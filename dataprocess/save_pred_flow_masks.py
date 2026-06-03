"""Save per-frame predicted flow as polar-grid npz files for offline_viewer.

Mirrors `dataprocess/save_flow_masks.py` but reads the prediction tensor from
an inference-run HDF5 (written by `save.py` under `f[ts][res_name]`) instead
of computing rigid box-to-box flow. Outputs match the GT-sidecar layout — one
``<seq>/<subdir>/<raw_id>.npz`` per frame with keys ``flow`` (H, W, 3) float32
and ``has_flow`` (H, W) bool — so the viewer can swap GT for prediction by
pointing ``--flow-dir`` at the new subdir.

Usage:
    python dataprocess/save_pred_flow_masks.py \\
        --data_root /mnt/data/lidar/processed \\
        --sequences gan_shomron_27_11_2025/sprint \\
        --h5_path  /home/ubuntu/orr/data/innoviz_h5/val_zshift/gan_shomron_27_11_2025__sprint.h5 \\
        --res_name deltaflow_waymo \\
        --subdir   pred_flow

Then visualise with:
    uv run python offline_viewer.py \\
        --polar-folder /mnt/data/lidar/processed/<day>/<seq>/polar \\
        --lut          /mnt/data/lidar/processed/<day>/<seq>/lut.npz \\
        --flow-dir     /mnt/data/lidar/processed/<day>/<seq>/pred_flow
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import fire
import h5py
import numpy as np
from tqdm import tqdm

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from dataprocess.extract_innoviz import (  # noqa: E402  reuse helpers
    DEFAULT_FRAME_PERIOD_NS,
    _enumerate_frames,
)


def process_sequence(
    seq_dir: Path,
    h5_path: Path,
    res_name: str,
    subdir: str,
    frame_period_ns: int,
    has_flow_threshold: float,
    overwrite: bool,
    no_annotations: bool = False,
) -> str:
    """Project ``f[<ts>][res_name]`` into a (H, W, 3) polar grid per frame."""
    if not seq_dir.is_dir():
        raise FileNotFoundError(seq_dir)
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)

    lut_path = seq_dir / "lut.npz"
    with np.load(lut_path) as lut_npz:
        unit_vec = lut_npz["unit_vec"].astype(np.float32)
    H, W = unit_vec.shape[:2]

    frame_pairs = _enumerate_frames(seq_dir, no_annotations=no_annotations)
    if not frame_pairs:
        return f"skip {seq_dir.name}: no polar frames found"

    out_dir = seq_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_missing = 0
    with h5py.File(h5_path, "r") as h5:
        for frame_idx, raw_id in frame_pairs:
            out_path = out_dir / f"{raw_id}.npz"
            if out_path.exists() and not overwrite:
                n_skipped += 1
                continue

            ts = str(frame_idx * frame_period_ns)
            if ts not in h5 or res_name not in h5[ts]:
                # Frames without prediction (history-warmup at start, final frame).
                n_missing += 1
                continue

            pred = h5[ts][res_name][:]  # (N, 3) float32

            # Recompute the same flatten/unflatten mask the extractor used so
            # the N predictions land at the right pixels.
            with np.load(seq_dir / "polar" / f"{raw_id}.npz") as polar:
                distance = polar["distance"].astype(np.float32)
            keep = distance.reshape(-1) > 0.0
            if int(keep.sum()) != pred.shape[0]:
                # Polar grid was modified after extraction, or wrong h5 — skip
                # rather than scatter wrong indices into a misleading sidecar.
                n_missing += 1
                continue

            flow_grid = np.zeros((H * W, 3), dtype=np.float32)
            has_grid = np.zeros(H * W, dtype=bool)
            flat_indices = np.where(keep)[0]
            flow_grid[flat_indices] = pred
            has_grid[flat_indices] = np.linalg.norm(pred, axis=1) >= has_flow_threshold

            np.savez_compressed(
                out_path,
                flow=flow_grid.reshape(H, W, 3),
                has_flow=has_grid.reshape(H, W),
            )
            n_written += 1

    return (
        f"{seq_dir.name}: wrote {n_written}, skipped {n_skipped}, "
        f"no-pred {n_missing}"
    )


def _worker(args: dict) -> str:
    return process_sequence(
        seq_dir=Path(args["seq_dir"]),
        h5_path=Path(args["h5_path"]),
        res_name=args["res_name"],
        subdir=args["subdir"],
        frame_period_ns=args["frame_period_ns"],
        has_flow_threshold=args["has_flow_threshold"],
        overwrite=args["overwrite"],
        no_annotations=args["no_annotations"],
    )


def _resolve_sequences(data_root: str, sequences) -> List[Path]:
    if sequences is None:
        raise ValueError("--sequences is required")
    if isinstance(sequences, str):
        raw = [s for s in sequences.split(",") if s.strip()]
    elif isinstance(sequences, (list, tuple)):
        raw = list(sequences)
    else:
        raise ValueError(f"--sequences must be a string or list, got {type(sequences)}")
    root = Path(data_root)
    out: List[Path] = []
    for entry in raw:
        path = root / entry.strip().strip("/")
        if not path.is_dir():
            raise FileNotFoundError(f"Sequence dir not found: {path}")
        out.append(path)
    return out


def main(
    data_root: str = "/mnt/data/lidar/processed",
    sequences=None,
    h5_path: Optional[str] = None,
    res_name: str = "deltaflow_waymo",
    subdir: str = "pred_flow",
    frame_period_ns: int = DEFAULT_FRAME_PERIOD_NS,
    has_flow_threshold: float = 0.05,
    overwrite: bool = False,
    num_workers: int = 1,
    no_annotations: bool = False,
):
    """CLI entry point. See module docstring for example invocation.

    ``has_flow_threshold`` controls which pixels are tagged ``has_flow=True`` in
    the sidecar; the viewer uses that mask to decide overlay membership.
    Defaults to 5 cm / frame to suppress visual noise from near-zero
    predictions while keeping any meaningful motion.
    """
    if h5_path is None:
        raise ValueError(
            "--h5_path is required (e.g. /home/.../innoviz_h5/val/<scene>.h5)"
        )
    seq_dirs = _resolve_sequences(data_root, sequences)
    print(
        f"Writing predicted flow grids under <seq>/{subdir}/ "
        f"(res_name={res_name}, has_flow_threshold={has_flow_threshold})"
    )
    for sd in seq_dirs:
        print(f"  - {sd}")

    tasks = [
        dict(
            seq_dir=str(sd),
            h5_path=h5_path,
            res_name=res_name,
            subdir=subdir,
            frame_period_ns=frame_period_ns,
            has_flow_threshold=has_flow_threshold,
            overwrite=overwrite,
            no_annotations=no_annotations,
        )
        for sd in seq_dirs
    ]
    start = time.time()
    if num_workers > 1 and len(tasks) > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for msg in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), ncols=100):
                tqdm.write(msg)
    else:
        for t in tqdm(tasks, ncols=100):
            tqdm.write(_worker(t))
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    fire.Fire(main)
