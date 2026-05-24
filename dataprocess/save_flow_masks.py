"""Save per-frame ground-truth flow as polar-grid npz files for offline_viewer.

Mirrors `dataprocess/extract_innoviz.py`'s box-rigid-flow step but produces
sidecar files in the sequence directory instead of HDF5. Each output is shaped
``(H, W, 3) float32`` aligned with the polar grid, plus a ``(H, W) bool`` mask
marking pixels that received a flow value (non-empty intersection with a
tracked box at both t0 and t1).

Usage:
    python dataprocess/save_flow_masks.py \
        --data-root /mnt/data/lidar/processed \
        --sequences gan_shomron_27_11_2025/sprint \
        --subdir flow \
        --num-workers 4

Then visualise with offline_viewer's new flow overlay (see --flow-dir).
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import fire
import numpy as np
from tqdm import tqdm

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from dataprocess.extract_innoviz import (  # noqa: E402  reuse helpers
    DEFAULT_BOX_PAD,
    _enumerate_frames,
    _parse_annotations,
    _per_point_flow,
)


def _list_polar_int(seq_dir: Path) -> Dict[int, str]:
    polar_dir = seq_dir / "polar"
    return {int(p.stem): p.stem for p in polar_dir.glob("*.npz")}


def process_sequence(
    seq_dir: Path,
    subdir: str,
    box_pad: float,
    overwrite: bool,
) -> str:
    """Write per-frame polar-grid flow tensors under ``<seq_dir>/<subdir>/``."""
    if not seq_dir.is_dir():
        raise FileNotFoundError(seq_dir)
    lut_path = seq_dir / "lut.npz"
    with np.load(lut_path) as lut_npz:
        unit_vec = lut_npz["unit_vec"].astype(np.float32)
    H, W = unit_vec.shape[:2]

    frame_pairs = _enumerate_frames(seq_dir)
    if len(frame_pairs) < 2:
        return f"skip {seq_dir.name}: <2 polar frames with annotations"

    tracks, _ = _parse_annotations(seq_dir / "annotations" / "manual_gt.json")
    polar_int = _list_polar_int(seq_dir)
    dclass: Dict[int, int] = defaultdict(lambda: len(dclass))

    out_dir = seq_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    # Iterate consecutive (t0, t1) pairs; last frame has no flow and is left absent.
    for i in range(len(frame_pairs) - 1):
        frame_t0, raw_t0 = frame_pairs[i]
        frame_t1, _ = frame_pairs[i + 1]
        out_path = out_dir / f"{raw_t0}.npz"
        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue

        with np.load(seq_dir / "polar" / f"{raw_t0}.npz") as f:
            distance = f["distance"].astype(np.float32)
        xyz_grid = (distance[..., None] * unit_vec).reshape(-1, 3)
        keep = distance.reshape(-1) > 0.0
        xyz0 = xyz_grid[keep].astype(np.float32)

        flow_pts, _, _, instances_pts, _ = _per_point_flow(
            xyz0=xyz0,
            tracks=tracks,
            frame_t0=frame_t0,
            frame_t1=frame_t1,
            box_pad=box_pad,
            dclass=dclass,
        )
        has_flow_pts = instances_pts > 0  # only object points have non-trivial flow

        flow_grid = np.zeros((H * W, 3), dtype=np.float32)
        has_grid = np.zeros(H * W, dtype=bool)
        flat_indices = np.where(keep)[0]
        flow_grid[flat_indices] = flow_pts
        has_grid[flat_indices] = has_flow_pts
        flow_grid = flow_grid.reshape(H, W, 3)
        has_grid = has_grid.reshape(H, W)

        np.savez_compressed(out_path, flow=flow_grid, has_flow=has_grid)
        n_written += 1

    return f"{seq_dir.name}: wrote {n_written}, skipped {n_skipped}"


def _worker(args: dict) -> str:
    return process_sequence(
        seq_dir=Path(args["seq_dir"]),
        subdir=args["subdir"],
        box_pad=args["box_pad"],
        overwrite=args["overwrite"],
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
    subdir: str = "flow",
    box_pad: float = DEFAULT_BOX_PAD,
    overwrite: bool = False,
    num_workers: int = 1,
):
    """CLI entry point. See module docstring for example invocation."""
    seq_dirs = _resolve_sequences(data_root, sequences)
    print(f"Writing flow grids under <seq>/{subdir}/  (box_pad={box_pad})")
    for sd in seq_dirs:
        print(f"  - {sd}")

    tasks = [
        dict(
            seq_dir=str(sd),
            subdir=subdir,
            box_pad=box_pad,
            overwrite=overwrite,
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
