"""Save per-frame ground masks as polar-grid npz files for offline_viewer.

Mirrors `dataprocess/extract_innoviz.py`'s ground-segmentation step but produces
sidecar files in the sequence directory instead of HDF5. Each output is a
``(H, W)`` bool mask aligned with the polar grid (same shape as ``polar/*.npz``
and ``fg/*.npz``), saved as ``<seq>/<subdir>/<raw_id>.npz`` under the key
``is_foreground``. This is the format `offline_viewer.py` reads via ``--fg-dir``.

Usage:
    python dataprocess/save_ground_masks.py \
        --data-root /mnt/data/lidar/processed \
        --sequences gan_shomron_27_11_2025/sprint \
        --subdir ground \
        --num-workers 4

Then visualise with:
    uv run python offline_viewer.py \
        --polar-folder /mnt/data/lidar/processed/gan_shomron_27_11_2025/sprint/polar \
        --lut         /mnt/data/lidar/processed/gan_shomron_27_11_2025/sprint/lut.npz \
        --fg-dir      /mnt/data/lidar/processed/gan_shomron_27_11_2025/sprint/ground
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import fire
import numpy as np
from tqdm import tqdm

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from dataprocess.extract_innoviz import (  # noqa: E402  reuse helpers
    DEFAULT_GROUND_CONFIG,
    _autodetect_height,
    _materialise_ground_config,
)


def _list_polar_ids(seq_dir: Path) -> List[str]:
    polar_dir = seq_dir / "polar"
    return sorted(p.stem for p in polar_dir.glob("*.npz"))


def _load_distance(seq_dir: Path, raw_id: str) -> np.ndarray:
    with np.load(seq_dir / "polar" / f"{raw_id}.npz") as f:
        return f["distance"].astype(np.float32)


def process_sequence(
    seq_dir: Path,
    subdir: str,
    ground_config: str,
    auto_height: bool,
    height_override: Optional[float],
    overwrite: bool,
) -> str:
    """Write per-frame polar-grid ground masks under ``<seq_dir>/<subdir>/``."""
    if not seq_dir.is_dir():
        raise FileNotFoundError(seq_dir)
    lut_path = seq_dir / "lut.npz"
    with np.load(lut_path) as lut_npz:
        unit_vec = lut_npz["unit_vec"].astype(np.float32)
    H, W = unit_vec.shape[:2]

    raw_ids = _list_polar_ids(seq_dir)
    if not raw_ids:
        return f"skip {seq_dir}: no polar frames"

    # Resolve height: override > autodetect from frame 0 > toml default.
    tmp_config_path: Optional[Path] = None
    if height_override is not None:
        chosen_height = float(height_override)
    elif auto_height:
        dist0 = _load_distance(seq_dir, raw_ids[0])
        xyz0 = (dist0[..., None] * unit_vec).reshape(-1, 3)
        keep0 = dist0.reshape(-1) > 0.0
        chosen_height = _autodetect_height(xyz0[keep0])
    else:
        chosen_height = None
    if chosen_height is not None:
        tmp_config_path = _materialise_ground_config(Path(ground_config), chosen_height)
        effective_config = str(tmp_config_path)
    else:
        effective_config = ground_config

    out_dir = seq_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    from linefit import ground_seg
    seg = ground_seg(effective_config)

    n_written = 0
    n_skipped = 0
    for raw_id in raw_ids:
        out_path = out_dir / f"{raw_id}.npz"
        if out_path.exists() and not overwrite:
            n_skipped += 1
            continue
        distance = _load_distance(seq_dir, raw_id)
        xyz_flat = (distance[..., None] * unit_vec).reshape(-1, 3)
        keep = distance.reshape(-1) > 0.0
        xyz = xyz_flat[keep].astype(np.float32)
        gm = np.asarray(seg.run(xyz), dtype=bool)
        # Unflatten into the polar grid: only "kept" pixels can be ground.
        mask_grid = np.zeros(H * W, dtype=bool)
        mask_grid[np.where(keep)[0][gm]] = True
        mask_grid = mask_grid.reshape(H, W)
        np.savez_compressed(out_path, is_foreground=mask_grid)
        n_written += 1

    if tmp_config_path is not None:
        tmp_config_path.unlink(missing_ok=True)

    return f"{seq_dir.name}: wrote {n_written}, skipped {n_skipped} (height={chosen_height})"


def _worker(args: dict) -> str:
    return process_sequence(
        seq_dir=Path(args["seq_dir"]),
        subdir=args["subdir"],
        ground_config=args["ground_config"],
        auto_height=args["auto_height"],
        height_override=args["height_override"],
        overwrite=args["overwrite"],
    )


def _resolve_sequences(data_root: str, sequences) -> List[Path]:
    """Expand the --sequences arg (string, list, or comma-sep) into absolute paths."""
    if sequences is None:
        raise ValueError("--sequences is required (e.g. 'day/seq' or list of them)")
    if isinstance(sequences, str):
        raw = [s for s in sequences.split(",") if s.strip()]
    elif isinstance(sequences, (list, tuple)):
        raw = list(sequences)
    else:
        raise ValueError(f"--sequences must be a string or list, got {type(sequences)}")
    root = Path(data_root)
    out: List[Path] = []
    for entry in raw:
        entry = entry.strip().strip("/")
        path = root / entry
        if not path.is_dir():
            raise FileNotFoundError(f"Sequence dir not found: {path}")
        out.append(path)
    return out


def main(
    data_root: str = "/mnt/data/lidar/processed",
    sequences=None,                     # "<day>/<seq>" or comma-sep / list
    subdir: str = "ground",
    ground_config: str = DEFAULT_GROUND_CONFIG,
    auto_height: bool = True,
    height: Optional[float] = None,
    overwrite: bool = False,
    num_workers: int = 1,
):
    """CLI entry point. See module docstring for example invocation."""
    seq_dirs = _resolve_sequences(data_root, sequences)
    print(f"Writing ground masks under <seq>/{subdir}/  (config: {ground_config})")
    for sd in seq_dirs:
        print(f"  - {sd}")

    tasks = [
        dict(
            seq_dir=str(sd),
            subdir=subdir,
            ground_config=ground_config,
            auto_height=auto_height,
            height_override=height,
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
