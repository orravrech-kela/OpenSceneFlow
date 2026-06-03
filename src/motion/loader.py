"""Load Innoviz polar frames + predicted flow, unproject to moving-point candidates.

The predicted-flow sidecar (``<seq>/<pred_subdir>/<raw_id>.npz``, written by
``dataprocess/save_pred_flow_masks.py``) is pixel-aligned with the polar
``distance`` grid: ``flow[r,c]`` is the prediction for the point
``distance[r,c] * unit_vec[r,c]``. Sensor is static, so flow == object motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    idx: int           # position in the sorted sequence
    raw_id: str
    xyz: np.ndarray    # (M, 3) candidate moving points, sensor frame
    vel: np.ndarray    # (M, 3) per-point flow, m/frame
    n_valid: int       # total distance>0 points (context only)


def list_frame_ids(pred_dir: Path) -> list[str]:
    ids = [p.stem for p in pred_dir.glob("*.npz")]
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else s)


@lru_cache(maxsize=2)
def _load_unit_vec(lut_path: str) -> np.ndarray:
    with np.load(lut_path) as f:
        return f["unit_vec"].astype(np.float32)


def load_candidates(
    seq_dir: Path,
    pred_subdir: str,
    raw_id: str,
    tau: float,
    use_snp: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (xyz, vel, n_valid) for pixels with |flow| >= tau."""
    unit_vec = _load_unit_vec(str(seq_dir / "lut.npz"))
    with np.load(seq_dir / "polar" / f"{raw_id}.npz") as pf:
        distance = pf["distance"].astype(np.float32)
    with np.load(seq_dir / pred_subdir / f"{raw_id}.npz") as pf:
        flow = pf["flow"].astype(np.float32)
        has_flow = pf["has_flow"]

    valid = distance > 0.0
    mag = np.linalg.norm(flow, axis=-1)
    mask = valid & has_flow & (mag >= tau)
    if use_snp:
        snp_path = seq_dir / "filter_snp" / f"{raw_id}.npz"
        if snp_path.is_file():
            with np.load(snp_path) as sf:
                mask &= ~sf["is_filtered_out"]

    xyz = distance[..., None] * unit_vec
    return xyz[mask], flow[mask], int(valid.sum())


def load_full_cloud(seq_dir: Path, raw_id: str, max_points: int = 60_000) -> np.ndarray:
    """All distance>0 points (sensor frame), uniformly subsampled for context viz."""
    unit_vec = _load_unit_vec(str(seq_dir / "lut.npz"))
    with np.load(seq_dir / "polar" / f"{raw_id}.npz") as pf:
        distance = pf["distance"].astype(np.float32)
    valid = distance > 0.0
    pts = (distance[..., None] * unit_vec)[valid]
    if pts.shape[0] > max_points:
        step = pts.shape[0] // max_points
        pts = pts[::step]
    return pts
