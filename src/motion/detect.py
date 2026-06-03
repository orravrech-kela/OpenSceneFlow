"""Temporal accumulation (D) + DBSCAN clustering (A) + oriented box fitting.

For target frame *t*, candidate moving points from a window [t-w, t+w] are
de-translated to *t* along their own flow (``p - off*vel``), pooled, and
clustered in 3D. Pooling densifies real movers and lets a multi-frame support
test drop transient single-frame noise. Sensor is static so no warping is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

BOX_LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


@dataclass
class Detection:
    frame_idx: int
    raw_id: str
    box: np.ndarray        # [cx, cy, cz, l, w, h, yaw]
    velocity: np.ndarray   # [vx, vy, vz], m/frame
    bev_corners: np.ndarray  # (4, 2) BEV footprint
    z_lo: float
    z_hi: float
    points: np.ndarray     # (n, 3) member points, de-translated to t
    track_id: int = -1

    def corners3d(self) -> np.ndarray:
        c = self.bev_corners
        bottom = np.column_stack([c, np.full(4, self.z_lo)])
        top = np.column_stack([c, np.full(4, self.z_hi)])
        return np.vstack([top, bottom])


def fit_box(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Oriented BEV box from points. Returns (box7, bev_corners, z_lo, z_hi)."""
    xy = np.ascontiguousarray(pts[:, :2], dtype=np.float32)
    (cx, cy), (a, b), ang = cv2.minAreaRect(xy)
    corners = cv2.boxPoints(((cx, cy), (a, b), ang)).astype(np.float64)
    z_lo, z_hi = (float(v) for v in np.percentile(pts[:, 2], [5, 95]))
    cz = 0.5 * (z_lo + z_hi)
    h = max(z_hi - z_lo, 0.1)
    l, w = max(a, b), min(a, b)
    yaw = np.deg2rad(ang if a >= b else ang + 90.0)
    box = np.array([cx, cy, cz, max(l, 0.1), max(w, 0.1), h, yaw], dtype=np.float64)
    return box, corners, z_lo, z_hi


def detect_frame(
    window: List[Tuple[int, np.ndarray, np.ndarray]],
    frame_idx: int,
    raw_id: str,
    eps: float = 0.6,
    min_samples: int = 6,
    min_points: int = 12,
    min_frames: int = 2,
    max_range: float = 120.0,
) -> List[Detection]:
    """window: list of (offset, xyz, vel) where offset = neighbor_idx - frame_idx."""
    pos, vel_list, frm = [], [], []
    for off, xyz, vel in window:
        if xyz.shape[0] == 0:
            continue
        pos.append(xyz - off * vel)   # de-translate neighbor to target frame
        vel_list.append(vel)
        frm.append(np.full(xyz.shape[0], off, dtype=np.int32))
    if not pos:
        return []
    P = np.concatenate(pos)
    V = np.concatenate(vel_list)
    F = np.concatenate(frm)

    if max_range:
        keep = np.linalg.norm(P, axis=1) <= max_range
        P, V, F = P[keep], V[keep], F[keep]
    if P.shape[0] < min_points:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(P).labels_
    n_window = max(1, len({off for off, _, _ in window}))
    min_frames_eff = min(min_frames, n_window)

    dets: List[Detection] = []
    for c in np.unique(labels):
        if c < 0:
            continue
        m = labels == c
        if m.sum() < min_points:
            continue
        if np.unique(F[m]).size < min_frames_eff:
            continue
        pts = P[m]
        box, corners, z_lo, z_hi = fit_box(pts)
        dets.append(
            Detection(
                frame_idx=frame_idx,
                raw_id=raw_id,
                box=box,
                velocity=np.median(V[m], axis=0),
                bev_corners=corners,
                z_lo=z_lo,
                z_hi=z_hi,
                points=pts,
            )
        )
    return dets
