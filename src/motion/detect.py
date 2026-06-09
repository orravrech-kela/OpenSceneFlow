"""Temporal accumulation (D) + DBSCAN clustering (A) + oriented box fitting.

For target frame *t*, candidate moving points from a window [t-w, t+w] are
de-translated to *t* along their own flow (``p - off*vel``), pooled, and
clustered in 3D. Pooling densifies real movers and lets a multi-frame support
test drop transient single-frame noise. Sensor is static so no warping is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from src.motion.types import Measurement

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

    def to_measurement(self) -> Measurement:
        """Lean tracker input (class-agnostic 'mover'); carries flow velocity."""
        cx, cy, cz, l, w, h, yaw = self.box
        return Measurement(
            class_name="mover", score=1.0,
            x=float(cx), y=float(cy), z=float(cz),
            dx=float(l), dy=float(w), dz=float(h), heading=float(yaw),
            vx=float(self.velocity[0]), vy=float(self.velocity[1]), vz=float(self.velocity[2]),
            num_points=int(self.points.shape[0]),
        )


def fit_box(
    pts: np.ndarray,
    vel: Optional[np.ndarray] = None,
    speed_yaw_thresh: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Oriented BEV box from points. Returns (box7, bev_corners, z_lo, z_hi).

    When the cluster is clearly moving (|vel_xy| >= speed_yaw_thresh) the yaw is
    taken from the flow direction (a stable, frame-consistent heading) instead of
    the minAreaRect angle, which is only defined mod-180 and flips between frames.
    """
    xy = np.ascontiguousarray(pts[:, :2], dtype=np.float32)
    (cx, cy), (a, b), ang = cv2.minAreaRect(xy)
    corners = cv2.boxPoints(((cx, cy), (a, b), ang)).astype(np.float64)
    z_lo, z_hi = (float(v) for v in np.percentile(pts[:, 2], [2, 98]))
    cz = 0.5 * (z_lo + z_hi)
    h = max(z_hi - z_lo, 0.1)
    l, w = max(a, b), min(a, b)
    yaw = np.deg2rad(ang if a >= b else ang + 90.0)
    if vel is not None and float(np.hypot(vel[0], vel[1])) >= speed_yaw_thresh:
        yaw = float(np.arctan2(vel[1], vel[0]))
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
    vel_weight: float = 1.0,
    speed_yaw_thresh: float = 0.15,
) -> List[Detection]:
    """window: list of (offset, xyz, vel) where offset = neighbor_idx - frame_idx.

    vel_weight scales the per-point flow into the clustering feature space so
    adjacent objects moving differently split into separate clusters: a velocity
    difference of 1/vel_weight (m/frame) separates points at the same location.
    vel_weight=0 reproduces position-only DBSCAN.
    """
    chunks = [(off, xyz, vel) for off, xyz, vel in window if xyz.shape[0] > 0]
    if not chunks:
        return []

    # De-translate neighbors to the target frame and pool, pre-allocating once.
    total = sum(xyz.shape[0] for _, xyz, _ in chunks)
    P = np.empty((total, 3), dtype=np.float64)
    V = np.empty((total, 3), dtype=np.float64)
    F = np.empty(total, dtype=np.int32)
    i = 0
    for off, xyz, vel in chunks:
        n = xyz.shape[0]
        P[i:i + n] = xyz - off * vel
        V[i:i + n] = vel
        F[i:i + n] = off
        i += n

    if max_range:
        keep = (P ** 2).sum(axis=1) <= max_range * max_range
        P, V, F = P[keep], V[keep], F[keep]
    if P.shape[0] < min_points:
        return []

    if vel_weight > 0.0:
        feat = np.concatenate([P, (vel_weight * eps) * V], axis=1)
    else:
        feat = P
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(feat).labels_
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
        vel_med = np.median(V[m], axis=0)
        box, corners, z_lo, z_hi = fit_box(pts, vel=vel_med, speed_yaw_thresh=speed_yaw_thresh)
        dets.append(
            Detection(
                frame_idx=frame_idx,
                raw_id=raw_id,
                box=box,
                velocity=vel_med,
                bev_corners=corners,
                z_lo=z_lo,
                z_hi=z_hi,
                points=pts,
            )
        )
    return dets
