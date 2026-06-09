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
    extent_pct: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Oriented BEV box from points. Returns (box7, bev_corners, z_lo, z_hi).

    When the cluster is clearly moving (|vel_xy| >= speed_yaw_thresh) the heading is
    taken from the flow direction and length/width are measured as percentile-
    trimmed extents *along that heading* -- a tight, orientation-consistent,
    outlier-resistant box (unlike minAreaRect, which encloses every stray point and
    whose axes need not align with travel). Otherwise size/orientation come from
    minAreaRect. extent_pct (%) is trimmed off each end when measuring extents.
    """
    xy = np.ascontiguousarray(pts[:, :2], dtype=np.float64)
    lo_pct, hi_pct = extent_pct, 100.0 - extent_pct
    z_lo, z_hi = (float(v) for v in np.percentile(pts[:, 2], [lo_pct, hi_pct]))
    cz = 0.5 * (z_lo + z_hi)
    h = max(z_hi - z_lo, 0.1)

    if vel is not None and float(np.hypot(vel[0], vel[1])) >= speed_yaw_thresh:
        yaw = float(np.arctan2(vel[1], vel[0]))
        u = np.array([np.cos(yaw), np.sin(yaw)])    # along heading
        v = np.array([-np.sin(yaw), np.cos(yaw)])   # cross heading
        pu, pv = xy @ u, xy @ v
        lo_u, hi_u = np.percentile(pu, [lo_pct, hi_pct])
        lo_v, hi_v = np.percentile(pv, [lo_pct, hi_pct])
        l, w = hi_u - lo_u, hi_v - lo_v
        center = 0.5 * (lo_u + hi_u) * u + 0.5 * (lo_v + hi_v) * v
        cx, cy = float(center[0]), float(center[1])
        hl, hw = l / 2.0, w / 2.0
        corners = np.array([
            center + hl * u + hw * v,
            center + hl * u - hw * v,
            center - hl * u - hw * v,
            center - hl * u + hw * v,
        ], dtype=np.float64)
    else:
        (cx, cy), (a, b), ang = cv2.minAreaRect(xy.astype(np.float32))
        corners = cv2.boxPoints(((cx, cy), (a, b), ang)).astype(np.float64)
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
    vel_weight: float = 1.0,
    speed_yaw_thresh: float = 0.15,
    min_box_points: int = 6,
    min_current_points: int = 2,
) -> List[Detection]:
    """window: list of (offset, xyz, vel) where offset = neighbor_idx - frame_idx.

    vel_weight scales the per-point flow into the clustering feature space so
    adjacent objects moving differently split into separate clusters: a velocity
    difference of 1/vel_weight (m/frame) separates points at the same location.
    vel_weight=0 reproduces position-only DBSCAN.

    min_current_points requires a cluster to have at least this many points at the
    target frame (offset 0) to be emitted -- it rejects pure temporal smear from
    objects only present in past/future window frames (which otherwise produce
    boxes ~window frames before an object arrives and after it leaves). 0 disables.
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
        cur = F[m] == 0
        # Require the object to actually be present at the target frame (not pure
        # past/future smear).
        if int(cur.sum()) < min_current_points:
            continue
        pts = P[m]
        vel_med = np.median(V[m], axis=0)
        # Size the box from current-frame members only (offset 0, un-de-translated)
        # to avoid flow-smear; the pooled window is for finding/velocity/persistence.
        pts_box = pts[cur] if int(cur.sum()) >= min_box_points else pts
        box, corners, z_lo, z_hi = fit_box(pts_box, vel=vel_med, speed_yaw_thresh=speed_yaw_thresh)
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
