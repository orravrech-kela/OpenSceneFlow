"""Detection-to-track association cost matrices (mirrors lidar core.tracking).

Provides Euclidean-distance and BEV-IoU cost matrices for matching measurements
to existing tracks via the Hungarian algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.motion.types import Measurement

if TYPE_CHECKING:
    from src.motion.kalman_box_tracker import KalmanBoxTracker


def compute_distance_cost_matrix(
    trackers: list["KalmanBoxTracker"],
    detections: list[Measurement],
) -> np.ndarray:
    """Euclidean distance between each tracker's predicted position and detection.

    Returns an (M, N) cost matrix; cost[i, j] = ||track_i - det_j||_2.
    """
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((len(trackers), len(detections)))

    tracker_positions = np.array([t.position for t in trackers])  # (M, 3)
    detection_positions = np.array([[d.x, d.y, d.z] for d in detections])  # (N, 3)

    diff = tracker_positions[:, np.newaxis, :] - detection_positions[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))  # (M, N)


def compute_bev_iou_cost_matrix(
    trackers: list["KalmanBoxTracker"],
    detections: list[Measurement],
) -> np.ndarray:
    """BEV IoU cost (1 - IoU) for dense scenes.

    Uses the orientation-aware axis-aligned box that *encloses* each oriented BEV
    box (so the box heading is honored, unlike a square max(l, w) bound).
    """
    if len(trackers) == 0 or len(detections) == 0:
        return np.empty((len(trackers), len(detections)))

    cost_matrix = np.ones((len(trackers), len(detections)), dtype=np.float32)
    for i, tracker in enumerate(trackers):
        tx, ty, _ = tracker.position
        tl, tw, _ = tracker.dimensions
        tth = tracker.heading
        for j, det in enumerate(detections):
            iou = _compute_bev_iou(tx, ty, tl, tw, tth, det.x, det.y, det.dx, det.dy, det.heading)
            cost_matrix[i, j] = 1.0 - iou
    return cost_matrix


def _aabb_half_extents(l: float, w: float, theta: float) -> tuple[float, float]:
    """Half-extents of the axis-aligned box enclosing an oriented (l, w, theta) box."""
    c, s = abs(np.cos(theta)), abs(np.sin(theta))
    return 0.5 * (l * c + w * s), 0.5 * (l * s + w * c)


def _compute_bev_iou(
    x1: float, y1: float, l1: float, w1: float, theta1: float,
    x2: float, y2: float, l2: float, w2: float, theta2: float,
) -> float:
    """IoU of the orientation-aware enclosing AABBs of two oriented BEV boxes."""
    hx1, hy1 = _aabb_half_extents(l1, w1, theta1)
    hx2, hy2 = _aabb_half_extents(l2, w2, theta2)

    inter_w = max(0.0, min(x1 + hx1, x2 + hx2) - max(x1 - hx1, x2 - hx2))
    inter_h = max(0.0, min(y1 + hy1, y2 + hy2) - max(y1 - hy1, y2 - hy2))
    inter_area = inter_w * inter_h

    area1 = (2 * hx1) * (2 * hy1)
    area2 = (2 * hx2) * (2 * hy2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
