"""Detect and track moving objects from predicted scene flow (Innoviz, static sensor).

The tracker mirrors the lidar ``core.tracking`` interface: a runtime-checkable
``Tracker`` protocol, a ``TrackState`` enum, params dataclasses, an association
cost module, and a typed ``TrackedDetection`` output with ``to_dict()``.
"""

from src.motion.loader import Frame, list_frame_ids, load_candidates, load_full_cloud
from src.motion.detect import Detection, detect_frame, fit_box
from src.motion.types import Measurement, TrackState, TrackedDetection
from src.motion.protocol import Tracker
from src.motion.association import (
    compute_bev_iou_cost_matrix,
    compute_distance_cost_matrix,
)
from src.motion.kalman_box_tracker import KalmanBoxTracker, KalmanTrackerParams
from src.motion.multi_object_tracker import MOTParams, MultiObjectTracker

__all__ = [
    # loader
    "Frame",
    "list_frame_ids",
    "load_candidates",
    "load_full_cloud",
    # detector
    "Detection",
    "detect_frame",
    "fit_box",
    # tracking interface (mirrors lidar core.tracking)
    "Measurement",
    "TrackState",
    "TrackedDetection",
    "Tracker",
    "MOTParams",
    "MultiObjectTracker",
    "KalmanTrackerParams",
    "KalmanBoxTracker",
    "compute_distance_cost_matrix",
    "compute_bev_iou_cost_matrix",
]
