"""Deprecated module. Use ``src.motion.MultiObjectTracker`` / ``MOTParams``.

The legacy ``Tracker`` (constant-velocity + Hungarian, mutating Detection in
place) was replaced by the lidar-mirrored Kalman ``MultiObjectTracker``. This
shim keeps ``from src.motion.track import Tracker`` working for stray callers;
prefer importing from ``src.motion`` directly.
"""

from __future__ import annotations

import warnings

from src.motion.multi_object_tracker import MOTParams, MultiObjectTracker  # noqa: F401

warnings.warn(
    "src.motion.track is deprecated; import MultiObjectTracker/MOTParams from src.motion",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy name: "Tracker" used to be the implementation (now MultiObjectTracker).
Tracker = MultiObjectTracker
