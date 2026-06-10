"""Core types for flow-based moving-object tracking.

Mirrors the lidar ``core.tracking`` contract (TrackState / typed output with
``to_dict``) so the two trackers share an interface. ``Measurement`` is the
lean per-frame tracker input (analog of lidar ``Detection3D``); unlike the lidar
detection it carries a flow ``velocity`` (m/frame) used to seed the Kalman filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackState(Enum):
    """Track lifecycle state: TENTATIVE -> CONFIRMED -> LOST -> (deleted)."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"


@dataclass
class Measurement:
    """Lean per-frame detection consumed by the tracker (analog of Detection3D).

    vx, vy, vz are the per-detection flow velocity (m/frame) — OSF-specific, used
    to seed/fuse the Kalman filter. num_points carries the cluster size for output.
    """

    class_name: str
    score: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    num_points: int = 0


@dataclass
class TrackedDetection:
    """Tracked detection output (analog of lidar TrackedDetection3D).

    OSF extensions over the lidar type: ``displacement`` (net translation from
    birth, drives the static-structure gate) and ``num_points``.
    """

    track_id: int
    class_name: str
    score: float
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    heading: float
    track_state: TrackState
    age: int
    hits: int
    time_since_update: int
    vx: float
    vy: float
    vz: float
    displacement: float = 0.0
    num_points: int = 0

    def to_dict(self) -> dict:
        """JSON dict. Superset of the offline_viewer detection schema: the viewer
        keys (``class_name``, string ``track_id``, x/y/z/dx/dy/dz/heading/score)
        are preserved verbatim; lidar lifecycle fields are added alongside."""
        return {
            # keys the external offline_viewer.py reads -- preserve verbatim
            "class_name": self.class_name,
            "score": self.score,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "heading": self.heading,
            "track_id": str(self.track_id),
            # lidar-parity additions (viewer ignores unknown keys)
            "class": self.class_name,
            "state": self.track_state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "velocity": {"vx": self.vx, "vy": self.vy, "vz": self.vz},
            "displacement": self.displacement,
            "num_points": self.num_points,
        }

    def to_cvat_dict(self) -> dict:
        """detections.json shape matching the lidar inference ``save_results``
        contract (CVAT-import / offline_viewer ready): ``class`` label, **int**
        ``track_id``, and flat ``vel_x``/``vel_y`` + ``track_vel_x``/``track_vel_y``.
        OSF lifecycle fields are kept as a harmless superset."""
        return {
            "class": self.class_name,
            "score": self.score,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "heading": self.heading,
            "vel_x": self.vx,
            "vel_y": self.vy,
            "track_id": self.track_id,
            "track_vel_x": self.vx,
            "track_vel_y": self.vy,
            # OSF lifecycle extras (superset; ignored by CVAT/viewer)
            "state": self.track_state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "displacement": self.displacement,
            "num_points": self.num_points,
        }
