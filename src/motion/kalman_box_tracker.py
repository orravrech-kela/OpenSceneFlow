"""Kalman filter 3D box tracker for flow-based movers (mirrors lidar core.tracking).

filterpy constant-velocity KF. State [x, y, z, theta, l, w, h, vx, vy, vz].

Differences from the lidar tracker, all motivated by the flow signal:
- dt defaults to 1.0 *frame* (flow velocity is m/frame, so predict is ``x += v``).
- velocity is seeded from the flow measurement at birth (lidar seeds 0).
- when ``use_flow_measurement`` (default) the filter measures velocity too
  (dim_z=10, H=I), directly fusing the per-frame flow; otherwise it mirrors
  lidar's pose/size-only measurement (dim_z=7).
- heading is wrapped onto the prediction's branch and the minAreaRect 180-deg
  box-flip is resolved on update (lidar does no angle wrapping).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from src.motion.types import Measurement, TrackedDetection, TrackState


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class KalmanTrackerParams:
    """Parameters for the Kalman box tracker.

    dt is in frames (flow is m/frame). r_vel/p_vel_init govern how much the
    flow velocity is trusted as a measurement / birth seed.
    """

    dt: float = 1.0
    q_pos: float = 0.1
    q_vel: float = 0.5
    q_dim: float = 0.01
    q_theta: float = 0.1
    r_pos: float = 0.5
    r_dim: float = 0.1
    r_theta: float = 0.2
    r_vel: float = 1.0
    p_vel_init: float = 2.0
    use_flow_measurement: bool = True


class KalmanBoxTracker:
    """Kalman filter tracker for a single 3D bounding box.

    State (10D): [x, y, z, theta, l, w, h, vx, vy, vz].
    Measurement: [x, y, z, theta, l, w, h] (+ [vx, vy, vz] when fusing flow).
    """

    _count = 0  # class-level counter for unique track IDs

    def __init__(
        self,
        detection: Measurement,
        params: KalmanTrackerParams | None = None,
    ) -> None:
        if params is None:
            params = KalmanTrackerParams()
        self._params = params
        self._use_flow = params.use_flow_measurement

        KalmanBoxTracker._count += 1
        self.track_id = KalmanBoxTracker._count

        self.class_name = detection.class_name
        self.score = detection.score
        self.num_points = detection.num_points

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE

        # Anchor for the net-displacement gate (removes static spurious-flow boxes).
        self.first_center = np.array(
            [detection.x, detection.y, detection.z], dtype=np.float64
        )

        dim_z = 10 if self._use_flow else 7
        self._kf = KalmanFilter(dim_x=10, dim_z=dim_z)

        dt = params.dt
        self._kf.F = np.array([
            [1, 0, 0, 0, 0, 0, 0, dt, 0,  0],   # x
            [0, 1, 0, 0, 0, 0, 0, 0,  dt, 0],   # y
            [0, 0, 1, 0, 0, 0, 0, 0,  0,  dt],  # z
            [0, 0, 0, 1, 0, 0, 0, 0,  0,  0],   # theta
            [0, 0, 0, 0, 1, 0, 0, 0,  0,  0],   # l
            [0, 0, 0, 0, 0, 1, 0, 0,  0,  0],   # w
            [0, 0, 0, 0, 0, 0, 1, 0,  0,  0],   # h
            [0, 0, 0, 0, 0, 0, 0, 1,  0,  0],   # vx
            [0, 0, 0, 0, 0, 0, 0, 0,  1,  0],   # vy
            [0, 0, 0, 0, 0, 0, 0, 0,  0,  1],   # vz
        ], dtype=np.float64)

        if self._use_flow:
            # Measure pose, size, AND flow velocity directly.
            self._kf.H = np.eye(10, dtype=np.float64)
            self._kf.R = np.diag([
                params.r_pos, params.r_pos, params.r_pos, params.r_theta,
                params.r_dim, params.r_dim, params.r_dim,
                params.r_vel, params.r_vel, params.r_vel,
            ])
        else:
            # Pose/size-only measurement, exactly like the lidar tracker.
            self._kf.H = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # x
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # y
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # z
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # theta
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # l
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # w
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # h
            ], dtype=np.float64)
            self._kf.R = np.diag([
                params.r_pos, params.r_pos, params.r_pos, params.r_theta,
                params.r_dim, params.r_dim, params.r_dim,
            ])

        self._kf.Q = np.diag([
            params.q_pos, params.q_pos, params.q_pos, params.q_theta,
            params.q_dim, params.q_dim, params.q_dim,
            params.q_vel, params.q_vel, params.q_vel,
        ])

        # Lower velocity uncertainty than lidar -- we seed it from flow.
        self._kf.P = np.diag([
            1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1,
            params.p_vel_init, params.p_vel_init, params.p_vel_init,
        ])

        self._kf.x = np.array([
            detection.x, detection.y, detection.z, detection.heading,
            detection.dx, detection.dy, detection.dz,
            detection.vx, detection.vy, detection.vz,  # seed velocity from flow
        ], dtype=np.float64).reshape(-1, 1)

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the track ID counter (per-sequence numbering / testing)."""
        cls._count = 0

    def predict(self) -> None:
        """Advance the motion model one frame. Call once per frame before update."""
        self._kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Measurement) -> None:
        """Fuse a matched detection into the filter."""
        # Bring the measured heading onto the prediction's branch and resolve the
        # minAreaRect 180-deg box-flip ambiguity before measuring.
        pred_theta = float(self._kf.x[3, 0])
        dtheta = _wrap_to_pi(detection.heading - pred_theta)
        if abs(dtheta) > np.pi / 2.0:
            dtheta = _wrap_to_pi(dtheta + np.pi)
        meas_theta = pred_theta + dtheta

        z = [detection.x, detection.y, detection.z, meas_theta,
             detection.dx, detection.dy, detection.dz]
        if self._use_flow:
            z += [detection.vx, detection.vy, detection.vz]
        self._kf.update(np.array(z, dtype=np.float64).reshape(-1, 1))
        self._kf.x[3, 0] = _wrap_to_pi(float(self._kf.x[3, 0]))

        self.hits += 1
        self.time_since_update = 0
        self.score = detection.score
        self.num_points = detection.num_points

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def confirm(self) -> None:
        """Promote TENTATIVE/LOST to CONFIRMED (re-association recovers LOST)."""
        if self.state in (TrackState.TENTATIVE, TrackState.LOST):
            self.state = TrackState.CONFIRMED

    @property
    def position(self) -> tuple[float, float, float]:
        x = self._kf.x
        return (float(x[0, 0]), float(x[1, 0]), float(x[2, 0]))

    @property
    def velocity(self) -> tuple[float, float, float]:
        x = self._kf.x
        return (float(x[7, 0]), float(x[8, 0]), float(x[9, 0]))

    @property
    def heading(self) -> float:
        return _wrap_to_pi(float(self._kf.x[3, 0]))

    @property
    def dimensions(self) -> tuple[float, float, float]:
        x = self._kf.x
        return (float(x[4, 0]), float(x[5, 0]), float(x[6, 0]))

    @property
    def displacement(self) -> float:
        """Net translation from birth center (drives the static-structure gate)."""
        x = self._kf.x
        cur = np.array([x[0, 0], x[1, 0], x[2, 0]])
        return float(np.linalg.norm(cur - self.first_center))

    def get_state(self) -> np.ndarray:
        return self._kf.x.flatten()

    def to_tracked_detection(self) -> TrackedDetection:
        x, y, z = self.position
        vx, vy, vz = self.velocity
        dx, dy, dz = self.dimensions
        return TrackedDetection(
            track_id=self.track_id,
            class_name=self.class_name,
            score=self.score,
            x=x, y=y, z=z,
            dx=dx, dy=dy, dz=dz,
            heading=self.heading,
            track_state=self.state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            vx=vx, vy=vy, vz=vz,
            displacement=self.displacement,
            num_points=self.num_points,
        )
