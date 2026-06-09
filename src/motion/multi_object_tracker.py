"""Multi-object tracker: Kalman filtering + Hungarian assignment.

Mirrors the lidar ``MultiObjectTracker`` lifecycle (predict -> associate ->
update -> birth -> death). OSF extension: emission additionally requires the
track to have net-translated past ``disp_gate`` -- this drops static structures
that carry a persistent but spurious flow prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import linear_sum_assignment

from src.motion.association import (
    compute_bev_iou_cost_matrix,
    compute_distance_cost_matrix,
)
from src.motion.kalman_box_tracker import KalmanBoxTracker, KalmanTrackerParams
from src.motion.types import Measurement, TrackedDetection, TrackState


@dataclass
class MOTParams:
    """Parameters for the multi-object tracker.

    max_age: frames without detection before deletion (legacy ``max_misses``).
    min_hits: associations before a track is confirmed.
    gate_distance: max association distance, meters (legacy ``max_dist``).
    disp_gate: net translation (m) a track must reach before it is emitted
        (OSF static-structure filter; no lidar equivalent).
    min_speed: min coherent BEV speed (m/frame) to emit -- rejects near-static jitter.
    max_motion_angle: max angle (deg) between a track's net displacement and its
        flow velocity to emit. Real movers travel along their velocity; noisy
        false-flow clusters drift incoherently and are rejected. 180 disables it.
    use_bev_iou: associate by BEV-IoU instead of centroid distance.
    """

    max_age: int = 4
    min_hits: int = 3
    gate_distance: float = 2.5
    disp_gate: float = 0.5
    min_speed: float = 0.05
    max_motion_angle: float = 60.0
    use_bev_iou: bool = False
    kalman_params: KalmanTrackerParams | None = None


class MultiObjectTracker:
    """Class-agnostic flow-based multi-object tracker."""

    def __init__(self, params: MOTParams | None = None) -> None:
        if params is None:
            params = MOTParams()
        self._params = params
        self._kalman_params = params.kalman_params or KalmanTrackerParams()

        self._trackers: list[KalmanBoxTracker] = []
        self._frame_count = 0
        self._last_matches: dict[int, Measurement] = {}

    @property
    def last_matches(self) -> dict[int, Measurement]:
        """Map of track_id -> measurement matched this frame."""
        return self._last_matches

    @property
    def num_tracks(self) -> int:
        return len(self._trackers)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        """Clear all state and restart track ID numbering."""
        self._trackers.clear()
        self._frame_count = 0
        self._last_matches = {}
        KalmanBoxTracker.reset_counter()

    def update(self, detections: list[Measurement]) -> list[TrackedDetection]:
        """Process one frame; return confirmed, moving tracks updated this frame."""
        self._frame_count += 1
        self._last_matches = {}

        # Step 1: predict
        for tracker in self._trackers:
            tracker.predict()

        # Step 2: associate
        matched, unmatched_dets, _ = self._associate(detections)

        # Step 3: update matched tracks
        for trk_idx, det_idx in matched:
            self._trackers[trk_idx].update(detections[det_idx])
            self._last_matches[self._trackers[trk_idx].track_id] = detections[det_idx]
            if self._trackers[trk_idx].hits >= self._params.min_hits:
                self._trackers[trk_idx].confirm()

        # Step 4: birth
        for det_idx in unmatched_dets:
            self._trackers.append(
                KalmanBoxTracker(detections[det_idx], params=self._kalman_params)
            )

        # Step 5: death
        trackers_to_keep = []
        for tracker in self._trackers:
            if tracker.time_since_update > self._params.max_age:
                continue
            elif tracker.time_since_update > 0:
                tracker.mark_lost()
            trackers_to_keep.append(tracker)
        self._trackers = trackers_to_keep

        # Emit confirmed tracks updated this frame that pass the motion gates.
        cos_gate = math.cos(math.radians(self._params.max_motion_angle))
        results = []
        for tracker in self._trackers:
            if tracker.state != TrackState.CONFIRMED or tracker.time_since_update != 0:
                continue
            if tracker.displacement < self._params.disp_gate:
                continue
            # Velocity-coherence gate: a real mover travels along its velocity; a
            # noisy false-flow cluster's centroid drifts off its erratic velocity.
            speed, cos_ang = tracker.motion_coherence()
            if speed < self._params.min_speed or cos_ang < cos_gate:
                continue
            results.append(tracker.to_tracked_detection())
        return results

    def get_all_tracks(self) -> list[TrackedDetection]:
        """All active tracks regardless of state (tentative/confirmed/lost)."""
        return [t.to_tracked_detection() for t in self._trackers]

    def _associate(
        self, detections: list[Measurement]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Hungarian assignment with gating. Returns (matched, unmatched_dets,
        unmatched_trackers)."""
        if len(self._trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self._trackers)))

        if self._params.use_bev_iou:
            # cost = 1 - IoU; accept only positive overlap (strict < gate).
            cost_matrix = compute_bev_iou_cost_matrix(self._trackers, detections)
            gate, inclusive = 1.0, False
        else:
            # Euclidean distance; accept within the gate (inclusive, like lidar).
            cost_matrix = compute_distance_cost_matrix(self._trackers, detections)
            gate, inclusive = self._params.gate_distance, True

        gated_cost = cost_matrix.copy()
        gated_cost[cost_matrix > gate if inclusive else cost_matrix >= gate] = 1e6
        row_indices, col_indices = linear_sum_assignment(gated_cost)

        matched = [
            (row, col)
            for row, col in zip(row_indices, col_indices)
            if (cost_matrix[row, col] <= gate if inclusive else cost_matrix[row, col] < gate)
        ]
        matched_trk = {m[0] for m in matched}
        matched_det = {m[1] for m in matched}
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det]
        unmatched_trackers = [i for i in range(len(self._trackers)) if i not in matched_trk]
        return matched, unmatched_detections, unmatched_trackers
