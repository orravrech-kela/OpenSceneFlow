"""Tests for the flow-based multi-object tracker (mirrors lidar test_tracking.py)."""

import numpy as np
import pytest

from src.motion import (
    KalmanBoxTracker,
    KalmanTrackerParams,
    MOTParams,
    Measurement,
    MultiObjectTracker,
    Tracker,
    TrackedDetection,
    TrackState,
    compute_bev_iou_cost_matrix,
    compute_distance_cost_matrix,
)


def meas(x, y, z, dx=1.0, dy=1.0, dz=1.0, heading=0.0, vx=0.0, vy=0.0, vz=0.0):
    return Measurement("mover", 1.0, x, y, z, dx, dy, dz, heading, vx, vy, vz)


class TestTrackState:
    def test_track_states_exist(self) -> None:
        assert TrackState.TENTATIVE.value == "TENTATIVE"
        assert TrackState.CONFIRMED.value == "CONFIRMED"
        assert TrackState.LOST.value == "LOST"


class TestTrackedDetection:
    def test_to_dict_is_viewer_superset(self) -> None:
        tracked = TrackedDetection(
            track_id=1, class_name="mover", score=1.0,
            x=10.0, y=5.0, z=0.8, dx=0.5, dy=0.5, dz=1.7, heading=1.57,
            track_state=TrackState.CONFIRMED, age=10, hits=8, time_since_update=0,
            vx=1.0, vy=0.5, vz=0.0, displacement=2.3, num_points=42,
        )
        d = tracked.to_dict()
        # viewer-critical keys preserved verbatim
        assert d["class_name"] == "mover"
        assert d["track_id"] == "1"  # string for the offline viewer
        assert d["x"] == 10.0 and d["heading"] == 1.57
        # lidar-parity additions
        assert d["class"] == "mover"
        assert d["state"] == "CONFIRMED"
        assert d["velocity"] == {"vx": 1.0, "vy": 0.5, "vz": 0.0}
        assert d["age"] == 10 and d["hits"] == 8 and d["time_since_update"] == 0
        assert d["displacement"] == 2.3 and d["num_points"] == 42


class TestKalmanBoxTracker:
    @pytest.fixture(autouse=True)
    def reset_counter(self) -> None:
        KalmanBoxTracker.reset_counter()

    def test_initialization(self) -> None:
        trk = KalmanBoxTracker(meas(10.0, 5.0, 0.8, 0.5, 0.5, 1.7, 1.57))
        assert trk.track_id == 1
        assert trk.class_name == "mover"
        assert trk.age == 1 and trk.hits == 1 and trk.time_since_update == 0
        assert trk.state == TrackState.TENTATIVE
        assert trk.position == pytest.approx((10.0, 5.0, 0.8))
        assert trk.dimensions == pytest.approx((0.5, 0.5, 1.7))

    def test_velocity_seeded_from_flow(self) -> None:
        # Unlike lidar (seeds 0), OSF seeds velocity from the flow measurement.
        trk = KalmanBoxTracker(meas(0, 0, 0, vx=2.0, vy=1.0, vz=0.0))
        assert trk.velocity == pytest.approx((2.0, 1.0, 0.0))

    def test_unique_track_ids(self) -> None:
        a, b, c = (KalmanBoxTracker(meas(0, 0, 0)) for _ in range(3))
        assert (a.track_id, b.track_id, c.track_id) == (1, 2, 3)

    def test_predict_updates_age(self) -> None:
        trk = KalmanBoxTracker(meas(0, 0, 0))
        trk.predict()
        assert trk.age == 2 and trk.time_since_update == 1

    def test_predict_propagates_velocity(self) -> None:
        trk = KalmanBoxTracker(meas(0, 0, 0, vx=2.0, vy=1.0, vz=0.0),
                               KalmanTrackerParams(dt=1.0))
        trk.predict()
        x, y, z = trk.position
        assert x == pytest.approx(2.0, abs=0.5)
        assert y == pytest.approx(1.0, abs=0.5)
        assert z == pytest.approx(0.0, abs=0.5)

    def test_update_resets_time_since_update(self) -> None:
        trk = KalmanBoxTracker(meas(0, 0, 0))
        trk.predict()
        assert trk.time_since_update == 1
        trk.update(meas(1, 0, 0))
        assert trk.time_since_update == 0 and trk.hits == 2

    def test_heading_wraps_across_pi(self) -> None:
        # Object pointing near +pi; measurements jitter across the +/-pi seam.
        trk = KalmanBoxTracker(meas(0, 0, 0, 4, 2, 2, heading=3.10),
                               KalmanTrackerParams(use_flow_measurement=False))
        for h in (3.13, -3.13, 3.12, -3.10):
            trk.predict()
            trk.update(meas(0, 0, 0, 4, 2, 2, heading=h))
            assert -np.pi <= trk.heading <= np.pi
        # Stays near +/-pi (continuous), does not collapse toward 0.
        assert abs(abs(trk.heading) - np.pi) < 0.3

    def test_to_tracked_detection(self) -> None:
        trk = KalmanBoxTracker(meas(10, 5, 0.8, 0.5, 0.5, 1.7, 1.57))
        trk.confirm()
        td = trk.to_tracked_detection()
        assert isinstance(td, TrackedDetection)
        assert td.track_id == trk.track_id
        assert td.track_state == TrackState.CONFIRMED


class TestAssociation:
    @pytest.fixture(autouse=True)
    def reset_counter(self) -> None:
        KalmanBoxTracker.reset_counter()

    def test_empty_trackers_returns_empty_matrix(self) -> None:
        assert compute_distance_cost_matrix([], [meas(0, 0, 0)]).shape == (0, 1)

    def test_empty_detections_returns_empty_matrix(self) -> None:
        trackers = [KalmanBoxTracker(meas(0, 0, 0))]
        assert compute_distance_cost_matrix(trackers, []).shape == (1, 0)

    def test_distance_cost_matrix(self) -> None:
        trackers = [KalmanBoxTracker(meas(0, 0, 0)), KalmanBoxTracker(meas(10, 0, 0))]
        cost = compute_distance_cost_matrix(trackers, [meas(0.5, 0, 0), meas(9.5, 0, 0)])
        assert cost.shape == (2, 2)
        assert cost[0, 0] < cost[0, 1]
        assert cost[1, 1] < cost[1, 0]

    def test_bev_iou_cost_matrix(self) -> None:
        trackers = [KalmanBoxTracker(meas(0, 0, 0, 4, 2, 2, 0.0))]
        cost = compute_bev_iou_cost_matrix(
            trackers, [meas(0, 0, 0, 4, 2, 2, 0.0), meas(20, 0, 0, 4, 2, 2, 0.0)]
        )
        assert cost.shape == (1, 2)
        assert cost[0, 0] < 0.1            # overlapping -> low cost (high IoU)
        assert cost[0, 1] == pytest.approx(1.0)  # disjoint -> cost 1 (IoU 0)


class TestMultiObjectTracker:
    @pytest.fixture(autouse=True)
    def reset_counter(self) -> None:
        KalmanBoxTracker.reset_counter()

    def test_protocol_runtime_checkable(self) -> None:
        assert isinstance(MultiObjectTracker(), Tracker)

    def test_initialization(self) -> None:
        tracker = MultiObjectTracker()
        assert tracker.num_tracks == 0 and tracker.frame_count == 0

    def test_first_detection_creates_track(self) -> None:
        tracker = MultiObjectTracker()
        results = tracker.update([meas(0, 0, 0)])
        assert len(results) == 0  # tentative
        assert tracker.num_tracks == 1

    def test_track_confirmed_after_min_hits(self) -> None:
        # disp_gate=0 isolates confirmation from the static-structure gate.
        tracker = MultiObjectTracker(MOTParams(min_hits=3, max_age=5, disp_gate=0.0))
        assert tracker.update([meas(0, 0, 0)]) == []
        assert tracker.update([meas(0, 0, 0)]) == []
        results = tracker.update([meas(0, 0, 0)])
        assert len(results) == 1
        assert results[0].track_state == TrackState.CONFIRMED

    def test_track_deleted_after_max_age(self) -> None:
        tracker = MultiObjectTracker(MOTParams(min_hits=1, max_age=3, disp_gate=0.0))
        tracker.update([meas(0, 0, 0)])
        assert tracker.num_tracks == 1
        for _ in range(4):
            tracker.update([])
        assert tracker.num_tracks == 0

    def test_correct_association(self) -> None:
        tracker = MultiObjectTracker(
            MOTParams(min_hits=1, max_age=5, gate_distance=2.0, disp_gate=0.0)
        )
        tracker.update([meas(0, 0, 0), meas(10, 0, 0)])
        results = tracker.update([meas(0.5, 0, 0), meas(9.5, 0, 0)])
        assert len(results) == 2
        assert {r.track_id for r in results} == {1, 2}

    def test_unmatched_detection_creates_new_track(self) -> None:
        tracker = MultiObjectTracker(
            MOTParams(min_hits=1, max_age=5, gate_distance=2.0, disp_gate=0.0)
        )
        tracker.update([meas(0, 0, 0)])
        tracker.update([meas(0, 0, 0), meas(100, 0, 0)])
        assert tracker.num_tracks == 2

    def test_disp_gate_suppresses_static_emits_moving(self) -> None:
        params = MOTParams(min_hits=2, max_age=5, gate_distance=3.0, disp_gate=0.5)
        tracker = MultiObjectTracker(params)
        # Static (spurious-flow) structure: confirmed but never emitted.
        tracker.update([meas(0, 0, 0, vx=0.0)])
        assert tracker.update([meas(0, 0, 0, vx=0.0)]) == []
        # Genuinely moving object crosses the gate and is emitted.
        tracker.reset()
        results: list = []
        for i in range(3):
            results = tracker.update([meas(i * 1.0, 0, 0, vx=1.0)])
        assert len(results) == 1
        assert results[0].displacement >= 0.5

    def test_bev_iou_association(self) -> None:
        tracker = MultiObjectTracker(
            MOTParams(min_hits=1, max_age=5, use_bev_iou=True, disp_gate=0.0)
        )
        tracker.update([meas(0, 0, 0, 4, 2, 2, 0.0)])
        res = tracker.update([meas(0.3, 0, 0, 4, 2, 2, 0.0)])  # overlaps -> matched
        assert len(res) == 1 and res[0].track_id == 1
        # disjoint detection -> no IoU overlap -> new track
        tracker.update([meas(0.3, 0, 0, 4, 2, 2, 0.0), meas(50, 0, 0, 4, 2, 2, 0.0)])
        assert tracker.num_tracks == 2

    def test_velocity_estimation_over_time(self) -> None:
        tracker = MultiObjectTracker(
            MOTParams(min_hits=1, max_age=10, gate_distance=5.0, disp_gate=0.0)
        )
        results: list = []
        for i in range(15):
            results = tracker.update([meas(i * 1.0, 0, 0, vx=1.0)])
        assert len(results) == 1
        assert results[0].vx == pytest.approx(1.0, abs=0.3)
        assert abs(results[0].vy) < 0.3 and abs(results[0].vz) < 0.3

    def test_last_matches(self) -> None:
        tracker = MultiObjectTracker(MOTParams(min_hits=1, disp_gate=0.0))
        tracker.update([meas(0, 0, 0)])
        tracker.update([meas(0.1, 0, 0)])
        assert set(tracker.last_matches.keys()) == {1}

    def test_reset(self) -> None:
        tracker = MultiObjectTracker()
        tracker.update([meas(0, 0, 0)])
        assert tracker.num_tracks == 1 and tracker.frame_count == 1
        tracker.reset()
        assert tracker.num_tracks == 0 and tracker.frame_count == 0

    def test_get_all_tracks_includes_tentative(self) -> None:
        tracker = MultiObjectTracker(MOTParams(min_hits=3))
        assert tracker.update([meas(0, 0, 0)]) == []
        all_tracks = tracker.get_all_tracks()
        assert len(all_tracks) == 1
        assert all_tracks[0].track_state == TrackState.TENTATIVE


class TestTrackLifecycle:
    @pytest.fixture(autouse=True)
    def reset_counter(self) -> None:
        KalmanBoxTracker.reset_counter()

    def test_full_lifecycle(self) -> None:
        tracker = MultiObjectTracker(MOTParams(min_hits=2, max_age=2, disp_gate=0.0))
        det = meas(0, 0, 0)

        assert tracker.update([det]) == []  # birth -> tentative
        assert tracker.get_all_tracks()[0].track_state == TrackState.TENTATIVE

        results = tracker.update([det])  # confirm
        assert len(results) == 1 and results[0].track_state == TrackState.CONFIRMED

        assert tracker.update([]) == []  # miss -> lost (not returned)
        assert tracker.get_all_tracks()[0].track_state == TrackState.LOST

        tracker.update([])  # still within max_age
        assert tracker.num_tracks == 1

        tracker.update([])  # exceeds max_age -> deleted
        assert tracker.num_tracks == 0
