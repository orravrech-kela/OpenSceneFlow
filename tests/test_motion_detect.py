"""Tests for the flow-based detector improvements (clustering, yaw, de-translation)."""

import numpy as np
import pytest

from src.motion import detect_frame, fit_box


class TestVelocityAwareClustering:
    def test_splits_overlapping_objects_by_velocity(self) -> None:
        # Two point sets at the SAME location, moving in opposite directions.
        rng = np.random.RandomState(0)
        a = rng.uniform(-0.2, 0.2, (20, 3))
        b = rng.uniform(-0.2, 0.2, (20, 3))
        xyz = np.vstack([a, b])
        vel = np.vstack([np.tile([2.0, 0, 0], (20, 1)),
                         np.tile([-2.0, 0, 0], (20, 1))])
        window = [(0, xyz, vel)]
        kw = dict(eps=0.6, min_samples=6, min_points=12, min_frames=1)

        # Position-only clustering merges them.
        d0 = detect_frame(window, 0, "0", vel_weight=0.0, **kw)
        assert len(d0) == 1
        # Velocity-aware clustering separates them.
        d1 = detect_frame(window, 0, "0", vel_weight=1.0, **kw)
        assert len(d1) == 2


class TestDeTranslation:
    def test_window_detranslation_collapses_moving_blob(self) -> None:
        # A blob moving at v=(1,0,0): observed positions are shifted by off*v,
        # so de-translation (p - off*v) must collapse them onto one cluster.
        rng = np.random.RandomState(1)
        base = rng.uniform(-0.2, 0.2, (15, 3))
        v = np.array([1.0, 0.0, 0.0])
        window = [(off, base + off * v, np.tile(v, (15, 1))) for off in (-1, 0, 1)]
        dets = detect_frame(window, 0, "0", eps=0.5, min_samples=5,
                            min_points=12, min_frames=3, vel_weight=0.0)
        assert len(dets) == 1
        assert dets[0].points.shape[0] == 45
        assert dets[0].velocity == pytest.approx([1.0, 0.0, 0.0])


class TestCurrentFrameSupport:
    def test_rejects_pure_smear(self) -> None:
        # Object only present in PAST frames (offsets -3,-2,-1), nothing at t=0.
        rng = np.random.RandomState(5)
        base = rng.uniform(-0.2, 0.2, (20, 3))
        v = np.array([1.0, 0.0, 0.0])
        window = [(off, base + off * v, np.tile(v, (20, 1))) for off in (-3, -2, -1)]
        kw = dict(eps=0.5, min_samples=5, min_points=12, min_frames=2)
        # Gate on (default): no current-frame support -> not detected.
        assert detect_frame(window, 0, "0", min_current_points=3, **kw) == []
        # Gate off: the smear is detected (the old behavior).
        assert len(detect_frame(window, 0, "0", min_current_points=0, **kw)) == 1


class TestFitBox:
    def test_yaw_from_flow_when_moving(self) -> None:
        rng = np.random.RandomState(2)
        pts = rng.uniform(-0.5, 0.5, (30, 3))
        vel = np.array([np.cos(0.5), np.sin(0.5), 0.0])  # speed 1.0 >= thresh
        box, _, _, _ = fit_box(pts, vel=vel, speed_yaw_thresh=0.15)
        assert box[6] == pytest.approx(0.5, abs=1e-6)

    def test_yaw_from_geometry_when_slow(self) -> None:
        # Elongated along x; near-zero velocity -> yaw from minAreaRect, not flow.
        pts = np.zeros((40, 3))
        pts[:, 0] = np.linspace(-2, 2, 40)
        pts[:, 1] = np.random.RandomState(3).uniform(-0.05, 0.05, 40)
        box, _, _, _ = fit_box(pts, vel=np.array([0.01, 0.0, 0.0]), speed_yaw_thresh=0.15)
        # Long axis is x, so heading is ~0 (mod pi).
        assert abs(np.sin(box[6])) < 0.1
