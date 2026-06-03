"""Headless Open3D (EGL) renderer: context cloud + per-track boxes, top-down BEV.

Mirrors the offscreen pattern in ``dataprocess/viz_ground_coverage.py``.
"""

from __future__ import annotations

import os
import sys

if sys.platform != "darwin":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

from src.motion.detect import BOX_LINES, Detection

_BG = (0.04, 0.04, 0.06, 1.0)
_CLOUD_RGB = (0.55, 0.55, 0.60)
# distinct, high-contrast palette cycled by track id
_PALETTE = np.array([
    [0.96, 0.26, 0.21], [0.30, 0.69, 0.31], [0.13, 0.59, 0.95],
    [1.00, 0.76, 0.03], [0.61, 0.15, 0.69], [0.00, 0.74, 0.83],
    [1.00, 0.34, 0.13], [0.55, 0.76, 0.29], [0.91, 0.12, 0.39],
    [0.40, 0.23, 0.72],
], dtype=np.float64)


def track_color(track_id: int) -> np.ndarray:
    return _PALETTE[track_id % len(_PALETTE)]


def _topdown(center: np.ndarray, half: float, fov: float) -> Tuple[list, list, list, float]:
    cx, cy, cz = center
    height = cz + half / np.tan(np.deg2rad(fov / 2.0)) * 1.1
    return [cx, cy, height], [cx, cy, cz], [1.0, 0.0, 0.0], fov


def fit_topdown_camera(
    centers: np.ndarray, margin: float = 6.0, fov: float = 60.0,
    min_half: float = 12.0, max_half: float = 45.0,
) -> Tuple[list, list, list, float]:
    """Fixed top-down view robust to far outlier tracks (median center, p90 span)."""
    if centers.shape[0] == 0:
        return _topdown(np.array([30.0, 0.0, 0.0]), 40.0, fov)
    c = np.median(centers, axis=0)
    half = float(np.percentile(np.abs(centers[:, :2] - c[:2]), 90)) + margin
    return _topdown(c, float(np.clip(half, min_half, max_half)), fov)


def follow_camera(center: np.ndarray, half: float = 18.0, fov: float = 60.0):
    return _topdown(center, half, fov)


class MoverRenderer:
    def __init__(self, width: int = 1280, height: int = 960, point_size: float = 2.0):
        self.r = rendering.OffscreenRenderer(width, height)
        self._cloud_mat = rendering.MaterialRecord()
        self._cloud_mat.shader = "defaultUnlit"
        self._cloud_mat.point_size = point_size
        self._pts_mat = rendering.MaterialRecord()
        self._pts_mat.shader = "defaultUnlit"
        self._pts_mat.point_size = point_size + 6.0
        self._line_mat = rendering.MaterialRecord()
        self._line_mat.shader = "unlitLine"
        self._line_mat.line_width = 5.0

    def render(
        self,
        cloud: np.ndarray,
        dets: List[Detection],
        camera: Tuple[list, list, list, float],
    ) -> np.ndarray:
        scene = self.r.scene
        scene.clear_geometry()
        scene.set_background(list(_BG))

        if cloud.shape[0]:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud.astype(np.float64))
            pcd.paint_uniform_color(list(_CLOUD_RGB))
            scene.add_geometry("cloud", pcd, self._cloud_mat)

        for i, det in enumerate(dets):
            col = track_color(det.track_id)
            if det.points.shape[0]:
                mp = o3d.geometry.PointCloud()
                mp.points = o3d.utility.Vector3dVector(det.points.astype(np.float64))
                mp.paint_uniform_color(list(col))
                scene.add_geometry(f"pts_{i}", mp, self._pts_mat)

            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(det.corners3d())
            ls.lines = o3d.utility.Vector2iVector(np.array(BOX_LINES))
            ls.colors = o3d.utility.Vector3dVector(np.tile(col, (len(BOX_LINES), 1)))
            scene.add_geometry(f"box_{i}", ls, self._line_mat)

        eye, target, up, fov = camera
        self.r.setup_camera(fov, target, eye, up)
        img = self.r.render_to_image()
        return np.asarray(img)
