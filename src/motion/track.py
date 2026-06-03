"""Flow-guided tracker: constant-velocity prediction + Hungarian association.

Predicted flow gives each detection an instantaneous velocity, so prediction is
``center += velocity`` (m/frame). A track is only *emitted* once it is both
confirmed (>= min_hits) and has net-translated past ``disp_gate`` — this is what
removes static structures that carry a persistent but spurious flow prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.motion.detect import Detection


@dataclass
class Track:
    id: int
    box: np.ndarray
    velocity: np.ndarray
    first_center: np.ndarray
    hits: int = 1
    misses: int = 0

    @property
    def center(self) -> np.ndarray:
        return self.box[:3]

    @property
    def displacement(self) -> float:
        return float(np.linalg.norm(self.box[:3] - self.first_center))


class Tracker:
    def __init__(
        self,
        max_dist: float = 2.5,
        min_hits: int = 3,
        max_misses: int = 4,
        disp_gate: float = 0.5,
    ):
        self.max_dist = max_dist
        self.min_hits = min_hits
        self.max_misses = max_misses
        self.disp_gate = disp_gate
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, dets: List[Detection]) -> List[Detection]:
        for t in self.tracks:
            t.box = t.box.copy()
            t.box[:3] = t.box[:3] + t.velocity   # constant-velocity predict

        matches, un_tracks, un_dets = self._associate(dets)

        emitted: List[Detection] = []
        for ti, di in matches:
            t = self.tracks[ti]
            det = dets[di]
            t.box = det.box.copy()
            t.velocity = det.velocity.copy()
            t.hits += 1
            t.misses = 0
            if t.hits >= self.min_hits and t.displacement >= self.disp_gate:
                det.track_id = t.id
                emitted.append(det)

        for ti in un_tracks:
            self.tracks[ti].misses += 1
        for di in un_dets:
            det = dets[di]
            self.tracks.append(
                Track(id=self._next_id, box=det.box.copy(),
                      velocity=det.velocity.copy(), first_center=det.box[:3].copy())
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return emitted

    def _associate(self, dets: List[Detection]):
        if not self.tracks or not dets:
            return [], list(range(len(self.tracks))), list(range(len(dets)))
        tc = np.array([t.center for t in self.tracks])
        dc = np.array([d.box[:3] for d in dets])
        cost = np.linalg.norm(tc[:, None, :] - dc[None, :, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        matches, mt, md = [], set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= self.max_dist:
                matches.append((r, c))
                mt.add(r)
                md.add(c)
        un_tracks = [i for i in range(len(self.tracks)) if i not in mt]
        un_dets = [i for i in range(len(dets)) if i not in md]
        return matches, un_tracks, un_dets
