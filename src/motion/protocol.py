"""Tracker protocol defining the interface all tracker implementations satisfy.

Mirrors the lidar ``core.tracking`` Tracker protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.motion.types import Measurement, TrackedDetection


@runtime_checkable
class Tracker(Protocol):
    def update(self, detections: list[Measurement]) -> list[TrackedDetection]: ...

    @property
    def last_matches(self) -> dict[int, Measurement]: ...

    def reset(self) -> None: ...
