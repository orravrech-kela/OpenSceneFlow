"""Predicted-flow polar sidecars: sparse fp16 writer + dual-format loader.

Sparse format (v2), one ``<seq>/<subdir>/<raw_id>.npz`` per frame:
  - ``pix``  (M,)  uint32  — flat row-major indices into H*W where |flow| >= threshold
  - ``flow`` (M,3) float16 — predicted flow at those pixels
  - ``shape`` (2,) int32   — (H, W) of the polar grid

Equivalent to the legacy dense format (``flow`` (H,W,3) fp32 + ``has_flow``
(H,W) bool from ``dataprocess/save_pred_flow_masks.py``) for every consumer
that reads flow only at ``has_flow`` pixels — at ~1/20 the size. Sub-threshold
flow values are NOT stored (they reconstruct as zero), so pick the write-time
threshold no higher than the smallest detection tau you'll ever use.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def save_sparse_sidecar(path: Path, pix: np.ndarray, flow: np.ndarray, hw: Tuple[int, int]) -> None:
    np.savez_compressed(
        path,
        pix=pix.astype(np.uint32),
        flow=flow.astype(np.float16),
        shape=np.asarray(hw, dtype=np.int32),
    )


def load_flow_sidecar(path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (flow (H,W,3) float32, has_flow (H,W) bool) from either format."""
    with np.load(path) as d:
        if "pix" in d.files:
            h, w = (int(x) for x in d["shape"])
            flow = np.zeros((h * w, 3), dtype=np.float32)
            flow[d["pix"]] = d["flow"].astype(np.float32)
            has = np.zeros(h * w, dtype=bool)
            has[d["pix"]] = True
            return flow.reshape(h, w, 3), has.reshape(h, w)
        return d["flow"].astype(np.float32), d["has_flow"].astype(bool)


class PolarSidecarWriter:
    """Write per-frame sparse flow sidecars next to the raw polar data.

    Maps H5 ``scene_id``/``timestamp`` back to ``<seq_dir>/polar/<raw_id>.npz``
    by walking ``data_root`` for dirs containing ``polar/`` and rebuilding the
    extractor's scene_id (``day__seq-with-slashes-as-underscores``).
    """

    def __init__(self, data_root: str, subdir: str = "pred_flow",
                 threshold: float = 0.05, frame_period_ns: Optional[int] = None):
        # lazy import: keeps trainer importable without the dataprocess deps
        from dataprocess.extract_innoviz import DEFAULT_FRAME_PERIOD_NS, _scene_id_for

        self.data_root = Path(data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"save_sidecar_root does not exist: {data_root}")
        self.subdir = subdir
        self.threshold = float(threshold)
        self.frame_period_ns = int(frame_period_ns or DEFAULT_FRAME_PERIOD_NS)

        self._scene_dirs: Dict[str, Path] = {}
        for dirpath, dirnames, _ in os.walk(self.data_root):
            if "polar" not in dirnames:
                continue
            dirnames[:] = []  # sequence dir found — don't descend into its data
            seq_dir = Path(dirpath)
            rel = seq_dir.relative_to(self.data_root).as_posix()
            day, _, seq = rel.partition("/")
            scene_id = day if not seq else _scene_id_for(day, seq)
            if scene_id in self._scene_dirs:
                raise ValueError(
                    f"scene_id collision under {data_root}: '{scene_id}' maps to both "
                    f"{self._scene_dirs[scene_id]} and {seq_dir}")
            self._scene_dirs[scene_id] = seq_dir

        self._raw_ids: Dict[str, Dict[int, str]] = {}  # scene_id -> frame_idx -> raw_id
        self.n_written = 0
        self.n_skipped = 0

    def _frame_map(self, scene_id: str) -> Dict[int, str]:
        if scene_id not in self._raw_ids:
            from dataprocess.extract_innoviz import _enumerate_frames

            seq_dir = self._scene_dirs[scene_id]
            no_annotations = not (seq_dir / "annotations").is_dir()
            pairs = _enumerate_frames(seq_dir, no_annotations=no_annotations)
            self._raw_ids[scene_id] = dict(pairs)
            (seq_dir / self.subdir).mkdir(exist_ok=True)
        return self._raw_ids[scene_id]

    def write(self, scene_id: str, timestamp, flow: np.ndarray) -> bool:
        """Scatter (N,3) flow over keep pixels into the polar grid; write sidecar."""
        if scene_id not in self._scene_dirs:
            print(f"[sidecar] WARNING: scene_id '{scene_id}' not found under {self.data_root}, skipping")
            self.n_skipped += 1
            return False
        seq_dir = self._scene_dirs[scene_id]
        raw_id = self._frame_map(scene_id).get(int(timestamp) // self.frame_period_ns)
        if raw_id is None:
            print(f"[sidecar] WARNING: no polar frame for {scene_id} ts={timestamp}, skipping")
            self.n_skipped += 1
            return False

        with np.load(seq_dir / "polar" / f"{raw_id}.npz") as polar:
            distance = polar["distance"]
        h, w = distance.shape
        keep = distance.reshape(-1) > 0.0
        if int(keep.sum()) != flow.shape[0]:
            print(f"[sidecar] WARNING: {scene_id}/{raw_id}: {flow.shape[0]} predictions vs "
                  f"{int(keep.sum())} keep pixels — polar data changed since extraction? skipping")
            self.n_skipped += 1
            return False

        valid = np.linalg.norm(flow, axis=1) >= self.threshold
        pix = np.where(keep)[0][valid].astype(np.uint32)
        save_sparse_sidecar(seq_dir / self.subdir / f"{raw_id}.npz", pix, flow[valid], (h, w))
        self.n_written += 1
        return True

    def summary(self) -> str:
        return (f"wrote {self.n_written} sidecars under <seq>/{self.subdir}/ "
                f"(threshold={self.threshold}, skipped {self.n_skipped})")
