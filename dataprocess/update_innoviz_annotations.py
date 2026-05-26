"""Refresh ``flow_*`` datasets in extracted innoviz H5s after a relabel.

When the operator updates `annotations/ground_truth.json` (or `manual_gt.json`)
for a sequence — fixing a box, adding a track, changing a class, etc. — only
the four annotation-derived datasets per frame need to change:

    flow, flow_is_valid, flow_category_indices, flow_instance_id

The other per-frame datasets (``lidar``, ``pose``, ``ground_mask``,
``ego_motion``) are functions of the polar data + LineFit, which did not
change. So we can rewrite *only* the flow datasets, avoiding the polar→
cartesian conversion and LineFit re-run. Two practical wins:

* ~3–5× faster than a full re-extract per sequence.
* No LineFit thread-non-determinism: ``ground_mask`` stays bit-exact.

How frames are reconciled with the new annotations:

* **kept**   (frame in both H5 and new annotations) → flow recomputed in place.
* **removed**(frame in H5 only)                     → group dropped from output.
* **added**  (frame in new annotations only)        → ABORT. ``lidar`` /
  ``ground_mask`` are missing for these frames; needs a full re-extract.

The script writes to a sibling ``.h5.tmp`` file and atomically renames at the
end, so an interrupted run never leaves the H5 in a partial state.

``index_total.pkl`` and ``index_flow.pkl`` are NOT touched — timestamps are
unchanged for kept frames, and removed frames are absent from both, so the
existing indices remain consistent. (If you want a clean rebuild, run
``create_reading_index`` against the split dir afterwards.)

Usage::

    # Dry-run on a few sequences
    python dataprocess/update_innoviz_annotations.py \
        --sequences "gan_shomron_27_11_2025/sprint, meginim_11_02_2026/tirgolet16"

    # Apply
    python dataprocess/update_innoviz_annotations.py \
        --sequences "gan_shomron_27_11_2025/sprint, meginim_11_02_2026/tirgolet16" \
        --execute
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import h5py
import numpy as np
from tqdm import tqdm

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from dataprocess.extract_innoviz import (  # noqa: E402
    DEFAULT_BOX_PAD,
    DEFAULT_FRAME_PERIOD_NS,
    _annotations_path,
    _enumerate_frames,
    _parse_annotations,
    _per_point_flow,
)


def _scene_id(day: str, seq: str) -> str:
    return f"{day}__{seq}".replace("/", "_")


def _find_h5(output_dir: Path, scene_id: str) -> Path:
    """Probe train/ and val/ for the sequence's H5; error if neither or both exist."""
    candidates = [output_dir / sp / f"{scene_id}.h5" for sp in ("train", "val")]
    hits = [c for c in candidates if c.exists()]
    if not hits:
        raise FileNotFoundError(
            f"No H5 for scene_id={scene_id} under {output_dir}/train or {output_dir}/val"
        )
    if len(hits) > 1:
        raise RuntimeError(
            f"Scene {scene_id} exists in both splits: {hits}. Resolve manually."
        )
    return hits[0]


def update_one(
    day: str,
    seq: str,
    data_root: Path,
    output_dir: Path,
    *,
    frame_period_ns: int = DEFAULT_FRAME_PERIOD_NS,
    box_pad: float = DEFAULT_BOX_PAD,
    execute: bool = False,
) -> str:
    """Refresh flow datasets for one sequence. Returns a one-line status."""
    scene_id = _scene_id(day, seq)
    seq_dir = data_root / day / seq
    if not seq_dir.is_dir():
        return f"{scene_id}: SKIP — sequence dir missing ({seq_dir})"
    h5_path = _find_h5(output_dir, scene_id)

    # Strict POLAR-NUM lookup; raises with pointer to convert_annotations_to_global.py
    # if any annotation id doesn't resolve to a polar file.
    tracks, _ = _parse_annotations(_annotations_path(seq_dir))
    frame_pairs = _enumerate_frames(seq_dir)
    expected_ts = [str(idx * frame_period_ns) for idx, _ in frame_pairs]
    expected_set = set(expected_ts)

    with h5py.File(h5_path, "r") as f_old:
        existing_ts = set(f_old.keys())

    kept = expected_set & existing_ts
    added = expected_set - existing_ts
    removed = existing_ts - expected_set

    if added:
        sample = sorted(added, key=int)[:3]
        return (
            f"{scene_id}: ABORT — {len(added)} frames present in new annotations "
            f"but missing from H5 (sample timestamps: {sample}). "
            f"Run a full re-extract for this sequence — lidar/ground_mask "
            f"haven't been computed for these frames."
        )

    if not execute:
        return f"{scene_id}: DRY-RUN — {len(kept)} updated, {len(removed)} removed"

    # frame_idx → next-frame_idx for flow computation (sorted, monotonic).
    sorted_pairs = sorted(frame_pairs, key=lambda p: p[0])
    frame_idx_list = [p[0] for p in sorted_pairs]
    next_idx = dict(zip(frame_idx_list, frame_idx_list[1:]))

    tmp_path = h5_path.with_suffix(".h5.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    # Per-track-id instance counter, matching process_sequence's auto-assignment
    # (see extract_innoviz.py:474). The lambda captures `dclass` by name so each
    # new track gets the current length as its instance index.
    dclass: Dict[int, int] = defaultdict(lambda: len(dclass))

    with h5py.File(h5_path, "r") as f_old, h5py.File(tmp_path, "w") as f_new:
        for ts in sorted(expected_ts, key=int):
            g_old = f_old[ts]
            g_new = f_new.create_group(ts)

            # Verbatim copies (polar-derived; unchanged by relabel).
            for key in ("lidar", "pose", "ground_mask"):
                g_new.create_dataset(key, data=g_old[key][:])

            frame_idx = int(ts) // frame_period_ns
            if frame_idx not in next_idx:
                # Last frame in the sequence — no flow datasets to write.
                continue

            # ego_motion stayed identity; copy verbatim.
            if "ego_motion" in g_old:
                g_new.create_dataset("ego_motion", data=g_old["ego_motion"][:])

            xyz0 = g_old["lidar"][:]
            flow, valid, classes, instances, _ = _per_point_flow(
                xyz0,
                tracks,
                frame_idx,
                next_idx[frame_idx],
                box_pad,
                dclass,
            )
            g_new.create_dataset("flow", data=flow.astype(np.float32))
            g_new.create_dataset("flow_is_valid", data=valid.astype(bool))
            g_new.create_dataset("flow_category_indices", data=classes.astype(np.uint8))
            g_new.create_dataset("flow_instance_id", data=instances.astype(np.int16))

    # Atomic on POSIX within the same directory.
    tmp_path.replace(h5_path)
    return f"{scene_id}: UPDATED — {len(kept)} frames, {len(removed)} removed"


def _worker(args: dict) -> str:
    try:
        return update_one(**args)
    except Exception as e:  # surface per-sequence errors without killing the pool
        return f"{args['day']}/{args['seq']}: ERROR — {type(e).__name__}: {e}"


def main(
    data_root: str = "/mnt/data/lidar/processed",
    output_dir: str = "/mnt/data/lidar/h5/innoviz",
    sequences: str = "",
    num_workers: int = 1,
    frame_period_ns: int = DEFAULT_FRAME_PERIOD_NS,
    box_pad: float = DEFAULT_BOX_PAD,
    execute: bool = False,
):
    """CLI entry point. See module docstring for examples."""
    if not sequences:
        raise ValueError(
            "--sequences must be a non-empty comma-separated list of "
            "'<recording_day>/<sequence_path>' entries"
        )
    items = [s.strip() for s in sequences.split(",") if s.strip()]
    parsed: List[Tuple[str, str]] = []
    for entry in items:
        if "/" not in entry:
            raise ValueError(
                f"Invalid sequence entry: {entry!r} (expected '<day>/<seq>')"
            )
        day, rest = entry.split("/", 1)
        parsed.append((day, rest))

    mode = "EXECUTING" if execute else "DRY-RUN"
    print(f"{mode} on {len(parsed)} sequence(s)")
    tasks = [
        dict(
            day=day,
            seq=seq,
            data_root=Path(data_root),
            output_dir=Path(output_dir),
            frame_period_ns=frame_period_ns,
            box_pad=box_pad,
            execute=execute,
        )
        for (day, seq) in parsed
    ]
    start = time.time()
    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            for msg in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), ncols=100):
                tqdm.write(msg)
    else:
        for t in tqdm(tasks, ncols=100):
            tqdm.write(_worker(t))
    print(f"\nDone in {time.time() - start:.1f}s")
    if not execute:
        print("DRY-RUN: pass --execute to apply.")


if __name__ == "__main__":
    fire.Fire(main)
