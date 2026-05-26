"""Extract Innoviz LiDAR recordings into the OpenSceneFlow HDF5 schema.

Mirrors `dataprocess/extract_nus.py` but for the custom Innoviz dataset stored at
`/mnt/data/lidar/processed/<recording_day>/<sequence_name>/` with:

  * `polar/<frame_id>.npz` — (480, 1200) `distance`, `reflectivity`, `pixel_time`.
  * `lut.npz`              — shared LUT, `unit_vec` shape (480, 1200, 3) so that
                              cartesian = distance[..., None] * unit_vec.
  * `fg/<frame_id>.npz`    — `is_foreground` mask (unused here; reserved for SSL).
  * `annotations/ground_truth.json` (preferred) or `annotations/manual_gt.json`
                                    (fallback) — CVAT cuboid_3d annotations with
                                    per-frame items keyed by `attr.frame`
                                    (sequence-relative index).

Sensor is statically mounted, so ego_pose == identity and ego_flow == 0 everywhere.
Per-point flow comes from rigid box-to-box transforms applied to interior points.

The output HDF5 layout follows `extract_nus.py::create_group_data`:
  group `<timestamp_ns>`:
    lidar               (N, 3) float32
    pose                (4, 4) float32
    ground_mask         (N,)   bool
    ego_motion          (4, 4) float32     # absent on the last frame
    flow                (N, 3) float32     # absent on the last frame
    flow_is_valid       (N,)   bool        # absent on the last frame
    flow_category_indices (N,) uint8       # absent on the last frame
    flow_instance_id    (N,)   int16       # absent on the last frame

After all sequences in a split are written, `index_total.pkl` and `index_flow.pkl`
are created next to the .h5 files via `dataprocess/misc_data.py::create_reading_index`.

Usage:
    python dataprocess/extract_innoviz.py \
        --data-root /mnt/data/lidar/processed \
        --output-dir /mnt/data/lidar/h5/innoviz \
        --splits-file conf/innoviz_splits.yaml \
        --split both \
        --num-workers 4
"""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fire
import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from dataprocess.misc_data import check_h5py_file_exists, create_reading_index  # noqa: E402
from src.utils.innoviz_eval import (  # noqa: E402
    INNOVIZ_CATEGORY_TO_INDEX,
    canonical_label,
    category_index,
)

DEFAULT_GROUND_CONFIG = f"{PARENT_DIR}/conf/others/innoviz.toml"
DEFAULT_SPLITS_FILE = f"{PARENT_DIR}/conf/innoviz_splits.yaml"
DEFAULT_FRAME_PERIOD_NS = 100_000_000  # 10 Hz nominal
DEFAULT_BOX_PAD = 1.1
MIN_POINTS_IN_BOX = 5
DEFAULT_Z_FLOOR: Optional[float] = None  # supplemental ground cutoff disabled by default
AUTO_HEIGHT_PERCENTILE = 5.0           # robust estimator: sensor-frame ground ≈ p5(Z)
# Below STATIC_DISP_EPS (m) we treat a 1-frame box motion as zero — both because
# numerical noise in T1 @ T0^-1 can produce ~1e-14 m fake flow, and because
# annotation jitter rarely encodes meaningful sub-cm motion. CVAT's `is_static`
# attribute, when present, takes precedence regardless of magnitude.
STATIC_DISP_EPS = 0.01
# Above MAX_PLAUSIBLE_DISP (m / frame) we treat the box pair as an annotation
# glitch (e.g. observed 17 m teleport in sprint). At 10 Hz this caps speed at
# 30 m/s ≈ 108 km/h, well above any plausible pedestrian/vehicle in our data.
MAX_PLAUSIBLE_DISP = 3.0

IDENTITY_4 = np.eye(4, dtype=np.float32)


def _autodetect_linefit_params(xyz: np.ndarray) -> Dict[str, float]:
    """Estimate per-sequence LineFit ground-segmentation parameters from the cloud.

    Anchor: ``height = -p5(Z)`` — sensor mount height above expected ground plane
    (5th-percentile Z is robust to object returns and to occasional cliff/sensor
    outliers below p1).

    Ground-band thickness estimator: ``spread = p15(Z) - p5(Z)``. This is the
    vertical extent of the bottom decile of returns and serves as a proxy for
    how non-flat the ground is. A flat sports field gives ~0.9 m; rolling
    outdoor terrain with slopes gives 2–4 m; a deep cliff sequence gives more.

    The thickness is then mapped onto the three LineFit parameters that scale
    with terrain roughness:

      * ``max_start_height``: how far below the expected plane LineFit may anchor
        its first ground line — must be ≥ the spread for the band to be reachable.
      * ``max_long_height``: tolerated Z jump between consecutive radial points;
        scales roughly with half the spread.
      * ``max_dist_to_line``: lateral tolerance for points labelled ground —
        scales with one-fifth of the spread, clamped to a usable range.

    ``max_slope`` is **not** auto-derived (it depends on lateral terrain rise/run,
    not purely on Z statistics) and is kept at the toml default.
    """
    z = xyz[:, 2]
    p5 = float(np.percentile(z, 5))
    p15 = float(np.percentile(z, 15))
    spread = max(0.0, p15 - p5)
    return {
        "height": float(-p5),
        # Floors of 2.0 / 1.0 / 0.3 reproduce the sprint-tuned defaults exactly;
        # ceilings keep the estimator from blowing up on hovering-over-valley scenes.
        "max_start_height": float(np.clip(spread, 2.0, 15.0)),
        "max_long_height": float(np.clip(spread / 2.0, 1.0, 5.0)),
        "max_dist_to_line": float(np.clip(spread / 5.0, 0.3, 0.8)),
    }


# Backwards-compatible thin wrapper used by callers that only want the height.
def _autodetect_height(xyz: np.ndarray) -> float:
    return _autodetect_linefit_params(xyz)["height"]


_PARAM_LINE_RES = {
    "height":           re.compile(r"^(\s*height\s*=\s*)([^#\n]+?)(\s*(?:#.*)?)$",           re.MULTILINE),
    "max_start_height": re.compile(r"^(\s*max_start_height\s*=\s*)([^#\n]+?)(\s*(?:#.*)?)$", re.MULTILINE),
    "max_long_height":  re.compile(r"^(\s*max_long_height\s*=\s*)([^#\n]+?)(\s*(?:#.*)?)$",  re.MULTILINE),
    "max_dist_to_line": re.compile(r"^(\s*max_dist_to_line\s*=\s*)([^#\n]+?)(\s*(?:#.*)?)$", re.MULTILINE),
}


def _materialise_ground_config(base_toml: Path, params: Dict[str, float]) -> Path:
    """Copy ``base_toml`` to a tempfile with the named parameters substituted.

    LineFit reads its parameters from disk on construction, so per-sequence
    overrides need an on-disk file. Caller is responsible for cleaning up.
    Only keys present in :data:`_PARAM_LINE_RES` are honoured; missing keys
    raise ``RuntimeError`` so a typo can't silently fall through to the default.
    """
    text = base_toml.read_text()
    for key, value in params.items():
        regex = _PARAM_LINE_RES.get(key)
        if regex is None:
            raise RuntimeError(f"No substitution regex registered for LineFit param '{key}'")
        text, n = regex.subn(rf"\g<1>{value:.4f}\g<3>", text, count=1)
        if n == 0:
            raise RuntimeError(f"Could not find a '{key} = ...' line in {base_toml}")
    fd, tmp_path = tempfile.mkstemp(prefix="innoviz_ground_", suffix=".toml")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return Path(tmp_path)


@dataclass(frozen=True)
class BoxAnnotation:
    """Per-frame, per-track cuboid annotation, already canonicalised."""

    position: np.ndarray  # (3,) float64, sensor frame
    rotation: np.ndarray  # (3,) float64, Euler XYZ radians (CVAT convention)
    scale: np.ndarray     # (3,) float64, full extents (w, l, h)
    category_index: int   # mapped via INNOVIZ_CATEGORY_TO_INDEX
    is_static: bool       # CVAT attribute; True = parked / non-moving object


def _scene_id_for(day: str, seq: str) -> str:
    """Flatten ``day/seq`` into a single safe HDF5 file stem."""
    return f"{day}__{seq}".replace("/", "_")


def _read_polar_frame(seq_dir: Path, raw_id: str) -> np.ndarray:
    """Load distance from a polar npz; returns (H, W) float32."""
    with np.load(seq_dir / "polar" / f"{raw_id}.npz") as f:
        return f["distance"].astype(np.float32)


def _build_box_pose(position: np.ndarray, rotation_euler: np.ndarray) -> np.ndarray:
    """Return a (4,4) float64 SE(3) box-to-sensor transform for an oriented cuboid.

    CVAT cuboid_3d rotations are stored as `[rx, ry, rz]` in radians; we treat the
    triple as an intrinsic XYZ Euler. If a future inspection shows the convention
    is different, this is the only place to change.
    """
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_euler("xyz", rotation_euler, degrees=False).as_matrix()
    pose[:3, 3] = position
    return pose


def _se3_inverse(T: np.ndarray) -> np.ndarray:
    Tinv = np.eye(4, dtype=T.dtype)
    Tinv[:3, :3] = T[:3, :3].T
    Tinv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Tinv


def _annotations_path(seq_dir: Path) -> Path:
    """Prefer ground_truth.json (newer canonical export); fall back to manual_gt.json."""
    gt = seq_dir / "annotations" / "ground_truth.json"
    if gt.exists():
        return gt
    mgt = seq_dir / "annotations" / "manual_gt.json"
    if mgt.exists():
        return mgt
    raise FileNotFoundError(
        f"No annotations found under {seq_dir / 'annotations'}: "
        f"expected ground_truth.json or manual_gt.json"
    )


def _parse_annotations(
    annotations_path: Path,
) -> Tuple[Dict[int, Dict[int, BoxAnnotation]], int]:
    """Group CVAT annotations as ``tracks[track_id][frame_idx] = BoxAnnotation``.

    Returns the track map plus the largest frame index encountered in the file
    (so callers can pad sequences whose annotation file is shorter than expected).
    """
    with open(annotations_path) as f:
        data = json.load(f)
    label_names = [entry["name"] for entry in data["categories"]["label"]["labels"]]
    tracks: Dict[int, Dict[int, BoxAnnotation]] = defaultdict(dict)
    max_frame = -1
    for item in data["items"]:
        frame_idx = int(item["attr"]["frame"])
        max_frame = max(max_frame, frame_idx)
        for ann in item["annotations"]:
            if ann.get("type") != "cuboid_3d":
                continue
            label_id = int(ann["label_id"])
            if not (0 <= label_id < len(label_names)):
                continue
            canon = canonical_label(label_names[label_id])
            if canon is None:
                continue
            track_id = int(ann["attributes"]["track_id"])
            tracks[track_id][frame_idx] = BoxAnnotation(
                position=np.asarray(ann["position"], dtype=np.float64),
                rotation=np.asarray(ann["rotation"], dtype=np.float64),
                scale=np.asarray(ann["scale"], dtype=np.float64),
                category_index=category_index(canon),
                is_static=bool(ann["attributes"].get("is_static", False)),
            )
    return tracks, max_frame


def _enumerate_frames(seq_dir: Path) -> List[Tuple[int, str]]:
    """Discover available polar frames; returns sorted ``(frame_idx, raw_id)`` pairs.

    ``frame_idx`` is the polar-file integer (= ``attr.frame`` in annotations),
    used as the HDF5 group name basis via ``frame_idx * frame_period_ns``.
    ``raw_id`` is the polar filename stem (e.g. ``"00007321"``).

    Annotations must use polar-file numbering: ``id`` parses to an integer
    matching a polar stem. If you have a 0-based serial export from CVAT,
    normalize it first with
    ``algorithms/projects/lidar/scripts/data_prep/convert_annotations_to_global.py``.
    Items whose ``id`` doesn't resolve to any polar file are reported and the
    extraction fails fast — the silent positional fallback that used to live
    here masked real corruption (e.g. duplicate polar-frame extraction when
    the id range overlapped polar but was offset).
    """
    annotations_path = _annotations_path(seq_dir)
    with open(annotations_path) as f:
        data = json.load(f)
    polar_dir = seq_dir / "polar"
    polar_files = {p.stem: p for p in polar_dir.glob("*.npz")}
    polar_by_int = {int(stem): stem for stem in polar_files}
    pairs: List[Tuple[int, str]] = []
    unresolved: List[str] = []
    for item in data["items"]:
        try:
            raw_int = int(item["id"])
        except (TypeError, ValueError):
            continue
        frame_idx = int(item["attr"]["frame"])
        raw_id = polar_by_int.get(raw_int)
        if raw_id is None:
            unresolved.append(str(item["id"]))
            continue
        pairs.append((frame_idx, raw_id))
    if unresolved:
        sample = ", ".join(unresolved[:5])
        more = f" (+{len(unresolved)-5} more)" if len(unresolved) > 5 else ""
        raise RuntimeError(
            f"{annotations_path}: {len(unresolved)} annotation ids do not match any "
            f"polar file in {polar_dir} (e.g. {sample}{more}). Run "
            f"convert_annotations_to_global.py on this sequence first."
        )
    pairs.sort(key=lambda p: p[0])
    return pairs


def _per_point_flow(
    xyz0: np.ndarray,
    tracks: Dict[int, Dict[int, BoxAnnotation]],
    frame_t0: int,
    frame_t1: int,
    box_pad: float,
    dclass: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-point flow + auxiliary masks for frame pair ``(t0, t1)``.

    Returns ``(flow, valid, classes, instances, ego_motion)``. ``ego_motion`` is
    always identity for the static Innoviz sensor; we still emit it so the HDF5
    schema mirrors `extract_nus.py` exactly.
    """
    N = xyz0.shape[0]
    ego_motion = IDENTITY_4.copy()
    flow = np.zeros((N, 3), dtype=np.float32)
    valid = np.ones(N, dtype=bool)
    classes = np.zeros(N, dtype=np.uint8)
    instances = np.zeros(N, dtype=np.int16)
    obj_flow_mag = np.zeros(N, dtype=np.float32)

    for track_id, frames in tracks.items():
        box0 = frames.get(frame_t0)
        box1 = frames.get(frame_t1)
        if box0 is None or box1 is None:
            continue

        T0 = _build_box_pose(box0.position, box0.rotation)
        T0_inv = _se3_inverse(T0)

        # Test interior points in box-local frame at t0 with the standard 1.1x padding.
        local = (xyz0.astype(np.float64) - box0.position) @ T0[:3, :3]
        half_extent = (box0.scale * 0.5) * box_pad
        inside = np.all(np.abs(local) <= half_extent, axis=1)
        if not inside.any():
            continue

        # Always label the class for interior points, even when the object is
        # static or the box pair is rejected for flow purposes — this matches
        # extract_nus.py's behaviour and lets DeltaFlow learn "object → 0 flow".
        classes[inside] = box0.category_index

        n_inside = int(inside.sum())
        if n_inside < MIN_POINTS_IN_BOX:
            # Too few points to compute a reliable rigid-body flow; mark invalid.
            valid[inside] = False
            continue

        # Skip flow computation for objects tagged static by the annotator, or
        # for box pairs with implausible motion (annotation glitches), or with
        # sub-cm displacement (numerical/jitter noise). Flow stays at zero and
        # the instance_id is *not* set — matching what nuScenes/AV2 produce for
        # parked vehicles and what offline_viewer.py uses to decide overlay
        # membership (has_flow = instance_id > 0).
        centroid_disp = float(np.linalg.norm(box1.position - box0.position))
        if (
            box0.is_static
            or box1.is_static
            or centroid_disp < STATIC_DISP_EPS
            or centroid_disp > MAX_PLAUSIBLE_DISP
        ):
            continue

        # Rigid transform t0 → t1, applied to interior sensor-frame points.
        T1 = _build_box_pose(box1.position, box1.rotation)
        T_1_from_0 = T1 @ T0_inv
        pts0 = xyz0[inside].astype(np.float64)
        pts1 = pts0 @ T_1_from_0[:3, :3].T + T_1_from_0[:3, 3]
        delta = (pts1 - pts0).astype(np.float32)
        new_mag = np.linalg.norm(delta, axis=1).astype(np.float32)

        # Resolve overlapping boxes by keeping the larger-magnitude flow per point.
        # Mirrors extract_nus.py:185-194 (object-flow max selection).
        better = np.zeros(N, dtype=bool)
        better[inside] = new_mag > obj_flow_mag[inside]
        flow[better] = delta[better[inside]]
        obj_flow_mag[better] = new_mag[better[inside]]
        instance_idx = dclass[track_id] + 1
        instances[better] = instance_idx

    return flow, valid, classes, instances, ego_motion


def _create_group_data(
    group: h5py.Group,
    pc: np.ndarray,
    pose: np.ndarray,
    gm: np.ndarray,
    flow_0to1: Optional[np.ndarray] = None,
    flow_valid: Optional[np.ndarray] = None,
    flow_category: Optional[np.ndarray] = None,
    flow_instance: Optional[np.ndarray] = None,
    ego_motion: Optional[np.ndarray] = None,
) -> None:
    """Write one frame's datasets into ``group``; mirrors ``extract_nus.py``."""
    group.create_dataset("lidar", data=pc.astype(np.float32))
    group.create_dataset("pose", data=pose.astype(np.float32))
    group.create_dataset("ground_mask", data=gm.astype(bool))
    if ego_motion is not None:
        group.create_dataset("ego_motion", data=ego_motion.astype(np.float32))
    if flow_0to1 is not None:
        group.create_dataset("flow", data=flow_0to1.astype(np.float32))
        group.create_dataset("flow_is_valid", data=flow_valid.astype(bool))
        group.create_dataset("flow_category_indices", data=flow_category.astype(np.uint8))
        group.create_dataset("flow_instance_id", data=flow_instance.astype(np.int16))


def process_sequence(
    day: str,
    seq: str,
    data_root: Path,
    output_dir: Path,
    ground_config: str,
    frame_period_ns: int,
    z_floor: Optional[float],
    box_pad: float,
    max_frames: Optional[int] = None,
    auto_height: bool = True,
    height_override: Optional[float] = None,
) -> str:
    """Convert one Innoviz sequence into a single `<scene_id>.h5` file."""
    seq_dir = data_root / day / seq
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

    scene_id = _scene_id_for(day, seq)
    output_h5 = output_dir / f"{scene_id}.h5"

    lut_path = seq_dir / "lut.npz"
    if not lut_path.is_file():
        raise FileNotFoundError(f"Missing LUT: {lut_path}")
    with np.load(lut_path) as lut_npz:
        unit_vec = lut_npz["unit_vec"].astype(np.float32)  # (H, W, 3)

    frame_pairs = _enumerate_frames(seq_dir)
    if max_frames is not None:
        frame_pairs = frame_pairs[:max_frames]
    if len(frame_pairs) < 2:
        raise RuntimeError(f"Not enough usable polar frames for {scene_id}")

    timestamps = [str(frame_idx * frame_period_ns) for frame_idx, _ in frame_pairs]
    if check_h5py_file_exists(output_h5, timestamps):
        return f"skip {scene_id} (already complete)"

    tracks, _ = _parse_annotations(_annotations_path(seq_dir))

    # Decide LineFit params: explicit override > autodetect from frame 0 > toml default.
    tmp_config_path: Optional[Path] = None
    params: Optional[Dict[str, float]] = None
    if height_override is not None:
        params = {"height": float(height_override)}
    elif auto_height:
        _, first_raw_id = frame_pairs[0]
        distance0 = _read_polar_frame(seq_dir, first_raw_id)
        xyz0_all = (distance0[..., None] * unit_vec).reshape(-1, 3)
        keep0 = distance0.reshape(-1) > 0.0
        params = _autodetect_linefit_params(xyz0_all[keep0])
    if params is not None:
        tmp_config_path = _materialise_ground_config(Path(ground_config), params)
        effective_config = str(tmp_config_path)
        print(f"[{scene_id}] LineFit params: " + ", ".join(f"{k}={v:.2f}" for k, v in params.items()))
    else:
        effective_config = ground_config

    # Lazy linefit import — keeps test/inspection environments from needing the C++ ext.
    from linefit import ground_seg
    ground_seg_runner = ground_seg(effective_config)

    dclass: Dict[int, int] = defaultdict(lambda: len(dclass))

    if output_h5.exists():
        output_h5.unlink()  # clean overwrite, since check_h5py_file_exists deletes partials
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "a") as f:
        # Cache the previous-frame state so we only need to load each polar npz once.
        prev_xyz: Optional[np.ndarray] = None
        prev_frame_idx: Optional[int] = None
        prev_ground: Optional[np.ndarray] = None

        for i, (frame_idx, raw_id) in enumerate(frame_pairs):
            distance = _read_polar_frame(seq_dir, raw_id)
            xyz_all = distance[..., None] * unit_vec  # (H, W, 3)
            xyz_flat = xyz_all.reshape(-1, 3)
            keep = distance.reshape(-1) > 0.0
            xyz = xyz_flat[keep].astype(np.float32)

            ground = ground_seg_runner.run(xyz).astype(bool)
            if z_floor is not None:
                ground = ground | (xyz[:, 2] < z_floor)

            # Write the previous frame using the just-loaded current frame to derive flow.
            if prev_xyz is not None and prev_frame_idx is not None:
                flow, valid, classes, instances, ego_motion = _per_point_flow(
                    xyz0=prev_xyz,
                    tracks=tracks,
                    frame_t0=prev_frame_idx,
                    frame_t1=frame_idx,
                    box_pad=box_pad,
                    dclass=dclass,
                )
                ts0 = str(prev_frame_idx * frame_period_ns)
                grp = f.create_group(ts0)
                _create_group_data(
                    group=grp,
                    pc=prev_xyz,
                    pose=IDENTITY_4,
                    gm=prev_ground,
                    flow_0to1=flow,
                    flow_valid=valid,
                    flow_category=classes,
                    flow_instance=instances,
                    ego_motion=ego_motion,
                )

            prev_xyz = xyz
            prev_frame_idx = frame_idx
            prev_ground = ground

        # Last frame — no flow.
        assert prev_xyz is not None and prev_frame_idx is not None and prev_ground is not None
        ts_last = str(prev_frame_idx * frame_period_ns)
        grp = f.create_group(ts_last)
        _create_group_data(
            group=grp,
            pc=prev_xyz,
            pose=IDENTITY_4,
            gm=prev_ground,
        )

    if tmp_config_path is not None:
        tmp_config_path.unlink(missing_ok=True)
    return f"wrote {scene_id} ({len(frame_pairs)} frames)"


def _load_splits(splits_file: Path) -> Dict[str, List[Tuple[str, str]]]:
    with open(splits_file) as f:
        data = yaml.safe_load(f) or {}
    out: Dict[str, List[Tuple[str, str]]] = {}
    for key in ("train", "val"):
        items = data.get(key) or []
        parsed: List[Tuple[str, str]] = []
        for entry in items:
            if "/" not in entry:
                raise ValueError(f"Split entry must be '<day>/<seq>', got: {entry!r}")
            day, seq = entry.split("/", 1)
            parsed.append((day, seq))
        out[key] = parsed
    return out


def _worker(args: dict) -> str:
    return process_sequence(
        day=args["day"],
        seq=args["seq"],
        data_root=Path(args["data_root"]),
        output_dir=Path(args["output_dir"]),
        ground_config=args["ground_config"],
        frame_period_ns=args["frame_period_ns"],
        z_floor=args["z_floor"],
        box_pad=args["box_pad"],
        max_frames=args["max_frames"],
        auto_height=args["auto_height"],
        height_override=args["height_override"],
    )


def main(
    data_root: str = "/mnt/data/lidar/processed",
    output_dir: str = "/mnt/data/lidar/h5/innoviz",
    splits_file: str = DEFAULT_SPLITS_FILE,
    split: str = "both",  # "train" | "val" | "both"
    num_workers: int = 1,
    ground_config: str = DEFAULT_GROUND_CONFIG,
    frame_period_ns: int = DEFAULT_FRAME_PERIOD_NS,
    z_floor: Optional[float] = DEFAULT_Z_FLOOR,
    box_pad: float = DEFAULT_BOX_PAD,
    max_frames: Optional[int] = None,
    auto_height: bool = True,
    height: Optional[float] = None,
):
    """CLI entry point. See module docstring for example invocation."""
    splits = _load_splits(Path(splits_file))
    if split not in ("train", "val", "both"):
        raise ValueError("--split must be one of: train, val, both")
    active_splits = ["train", "val"] if split == "both" else [split]

    print(f"INNOVIZ_CATEGORY_TO_INDEX = {dict(INNOVIZ_CATEGORY_TO_INDEX)}")
    print(f"Output dir: {output_dir}")
    print(f"Splits file: {splits_file}")

    for sp in active_splits:
        seqs = splits[sp]
        if not seqs:
            print(f"--- split '{sp}' is empty, skipping (fill {splits_file})")
            continue
        sp_out = Path(output_dir) / sp
        sp_out.mkdir(parents=True, exist_ok=True)
        print(f"--- processing split '{sp}' ({len(seqs)} sequences) → {sp_out}")
        start = time.time()
        tasks = [
            dict(
                day=day,
                seq=seq,
                data_root=data_root,
                output_dir=str(sp_out),
                ground_config=ground_config,
                frame_period_ns=frame_period_ns,
                z_floor=z_floor,
                box_pad=box_pad,
                max_frames=max_frames,
                auto_height=auto_height,
                height_override=height,
            )
            for (day, seq) in seqs
        ]
        if num_workers > 1:
            with multiprocessing.Pool(processes=num_workers) as pool:
                for msg in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), ncols=100):
                    tqdm.write(msg)
        else:
            for t in tqdm(tasks, ncols=100):
                tqdm.write(_worker(t))
        print(f"--- split '{sp}' done in {time.time() - start:.1f}s; building indices...")
        create_reading_index(sp_out, flow_inside_check=False)
        create_reading_index(sp_out, flow_inside_check=True)


if __name__ == "__main__":
    fire.Fire(main)
