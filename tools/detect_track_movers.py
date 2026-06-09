"""Detect and track moving objects in an Innoviz sequence from predicted flow.

Pipeline: candidate motion mask -> temporal accumulation -> 3D DBSCAN ->
oriented box -> flow-guided tracking with a net-displacement gate.

Outputs per-frame detections (JSON + npz). Load detections.json in
offline_viewer.py with --detections for visualization.

Example:
    .venv/bin/python tools/detect_track_movers.py \\
        --seq /mnt/data/lidar/processed/gan_shomron_27_11_2025/sprint \\
        --pred-subdir pred_flow_deltaflow5f --limit 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PARENT_DIR)

from src.motion import (  # noqa: E402
    KalmanTrackerParams,
    MOTParams,
    MultiObjectTracker,
    TrackedDetection,
    detect_frame,
)
from src.motion.loader import list_frame_ids, load_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=Path, required=True, help="sequence dir under processed/")
    ap.add_argument("--pred-subdir", type=str, default="pred_flow_deltaflow5f")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default <seq>/movers_<pred-subdir>)")
    # candidate / accumulation
    ap.add_argument("--tau", type=float, default=0.08, help="min |flow| (m/frame) for a moving candidate")
    ap.add_argument("--window", type=int, default=3, help="temporal half-window (frames each side)")
    ap.add_argument("--use-snp", action="store_true", help="also drop snow/noise-filtered pixels")
    # clustering
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--min-samples", type=int, default=6)
    ap.add_argument("--min-points", type=int, default=12)
    ap.add_argument("--min-frames", type=int, default=2, help="distinct source frames a cluster must span")
    ap.add_argument("--max-range", type=float, default=120.0)
    ap.add_argument("--vel-weight", type=float, default=1.0, help="velocity weight in clustering (0 = position-only)")
    ap.add_argument("--speed-yaw", type=float, default=0.15, help="min |vel_xy| (m/frame) to derive yaw from flow")
    # tracking
    ap.add_argument("--max-dist", type=float, default=2.5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=4)
    ap.add_argument("--disp-gate", type=float, default=0.5, help="net translation (m) before a track is emitted")
    ap.add_argument("--use-bev-iou", action="store_true", help="associate by BEV-IoU instead of distance")
    ap.add_argument("--no-flow-meas", action="store_true", help="don't fuse flow velocity as a KF measurement")
    # run
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pred_dir = args.seq / args.pred_subdir
    if not pred_dir.is_dir():
        print(f"[error] no prediction dir: {pred_dir}", file=sys.stderr)
        return 1
    out_dir = args.out or (args.seq / f"movers_{args.pred_subdir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list_frame_ids(pred_dir)
    sel = ids[args.start:]
    if args.limit > 0:
        sel = sel[: args.limit]
    print(f"[info] {len(ids)} frames; processing {len(sel)} ({sel[0]}..{sel[-1]})", flush=True)

    # Candidate points are cheap (only moving pixels) -> preload for window access.
    cands = {}
    for j, rid in enumerate(ids):
        if j < args.start - args.window or j > args.start + len(sel) + args.window:
            continue
        xyz, vel, _ = load_candidates(args.seq, args.pred_subdir, rid, args.tau, args.use_snp)
        cands[j] = (xyz, vel)

    per_frame: dict[str, tuple[int, list[TrackedDetection]]] = {}
    tracker = MultiObjectTracker(MOTParams(
        gate_distance=args.max_dist,
        min_hits=args.min_hits,
        max_age=args.max_misses,
        disp_gate=args.disp_gate,
        use_bev_iou=args.use_bev_iou,
        kalman_params=KalmanTrackerParams(dt=1.0, use_flow_measurement=not args.no_flow_meas),
    ))
    n_emitted = 0
    for j in range(args.start, args.start + len(sel)):
        rid = ids[j]
        window = []
        for off in range(-args.window, args.window + 1):
            k = j + off
            if k in cands:
                xyz, vel = cands[k]
                window.append((off, xyz, vel))
        dets = detect_frame(
            window, j, rid,
            eps=args.eps, min_samples=args.min_samples, min_points=args.min_points,
            min_frames=args.min_frames, max_range=args.max_range,
            vel_weight=args.vel_weight, speed_yaw_thresh=args.speed_yaw,
        )
        emitted = tracker.update([d.to_measurement() for d in dets])
        per_frame[rid] = (j, emitted)
        n_emitted += len(emitted)
        if emitted:
            ids_str = ",".join(
                f"#{d.track_id}(|v|={float(np.linalg.norm([d.vx, d.vy, d.vz])):.2f})" for d in emitted
            )
            print(f"  {rid}: {len(dets)} clusters -> {len(emitted)} tracked [{ids_str}]", flush=True)

    write_outputs(out_dir, per_frame)
    n_tracks = len({d.track_id for _, ds in per_frame.values() for d in ds})
    print(f"[done] {n_emitted} detections across {n_tracks} tracks -> {out_dir}", flush=True)
    return 0


def write_outputs(out_dir: Path, per_frame: dict) -> None:
    summary = []
    for rid, (frame_idx, dets) in per_frame.items():
        for d in dets:
            summary.append(dict(
                raw_id=rid, frame_idx=frame_idx, track_id=d.track_id,
                box=[round(v, 4) for v in (d.x, d.y, d.z, d.dx, d.dy, d.dz, d.heading)],
                velocity=[round(v, 4) for v in (d.vx, d.vy, d.vz)],
                num_points=d.num_points,
                state=d.track_state.value, age=d.age, hits=d.hits,
                time_since_update=d.time_since_update, displacement=round(d.displacement, 4),
            ))
        np.savez_compressed(
            out_dir / f"{rid}.npz",
            track_ids=np.array([d.track_id for d in dets], dtype=np.int32),
            boxes=np.array([[d.x, d.y, d.z, d.dx, d.dy, d.dz, d.heading] for d in dets],
                           dtype=np.float32).reshape(-1, 7),
            velocities=np.array([[d.vx, d.vy, d.vz] for d in dets],
                                dtype=np.float32).reshape(-1, 3),
        )
    with (out_dir / "tracks.json").open("w") as f:
        json.dump(summary, f, indent=1)

    _write_viewer_detections(out_dir, per_frame)


def _write_viewer_detections(out_dir: Path, per_frame: dict) -> None:
    """Write detections.json compatible with offline_viewer.py --detections.

    Each detection is TrackedDetection.to_dict() -- a superset of the viewer
    schema (original keys preserved, lidar lifecycle fields added).
    """
    results = []
    for rid, (_, dets) in per_frame.items():
        results.append({
            "frame_id": rid,
            "frame_idx": int(rid),
            "detections": [d.to_dict() for d in dets],
        })
    with (out_dir / "detections.json").open("w") as f:
        json.dump({"results": results}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
