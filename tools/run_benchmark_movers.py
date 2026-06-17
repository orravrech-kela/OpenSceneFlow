"""Run detect_track_movers across a benchmark tree of LiDAR sequences.

Walks ``--benchmark-dir`` for sequences (directories containing both a
``<pred-subdir>/`` and ``polar/`` folder), runs the motion-detection pipeline
on each, and writes results in a layout the offline viewer can pick up with
``--detections-root``:

    <results-root>/
      <run-name>/
        run_metadata.json
        summary.json
        sequences/
          <rel-path-from-benchmark>/
            detections.json   -- offline_viewer compatible
            tracks.json       -- full track list
            <frame_id>.npz    -- per-frame boxes/velocities

Example:
    .venv/bin/python tools/run_benchmark_movers.py \\
        --benchmark-dir /mnt/data/lidar/pbench \\
        --results-root  /mnt/data/lidar/bench_results/pbench \\
        --run-name      movers_deltaflow5f \\
        --pred-subdir   pred_flow
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
    # benchmark layout
    ap.add_argument("--benchmark-dir", type=Path, required=True)
    ap.add_argument("--results-root", type=Path, required=True)
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--pred-subdir", type=str, default="pred_flow_deltaflow5f")
    # candidate / accumulation
    ap.add_argument("--tau", type=float, default=0.08)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--use-snp", action="store_true")
    # clustering
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--min-samples", type=int, default=6)
    ap.add_argument("--min-points", type=int, default=12)
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--max-range", type=float, default=120.0)
    ap.add_argument("--vel-weight", type=float, default=1.0, help="velocity weight in clustering (0 = position-only)")
    ap.add_argument("--speed-yaw", type=float, default=0.15, help="min |vel_xy| (m/frame) to derive yaw from flow")
    ap.add_argument("--min-current-points", type=int, default=2, help="min points at target frame to emit (rejects temporal smear; 0 disables)")
    # tracking
    ap.add_argument("--max-dist", type=float, default=2.5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=4)
    ap.add_argument("--disp-gate", type=float, default=0.5)
    ap.add_argument("--min-speed", type=float, default=0.05, help="min coherent BEV speed (m/frame) to emit a track")
    ap.add_argument("--max-motion-angle", type=float, default=60.0, help="max angle (deg) between net displacement and velocity to emit (180 disables)")
    ap.add_argument("--use-bev-iou", action="store_true", help="associate by BEV-IoU instead of distance")
    ap.add_argument("--no-flow-meas", action="store_true", help="don't fuse flow velocity as a KF measurement")
    # frame range
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    return ap.parse_args()


def discover_sequences(benchmark_dir: Path, pred_subdir: str) -> list[Path]:
    """Return sequence dirs that have <pred_subdir>/ and polar/ subdirs."""
    seqs = []
    for root, dirs, _ in os.walk(benchmark_dir):
        root_path = Path(root)
        if (root_path / pred_subdir).is_dir() and (root_path / "polar").is_dir():
            seqs.append(root_path)
            dirs.clear()  # don't recurse into matched sequence
    return sorted(seqs)


def run_sequence(
    seq: Path,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict:
    pred_dir = seq / args.pred_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = list_frame_ids(pred_dir)
    sel = ids[args.start:]
    if args.limit > 0:
        sel = sel[: args.limit]

    # Streaming window: load frames on demand, evict outside window.
    cands: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _ensure_loaded(j: int) -> None:
        if j < 0 or j >= len(ids) or j in cands:
            return
        xyz, vel, _ = load_candidates(seq, args.pred_subdir, ids[j], args.tau, args.use_snp)
        cands[j] = (xyz, vel)

    per_frame: dict[str, tuple[int, list[TrackedDetection]]] = {}
    # Fresh tracker per sequence; reset() also restarts the global track-ID counter.
    tracker = MultiObjectTracker(MOTParams(
        gate_distance=args.max_dist,
        min_hits=args.min_hits,
        max_age=args.max_misses,
        disp_gate=args.disp_gate,
        min_speed=args.min_speed,
        max_motion_angle=args.max_motion_angle,
        use_bev_iou=args.use_bev_iou,
        kalman_params=KalmanTrackerParams(dt=1.0, use_flow_measurement=not args.no_flow_meas),
    ))
    tracker.reset()
    n_emitted = 0
    for j in range(args.start, args.start + len(sel)):
        rid = ids[j]
        # Load frames needed for this window.
        for off in range(-args.window, args.window + 1):
            _ensure_loaded(j + off)
        # Evict frames that are no longer needed.
        evict_before = j - args.window
        for k in list(cands.keys()):
            if k < evict_before:
                del cands[k]

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
            min_current_points=args.min_current_points,
        )
        emitted = tracker.update([d.to_measurement() for d in dets])
        per_frame[rid] = (j, emitted)
        n_emitted += len(emitted)
        local_j = j - args.start
        if local_j % 100 == 0:
            print(f"  [{local_j}/{len(sel)}] {rid} dets_so_far={n_emitted}", flush=True)

    _write_outputs(out_dir, per_frame)
    n_tracks = len({d.track_id for _, ds in per_frame.values() for d in ds})
    return {"n_frames": len(sel), "n_detections": n_emitted, "n_tracks": n_tracks}


def _write_outputs(out_dir: Path, per_frame: dict) -> None:
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

    # detections.json: lidar inference `save_results` schema (CVAT-import /
    # offline_viewer ready) -- `class`, int `track_id`, flat velocity + superset.
    results = []
    for rid, (_, dets) in per_frame.items():
        results.append({
            "frame_id": rid,
            "frame_idx": int(rid),
            "detections": [d.to_cvat_dict() for d in dets],
        })
    with (out_dir / "detections.json").open("w") as f:
        json.dump({"results": results}, f, indent=1)


def main() -> int:
    args = parse_args()
    run_root = args.results_root / args.run_name
    seq_root = run_root / "sequences"
    seq_root.mkdir(parents=True, exist_ok=True)

    seqs = discover_sequences(args.benchmark_dir, args.pred_subdir)
    if not seqs:
        print(f"[error] no sequences found under {args.benchmark_dir} with {args.pred_subdir}/", file=sys.stderr)
        return 1
    # Sort smallest-first by frame count for faster early results.
    seqs.sort(key=lambda s: len(list((s / args.pred_subdir).glob("*.npz"))))
    print(f"[info] found {len(seqs)} sequences", flush=True)

    run_meta = {
        "benchmark_dir": str(args.benchmark_dir),
        "pred_subdir": args.pred_subdir,
        "tau": args.tau,
        "window": args.window,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "min_points": args.min_points,
        "min_frames": args.min_frames,
        "max_range": args.max_range,
        "vel_weight": args.vel_weight,
        "speed_yaw_thresh": args.speed_yaw,
        "min_current_points": args.min_current_points,
        "max_dist": args.max_dist,
        "min_hits": args.min_hits,
        "max_misses": args.max_misses,
        "disp_gate": args.disp_gate,
        "min_speed": args.min_speed,
        "max_motion_angle": args.max_motion_angle,
        "use_bev_iou": args.use_bev_iou,
        "use_flow_measurement": not args.no_flow_meas,
        "kalman_dt": 1.0,
        "tracker_api": "MultiObjectTracker",
    }
    with (run_root / "run_metadata.json").open("w") as f:
        json.dump(run_meta, f, indent=2)

    summary = []
    t0_total = time.time()
    for seq in seqs:
        rel = seq.relative_to(args.benchmark_dir)
        out_dir = seq_root / rel
        if (out_dir / "detections.json").exists():
            print(f"\n[skip] {rel} (already done)", flush=True)
            summary.append({"sequence": str(rel), "status": "skipped"})
            continue
        print(f"\n[seq] {rel}", flush=True)
        t0 = time.time()
        try:
            stats = run_sequence(seq, args, out_dir)
            elapsed = time.time() - t0
            stats["status"] = "ok"
            stats["elapsed_s"] = round(elapsed, 2)
            print(f"  -> {stats['n_detections']} dets / {stats['n_tracks']} tracks in {elapsed:.1f}s", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            stats = {"status": "failed", "error": str(e), "elapsed_s": round(elapsed, 2)}
            print(f"  [FAILED] {e}", file=sys.stderr)
            traceback.print_exc()
        summary.append({"sequence": str(rel), **stats})

    with (run_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    n_ok = sum(1 for s in summary if s["status"] == "ok")
    n_fail = len(summary) - n_ok
    print(f"\n[done] {n_ok}/{len(summary)} sequences ok, {n_fail} failed in {time.time()-t0_total:.1f}s -> {run_root}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
