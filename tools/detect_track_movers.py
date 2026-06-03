"""Detect and track moving objects in an Innoviz sequence from predicted flow.

Pipeline: candidate motion mask -> temporal accumulation -> 3D DBSCAN ->
oriented box -> flow-guided tracking with a net-displacement gate.

Outputs per-frame detections (JSON + npz) and, with --viz, a top-down mp4.

Example:
    .venv/bin/python tools/detect_track_movers.py \\
        --seq /mnt/data/lidar/processed/gan_shomron_27_11_2025/sprint \\
        --pred-subdir pred_flow_deltaflow5f --viz --limit 120
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

from src.motion.detect import Detection, detect_frame  # noqa: E402
from src.motion.frontend import list_frame_ids, load_candidates, load_full_cloud  # noqa: E402
from src.motion.track import Tracker  # noqa: E402


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
    # tracking
    ap.add_argument("--max-dist", type=float, default=2.5)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-misses", type=int, default=4)
    ap.add_argument("--disp-gate", type=float, default=0.5, help="net translation (m) before a track is emitted")
    # run / viz
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--follow", action="store_true", help="per-frame follow cam instead of a fixed view")
    ap.add_argument("--view-half", type=float, default=18.0, help="follow-cam half-extent (m)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
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
    sel_set = {i for i in range(args.start, args.start + len(sel))}
    print(f"[info] {len(ids)} frames; processing {len(sel)} ({sel[0]}..{sel[-1]})", flush=True)

    # Candidate points are cheap (only moving pixels) -> preload for window access.
    cands = {}
    for j, rid in enumerate(ids):
        if j < args.start - args.window or j > args.start + len(sel) + args.window:
            continue
        xyz, vel, _ = load_candidates(args.seq, args.pred_subdir, rid, args.tau, args.use_snp)
        cands[j] = (xyz, vel)

    per_frame: dict[str, list[Detection]] = {}
    tracker = Tracker(args.max_dist, args.min_hits, args.max_misses, args.disp_gate)
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
        )
        emitted = tracker.update(dets)
        per_frame[rid] = emitted
        n_emitted += len(emitted)
        if emitted:
            ids_str = ",".join(f"#{d.track_id}(|v|={np.linalg.norm(d.velocity):.2f})" for d in emitted)
            print(f"  {rid}: {len(dets)} clusters -> {len(emitted)} tracked [{ids_str}]", flush=True)

    write_outputs(out_dir, per_frame)
    n_tracks = len({d.track_id for ds in per_frame.values() for d in ds})
    print(f"[done] {n_emitted} detections across {n_tracks} tracks -> {out_dir}", flush=True)

    if args.viz:
        render_video(args, ids, per_frame, out_dir)
    return 0


def write_outputs(out_dir: Path, per_frame: dict) -> None:
    summary = []
    for rid, dets in per_frame.items():
        for d in dets:
            summary.append(dict(
                raw_id=rid, frame_idx=d.frame_idx, track_id=d.track_id,
                box=[round(float(v), 4) for v in d.box],
                velocity=[round(float(v), 4) for v in d.velocity],
                num_points=int(d.points.shape[0]),
            ))
        np.savez_compressed(
            out_dir / f"{rid}.npz",
            track_ids=np.array([d.track_id for d in dets], dtype=np.int32),
            boxes=np.array([d.box for d in dets], dtype=np.float32).reshape(-1, 7),
            velocities=np.array([d.velocity for d in dets], dtype=np.float32).reshape(-1, 3),
        )
    with (out_dir / "tracks.json").open("w") as f:
        json.dump(summary, f, indent=1)

    _write_viewer_detections(out_dir, per_frame)


def _write_viewer_detections(out_dir: Path, per_frame: dict) -> None:
    """Write detections.json compatible with offline_viewer.py --detections."""
    results = []
    for rid, dets in per_frame.items():
        frame_dets = []
        for d in dets:
            cx, cy, cz, l, w, h, yaw = d.box
            frame_dets.append({
                "class_name": "mover",
                "score": 1.0,
                "x": round(float(cx), 4),
                "y": round(float(cy), 4),
                "z": round(float(cz), 4),
                "dx": round(float(l), 4),
                "dy": round(float(w), 4),
                "dz": round(float(h), 4),
                "heading": round(float(yaw), 4),
                "track_id": str(d.track_id),
            })
        results.append({"frame_id": rid, "frame_idx": int(rid), "detections": frame_dets})
    with (out_dir / "detections.json").open("w") as f:
        json.dump({"results": results}, f, indent=1)


def render_video(args, ids, per_frame, out_dir) -> None:
    import cv2
    from src.motion.render import MoverRenderer, fit_topdown_camera, follow_camera

    centers = np.array([d.box[:3] for ds in per_frame.values() for d in ds]).reshape(-1, 3)
    fixed_cam = fit_topdown_camera(centers)
    renderer = MoverRenderer(args.width, args.height)
    out_mp4 = out_dir / "movers.mp4"
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (args.width, args.height))
    sel = ids[args.start: args.start + (args.limit or len(ids))]
    last_center = np.median(centers, axis=0) if centers.shape[0] else np.array([30.0, 0.0, 0.0])
    for rid in sel:
        cloud = load_full_cloud(args.seq, rid)
        dets = per_frame.get(rid, [])
        if args.follow:
            if dets:
                last_center = max(dets, key=lambda d: d.points.shape[0]).box[:3]
            camera = follow_camera(last_center, half=args.view_half)
        else:
            camera = fixed_cam
        rgb = renderer.render(cloud, dets, camera)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        n = len(per_frame.get(rid, []))
        cv2.putText(bgr, f"{rid}  objects={n}", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(bgr)
    vw.release()
    print(f"[viz] {out_mp4}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
