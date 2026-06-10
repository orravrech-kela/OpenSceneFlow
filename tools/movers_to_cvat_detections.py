#!/usr/bin/env python3
"""Convert movers detections.json (OSF tracker) to the lidar inference schema.

The lidar ``run_streaming_inference.save_results`` schema is the CVAT-import /
offline_viewer contract. The OSF tracker already emits all the geometry CVAT
needs (class, x/y/z, dx/dy/dz, heading, track_id); this normalizes the cosmetic
differences so the output is a drop-in for the same downstream tooling:

  class        <- "class" / "class_name"   (or --class override)
  score, x, y, z, dx, dy, dz, heading      copied
  track_id     <- int(track_id)            (OSF emits a string)
  vel_x, vel_y <- velocity.vx, velocity.vy (flat; CVAT ignores velocity)
  track_vel_x, track_vel_y <- velocity.vx, velocity.vy

The emitted detection is a SUPERSET (keeps the OSF lifecycle fields too), so it
stays valid for the offline viewer as well.

Usage:
    # single file
    python tools/movers_to_cvat_detections.py --input <seq>/detections.json --output out.json
    # a whole benchmark run (mirrors the sequences/ layout)
    python tools/movers_to_cvat_detections.py --input <run_dir> --output <run_dir_cvat> [--class Pedestrian]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_detection(det: dict, class_override: str | None) -> dict:
    vel = det.get("velocity") or {}
    vx = float(vel.get("vx", det.get("vx", 0.0)))
    vy = float(vel.get("vy", det.get("vy", 0.0)))
    raw_id = det.get("track_id")
    try:
        track_id = int(raw_id)
    except (TypeError, ValueError):
        track_id = raw_id
    out = {
        "class": class_override or det.get("class") or det.get("class_name"),
        "score": det.get("score", 1.0),
        "x": det["x"], "y": det["y"], "z": det["z"],
        "dx": det["dx"], "dy": det["dy"], "dz": det["dz"],
        "heading": det["heading"],
        "vel_x": vx, "vel_y": vy,
        "track_id": track_id,
        "track_vel_x": vx, "track_vel_y": vy,
    }
    # Keep OSF lifecycle extras (superset; ignored by CVAT/viewer).
    for k in ("state", "age", "hits", "time_since_update", "displacement", "num_points"):
        if k in det:
            out[k] = det[k]
    return out


def convert_file(src: Path, dst: Path, class_override: str | None) -> int:
    data = json.loads(src.read_text())
    out: dict = {"source": str(src), "tracking_enabled": True, "results": []}
    for fr in data.get("results", []):
        dets = fr.get("detections", [])
        out["results"].append({
            "frame_idx": fr.get("frame_idx"),
            "frame_id": fr.get("frame_id", str(fr.get("frame_idx"))),
            "num_points": fr.get("num_points", 0),
            "detections": [convert_detection(d, class_override) for d in dets],
        })
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))
    return sum(len(f["detections"]) for f in out["results"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="a detections.json file OR a benchmark run dir (with sequences/**/detections.json)")
    ap.add_argument("--output", type=Path, required=True,
                    help="output file (single input) or output dir (mirrors the input's detections.json layout)")
    ap.add_argument("--class", dest="class_override", default=None,
                    help="override the label for every box (e.g. a CVAT task label to pre-fill)")
    args = ap.parse_args()

    if args.input.is_file():
        n = convert_file(args.input, args.output, args.class_override)
        print(f"[done] {n} detections -> {args.output}")
        return 0

    srcs = sorted(args.input.glob("sequences/**/detections.json")) or \
        sorted(args.input.glob("**/detections.json"))
    if not srcs:
        print(f"[error] no detections.json under {args.input}")
        return 1
    total = 0
    for s in srcs:
        rel = s.relative_to(args.input)
        total += convert_file(s, args.output / rel, args.class_override)
    print(f"[done] {len(srcs)} files, {total} detections -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
