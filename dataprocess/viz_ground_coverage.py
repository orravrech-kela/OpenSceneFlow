"""Render ground-coverage percentile frames (p05/p50/p95) per innoviz H5 sequence.

For each .h5 under --data_root, compute per-frame ground_mask fraction, pick the
frames at p05/p50/p95, and save Open3D offscreen renders from the sensor
viewpoint plus a per-sequence histogram. A summary CSV is written at the root.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Headless EGL: surfaceless platform works on NVIDIA without an X server.
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from open3d.visualization import rendering


GROUND_RGB = (0.20, 0.80, 0.35)
NONGROUND_RGB = (0.75, 0.75, 0.78)


def per_frame_stats(h5_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (frame_keys_sorted, fractions, num_points) for one sequence H5."""
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k) if k.isdigit() else k)
        fracs = np.empty(len(keys), dtype=np.float32)
        npts = np.empty(len(keys), dtype=np.int64)
        for i, k in enumerate(keys):
            gm = f[k]["ground_mask"][:]
            npts[i] = len(gm)
            fracs[i] = float(gm.sum()) / max(len(gm), 1)
    return keys, fracs, npts


def pick_percentile_frames(
    keys: list[str], fracs: np.ndarray, percentiles=(5, 50, 95)
) -> list[tuple[int, str, float, float]]:
    """For each percentile, find the frame whose fraction is closest to it.

    Returns list of (index, key, fraction, target_percentile_value).
    """
    out = []
    for p in percentiles:
        target = float(np.percentile(fracs, p))
        idx = int(np.argmin(np.abs(fracs - target)))
        out.append((idx, keys[idx], float(fracs[idx]), target))
    return out


def load_frame(h5_path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        g = f[key]
        return g["lidar"][:], g["ground_mask"][:]


def render_frame(
    renderer: rendering.OffscreenRenderer,
    points: np.ndarray,
    ground_mask: np.ndarray,
    out_png: Path,
    point_size: float = 2.0,
) -> None:
    """Render one frame from the sensor viewpoint (camera at origin, +X forward).

    Coordinates are assumed lidar-frame: +X forward, +Y left, +Z up.
    """
    colors = np.where(
        ground_mask[:, None],
        np.array(GROUND_RGB, dtype=np.float32),
        np.array(NONGROUND_RGB, dtype=np.float32),
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    scene = renderer.scene
    scene.clear_geometry()
    scene.set_background([0.05, 0.05, 0.07, 1.0])
    scene.add_geometry("pcd", pcd, mat)

    # Slight elevation above the sensor so the ground in front is visible;
    # eye and target are both in the lidar frame.
    eye = [0.0, 0.0, 2.0]
    target = [25.0, 0.0, 0.0]
    up = [0.0, 0.0, 1.0]
    renderer.setup_camera(70.0, target, eye, up)

    img = renderer.render_to_image()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out_png), img)


def save_histogram(
    fracs: np.ndarray,
    picks: list[tuple[int, str, float, float]],
    out_png: Path,
    sequence_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    ax.hist(fracs, bins=40, color="#5a8fd4", edgecolor="#1f3b6b", alpha=0.85)
    for _, _, frac, _ in picks:
        ax.axvline(frac, color="#d4533a", linewidth=1.4, linestyle="--")
    ax.set_xlabel("ground fraction (per frame)")
    ax.set_ylabel("# frames")
    ax.set_title(sequence_name, fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def process_sequence(
    h5_path: Path,
    out_dir: Path,
    renderer: rendering.OffscreenRenderer,
    point_size: float,
    percentiles: tuple[int, ...],
    min_points: int,
    min_ground_frac: float,
) -> list[dict]:
    keys, fracs, npts = per_frame_stats(h5_path)
    if len(keys) == 0:
        return []
    # Drop warmup/junk frames so percentiles describe real scenes.
    valid = (npts >= min_points) & (fracs >= min_ground_frac)
    if valid.sum() == 0:
        print(
            f"  no valid frames after filter "
            f"(min_points={min_points}, min_ground_frac={min_ground_frac}); skipping",
            flush=True,
        )
        return []
    keys_v = [k for k, ok in zip(keys, valid) if ok]
    fracs_v = fracs[valid]
    picks = pick_percentile_frames(keys_v, fracs_v, percentiles=percentiles)
    rows = []
    for percentile, (_, key, frac, _) in zip(percentiles, picks):
        points, gm = load_frame(h5_path, key)
        out_png = out_dir / f"p{percentile:02d}_frac={frac:.3f}_frame={key}.png"
        render_frame(renderer, points, gm, out_png, point_size=point_size)
        rows.append(
            dict(
                sequence=h5_path.stem,
                percentile=percentile,
                frame_key=key,
                ground_fraction=frac,
                num_points=int(len(gm)),
                num_ground=int(gm.sum()),
            )
        )
    save_histogram(fracs_v, picks, out_dir / "histogram.png", h5_path.stem)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, default=Path("/mnt/data/lidar/h5/innoviz"))
    ap.add_argument(
        "--out_root",
        type=Path,
        default=Path("/home/ubuntu/orr/dev/forks/OpenSceneFlow/assets/ground_coverage_viz"),
    )
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = process all sequences")
    ap.add_argument("--match", type=str, default="", help="substring filter for sequence stem")
    ap.add_argument("--percentiles", type=int, nargs="+", default=[5, 50, 95])
    ap.add_argument("--min_points", type=int, default=50_000,
                    help="drop frames with fewer points (sensor warmup) before computing percentiles")
    ap.add_argument("--min_ground_frac", type=float, default=0.005,
                    help="drop frames with ground fraction below this before computing percentiles")
    args = ap.parse_args()

    renderer = rendering.OffscreenRenderer(args.width, args.height)

    summary_rows: list[dict] = []
    for split in args.splits:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            print(f"[skip] {split_dir} (not a directory)", file=sys.stderr)
            continue
        h5_files = sorted(p for p in split_dir.glob("*.h5") if p.is_file())
        if args.match:
            h5_files = [p for p in h5_files if args.match in p.stem]
        if args.limit > 0:
            h5_files = h5_files[: args.limit]
        for h5_path in h5_files:
            out_dir = args.out_root / split / h5_path.stem
            print(f"[{split}] {h5_path.stem}", flush=True)
            try:
                rows = process_sequence(
                    h5_path,
                    out_dir,
                    renderer,
                    point_size=args.point_size,
                    percentiles=tuple(args.percentiles),
                    min_points=args.min_points,
                    min_ground_frac=args.min_ground_frac,
                )
            except Exception as e:  # noqa: BLE001 — surface the offender, keep going
                print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            for r in rows:
                r["split"] = split
            summary_rows.extend(rows)

    if summary_rows:
        csv_path = args.out_root / "summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "sequence",
                    "percentile",
                    "frame_key",
                    "ground_fraction",
                    "num_points",
                    "num_ground",
                ],
            )
            w.writeheader()
            w.writerows(summary_rows)
        print(f"summary -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
