"""Visualize training data before/after augmentation and save PNGs to assets/.

Two outputs (Open3D headless render, BEV + 3D-oblique rows, flow drawn as
class-colored segments on moving foreground points):

  showcase.png      one sample, each augmentation isolated:
                    ORIGINAL | FLIP-Y | FLIP-X | HEIGHT +dz | JITTER
  sample_NN.png     N real samples: PRE-AUG vs POST-AUG using the exact
                    transforms.Compose the trainer uses (train.py:62).

Run from repo root with the project venv:
  .venv/bin/python tools/viz_train_aug.py --split train --num_samples 6

Add --wandb to also log the panels to a wandb run (project from conf/config.yaml).
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from torchvision import transforms

from src.dataset import HDF5Dataset, RandomHeight, RandomFlip, RandomJitter
from src.utils.aug_viz import (
    AugRenderer, aug_label, decode_aug, render_columns, split_frame,
    pick_foreground_indices, sample_to_column as column_from_dict, FLOW_RGB,
)


def _fg_count(d: dict) -> int:
    cat = d.get("flow_category_indices")
    return 0 if cat is None else int((np.asarray(cat) > 0).sum())


def make_showcase(renderer: AugRenderer, d: dict, views, show_pc1: bool, dz: float = 1.5,
                  jitter_sigma: float = 0.05) -> "Image.Image":
    """One sample, each augmentation applied in isolation and explicitly labeled."""
    orig = d
    flip_y = copy.deepcopy(d); flip_y["pc0"] = flip_y["pc0"].copy(); flip_y["pc0"][:, 1] *= -1
    if flip_y.get("flow") is not None:
        flip_y["flow"] = flip_y["flow"].copy(); flip_y["flow"][:, 1] *= -1
    flip_x = copy.deepcopy(d); flip_x["pc0"] = flip_x["pc0"].copy(); flip_x["pc0"][:, 0] *= -1
    if flip_x.get("flow") is not None:
        flip_x["flow"] = flip_x["flow"].copy(); flip_x["flow"][:, 0] *= -1
    height = copy.deepcopy(d); height["pc0"] = height["pc0"].copy(); height["pc0"][:, 2] += dz
    jit = copy.deepcopy(d); jit["pc0"] = jit["pc0"] + np.clip(
        jitter_sigma * np.random.randn(*jit["pc0"].shape), -0.1, 0.1)

    cols = [
        column_from_dict("ORIGINAL", "", orig, show_pc1),
        column_from_dict("FLIP-Y (mirror left/right)", "", flip_y, show_pc1),
        column_from_dict("FLIP-X (mirror front/back)", "", flip_x, show_pc1),
        column_from_dict(f"HEIGHT dz=+{dz:.1f}m", "", height, show_pc1),
        column_from_dict(f"JITTER sigma={jitter_sigma:.2f}", "", jit, show_pc1),
    ]
    cap = (f"AUGMENTATION SHOWCASE  |  {split_frame(d['scene_id'], d['timestamp'])}  |  "
           f"flow x{renderer.flow_scale:g}")
    return render_columns(renderer, cols, views=views, caption=cap)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, default=Path("/mnt/data/lidar/h5/innoviz"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_dir", type=Path,
                    default=Path(BASE_DIR) / "assets" / "aug_viz")
    ap.add_argument("--num_samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=2)
    ap.add_argument("--min_fg", type=int, default=150, help="min foreground points for sample picking")
    # match train.py:62 defaults; override --flip_prob 1.0 to force a flip every sample.
    ap.add_argument("--height_prob", type=float, default=0.8)
    ap.add_argument("--flip_prob", type=float, default=0.5)
    ap.add_argument("--jitter_sigma", type=float, default=0.01)
    ap.add_argument("--flow_scale", type=float, default=10.0)
    ap.add_argument("--flow_min_mag", type=float, default=0.02)
    ap.add_argument("--flow_color", choices=["class", "yellow"], default="class",
                    help="color flow arrows by object class (default) or a single yellow")
    ap.add_argument("--point_size", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--views", nargs="+", default=["bev", "zoom"],
                    help="rows per column: bev, oblique, sensor, zoom (auto-center on foreground)")
    ap.add_argument("--show_pc1", action="store_true", help="overlay next-frame points (faint blue)")
    ap.add_argument("--no_showcase", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="also log panels to a wandb run")
    args = ap.parse_args()

    split_dir = args.data_root / args.split
    ds = HDF5Dataset(str(split_dir), n_frames=args.num_frames, transform=None)
    print(f"Loaded {len(ds)} samples from {split_dir}")

    flow_color = None if args.flow_color == "class" else FLOW_RGB
    renderer = AugRenderer(width=args.width, height=args.height, point_size=args.point_size,
                           flow_scale=args.flow_scale, flow_min_mag=args.flow_min_mag,
                           flow_color=flow_color)

    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    views = tuple(args.views)

    wandb_run = None
    if args.wandb:
        import wandb
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(Path(BASE_DIR) / "conf" / "config.yaml")
        wandb_run = wandb.init(entity=cfg.get("wandb_entity"), project=cfg.get("wandb_project_name"),
                               job_type="aug_viz", name=f"aug_viz_{args.split}")

    indices = pick_foreground_indices(ds, args.num_samples, args.seed, args.min_fg)
    print(f"Picked indices: {indices}")

    if not args.no_showcase:
        np.random.seed(args.seed)
        show = make_showcase(renderer, ds[indices[0]], ("bev", "oblique"), args.show_pc1,
                             jitter_sigma=max(args.jitter_sigma, 0.05))
        p = out_dir / "showcase.png"
        show.save(p)
        print(f"  wrote {p}  ({show.width}x{show.height})")
        if wandb_run is not None:
            import wandb
            wandb_run.log({"aug/showcase": wandb.Image(str(p))})

    wandb_images = []
    for n, idx in enumerate(indices):
        np.random.seed(args.seed + idx)
        raw = ds[idx]
        pre = copy.deepcopy(raw)
        post = copy.deepcopy(raw)
        aug = transforms.Compose([
            RandomHeight(p=args.height_prob),
            RandomFlip(p=args.flip_prob),
            RandomJitter(sigma=args.jitter_sigma),
        ])
        post = aug(post)
        info = decode_aug(pre["pc0"][:, :3], post["pc0"][:, :3])
        fg = _fg_count(raw)
        cap = (f"{split_frame(raw['scene_id'], raw['timestamp'])}  |  fg_pts={fg}  |  "
               f"POST-AUG: {aug_label(info)}  |  flow x{args.flow_scale:g}")
        cols = [
            column_from_dict("PRE-AUG", "", pre, args.show_pc1),
            column_from_dict("POST-AUG", aug_label(info), post, args.show_pc1),
        ]
        img = render_columns(renderer, cols, views=views, caption=cap)
        seq = str(raw["scene_id"])[:28]
        p = out_dir / f"sample_{n:02d}_{seq}_{raw['timestamp']}.png"
        img.save(p)
        print(f"  wrote {p.name}  ({img.width}x{img.height})  {aug_label(info)}")
        if wandb_run is not None:
            import wandb
            wandb_images.append(wandb.Image(str(p), caption=f"#{n} {cap}"))

    if wandb_run is not None:
        if wandb_images:
            wandb_run.log({"aug/samples": wandb_images})  # one panel, index slider
        wandb_run.finish()
    print(f"Done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
