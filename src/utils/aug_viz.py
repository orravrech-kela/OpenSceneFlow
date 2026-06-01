"""Open3D offscreen renderer for before/after training-augmentation visualization.

Used by tools/viz_train_aug.py (PNG dumps to assets/) and by the
TrainAugVizCallback (wandb logging during training). Mirrors the structure of
the OpenPCDet vedo_render but uses Open3D's headless-EGL OffscreenRenderer,
which is the proven path in this repo (see dataprocess/viz_ground_coverage.py)
and avoids the vedo/VTK dependency that isn't installed here.

Scene-flow data, not boxes: a sample is pc0/gm0/flow(+category). The Innoviz
sensor is static (pose_flow ~ 0), so the stored `flow` is already the non-rigid
object motion. Per-frame motion is small, so flow segments are drawn scaled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# flow_category_indices taxonomy (src/utils/innoviz_eval.py): 0 background.
CLASS_COLORS = {
    0: (0.62, 0.62, 0.66),   # background / non-ground
    1: (0.20, 0.55, 1.00),   # vehicle
    2: (1.00, 0.25, 0.85),   # person
    3: (1.00, 0.55, 0.05),   # drone
    4: (0.25, 0.92, 0.45),   # animal
}
CLASS_NAMES = {0: "background", 1: "vehicle", 2: "person", 3: "drone", 4: "animal"}
GROUND_RGB = (0.22, 0.24, 0.30)
PC1_RGB = (0.35, 0.45, 0.75)
FLOW_RGB = (1.0, 0.95, 0.3)   # GT flow segments: one distinct color, not per-class
BG_RGBA = [0.04, 0.05, 0.07, 1.0]


@dataclass
class View:
    """A named camera (lidar frame: +X forward, +Y left, +Z up)."""
    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    fov: float = 60.0


VIEWS = {
    "bev": View("BEV (top-down)", eye=(12.0, 0.0, 72.0), target=(12.0, 0.0, 0.0),
                up=(1.0, 0.0, 0.0), fov=50.0),
    "oblique": View("3D oblique", eye=(-24.0, -30.0, 22.0), target=(15.0, 0.0, -1.0),
                    up=(0.0, 0.0, 1.0), fov=60.0),
    "sensor": View("sensor", eye=(0.0, 0.0, 2.5), target=(28.0, 0.0, 0.0),
                   up=(0.0, 0.0, 1.0), fov=70.0),
}

DEFAULT_CROP = (-40.0, -40.0, -4.0, 40.0, 40.0, 6.0)


def fg_zoom_view(points, category, ground_mask=None, dist: float = 20.0) -> View | None:
    """A 3D view centered on the moving-foreground centroid (None if no foreground)."""
    if category is None:
        return None
    cat = np.asarray(category)
    fg = cat > 0
    if ground_mask is not None:
        fg &= ~np.asarray(ground_mask).astype(bool)
    if int(fg.sum()) < 5:
        return None
    c = np.asarray(points)[fg, :3].mean(0)
    return View("zoom-3D (foreground)", eye=(c[0] - dist, c[1] - dist, dist * 0.7),
                target=(float(c[0]), float(c[1]), float(c[2])), up=(0.0, 0.0, 1.0), fov=55.0)


def decode_aug(pre_xyz: np.ndarray, post_xyz: np.ndarray) -> dict:
    """Recover the realized flip/height/jitter from a pre/post point pair.

    Augmentation is per-point with no reordering, so post[i] corresponds to
    pre[i]: flip negates x and/or y, height adds a constant to z, jitter adds
    bounded gaussian noise. We read each back from the aligned coordinates.
    """
    pre_xyz, post_xyz = np.asarray(pre_xyz), np.asarray(post_xyz)
    if pre_xyz.shape != post_xyz.shape or len(pre_xyz) == 0:
        return {"flip_x": False, "flip_y": False, "dz": 0.0, "jitter": 0.0}
    dz = float(np.median(post_xyz[:, 2] - pre_xyz[:, 2]))

    def _flipped(axis: int) -> bool:
        a_pre, a_post = pre_xyz[:, axis], post_xyz[:, axis]
        m = np.abs(a_pre) > 1.0
        if int(m.sum()) < 10:
            return False
        return float(np.median(a_post[m] / a_pre[m])) < 0.0

    fx, fy = _flipped(0), _flipped(1)
    rec = post_xyz.copy()
    if fx:
        rec[:, 0] *= -1
    if fy:
        rec[:, 1] *= -1
    rec[:, 2] -= dz
    return {"flip_x": fx, "flip_y": fy, "dz": dz, "jitter": float(np.std(rec - pre_xyz))}


def aug_label(d: dict) -> str:
    flips = "".join(a for a, on in (("x", d["flip_x"]), ("y", d["flip_y"])) if on) or "none"
    return f"flip={flips}  dz={d['dz']:+.2f}m  jitter_sigma={d['jitter']:.3f}"


def _crop_mask(xyz: np.ndarray, crop) -> np.ndarray:
    x0, y0, z0, x1, y1, z1 = crop
    return ((xyz[:, 0] >= x0) & (xyz[:, 0] < x1) &
            (xyz[:, 1] >= y0) & (xyz[:, 1] < y1) &
            (xyz[:, 2] >= z0) & (xyz[:, 2] < z1))


def _arrow_segments(s: np.ndarray, e: np.ndarray, head_frac: float = 0.22,
                    head_min: float = 0.5, deg: float = 26.0):
    """Shaft + two arrowhead segments for arrow s->e (head splayed in the xy plane)."""
    s, e = np.asarray(s, float), np.asarray(e, float)
    d = e - s
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return [(s, e)]
    h = max(head_min, head_frac * L)
    back = -d / L
    a = np.deg2rad(deg)
    ca, sa = np.cos(a), np.sin(a)
    left = np.array([back[0] * ca - back[1] * sa, back[0] * sa + back[1] * ca, back[2]])
    right = np.array([back[0] * ca + back[1] * sa, -back[0] * sa + back[1] * ca, back[2]])
    return [(s, e), (e, e + h * left), (e, e + h * right)]


def _point_colors(category: np.ndarray | None, ground: np.ndarray | None, n: int) -> np.ndarray:
    colors = np.tile(CLASS_COLORS[0], (n, 1)).astype(np.float64)
    if category is not None:
        for cid, rgb in CLASS_COLORS.items():
            colors[category == cid] = rgb
    if ground is not None:
        colors[ground.astype(bool)] = GROUND_RGB
    return colors


class AugRenderer:
    """Headless Open3D renderer producing RGB numpy frames for one sample."""

    def __init__(self, width: int = 720, height: int = 480, point_size: float = 2.0,
                 flow_scale: float = 10.0, flow_min_mag: float = 0.02,
                 flow_color=None, crop=DEFAULT_CROP):
        # flow_color: None -> color each flow segment by its point's class;
        # an (r,g,b) tuple -> draw all flow segments in that single color.
        from open3d.visualization import rendering
        self._rendering = rendering
        self.width, self.height = width, height
        self.point_size = point_size
        self.flow_scale = flow_scale
        self.flow_min_mag = flow_min_mag
        self.flow_color = flow_color
        self.crop = crop
        self._r = rendering.OffscreenRenderer(width, height)

    def render(self, points, ground_mask=None, category=None, flow=None,
               pc1=None, instance=None, view="oblique") -> np.ndarray:
        import open3d as o3d
        rendering = self._rendering
        v = view if isinstance(view, View) else VIEWS[view]
        crop = self.crop
        if v.name.startswith("zoom"):
            half = 14.0
            crop = (v.target[0] - half, v.target[1] - half, -4.0,
                    v.target[0] + half, v.target[1] + half, 8.0)
        xyz = np.asarray(points)[:, :3].astype(np.float64)
        keep = _crop_mask(xyz, crop)
        xyz = xyz[keep]
        cat = None if category is None else np.asarray(category)[keep]
        gm = None if ground_mask is None else np.asarray(ground_mask)[keep]
        iid = None if instance is None else np.asarray(instance)[keep]

        scene = self._r.scene
        scene.clear_geometry()
        scene.set_background(BG_RGBA)

        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = self.point_size

        if pc1 is not None:
            p1 = np.asarray(pc1)[:, :3].astype(np.float64)
            p1 = p1[_crop_mask(p1, crop)]
            g1 = o3d.geometry.PointCloud()
            g1.points = o3d.utility.Vector3dVector(p1)
            g1.colors = o3d.utility.Vector3dVector(np.tile(PC1_RGB, (len(p1), 1)))
            m1 = rendering.MaterialRecord(); m1.shader = "defaultUnlit"; m1.point_size = max(1.0, self.point_size - 1)
            scene.add_geometry("pc1", g1, m1)

        # Split background (small) from foreground classes (large, saturated) so
        # the few moving objects pop against the dense gray background.
        fg = np.zeros(len(xyz), dtype=bool)
        if cat is not None:
            fg = cat > 0
        bg = ~fg
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(xyz[bg])
        bg_pcd.colors = o3d.utility.Vector3dVector(
            _point_colors(None if cat is None else cat[bg], None if gm is None else gm[bg], int(bg.sum())))
        scene.add_geometry("pc0_bg", bg_pcd, mat)
        if int(fg.sum()) > 0:
            fg_pcd = o3d.geometry.PointCloud()
            fg_pcd.points = o3d.utility.Vector3dVector(xyz[fg])
            fg_pcd.colors = o3d.utility.Vector3dVector(_point_colors(cat[fg], None, int(fg.sum())))
            mfg = rendering.MaterialRecord(); mfg.shader = "defaultUnlit"; mfg.point_size = self.point_size * 3.0
            scene.add_geometry("pc0_fg", fg_pcd, mfg)

        if flow is not None:
            fl = np.asarray(flow)[keep, :3].astype(np.float64)
            # One arrow per object instance (centroid -> centroid + mean flow), so a
            # few bold class-colored vectors stand out instead of thousands of tiny
            # per-point segments that pile onto the same-colored object points.
            arrows = []  # (start, end, class_id)
            if iid is not None and cat is not None:
                fgm = cat > 0
                if gm is not None:
                    fgm &= ~gm.astype(bool)
                for u in np.unique(iid[fgm]):
                    m = fgm & (iid == u)
                    mf = fl[m].mean(0)
                    if np.linalg.norm(mf) >= self.flow_min_mag:
                        c0 = xyz[m].mean(0)
                        arrows.append((c0, c0 + mf * self.flow_scale, int(cat[m][0])))
            else:  # no instance ids: fall back to per-point segments
                sel = np.linalg.norm(fl, axis=1) >= self.flow_min_mag
                if gm is not None:
                    sel &= ~gm.astype(bool)
                cc = cat[sel] if cat is not None else np.zeros(int(sel.sum()), dtype=int)
                for st, fv, c in zip(xyz[sel], fl[sel], cc):
                    arrows.append((st, st + fv * self.flow_scale, int(c)))
            if arrows:
                pts, lines, cols = [], [], []
                for s, e, cid in arrows:
                    col = self.flow_color if self.flow_color is not None else CLASS_COLORS.get(cid, CLASS_COLORS[0])
                    for a, b in _arrow_segments(s, e):
                        i0 = len(pts)
                        pts.extend([a, b]); lines.append([i0, i0 + 1]); cols.append(col)
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
                ls.lines = o3d.utility.Vector2iVector(np.asarray(lines))
                ls.colors = o3d.utility.Vector3dVector(np.asarray(cols, dtype=np.float64))
                ml = rendering.MaterialRecord(); ml.shader = "unlitLine"; ml.line_width = 4.0
                scene.add_geometry("flow", ls, ml)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4.0, origin=[0, 0, 0])
        ma = rendering.MaterialRecord(); ma.shader = "defaultUnlit"
        scene.add_geometry("ego", axes, ma)

        self._r.setup_camera(v.fov, list(v.target), list(v.eye), list(v.up))
        return np.asarray(self._r.render_to_image())


# ---------- PIL composition ----------

def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(arr: np.ndarray, title: str, sub: str = "") -> Image.Image:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (img.width, 26)], fill=(0, 0, 0))
    draw.text((8, 4), title, fill=(255, 255, 255), font=_font(16))
    if sub:
        draw.rectangle([(0, img.height - 22), (img.width, img.height)], fill=(0, 0, 0))
        draw.text((8, img.height - 19), sub, fill=(210, 210, 210), font=_font(13))
    return img


CLASS_LEGEND = [
    ("ground", GROUND_RGB),
    ("background", CLASS_COLORS[0]),
    ("vehicle", CLASS_COLORS[1]),
    ("person", CLASS_COLORS[2]),
    ("drone", CLASS_COLORS[3]),
    ("animal", CLASS_COLORS[4]),
]


def _text_w(font, text: str) -> float:
    try:
        return font.getlength(text)
    except Exception:
        b = font.getbbox(text)
        return b[2] - b[0]


def _legend_bar(width: int, height: int = 30, bg=(10, 10, 14), flow_color=None) -> Image.Image:
    """Centered row: class color swatches + a flow arrow glyph.

    The flow glyph is an arrow (not a swatch) since flow is per-class by default;
    its label says so, or names the uniform color when flow_color is set.
    """
    # (label, color, kind): swatch for classes, arrow for flow.
    items = [(name, rgb, "swatch") for name, rgb in CLASS_LEGEND]
    if flow_color is None:
        items.append(("GT flow per object (by class)", (0.9, 0.9, 0.9), "arrow"))
    else:
        items.append(("GT flow per object", flow_color, "arrow"))

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = _font(14)
    sw, gap, spacing = 15, 6, 20
    widths = [_text_w(font, name) for name, _, _ in items]
    total = sum(sw + gap + w for w in widths) + spacing * (len(items) - 1)
    x = max(8.0, (width - total) / 2.0)
    y = (height - sw) // 2
    for (name, rgb, kind), tw in zip(items, widths):
        col = tuple(int(round(c * 255)) for c in rgb)
        if kind == "arrow":
            cy = y + sw // 2
            draw.line([(x, cy), (x + sw, cy)], fill=col, width=2)
            draw.line([(x + sw - 5, cy - 3), (x + sw, cy)], fill=col, width=2)
            draw.line([(x + sw - 5, cy + 3), (x + sw, cy)], fill=col, width=2)
        else:
            draw.rectangle([x, y, x + sw, y + sw], fill=col, outline=(60, 60, 66))
        draw.text((x + sw + gap, y + 1), name, fill=(225, 225, 225), font=font)
        x += sw + gap + tw + spacing
    return img


def _grid(cells: list[list[Image.Image]], pad: int = 4, bg=(10, 10, 14)) -> Image.Image:
    rows = len(cells)
    cols = max(len(r) for r in cells)
    cw = max(c.width for r in cells for c in r)
    ch = max(c.height for r in cells for c in r)
    out = Image.new("RGB", (cols * cw + (cols + 1) * pad, rows * ch + (rows + 1) * pad), bg)
    for i, row in enumerate(cells):
        for j, c in enumerate(row):
            out.paste(c, (pad + j * (cw + pad), pad + i * (ch + pad)))
    return out


def render_columns(renderer: AugRenderer, columns: list[dict],
                   views=("bev", "oblique"), caption: str = "") -> Image.Image:
    """Render labeled columns side by side; each column renders the given views as rows.

    A column dict: {title, sub, points, ground_mask, category, flow, pc1}.
    """
    cells: list[list[Image.Image]] = [[] for _ in views]
    for col in columns:
        for r, view in enumerate(views):
            resolved = view
            if view == "zoom":
                resolved = fg_zoom_view(col["points"], col.get("category"),
                                        col.get("ground_mask")) or VIEWS["oblique"]
            arr = renderer.render(col["points"], col.get("ground_mask"), col.get("category"),
                                  col.get("flow"), col.get("pc1"), col.get("instance"),
                                  view=resolved)
            vname = resolved.name if isinstance(resolved, View) else VIEWS[resolved].name
            title = col["title"] if r == 0 else vname
            sub = col.get("sub", "") if r == 0 else ""
            cells[r].append(_label(arr, title, sub))
    grid = _grid(cells)
    top = 30 if caption else 0
    legend_h = 30
    out = Image.new("RGB", (grid.width, grid.height + top + legend_h), (10, 10, 14))
    if caption:
        ImageDraw.Draw(out).text((8, 7), caption, fill=(255, 255, 255), font=_font(15))
    out.paste(grid, (0, top))
    out.paste(_legend_bar(grid.width, legend_h, flow_color=renderer.flow_color),
              (0, top + grid.height))
    return out


def split_frame(scene_id: str, timestamp) -> str:
    return f"seq={scene_id}  ts={timestamp}"


def pick_foreground_indices(ds, n: int, seed: int = 0, min_fg: int = 150) -> list[int]:
    """Spread n dataset indices, preferring frames with the most moving foreground."""
    rng = np.random.default_rng(seed)
    cand = rng.integers(0, len(ds), size=min(len(ds), max(n * 8, 64)))
    scored = []
    for i in cand:
        d = ds[int(i)]
        cat, fl = d.get("flow_category_indices"), d.get("flow")
        fg = 0 if cat is None else int((np.asarray(cat) > 0).sum())
        moving = 0.0
        if cat is not None and fl is not None and fg > 0:
            m = np.asarray(cat) > 0
            moving = float(np.linalg.norm(np.asarray(fl)[m], axis=1).mean())
        scored.append((moving if fg >= min_fg else -1.0, int(i)))
    scored.sort(reverse=True)
    picked = [i for _, i in scored[:n]]
    return picked or [int(x) for x in cand[:n]]


def sample_to_column(title: str, sub: str, d: dict, show_pc1: bool = False) -> dict:
    return {
        "title": title, "sub": sub,
        "points": d["pc0"], "ground_mask": d.get("gm0"),
        "category": d.get("flow_category_indices"), "flow": d.get("flow"),
        "instance": d.get("flow_instance_id"),
        "pc1": d.get("pc1") if show_pc1 else None,
    }


try:
    from pytorch_lightning import Callback as _Callback
except Exception:  # pl absent (e.g. dataprocess hosts) — callback just won't be used there
    _Callback = object


class TrainAugVizCallback(_Callback):
    """Lightning callback: log PRE/POST-aug panels for fixed samples to wandb.

    Seamless wiring — append to the Trainer's callbacks list (see train.py).
    Renders rank-0 only; reuses the training dataset by toggling its transform
    off to capture the raw (PRE) frame, then re-applies it for POST.
    """

    def __init__(self, num_samples: int = 4, seed: int = 0, every_n_epochs: int = 1,
                 min_fg: int = 150, views=("bev", "zoom"), flow_scale: float = 25.0,
                 flow_min_mag: float = 0.03, width: int = 720, height: int = 460,
                 point_size: float = 2.0, dataset=None):
        self.num_samples = num_samples
        self.seed = seed
        self.every_n_epochs = max(1, every_n_epochs)
        self.min_fg = min_fg
        self.views = tuple(views)
        self.dataset = dataset  # explicit train dataset; falls back to trainer's if None
        self._rkw = dict(flow_scale=flow_scale, flow_min_mag=flow_min_mag,
                         width=width, height=height, point_size=point_size)
        self._renderer = None
        self._indices = None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_rank != 0 or (trainer.current_epoch % self.every_n_epochs) != 0:
            return
        run = self._wandb_run(trainer)
        if run is None:
            return
        ds = self.dataset
        if ds is None:
            loader = getattr(trainer, "train_dataloader", None)
            ds = getattr(loader, "dataset", None)
        if ds is None or not hasattr(ds, "transform"):
            return
        import copy
        if self._renderer is None:
            self._renderer = AugRenderer(**self._rkw)
        if self._indices is None:
            saved = ds.transform; ds.transform = None
            self._indices = pick_foreground_indices(ds, self.num_samples, self.seed, self.min_fg)
            ds.transform = saved

        import wandb
        aug = ds.transform
        # All samples under one key as a list -> a single wandb panel with an index
        # slider, instead of one panel per sample.
        images = []
        for n, idx in enumerate(self._indices):
            saved = ds.transform; ds.transform = None
            raw = ds[idx]; ds.transform = saved
            pre = copy.deepcopy(raw)
            post = aug(copy.deepcopy(raw)) if aug is not None else raw
            info = decode_aug(pre["pc0"][:, :3], post["pc0"][:, :3])
            cap = (f"#{n} epoch {trainer.current_epoch} | {split_frame(raw['scene_id'], raw['timestamp'])}"
                   f" | POST: {aug_label(info)}")
            img = render_columns(self._renderer, [
                sample_to_column("PRE-AUG", "", pre),
                sample_to_column("POST-AUG", aug_label(info), post),
            ], views=self.views, caption=cap)
            images.append(wandb.Image(np.asarray(img), caption=cap))
        run.log({"aug/samples": images}, step=trainer.global_step)

    @staticmethod
    def _wandb_run(trainer):
        logger = getattr(trainer, "logger", None)
        exp = getattr(logger, "experiment", None)
        return exp if (exp is not None and hasattr(exp, "log")
                       and type(logger).__name__ == "WandbLogger") else None
