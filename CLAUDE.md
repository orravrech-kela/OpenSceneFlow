# Project notes for Claude Code

## Environment

Always run Python from the project venv at `.venv/`. The venv lives in the
top-level checkout; worktrees under `.claude/worktrees/...` share it (no need
to create a per-worktree venv).

```bash
/home/ubuntu/orr/dev/forks/OpenSceneFlow/.venv/bin/python <script>
```

For provisioning from scratch (and the rationale behind each step), see
[SETUP.md](SETUP.md). In one line: `uv pip install -r requirements-train.txt`
gets you 90% of the way; the remaining 10% is upgrading torch to cu121,
restoring modern pydantic/fastapi, and building the two in-repo CUDA extensions
(`assets/cuda/chamfer3D`, `assets/cuda/mmcv`).

Do **not** use `import lightning.pytorch as pl` — use `import pytorch_lightning
as pl`. The `lightning` metapackage's `app` subpackage drags in a broken old
`fastapi`/`pydantic` chain. We've already migrated `train.py`, `eval.py`, and
`src/trainer.py`; please keep new code on `pytorch_lightning`.

## Innoviz pipeline

End-to-end finetune from the Waymo DeltaFlow ckpt is documented in
[dataprocess/README.md#Innoviz](dataprocess/README.md). High-level:

1. Author/update `conf/innoviz_splits.yaml` (gitignored; per-operator). Entries
   are `<recording_day>/<sequence_path>` relative to `/mnt/data/lidar/processed/`.
   Arbitrary nesting depth is fine (`25_3_maya_amit/holon_dune_2/Lidar_holon_dune2/Recording_…`
   works just like a one-level entry).
2. `python dataprocess/extract_innoviz.py --splits_file conf/innoviz_splits.yaml
   --data_root /mnt/data/lidar/processed --output_dir /mnt/data/lidar/h5/innoviz`
   produces one `<scene_id>.h5` per sequence in `train/` and `val/` plus
   `index_total.pkl` and `index_flow.pkl` per split.
3. `python train.py model=deltaflow pretrained_weights=model_zoo/deltaflow-waymo.ckpt
   train_data=/mnt/data/lidar/h5/innoviz/train val_data=/mnt/data/lidar/h5/innoviz/val
   "voxel_size=[0.15, 0.15, 0.15]" "point_cloud_range=[-38.4, -38.4, -3, 38.4, 38.4, 3]"
   loss_fn=deltaflowLoss …`

**Why those voxel/point_cloud_range overrides?** The base `conf/config.yaml`
defaults to `voxel_size=[0.2, 0.2, 6]` (intended for SSF). With DeltaFlow that
collapses the Z dim to 1, which then becomes 0 after the first stride-2 conv
in MinkUNet. The Waymo ckpt's `hyper_parameters.cfg` records the exact training
config — always cross-check the ckpt before changing these values.

**Why `deltaflowLoss` and not `deflowLoss`?** The Waymo ckpt was trained with
`deltaflowLoss`. Using a different loss for finetune is allowed but you'd lose
the loss-scale matching that the ckpt's optimizer state encodes.

## Extractor: annotation file resolution

`dataprocess/extract_innoviz.py::_annotations_path` prefers
`annotations/ground_truth.json` and falls back to `annotations/manual_gt.json`.
Both share the same CVAT cuboid_3d schema. Always cross-check on disk before
assuming one or the other is present.

`_enumerate_frames` requires annotation `id`s to parse to integers that match
polar filename stems (polar-file numbering, e.g. `id="00007321"` →
`polar/00007321.npz`). If you have a 0-based serial export from CVAT, normalize
it first with
`algorithms/projects/lidar/scripts/data_prep/convert_annotations_to_global.py`
— the extractor will fail fast with a pointer to that tool if any id doesn't
resolve. A previous version of the extractor had a silent positional fallback;
it was removed because it could produce structurally corrupt H5s when the
polar id range partially overlapped the annotation id range (duplicate frames
for one polar range, missing frames for another).

## Models / dependencies

Several model imports are gated behind try/except in `src/models/__init__.py`
because their CUDA extensions or pip deps are optional:
- `DeFlow` / `FastFlow3D`: need `assets/cuda/mmcv` (Voxelization, DynamicScatter)
- `DeltaFlow` / `Flow4D`: need spconv-cu121 + mmcv
- `VoteFlow`: needs pytorch3d
- `SSF`: needs torch-scatter + mmengine-lite
- `FastNSF`: needs FastGeodis
- `ICPFlow`: needs pytorch3d

Don't drop the try/except wrappers — they let the rest of the codebase load
on hosts that haven't built every extension.

Similarly, `src/lossfuncs/selfsupervise.py` guards `chamfer3D` import; calling
the SSL losses (`teflowLoss`, `seflowLoss`, `seflowppLoss`) without that
extension raises at runtime, but `deflowLoss` / `deltaflowLoss` work without it.

## Data root and sizes (current)

- Raw innoviz polar + annotations: `/mnt/data/lidar/processed/`
- Extracted HDF5: `/mnt/data/lidar/h5/innoviz/{train,val}/<scene_id>.h5`
- Per-sequence H5s are several GB each (~14 GB max). Plan disk accordingly
  before running the full extractor.
- Per-frame data: ~480×1200 = 576k polar points, ~50–100k cartesian after
  range cropping and ground masking. Dynamic objects are <1% of points.

## Tracking

We use Weights & Biases for experiment tracking. Defaults are wired up in
`conf/config.yaml`:
- `wandb_entity: orr-a-kelasys`
- `wandb_project_name: opensceneflow`
- `wandb_mode: online`

Override with `wandb_mode=disabled` for smoke / debug runs.
