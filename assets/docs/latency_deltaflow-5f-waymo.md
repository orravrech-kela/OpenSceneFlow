# DeltaFlow-5f runtime latency report — deployability @ 10 FPS

**Checkpoint:** `logs/jobs/deltaflow-5f-waymo/05-27-22-16/checkpoints/last.ckpt`
(DeltaFlow, 19.94M params, `num_frames=5`, `voxel_size=[0.32]³`,
`point_cloud_range=[0,-160,-25.6, 204.8,160,25.6]` → grid 640×1000×160)
**GPU:** NVIDIA L4 (24 GB) · torch 2.1.2+cu121 · spconv `ConvAlgo.Native`
**Data:** 120 frames sampled evenly across all 8 pbench sequences
(`/home/ubuntu/orr/data/innoviz_h5/pbench/val`), ground-stripped exactly like
the val/test path (`run_model_wo_ground_data`). Sensor is natively 10 Hz, so
the 100 ms budget == keeping up with the sensor.
**Tool:** `tools/benchmark_latency.py` (CUDA-sync'd per-stage timers; the
instrumented forward is verified equal to plain `model(batch)` to 4e-8).

## TL;DR

| Config | mean | p50 | p90 | p99 | FPS (mean) | 10 FPS? |
|---|---|---|---|---|---|---|
| **Stock model** | 359 ms | 357 | 368 | 406 | 2.8 | **NO — 3.6× over budget** |
| + sparse-gather decoder (exact, no retrain) | 86 ms | 84 | 95 | 136 | 11.6 | borderline (11/120 frames >100 ms) |
| + fp16 backbone | 84 ms | 81 | 97 | 104 | 12.0 | borderline-pass (4/120 frames >100 ms) |
| + history-voxelization caching (estimate) | ~50–65 ms | | | | ~16–20 | **YES, with margin** |

The model is **not deployable as-is**, but the blocker is a single
inference-only inefficiency, not the network itself: **76 % of the frame time
is the decoder materializing a 6.5 GB dense voxel grid (`.dense()`) just to
gather ~10⁵ point features out of it.** Replacing that gather with a sparse
lookup is numerically identical (max flow diff 3.7e-8) and takes the model
from 2.8 FPS to ~11.6 FPS, and peak GPU memory from **13.6 GB → 1.7 GB**.

## Where the time goes (stock, mean over 120 frames)

| Block | mean ms | p90 ms | % of e2e |
|---|---|---|---|
| preprocess: pose-warp 4 frames → pc1 frame | 1.4 | 2.0 | 0.4 % |
| voxelize (5 frames) | 48.4 | 58.3 | 13.5 % |
| — `DynamicVoxelizer` ×5 | 6.3 | | |
| — `DynamicPillarFeatureNet` ×5 | 40.0 | | |
| — sparse delta accumulate + coalesce | 0.8 | | |
| backbone: MinkUNet (spconv) | 37.2 | 43.7 | 10.4 % |
| — down (conv_input + stage1–4) | 21.3 | | |
| — up (up1–4) | 15.7 | | |
| **decoder: `Point_head`** | **274.1** | **275.7** | **76.3 %** |
| — **`sparse_tensor.dense()`** | **272.3** | 272.3 | **75.8 %** |
| — voxel→point gather + MLP | 1.7 | 3.3 | 0.5 % |
| **total e2e** | **359.2** | 368.2 | 100 % |

Data-side costs (outside the model, replaced by the live driver in deployment):
H5 fetch ≈ 76 ms p50 (CPU, can pipeline), H2D copy 2.4 ms, ground-strip 0.5 ms.
Note ground segmentation itself (LineFit) runs upstream of this model and is
**not** included in any number here — it needs its own budget line.

### Root cause: `Point_head.forward` calls `sparse_tensor.dense()`

The decoder only needs backbone features at pc0's occupied voxels
(7k–177k voxels on these scenes), but `.dense()` allocates + zero-fills +
scatters a `1×16×640×1000×160` fp32 tensor (6.55 GB) every frame. The op-level
profiler confirms it: **79 % of all CUDA time is `aten::copy_` and 15.6 % is
`aten::fill_`** — i.e. ~95 % of GPU time is spent writing that dense tensor.
The entire sparse UNet is 3.4 %. At Waymo-scale grids (512×512×32, 25× fewer
voxels) this op was cheap; our long-range Innoviz grid is what exposed it.

This was a non-issue during training only because training never measured
per-frame wall time; it also explains the high training-time GPU memory.

## Fixes, measured

### 1. Sparse-gather decoder (do this first — it is "free")

Look the pc0 voxel coords up directly in the sparse tensor (sorted-key binary
search over `indices`) instead of densifying. Implemented + validated in
`tools/benchmark_latency.py::decoder_forward_sparse_gather`:

- exact: max |Δflow| = 3.7e-8 vs stock (pure float reordering)
- e2e 359 → **86.4 ms** (11.6 FPS), peak GPU mem 13.6 → **1.67 GB**
- inference-only change; no retraining; ~25 lines

To productionize, add a `decoder_option` variant in `DeltaFlow` /
`Point_head` that uses the sparse gather at inference.

### 2. fp16 backbone (small win)

Autocasting MinkUNet to fp16 (voxelizer must stay fp32 — mmcv's
`feats_reduce_kernel` has no Half support) saves only ~3 ms mean / ~32 ms p99:
the spconv backbone is index-bookkeeping-bound, not FLOP-bound. Flow deviation
4.4e-4 m (vs the 0.05 m dynamic threshold). Worth keeping (p99 104 vs 136 ms)
but validate metrics first; not the main lever.

### 3. Cache history-frame voxelization (estimated, static sensor only)

Each forward voxelizes 5 frames, but 4 of them (pc0, pch1–3) were already
voxelized on previous ticks. Because the sensor is static (pose warp ≈
identity), voxel coords and PFN features of past frames don't change and can
be cached, leaving only the newest frame's voxelize+PFN (~9.5 ms) per tick.
Estimated saving ≈ 4 × 9.2 ≈ **37 ms** → steady-state ≈ 50–65 ms typical.
Caveat: estimate only (not implemented); breaks if the sensor ever moves.

## Scaling behavior / tail risk

Latency correlates strongly with point count (r = 0.81 vs pc0 points).
Per-scene means with the sparse-gather decoder:

| Sequence | pc0 pts | Δ-voxels | opt e2e (max) | +fp16 |
|---|---|---|---|---|
| dune_bushes_hills (23_3) | 30k | 6.5k | 72 ms (77) | 76 |
| innoviz_office | 98k | 41k | 76 ms (83) | 76 |
| meginim_11_02_2026 | 224k | 84k | 87 ms (88) | 81 |
| yellow_field_location2 (23_3) | 250k | 32k | 84 ms (98) | 83 |
| palmam_test (car/even4/event1) | ~272k | ~74k | 81 ms (85) | 81 |
| dark_yarkon_park #1 (22_3) | 490k | 76k | 93 ms (95) | 95 |
| **dark_yarkon_park #2 (22_3)** | **518k** | **177k** | **122 ms (142)** | **99** |

The dark-park scenes carry 5× the median point load (~520k non-ground points
vs 225k median) and are the only ones that miss the 100 ms deadline after the
decoder fix. Mitigations, in order of preference: voxelization caching (#3
above, covers it with margin), point-count capping / range-based subsampling
on the input cloud, or investigating why these recordings keep so many
non-ground returns.

## Verdict

- **As trained/checkpointed: not deployable** — 2.8 FPS on an L4, 3.6× over
  the 100 ms budget, 13.6 GB GPU memory.
- **With the sparse-gather decoder (exact, inference-only): yes, marginally** —
  11.6 FPS mean, 1.7 GB memory, but ~9 % of heavy-scene frames still exceed
  100 ms.
- **With decoder fix + history-voxel caching (+ optional fp16 backbone):
  deployable with headroom** — estimated 50–65 ms/frame (~16–20 FPS) on the L4,
  worst observed scene ≈ 90 ms.

## Reproduce

```bash
.venv/bin/python tools/benchmark_latency.py \
  --checkpoint logs/jobs/deltaflow-5f-waymo/05-27-22-16/checkpoints/last.ckpt \
  --data_dir /home/ubuntu/orr/data/innoviz_h5/pbench/val \
  --num_samples 120 --warmup 10 --opt_decoder --amp --profile
```

Raw outputs (per-sample JSON, op-level profiler table, chrome trace):
`logs/benchmark/deltaflow-5f-waymo/{results.json,profiler_top_ops.txt,trace.json}`
(benchmarked 2026-06-04).
