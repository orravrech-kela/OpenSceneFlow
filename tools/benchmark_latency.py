"""
Benchmark per-frame inference latency of a trained OpenSceneFlow checkpoint
(DeltaFlow) on real H5 sequences, broken down by block/method/operation.

Answers: can this model run in realtime (e.g. 10FPS => 100 ms/frame budget)?

Measures, per sample:
  * data side  : H5 fetch + collate (CPU), H2D copy, ground-mask strip
  * model e2e  : free-running model(batch) wall time (sync at end only)
  * stages     : instrumented re-run of the same forward with CUDA-sync'd
                 timers around every sub-block (voxelizer per frame, PFN per
                 frame, sparse accumulate, coalesce, each MinkUNet stage,
                 decoder dense() and gather+MLP)

The instrumented forward replicates DeltaFlow.forward / SparseVoxelNet.forward
/ MinkUNet.forward / Point_head.forward 1:1 and is verified against the plain
model(batch) output on the first sample (asserts allclose).

Usage:
  .venv/bin/python tools/benchmark_latency.py \
    --checkpoint logs/jobs/deltaflow-5f-waymo/05-27-22-16/checkpoints/last.ckpt \
    --data_dir /home/ubuntu/orr/data/innoviz_h5/pbench/val \
    --num_samples 100 --warmup 10 --profile
"""

import argparse, json, os, sys, time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.dataset import HDF5Dataset  # noqa: E402


# ---------------------------------------------------------------- timing ----
class StageTimer:
    """Wall-clock timers with CUDA sync at both edges; keeps all samples."""

    def __init__(self):
        self.records = defaultdict(list)
        self.enabled = True

    @contextmanager
    def time(self, name):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        self.records[name].append((time.perf_counter() - t0) * 1e3)

    def pop_last(self):
        return {k: v[-1] for k, v in self.records.items()}


def percentiles(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)), "min": float(a.min()), "max": float(a.max()),
        "n": int(a.size),
    }


# ----------------------------------------------------------- model build ----
def build_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = DictConfig(ckpt["hyper_parameters"]).cfg
    print(f"[ckpt] model={cfg.model.name} num_frames={cfg.num_frames} "
          f"voxel_size={list(cfg.voxel_size)} pcr={list(cfg.point_cloud_range)} "
          f"grid={list(cfg.model.target.grid_feature_size)} epoch={ckpt.get('epoch')}")
    model = instantiate(cfg.model.target)
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    if missing:
        print(f"[ckpt][warn] missing keys (left at init): {missing[:5]}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M")
    return model.cuda().eval(), cfg


# ------------------------------------------------------ batch preparation ----
def batch_to_cuda(batch):
    return {k: (v.cuda(non_blocking=False) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def strip_ground(batch, num_frames):
    # mirrors ModelWrapper.run_model_wo_ground_data (val/test path)
    batch["pc0"] = batch["pc0"][~batch["gm0"]].unsqueeze(0)
    batch["pc1"] = batch["pc1"][~batch["gm1"]].unsqueeze(0)
    for i in range(1, num_frames - 1):
        batch[f"pch{i}"] = batch[f"pch{i}"][~batch[f"gmh{i}"]].unsqueeze(0)
    return batch


# ------------------------------------------------- instrumented forwards ----
def voxel_forward_instrumented(net, input_dict, t):
    """Replicates SparseVoxelNet.forward with per-step timers."""
    import spconv.pytorch as spconv

    bz_ = len(input_dict["pc0s"])
    frame_keys = sorted([k for k in input_dict if k.startswith("pch")], reverse=True)
    frame_keys += ["pc0s"]

    with t.time("voxelize/pc1/dyn_voxelize"):
        pc1_voxel_info_list = net.voxelizer(input_dict["pc1s"])
    with t.time("voxelize/pc1/pfn"):
        pc1_voxel_feats_sp, pc1_coors_batch_sp = net.process_batch(pc1_voxel_info_list)
    pc1s_num_voxels = pc1_voxel_feats_sp.shape[0]
    sparse_max_size = [bz_, *net.voxel_spatial_shape, net.num_feature]
    with t.time("voxelize/sparse_accumulate"):
        sparse_pc1 = torch.sparse_coo_tensor(pc1_coors_batch_sp.t(), pc1_voxel_feats_sp, size=sparse_max_size)
        sparse_diff = torch.sparse_coo_tensor(pc1_coors_batch_sp.t(), pc1_voxel_feats_sp * 0.0, size=sparse_max_size)
    pch1s_3dvoxel_infos_lst = None
    pc0_point_feats_lst = []
    pc0s_3dvoxel_infos_lst, pc0s_num_voxels = None, 0

    for time_index, frame_key in enumerate(reversed(frame_keys)):
        tag = frame_key[:-1]  # pc0s -> pc0
        pc = input_dict[frame_key]
        with t.time(f"voxelize/{tag}/dyn_voxelize"):
            voxel_info_list = net.voxelizer(pc)
        with t.time(f"voxelize/{tag}/pfn"):
            if frame_key == "pc0s":
                voxel_feats_sp, coors_batch_sp, pc0_point_feats_lst = net.process_batch(
                    voxel_info_list, if_return_point_feats=True)
            else:
                voxel_feats_sp, coors_batch_sp = net.process_batch(voxel_info_list)
        with t.time("voxelize/sparse_accumulate"):
            sparse_pcx = torch.sparse_coo_tensor(coors_batch_sp.t(), voxel_feats_sp, size=sparse_max_size)
            sparse_diff = sparse_diff + (sparse_pc1 - sparse_pcx) * pow(net.decay_factor, time_index)

        if frame_key == "pc0s":
            pc0s_3dvoxel_infos_lst = voxel_info_list
            pc0s_num_voxels = voxel_feats_sp.shape[0]
        elif frame_key == "pch1s":
            pch1s_3dvoxel_infos_lst = voxel_info_list

    with t.time("voxelize/coalesce"):
        coalesced = sparse_diff.coalesce()
        features = coalesced.values() / (time_index + 1)
        indices = coalesced.indices().t().to(dtype=torch.int32)
        all_pcdiff_sparse = spconv.SparseConvTensor(
            features.contiguous(), indices.contiguous(), net.voxel_spatial_shape, bz_)

    return {
        "delta_sparse": all_pcdiff_sparse,
        "pch1_3dvoxel_infos_lst": pch1s_3dvoxel_infos_lst,
        "pc0_3dvoxel_infos_lst": pc0s_3dvoxel_infos_lst,
        "pc0_point_feats_lst": pc0_point_feats_lst,
        "pc0_num_voxels": pc0s_num_voxels,
        "pc1_3dvoxel_infos_lst": pc1_voxel_info_list,
        "pc1_num_voxels": pc1s_num_voxels,
        "d_num_voxels": indices.shape[0],
    }


def unet_forward_instrumented(net, x, t):
    """Replicates MinkUNet.forward with per-stage timers."""
    with t.time("backbone/conv_input"):
        x = net.conv_input(x)
    with t.time("backbone/stage1"):
        x1 = net.stage1(x)
    with t.time("backbone/stage2"):
        x2 = net.stage2(x1)
    with t.time("backbone/stage3"):
        x3 = net.stage3(x2)
    with t.time("backbone/stage4"):
        x4 = net.stage4(x3)
    with t.time("backbone/up1"):
        y1 = net.up1[0](x4)
        y1 = y1.replace_feature(torch.cat([y1.features, x3.features], dim=1))
        y1 = net.up1[1](y1)
    with t.time("backbone/up2"):
        y2 = net.up2[0](y1)
        y2 = y2.replace_feature(torch.cat([y2.features, x2.features], dim=1))
        y2 = net.up2[1](y2)
    with t.time("backbone/up3"):
        y3 = net.up3[0](y2)
        y3 = y3.replace_feature(torch.cat([y3.features, x1.features], dim=1))
        y3 = net.up3[1](y3)
    with t.time("backbone/up4"):
        y4 = net.up4[0](y3)
        y4 = y4.replace_feature(torch.cat([y4.features, x.features], dim=1))
        y4 = net.up4[1](y4)
    return y4


def decoder_forward_instrumented(net, sparse_tensor, voxelizer_infos, pc0_point_feats_lst, t):
    """Replicates Point_head.forward with dense() timed separately."""
    with t.time("decoder/to_dense"):
        voxel_feats = sparse_tensor.dense()
    with t.time("decoder/gather_mlp"):
        flow_outputs = []
        for batch_idx, voxelizer_info in enumerate(voxelizer_infos):
            voxel_coords = voxelizer_info["voxel_coords"]
            point_feat = pc0_point_feats_lst[batch_idx]
            voxel_feat = voxel_feats[batch_idx, :]
            flow = net.forward_single(voxel_feat, voxel_coords, point_feat)
            flow_outputs.append(flow)
    return flow_outputs


def model_forward_instrumented(model, batch, t):
    from src.models.basic import wrap_batch_pcs
    with t.time("preprocess/pose_warp"):
        pcs_dict = wrap_batch_pcs(batch, num_frames=model.num_frames)
    with t.time("voxelize"):
        sparse_dict = voxel_forward_instrumented(model.pc2voxel, pcs_dict, t)
    with t.time("backbone"):
        backbone_res = unet_forward_instrumented(model.backbone, sparse_dict["delta_sparse"], t)
    with t.time("decoder"):
        flows = decoder_forward_instrumented(
            model.flowdecoder, backbone_res,
            sparse_dict["pc0_3dvoxel_infos_lst"], sparse_dict["pc0_point_feats_lst"], t)
    return flows, sparse_dict


def decoder_forward_sparse_gather(net, sparse_tensor, voxelizer_infos, pc0_point_feats_lst):
    """Point_head equivalent that skips .dense(): looks up per-point voxel
    features directly in the sparse tensor via sorted-key binary search."""
    indices = sparse_tensor.indices.long()  # [N, 4] = (batch, s0, s1, s2)
    feats = sparse_tensor.features
    s0, s1, s2 = (int(x) for x in sparse_tensor.spatial_shape)
    keys = ((indices[:, 0] * s0 + indices[:, 1]) * s1 + indices[:, 2]) * s2 + indices[:, 3]
    order = torch.argsort(keys)
    sorted_keys = keys[order]

    flow_outputs = []
    for b, info in enumerate(voxelizer_infos):
        vc = info["voxel_coords"].long()
        # mirror Point_head.forward_single's dense indexing [:, vc2, vc1, vc0]
        q = ((torch.full_like(vc[:, 2], b) * s0 + vc[:, 2]) * s1 + vc[:, 1]) * s2 + vc[:, 0]
        pos = torch.searchsorted(sorted_keys, q).clamp(max=sorted_keys.numel() - 1)
        voxel_to_point_feat = feats[order[pos]]
        miss = sorted_keys[pos] != q  # absent voxel == zero in the dense path
        if miss.any():
            voxel_to_point_feat = voxel_to_point_feat.masked_fill(miss.unsqueeze(1), 0)
        concat = torch.cat([voxel_to_point_feat, pc0_point_feats_lst[b]], dim=-1)
        flow_outputs.append(net.PPmodel_flow(concat))
    return flow_outputs


def model_forward_opt_decoder(model, batch, amp_backbone=False):
    """Full forward with the sparse-gather decoder (no nested timers).
    amp_backbone: run MinkUNet under fp16 autocast (mmcv voxelizer kernels
    have no Half support, so the voxelizer stays fp32)."""
    from src.models.basic import wrap_batch_pcs
    pcs_dict = wrap_batch_pcs(batch, num_frames=model.num_frames)
    sparse_dict = model.pc2voxel(pcs_dict)
    if amp_backbone:
        with torch.autocast("cuda", dtype=torch.float16):
            backbone_res = model.backbone(sparse_dict["delta_sparse"])
        backbone_res = backbone_res.replace_feature(backbone_res.features.float())
    else:
        backbone_res = model.backbone(sparse_dict["delta_sparse"])
    flows = decoder_forward_sparse_gather(
        model.flowdecoder, backbone_res,
        sparse_dict["pc0_3dvoxel_infos_lst"], sparse_dict["pc0_point_feats_lst"])
    return flows


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="/home/ubuntu/orr/dev/forks/OpenSceneFlow/logs/jobs/"
                                            "deltaflow-5f-waymo/05-27-22-16/checkpoints/last.ckpt")
    ap.add_argument("--data_dir", default="/home/ubuntu/orr/data/innoviz_h5/pbench/val")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--target_fps", type=float, default=10.0)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--profile", action="store_true", help="also run torch.profiler on a few batches")
    ap.add_argument("--amp", action="store_true", help="additionally measure fp16 autocast e2e latency")
    ap.add_argument("--opt_decoder", action="store_true",
                    help="additionally measure e2e with a sparse-gather decoder (skips .dense())")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    gpu = torch.cuda.get_device_name(0)
    out_dir = args.output_dir or os.path.join(
        BASE_DIR, "logs", "benchmark", time.strftime("%m-%d-%H-%M"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[env] gpu={gpu} torch={torch.__version__} out={out_dir}")

    model, cfg = build_model(args.checkpoint)
    num_frames = int(cfg.num_frames)

    ds = HDF5Dataset(args.data_dir, n_frames=num_frames)
    n = len(ds)
    idxs = np.unique(np.linspace(0, n - 1, args.num_samples).astype(int))
    loader = DataLoader(Subset(ds, idxs.tolist()), batch_size=1, shuffle=False, num_workers=0)
    print(f"[data] {n} frames in index, sampling {len(idxs)} evenly")

    timer = StageTimer()
    records = []
    budget_ms = 1000.0 / args.target_fps

    # ---- warmup (spconv/cudnn/allocator) on the first few batches
    warm_iter = iter(loader)
    with torch.inference_mode():
        for i in range(min(args.warmup, len(idxs))):
            batch = strip_ground(batch_to_cuda(next(warm_iter)), num_frames)
            model(batch)
    torch.cuda.synchronize()
    print(f"[warmup] done ({min(args.warmup, len(idxs))} iters)")

    # ---- sanity: instrumented forward == plain forward on one batch
    with torch.inference_mode():
        batch = strip_ground(batch_to_cuda(next(iter(loader))), num_frames)
        ref = model(batch)["flow"][0]
        t_check = StageTimer()
        flows, _ = model_forward_instrumented(model, batch, t_check)
        max_diff = (ref - flows[0]).abs().max().item()
        assert max_diff < 1e-3, f"instrumented forward diverges: max_diff={max_diff}"
        print(f"[sanity] instrumented == plain forward (max_diff={max_diff:.2e})")
        if args.opt_decoder:
            opt_flows = model_forward_opt_decoder(model, batch)
            opt_diff = (ref - opt_flows[0]).abs().max().item()
            assert opt_diff < 1e-3, f"sparse-gather decoder diverges: max_diff={opt_diff}"
            print(f"[sanity] sparse-gather decoder == dense decoder (max_diff={opt_diff:.2e})")
        if args.amp:
            try:
                amp_flows = model_forward_opt_decoder(model, batch, amp_backbone=True)
                amp_diff = (ref - amp_flows[0]).abs().max().item()
                print(f"[sanity] fp16-backbone flow vs fp32: max_diff={amp_diff:.2e} (info only)")
            except RuntimeError as e:
                print(f"[amp][warn] fp16 backbone unsupported, disabling --amp: {e}")
                args.amp = False

    torch.cuda.reset_peak_memory_stats()

    # ---- main measurement loop
    fetch_t0 = time.perf_counter()
    with torch.inference_mode():
        for bi, batch_cpu in enumerate(loader):
            data_ms = (time.perf_counter() - fetch_t0) * 1e3

            torch.cuda.synchronize(); t0 = time.perf_counter()
            batch = batch_to_cuda(batch_cpu)
            torch.cuda.synchronize(); h2d_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            batch = strip_ground(batch, num_frames)
            torch.cuda.synchronize(); strip_ms = (time.perf_counter() - t0) * 1e3

            n_pts = {k: int(batch[k].shape[1]) for k in batch
                     if k.startswith("pc") and isinstance(batch[k], torch.Tensor) and batch[k].dim() == 3}

            # free-running end-to-end (the deployment-relevant number)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            res = model(batch)
            torch.cuda.synchronize(); e2e_ms = (time.perf_counter() - t0) * 1e3

            # instrumented re-run on the identical input
            _, sparse_dict = model_forward_instrumented(model, batch, timer)
            stages = timer.pop_last()

            amp_ms = None
            if args.amp:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model_forward_opt_decoder(model, batch, amp_backbone=True)
                torch.cuda.synchronize(); amp_ms = (time.perf_counter() - t0) * 1e3

            opt_ms = None
            if args.opt_decoder:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model_forward_opt_decoder(model, batch)
                torch.cuda.synchronize(); opt_ms = (time.perf_counter() - t0) * 1e3

            records.append({
                "scene_id": batch["scene_id"][0],
                "timestamp": str(batch["timestamp"][0].item() if isinstance(batch["timestamp"], torch.Tensor)
                                 else batch["timestamp"][0]),
                "n_points": n_pts,
                "pc0_num_voxels": int(sparse_dict["pc0_num_voxels"]),
                "pc1_num_voxels": int(sparse_dict["pc1_num_voxels"]),
                "d_num_voxels": int(sparse_dict["d_num_voxels"]),
                "data_ms": data_ms, "h2d_ms": h2d_ms, "strip_ms": strip_ms,
                "e2e_ms": e2e_ms, "amp_e2e_ms": amp_ms, "opt_e2e_ms": opt_ms,
                "stages_ms": stages,
            })
            if (bi + 1) % 20 == 0:
                print(f"  [{bi+1}/{len(idxs)}] e2e={e2e_ms:.1f}ms")
            fetch_t0 = time.perf_counter()

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    # ---- optional op-level profiler on a few batches
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        prof_iter = iter(loader)
        with torch.inference_mode():
            pb = strip_ground(batch_to_cuda(next(prof_iter)), num_frames)
            model(pb)  # re-warm
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                for _ in range(3):
                    model(pb)
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        with open(os.path.join(out_dir, "profiler_top_ops.txt"), "w") as f:
            f.write(table)
        prof.export_chrome_trace(os.path.join(out_dir, "trace.json"))
        print(f"[profile] wrote profiler_top_ops.txt and trace.json")

    # ---- aggregate + write
    agg = {
        "e2e_ms": percentiles([r["e2e_ms"] for r in records]),
        "data_ms": percentiles([r["data_ms"] for r in records]),
        "h2d_ms": percentiles([r["h2d_ms"] for r in records]),
        "strip_ms": percentiles([r["strip_ms"] for r in records]),
        "stages_ms": {},
    }
    if args.amp:
        agg["amp_e2e_ms"] = percentiles([r["amp_e2e_ms"] for r in records])
    if args.opt_decoder:
        agg["opt_e2e_ms"] = percentiles([r["opt_e2e_ms"] for r in records])
    for k in sorted(timer.records):
        agg["stages_ms"][k] = percentiles(timer.records[k])
    per_scene = defaultdict(list)
    for r in records:
        per_scene[r["scene_id"]].append(r["e2e_ms"])
    agg["e2e_ms_per_scene"] = {s: percentiles(v) for s, v in sorted(per_scene.items())}

    result = {
        "checkpoint": args.checkpoint, "data_dir": args.data_dir,
        "gpu": gpu, "torch": torch.__version__,
        "num_frames": num_frames,
        "voxel_size": list(cfg.voxel_size), "point_cloud_range": list(cfg.point_cloud_range),
        "grid_feature_size": list(cfg.model.target.grid_feature_size),
        "target_fps": args.target_fps, "budget_ms": budget_ms,
        "peak_gpu_mem_gb": peak_mem_gb,
        "aggregates": agg, "samples": records,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=1)

    # ---- console summary
    e2e = agg["e2e_ms"]
    print("\n================ LATENCY SUMMARY ================")
    print(f"GPU: {gpu} | peak mem {peak_mem_gb:.2f} GB | {len(records)} samples")
    print(f"e2e model forward: mean {e2e['mean']:.1f} | p50 {e2e['p50']:.1f} | "
          f"p90 {e2e['p90']:.1f} | p99 {e2e['p99']:.1f} ms")
    print(f"=> sustained {1000.0/e2e['mean']:.2f} FPS vs target {args.target_fps} FPS "
          f"(budget {budget_ms:.0f} ms): "
          f"{'PASS' if e2e['p90'] <= budget_ms else 'FAIL'} (p90 basis)")
    if args.amp:
        amp = agg["amp_e2e_ms"]
        print(f"opt-decoder + fp16-backbone e2e: mean {amp['mean']:.1f} | p50 {amp['p50']:.1f} | "
              f"p90 {amp['p90']:.1f} ms => {1000.0/amp['mean']:.2f} FPS")
    if args.opt_decoder:
        opt = agg["opt_e2e_ms"]
        print(f"sparse-gather-decoder e2e: mean {opt['mean']:.1f} | p50 {opt['p50']:.1f} | "
              f"p90 {opt['p90']:.1f} | p99 {opt['p99']:.1f} ms "
              f"=> {1000.0/opt['mean']:.2f} FPS "
              f"({'PASS' if opt['p90'] <= budget_ms else 'FAIL'} vs {args.target_fps} FPS, p90 basis)")
    print("\nper-stage mean ms (sync-bounded):")
    for k in sorted(agg["stages_ms"]):
        s = agg["stages_ms"][k]
        indent = "  " * k.count("/")
        print(f"  {indent}{k:<42s} {s['mean']:8.2f} (p90 {s['p90']:.2f})")
    print(f"\nresults.json written to {out_dir}")


if __name__ == "__main__":
    main()
