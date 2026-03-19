"""
spec_kan_fps_opt_streaming_eval.py：Streaming/block-wise seg_filter FPS 与已有 seg_filter 的对比评估

目的：
  - 将「逐 block 更新 min_dist + segment 阈值 + heap 候选」的 fps_streaming_block_seg_filter
    与 spec_kan_fps_opt 中的 seg_filter（全点一次扫描 + 同阈值规则）做 IOU 与耗时对比。
  - 输出：exact / seg_filter / streaming 三者的 IOU（相对 exact）、耗时、speedup。

流程：
  1. 每批先跑 exact 得到 idx_exact、radii_ref
  2. 跑 seg_filter（来自 spec_kan_fps_opt）得到 idx_seg，计时
  3. 跑 fps_streaming_block_seg_filter 得到 idx_stream，计时
  4. 计算 IOU(exact, seg_filter)、IOU(exact, streaming)
  5. 汇总 mean_iou、std_iou、总耗时、speedup（exact/seg_filter、exact/streaming）
"""
import os
import sys
import json
import argparse
import time
import numpy as np
from typing import Optional, Tuple, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 已有 seg_filter、exact、数据集、IOU
from spec_kan_fps_opt import (
    fps_adaptive_argmax,
    compute_iou_batch,
    ModelNet40Wrapper,
    collate_points,
)


# =========================================================
# 1. Streaming/block-wise FPS with segment-threshold (你提供的实现)
# =========================================================

@torch.no_grad()
def fps_streaming_block_seg_filter(
    X: torch.Tensor,
    K: int,
    seed: str = "first",
    radii_ref: Optional[torch.Tensor] = None,
    segment_len: int = 32,
    block_len: int = 64,
    heap_cap: int = 32,
    empty_fallback: str = "global",
    store_distances: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Streaming/block-wise FPS with segment-threshold candidate filtering.
    Returns: idx (B,K), radii (B,K), distances_matrix or None, stats dict.
    """
    assert X.ndim == 3, f"X must be (B,N,D), got shape={tuple(X.shape)}"
    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    K = min(K, N)
    if K <= 0:
        raise ValueError("K must be positive")
    if heap_cap <= 0:
        raise ValueError("heap_cap must be positive")
    if block_len <= 0:
        raise ValueError("block_len must be positive")

    x2 = (X * X).sum(dim=-1)

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=dtype, device=device)
    distances_matrix = (
        torch.zeros((B, K, N), dtype=dtype, device=device) if store_distances else None
    )

    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)

    candidate_ratio = torch.ones((B, K), dtype=dtype, device=device)
    heap_fill_ratio = torch.zeros((B, K), dtype=dtype, device=device)
    used_fallback = torch.zeros((B, K), dtype=torch.bool, device=device)
    threshold_tau = torch.zeros((B, K), dtype=dtype, device=device)
    num_candidates_before_heap = torch.zeros((B, K), dtype=torch.long, device=device)

    batch_arange = torch.arange(B, device=device)

    def sqdist_to_center_all(C: torch.Tensor) -> torch.Tensor:
        c2 = (C * C).sum(dim=-1, keepdim=True)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0)
        return d2

    def sqdist_to_center_block(Xb: torch.Tensor, xb2: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        c2 = (C * C).sum(dim=-1, keepdim=True)
        dot = (Xb @ C.unsqueeze(-1)).squeeze(-1)
        d2 = (xb2 + c2 - 2.0 * dot).clamp_min_(0.0)
        return d2

    def normalize_radii_ref(rref: torch.Tensor) -> torch.Tensor:
        if rref.shape[:2] != (B, K):
            raise ValueError(f"radii_ref shape must be (B,K)={(B,K)}, got {tuple(rref.shape)}")
        if torch.numel(rref) == 0:
            return rref
        med_x2 = torch.median(x2)
        med_r = torch.median(rref)
        if med_r > 0 and (med_r * med_r - med_x2).abs() < (med_r - med_x2).abs():
            return rref * rref
        return rref

    def get_tau_for_step(t: int, rref: torch.Tensor) -> torch.Tensor:
        L = max(1, int(segment_len))
        seg_id = t // L
        seg_end = min((seg_id + 1) * L - 1, K - 1)
        return rref[:, seg_end]

    def merge_heap(heap_vals: torch.Tensor, heap_idx: torch.Tensor,
                   cand_vals: torch.Tensor, cand_idx: torch.Tensor, H: int) -> Tuple[torch.Tensor, torch.Tensor]:
        all_vals = torch.cat([heap_vals, cand_vals], dim=1)
        all_idx = torch.cat([heap_idx, cand_idx], dim=1)
        new_vals, sel = torch.topk(all_vals, k=H, dim=1)
        new_idx = all_idx.gather(1, sel)
        return new_vals, new_idx

    if seed == "first":
        first = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        first = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    idx[:, 0] = first
    C0 = X[batch_arange, first]
    d2_0 = sqdist_to_center_all(C0)

    if store_distances:
        distances_matrix[:, 0, :] = d2_0

    min_dist.copy_(d2_0)
    radii[:, 0], _ = torch.max(min_dist, dim=1)
    min_dist[batch_arange, first] = 0.0

    candidate_ratio[:, 0] = 1.0
    heap_fill_ratio[:, 0] = 1.0
    threshold_tau[:, 0] = 0.0
    num_candidates_before_heap[:, 0] = N

    if K > 1:
        if radii_ref is None:
            raise ValueError("radii_ref must be provided for streaming_block_seg_filter")
        radii_ref = normalize_radii_ref(radii_ref)

    for t in range(1, K):
        prev_center_idx = idx[:, t - 1]
        prev_center = X[batch_arange, prev_center_idx]

        tau = get_tau_for_step(t, radii_ref)
        threshold_tau[:, t] = tau

        heap_vals = torch.full((B, heap_cap), -float("inf"), dtype=dtype, device=device)
        heap_idx = torch.full((B, heap_cap), -1, dtype=torch.long, device=device)

        global_best_val = torch.full((B,), -float("inf"), dtype=dtype, device=device)
        global_best_idx = torch.zeros((B,), dtype=torch.long, device=device)

        cand_cnt_total = torch.zeros((B,), dtype=torch.long, device=device)

        if store_distances:
            d2_full_t = torch.empty((B, N), dtype=dtype, device=device)

        for s in range(0, N, block_len):
            e = min(s + block_len, N)
            Lb = e - s

            Xb = X[:, s:e, :]
            xb2 = x2[:, s:e]
            old_md = min_dist[:, s:e]

            d2_blk = sqdist_to_center_block(Xb, xb2, prev_center)

            if store_distances:
                d2_full_t[:, s:e] = d2_blk

            new_md = torch.minimum(old_md, d2_blk)
            min_dist[:, s:e] = new_md

            blk_best_val, blk_best_pos = torch.max(new_md, dim=1)
            blk_best_idx = blk_best_pos + s

            better_global = blk_best_val > global_best_val
            global_best_val = torch.where(better_global, blk_best_val, global_best_val)
            global_best_idx = torch.where(better_global, blk_best_idx, global_best_idx)

            mask = new_md >= tau.unsqueeze(1)
            cand_cnt_total += mask.sum(dim=1)

            vals_masked = new_md.masked_fill(~mask, -float("inf"))

            local_k = min(heap_cap, Lb)
            blk_top_vals, blk_top_pos = torch.topk(vals_masked, k=local_k, dim=1)
            blk_top_idx = blk_top_pos + s

            heap_vals, heap_idx = merge_heap(heap_vals, heap_idx, blk_top_vals, blk_top_idx, heap_cap)

        if store_distances:
            distances_matrix[:, t, :] = d2_full_t

        num_candidates_before_heap[:, t] = cand_cnt_total
        candidate_ratio[:, t] = cand_cnt_total.to(dtype) / float(N)

        heap_valid = heap_idx >= 0
        heap_fill_ratio[:, t] = heap_valid.sum(dim=1).to(dtype) / float(heap_cap)

        heap_best_val, heap_best_slot = torch.max(heap_vals, dim=1)
        heap_best_idx = heap_idx[batch_arange, heap_best_slot]
        has_candidate = heap_best_idx >= 0

        if empty_fallback != "global":
            raise NotImplementedError(f"empty_fallback={empty_fallback} is not implemented")

        farthest = torch.where(has_candidate, heap_best_idx, global_best_idx)
        used_fallback[:, t] = ~has_candidate

        idx[:, t] = farthest
        radii[:, t] = global_best_val

        min_dist[batch_arange, farthest] = 0.0

    stats = {
        "candidate_ratio": candidate_ratio,
        "heap_fill_ratio": heap_fill_ratio,
        "used_fallback": used_fallback,
        "threshold_tau": threshold_tau,
        "num_candidates_before_heap": num_candidates_before_heap,
    }
    return idx, radii, distances_matrix, stats


# =========================================================
# 2. 评估：exact / seg_filter / streaming，计时与 IOU
# =========================================================

def _time_cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run_evaluation(
    loader: DataLoader,
    K: int,
    segment_len: int,
    seed: str,
    device: torch.device,
    num_points: int,
    block_len: int = 64,
    heap_cap: int = 32,
    warmup_batches: int = 2,
) -> Dict:
    """
    对每批：exact → radii_ref；再分别跑 seg_filter 与 streaming_block_seg_filter，计时并算 IOU。
    返回汇总：mean_iou_seg, std_iou_seg, mean_iou_streaming, std_iou_streaming,
             time_exact, time_seg, time_streaming, speedup_seg, speedup_streaming, 以及两套 stats。
    """
    all_ious_seg: List[float] = []
    all_ious_streaming: List[float] = []
    all_cr_seg: List[torch.Tensor] = []
    all_cr_streaming: List[torch.Tensor] = []
    all_fallback_seg: List[torch.Tensor] = []
    all_fallback_streaming: List[torch.Tensor] = []

    time_exact = 0.0
    time_seg = 0.0
    time_streaming = 0.0
    n_batches = 0

    it = tqdm(loader, desc="IOU & timing", ncols=100)
    for batch_idx, (pts, _) in enumerate(it):
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)

        # ----- Warmup (skip timing) -----
        if batch_idx < warmup_batches:
            idx_exact, radii_ref, _, _ = fps_adaptive_argmax(pts, K, seed=seed, mode="exact")
            _time_cuda_sync(device)
            fps_adaptive_argmax(pts, K, seed=seed, mode="seg_filter", radii_ref=radii_ref, segment_len=segment_len)
            _time_cuda_sync(device)
            fps_streaming_block_seg_filter(
                pts, K, seed=seed, radii_ref=radii_ref, segment_len=segment_len,
                block_len=block_len, heap_cap=heap_cap,
            )
            _time_cuda_sync(device)
            continue

        # ----- Exact + radii_ref -----
        _time_cuda_sync(device)
        t0 = time.perf_counter()
        idx_exact, radii_ref, _, _ = fps_adaptive_argmax(pts, K, seed=seed, mode="exact")
        _time_cuda_sync(device)
        time_exact += time.perf_counter() - t0

        # ----- seg_filter -----
        _time_cuda_sync(device)
        t0 = time.perf_counter()
        idx_seg, _, _, stats_seg = fps_adaptive_argmax(
            pts, K, seed=seed, mode="seg_filter",
            radii_ref=radii_ref, segment_len=segment_len,
        )
        _time_cuda_sync(device)
        time_seg += time.perf_counter() - t0

        # ----- streaming_block_seg_filter -----
        _time_cuda_sync(device)
        t0 = time.perf_counter()
        idx_stream, _, _, stats_stream = fps_streaming_block_seg_filter(
            pts, K, seed=seed, radii_ref=radii_ref, segment_len=segment_len,
            block_len=block_len, heap_cap=heap_cap,
        )
        _time_cuda_sync(device)
        time_streaming += time.perf_counter() - t0

        # ----- IOU -----
        iou_seg_per, _, _ = compute_iou_batch(idx_exact, idx_seg)
        iou_stream_per, _, _ = compute_iou_batch(idx_exact, idx_stream)
        all_ious_seg.extend(iou_seg_per.cpu().numpy().tolist())
        all_ious_streaming.extend(iou_stream_per.cpu().numpy().tolist())
        all_cr_seg.append(stats_seg["candidate_ratio"])
        all_cr_streaming.append(stats_stream["candidate_ratio"])
        all_fallback_seg.append(stats_seg["used_fallback"])
        all_fallback_streaming.append(stats_stream["used_fallback"])

        n_batches += 1

    mean_iou_seg = float(np.mean(all_ious_seg))
    std_iou_seg = float(np.std(all_ious_seg))
    mean_iou_streaming = float(np.mean(all_ious_streaming))
    std_iou_streaming = float(np.std(all_ious_streaming))

    # Speedup: exact 为基准，seg_filter / streaming 的加速比 = time_exact / time_*
    speedup_seg = time_exact / time_seg if time_seg > 0 else 0.0
    speedup_streaming = time_exact / time_streaming if time_streaming > 0 else 0.0

    # 与 spec_kan_fps_opt 一致的统计：candidate_ratio、used_fallback、scan 节省
    def _scan_stats(cr_list: List[torch.Tensor], fallback_list: List[torch.Tensor]) -> Dict:
        cr_cat = torch.cat([t.flatten() for t in cr_list])
        fb_cat = torch.cat([t.flatten() for t in fallback_list])
        cr_stack = torch.cat(cr_list, dim=0)
        per_step_mean = cr_stack.mean(dim=0).cpu().numpy()
        remaining = np.maximum(0, num_points - 1 - np.arange(K))
        total_vanilla = K * num_points - K * (K + 1) // 2
        total_filtered = float(np.sum(per_step_mean * remaining))
        scan_saved = 1.0 - (total_filtered / total_vanilla) if total_vanilla > 0 else 0.0
        return {
            "candidate_ratio_mean": cr_cat.mean().item(),
            "candidate_ratio_std": cr_cat.std().item(),
            "used_fallback_ratio": fb_cat.float().mean().item(),
            "total_scan_vanilla": total_vanilla,
            "total_scan_filtered": total_filtered,
            "scan_saved_ratio": scan_saved,
            "per_step_candidate_ratio_mean": per_step_mean.tolist(),
        }

    summary_seg = _scan_stats(all_cr_seg, all_fallback_seg)
    summary_streaming = _scan_stats(all_cr_streaming, all_fallback_streaming)

    return {
        "num_samples": len(all_ious_seg),
        "n_batches_timed": n_batches,
        "mean_iou_seg_filter": mean_iou_seg,
        "std_iou_seg_filter": std_iou_seg,
        "mean_iou_streaming": mean_iou_streaming,
        "std_iou_streaming": std_iou_streaming,
        "time_exact_sec": time_exact,
        "time_seg_filter_sec": time_seg,
        "time_streaming_sec": time_streaming,
        "speedup_seg_filter": speedup_seg,
        "speedup_streaming": speedup_streaming,
        "summary_seg_filter": summary_seg,
        "summary_streaming": summary_streaming,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Streaming block seg_filter FPS 与 seg_filter 的 IOU/耗时/speedup 对比（ModelNet40）"
    )
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--block_len", type=int, default=64, help="streaming 每块点数")
    p.add_argument("--heap_cap", type=int, default=32, help="streaming 候选堆容量")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=str, default="first")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./runs/fps_seg_filter_streaming_eval")
    p.add_argument("--warmup_batches", type=int, default=2, help="前几批不计时，用于 warmup")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed or "first"

    ds = ModelNet40Wrapper(partition=args.partition, num_points=args.num_points, root=args.root)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        drop_last=False, collate_fn=collate_points,
    )

    result = run_evaluation(
        loader,
        K=args.K,
        segment_len=args.segment_len,
        seed=seed,
        device=device,
        num_points=args.num_points,
        block_len=args.block_len,
        heap_cap=args.heap_cap,
        warmup_batches=args.warmup_batches,
    )

    # 写入 JSON（可序列化：去掉 summary 里的 tensor 等）
    results_json = {
        "partition": args.partition,
        "num_points": args.num_points,
        "K": args.K,
        "segment_len": args.segment_len,
        "block_len": args.block_len,
        "heap_cap": args.heap_cap,
        "seed": seed,
        "num_samples": result["num_samples"],
        "n_batches_timed": result["n_batches_timed"],
        "mean_iou_seg_filter": result["mean_iou_seg_filter"],
        "std_iou_seg_filter": result["std_iou_seg_filter"],
        "mean_iou_streaming": result["mean_iou_streaming"],
        "std_iou_streaming": result["std_iou_streaming"],
        "time_exact_sec": result["time_exact_sec"],
        "time_seg_filter_sec": result["time_seg_filter_sec"],
        "time_streaming_sec": result["time_streaming_sec"],
        "speedup_seg_filter": result["speedup_seg_filter"],
        "speedup_streaming": result["speedup_streaming"],
        "summary_seg_filter": {
            "candidate_ratio_mean": result["summary_seg_filter"]["candidate_ratio_mean"],
            "candidate_ratio_std": result["summary_seg_filter"]["candidate_ratio_std"],
            "used_fallback_ratio": result["summary_seg_filter"]["used_fallback_ratio"],
            "total_scan_vanilla": result["summary_seg_filter"]["total_scan_vanilla"],
            "total_scan_filtered": result["summary_seg_filter"]["total_scan_filtered"],
            "scan_saved_ratio": result["summary_seg_filter"]["scan_saved_ratio"],
        },
        "summary_streaming": {
            "candidate_ratio_mean": result["summary_streaming"]["candidate_ratio_mean"],
            "candidate_ratio_std": result["summary_streaming"]["candidate_ratio_std"],
            "used_fallback_ratio": result["summary_streaming"]["used_fallback_ratio"],
            "total_scan_vanilla": result["summary_streaming"]["total_scan_vanilla"],
            "total_scan_filtered": result["summary_streaming"]["total_scan_filtered"],
            "scan_saved_ratio": result["summary_streaming"]["scan_saved_ratio"],
        },
    }
    out_path = os.path.join(args.save_dir, "streaming_vs_seg_filter_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # 控制台输出
    print("=" * 70)
    print("Streaming block seg_filter vs seg_filter：IOU & 性能 / speedup 分析 (ModelNet40)")
    print("=" * 70)
    print(f"  分区: {args.partition}  N={args.num_points}  K={args.K}  segment_len={args.segment_len}")
    print(f"  block_len={args.block_len}  heap_cap={args.heap_cap}  seed={seed}")
    print(f"  计时的 batch 数: {result['n_batches_timed']}  样本数: {result['num_samples']}")
    print("-" * 70)
    print("IOU (vs exact FPS):")
    print(f"  seg_filter:   mean_iou = {result['mean_iou_seg_filter']:.6f} ± {result['std_iou_seg_filter']:.6f}")
    print(f"  streaming:    mean_iou = {result['mean_iou_streaming']:.6f} ± {result['std_iou_streaming']:.6f}")
    print("-" * 70)
    print("耗时 (秒):")
    print(f"  exact:      {result['time_exact_sec']:.4f}")
    print(f"  seg_filter: {result['time_seg_filter_sec']:.4f}  (speedup vs exact: {result['speedup_seg_filter']:.4f}x)")
    print(f"  streaming:  {result['time_streaming_sec']:.4f}  (speedup vs exact: {result['speedup_streaming']:.4f}x)")
    print("-" * 70)
    print("候选/扫描统计:")
    s_seg = result["summary_seg_filter"]
    s_str = result["summary_streaming"]
    print(f"  seg_filter:  candidate_ratio = {s_seg['candidate_ratio_mean']:.4f}  scan_saved = {s_seg['scan_saved_ratio']:.4f}  fallback_ratio = {s_seg['used_fallback_ratio']:.4f}")
    print(f"  streaming:   candidate_ratio = {s_str['candidate_ratio_mean']:.4f}  scan_saved = {s_str['scan_saved_ratio']:.4f}  fallback_ratio = {s_str['used_fallback_ratio']:.4f}")
    print("=" * 70)
    print(f"  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
