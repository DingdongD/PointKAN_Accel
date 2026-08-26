#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证：max_iter 过小时粗筛是否大量触发 hi_full 回退（h ≈ hi_full），从而 KNN IoU 对精确 topk 仍为 1。

用法（与 sweep 一致的超参）:
  python verify_knn_coarse_hi_fallback.py --device cuda
  python verify_knn_coarse_hi_fallback.py --root ./data --partition test --use_dataloader

默认若无数据则使用固定种子的合成点云 [B,N,3]。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torch.utils.data import DataLoader

from spec_kan_fps import _knn_coarse_hi_full_per_row, knn_neighbor_set_iou
from spec_kan_fps_opt import ModelNet40Wrapper, collate_points, fps_adaptive_argmax
from spec_kan_fps_opt_kan import kan_prefix_len_from_K
from spec_kan_fps_opt_kan_coarse_fine_fused_eval import (
    fps_segfilter_knn_coarse_fine_fused,
    knn_exact_topk_at_centers,
    resolve_knn_coarse_distance_hi_cap,
)


def _ns_for_cap(json_path: str, cap_cli: float) -> argparse.Namespace:
    return argparse.Namespace(
        knn_coarse_hi_json=json_path or "",
        knn_coarse_distance_hi_cap=float(cap_cli),
    )


@torch.no_grad()
def analyze_one_setting(
    pts: torch.Tensor,
    *,
    K: int,
    segment_len: int,
    k_neighbors: int,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    seed: str,
    exclude_self: bool,
    prefix_ratio: float,
    defer_seg_filter: bool,
    distance_hi_cap: Optional[float],
    coarse_schedule: str,
    mult_alpha_clip: float,
    knn_coarse_pool_shortage: str,
    device: torch.device,
) -> dict:
    pts = pts.to(device, dtype=torch.float32)
    B, N, _ = pts.shape
    seg_filter_start_t = (
        kan_prefix_len_from_K(K, prefix_ratio) if defer_seg_filter else 1
    )

    idx_exact, radii_sq_exact, _, _ = fps_adaptive_argmax(
        pts, K, seed=seed, mode="exact"
    )

    idx_f, _r_fps, knn_f, _knr, knn_th, dist_mat, stats = fps_segfilter_knn_coarse_fine_fused(
        pts,
        K,
        k_neighbors,
        radii_sq_exact,
        segment_len,
        init_threshold=init_threshold,
        max_iter=max_iter,
        min_candidates_ratio=min_candidates_ratio,
        seed=seed,
        exclude_self=exclude_self,
        return_distances_matrix=True,
        seg_filter_start_t=seg_filter_start_t,
        distance_hi_cap=distance_hi_cap,
        coarse_schedule=coarse_schedule,
        mult_alpha_clip=mult_alpha_clip,
        knn_coarse_pool_shortage=knn_coarse_pool_shortage,
    )

    cand = stats["knn_coarse_candidate_count"]  # [B, K]
    mean_ratio = float((cand / float(N)).mean().item())

    # 逐步比较 h 与「未 cap 的」hi_full（与 knn_coarse_fine_ball_mask 中回退目标一致）
    hi_full_hits = 0
    total = B * K
    for t in range(K):
        d_bt = dist_mat[:, t, :].clone()
        if exclude_self:
            b_idx = torch.arange(B, device=device)
            cen = idx_f[:, t]
            d_bt[b_idx, cen] = float("inf")
        d_max = d_bt.max(dim=1).values
        hi_full = _knn_coarse_hi_full_per_row(d_max, pts.dtype)
        th = knn_th[:, t]
        # 回退赋值即为 hi_full；允许 float32 误差
        hit = torch.isclose(th, hi_full, rtol=2e-4, atol=1e-5)
        hi_full_hits += int(hit.sum().item())

    frac_hi = hi_full_hits / float(total)

    knn_ref = knn_exact_topk_at_centers(
        pts, idx_f, k_neighbors, exclude_self=exclude_self
    )
    iou = float(knn_neighbor_set_iou(knn_f, knn_ref).mean().item())

    return {
        "mean_knn_iou_vs_exact_at_fused_centers": iou,
        "frac_steps_threshold_eq_hi_full": frac_hi,
        "mean_coarse_candidate_count_ratio": mean_ratio,
        "seg_filter_start_t": int(seg_filter_start_t),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--k_neighbors", type=int, default=24)
    p.add_argument("--min_candidates_ratio", type=float, default=1.2)
    p.add_argument("--seed", type=str, default="first")
    p.add_argument("--exclude_self", action="store_true")
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--defer_seg_filter", action="store_true", default=True)
    p.add_argument("--no_defer_seg_filter", dest="defer_seg_filter", action="store_false")
    p.add_argument("--knn_coarse_schedule", type=str, default="mult_then_bisect")
    p.add_argument("--mult_alpha_clip", type=float, default=0.1)
    p.add_argument("--init_threshold", type=float, default=0.05)
    p.add_argument("--max_iters", type=str, default="2,10")
    p.add_argument("--knn_coarse_hi_json", type=str, default="./runs/knn_coarse_hi_cap.json")
    p.add_argument("--knn_coarse_distance_hi_cap", type=float, default=0.0)
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--partition", type=str, default="test")
    p.add_argument("--use_dataloader", action="store_true")
    p.add_argument("--synthetic_batches", type=int, default=1)
    p.add_argument(
        "--knn_coarse_pool_shortage",
        type=str,
        default="keep_mid",
        choices=("keep_mid", "search_hi"),
        help="与 spec_kan_fps_opt_kan_coarse_fine_fused_eval 一致",
    )
    args = p.parse_args()

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    cap = resolve_knn_coarse_distance_hi_cap(
        _ns_for_cap(args.knn_coarse_hi_json, args.knn_coarse_distance_hi_cap)
    )

    max_iter_list = [int(x.strip()) for x in args.max_iters.split(",") if x.strip()]

    batches: list[torch.Tensor] = []
    if args.use_dataloader and os.path.isdir(args.root):
        ds = ModelNet40Wrapper(
            partition=args.partition, num_points=args.num_points, root=args.root
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=collate_points,
        )
        for i, (pts, _) in enumerate(loader):
            batches.append(pts)
            if i + 1 >= args.synthetic_batches:
                break
        if not batches:
            print("[warn] DataLoader 为空，改用合成点云")
    if not batches:
        torch.manual_seed(0)
        for _ in range(args.synthetic_batches):
            batches.append(torch.rand(args.batch_size, args.num_points, 3) * 2.0 - 1.0)

    print(
        f"设备={device}  cap={cap}  schedule={args.knn_coarse_schedule}  "
        f"mult_alpha_clip={args.mult_alpha_clip}  init_threshold={args.init_threshold}  "
        f"pool_shortage={args.knn_coarse_pool_shortage}  "
        f"defer_seg_filter={args.defer_seg_filter}  prefix_ratio={args.prefix_ratio}"
    )
    print(
        "max_iter | mean IoU | frac(h≈hi_full) | mean |候选|/N | 说明"
    )
    print("-" * 72)

    for mi in max_iter_list:
        mi_u = max(1, int(mi))
        agg = {"iou": [], "fh": [], "cr": []}
        for pts in batches:
            r = analyze_one_setting(
                pts,
                K=args.K,
                segment_len=args.segment_len,
                k_neighbors=args.k_neighbors,
                init_threshold=float(args.init_threshold),
                max_iter=mi_u,
                min_candidates_ratio=args.min_candidates_ratio,
                seed=args.seed,
                exclude_self=args.exclude_self,
                prefix_ratio=args.prefix_ratio,
                defer_seg_filter=args.defer_seg_filter,
                distance_hi_cap=cap,
                coarse_schedule=args.knn_coarse_schedule,
                mult_alpha_clip=float(args.mult_alpha_clip),
                knn_coarse_pool_shortage=str(args.knn_coarse_pool_shortage),
                device=device,
            )
            agg["iou"].append(r["mean_knn_iou_vs_exact_at_fused_centers"])
            agg["fh"].append(r["frac_steps_threshold_eq_hi_full"])
            agg["cr"].append(r["mean_coarse_candidate_count_ratio"])
        print(
            f"{mi_u:8d} | {sum(agg['iou'])/len(agg['iou']):.6f} | "
            f"{sum(agg['fh'])/len(agg['fh']):.4f}     | "
            f"{sum(agg['cr'])/len(agg['cr']):.4f}      | "
        )

    s0 = kan_prefix_len_from_K(args.K, args.prefix_ratio) if args.defer_seg_filter else 1
    print("-" * 72)
    print(
        f"解释: frac(h≈hi_full) 为 (batch, FPS步) 上最终阈值 h 与 hi_full 一致的比例"
        f"（spec_kan_fps.knn_coarse_fine_ball_mask 在 cnt<pool_lo 时令 h=hi_full）。"
        f" seg_filter_start_t={s0}。"
    )
    print(
        "IoU=1 可能来自：(A) hi_full 回退→候选≈全点；(B) 未回退但球仍包含真实 k 近邻。"
        " 若 cap 过小，二分上界 min(hi_full,cap) 下常达不到 pool_lo，易触发 (A)。"
    )


if __name__ == "__main__":
    main()
