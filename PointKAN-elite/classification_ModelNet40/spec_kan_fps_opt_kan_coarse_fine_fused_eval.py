#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN 预测 FPS 阈值曲线（seg_filter） + 融合 coarse-fine KNN 的联合评估。

流程要点（默认与「前 10% FPS 采样点 → KAN → 全长阈值曲线 → 仅后缀步 seg_filter」一致）：
  1) 先用 exact FPS 得到前 ``prefix_len`` 个采样点索引；KAN 输入为这 ``prefix_len`` 个点的 3D 坐标
     展平后重采样到 ``in_dim``（如 32），输出 ``out_dim``（如 64）再上采样到长度 ``K``，得到整段平方阈值
     ``radii_ref_sq``（``--kan_input fps_points``；需匹配训练的权重）。
  2) 融合循环内：``t = 0..prefix_len-1`` 与标准 exact FPS 相同（种子 + argmax）；自 ``t = prefix_len`` 起对 farthest
     候选做 segment 阈值过滤（``--defer_seg_filter``，默认开启）。``--no_defer_seg_filter`` 则与旧版一致（``t>=1`` 即 seg_filter）。
  3) 每步 coarse-fine KNN 同 ``fps_with_knn_coarse_fine_fused``。

``--kan_input radii``：仍用半径序列前缀预测后缀（旧 checkpoint）；与「坐标前缀 → 全长曲线」不同。

评估指标：FPS IoU、KNN IoU、过滤统计、耗时（见运行日志与 JSON）。

结果 JSON 中 ``fps_seg_filter_candidate_report_*`` 含：
  - ``per_step_mean_candidate_count``：各 FPS 步上 seg_filter 后候选点数的 batch 平均（前缀 exact 步≈N）；
  - ``seg_filter_only_*``：仅 ``t >= seg_filter_start_t`` 的步上候选数均值/最值（反映阈值截断强度）。

可选 ``--plot_fps_candidate_curve``：按 hw_sim/plot_figs/accuracy.ipynb 风格（Arial、字号、网格、配色）
输出 ``save_dir/fps_candidate_count_vs_iteration.png``（600 dpi）与 ``.pdf``（透明，fonttype 42）；
``--plot_fps_candidate_curve_std`` 叠加跨 batch 标准差阴影。

KNN coarse：JSON 中 ``knn_coarse_candidate_report_*`` 含每步粗筛候选数、粗筛阈值循环迭代次数；
``--plot_knn_candidate_curve`` 输出 ``knn_coarse_vs_fps_iteration.png/.pdf``（双面板：候选数 + 迭代次数 vs FPS Iteration）；
``--plot_knn_annotate_stride``（默认 -1 自动）在**上图**沿曲线标注平均候选数；``--plot_knn_threshold_trace`` 配合 ``--knn_trace_*``（可用 ``--knn_trace_samples 0,1,2`` 同图多条轨迹）
输出阈值–候选数轨迹 ``knn_coarse_threshold_trace_example.png/.pdf``，并写入 JSON ``knn_threshold_trace_example``（含 ``per_sample_traces``）。

粗筛（``knn_coarse_fine_ball_mask``）：``--knn_coarse_schedule bisect``（默认）为围绕 ``init_threshold`` 的经典二分；
``mult_then_bisect`` 为先乘性扩缩形成 ``(h_lo,h_hi)`` 括号再括号内二分（见该函数文档）。
目标候选数 **大于** ``k_neighbors`` 且 **不超过** ``⌊k·(1+τ)⌋``，``1+τ`` 即 ``--min_candidates_ratio``。
候选不足时由 ``--knn_coarse_pool_shortage`` 控制：默认 ``keep_mid``（不扩半径）；可选 ``search_hi``（扩到 ``min(hi_full,cap)``）。
``--max_iter`` 为 KNN 粗筛主循环步数上限 ``max(1, max_iter)``（可提前结束）。
默认 ``--k_neighbors 24`` 即细筛 top-24。

可选 ``--sweep_fused_knn_surface``：扫三维网格 ``mult_alpha_clip``×``max_iter``×``init_threshold``。
默认输出两类曲面：（1）每个 clip 固定下 ``init_threshold``×``max_iter``；（2）每个 init 固定下 ``mult_alpha_clip``×``max_iter``。
可用 ``--no_sweep_clip_maxiter_surface`` 关闭第（2）类。绘图见 ``run_knn_coarse_fine_eval.plot_iou_surface``；参数见 ``--sweep_*``。

用法示例：
  python spec_kan_fps_opt_kan_coarse_fine_fused_eval.py \\
    --K 512 --segment_len 32 --k_neighbors 24 \\
    --kan_save_dir ./runs/fps_radii_kan_K512_pr0.10

  python spec_kan_fps_opt_kan_coarse_fine_fused_eval.py --kan_save_dir ""
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 路径：保证可导入同目录下的 spec_kan_fps_opt / spec_kan_fps / spec_kan_fps_opt_kan
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from spec_kan_fps_opt import (
    ModelNet40Wrapper,
    collate_points,
    compute_iou_batch,
    fps_adaptive_argmax,
)
from spec_kan_fps import (
    Config,
    _knn_coarse_hi_full_per_row,
    _knn_coarse_mult_bisect_alpha,
    fps_with_knn_coarse_fine_fused,
    knn_coarse_fine_ball_mask,
    knn_neighbor_set_iou,
)
from spec_kan_fps_opt_kan import (
    _aggregate_filter_stats,
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
    predict_full_radii_squared_from_fps_prefix_points,
)


# ---------------------------------------------------------------------------
# 参考标准：在指定 FPS 中心上做「全点云精确 topk」，用于与 coarse-fine KNN 算 IoU
# ---------------------------------------------------------------------------
@torch.no_grad()
def knn_exact_topk_at_centers(
    X: torch.Tensor,
    idx: torch.Tensor,
    k_neighbors: int,
    exclude_self: bool = False,
) -> torch.Tensor:
    """
    在给定 FPS 索引 idx [B,K] 的每个 center 上，对全点云做精确 topk（欧氏距离）。
    返回 knn_idx [B, K, k_actual]。
    """
    B, N, _ = X.shape
    K = idx.size(1)
    device = X.device
    dtype = X.dtype
    k_actual = min(k_neighbors, N)
    x2 = (X * X).sum(dim=-1)
    batch_arange = torch.arange(B, device=device)
    knn_idx = torch.zeros((B, K, k_actual), dtype=torch.long, device=device)

    # 与融合路径在同一批 center 上算「精确 KNN」，便于 knn_neighbor_set_iou
    for t in range(K):
        C = X[batch_arange, idx[:, t]]
        c2 = (C * C).sum(dim=-1, keepdim=True)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)
        d = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()
        if exclude_self:
            d = d.clone()
            d[batch_arange, idx[:, t]] = float("inf")
        _, knn_i = torch.topk(d, k=k_actual, dim=1, largest=False)
        knn_idx[:, t, :] = knn_i
    return knn_idx


# ---------------------------------------------------------------------------
# 核心融合核：FPS（seg_filter 用平方 min_dist + 参考半径曲线 tau）+ 每步 coarse-fine KNN
# ---------------------------------------------------------------------------
@torch.no_grad()
def fps_segfilter_knn_coarse_fine_fused(
    X: torch.Tensor,
    K: int,
    k_neighbors: int,
    radii_ref_sq: torch.Tensor,
    segment_len: int,
    init_threshold: float = 0.1,
    max_iter: int = 10,
    min_candidates_ratio: float = 1.2,
    seed: str = "first",
    exclude_self: bool = False,
    return_distances_matrix: bool = False,
    empty_fallback: str = "global",
    seg_filter_start_t: int = 1,
    distance_hi_cap: Optional[float] = None,
    coarse_schedule: str = "bisect",
    mult_alpha_clip: float = 0.25,
    knn_coarse_pool_shortage: str = "keep_mid",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, torch.Tensor],
]:
    """
    seg_filter FPS（平方距离 min_dist，与 fps_adaptive_argmax 一致）+ 每步 coarse-fine KNN（欧氏距离）。

    radii_ref_sq: (B, K)，平方距离阈值参考曲线（exact 或 KAN 整段预测）。
    seg_filter_start_t: 从该步开始使用 seg_filter 选 farthest；``t < seg_filter_start_t`` 且 ``t>=1`` 时为
    标准 ``argmax(min_dist_sq)``（与 exact FPS 一致）。``t==0`` 恒为种子点。默认 ``1`` 即与原实现一致（除首点外全程 seg_filter）。
    设为 ``prefix_len`` 时表示前 ``prefix_len`` 个 FPS 点与 exact 一致，之后再用 segment 阈值过滤。

    min_candidates_ratio：即 ``1+τ``，粗筛候选数上界 ``⌊k·(1+τ)⌋``（见 ``knn_coarse_fine_ball_mask``）。
    返回: idx, radii_fps_sq, knn_idx, knn_radii, knn_thresholds, distances_matrix, stats
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    K = min(K, N)
    k_actual = min(k_neighbors, N)

    if radii_ref_sq.shape != (B, K):
        raise ValueError(f"radii_ref_sq 应为 (B,K)={(B,K)}，当前 {tuple(radii_ref_sq.shape)}")

    x2 = (X * X).sum(dim=-1)
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii_fps_sq = torch.zeros((B, K), dtype=dtype, device=device)
    knn_idx = torch.zeros((B, K, k_actual), dtype=torch.long, device=device)
    knn_radii = torch.zeros((B, K), dtype=dtype, device=device)
    knn_thresholds = torch.zeros((B, K), dtype=dtype, device=device)
    # 每步 coarse 阶段：距离阈值内的点数（细筛 topk 的候选池大小）
    knn_coarse_candidate_count = torch.zeros((B, K), dtype=dtype, device=device)
    # 每步粗筛：单次二分迭代次数（见 knn_coarse_fine_ball_mask）
    knn_coarse_threshold_iters = torch.zeros((B, K), dtype=torch.int32, device=device)
    # candidate_ratio[t] = 本步 farthest 筛选后「通过 tau 的点数 / N」；exact 步固定为 1.0
    candidate_ratio = torch.ones((B, K), dtype=dtype, device=device)
    used_fallback = torch.zeros((B, K), dtype=torch.bool, device=device)

    if return_distances_matrix:
        distances_matrix = torch.empty((B, K, N), dtype=dtype, device=device)
    else:
        distances_matrix = torch.empty((0,), dtype=dtype, device=device)

    min_dist_sq = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)
    L = max(1, int(segment_len))
    # 步 t < s0：与 exact FPS 相同（种子 + argmax）；t >= s0：segment 阈值 seg_filter
    s0 = max(1, min(int(seg_filter_start_t), K))

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    def sqdist_to_center(C: torch.Tensor) -> torch.Tensor:
        c2 = (C * C).sum(dim=-1, keepdim=True)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)
        return (x2 + c2 - 2.0 * dot).clamp_min_(0.0)

    def select_farthest_seg_filter(md_sq: torch.Tensor, t: int) -> torch.Tensor:
        # 与 spec_kan_fps_opt.fps_adaptive_argmax 一致：用当前段末尾的参考半径作阈值（平方距离）
        j = t // L
        seg_end = min((j + 1) * L - 1, K - 1)
        tau = radii_ref_sq[:, seg_end]
        mask = md_sq >= tau.unsqueeze(1)
        cand_cnt = mask.sum(dim=1).to(dtype)
        candidate_ratio[:, t] = cand_cnt / float(N)  # 候选比例；实际点数 = ratio * N
        empty = ~mask.any(dim=1)
        if empty.any():
            used_fallback[:, t] = True
            if empty_fallback == "global":
                mask = mask.clone()
                mask[empty] = True
        cand = md_sq.clone()
        cand[~mask] = -float("inf")
        return torch.argmax(cand, dim=1)

    for t in range(K):
        if t == 0:
            idx[:, 0] = farthest
        elif t < s0:
            # 前缀步：不截断候选，等价 vanilla farthest（与 idx_exact 前缀一致）
            farthest = torch.argmax(min_dist_sq, dim=1)
            idx[:, t] = farthest
            candidate_ratio[:, t] = 1.0
        else:
            farthest = select_farthest_seg_filter(min_dist_sq, t)
            idx[:, t] = farthest

        C = X[batch_arange, farthest]
        d2 = sqdist_to_center(C)
        if return_distances_matrix:
            distances_matrix[:, t, :] = d2.sqrt()

        d = d2.sqrt()
        if exclude_self:
            d_knn = d.clone()
            d_knn[batch_arange, farthest] = float("inf")
        else:
            d_knn = d

        # coarse-fine KNN：对 h 二分，使 |d<h| 落在 (k, ⌊k·(1+τ)⌋]（见 knn_coarse_fine_ball_mask）
        mask, threshold, inner_total = knn_coarse_fine_ball_mask(
            d_knn,
            init_threshold,
            max_iter,
            k_actual,
            dtype,
            min_candidates_ratio=min_candidates_ratio,
            distance_hi_cap=distance_hi_cap,
            coarse_schedule=coarse_schedule,
            mult_alpha_clip=mult_alpha_clip,
            pool_shortage=knn_coarse_pool_shortage,
        )

        knn_coarse_candidate_count[:, t] = mask.sum(dim=1).to(dtype)
        knn_coarse_threshold_iters[:, t] = int(inner_total)

        knn_thresholds[:, t] = threshold
        masked_d = d_knn.masked_fill(~mask, float("inf"))
        knn_vals, knn_idx_t = torch.topk(masked_d, k=k_actual, dim=1, largest=False)
        knn_idx[:, t, :] = knn_idx_t
        knn_radii[:, t] = knn_vals[:, -1]

        torch.minimum(min_dist_sq, d2, out=min_dist_sq)
        min_dist_sq[batch_arange, farthest] = 0.0
        radii_fps_sq[:, t], _ = torch.max(min_dist_sq, dim=1)

    # 候选点数 = candidate_ratio * N（逐样本 B,K；汇总见 _aggregate_filter_stats）
    stats = {
        "candidate_ratio": candidate_ratio,
        "used_fallback": used_fallback,
        "knn_coarse_candidate_count": knn_coarse_candidate_count,
        "knn_coarse_threshold_iters": knn_coarse_threshold_iters,
    }
    return idx, radii_fps_sq, knn_idx, knn_radii, knn_thresholds, distances_matrix, stats


@torch.no_grad()
def fps_segfilter_knn_precompute_d_knn(
    X: torch.Tensor,
    K: int,
    radii_ref_sq: torch.Tensor,
    segment_len: int,
    seed: str = "first",
    exclude_self: bool = False,
    empty_fallback: str = "global",
    seg_filter_start_t: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    与 ``fps_segfilter_knn_coarse_fine_fused`` 相同的 seg_filter FPS 轨迹，但**不**做 coarse-fine KNN。

    在 ``radii_ref_sq`` 固定时，FPS 索引与 ``init_threshold`` / ``max_iter`` / ``mult_alpha_clip`` 无关，
    仅每步到各点的距离 ``d_knn`` 相同；扫参时可本函数算一次 ``idx`` 与 ``d_knn_stack``，
    再对每个超参网格只做 ``knn_coarse_fine_ball_mask`` + ``topk``。

    返回:
      - idx: (B, K_eff)
      - d_knn_stack: (B, K_eff, N)，与融合核每步传入 ``knn_coarse_fine_ball_mask`` 的 ``d_knn`` 一致
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    K = min(int(K), int(N))
    if radii_ref_sq.shape[0] != B or radii_ref_sq.shape[1] < K:
        raise ValueError(
            f"radii_ref_sq 至少为 (B, K_eff)=({B}, {K})，当前 {tuple(radii_ref_sq.shape)}"
        )
    radii_ref_sq = radii_ref_sq[:, :K].contiguous()

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    d_knn_stack = torch.empty((B, K, N), dtype=dtype, device=device)

    x2 = (X * X).sum(dim=-1)
    min_dist_sq = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)
    L = max(1, int(segment_len))
    s0 = max(1, min(int(seg_filter_start_t), K))

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    def sqdist_to_center(C: torch.Tensor) -> torch.Tensor:
        c2 = (C * C).sum(dim=-1, keepdim=True)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)
        return (x2 + c2 - 2.0 * dot).clamp_min_(0.0)

    def select_farthest_seg_filter(md_sq: torch.Tensor, t: int) -> torch.Tensor:
        j = t // L
        seg_end = min((j + 1) * L - 1, K - 1)
        tau = radii_ref_sq[:, seg_end]
        mask = md_sq >= tau.unsqueeze(1)
        empty = ~mask.any(dim=1)
        if empty.any():
            if empty_fallback == "global":
                mask = mask.clone()
                mask[empty] = True
            else:
                raise NotImplementedError(f"empty_fallback={empty_fallback}")
        cand = md_sq.clone()
        cand[~mask] = -float("inf")
        return torch.argmax(cand, dim=1)

    for t in range(K):
        if t == 0:
            idx[:, 0] = farthest
        elif t < s0:
            farthest = torch.argmax(min_dist_sq, dim=1)
            idx[:, t] = farthest
        else:
            farthest = select_farthest_seg_filter(min_dist_sq, t)
            idx[:, t] = farthest

        C = X[batch_arange, farthest]
        d2 = sqdist_to_center(C)
        d = d2.sqrt()
        if exclude_self:
            d_knn = d.clone()
            d_knn[batch_arange, farthest] = float("inf")
        else:
            d_knn = d
        d_knn_stack[:, t, :] = d_knn

        torch.minimum(min_dist_sq, d2, out=min_dist_sq)
        min_dist_sq[batch_arange, farthest] = 0.0

    return idx, d_knn_stack


@torch.no_grad()
def sweep_mean_knn_iou_from_d_stack(
    d_knn_stack: torch.Tensor,
    idx_f: torch.Tensor,
    X: torch.Tensor,
    k_neighbors: int,
    exclude_self: bool,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    dtype: torch.dtype,
    distance_hi_cap: Optional[float],
    coarse_schedule: str,
    mult_alpha_clip: float,
    knn_coarse_pool_shortage: str,
) -> float:
    """
    由 ``fps_segfilter_knn_precompute_d_knn`` 的 ``d_knn_stack`` 与融合路径 ``idx_f``，
    对单组粗筛超参复现 ``pack_fused_branch`` 中的 KNN 与 ``knn_neighbor_set_iou``，返回 batch 平均 IoU。
    """
    B, Ks, N = d_knn_stack.shape
    assert idx_f.shape == (B, Ks)
    k_actual = min(int(k_neighbors), int(N))
    device = X.device
    knn_ref_at_fused = knn_exact_topk_at_centers(
        X, idx_f, k_neighbors, exclude_self=exclude_self
    )
    # 将 K 个时间步拼成一批 [B*Ks, N]，一次粗筛 + 一次 topk，避免 512 次小 kernel 与
    # knn_coarse_fine_ball_mask 内 .item() 同步；sync_early_exit=False 跑满步数、全程 GPU。
    d_flat = d_knn_stack.reshape(B * Ks, N)
    mask, _threshold, _inner = knn_coarse_fine_ball_mask(
        d_flat,
        init_threshold,
        max_iter,
        k_actual,
        dtype,
        min_candidates_ratio=min_candidates_ratio,
        distance_hi_cap=distance_hi_cap,
        coarse_schedule=coarse_schedule,
        mult_alpha_clip=mult_alpha_clip,
        pool_shortage=knn_coarse_pool_shortage,
        sync_early_exit=False,
    )
    masked_d = d_flat.masked_fill(~mask, float("inf"))
    _vals, knn_idx_t = torch.topk(masked_d, k=k_actual, dim=1, largest=False)
    knn_idx = knn_idx_t.view(B, Ks, k_actual)
    knn_iou_vs_ref = knn_neighbor_set_iou(knn_idx, knn_ref_at_fused)
    return float(knn_iou_vs_ref.mean().item())


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def time_callable(fn, warmup: int, device: torch.device) -> float:
    """含 CUDA 同步的单次计时（用于对比各分支耗时）。"""
    for _ in range(warmup):
        fn()
    _sync_cuda()
    t0 = time.perf_counter()
    fn()
    _sync_cuda()
    return time.perf_counter() - t0


def parse_args():
    # 参数说明见模块顶部 docstring；此处仅注册 CLI
    p = argparse.ArgumentParser(
        description="KAN radii seg_filter + coarse-fine KNN 融合：FPS/KNN IoU 与耗时"
    )
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--k_neighbors", type=int, default=24)
    p.add_argument("--init_threshold", type=float, default=0.1)
    p.add_argument(
        "--max_iter",
        type=int,
        default=10,
        help="KNN 粗筛主循环最大步数=max(1,max_iter)（可提前结束）",
    )
    p.add_argument(
        "--min_candidates_ratio",
        type=float,
        default=1.2,
        help="粗筛 1+τ：候选数目标上界 ⌊k·ratio⌋，下界为 k+1（不可行时退化为 k）",
    )
    p.add_argument(
        "--knn_coarse_distance_hi_cap",
        type=float,
        default=0.0,
        help="Coarse bisection hi = min(hi_full, cap); 0 = use hi_full only. Prefer JSON from calibration script.",
    )
    p.add_argument(
        "--knn_coarse_hi_json",
        type=str,
        default="",
        help="Path to JSON with distance_hi_cap (from calibrate_knn_coarse_distance_hi.py). Overrides --knn_coarse_distance_hi_cap.",
    )
    p.add_argument(
        "--knn_coarse_schedule",
        type=str,
        default="bisect",
        choices=("bisect", "mult_then_bisect"),
        help=(
            "粗筛阈值搜索：bisect=全程二分；mult_then_bisect=围绕 init_threshold 乘性扩缩形成括号后再二分"
        ),
    )
    p.add_argument(
        "--knn_coarse_mult_alpha_clip",
        type=float,
        default=0.25,
        help=(
            "mult_then_bisect 下乘性步长 α=min(τ, clip)，τ=ratio-1；≤0 表示不裁剪（α=τ）"
        ),
    )
    p.add_argument(
        "--knn_coarse_pool_shortage",
        type=str,
        default="keep_mid",
        choices=("keep_mid", "search_hi"),
        help=(
            "粗筛最终 h 后若 cnt<pool_lo：keep_mid=不扩半径（默认）；"
            "search_hi=扩到搜索上界 hi=min(hi_full,cap)（无 cap 时为 hi_full）。"
            "已移除 hi_full 档（旧版会突破 cap）。"
        ),
    )
    p.add_argument("--exclude_self", action="store_true", help="KNN 排除 center 自身")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=str, default="first")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--kan_save_dir",
        type=str,
        default="",
        help="KAN 权重目录（含 best.pt）；空字符串则跳过 KAN 分支",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="./runs/fps_kan_segfilter_coarsefine_fused",
    )
    p.add_argument("--warmup", type=int, default=2, help="GPU 计时预热次数")
    p.add_argument(
        "--time_batches",
        type=int,
        default=1,
        help="对前若干个 batch 做耗时统计（取平均秒/batch）",
    )
    p.add_argument(
        "--skip_cf_baseline",
        action="store_true",
        help="不跑「仅 coarse-fine KNN、无 seg_filter」的 baseline",
    )
    p.add_argument(
        "--prefix_ratio",
        type=float,
        default=0.10,
        help="前缀 FPS 点数比例：决定 kan_prefix_len 与（默认下）seg_filter 起始步",
    )
    p.add_argument(
        "--no_defer_seg_filter",
        dest="defer_seg_filter",
        action="store_false",
        help="恢复旧行为：除种子外每一步都用 seg_filter（seg_filter_start_t=1）",
    )
    p.add_argument(
        "--kan_input",
        type=str,
        default="fps_points",
        choices=("fps_points", "radii"),
        help="fps_points=前若干 FPS 点坐标→KAN→全长 K 阈值；radii=半径序列前缀（旧训练）",
    )
    p.set_defaults(defer_seg_filter=True)
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="仅处理前 N 个 batch 后退出（快速冒烟）；0 表示跑完整 DataLoader",
    )
    p.add_argument(
        "--plot_fps_candidate_curve",
        action="store_true",
        help="将「每步平均 farthest 候选点数」随 FPS 迭代 t 绘图保存到 save_dir",
    )
    p.add_argument(
        "--plot_fps_candidate_curve_std",
        action="store_true",
        help="与上图一并绘制跨 batch 标准差阴影带（需 matplotlib）",
    )
    p.add_argument(
        "--plot_knn_candidate_curve",
        action="store_true",
        help="绘制 KNN coarse：每步候选数 + 阈值迭代次数 vs FPS 迭代 t（双面板 PNG/PDF）",
    )
    p.add_argument(
        "--plot_knn_candidate_curve_std",
        action="store_true",
        help="KNN 曲线图叠加跨 batch 标准差阴影",
    )
    p.add_argument(
        "--plot_knn_annotate_stride",
        type=int,
        default=-1,
        help="KNN 上图每步平均候选数文字标注：-1=自动 max(1,K//25)，0=不标注，>0=每隔 N 个 FPS 步",
    )
    p.add_argument(
        "--plot_knn_threshold_trace",
        action="store_true",
        help="在指定 batch/样本/FPS 步上绘制粗筛阈值 h 与候选数变化（融合 exact radii_ref 路径下的中心点）",
    )
    p.add_argument(
        "--plot_knn_threshold_trace_distribution",
        dest="plot_knn_threshold_trace_distribution",
        action="store_true",
        help=(
            "在轨迹条数≥3 时输出跨样本 mean±std 分布图（*_distribution.png/pdf）；"
            "默认开启，可不传；若曾关闭则需再传此开关以开启。"
        ),
    )
    p.add_argument(
        "--no_plot_knn_threshold_trace_distribution",
        "--no-plot-knn-threshold-trace-distribution",
        dest="plot_knn_threshold_trace_distribution",
        action="store_false",
        help="关闭分布图输出（适用于 Python 3.8 等无 BooleanOptionalAction 的环境）。",
    )
    p.add_argument(
        "--knn_trace_batch",
        type=int,
        default=0,
        help="轨迹示例取自第几个 batch（0 起，需 max_batches 足够大或跑全量）",
    )
    p.add_argument(
        "--knn_trace_sample",
        type=int,
        default=0,
        help="batch 内单个样本索引；若设置了 --knn_trace_samples 则忽略此项",
    )
    p.add_argument(
        "--knn_trace_samples",
        type=str,
        default="",
        help="逗号分隔的 batch 内样本索引，例如 0,1,2,3；同一张图画多条轨迹。空则仅用 --knn_trace_sample",
    )
    p.add_argument(
        "--knn_trace_max_samples",
        type=int,
        default=0,
        help=(
            "目标轨迹条数（如 128）：从 --knn_trace_batch 起按 batch 顺序取样本 0..B-1 累积，"
            "跨多个 batch 直到凑满或 DataLoader 结束；>0 时忽略 --knn_trace_samples。"
            "需保证 max_batches 足够（例 B=8 时需约 16 个 batch）。"
        ),
    )
    p.add_argument(
        "--knn_trace_t",
        type=int,
        default=-1,
        help="FPS 迭代步 t；-1 表示 min(seg_filter_start_t+10, K-1)",
    )
    p.add_argument(
        "--sweep_fused_knn_surface",
        action="store_true",
        help=(
            "扫参三维网格：对每个 mult_alpha_clip 输出 init_threshold×max_iter 曲面；"
            "默认另对每个 init_threshold 输出 mult_alpha_clip×max_iter 曲面（可用 --no_sweep_clip_maxiter_surface 关闭）。"
            "clip 仅在 mult_then_bisect 下有效；bisect 时随 clip 变化通常重合。"
        ),
    )
    p.add_argument(
        "--no_sweep_clip_maxiter_surface",
        dest="sweep_clip_maxiter_surface",
        action="store_false",
        help="不输出「固定 init、横轴 mult_alpha_clip、纵轴 max_iter」的 IoU 曲面",
    )
    p.add_argument(
        "--sweep_init_thresholds",
        type=str,
        default="0.05,0.1,0.15,0.2,0.25",
        help="逗号分隔 init_threshold（曲面横轴）",
    )
    p.add_argument(
        "--sweep_max_iters",
        type=str,
        default="4,8,16,24,32,48",
        help="逗号分隔 max_iter（曲面纵轴）",
    )
    p.add_argument(
        "--sweep_mult_alpha_clips",
        type=str,
        default="0.1,0.25,0.5",
        help="逗号分隔 knn_coarse_mult_alpha_clip；每个 clip 单独一张 PNG",
    )
    p.add_argument(
        "--sweep_surface_out_prefix",
        type=str,
        default="./runs/fused_knn_sweep",
        help="扫参输出 PNG / JSON 的路径前缀",
    )
    p.add_argument(
        "--sweep_max_batches",
        type=int,
        default=16,
        help="每个网格点用多少个 batch 求平均 IoU；0=整份 DataLoader（很慢）。想加速可改小（如 4～8）",
    )
    p.add_argument(
        "--sweep_exact_only",
        action="store_true",
        help=(
            "扫参时只算 fused_exact_radii 的 KNN IoU，不加载 KAN、不画 kan 曲面；"
            "与粗筛超参无关的 KAN 对比可关掉，通常接近减半耗时"
        ),
    )
    p.add_argument(
        "--no_sweep_d_knn_precompute",
        action="store_true",
        help=(
            "关闭扫参的 d_knn 预计算加速（默认在 --skip_cf_baseline 下开启）："
            "每 batch 存 [B,K,N] 距离、网格上只做粗筛+topk，通常比逐格全融合快很多；"
            "显存紧张或需与旧实现完全逐格对齐时再加本项"
        ),
    )
    p.set_defaults(
        plot_knn_threshold_trace_distribution=True,
        sweep_clip_maxiter_surface=True,
    )
    return p.parse_args()


def parse_knn_trace_sample_indices(args: argparse.Namespace, batch_size: int) -> List[int]:
    """Unique in-order sample indices in [0, B-1]."""
    raw = str(getattr(args, "knn_trace_samples", "")).strip()
    if raw:
        ids = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    else:
        ids = [int(getattr(args, "knn_trace_sample", 0))]
    Bm = max(0, int(batch_size) - 1)
    out: List[int] = []
    seen = set()
    for si in ids:
        si = int(min(max(0, si), Bm))
        if si not in seen:
            seen.add(si)
            out.append(si)
    return out if out else [0]


def resolve_knn_trace_target_count(args: argparse.Namespace) -> int:
    """0 = use per-batch index list; >0 = accumulate up to this many (cross-batch)."""
    n = int(getattr(args, "knn_trace_max_samples", 0) or 0)
    return max(0, n)


def resolve_knn_trace_payload_done(payload: Optional[Dict], target: int) -> bool:
    if target <= 0 or payload is None:
        return False
    return len(payload.get("per_sample_traces") or []) >= target


def plot_knn_coarse_threshold_trace_distribution(
    save_path_base: str,
    per_sample_traces: List[Dict],
    pool_lo: int,
    pool_hi: int,
    meta: Dict,
    font_size: int = 16,
) -> Optional[Dict[str, str]]:
    """
    多样本：按 probe 索引对齐，画候选数 ``|d<h|`` 的跨样本 mean ± std（短轨迹为 NaN 不参与该点）。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] 未安装 matplotlib，跳过 KNN 阈值分布图。")
        return None

    rows: List[List[int]] = []
    max_len = 0
    for item in per_sample_traces:
        pairs = item.get("trace_pairs_h_count") or []
        cseq = [int(p[1]) for p in pairs]
        if not cseq:
            continue
        rows.append(cseq)
        max_len = max(max_len, len(cseq))
    if max_len == 0 or not rows:
        return None

    M = np.full((len(rows), max_len), np.nan, dtype=np.float64)
    for i, cseq in enumerate(rows):
        M[i, : len(cseq)] = np.array(cseq, dtype=np.float64)
    mean_c = np.nanmean(M, axis=0)
    std_c = np.nanstd(M, axis=0)
    std_c = np.nan_to_num(std_c, nan=0.0)
    x = np.arange(max_len, dtype=np.float64)

    _configure_matplotlib_paper_style(font_size=font_size)
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    lo = np.maximum(0.0, mean_c - std_c)
    hi = mean_c + std_c
    ax.fill_between(x, lo, hi, alpha=0.35, color="#2c5282", linewidth=0)
    ax.plot(x, mean_c, color="#1a365d", linewidth=2.2)
    ax.axhline(
        float(pool_lo),
        color="#555555",
        linestyle="--",
        linewidth=1.0,
    )
    ax.axhline(
        float(pool_hi),
        color="#2d6a4f",
        linestyle=":",
        linewidth=1.2,
    )
    ax.set_xlabel("Trace index (bisection probes)")
    ax.set_ylabel("Candidate count |{j: d_j < h}|")
    ax.grid(axis="y", alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    n = len(rows)
    sub = (
        f"n={n} clouds×samples  batch≥{meta.get('knn_trace_batch_start')}  "
        f"FPS_t={meta.get('fps_t')}"
    )
    fig.suptitle(
        f"KNN coarse candidate count vs probe (mean ± std)\n{sub}",
        fontsize=font_size - 1,
    )
    fig.text(
        0.5,
        0.02,
        "Per-trace series: JSON knn_threshold_trace_example.per_sample_traces",
        ha="center",
        va="bottom",
        fontsize=max(7, font_size - 5),
        color="#666666",
    )
    plt.tight_layout(pad=0.35, rect=[0, 0.07, 1, 0.90])

    png_path = f"{save_path_base}_distribution.png"
    pdf_path = f"{save_path_base}_distribution.pdf"
    fig.savefig(
        pdf_path,
        transparent=True,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.02,
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        format="png",
        pad_inches=0.02,
    )
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def resolve_knn_coarse_distance_hi_cap(args: argparse.Namespace) -> Optional[float]:
    """CLI / JSON -> optional cap for knn_coarse_fine_ball_mask (None = uncapped beyond hi_full)."""
    jp = str(getattr(args, "knn_coarse_hi_json", "")).strip()
    if jp:
        path = jp if os.path.isabs(jp) else os.path.normpath(jp)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "distance_hi_cap" not in data:
            raise KeyError(f"{path} must contain key 'distance_hi_cap'")
        return float(data["distance_hi_cap"])
    v = float(getattr(args, "knn_coarse_distance_hi_cap", 0.0))
    return v if v > 0.0 else None


def _build_fps_seg_filter_candidate_report(
    rows: List[Dict],
    seg_filter_start_t: int,
    num_points: int,
) -> Optional[Dict]:
    """
    汇总多 batch 的 ``per_step_candidates_mean``（即每步 FPS farthest 在 seg_filter 后**候选点个数**，
    前缀 exact 步为 N），写入 JSON；并单独统计 **仅 seg_filter 步**（t >= seg_filter_start_t）上的平均/最值。
    """
    if not rows:
        return None
    key = "per_step_candidates_mean"
    if key not in rows[0]:
        return None
    pcs = np.stack([np.array(r[key], dtype=np.float64) for r in rows], axis=0)
    mean_per_step = pcs.mean(axis=0)
    std_across_batches = pcs.std(axis=0)
    s0 = int(seg_filter_start_t)
    n = int(mean_per_step.shape[0])
    rep: Dict = {
        "N_cloud": int(num_points),
        "seg_filter_start_step_t": s0,
        "num_fps_steps_K": n,
        "per_step_mean_candidate_count": mean_per_step.tolist(),
        "per_step_std_candidate_count_across_batches": std_across_batches.tolist(),
        "description": (
            "Per-step seg_filter candidate count: points with min_dist_sq >= segment tau; "
            "when defer is on, steps t < s0 are exact FPS and count is about N_cloud."
        ),
    }
    if s0 < n:
        seg_slice = pcs[:, s0:]
        rep["seg_filter_only_num_steps"] = n - s0
        rep["seg_filter_only_mean_candidate_count"] = float(seg_slice.mean())
        rep["seg_filter_only_std_over_batches_and_steps"] = float(seg_slice.std())
        rep["seg_filter_only_min_candidate_count"] = float(seg_slice.min())
        rep["seg_filter_only_max_candidate_count"] = float(seg_slice.max())
    return rep


def _knn_coarse_stats_one_batch(
    count_bk: torch.Tensor,
    iter_bk: torch.Tensor,
    n_cloud: int,
    max_iter_cap: int,
) -> Dict:
    """单 batch：KNN coarse 每步候选数、阈值循环迭代次数（batch 内 mean/std）。"""
    k_len = int(count_bk.shape[1])
    itf = iter_bk.float()
    std_c = count_bk.std(dim=0)
    std_i = itf.std(dim=0)
    std_c = torch.nan_to_num(std_c, nan=0.0)
    std_i = torch.nan_to_num(std_i, nan=0.0)
    return {
        "per_step_mean_knn_coarse_candidate_count": count_bk.mean(dim=0).cpu().numpy().tolist(),
        "per_step_std_knn_coarse_candidate_count_within_batch": std_c.cpu().numpy().tolist(),
        "per_step_mean_knn_coarse_threshold_iterations": itf.mean(dim=0).cpu().numpy().tolist(),
        "per_step_std_knn_coarse_threshold_iters_within_batch": std_i.cpu().numpy().tolist(),
        "N_cloud": int(n_cloud),
        "num_fps_steps_K": k_len,
        "knn_coarse_max_iter_cap": int(max_iter_cap),
    }


def _build_knn_coarse_candidate_report(rows: List[Dict]) -> Optional[Dict]:
    """多 batch：对 ``_knn_coarse_stats_one_batch`` 结果按步聚合（跨 batch mean/std）。"""
    if not rows:
        return None
    kc = "per_step_mean_knn_coarse_candidate_count"
    ki = "per_step_mean_knn_coarse_threshold_iterations"
    if kc not in rows[0] or ki not in rows[0]:
        return None
    arr_c = np.stack([np.array(r[kc], dtype=np.float64) for r in rows], axis=0)
    arr_i = np.stack([np.array(r[ki], dtype=np.float64) for r in rows], axis=0)
    mean_per_t = arr_c.mean(axis=0)
    mean_per_t_i = arr_i.mean(axis=0)
    return {
        "num_batches_aggregated": len(rows),
        "N_cloud": rows[0].get("N_cloud"),
        "num_fps_steps_K": int(arr_c.shape[1]),
        "knn_coarse_max_iter_cap": rows[0].get("knn_coarse_max_iter_cap"),
        "per_step_mean_knn_coarse_candidate_count": mean_per_t.tolist(),
        "per_step_std_knn_coarse_candidate_count_across_batches": arr_c.std(axis=0).tolist(),
        "per_step_mean_knn_coarse_threshold_iterations": mean_per_t_i.tolist(),
        "per_step_std_knn_coarse_threshold_iterations_across_batches": arr_i.std(axis=0).tolist(),
        "mean_knn_coarse_candidate_count_all_steps": float(mean_per_t.mean()),
        "mean_knn_coarse_threshold_iterations_all_steps": float(mean_per_t_i.mean()),
        "min_per_step_mean_knn_coarse_threshold_iterations": float(mean_per_t_i.min()),
        "max_per_step_mean_knn_coarse_threshold_iterations": float(mean_per_t_i.max()),
        "min_per_step_mean_knn_coarse_candidate_count": float(mean_per_t.min()),
        "max_per_step_mean_knn_coarse_candidate_count": float(mean_per_t.max()),
        "description": (
            "Per FPS step t: points in coarse ball {j: d_j < h} before fine top-k; "
            "bisection probes (batch-synchronized), target count in (k, floor(k*(1+tau))] when feasible."
        ),
    }


def knn_coarse_threshold_trace_for_center(
    X_bn3: torch.Tensor,
    batch_idx: int,
    center_idx: int,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    k_neighbors: int,
    exclude_self: bool,
    distance_hi_cap: Optional[float] = None,
    coarse_schedule: str = "bisect",
    mult_alpha_clip: float = 0.25,
    pool_shortage: str = "keep_mid",
) -> Tuple[List[Tuple[float, int]], int, int]:
    """
    单样本、单 FPS 步：与 ``knn_coarse_fine_ball_mask`` 同调度，记录 ``(h, |d<h|)`` 序列。
    返回 ``(trace, pool_lo, pool_hi)``，供绘图与 JSON 标注目标候选数区间。
    """
    ps = str(pool_shortage).strip().lower()
    if ps == "hi_full":
        raise ValueError(
            "pool_shortage='hi_full' 已移除，请用 'keep_mid' 或 'search_hi'（与 knn_coarse_fine_ball_mask 一致）。"
        )
    if ps not in ("search_hi", "keep_mid"):
        raise ValueError(
            f"pool_shortage 须为 'search_hi' | 'keep_mid'，当前 {pool_shortage!r}"
        )
    X = X_bn3[batch_idx].detach()
    N = int(X.shape[0])
    c = X[int(center_idx)]
    d = torch.norm(X - c, dim=-1)
    if exclude_self:
        d = d.clone()
        d[int(center_idx)] = float("inf")
    k = min(int(k_neighbors), N)
    ratio = float(min_candidates_ratio)
    pool_hi = max(int(math.floor(k * ratio)), k)
    pool_lo = k + 1
    if pool_lo > pool_hi:
        pool_lo = k

    steps = max(1, int(max_iter))
    d_max = float(d.max().item())
    if d_max < 1e-12:
        d_max = 1e-8

    d_max_t = torch.tensor([d_max], dtype=torch.float32)
    hi_full = float(_knn_coarse_hi_full_per_row(d_max_t, torch.float32).item())
    hi = hi_full
    if distance_hi_cap is not None and float(distance_hi_cap) > 0.0:
        hi = min(hi_full, float(distance_hi_cap))

    trace: List[Tuple[float, int]] = []
    lo = 0.0
    eps = max(hi_full, 1e-8) * (math.sqrt(float(torch.finfo(torch.float32).eps)))

    init_h = float(init_threshold)
    init_h = min(max(init_h, lo + eps), hi - eps)

    def _count_at(hp: float) -> int:
        return int((d < hp).sum().item())

    if coarse_schedule == "bisect":
        b_lo, b_hi = lo, hi
        for it in range(steps):
            mid = init_h if it == 0 else (b_lo + b_hi) / 2.0
            cnt = _count_at(mid)
            trace.append((mid, cnt))
            ok = pool_lo <= cnt <= pool_hi
            too_few = cnt < pool_lo
            too_many = cnt > pool_hi
            if ok:
                b_lo = mid
                b_hi = mid
                break
            if too_few:
                b_lo = mid
            if too_many:
                b_hi = mid

        h = (b_lo + b_hi) / 2.0
        cnt = _count_at(h)
        if cnt < pool_lo:
            if ps == "search_hi":
                h = hi
                cnt = _count_at(h)
                trace.append((h, cnt))
            else:
                last = trace[-1]
                if abs(last[0] - h) > 1e-9 * max(abs(h), 1.0) or last[1] != cnt:
                    trace.append((h, cnt))
        else:
            last = trace[-1]
            if abs(last[0] - h) > 1e-9 * max(abs(h), 1.0) or last[1] != cnt:
                trace.append((h, cnt))

        return trace, pool_lo, pool_hi

    if coarse_schedule != "mult_then_bisect":
        raise ValueError(
            f"coarse_schedule 须为 'bisect' 或 'mult_then_bisect'，当前 {coarse_schedule!r}"
        )

    alpha = _knn_coarse_mult_bisect_alpha(min_candidates_ratio, float(mult_alpha_clip))
    one_pa = 1.0 + alpha

    done_ok = False
    use_bisect = False
    h_cur = init_h
    bl = 0.0
    bh = float("inf")
    hl = False
    hh = False
    lo_b = 0.0
    hi_b = hi
    ans_h = 0.0

    for _it in range(steps):
        if done_ok:
            break
        if not use_bisect:
            hp = h_cur
        else:
            hp = (lo_b + hi_b) * 0.5
        cnt = _count_at(hp)
        trace.append((hp, cnt))
        ok = pool_lo <= cnt <= pool_hi
        if ok:
            done_ok = True
            ans_h = hp
            break

        if not use_bisect:
            tf = cnt < pool_lo
            tm = cnt > pool_hi
            if tf:
                bl = hp if not hl else max(bl, hp)
                hl = True
            if tm:
                bh = hp if not hh else min(bh, hp)
                hh = True
            ready = hl and hh and bl < bh
            if ready:
                use_bisect = True
                lo_b, hi_b = bl, bh
            else:
                if tf:
                    h_cur = h_cur * one_pa
                elif tm:
                    h_cur = h_cur / one_pa
                h_cur = min(max(h_cur, lo + eps), hi - eps)
        else:
            if cnt < pool_lo:
                lo_b = hp
            elif cnt > pool_hi:
                hi_b = hp

    if not use_bisect and not done_ok:
        lo_b = lo + eps
        hi_b = hi

    h = ans_h if done_ok else (lo_b + hi_b) * 0.5
    cnt = _count_at(h)
    if cnt < pool_lo:
        if ps == "search_hi":
            h = hi
            cnt = _count_at(h)
            trace.append((h, cnt))
        else:
            last = trace[-1]
            if abs(last[0] - h) > 1e-9 * max(abs(h), 1.0) or last[1] != cnt:
                trace.append((h, cnt))
    else:
        last = trace[-1]
        if abs(last[0] - h) > 1e-9 * max(abs(h), 1.0) or last[1] != cnt:
            trace.append((h, cnt))

    return trace, pool_lo, pool_hi


def _configure_matplotlib_paper_style(font_size: int = 18) -> None:
    """
    与 hw_sim/plot_figs/accuracy.ipynb 一致：注册 Arial（若存在）、全局 rcParams、PDF 可编辑文字。
    无 Arial 时回退 DejaVu Sans，不抛错。
    """
    from matplotlib import font_manager

    candidate_fonts = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((fp for fp in candidate_fonts if os.path.exists(fp)), None)
    font_family = "DejaVu Sans"
    if font_path:
        try:
            font_manager.fontManager.addfont(font_path)
            font_family = font_manager.FontProperties(fname=font_path).get_name()
        except Exception:
            pass

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "hatch.linewidth": 2.5,
        }
    )


def plot_fps_candidate_count_vs_iteration(
    save_path_base: str,
    rep_exact: Optional[Dict],
    rep_kan: Optional[Dict],
    seg_filter_start_t: int,
    num_points: int,
    with_std_band: bool = False,
    font_size: int = 18,
) -> Optional[Dict[str, str]]:
    """
    绘制完整 K 步：纵轴为每步跨 batch 平均候选点数。
    水平均值线仅对 ``t >= seg_filter_start_t``（真正 seg_filter 的步）取平均，不含前缀 exact 步。
    样式对齐 accuracy.ipynb；同时导出 PNG（600 dpi）与透明 PDF。

    参数:
        save_path_base: 不含扩展名，如 ``.../fps_candidate_count_vs_iteration``
    返回:
        成功时 ``{"png": ..., "pdf": ...}``，否则 ``None``。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] 未安装 matplotlib，跳过绘图。")
        return None

    if rep_exact is None and rep_kan is None:
        return None

    _configure_matplotlib_paper_style(font_size=font_size)

    # 与 accuracy.ipynb 柱状配色同系的线条色（白底上可读）
    color_exact = "#00b0f0"
    color_kan = "#cc8400"
    color_ref = "#555555"

    fig, ax = plt.subplots(figsize=(9.2, 3.4))

    def _one_series_line(rep: Dict, label: str, color: str) -> None:
        y = np.array(rep["per_step_mean_candidate_count"], dtype=np.float64)
        t = np.arange(len(y))
        ax.plot(t, y, color=color, linewidth=2.2, label=label)
        if with_std_band and "per_step_std_candidate_count_across_batches" in rep:
            std = np.array(
                rep["per_step_std_candidate_count_across_batches"], dtype=np.float64
            )
            if std.shape == y.shape:
                ax.fill_between(
                    t,
                    np.maximum(0, y - std),
                    y + std,
                    color=color,
                    alpha=0.22,
                )

    def _one_series_kan_markers(rep: Dict, label: str, color: str) -> None:
        y = np.array(rep["per_step_mean_candidate_count"], dtype=np.float64)
        t = np.arange(len(y))
        ax.plot(
            t,
            y,
            linestyle="none",
            marker="^",
            markersize=4.0,
            markeredgewidth=0.8,
            markerfacecolor=color,
            markeredgecolor=color,
            label=label,
        )
        if with_std_band and "per_step_std_candidate_count_across_batches" in rep:
            std = np.array(
                rep["per_step_std_candidate_count_across_batches"], dtype=np.float64
            )
            if std.shape == y.shape:
                ax.fill_between(
                    t,
                    np.maximum(0, y - std),
                    y + std,
                    color=color,
                    alpha=0.22,
                )

    if rep_exact is not None:
        _one_series_line(rep_exact, "radii_ref = exact FPS", color_exact)
    if rep_kan is not None:
        _one_series_kan_markers(rep_kan, "radii_ref = KAN", color_kan)

    # 水平均值：仅对 t >= seg_filter_start_t（真正 seg_filter 的步），不含前缀 exact「暖步」
    s0 = int(seg_filter_start_t)
    dash_mean = (0, (6, 3))

    def _mean_candidates_seg_only(y: np.ndarray) -> float:
        if y.size == 0:
            return float("nan")
        if s0 < len(y):
            return float(np.mean(y[s0:]))
        return float(np.mean(y))

    if rep_exact is not None:
        y_e = np.array(rep_exact["per_step_mean_candidate_count"], dtype=np.float64)
        mu_e = _mean_candidates_seg_only(y_e)
        t_last_e = max(0, len(y_e) - 1)
        if not np.isnan(mu_e) and s0 <= t_last_e:
            ax.plot(
                [s0, t_last_e],
                [mu_e, mu_e],
                linestyle=dash_mean,
                color=color_exact,
                linewidth=1.5,
                alpha=0.9,
                label=f"Mean (exact, t≥{s0}) = {mu_e:.1f}",
            )
    if rep_kan is not None:
        y_k = np.array(rep_kan["per_step_mean_candidate_count"], dtype=np.float64)
        mu_k = _mean_candidates_seg_only(y_k)
        t_last_k = max(0, len(y_k) - 1)
        if not np.isnan(mu_k) and s0 <= t_last_k:
            t_seg = np.arange(s0, len(y_k), dtype=np.float64)
            markevery = max(1, len(t_seg) // 25) if len(t_seg) > 1 else 1
            ax.plot(
                t_seg,
                np.full_like(t_seg, mu_k),
                linestyle=dash_mean,
                color=color_kan,
                linewidth=1.5,
                alpha=0.9,
                marker="^",
                markersize=5,
                markevery=markevery,
                label=f"Mean (KAN, t≥{s0}) = {mu_k:.1f}",
            )

    ax.axhline(
        num_points,
        color=color_ref,
        linestyle=":",
        linewidth=1.2,
        label=f"N = {num_points}",
    )
    ax.axvline(
        seg_filter_start_t,
        color=color_ref,
        linestyle="--",
        linewidth=1.2,
        label=f"seg_filter start t = {seg_filter_start_t}",
    )

    ax.set_xlabel("FPS Iteration")
    ax.set_ylabel("Candidate Count")
    ax.grid(axis="y", alpha=0.45, linestyle="--")
    ax.set_axisbelow(True)
    ax.margins(x=0.02)
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
    )

    plt.tight_layout(pad=0.25)
    fig.subplots_adjust(top=0.98, bottom=0.22, left=0.10, right=0.98)

    png_path = f"{save_path_base}.png"
    pdf_path = f"{save_path_base}.pdf"
    fig.savefig(
        pdf_path,
        transparent=True,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.02,
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        format="png",
        pad_inches=0.02,
    )
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_knn_coarse_vs_fps_iteration(
    save_path_base: str,
    rep_exact: Optional[Dict],
    rep_kan: Optional[Dict],
    num_points: int,
    with_std_band: bool = False,
    font_size: int = 18,
    annotate_candidate_stride: int = -1,
) -> Optional[Dict[str, str]]:
    """
    双面板：上 — 每步 KNN coarse 候选点数 vs FPS 迭代 t；下 — 粗筛阈值放大循环迭代次数 vs t。
    样式与 ``plot_fps_candidate_count_vs_iteration`` / accuracy 笔记本一致。

    annotate_candidate_stride:
        ``0`` 不标注数值；``<0`` 自动步长 ``max(1, K//25)``；``>0`` 每隔该步在曲线上标出候选数。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] 未安装 matplotlib，跳过 KNN 曲线。")
        return None

    if rep_exact is None and rep_kan is None:
        return None

    _configure_matplotlib_paper_style(font_size=font_size)
    color_exact = "#00b0f0"
    color_kan = "#cc8400"
    color_ref = "#555555"

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(9.2, 5.8), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )

    def _plot_candidates(ax, key_mean: str, key_std: str) -> None:
        def _one(rep: Dict, label: str, color: str) -> None:
            y = np.array(rep[key_mean], dtype=np.float64)
            t = np.arange(len(y))
            ax.plot(t, y, color=color, linewidth=2.0, label=label)
            if with_std_band and key_std in rep:
                std = np.array(rep[key_std], dtype=np.float64)
                if std.shape == y.shape:
                    ax.fill_between(
                        t, np.maximum(0, y - std), y + std, color=color, alpha=0.2
                    )

        if rep_exact is not None:
            _one(
                rep_exact,
                "radii_ref = exact FPS",
                color_exact,
            )
        if rep_kan is not None:
            _one(
                rep_kan,
                "radii_ref = KAN",
                color_kan,
            )
        ax.axhline(
            num_points,
            color=color_ref,
            linestyle=":",
            linewidth=1.0,
            label=f"N = {num_points}",
        )

    _plot_candidates(
        ax0,
        "per_step_mean_knn_coarse_candidate_count",
        "per_step_std_knn_coarse_candidate_count_across_batches",
    )
    ax0.set_ylabel("Coarse KNN\ncandidates")
    ax0.grid(axis="y", alpha=0.45, linestyle="--")
    ax0.set_axisbelow(True)

    # 上图：按步长标注跨 batch 平均候选数（优先 exact 曲线，与图例一致）
    if annotate_candidate_stride != 0:
        rep_ann = rep_exact if rep_exact is not None else rep_kan
        if rep_ann is not None:
            y_ann = np.array(
                rep_ann["per_step_mean_knn_coarse_candidate_count"], dtype=np.float64
            )
            n_t = len(y_ann)
            st = max(1, n_t // 25) if annotate_candidate_stride < 0 else int(
                annotate_candidate_stride
            )
            ann_fs = max(5, int(font_size) - 7)
            ann_color = color_exact if rep_exact is not None else color_kan
            for ti in range(0, n_t, st):
                ax0.annotate(
                    f"{y_ann[ti]:.0f}",
                    (ti, y_ann[ti]),
                    textcoords="offset points",
                    xytext=(0, 4),
                    ha="center",
                    fontsize=ann_fs,
                    color=ann_color,
                )

    ax0.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.1,
        handletextpad=0.35,
    )

    km = "per_step_mean_knn_coarse_threshold_iterations"
    ks = "per_step_std_knn_coarse_threshold_iterations_across_batches"

    def _plot_iters(rep: Dict, label: str, color: str, use_marker: bool) -> None:
        y = np.array(rep[km], dtype=np.float64)
        t = np.arange(len(y))
        if use_marker:
            ax1.plot(
                t,
                y,
                color=color,
                linewidth=2.0,
                marker="^",
                markersize=4,
                markevery=max(1, len(t) // 30),
                label=label,
            )
        else:
            ax1.plot(t, y, color=color, linewidth=2.0, label=label)
        if with_std_band and ks in rep:
            std = np.array(rep[ks], dtype=np.float64)
            if std.shape == y.shape:
                ax1.fill_between(
                    t, np.maximum(0, y - std), y + std, color=color, alpha=0.2
                )

    if rep_exact is not None:
        _plot_iters(rep_exact, "exact radii_ref", color_exact, use_marker=False)
    if rep_kan is not None:
        _plot_iters(rep_kan, "KAN radii_ref", color_kan, use_marker=True)

    cap = None
    if rep_exact is not None:
        cap = rep_exact.get("knn_coarse_max_iter_cap")
    elif rep_kan is not None:
        cap = rep_kan.get("knn_coarse_max_iter_cap")
    if cap is not None:
        ax1.axhline(
            float(cap),
            color=color_ref,
            linestyle=":",
            linewidth=1.0,
            label=f"max_iter = {cap}",
        )

    ax1.set_xlabel("FPS Iteration")
    ax1.set_ylabel("Coarse threshold\niterations")
    ax1.grid(axis="y", alpha=0.45, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.1,
        handletextpad=0.35,
    )

    plt.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.98, bottom=0.12, left=0.12, right=0.98, hspace=0.12)

    png_path = f"{save_path_base}.png"
    pdf_path = f"{save_path_base}.pdf"
    fig.savefig(
        pdf_path,
        transparent=True,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.02,
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        format="png",
        pad_inches=0.02,
    )
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def plot_knn_coarse_threshold_trace(
    save_path_base: str,
    trace_series: List[Tuple[str, Sequence[Tuple[float, int]]]],
    pool_lo: int,
    pool_hi: int,
    meta: Dict,
    font_size: int = 16,
) -> Optional[Dict[str, str]]:
    """
    粗筛二分 ``(h, |d<h|)`` 轨迹：支持多条序列 ``trace_series = [(label, pairs), ...]``。
    左图：各序列按各自 probe 索引 0..L-1 画 step；右图：h vs count（多色，无图例；对应关系见标题 trace_subtitle 与 JSON）。
    JSON：``knn_threshold_trace_example.per_sample_traces``（多样本）或单样本扁平字段。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except ImportError:
        print("  [plot] 未安装 matplotlib，跳过 KNN 阈值轨迹图。")
        return None

    series = [(lab, list(tr)) for lab, tr in trace_series if tr]
    if not series:
        return None

    def _plain_positive_float(x: float, _pos: int) -> str:
        if not np.isfinite(x) or x <= 0:
            return ""
        s = f"{x:.8f}".rstrip("0").rstrip(".")
        return s if s else "0"

    _configure_matplotlib_paper_style(font_size=font_size)
    _pal = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    )
    colors = [_pal[i % len(_pal)] for i in range(len(series))]
    n_multi = len(series) > 1
    w = 11.0 if n_multi else 10.0
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(w, 4.2), gridspec_kw={"width_ratios": [1.0, 1.05]}
    )

    max_len = 0
    all_h: List[float] = []
    for i, (_label, trace) in enumerate(series):
        col = colors[i]
        hs = np.array([t[0] for t in trace], dtype=np.float64)
        cs = np.array([t[1] for t in trace], dtype=np.int64)
        iters = np.arange(len(trace))
        max_len = max(max_len, len(trace))
        all_h.extend(hs[np.isfinite(hs) & (hs > 0)].tolist())
        ax0.step(
            iters,
            cs,
            where="post",
            color=col,
            linewidth=2.0,
        )
        ax0.scatter(iters, cs, color=col, s=28 if n_multi else 36, zorder=3, edgecolors="white", linewidths=0.4)
        ax1.plot(
            hs,
            cs,
            marker="o",
            color=col,
            linewidth=2.0,
            markersize=5 if n_multi else 6,
        )
        if not n_multi:
            ann_fs = max(6, font_size - 4)
            for h, c in trace:
                ax1.annotate(
                    str(c),
                    (h, c),
                    textcoords="offset points",
                    xytext=(5, 3),
                    fontsize=ann_fs,
                    color="#333333",
                )

    n_pts = max_len
    if n_pts <= 14:
        tick_idx = np.arange(n_pts)
    else:
        n_tick = min(12, n_pts)
        tick_idx = np.unique(np.linspace(0, n_pts - 1, num=n_tick, dtype=np.int64))
    ax0.set_xticks(tick_idx)
    if n_pts > 18:
        ax0.tick_params(axis="x", labelrotation=35)
    ax0.axhline(
        float(pool_lo),
        color="#555555",
        linestyle="--",
        linewidth=1.0,
    )
    ax0.axhline(
        float(pool_hi),
        color="#2d6a4f",
        linestyle=":",
        linewidth=1.2,
    )
    ax0.set_xlabel("Trace index (bisection probes)")
    ax0.set_ylabel("Candidate count |{j: d_j < h}|")
    ax0.grid(axis="y", alpha=0.45, linestyle="--")
    ax0.set_axisbelow(True)

    h_arr = np.array(all_h, dtype=np.float64)
    h_pos = h_arr[np.isfinite(h_arr) & (h_arr > 0)]
    h_ratio = float(np.max(h_pos) / np.min(h_pos)) if h_pos.size else 1.0
    ax1.yaxis.get_offset_text().set_visible(False)
    if h_ratio <= 80.0:
        ax1.set_xscale("linear")
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=9, prune="both"))
        ax1.xaxis.set_major_formatter(FuncFormatter(_plain_positive_float))
        ax1.xaxis.get_offset_text().set_visible(False)
        ax1.set_xlabel("Euclidean threshold h (linear)")
    else:
        ax1.set_xscale("log")
        ax1.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
        ax1.xaxis.set_major_formatter(FuncFormatter(_plain_positive_float))
        ax1.xaxis.set_minor_formatter(FuncFormatter(lambda x, p: ""))
        ax1.xaxis.get_offset_text().set_visible(False)
        ax1.set_xlabel("Euclidean threshold h (log scale)")
    ax1.axhline(
        float(pool_lo),
        color="#555555",
        linestyle="--",
        linewidth=1.0,
    )
    ax1.axhline(
        float(pool_hi),
        color="#2d6a4f",
        linestyle=":",
        linewidth=1.2,
    )
    ax1.set_ylabel("Candidate count")
    ax1.grid(True, alpha=0.45, linestyle="--")
    ax1.set_axisbelow(True)

    sub1 = f"batch={meta.get('batch_idx')}  FPS_t={meta.get('fps_t')}"
    sub2 = meta.get("trace_subtitle", "")
    fig.suptitle(
        f"KNN coarse threshold schedule ({len(series)} sample(s))\n{sub1}\n{sub2}",
        fontsize=font_size - 1,
    )
    foot = (
        "Full (h, count): JSON knn_threshold_trace_example.per_sample_traces "
        "or trace_pairs_h_count (single)"
    )
    fig.text(
        0.5,
        0.02,
        foot,
        ha="center",
        va="bottom",
        fontsize=max(7, font_size - 5),
        color="#666666",
    )

    for _ax in (ax0, ax1):
        _leg = _ax.get_legend()
        if _leg is not None:
            _leg.remove()
    try:
        for _fl in list(getattr(fig, "legends", ())):
            _fl.remove()
    except (AttributeError, TypeError, ValueError):
        pass

    plt.tight_layout(pad=0.35, rect=[0, 0.06, 1, 0.90])

    png_path = f"{save_path_base}.png"
    pdf_path = f"{save_path_base}.pdf"
    fig.savefig(
        pdf_path,
        transparent=True,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.02,
    )
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        format="png",
        pad_inches=0.02,
    )
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


# ---------------------------------------------------------------------------
# 单 batch：exact 参考、两路融合（exact / KAN radii_ref）、可选 coarse-fine baseline
# ---------------------------------------------------------------------------
def _knn_inner_iter_cap_for_plot(args) -> int:
    """KNN 粗筛探测步数上界（用于 JSON/图中参考线），即 max(1, max_iter)。"""
    return max(1, int(args.max_iter))


def _parse_sweep_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def _parse_sweep_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(float(x)))
    return out


def _sweep_clip_tag(clip: float) -> str:
    t = f"{float(clip):.6f}".rstrip("0").rstrip(".")
    return t.replace("-", "m").replace(".", "p")


def _sweep_init_tag(th: float) -> str:
    t = f"{float(th):.6f}".rstrip("0").rstrip(".")
    return t.replace("-", "m").replace(".", "p")


def run_fused_knn_hyperparam_sweep(args: argparse.Namespace) -> None:
    """
    在 seg_filter + coarse-fine 融合路径上扫 ``clip × max_iter × init_threshold`` 全网格。

    - 每个 ``knn_coarse_mult_alpha_clip``：一张 ``init_threshold × max_iter`` 曲面（Z = IoU）。
    - 若 ``sweep_clip_maxiter_surface``：每个 ``init_threshold`` 另输出一张
      ``mult_alpha_clip × max_iter`` 曲面。
    """
    from run_knn_coarse_fine_eval import plot_iou_surface

    init_list = _parse_sweep_csv_floats(args.sweep_init_thresholds)
    max_iter_list = _parse_sweep_csv_ints(args.sweep_max_iters)
    clip_list = _parse_sweep_csv_floats(args.sweep_mult_alpha_clips)
    if not init_list or not max_iter_list or not clip_list:
        raise ValueError("sweep_init_thresholds / sweep_max_iters / sweep_mult_alpha_clips 不能为空")

    prefix = str(args.sweep_surface_out_prefix).strip() or "./runs/fused_knn_sweep"
    out_dir = os.path.dirname(os.path.abspath(prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    knn_coarse_hi_cap = resolve_knn_coarse_distance_hi_cap(args)

    ds = ModelNet40Wrapper(
        partition=args.partition, num_points=args.num_points, root=args.root
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_points,
    )

    n_sb = int(args.sweep_max_batches)
    batches: List[torch.Tensor] = []
    for pts, _ in loader:
        batches.append(pts)
        if n_sb > 0 and len(batches) >= n_sb:
            break
    if not batches:
        raise RuntimeError("sweep: DataLoader 未取到任何 batch")

    try:
        n_ds = len(ds)
    except Exception:
        n_ds = -1
    approx_clouds = len(batches) * int(args.batch_size)
    print(
        f"  [sweep] 数据: ModelNet40 --partition={args.partition}"
        + (f"（约 {n_ds} 个样本）" if n_ds >= 0 else "")
        + f"；每网格点用 {len(batches)} 个 batch × batch_size={args.batch_size} "
        f"→ 每格约 {approx_clouds} 次 forward（总格数 {len(clip_list)*len(max_iter_list)*len(init_list)}）"
    )

    kan_dir = args.kan_save_dir.strip()
    model_kan = None
    cfg_kan = None
    if getattr(args, "sweep_exact_only", False):
        print("  [sweep] --sweep_exact_only：跳过 KAN 加载与 kan 曲面")
    elif kan_dir and os.path.isdir(kan_dir):
        model_kan, cfg_kan = load_kan_radii_model(kan_dir, device)
        if cfg_kan.fps_steps != args.K:
            raise ValueError(
                f"KAN cfg.fps_steps={cfg_kan.fps_steps} 与 --K={args.K} 不一致"
            )
        cfg_kan.prefix_ratio = args.prefix_ratio

    results: Dict = {
        "partition": args.partition,
        "num_points": args.num_points,
        "K": args.K,
        "k_neighbors": args.k_neighbors,
        "min_candidates_ratio": float(args.min_candidates_ratio),
        "knn_coarse_schedule": str(args.knn_coarse_schedule),
        "knn_coarse_pool_shortage": str(args.knn_coarse_pool_shortage),
        "knn_coarse_hi_cap": knn_coarse_hi_cap,
        "init_thresholds": init_list,
        "max_iters": max_iter_list,
        "mult_alpha_clips": clip_list,
        "num_batches_per_cell": len(batches),
        "sweep_exact_only": bool(getattr(args, "sweep_exact_only", False)),
        "sweep_clip_maxiter_surface": bool(
            getattr(args, "sweep_clip_maxiter_surface", True)
        ),
        "surfaces": [],
        "surfaces_clip_by_maxiter": [],
    }

    n_c = len(clip_list)
    n_mi = len(max_iter_list)
    n_in = len(init_list)
    Z_sum_ex = np.zeros((n_c, n_mi, n_in), dtype=np.float64)
    Z_sum_kan: Optional[np.ndarray] = (
        np.zeros_like(Z_sum_ex) if model_kan is not None else None
    )

    n_cells = n_c * n_mi * n_in
    nb = len(batches)
    n_ops = n_cells * nb
    use_sweep_d_stack = bool(args.skip_cf_baseline) and not bool(
        getattr(args, "no_sweep_d_knn_precompute", False)
    )
    results["sweep_d_knn_precompute"] = use_sweep_d_stack
    pbar = tqdm(total=n_ops, desc="fused knn sweep", ncols=100)
    if use_sweep_d_stack:
        print(
            f"  [sweep] 加速：每 batch 1 次 exact FPS + 1 次 seg_filter 轨迹，"
            f"缓存 d_knn [B,K,N]；{n_cells} 个网格仅做粗筛+topk（需 --skip_cf_baseline）"
        )
    else:
        print(
            f"  [sweep] 每个 batch 复用 1 次 exact FPS idx/radii，再全融合跑 {n_cells} 格"
            + (
                "（可加 --skip_cf_baseline 以启用 d_knn 预计算加速）"
                if not args.skip_cf_baseline
                else "（已加 --no_sweep_d_knn_precompute，关闭 d_knn 预计算）"
            )
        )

    dtype_eval = torch.float32

    for pts in batches:
        pts_d = pts.to(device, dtype=torch.float32)
        idx_e, radii_e, _, _ = fps_adaptive_argmax(
            pts_d, args.K, seed=args.seed, mode="exact"
        )
        if use_sweep_d_stack:
            seg0 = (
                int(kan_prefix_len_from_K(args.K, args.prefix_ratio))
                if args.defer_seg_filter
                else 1
            )
            idx_f_ex, d_ex = fps_segfilter_knn_precompute_d_knn(
                pts_d,
                args.K,
                radii_e,
                args.segment_len,
                seed=args.seed,
                exclude_self=args.exclude_self,
                empty_fallback="global",
                seg_filter_start_t=seg0,
            )
            d_kan: Optional[torch.Tensor] = None
            idx_f_kan: Optional[torch.Tensor] = None
            if Z_sum_kan is not None and model_kan is not None and cfg_kan is not None:
                if args.kan_input == "fps_points":
                    radii_kan, _pl = predict_full_radii_squared_from_fps_prefix_points(
                        pts_d, idx_e, args.K, model_kan, cfg_kan, device
                    )
                else:
                    radii_kan = predict_full_radii_squared_from_exact(
                        radii_e, model_kan, cfg_kan, device
                    )
                idx_f_kan, d_kan = fps_segfilter_knn_precompute_d_knn(
                    pts_d,
                    args.K,
                    radii_kan,
                    args.segment_len,
                    seed=args.seed,
                    exclude_self=args.exclude_self,
                    empty_fallback="global",
                    seg_filter_start_t=seg0,
                )
            for ci, clip in enumerate(clip_list):
                for yi, mi in enumerate(max_iter_list):
                    mi_u = max(1, int(mi))
                    for xi, th in enumerate(init_list):
                        Z_sum_ex[ci, yi, xi] += sweep_mean_knn_iou_from_d_stack(
                            d_ex,
                            idx_f_ex,
                            pts_d,
                            args.k_neighbors,
                            args.exclude_self,
                            float(th),
                            int(mi_u),
                            args.min_candidates_ratio,
                            dtype_eval,
                            knn_coarse_hi_cap,
                            args.knn_coarse_schedule,
                            float(clip),
                            args.knn_coarse_pool_shortage,
                        )
                        if (
                            Z_sum_kan is not None
                            and d_kan is not None
                            and idx_f_kan is not None
                        ):
                            Z_sum_kan[ci, yi, xi] += sweep_mean_knn_iou_from_d_stack(
                                d_kan,
                                idx_f_kan,
                                pts_d,
                                args.k_neighbors,
                                args.exclude_self,
                                float(th),
                                int(mi_u),
                                args.min_candidates_ratio,
                                dtype_eval,
                                knn_coarse_hi_cap,
                                args.knn_coarse_schedule,
                                float(clip),
                                args.knn_coarse_pool_shortage,
                            )
                        pbar.update(1)
        else:
            for ci, clip in enumerate(clip_list):
                for yi, mi in enumerate(max_iter_list):
                    mi_u = max(1, int(mi))
                    cap = mi_u
                    for xi, th in enumerate(init_list):
                        m = run_one_batch_metrics(
                            pts_d,
                            K=args.K,
                            segment_len=args.segment_len,
                            k_neighbors=args.k_neighbors,
                            init_threshold=float(th),
                            max_iter=int(mi_u),
                            min_candidates_ratio=args.min_candidates_ratio,
                            seed=args.seed,
                            exclude_self=args.exclude_self,
                            device=device,
                            model_kan=model_kan,
                            cfg_kan=cfg_kan,
                            skip_cf_baseline=args.skip_cf_baseline,
                            prefix_ratio=args.prefix_ratio,
                            defer_seg_filter=args.defer_seg_filter,
                            kan_input=args.kan_input,
                            knn_inner_iter_cap=cap,
                            distance_hi_cap=knn_coarse_hi_cap,
                            coarse_schedule=args.knn_coarse_schedule,
                            mult_alpha_clip=float(clip),
                            knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
                            idx_exact_pre=idx_e,
                            radii_sq_exact_pre=radii_e,
                        )
                        Z_sum_ex[ci, yi, xi] += float(
                            m["fused_exact_radii_mean_knn_iou"]
                        )
                        if Z_sum_kan is not None:
                            Z_sum_kan[ci, yi, xi] += float(
                                m["fused_kan_radii_mean_knn_iou"]
                            )
                        pbar.update(1)

    pbar.close()
    Z_ex_all = Z_sum_ex / float(nb)
    Z_kan_all = Z_sum_kan / float(nb) if Z_sum_kan is not None else None

    for ci, clip in enumerate(clip_list):
        Z_ex = Z_ex_all[ci]
        ctag = _sweep_clip_tag(float(clip))
        path_ex = f"{prefix}_fused_exact_knn_iou_surface_clip={ctag}.png"
        plot_iou_surface(
            init_list,
            max_iter_list,
            Z_ex,
            save_path=path_ex,
            zlabel="KNN IoU",
            xlabel="init_threshold",
            ylabel="max_iter",
            title=f"mult_alpha_clip={float(clip):g} (fixed per figure)",
        )
        rec: Dict = {
            "mult_alpha_clip": float(clip),
            "metric": "fused_exact_radii_mean_knn_iou",
            "png": path_ex,
            "Z": Z_ex.tolist(),
        }
        print(f"  [sweep] wrote {path_ex}")
        if Z_kan_all is not None:
            path_kan = f"{prefix}_fused_kan_knn_iou_surface_clip={ctag}.png"
            plot_iou_surface(
                init_list,
                max_iter_list,
                Z_kan_all[ci],
                save_path=path_kan,
                zlabel="KNN IoU",
                xlabel="init_threshold",
                ylabel="max_iter",
                title=f"mult_alpha_clip={float(clip):g} (fixed per figure)",
            )
            rec["metric_kan"] = "fused_kan_radii_mean_knn_iou"
            rec["png_kan"] = path_kan
            rec["Z_kan"] = Z_kan_all[ci].tolist()
            print(f"  [sweep] wrote {path_kan}")
        results["surfaces"].append(rec)

    if getattr(args, "sweep_clip_maxiter_surface", True):
        for xi, th in enumerate(init_list):
            # Z_cm[yi, ci] = IoU at (max_iter[yi], clip[ci]), fixed init
            Z_cm_ex = np.transpose(Z_ex_all[:, :, xi], (1, 0))
            htag = _sweep_init_tag(float(th))
            path_cm_ex = (
                f"{prefix}_fused_exact_knn_iou_surface_clip_by_maxiter_init={htag}.png"
            )
            plot_iou_surface(
                clip_list,
                max_iter_list,
                Z_cm_ex,
                save_path=path_cm_ex,
                zlabel="KNN IoU",
                xlabel="mult_alpha_clip",
                ylabel="max_iter",
                title=f"init_threshold={float(th):g} (fixed per figure)",
            )
            cm_rec: Dict = {
                "init_threshold": float(th),
                "metric": "fused_exact_radii_mean_knn_iou",
                "axes": "mult_alpha_clip x max_iter",
                "png": path_cm_ex,
                "Z": Z_cm_ex.tolist(),
            }
            print(f"  [sweep] wrote {path_cm_ex}")
            if Z_kan_all is not None:
                Z_cm_kan = np.transpose(Z_kan_all[:, :, xi], (1, 0))
                path_cm_kan = (
                    f"{prefix}_fused_kan_knn_iou_surface_clip_by_maxiter_init={htag}.png"
                )
                plot_iou_surface(
                    clip_list,
                    max_iter_list,
                    Z_cm_kan,
                    save_path=path_cm_kan,
                    zlabel="KNN IoU",
                    xlabel="mult_alpha_clip",
                    ylabel="max_iter",
                    title=f"init_threshold={float(th):g} (fixed per figure)",
                )
                cm_rec["metric_kan"] = "fused_kan_radii_mean_knn_iou"
                cm_rec["png_kan"] = path_cm_kan
                cm_rec["Z_kan"] = Z_cm_kan.tolist()
                print(f"  [sweep] wrote {path_cm_kan}")
            results["surfaces_clip_by_maxiter"].append(cm_rec)

    results["Z_ex_all_order"] = "[clip_i][max_iter_j][init_k] 与 mult_alpha_clips,max_iters,init_thresholds 下标一致"
    results["Z_ex_all"] = Z_ex_all.tolist()
    if Z_kan_all is not None:
        results["Z_kan_all"] = Z_kan_all.tolist()

    json_path = f"{prefix}_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    print(f"  [sweep] wrote {json_path}")


@torch.no_grad()
def run_one_batch_metrics(
    pts: torch.Tensor,
    K: int,
    segment_len: int,
    k_neighbors: int,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    seed: str,
    exclude_self: bool,
    device: torch.device,
    model_kan: Optional[torch.nn.Module],
    cfg_kan: Optional[Config],
    skip_cf_baseline: bool,
    prefix_ratio: float,
    defer_seg_filter: bool,
    kan_input: str,
    knn_inner_iter_cap: int,
    distance_hi_cap: Optional[float] = None,
    coarse_schedule: str = "bisect",
    mult_alpha_clip: float = 0.25,
    knn_coarse_pool_shortage: str = "keep_mid",
    idx_exact_pre: Optional[torch.Tensor] = None,
    radii_sq_exact_pre: Optional[torch.Tensor] = None,
) -> Dict:
    pts = pts.to(device, dtype=torch.float32)
    B, N, _ = pts.shape

    if (idx_exact_pre is None) ^ (radii_sq_exact_pre is None):
        raise ValueError("idx_exact_pre 与 radii_sq_exact_pre 须同时提供或同时为 None")

    # 自该步起启用 segment 阈值 farthest；与 kan_prefix_len 对齐（defer 开启时）
    seg_filter_start_t = (
        kan_prefix_len_from_K(K, prefix_ratio) if defer_seg_filter else 1
    )

    # 标准 exact FPS（平方距离）；扫参时可传入预计算结果以避免每格重复算
    if idx_exact_pre is not None:
        idx_exact = idx_exact_pre
        radii_sq_exact = radii_sq_exact_pre
    else:
        idx_exact, radii_sq_exact, _, _ = fps_adaptive_argmax(
            pts, K, seed=seed, mode="exact"
        )

    knn_ref_exact: Optional[torch.Tensor] = None
    if not skip_cf_baseline:
        knn_ref_exact = knn_exact_topk_at_centers(
            pts, idx_exact, k_neighbors, exclude_self=exclude_self
        )

    out: Dict = {
        "seg_filter_start_t": float(seg_filter_start_t),
        "kan_prefix_len": float(kan_prefix_len_from_K(K, prefix_ratio)),
    }

    def pack_fused_branch(
        label: str,
        radii_ref_sq: torch.Tensor,
    ) -> None:
        # radii_ref_sq：整段 K 步平方阈值（exact 或 KAN）；stats 含每步 candidate_ratio
        idx_f, _, knn_f, _, _, _, stats_f = fps_segfilter_knn_coarse_fine_fused(
            pts,
            K,
            k_neighbors,
            radii_ref_sq,
            segment_len,
            init_threshold=init_threshold,
            max_iter=max_iter,
            min_candidates_ratio=min_candidates_ratio,
            seed=seed,
            exclude_self=exclude_self,
            return_distances_matrix=False,
            seg_filter_start_t=seg_filter_start_t,
            distance_hi_cap=distance_hi_cap,
            coarse_schedule=coarse_schedule,
            mult_alpha_clip=mult_alpha_clip,
            knn_coarse_pool_shortage=knn_coarse_pool_shortage,
        )
        iou_fps, _, _ = compute_iou_batch(idx_exact, idx_f)
        knn_ref_at_fused = knn_exact_topk_at_centers(
            pts, idx_f, k_neighbors, exclude_self=exclude_self
        )
        knn_iou_vs_exact_topk = knn_neighbor_set_iou(knn_f, knn_ref_at_fused)
        out[f"{label}_mean_iou_fps"] = float(iou_fps.mean().item())
        out[f"{label}_std_iou_fps"] = float(iou_fps.std().item())
        out[f"{label}_mean_knn_iou"] = float(knn_iou_vs_exact_topk.mean().item())
        out[f"{label}_std_knn_iou"] = float(knn_iou_vs_exact_topk.std().item())
        out[f"{label}_filter_stats"] = _aggregate_filter_stats(
            [stats_f["candidate_ratio"]],
            [stats_f["used_fallback"]],
            N,
            K,
            segment_len,
        )
        out[f"{label}_knn_coarse_candidate_stats"] = _knn_coarse_stats_one_batch(
            stats_f["knn_coarse_candidate_count"],
            stats_f["knn_coarse_threshold_iters"],
            N,
            knn_inner_iter_cap,
        )

    pack_fused_branch("fused_exact_radii", radii_sq_exact)

    if model_kan is not None and cfg_kan is not None:
        if kan_input == "fps_points":
            radii_kan, _pl = predict_full_radii_squared_from_fps_prefix_points(
                pts, idx_exact, K, model_kan, cfg_kan, device
            )
        else:
            radii_kan = predict_full_radii_squared_from_exact(
                radii_sq_exact, model_kan, cfg_kan, device
            )
        pack_fused_branch("fused_kan_radii", radii_kan)
        out["iou_fps_drop_kan_vs_exact_radii"] = (
            out["fused_exact_radii_mean_iou_fps"] - out["fused_kan_radii_mean_iou_fps"]
        )
        out["knn_iou_drop_kan_vs_exact_radii"] = (
            out["fused_exact_radii_mean_knn_iou"] - out["fused_kan_radii_mean_knn_iou"]
        )

    if not skip_cf_baseline:
        idx_cf, _, knn_cf, _, _, _ = fps_with_knn_coarse_fine_fused(
            pts,
            K,
            k_neighbors=k_neighbors,
            init_threshold=init_threshold,
            max_iter=max_iter,
            min_candidates_ratio=min_candidates_ratio,
            seed=seed,
            exclude_self=exclude_self,
            return_distances_matrix=False,
            distance_hi_cap=distance_hi_cap,
            coarse_schedule=coarse_schedule,
            mult_alpha_clip=mult_alpha_clip,
            knn_coarse_pool_shortage=knn_coarse_pool_shortage,
        )
        knn_iou_cf = knn_neighbor_set_iou(knn_cf, knn_ref_exact)
        out["cf_only_mean_iou_fps"] = float(
            compute_iou_batch(idx_exact, idx_cf)[0].mean().item()
        )
        out["cf_only_mean_knn_iou"] = float(knn_iou_cf.mean().item())

    return out


# ---------------------------------------------------------------------------
# 主程序：遍历 DataLoader，聚合并写 JSON / 打印 IoU、seg_filter 候选点数、耗时
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if getattr(args, "sweep_fused_knn_surface", False):
        run_fused_knn_hyperparam_sweep(args)
        return

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    knn_iter_cap = _knn_inner_iter_cap_for_plot(args)
    knn_coarse_hi_cap = resolve_knn_coarse_distance_hi_cap(args)

    ds = ModelNet40Wrapper(
        partition=args.partition, num_points=args.num_points, root=args.root
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_points,
    )

    kan_dir = args.kan_save_dir.strip()
    model_kan = None
    cfg_kan = None
    if kan_dir and os.path.isdir(kan_dir):
        model_kan, cfg_kan = load_kan_radii_model(kan_dir, device)
        if cfg_kan.fps_steps != args.K:
            raise ValueError(
                f"KAN cfg.fps_steps={cfg_kan.fps_steps} 与 --K={args.K} 不一致"
            )
        cfg_kan.prefix_ratio = args.prefix_ratio

    agg: Dict[str, List[float]] = {}
    agg_filter_exact: List = []  # 每元素为 _aggregate_filter_stats 字典（含每步候选点数）
    agg_filter_kan: List = []
    agg_knn_coarse_exact: List = []
    agg_knn_coarse_kan: List = []

    time_rows: Dict[str, List[float]] = {
        "fused_exact_radii_sec": [],
        "fused_kan_radii_sec": [],
        "cf_only_sec": [],
        "exact_fps_sec": [],
    }

    knn_trace_payload: Optional[Dict] = None
    nb = 0
    for pts, _ in tqdm(loader, desc="fused eval", ncols=100):
        m = run_one_batch_metrics(
            pts,
            K=args.K,
            segment_len=args.segment_len,
            k_neighbors=args.k_neighbors,
            init_threshold=args.init_threshold,
            max_iter=args.max_iter,
            min_candidates_ratio=args.min_candidates_ratio,
            seed=args.seed,
            exclude_self=args.exclude_self,
            device=device,
            model_kan=model_kan,
            cfg_kan=cfg_kan,
            skip_cf_baseline=args.skip_cf_baseline,
            prefix_ratio=args.prefix_ratio,
            defer_seg_filter=args.defer_seg_filter,
            kan_input=args.kan_input,
            knn_inner_iter_cap=knn_iter_cap,
            distance_hi_cap=knn_coarse_hi_cap,
            coarse_schedule=args.knn_coarse_schedule,
            mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
            knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
        )
        for k, v in m.items():
            if k.endswith("_filter_stats") or k.endswith("_knn_coarse_candidate_stats"):
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in ("seg_filter_start_t", "kan_prefix_len"):
                    continue
                agg.setdefault(k, []).append(float(v))
        if "fused_exact_radii_filter_stats" in m:
            agg_filter_exact.append(m["fused_exact_radii_filter_stats"])
        if "fused_kan_radii_filter_stats" in m:
            agg_filter_kan.append(m["fused_kan_radii_filter_stats"])
        if "fused_exact_radii_knn_coarse_candidate_stats" in m:
            agg_knn_coarse_exact.append(m["fused_exact_radii_knn_coarse_candidate_stats"])
        if "fused_kan_radii_knn_coarse_candidate_stats" in m:
            agg_knn_coarse_kan.append(m["fused_kan_radii_knn_coarse_candidate_stats"])

        if nb < args.time_batches:
            pts_d = pts.to(device, dtype=torch.float32)
            idx_e, radii_e, _, _ = fps_adaptive_argmax(
                pts_d, args.K, seed=args.seed, mode="exact"
            )
            s0 = (
                kan_prefix_len_from_K(args.K, args.prefix_ratio)
                if args.defer_seg_filter
                else 1
            )

            def t_exact():
                fps_adaptive_argmax(pts_d, args.K, seed=args.seed, mode="exact")

            def t_fused_ex():
                fps_segfilter_knn_coarse_fine_fused(
                    pts_d,
                    args.K,
                    args.k_neighbors,
                    radii_e,
                    args.segment_len,
                    init_threshold=args.init_threshold,
                    max_iter=args.max_iter,
                    min_candidates_ratio=args.min_candidates_ratio,
                    seed=args.seed,
                    exclude_self=args.exclude_self,
                    return_distances_matrix=False,
                    seg_filter_start_t=s0,
                    distance_hi_cap=knn_coarse_hi_cap,
                    coarse_schedule=args.knn_coarse_schedule,
                    mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                    knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
                )

            def t_fused_kan():
                if model_kan is None:
                    return
                if args.kan_input == "fps_points":
                    rk, _ = predict_full_radii_squared_from_fps_prefix_points(
                        pts_d, idx_e, args.K, model_kan, cfg_kan, device
                    )
                else:
                    rk = predict_full_radii_squared_from_exact(
                        radii_e, model_kan, cfg_kan, device
                    )
                fps_segfilter_knn_coarse_fine_fused(
                    pts_d,
                    args.K,
                    args.k_neighbors,
                    rk,
                    args.segment_len,
                    init_threshold=args.init_threshold,
                    max_iter=args.max_iter,
                    min_candidates_ratio=args.min_candidates_ratio,
                    seed=args.seed,
                    exclude_self=args.exclude_self,
                    return_distances_matrix=False,
                    seg_filter_start_t=s0,
                    distance_hi_cap=knn_coarse_hi_cap,
                    coarse_schedule=args.knn_coarse_schedule,
                    mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                    knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
                )

            def t_cf():
                fps_with_knn_coarse_fine_fused(
                    pts_d,
                    args.K,
                    k_neighbors=args.k_neighbors,
                    init_threshold=args.init_threshold,
                    max_iter=args.max_iter,
                    min_candidates_ratio=args.min_candidates_ratio,
                    seed=args.seed,
                    exclude_self=args.exclude_self,
                    return_distances_matrix=False,
                    distance_hi_cap=knn_coarse_hi_cap,
                    coarse_schedule=args.knn_coarse_schedule,
                    mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                    knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
                )

            time_rows["exact_fps_sec"].append(
                time_callable(t_exact, args.warmup, device)
            )
            time_rows["fused_exact_radii_sec"].append(
                time_callable(t_fused_ex, args.warmup, device)
            )
            if model_kan is not None:
                time_rows["fused_kan_radii_sec"].append(
                    time_callable(t_fused_kan, args.warmup, device)
                )
            if not args.skip_cf_baseline:
                time_rows["cf_only_sec"].append(
                    time_callable(t_cf, args.warmup, device)
                )

        if args.plot_knn_threshold_trace:
            trace_target = resolve_knn_trace_target_count(args)
            if trace_target > 0:
                if nb >= args.knn_trace_batch and not resolve_knn_trace_payload_done(
                    knn_trace_payload, trace_target
                ):
                    pts_d = pts.to(device, dtype=torch.float32)
                    B = int(pts_d.size(0))
                    s0 = (
                        kan_prefix_len_from_K(args.K, args.prefix_ratio)
                        if args.defer_seg_filter
                        else 1
                    )
                    t_sel = args.knn_trace_t if args.knn_trace_t >= 0 else min(s0 + 10, args.K - 1)
                    t_sel = int(max(0, min(t_sel, args.K - 1)))
                    _idx_e, radii_e, _, _ = fps_adaptive_argmax(
                        pts_d, args.K, seed=args.seed, mode="exact"
                    )
                    idx_f, _, _, _, _, _, _ = fps_segfilter_knn_coarse_fine_fused(
                        pts_d,
                        args.K,
                        args.k_neighbors,
                        radii_e,
                        args.segment_len,
                        init_threshold=args.init_threshold,
                        max_iter=args.max_iter,
                        min_candidates_ratio=args.min_candidates_ratio,
                        seed=args.seed,
                        exclude_self=args.exclude_self,
                        return_distances_matrix=False,
                        seg_filter_start_t=s0,
                        distance_hi_cap=knn_coarse_hi_cap,
                        coarse_schedule=args.knn_coarse_schedule,
                        mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                        knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
                    )
                    pool_lo = pool_hi = 0
                    if knn_trace_payload is None:
                        knn_trace_payload = {
                            "batch_idx": nb,
                            "fps_t": t_sel,
                            "seg_filter_start_t": int(s0),
                            "init_threshold": float(args.init_threshold),
                            "max_iter_cap": int(knn_iter_cap),
                            "min_candidates_ratio": float(args.min_candidates_ratio),
                            "k_neighbors": int(args.k_neighbors),
                            "exclude_self": bool(args.exclude_self),
                            "knn_coarse_pool_lo": 0,
                            "knn_coarse_pool_hi": 0,
                            "per_sample_traces": [],
                            "num_trace_samples": 0,
                            "trace_subtitle": "",
                            "sample_idx": 0,
                            "center_idx": 0,
                            "trace_pairs_h_count": [],
                            "distance_hi_cap": knn_coarse_hi_cap,
                            "knn_trace_target": int(trace_target),
                            "knn_trace_batch_start": int(args.knn_trace_batch),
                            "knn_coarse_schedule": str(args.knn_coarse_schedule),
                            "knn_coarse_mult_alpha_clip": float(
                                args.knn_coarse_mult_alpha_clip
                            ),
                            "note_en": (
                                "Accumulated traces across batches from knn_trace_batch_start; "
                                "each entry has batch_idx, sample_idx, center_idx, trace_pairs_h_count. "
                                "Distribution plot: mean±std of candidate count vs probe index."
                            ),
                        }
                    for si in range(B):
                        if len(knn_trace_payload["per_sample_traces"]) >= trace_target:
                            break
                        center = int(idx_f[si, t_sel].item())
                        trace_pairs, pool_lo, pool_hi = knn_coarse_threshold_trace_for_center(
                            pts_d,
                            si,
                            center,
                            args.init_threshold,
                            args.max_iter,
                            args.min_candidates_ratio,
                            args.k_neighbors,
                            args.exclude_self,
                            distance_hi_cap=knn_coarse_hi_cap,
                            coarse_schedule=args.knn_coarse_schedule,
                            mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                            pool_shortage=args.knn_coarse_pool_shortage,
                        )
                        knn_trace_payload["per_sample_traces"].append(
                            {
                                "batch_idx": int(nb),
                                "sample_idx": int(si),
                                "center_idx": int(center),
                                "trace_pairs_h_count": [
                                    [float(h), int(c)] for h, c in trace_pairs
                                ],
                            }
                        )
                    knn_trace_payload["knn_coarse_pool_lo"] = int(pool_lo)
                    knn_trace_payload["knn_coarse_pool_hi"] = int(pool_hi)
                    knn_trace_payload["batch_idx"] = int(nb)
                    ps = knn_trace_payload["per_sample_traces"]
                    knn_trace_payload["num_trace_samples"] = int(len(ps))
                    knn_trace_payload["trace_subtitle"] = (
                        f"n={len(ps)}/{trace_target}  last_batch={nb}"
                    )
                    if ps:
                        knn_trace_payload["sample_idx"] = int(ps[0]["sample_idx"])
                        knn_trace_payload["center_idx"] = int(ps[0]["center_idx"])
                        knn_trace_payload["trace_pairs_h_count"] = ps[0][
                            "trace_pairs_h_count"
                        ]
            elif knn_trace_payload is None and nb == args.knn_trace_batch:
                pts_d = pts.to(device, dtype=torch.float32)
                B = int(pts_d.size(0))
                sample_ids = parse_knn_trace_sample_indices(args, B)
                s0 = (
                    kan_prefix_len_from_K(args.K, args.prefix_ratio)
                    if args.defer_seg_filter
                    else 1
                )
                t_sel = args.knn_trace_t if args.knn_trace_t >= 0 else min(s0 + 10, args.K - 1)
                t_sel = int(max(0, min(t_sel, args.K - 1)))
                _idx_e, radii_e, _, _ = fps_adaptive_argmax(
                    pts_d, args.K, seed=args.seed, mode="exact"
                )
                idx_f, _, _, _, _, _, _ = fps_segfilter_knn_coarse_fine_fused(
                    pts_d,
                    args.K,
                    args.k_neighbors,
                    radii_e,
                    args.segment_len,
                    init_threshold=args.init_threshold,
                    max_iter=args.max_iter,
                    min_candidates_ratio=args.min_candidates_ratio,
                    seed=args.seed,
                    exclude_self=args.exclude_self,
                    return_distances_matrix=False,
                    seg_filter_start_t=s0,
                    distance_hi_cap=knn_coarse_hi_cap,
                    coarse_schedule=args.knn_coarse_schedule,
                    mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                    knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
                )
                per_sample: List[Dict] = []
                for si in sample_ids:
                    center = int(idx_f[si, t_sel].item())
                    trace_pairs, pool_lo, pool_hi = knn_coarse_threshold_trace_for_center(
                        pts_d,
                        si,
                        center,
                        args.init_threshold,
                        args.max_iter,
                        args.min_candidates_ratio,
                        args.k_neighbors,
                        args.exclude_self,
                        distance_hi_cap=knn_coarse_hi_cap,
                        coarse_schedule=args.knn_coarse_schedule,
                        mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                        pool_shortage=args.knn_coarse_pool_shortage,
                    )
                    per_sample.append(
                        {
                            "batch_idx": int(nb),
                            "sample_idx": int(si),
                            "center_idx": int(center),
                            "trace_pairs_h_count": [
                                [float(h), int(c)] for h, c in trace_pairs
                            ],
                        }
                    )
                first = per_sample[0]
                trace_sub = "  ".join(
                    f"s{item['sample_idx']}:c{item['center_idx']}" for item in per_sample
                )
                if len(trace_sub) > 120:
                    trace_sub = trace_sub[:117] + "..."
                knn_trace_payload = {
                    "batch_idx": nb,
                    "fps_t": t_sel,
                    "seg_filter_start_t": int(s0),
                    "init_threshold": float(args.init_threshold),
                    "max_iter_cap": int(knn_iter_cap),
                    "min_candidates_ratio": float(args.min_candidates_ratio),
                    "k_neighbors": int(args.k_neighbors),
                    "exclude_self": bool(args.exclude_self),
                    "knn_coarse_pool_lo": int(pool_lo),
                    "knn_coarse_pool_hi": int(pool_hi),
                    "per_sample_traces": per_sample,
                    "num_trace_samples": int(len(per_sample)),
                    "knn_trace_target": 0,
                    "knn_trace_batch_start": int(args.knn_trace_batch),
                    "knn_coarse_schedule": str(args.knn_coarse_schedule),
                    "knn_coarse_mult_alpha_clip": float(args.knn_coarse_mult_alpha_clip),
                    "trace_subtitle": trace_sub,
                    "sample_idx": int(first["sample_idx"]),
                    "center_idx": int(first["center_idx"]),
                    "trace_pairs_h_count": first["trace_pairs_h_count"],
                    "distance_hi_cap": knn_coarse_hi_cap,
                    "note_en": (
                        "Bisection uses hi = min(hi_full, distance_hi_cap) with hi_full = d_max * (1+delta) + tiny "
                        "(strictly above d_max, no literal +1). First probe is init_threshold; the next probe is "
                        "midpoint of [lo, hi], so count can still jump if the bracket is wide. Optional distance_hi_cap "
                        "from a calibration JSON tightens hi. If count stays below pool_lo, knn_coarse_pool_shortage "
                        "is keep_mid (no expand) or search_hi (expand to hi=min(hi_full,cap)); then fine top-k runs. "
                        "Multiple batch samples: see per_sample_traces."
                    ),
                }

        nb += 1
        if args.max_batches > 0 and nb >= args.max_batches:
            break

    if knn_trace_payload is not None and int(knn_trace_payload.get("knn_trace_target") or 0) > 0:
        tgt = int(knn_trace_payload["knn_trace_target"])
        got = len(knn_trace_payload.get("per_sample_traces") or [])
        if got < tgt:
            print(
                f"  [warn] knn_trace: 已收集 {got}/{tgt} 条轨迹；请增大 --max_batches 或 --batch_size。"
            )

    def mean_std(key: str) -> Tuple[Optional[float], Optional[float]]:
        xs = agg.get(key, [])
        if not xs:
            return None, None
        return float(np.mean(xs)), float(np.std(xs))

    results: Dict = {
        "partition": args.partition,
        "num_points": args.num_points,
        "K": args.K,
        "segment_len": args.segment_len,
        "k_neighbors": args.k_neighbors,
        "kan_save_dir": kan_dir or None,
        "prefix_ratio": args.prefix_ratio,
        "defer_seg_filter": args.defer_seg_filter,
        "seg_filter_start_t": kan_prefix_len_from_K(args.K, args.prefix_ratio)
        if args.defer_seg_filter
        else 1,
        "kan_prefix_len": kan_prefix_len_from_K(args.K, args.prefix_ratio),
        "kan_input": args.kan_input,
        "max_batches_limit": args.max_batches,
        "num_batches": nb,
        "min_candidates_ratio_1_plus_tau": float(args.min_candidates_ratio),
        "knn_coarse_distance_hi_cap": knn_coarse_hi_cap,
        "knn_coarse_schedule": str(args.knn_coarse_schedule),
        "knn_coarse_mult_alpha_clip": float(args.knn_coarse_mult_alpha_clip),
        "knn_coarse_pool_shortage": str(args.knn_coarse_pool_shortage),
        "knn_inner_iter_cap_report": int(knn_iter_cap),
        "metrics_per_batch_mean_std": {},
        "timing_sec_per_batch": {},
    }

    metric_keys = [
        k
        for k in agg.keys()
        if k not in ("iou_fps_drop_kan_vs_exact_radii", "knn_iou_drop_kan_vs_exact_radii")
    ]
    for k in sorted(metric_keys):
        mu, sig = mean_std(k)
        results["metrics_per_batch_mean_std"][k] = {"mean": mu, "std": sig}

    for dk, dv in (
        ("iou_fps_drop_kan_vs_exact_radii", "iou_fps_drop_kan_vs_exact_radii"),
        ("knn_iou_drop_kan_vs_exact_radii", "knn_iou_drop_kan_vs_exact_radii"),
    ):
        mu, sig = mean_std(dv)
        if mu is not None:
            results["metrics_per_batch_mean_std"][dk] = {"mean": mu, "std": sig}

    def _mean_filter_scalars(rows: List[Dict], prefix: str) -> None:
        if not rows:
            return
        keys = (
            "candidate_ratio_mean",
            "candidate_ratio_std",
            "used_fallback_any_ratio",
            "search_space_reduction_ratio",
            "scan_saved_ratio",
            "total_scan_vanilla",
            "total_scan_seg_filter",
        )
        results[f"{prefix}_aggregated"] = {
            k: float(np.mean([float(d[k]) for d in rows])) for k in keys if k in rows[0]
        }

    _mean_filter_scalars(agg_filter_exact, "filter_stats_fused_exact_radii")
    _mean_filter_scalars(agg_filter_kan, "filter_stats_fused_kan_radii")

    s0_report = int(results["seg_filter_start_t"])
    rep_ex = _build_fps_seg_filter_candidate_report(
        agg_filter_exact, s0_report, args.num_points
    )
    if rep_ex is not None:
        results["fps_seg_filter_candidate_report_fused_exact_radii"] = rep_ex
    rep_kan = _build_fps_seg_filter_candidate_report(
        agg_filter_kan, s0_report, args.num_points
    )
    if rep_kan is not None:
        results["fps_seg_filter_candidate_report_fused_kan_radii"] = rep_kan

    knn_rep_ex = _build_knn_coarse_candidate_report(agg_knn_coarse_exact)
    if knn_rep_ex is not None:
        results["knn_coarse_candidate_report_fused_exact_radii"] = knn_rep_ex
    knn_rep_kan = _build_knn_coarse_candidate_report(agg_knn_coarse_kan)
    if knn_rep_kan is not None:
        results["knn_coarse_candidate_report_fused_kan_radii"] = knn_rep_kan

    if knn_trace_payload is not None:
        results["knn_threshold_trace_example"] = knn_trace_payload
    elif args.plot_knn_threshold_trace:
        print(
            "  [warn] --plot_knn_threshold_trace 已开启但未采集到轨迹："
            f"请确认 --knn_trace_batch={args.knn_trace_batch} 小于实际处理的 batch 数（当前 num_batches={nb}）；"
            f"固定 batch 内多样本用 --knn_trace_samples（索引 < batch_size）；"
            f"跨 batch 累积 N 条用 --knn_trace_max_samples N 并保证 max_batches 足够。"
        )

    for name, xs in time_rows.items():
        if xs:
            results["timing_sec_per_batch"][name] = {
                "mean": float(np.mean(xs)),
                "std": float(np.std(xs)),
                "n_batches_timed": len(xs),
            }

    out_path = os.path.join(args.save_dir, "fused_kan_segfilter_coarsefine_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    plot_base = os.path.join(args.save_dir, "fps_candidate_count_vs_iteration")
    if args.plot_fps_candidate_curve:
        plot_out = plot_fps_candidate_count_vs_iteration(
            plot_base,
            rep_ex,
            rep_kan,
            s0_report,
            args.num_points,
            with_std_band=args.plot_fps_candidate_curve_std,
        )
        if plot_out:
            results["plot_fps_candidate_curve_path"] = plot_out["png"]
            results["plot_fps_candidate_curve_pdf_path"] = plot_out["pdf"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=True)

    knn_plot_base = os.path.join(args.save_dir, "knn_coarse_vs_fps_iteration")
    if args.plot_knn_candidate_curve:
        knn_plot_out = plot_knn_coarse_vs_fps_iteration(
            knn_plot_base,
            knn_rep_ex,
            knn_rep_kan,
            args.num_points,
            with_std_band=args.plot_knn_candidate_curve_std,
            annotate_candidate_stride=args.plot_knn_annotate_stride,
        )
        if knn_plot_out:
            results["plot_knn_coarse_curve_path"] = knn_plot_out["png"]
            results["plot_knn_coarse_curve_pdf_path"] = knn_plot_out["pdf"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=True)

    trace_plot_base = os.path.join(args.save_dir, "knn_coarse_threshold_trace_example")
    if knn_trace_payload is not None and args.plot_knn_threshold_trace:
        ps_tr = knn_trace_payload.get("per_sample_traces") or []
        series_list: List[Tuple[str, List[Tuple[float, int]]]] = []
        for item in ps_tr:
            b = int(item.get("batch_idx", knn_trace_payload.get("batch_idx", 0)))
            lab = f"b{b} s{item['sample_idx']} c{item['center_idx']}"
            pairs = [(float(p[0]), int(p[1])) for p in item["trace_pairs_h_count"]]
            series_list.append((lab, pairs))
        if not series_list and knn_trace_payload.get("trace_pairs_h_count"):
            pairs = [
                (float(p[0]), int(p[1]))
                for p in knn_trace_payload["trace_pairs_h_count"]
            ]
            series_list.append(
                (
                    f"s{knn_trace_payload.get('sample_idx', 0)} "
                    f"c{knn_trace_payload.get('center_idx', 0)}",
                    pairs,
                )
            )
        _overlay_cap = 20
        if len(series_list) > _overlay_cap:
            series_overlay = series_list[:_overlay_cap]
            trace_sub_ov = (
                f"{knn_trace_payload.get('trace_subtitle', '')}  "
                f"overlay: first {_overlay_cap}/{len(series_list)} series"
            )
        else:
            series_overlay = series_list
            trace_sub_ov = knn_trace_payload.get("trace_subtitle", "")
        trace_plot_out = None
        if series_overlay:
            trace_plot_out = plot_knn_coarse_threshold_trace(
                trace_plot_base,
                series_overlay,
                int(knn_trace_payload["knn_coarse_pool_lo"]),
                int(knn_trace_payload["knn_coarse_pool_hi"]),
                {
                    "batch_idx": knn_trace_payload["batch_idx"],
                    "fps_t": knn_trace_payload["fps_t"],
                    "trace_subtitle": trace_sub_ov,
                },
            )
        if trace_plot_out:
            results["plot_knn_threshold_trace_png"] = trace_plot_out["png"]
            results["plot_knn_threshold_trace_pdf"] = trace_plot_out["pdf"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=True)

        if args.plot_knn_threshold_trace_distribution and len(ps_tr) >= 3:
            dist_meta = {
                "knn_trace_batch_start": int(
                    knn_trace_payload.get(
                        "knn_trace_batch_start", knn_trace_payload["batch_idx"]
                    )
                ),
                "fps_t": knn_trace_payload["fps_t"],
            }
            dist_out = plot_knn_coarse_threshold_trace_distribution(
                trace_plot_base,
                ps_tr,
                int(knn_trace_payload["knn_coarse_pool_lo"]),
                int(knn_trace_payload["knn_coarse_pool_hi"]),
                dist_meta,
            )
            if dist_out:
                results["plot_knn_threshold_trace_distribution_png"] = dist_out["png"]
                results["plot_knn_threshold_trace_distribution_pdf"] = dist_out["pdf"]
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=True)

    print("=" * 64)
    print("KAN radii seg_filter + coarse-fine KNN 融合评估")
    print("=" * 64)
    print(
        f"  N={args.num_points} K={args.K} seg_len={args.segment_len} "
        f"k_nn={args.k_neighbors} partition={args.partition} batches={nb}"
    )
    print(
        f"  prefix_ratio={args.prefix_ratio}  kan_prefix_len={kan_prefix_len_from_K(args.K, args.prefix_ratio)}  "
        f"seg_filter_start_t={results['seg_filter_start_t']}  defer={args.defer_seg_filter}  "
        f"kan_input={args.kan_input}"
    )
    fe_m, fe_s = mean_std("fused_exact_radii_mean_iou_fps")
    fe_k_m, fe_k_s = mean_std("fused_exact_radii_mean_knn_iou")
    if fe_m is not None:
        print(f"  [融合 + exact radii_ref] FPS IoU vs exact: {fe_m:.6f} ± {fe_s or 0:.6f}")
        print(f"  [融合 + exact radii_ref] KNN IoU vs exact topk: {fe_k_m:.6f} ± {fe_k_s or 0:.6f}")

    fk_m, fk_s = mean_std("fused_kan_radii_mean_iou_fps")
    fk_k_m, fk_k_s = mean_std("fused_kan_radii_mean_knn_iou")
    if fk_m is not None:
        print(f"  [融合 + KAN radii_ref] FPS IoU vs exact: {fk_m:.6f} ± {fk_s or 0:.6f}")
        print(f"  [融合 + KAN radii_ref] KNN IoU vs exact topk: {fk_k_m:.6f} ± {fk_k_s or 0:.6f}")
        dm, _ = mean_std("iou_fps_drop_kan_vs_exact_radii")
        dk, _ = mean_std("knn_iou_drop_kan_vs_exact_radii")
        if dm is not None:
            print(f"  KAN vs exact radii — FPS IoU 下降(越大越差): {dm:+.6f}")
        if dk is not None:
            print(f"  KAN vs exact radii — KNN IoU 下降: {dk:+.6f}")

    if not args.skip_cf_baseline:
        cfm, _ = mean_std("cf_only_mean_iou_fps")
        ckn, _ = mean_std("cf_only_mean_knn_iou")
        if cfm is not None:
            print(f"  [baseline 仅 coarse-fine KNN，无 seg_filter] FPS IoU vs exact: {cfm:.6f}")
        if ckn is not None:
            print(f"  [baseline] KNN IoU vs exact topk @ vanilla FPS centers: {ckn:.6f}")

    print("-" * 64)
    print("  FPS seg_filter 后 farthest 候选点数（全步平均 / 仅 seg 步；详见 JSON）")
    if rep_ex is not None:
        m_all = float(np.mean(rep_ex["per_step_mean_candidate_count"]))
        print(f"    [exact radii_ref] 每步平均候选数≈ {m_all:.1f} / N={args.num_points}")
        if "seg_filter_only_mean_candidate_count" in rep_ex:
            print(
                f"      仅 t>={s0_report} 的 seg 步: 平均 {rep_ex['seg_filter_only_mean_candidate_count']:.1f}  "
                f"(min {rep_ex['seg_filter_only_min_candidate_count']:.0f}, "
                f"max {rep_ex['seg_filter_only_max_candidate_count']:.0f})"
            )
    if rep_kan is not None:
        m_all = float(np.mean(rep_kan["per_step_mean_candidate_count"]))
        print(f"    [KAN radii_ref] 每步平均候选数≈ {m_all:.1f} / N={args.num_points}")
        if "seg_filter_only_mean_candidate_count" in rep_kan:
            print(
                f"      仅 t>={s0_report} 的 seg 步: 平均 {rep_kan['seg_filter_only_mean_candidate_count']:.1f}  "
                f"(min {rep_kan['seg_filter_only_min_candidate_count']:.0f}, "
                f"max {rep_kan['seg_filter_only_max_candidate_count']:.0f})"
            )

    print("-" * 64)
    print("  KNN coarse 粗筛候选点数（全 K 步跨 batch 平均；详见 JSON knn_coarse_candidate_report_*）")
    if knn_rep_ex is not None:
        m_k = knn_rep_ex.get("mean_knn_coarse_candidate_count_all_steps")
        if m_k is not None:
            mi = knn_rep_ex.get("mean_knn_coarse_threshold_iterations_all_steps")
            mi_tail = (
                f"\n      平均粗筛探测步数≈ {float(mi):.2f}（按 FPS 步聚合的 min≈ "
                f"{knn_rep_ex.get('min_per_step_mean_knn_coarse_threshold_iterations', 0):.2f}, "
                f"max≈ {knn_rep_ex.get('max_per_step_mean_knn_coarse_threshold_iterations', 0):.2f}）"
                if mi is not None
                else ""
            )
            print(
                f"    [exact radii_ref] 每步平均粗筛候选数≈ {float(m_k):.1f} / N={args.num_points}  "
                f"(按步 min≈ {knn_rep_ex.get('min_per_step_mean_knn_coarse_candidate_count', 0):.1f}, "
                f"max≈ {knn_rep_ex.get('max_per_step_mean_knn_coarse_candidate_count', 0):.1f})"
                f"{mi_tail}"
            )
    if knn_rep_kan is not None:
        m_k = knn_rep_kan.get("mean_knn_coarse_candidate_count_all_steps")
        if m_k is not None:
            mi = knn_rep_kan.get("mean_knn_coarse_threshold_iterations_all_steps")
            mi_tail = (
                f"\n      平均粗筛探测步数≈ {float(mi):.2f}（按 FPS 步聚合的 min≈ "
                f"{knn_rep_kan.get('min_per_step_mean_knn_coarse_threshold_iterations', 0):.2f}, "
                f"max≈ {knn_rep_kan.get('max_per_step_mean_knn_coarse_threshold_iterations', 0):.2f}）"
                if mi is not None
                else ""
            )
            print(
                f"    [KAN radii_ref] 每步平均粗筛候选数≈ {float(m_k):.1f} / N={args.num_points}  "
                f"(按步 min≈ {knn_rep_kan.get('min_per_step_mean_knn_coarse_candidate_count', 0):.1f}, "
                f"max≈ {knn_rep_kan.get('max_per_step_mean_knn_coarse_candidate_count', 0):.1f})"
                f"{mi_tail}"
            )

    print("-" * 64)
    print("  耗时 (秒/batch, CUDA 同步, 前 %d 个 batch 平均):" % max(1, args.time_batches))
    for name, xs in time_rows.items():
        if xs:
            print(f"    {name}: {np.mean(xs):.4f} ± {np.std(xs):.4f}")
    print("=" * 64)
    print(f"  JSON: {out_path}")
    if args.plot_fps_candidate_curve and results.get("plot_fps_candidate_curve_path"):
        print(f"  候选曲线图 PNG: {results['plot_fps_candidate_curve_path']}")
        if results.get("plot_fps_candidate_curve_pdf_path"):
            print(f"  候选曲线图 PDF: {results['plot_fps_candidate_curve_pdf_path']}")
    if args.plot_knn_candidate_curve and results.get("plot_knn_coarse_curve_path"):
        print(f"  KNN coarse 曲线 PNG: {results['plot_knn_coarse_curve_path']}")
        if results.get("plot_knn_coarse_curve_pdf_path"):
            print(f"  KNN coarse 曲线 PDF: {results['plot_knn_coarse_curve_pdf_path']}")
    if args.plot_knn_threshold_trace and results.get("plot_knn_threshold_trace_png"):
        print(f"  KNN 阈值轨迹示例 PNG: {results['plot_knn_threshold_trace_png']}")
        if results.get("plot_knn_threshold_trace_pdf"):
            print(f"  KNN 阈值轨迹示例 PDF: {results['plot_knn_threshold_trace_pdf']}")


if __name__ == "__main__":
    main()
