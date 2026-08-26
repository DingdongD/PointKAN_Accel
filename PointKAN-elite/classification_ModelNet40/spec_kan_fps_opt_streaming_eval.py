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
  5. 可选：若提供 KAN 权重，则用 KAN 预测 radii_ref（prefix->KAN->suffix）重跑 seg_filter/streaming，并额外汇总 KAN 的 IOU/耗时/speedup
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
    ModelNet40Wrapper,
    collate_points,
)

from spec_kan_fps_opt_kan import (
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


# =========================================================
# 1. Streaming/block-wise FPS with segment-threshold 
# =========================================================

@torch.no_grad()
def fps_blockwise_segment_threshold(
    X: torch.Tensor,
    K: int,
    seed: str = "first",
    radii_ref: Optional[torch.Tensor] = None,   # (B, K), usually from exact FPS on same sample
    segment_len: int = 32,                      # segment length for threshold sharing
    block_len: int = 64,                        # hardware block size           # "global" | "block_max"
    return_distances_matrix: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Block-wise FPS with segment-threshold + max buffer fine selection.

    核心流程:
        1) 维护 min_dist（点到已选中心集合的最小平方距离）
        2) 每轮 t 根据 segment-threshold 得到 tau
        3) 将 min_dist 按 block_len 分块
        4) 对每个 block 内 md_block >= tau 的候选，追加到 max buffer
        5) 最终仅在 max buffer 中做 argmax，选出 farthest
        6) 更新 min_dist, radii

    参数:
        X:              (B, N, D)
        K:              采样点数
        seed:           "first" | "random" | "max_norm"
        radii_ref:      (B, K)，通常来自 exact FPS 的半径曲线
        segment_len:    一个 segment 内共享 tau
        block_len:      block 大小，模拟硬件按 block 处理

    返回:
        idx:            (B, K) 选中点索引
        radii:          (B, K) 每轮覆盖半径（平方距离）
        distances_matrix:(B, K, N) 每个选中点到所有点的平方距离
        stats:
            candidate_ratio:    (B, K) 通过 threshold 的真实候选比例
            buffer_count:       (B, K) max buffer 实际装入的候选数
            buffer_ratio:       (B, K) buffer_count / N
            block_filtered_count:
                                (B, K, nb) 每步 t 每个 block 内通过 threshold 的候选点数
            used_fallback:      (B, K) 是否触发回退
            tau_used:           (B, K) 当前轮使用的 tau
            block_pass_ratio:   (B, K) 至少有一个候选通过的 block 比例

    注意:
        - 全程使用平方欧氏距离，无 sqrt
        - radii_ref 若看起来像欧氏半径，会自动平方
        - 为保证与 vanilla FPS 对齐，seed="first" 时初始点固定为 0
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    K = min(K, N)

    if radii_ref is None:
        raise ValueError("fps_blockwise_segment_threshold 必须传入 radii_ref（建议来自同一样本的 exact FPS 半径曲线）")
    if radii_ref.shape[:2] != (B, K):
        raise ValueError(f"radii_ref shape should be (B, K) = {(B, K)}, got {tuple(radii_ref.shape)}")

    # Precompute x^2
    x2 = (X * X).sum(dim=-1)  # (B, N)

    # Heuristic: if radii_ref looks like Euclidean radius, square it
    if torch.median(radii_ref) > 0:
        med_x2 = torch.median(x2)
        med_r = torch.median(radii_ref)
        if (med_r * med_r - med_x2).abs() < (med_r - med_x2).abs():
            radii_ref = radii_ref * radii_ref

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=dtype, device=device)
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    distances_matrix = torch.empty((0,), dtype=dtype, device=device)  # lazy alloc
    if return_distances_matrix:
        distances_matrix = torch.zeros((B, K, N), dtype=dtype, device=device)

    # stats
    candidate_ratio = torch.ones((B, K), dtype=dtype, device=device)
    buffer_count = torch.zeros((B, K), dtype=torch.long, device=device)
    buffer_ratio = torch.zeros((B, K), dtype=dtype, device=device)
    used_fallback = torch.zeros((B, K), dtype=torch.bool, device=device)
    tau_used = torch.zeros((B, K), dtype=dtype, device=device)
    block_pass_ratio = torch.ones((B, K), dtype=dtype, device=device)

    BL = max(1, int(block_len))
    nb = (N + BL - 1) // BL
    block_filtered_count = torch.zeros((B, K, nb), dtype=torch.long, device=device)

    # initial seed
    if seed == "first":
        first = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        first = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    def sqdist_to_center(C: torch.Tensor) -> torch.Tensor:
        # C: (B, D)
        c2 = (C * C).sum(dim=-1, keepdim=True)          # (B, 1)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)         # (B, N)
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0)      # (B, N)
        return d2

    def get_segment_tau(t: int) -> torch.Tensor:
        """
        当前步 t 所属 segment 的末尾半径作为 tau
        """
        L = max(1, int(segment_len))
        j = t // L
        seg_end = min((j + 1) * L - 1, K - 1)
        tau = radii_ref[:, seg_end]  # (B,)
        return tau

    def select_farthest_blockwise_maxbuffer(md: torch.Tensor, t: int) -> torch.Tensor:
        tau = get_segment_tau(t)  # (B,)
        tau_used[:, t] = tau

        BL_local = max(1, int(block_len))
        nb_local = (N + BL_local - 1) // BL_local
        padN = nb_local * BL_local

        if padN != N:
            pad = torch.full((B, padN - N), -float("inf"), dtype=dtype, device=device)
            md_pad = torch.cat([md, pad], dim=1)
        else:
            md_pad = md

        md_blk = md_pad.view(B, nb_local, BL_local)          # (B, nb, BL)
        mask_blk = md_blk >= tau.view(B, 1, 1)               # (B, nb, BL)

        # 每个 block 内通过筛选的点数
        blk_pass_counts = mask_blk.sum(dim=2).to(torch.long) # (B, nb)
        block_filtered_count[:, t, :] = blk_pass_counts

        blk_has_any = blk_pass_counts > 0
        pass_block_cnt = blk_has_any.sum(dim=1)
        block_pass_ratio[:, t] = pass_block_cnt.to(dtype) / float(nb_local)

        cand_cnt = blk_pass_counts.sum(dim=1).to(torch.long)
        # 若某个 batch 没有候选，则触发回退（避免 runtime error）
        used = cand_cnt == 0
        used_fallback[:, t] = used
        candidate_ratio[:, t] = cand_cnt.to(dtype) / float(N)
        buffer_count[:, t] = cand_cnt
        buffer_ratio[:, t] = cand_cnt.to(dtype) / float(N)

        # 在所有通过筛选的点中做 fine max
        # 在 block 粒度上找到通过阈值的全局 max（避免额外的 md>=tau 全量 mask）
        masked_blk = torch.where(mask_blk, md_blk, torch.full_like(md_blk, -float("inf")))  # (B,nb,BL)
        blk_vals, blk_pos = masked_blk.max(dim=2)  # (B,nb), (B,nb)
        best_blk = blk_vals.argmax(dim=1)          # (B,)
        best_pos = blk_pos[torch.arange(B, device=device), best_blk]  # (B,)
        best_idx = best_blk * BL_local + best_pos  # (B,)
        # 去掉兜底机制：一旦候选为空，直接报错（不再回退全局 argmax）
        if used.any():
            bad = torch.nonzero(used, as_tuple=False).squeeze(1).tolist()
            raise RuntimeError(
                f"[fps_blockwise_segment_threshold] tau produced empty candidate set at step t={t}. "
                f"empty_batches={bad}"
            )
        farthest = best_idx.clamp_max(N - 1)
        return farthest

    # init
    idx[:, 0] = first
    C0 = X[torch.arange(B, device=device), first]
    d2 = sqdist_to_center(C0)
    if return_distances_matrix:
        distances_matrix[:, 0, :] = d2
    min_dist.copy_(d2)
    radii[:, 0], _ = torch.max(min_dist, dim=1)
    min_dist[torch.arange(B, device=device), first] = 0.0

    # FPS loop
    for t in range(1, K):
        farthest = select_farthest_blockwise_maxbuffer(min_dist, t)
        idx[:, t] = farthest

        C = X[torch.arange(B, device=device), farthest]
        d2 = sqdist_to_center(C)
        if return_distances_matrix:
            distances_matrix[:, t, :] = d2

        torch.minimum(min_dist, d2, out=min_dist)
        radii[:, t], _ = torch.max(min_dist, dim=1)

        # avoid reselection
        min_dist[torch.arange(B, device=device), farthest] = 0.0

    stats = {
        "candidate_ratio": candidate_ratio,
        "buffer_count": buffer_count,
        "buffer_ratio": buffer_ratio,
        "used_fallback": used_fallback,
        "tau_used": tau_used,
        "block_pass_ratio": block_pass_ratio,
        "block_filtered_count": block_filtered_count,
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
    warmup_batches: int = 2,
    kan_save_dir: str = "",
) -> Dict:
    """
    对每批：exact → radii_ref；再分别跑 seg_filter 与 streaming_block_seg_filter，计时并算 IOU。
    返回汇总：
      - 不带 KAN：mean_iou_seg_filter、mean_iou_streaming、time_exact、time_seg_filter、time_streaming、speedup_seg_filter、speedup_streaming、以及两套 stats。
      - 带 KAN：额外包含 mean_iou_seg_filter_kan、mean_iou_streaming_kan、time_seg_filter_kan、time_streaming_kan、以及对应 speedup 与 summary。
    """
    all_ious_seg: List[float] = []
    all_ious_streaming: List[float] = []
    all_cr_seg: List[torch.Tensor] = []
    all_cr_streaming: List[torch.Tensor] = []
    all_fallback_seg: List[torch.Tensor] = []
    all_fallback_streaming: List[torch.Tensor] = []

    all_ious_seg_kan: List[float] = []
    all_ious_streaming_kan: List[float] = []
    all_cr_seg_kan: List[torch.Tensor] = []
    all_cr_streaming_kan: List[torch.Tensor] = []
    all_fallback_seg_kan: List[torch.Tensor] = []
    all_fallback_streaming_kan: List[torch.Tensor] = []

    time_exact = 0.0
    time_seg = 0.0
    time_streaming = 0.0
    time_seg_kan = 0.0
    time_streaming_kan = 0.0
    n_batches = 0

    model_kan = None
    cfg_kan = None
    kan_save_dir = kan_save_dir.strip()
    if kan_save_dir and os.path.isdir(kan_save_dir):
        model_kan, cfg_kan = load_kan_radii_model(kan_save_dir, device)
        assert cfg_kan.fps_steps == K, f"KAN 训练时 fps_steps={cfg_kan.fps_steps}，当前 K={K}，需一致"

    BL = max(1, int(block_len))
    nb = (num_points + BL - 1) // BL

    def _hist_to_quantiles(hist: np.ndarray, quantiles: List[float]) -> Dict[str, float]:
        """hist[k] = count, k is value; return quantile values."""
        total = int(hist.sum())
        if total <= 0:
            return {str(q): 0.0 for q in quantiles}
        cdf = np.cumsum(hist).astype(np.float64)
        out: Dict[str, float] = {}
        for q in quantiles:
            thr = q * total
            idx = int(np.searchsorted(cdf, thr, side="left"))
            idx = min(max(idx, 0), len(hist) - 1)
            out[str(q)] = float(idx)
        return out

    # streaming 分支（非 KAN）
    buffer_hist_streaming = np.zeros(num_points + 1, dtype=np.int64)  # value -> count
    block_hist_streaming = np.zeros((nb, BL + 1), dtype=np.int64)      # block -> (count -> freq)

    # streaming 分支（KAN）
    buffer_hist_streaming_kan = None
    block_hist_streaming_kan = None

    it = tqdm(loader, desc="IOU & timing", ncols=100)

    def compute_iou_batch_fast(
        idx_a: torch.Tensor,
        idx_b: torch.Tensor,
        num_points_local: int,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        纯 torch 版 IOU，避免 compute_iou_batch 里的 Python set 循环。
        idx_a/idx_b: (B, K)
        """
        B, K = idx_a.shape
        device_local = idx_a.device
        mask_a = torch.zeros((B, num_points_local), dtype=torch.bool, device=device_local)
        mask_b = torch.zeros((B, num_points_local), dtype=torch.bool, device=device_local)
        mask_a.scatter_(1, idx_a, True)
        mask_b.scatter_(1, idx_b, True)
        inter = (mask_a & mask_b).sum(dim=1).to(torch.float32)
        union = (mask_a | mask_b).sum(dim=1).to(torch.float32)
        iou = inter / torch.clamp_min(union, 1.0)
        return iou, float(iou.mean().item()), float(iou.std(unbiased=False).item())

    for batch_idx, (pts, _) in enumerate(it):
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)

        # ----- Warmup (skip timing) -----
        if batch_idx < warmup_batches:
            idx_exact, radii_ref, _, _ = fps_adaptive_argmax(pts, K, seed=seed, mode="exact")
            _time_cuda_sync(device)
            fps_adaptive_argmax(pts, K, seed=seed, mode="seg_filter", radii_ref=radii_ref, segment_len=segment_len)
            _time_cuda_sync(device)
            fps_blockwise_segment_threshold(
                pts, K, seed=seed, radii_ref=radii_ref, segment_len=segment_len,
                block_len=block_len,
            )
            _time_cuda_sync(device)

            if model_kan is not None and cfg_kan is not None:
                # KAN：用 radii_ref 做 prefix，预测完整曲线并替代 segment 阈值
                radii_ref_kan = predict_full_radii_squared_from_exact(radii_ref, model_kan, cfg_kan, device)
                fps_adaptive_argmax(
                    pts, K, seed=seed, mode="seg_filter",
                    radii_ref=radii_ref_kan, segment_len=segment_len,
                )
                _time_cuda_sync(device)
                fps_blockwise_segment_threshold(
                    pts, K, seed=seed, radii_ref=radii_ref_kan, segment_len=segment_len,
                    block_len=block_len,
                )
                _time_cuda_sync(device)
            continue

        # ----- Exact + radii_ref -----
        _time_cuda_sync(device)
        t0 = time.perf_counter()
        idx_exact, radii_ref, _, _ = fps_adaptive_argmax(pts, K, seed=seed, mode="exact")
        _time_cuda_sync(device)
        time_exact += time.perf_counter() - t0

        radii_ref_kan = None
        if model_kan is not None and cfg_kan is not None:
            # KAN 预测 radii_ref（仍为平方半径曲线），供 seg_filter / streaming 使用
            radii_ref_kan = predict_full_radii_squared_from_exact(radii_ref, model_kan, cfg_kan, device)

        # ----- seg_filter -----
        _time_cuda_sync(device)
        t0 = time.perf_counter()
        idx_seg, _, _, stats_seg = fps_adaptive_argmax(
            pts, K, seed=seed, mode="seg_filter",
            radii_ref=radii_ref, segment_len=segment_len,
        )
        _time_cuda_sync(device)
        time_seg += time.perf_counter() - t0

        # ----- seg_filter (KAN radii_ref) -----
        if radii_ref_kan is not None:
            _time_cuda_sync(device)
            t0 = time.perf_counter()
            idx_seg_kan, _, _, stats_seg_kan = fps_adaptive_argmax(
                pts, K, seed=seed, mode="seg_filter",
                radii_ref=radii_ref_kan, segment_len=segment_len,
            )
            _time_cuda_sync(device)
            time_seg_kan += time.perf_counter() - t0
        else:
            idx_seg_kan, stats_seg_kan = None, None

        # ----- streaming_block_seg_filter -----
        _time_cuda_sync(device)
        t0 = time.perf_counter()
        idx_stream, _, _, stats_stream = fps_blockwise_segment_threshold(
            pts, K, seed=seed, radii_ref=radii_ref, segment_len=segment_len,
            block_len=block_len,
        )
        _time_cuda_sync(device)
        time_streaming += time.perf_counter() - t0

        # ----- streaming_block_seg_filter (KAN radii_ref) -----
        if radii_ref_kan is not None:
            _time_cuda_sync(device)
            t0 = time.perf_counter()
            idx_stream_kan, _, _, stats_stream_kan = fps_blockwise_segment_threshold(
                pts, K, seed=seed, radii_ref=radii_ref_kan, segment_len=segment_len,
                block_len=block_len,
            )
            _time_cuda_sync(device)
            time_streaming_kan += time.perf_counter() - t0
        else:
            idx_stream_kan, stats_stream_kan = None, None

        # ----- 统计 streaming 的候选过滤与最终 buffer 分布 -----
        # buffer_count：最终 max buffer 中的候选数量（用于细选 argmax），形状 (B, K)
        # 第 0 步是固定 seed 点，不走阈值过滤/候选 buffer 逻辑；分布统计从 t=1 开始更合理
        bc = stats_stream["buffer_count"][:, 1:].detach().to("cpu", dtype=torch.long).numpy()
        buffer_hist_streaming += np.bincount(bc.reshape(-1), minlength=num_points + 1).astype(np.int64)

        bfc = stats_stream["block_filtered_count"][:, 1:].detach().to("cpu", dtype=torch.long).numpy()  # (B,K-1,nb)
        # block_filtered_count 取值范围 <= BL，所以用 (BL+1) 作为直方图轴
        for bidx in range(nb):
            vals = bfc[:, :, bidx].reshape(-1)
            block_hist_streaming[bidx] += np.bincount(vals, minlength=BL + 1).astype(np.int64)

        if stats_stream_kan is not None:
            if buffer_hist_streaming_kan is None:
                buffer_hist_streaming_kan = np.zeros(num_points + 1, dtype=np.int64)
                block_hist_streaming_kan = np.zeros((nb, BL + 1), dtype=np.int64)
            bc_kan = stats_stream_kan["buffer_count"][:, 1:].detach().to("cpu", dtype=torch.long).numpy()
            buffer_hist_streaming_kan += np.bincount(bc_kan.reshape(-1), minlength=num_points + 1).astype(np.int64)

            bfc_kan = stats_stream_kan["block_filtered_count"][:, 1:].detach().to("cpu", dtype=torch.long).numpy()
            for bidx in range(nb):
                vals = bfc_kan[:, :, bidx].reshape(-1)
                block_hist_streaming_kan[bidx] += np.bincount(vals, minlength=BL + 1).astype(np.int64)
            # 这里只做不严格的计数（只用于检查）

        # ----- IOU -----
        iou_seg_per, _, _ = compute_iou_batch_fast(idx_exact, idx_seg, num_points)
        iou_stream_per, _, _ = compute_iou_batch_fast(idx_exact, idx_stream, num_points)
        all_ious_seg.extend(iou_seg_per.cpu().numpy().tolist())
        all_ious_streaming.extend(iou_stream_per.cpu().numpy().tolist())
        all_cr_seg.append(stats_seg["candidate_ratio"])
        all_cr_streaming.append(stats_stream["candidate_ratio"])
        all_fallback_seg.append(stats_seg["used_fallback"])
        all_fallback_streaming.append(stats_stream["used_fallback"])

        if idx_seg_kan is not None and idx_stream_kan is not None:
            iou_seg_kan_per, _, _ = compute_iou_batch_fast(idx_exact, idx_seg_kan, num_points)
            iou_stream_kan_per, _, _ = compute_iou_batch_fast(idx_exact, idx_stream_kan, num_points)
            all_ious_seg_kan.extend(iou_seg_kan_per.cpu().numpy().tolist())
            all_ious_streaming_kan.extend(iou_stream_kan_per.cpu().numpy().tolist())
            all_cr_seg_kan.append(stats_seg_kan["candidate_ratio"])
            all_cr_streaming_kan.append(stats_stream_kan["candidate_ratio"])
            all_fallback_seg_kan.append(stats_seg_kan["used_fallback"])
            all_fallback_streaming_kan.append(stats_stream_kan["used_fallback"])

        n_batches += 1

    mean_iou_seg = float(np.mean(all_ious_seg))
    std_iou_seg = float(np.std(all_ious_seg))
    mean_iou_streaming = float(np.mean(all_ious_streaming))
    std_iou_streaming = float(np.std(all_ious_streaming))

    # Speedup: exact 为基准，seg_filter / streaming 的加速比 = time_exact / time_*
    speedup_seg = time_exact / time_seg if time_seg > 0 else 0.0
    speedup_streaming = time_exact / time_streaming if time_streaming > 0 else 0.0

    # ----- 分布摘要：streaming buffer 与每 block 候选 -----
    buffer_streaming_mean = float((np.arange(num_points + 1, dtype=np.float64) * buffer_hist_streaming).sum() / max(1, buffer_hist_streaming.sum()))
    buffer_streaming_quantiles = _hist_to_quantiles(buffer_hist_streaming, [0.5, 0.75, 0.9, 0.99])
    buffer_streaming_mode = int(np.argmax(buffer_hist_streaming))

    block_streaming_mean_by_block: List[float] = []
    block_streaming_mode_by_block: List[int] = []
    for bidx in range(nb):
        hist_b = block_hist_streaming[bidx]
        total_b = int(hist_b.sum())
        if total_b <= 0:
            block_streaming_mean_by_block.append(0.0)
            block_streaming_mode_by_block.append(0)
            continue
        mean_b = float((np.arange(BL + 1, dtype=np.float64) * hist_b).sum() / total_b)
        mode_b = int(np.argmax(hist_b))
        block_streaming_mean_by_block.append(mean_b)
        block_streaming_mode_by_block.append(mode_b)

    # KAN 分支（如果启用）
    block_streaming_mean_by_block_kan: Optional[List[float]] = None
    block_streaming_mode_by_block_kan: Optional[List[int]] = None
    if block_hist_streaming_kan is not None:
        block_streaming_mean_by_block_kan = []
        block_streaming_mode_by_block_kan = []
        for bidx in range(nb):
            hist_b = block_hist_streaming_kan[bidx]
            total_b = int(hist_b.sum())
            if total_b <= 0:
                block_streaming_mean_by_block_kan.append(0.0)
                block_streaming_mode_by_block_kan.append(0)
                continue
            mean_b = float((np.arange(BL + 1, dtype=np.float64) * hist_b).sum() / total_b)
            mode_b = int(np.argmax(hist_b))
            block_streaming_mean_by_block_kan.append(mean_b)
            block_streaming_mode_by_block_kan.append(mode_b)

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

    if all_ious_seg_kan:
        mean_iou_seg_kan = float(np.mean(all_ious_seg_kan))
        std_iou_seg_kan = float(np.std(all_ious_seg_kan))
        mean_iou_streaming_kan = float(np.mean(all_ious_streaming_kan))
        std_iou_streaming_kan = float(np.std(all_ious_streaming_kan))
        speedup_seg_kan = time_exact / time_seg_kan if time_seg_kan > 0 else 0.0
        speedup_streaming_kan = time_exact / time_streaming_kan if time_streaming_kan > 0 else 0.0
        summary_seg_kan = _scan_stats(all_cr_seg_kan, all_fallback_seg_kan)
        summary_streaming_kan = _scan_stats(all_cr_streaming_kan, all_fallback_streaming_kan)
    else:
        mean_iou_seg_kan = None
        std_iou_seg_kan = None
        mean_iou_streaming_kan = None
        std_iou_streaming_kan = None
        speedup_seg_kan = None
        speedup_streaming_kan = None
        summary_seg_kan = None
        summary_streaming_kan = None

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
        "mean_iou_seg_filter_kan": mean_iou_seg_kan,
        "std_iou_seg_filter_kan": std_iou_seg_kan,
        "mean_iou_streaming_kan": mean_iou_streaming_kan,
        "std_iou_streaming_kan": std_iou_streaming_kan,
        "time_seg_filter_kan_sec": time_seg_kan if mean_iou_seg_kan is not None else None,
        "time_streaming_kan_sec": time_streaming_kan if mean_iou_streaming_kan is not None else None,
        "speedup_seg_filter_kan": speedup_seg_kan,
        "speedup_streaming_kan": speedup_streaming_kan,
        "summary_seg_filter_kan": summary_seg_kan,
        "summary_streaming_kan": summary_streaming_kan,
        # streaming 分布统计
        "streaming_buffer_count_hist": buffer_hist_streaming,
        "streaming_buffer_count_mean": buffer_streaming_mean,
        "streaming_buffer_count_mode": buffer_streaming_mode,
        "streaming_buffer_count_quantiles": buffer_streaming_quantiles,
        "streaming_block_filtered_count_hist": block_hist_streaming,
        "streaming_block_filtered_count_mean_by_block": block_streaming_mean_by_block,
        "streaming_block_filtered_count_mode_by_block": block_streaming_mode_by_block,
        # KAN 分支分布（若启用）
        "streaming_buffer_count_hist_kan": buffer_hist_streaming_kan,
        "streaming_buffer_count_mean_kan": None if buffer_hist_streaming_kan is None else float((np.arange(num_points + 1, dtype=np.float64) * buffer_hist_streaming_kan).sum() / max(1, buffer_hist_streaming_kan.sum())),
        "streaming_buffer_count_mode_kan": None if buffer_hist_streaming_kan is None else int(np.argmax(buffer_hist_streaming_kan)),
        "streaming_buffer_count_quantiles_kan": None if buffer_hist_streaming_kan is None else _hist_to_quantiles(buffer_hist_streaming_kan, [0.5, 0.75, 0.9, 0.99]),
        "streaming_block_filtered_count_hist_kan": block_hist_streaming_kan,
        "streaming_block_filtered_count_mean_by_block_kan": block_streaming_mean_by_block_kan,
        "streaming_block_filtered_count_mode_by_block_kan": block_streaming_mode_by_block_kan,
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
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=str, default="first")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./runs/fps_seg_filter_streaming_eval")
    p.add_argument("--warmup_batches", type=int, default=2, help="前几批不计时，用于 warmup")
    p.add_argument("--kan_save_dir", type=str, default="",
                   help="KAN 权重目录，如 ./runs/fps_radii_kan_K512_pr0.10（有则启用 KAN radii_ref 分支）")
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
        warmup_batches=args.warmup_batches,
        kan_save_dir=args.kan_save_dir,
    )

    # 写入 JSON（可序列化：去掉 summary 里的 tensor 等）
    results_json = {
        "partition": args.partition,
        "num_points": args.num_points,
        "K": args.K,
        "segment_len": args.segment_len,
        "block_len": args.block_len,
        "seed": seed,
        "num_samples": result["num_samples"],
        "n_batches_timed": result["n_batches_timed"],
        "mean_iou_seg_filter": result["mean_iou_seg_filter"],
        "std_iou_seg_filter": result["std_iou_seg_filter"],
        "mean_iou_streaming": result["mean_iou_streaming"],
        "std_iou_streaming": result["std_iou_streaming"],
        "mean_iou_seg_filter_kan": result["mean_iou_seg_filter_kan"],
        "std_iou_seg_filter_kan": result["std_iou_seg_filter_kan"],
        "mean_iou_streaming_kan": result["mean_iou_streaming_kan"],
        "std_iou_streaming_kan": result["std_iou_streaming_kan"],
        "time_exact_sec": result["time_exact_sec"],
        "time_seg_filter_sec": result["time_seg_filter_sec"],
        "time_streaming_sec": result["time_streaming_sec"],
        "speedup_seg_filter": result["speedup_seg_filter"],
        "speedup_streaming": result["speedup_streaming"],
        "time_seg_filter_kan_sec": result["time_seg_filter_kan_sec"],
        "time_streaming_kan_sec": result["time_streaming_kan_sec"],
        "speedup_seg_filter_kan": result["speedup_seg_filter_kan"],
        "speedup_streaming_kan": result["speedup_streaming_kan"],
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
        "streaming_buffer_count_distribution": {
            "hist": result["streaming_buffer_count_hist"].tolist(),
            "mean": result["streaming_buffer_count_mean"],
            "mode": result["streaming_buffer_count_mode"],
            "quantiles": result["streaming_buffer_count_quantiles"],
        },
        "streaming_block_filtered_count_distribution": {
            "block_len": args.block_len,
            "nb": (args.num_points + max(1, args.block_len) - 1) // max(1, args.block_len),
            "hist": result["streaming_block_filtered_count_hist"].tolist(),  # shape: (nb, BL+1)
            "mean_by_block": result["streaming_block_filtered_count_mean_by_block"],
            "mode_by_block": result["streaming_block_filtered_count_mode_by_block"],
        },
    }

    # 追加 KAN summary（如启用）
    if result.get("summary_seg_filter_kan") is not None:
        results_json["summary_seg_filter_kan"] = {
            "candidate_ratio_mean": result["summary_seg_filter_kan"]["candidate_ratio_mean"],
            "candidate_ratio_std": result["summary_seg_filter_kan"]["candidate_ratio_std"],
            "used_fallback_ratio": result["summary_seg_filter_kan"]["used_fallback_ratio"],
            "total_scan_vanilla": result["summary_seg_filter_kan"]["total_scan_vanilla"],
            "total_scan_filtered": result["summary_seg_filter_kan"]["total_scan_filtered"],
            "scan_saved_ratio": result["summary_seg_filter_kan"]["scan_saved_ratio"],
        }
        results_json["summary_streaming_kan"] = {
            "candidate_ratio_mean": result["summary_streaming_kan"]["candidate_ratio_mean"],
            "candidate_ratio_std": result["summary_streaming_kan"]["candidate_ratio_std"],
            "used_fallback_ratio": result["summary_streaming_kan"]["used_fallback_ratio"],
            "total_scan_vanilla": result["summary_streaming_kan"]["total_scan_vanilla"],
            "total_scan_filtered": result["summary_streaming_kan"]["total_scan_filtered"],
            "scan_saved_ratio": result["summary_streaming_kan"]["scan_saved_ratio"],
        }
        results_json["streaming_buffer_count_distribution_kan"] = None if result["streaming_buffer_count_hist_kan"] is None else {
            "hist": result["streaming_buffer_count_hist_kan"].tolist(),
            "mean": result["streaming_buffer_count_mean_kan"],
            "mode": result["streaming_buffer_count_mode_kan"],
            "quantiles": result["streaming_buffer_count_quantiles_kan"],
        }
        results_json["streaming_block_filtered_count_distribution_kan"] = None if result["streaming_block_filtered_count_hist_kan"] is None else {
            "block_len": args.block_len,
            "nb": (args.num_points + max(1, args.block_len) - 1) // max(1, args.block_len),
            "hist": result["streaming_block_filtered_count_hist_kan"].tolist(),
            "mean_by_block": result["streaming_block_filtered_count_mean_by_block_kan"],
            "mode_by_block": result["streaming_block_filtered_count_mode_by_block_kan"],
        }
    out_path = os.path.join(args.save_dir, "streaming_vs_seg_filter_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # 控制台输出
    print("=" * 70)
    print("Streaming block seg_filter vs seg_filter：IOU & 性能 / speedup 分析 (ModelNet40)")
    print("=" * 70)
    print(f"  分区: {args.partition}  N={args.num_points}  K={args.K}  segment_len={args.segment_len}")
    print(f"  block_len={args.block_len}  seed={seed}")
    print(f"  计时的 batch 数: {result['n_batches_timed']}  样本数: {result['num_samples']}")
    print("-" * 70)
    print("IOU (vs exact FPS):")
    print(f"  seg_filter:   mean_iou = {result['mean_iou_seg_filter']:.6f} ± {result['std_iou_seg_filter']:.6f}")
    print(f"  streaming:    mean_iou = {result['mean_iou_streaming']:.6f} ± {result['std_iou_streaming']:.6f}")
    if result.get("mean_iou_seg_filter_kan") is not None:
        print(f"  seg_filter_kan:   mean_iou = {result['mean_iou_seg_filter_kan']:.6f} ± {result['std_iou_seg_filter_kan']:.6f}")
        print(f"  streaming_kan:    mean_iou = {result['mean_iou_streaming_kan']:.6f} ± {result['std_iou_streaming_kan']:.6f}")
    print("-" * 70)
    print("耗时 (秒):")
    print(f"  exact:      {result['time_exact_sec']:.4f}")
    print(f"  seg_filter: {result['time_seg_filter_sec']:.4f}  (speedup vs exact: {result['speedup_seg_filter']:.4f}x)")
    print(f"  streaming:  {result['time_streaming_sec']:.4f}  (speedup vs exact: {result['speedup_streaming']:.4f}x)")
    if result.get("mean_iou_seg_filter_kan") is not None:
        print(f"  seg_filter_kan: {result['time_seg_filter_kan_sec']:.4f}  (speedup vs exact: {result['speedup_seg_filter_kan']:.4f}x)")
        print(f"  streaming_kan:  {result['time_streaming_kan_sec']:.4f}  (speedup vs exact: {result['speedup_streaming_kan']:.4f}x)")
    print("-" * 70)
    print("候选/扫描统计:")
    s_seg = result["summary_seg_filter"]
    s_str = result["summary_streaming"]
    print(f"  seg_filter:  candidate_ratio = {s_seg['candidate_ratio_mean']:.4f}  scan_saved = {s_seg['scan_saved_ratio']:.4f}  fallback_ratio = {s_seg['used_fallback_ratio']:.4f}")
    print(f"  streaming:   candidate_ratio = {s_str['candidate_ratio_mean']:.4f}  scan_saved = {s_str['scan_saved_ratio']:.4f}  fallback_ratio = {s_str['used_fallback_ratio']:.4f}")
    if result.get("summary_seg_filter_kan") is not None:
        s_seg_kan = result["summary_seg_filter_kan"]
        s_str_kan = result["summary_streaming_kan"]
        print(f"  seg_filter_kan:  candidate_ratio = {s_seg_kan['candidate_ratio_mean']:.4f}  scan_saved = {s_seg_kan['scan_saved_ratio']:.4f}  fallback_ratio = {s_seg_kan['used_fallback_ratio']:.4f}")
        print(f"  streaming_kan:   candidate_ratio = {s_str_kan['candidate_ratio_mean']:.4f}  scan_saved = {s_str_kan['scan_saved_ratio']:.4f}  fallback_ratio = {s_str_kan['used_fallback_ratio']:.4f}")

    print("-" * 70)
    print("streaming 候选分布摘要（基于 buffer_count 与 block_filtered_count）")
    print(f"  buffer_count: mean={result['streaming_buffer_count_mean']:.2f}  mode={result['streaming_buffer_count_mode']}  q50={float(result['streaming_buffer_count_quantiles']['0.5']):.0f}  q75={float(result['streaming_buffer_count_quantiles']['0.75']):.0f}  q90={float(result['streaming_buffer_count_quantiles']['0.9']):.0f}  q99={float(result['streaming_buffer_count_quantiles']['0.99']):.0f}")
    print(f"  per-block filtered candidate mean (len={len(result['streaming_block_filtered_count_mean_by_block'])}): {result['streaming_block_filtered_count_mean_by_block']}")
    if result.get("summary_seg_filter_kan") is not None and result.get("streaming_buffer_count_hist_kan") is not None:
        print(f"  buffer_count (KAN): mean={result['streaming_buffer_count_mean_kan']:.2f}  mode={result['streaming_buffer_count_mode_kan']}  q50={float(result['streaming_buffer_count_quantiles_kan']['0.5']):.0f}  q75={float(result['streaming_buffer_count_quantiles_kan']['0.75']):.0f}  q90={float(result['streaming_buffer_count_quantiles_kan']['0.9']):.0f}  q99={float(result['streaming_buffer_count_quantiles_kan']['0.99']):.0f}")
        print(f"  per-block filtered candidate mean (KAN): {result['streaming_block_filtered_count_mean_by_block_kan']}")
    print("=" * 70)
    print(f"  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
