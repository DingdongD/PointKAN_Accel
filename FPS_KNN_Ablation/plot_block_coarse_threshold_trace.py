#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单样本：按 block 粗筛时记录「阈值迭代 + 块内累积」过程中候选数 M 的轨迹，并绘图。

说明（与 spec_kan_fps.fps_with_knn_coarse_fine_fused_blockwise 对齐）：
  - 当前实现：若 M < min_candidates 则放大阈值，直到 M >= min_candidates 或耗尽 max_iter，
    再在候选集上做 topk 细筛。
  - 若采用「细筛仅当 k <= M <= min_candidates」的严格语义（细筛输入有上界），
    本脚本用阈值二分 + 线性细搜记录轨迹（见 --plot_fine_cap_bisect）。

输出：
  - PNG：轨迹曲线
  - JSON：逐步数值，便于复现
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for d in (SCRIPT_DIR, REPO_ROOT, MODELNET40_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torch.utils.data import DataLoader

from spec_kan_fps import collate_points
from run_knn_coarse_fine_eval_kitti import KITTIPointCloudWrapper


@dataclass
class StepRecord:
    phase: str  # "threshold_round" | "block_partial"
    round_idx: int
    block_idx: Optional[int]
    threshold: float
    M: int
    done_current_policy: bool


def _fps_distances_one_center(X: torch.Tensor, t_fps: int, farthest_seq: torch.Tensor) -> torch.Tensor:
    """X: [1,N,3]，返回第 t_fps 步时 FPS 中心到全场的距离 [N]（与 fused_blockwise 一致）。"""
    assert X.shape[0] == 1
    x2 = (X * X).sum(dim=-1).squeeze(0)
    cidx = int(farthest_seq[t_fps].item())
    C = X[0, cidx]
    c2 = (C * C).sum()
    dot = (X[0] * C).sum(dim=-1)
    d = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()
    return d


def _fps_farthest_sequence(X: torch.Tensor, K: int, seed: str = "first") -> torch.Tensor:
    """返回每步 FPS 选中的 farthest 索引 [K]（batch=1）。"""
    assert X.ndim == 3 and X.shape[0] == 1
    N = X.shape[1]
    device, dtype = X.device, X.dtype
    K = min(K, N)
    x2 = (X * X).sum(dim=-1).squeeze(0)
    min_dist = torch.full((N,), float("inf"), dtype=dtype, device=device)
    seq = torch.zeros(K, dtype=torch.long, device=device)

    if seed == "first":
        first = 0
    elif seed == "random":
        first = int(torch.randint(0, N, (1,), device=device).item())
    elif seed == "max_norm":
        first = int(torch.argmax(x2).item())
    else:
        raise ValueError(seed)

    seq[0] = first
    C = X[0, first]
    c2 = (C * C).sum()
    dot = (X[0] * C).sum(dim=-1)
    d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()
    min_dist.copy_(d2)
    min_dist[first] = 0.0

    for t in range(1, K):
        farthest = int(torch.argmax(min_dist).item())
        seq[t] = farthest
        C = X[0, farthest]
        c2 = (C * C).sum()
        dot = (X[0] * C).sum(dim=-1)
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()
        torch.minimum(min_dist, d2, out=min_dist)
        min_dist[farthest] = 0.0
    return seq


@torch.no_grad()
def trace_blockwise_expand(
    d: torch.Tensor,
    init_threshold: float,
    max_iter: int,
    min_candidates: int,
    block_size: int,
) -> Tuple[List[StepRecord], float, int]:
    """
    与 fused_blockwise 一致：M < min_candidates 时 thr *= 1.1，按块累积 M。
    返回 (records, final_threshold, final_M)
    """
    N = d.numel()
    device, dtype = d.device, d.dtype
    thr = float(init_threshold)
    records: List[StepRecord] = []
    final_M = 0

    for r in range(max_iter):
        mask_acc = torch.zeros(N, dtype=torch.bool, device=device)
        nblk = (N + block_size - 1) // block_size
        for bi in range(nblk):
            start = bi * block_size
            end = min(start + block_size, N)
            blk = d[start:end]
            mblk = blk < thr
            mask_acc[start:end] |= mblk
            M = int(mask_acc.sum().item())
            records.append(
                StepRecord(
                    phase="block_partial",
                    round_idx=r,
                    block_idx=bi,
                    threshold=thr,
                    M=M,
                    done_current_policy=(M >= min_candidates),
                )
            )
        M = int(mask_acc.sum().item())
        final_M = M
        records.append(
            StepRecord(
                phase="threshold_round_end",
                round_idx=r,
                block_idx=None,
                threshold=thr,
                M=M,
                done_current_policy=(M >= min_candidates),
            )
        )
        if M >= min_candidates:
            break
        thr = thr * 1.1

    return records, thr, final_M


def _count_in_ball_blockwise(
    d: torch.Tensor, thr: float, block_size: int
) -> Tuple[int, torch.Tensor]:
    """返回 M = |{i : d_i < thr}| 与完整 mask（block 顺序与 spec 一致）。"""
    N = d.numel()
    device = d.device
    mask_acc = torch.zeros(N, dtype=torch.bool, device=device)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        mask_acc[start:end] |= d[start:end] < thr
    return int(mask_acc.sum().item()), mask_acc


@torch.no_grad()
def trace_bisect_for_fine_cap(
    d: torch.Tensor,
    min_candidates: int,
    k_neighbors: int,
    block_size: int,
    max_bisect_steps: int = 40,
) -> Tuple[List[StepRecord], float, int]:
    """
    严格语义：仅当 k <= M <= min_candidates 时才允许触发细筛（细筛输入规模有上界）。
    对阈值做二分：在 [0, d_max] 上寻找使 M 落入 [k, min_c] 的 thr；并记录每步 (thr, M)。
    若不存在（离散距离导致空隙），取最后一次使 M >= k 且尽可能小的 M（仍记录轨迹）。
    """
    hi_b = float(d.detach().float().max().item() * 1.001 + 1e-6)
    lo_b = 0.0
    records: List[StepRecord] = []

    def M_at(t: float) -> int:
        return _count_in_ball_blockwise(d, t, block_size)[0]

    M_hi = M_at(hi_b)
    if M_hi < k_neighbors:
        return records, hi_b, M_hi

    best_thr, best_M = hi_b, M_hi
    for r in range(max_bisect_steps):
        mid = 0.5 * (lo_b + hi_b)
        M_mid = M_at(mid)
        records.append(
            StepRecord(
                phase="bisect_step",
                round_idx=r,
                block_idx=None,
                threshold=mid,
                M=M_mid,
                done_current_policy=(k_neighbors <= M_mid <= min_candidates),
            )
        )
        if k_neighbors <= M_mid <= min_candidates:
            best_thr, best_M = mid, M_mid
            break
        if M_mid > min_candidates:
            hi_b = mid
        else:
            lo_b = mid
        if hi_b - lo_b < 1e-7 * max(hi_b, 1.0):
            break

    if not (k_neighbors <= best_M <= min_candidates):
        thr_lin = torch.linspace(lo_b, hi_b, steps=min(500, int(d.numel())), device=d.device, dtype=d.dtype)
        for i, t in enumerate(thr_lin):
            Mv = M_at(float(t.item()))
            if k_neighbors <= Mv <= min_candidates:
                best_thr, best_M = float(t.item()), Mv
                records.append(
                    StepRecord(
                        phase="linear_refine",
                        round_idx=max_bisect_steps + i,
                        block_idx=None,
                        threshold=best_thr,
                        M=best_M,
                        done_current_policy=True,
                    )
                )
                break

    return records, best_thr, best_M


def records_to_jsonable(records: List[StepRecord]) -> List[Dict[str, Any]]:
    return [asdict(s) for s in records]


def plot_traces(
    expand_recs: List[StepRecord],
    shrink_recs: Optional[List[StepRecord]],
    min_c: int,
    k: int,
    out_png: str,
    title: str,
):
    import matplotlib.pyplot as plt

    def series_round_ends(recs: List[StepRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, th = [], [], []
        step = 0
        for s in recs:
            if s.phase.startswith("threshold_round_end"):
                xs.append(step)
                ys.append(s.M)
                th.append(s.threshold)
                step += 1
        return np.array(xs), np.array(ys), np.array(th)

    def series_all_blocks(recs: List[StepRecord]) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i, s in enumerate(recs):
            if s.phase in ("block_partial", "block_partial_shrink"):
                xs.append(i)
                ys.append(s.M)
        return np.array(xs), np.array(ys)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(title, fontsize=11)

    # 上图：每轮阈值迭代结束时的 M（与 spec 一致：目标 M >= min_c）
    ax0 = axes[0]
    x0, y0, t0 = series_round_ends(expand_recs)
    ax0.plot(x0, y0, "o-", label="M @ round end (expand policy, spec)", color="C0")
    ax0.axhline(min_c, color="C2", ls="--", lw=1, label=f"min_candidates={min_c}")
    ax0.axhline(k, color="C3", ls=":", lw=1, label=f"k_neighbors={k}")
    ax0.set_ylabel("candidate count M")
    ax0.set_xlabel("threshold round index")
    ax0.legend(loc="best", fontsize=8)
    ax0.set_title("Spec: expand thr until M >= min_candidates, then fine topk")

    ax0b = ax0.twinx()
    ax0b.plot(x0, t0, "x--", color="C1", alpha=0.7, label="threshold (m)")
    ax0b.set_ylabel("threshold (m)", color="C1")
    ax0b.tick_params(axis="y", labelcolor="C1")

    # 下图：块级累积曲线（仅 expand 轨迹，取第一轮或全长）
    ax1 = axes[1]
    xb, yb = series_all_blocks(expand_recs)
    ax1.plot(xb, yb, lw=0.8, label="M after each block (within rounds)", color="C4")
    ax1.axhline(min_c, color="C2", ls="--", lw=1)
    ax1.axhline(k, color="C3", ls=":", lw=1)
    ax1.set_xlabel("block step index (all rounds concatenated)")
    ax1.set_ylabel("M (cumulative in round)")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_title("Block scan: cumulative M within each threshold round (non-decreasing per round)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    if shrink_recs:
        fig2, ax = plt.subplots(1, 1, figsize=(9, 4))
        xs, ys, ths = [], [], []
        for i, s in enumerate(shrink_recs):
            if s.phase in ("bisect_step", "linear_refine"):
                xs.append(i)
                ys.append(s.M)
                ths.append(s.threshold)
        xs, ys, ths = np.array(xs), np.array(ys), np.array(ths)
        ax.plot(xs, ys, "o-", label="M (bisect thr for fine-cap)", color="C0")
        ax.axhline(min_c, color="C2", ls="--", label=f"min_candidates cap={min_c}")
        ax.axhline(k, color="C3", ls=":", label=f"k_neighbors floor={k}")
        if xs.size:
            ax.fill_between(xs, k, min_c, alpha=0.08, color="gray", label="target band [k, min_c]")
        ax.set_xlabel("bisect / refine step index")
        ax.set_ylabel("M")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Strict cap: find thr s.t. k <= M <= min_c (then fine topk is bounded)")
        ax2 = ax.twinx()
        if xs.size:
            ax2.plot(xs, ths, "x--", color="C1", alpha=0.7)
        ax2.set_ylabel("threshold (m)", color="C1")
        plt.tight_layout()
        base, ext = os.path.splitext(out_png)
        p2 = base + "_fine_cap_bisect" + ext
        plt.savefig(p2, dpi=150)
        plt.close()
        return p2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/workspace/VoxelNet/KITTI_Mini/training/velodyne")
    ap.add_argument("--num_points", type=int, default=50000)
    ap.add_argument("--fps_steps", type=int, default=512, help="用于算 FPS 序列；轨迹只画其中一个 center")
    ap.add_argument("--center_t", type=int, default=256, help="选用第几个 FPS 步的中心（0..K-1）")
    ap.add_argument("--k_neighbors", type=int, default=32)
    ap.add_argument("--min_candidates_ratio", type=float, default=1.1)
    ap.add_argument("--init_threshold", type=float, default=2.4)
    ap.add_argument("--max_iter", type=int, default=10)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_png", type=str, default=os.path.join(SCRIPT_DIR, "block_coarse_threshold_trace.png"))
    ap.add_argument("--out_json", type=str, default=os.path.join(SCRIPT_DIR, "block_coarse_threshold_trace.json"))
    ap.add_argument(
        "--plot_fine_cap_bisect",
        action="store_true",
        help="另存一图：二分搜索阈值使 k<=M<=min_candidates（细筛输入有上界时的严格语义）",
    )
    ap.add_argument(
        "--full_trace",
        action="store_true",
        help="JSON 中写入每个 block 的轨迹（默认只写每轮 threshold 结束点，体积小）",
    )
    args = ap.parse_args()

    k = args.k_neighbors
    min_c = max(k + 1, int(k * args.min_candidates_ratio))

    ds = KITTIPointCloudWrapper(
        args.data_dir, num_points=args.num_points, do_roi=True, max_frames=1, seed=42
    )
    loader = DataLoader(ds, batch_size=1, collate_fn=collate_points)
    pts, _ = next(iter(loader))
    device = torch.device(args.device)
    pts = pts.to(device, dtype=torch.float32)

    seq = _fps_farthest_sequence(pts, min(args.fps_steps, args.num_points), seed="first")
    t = max(0, min(args.center_t, seq.numel() - 1))
    d = _fps_distances_one_center(pts, t, seq)

    expand_recs, thr_f, M_f = trace_blockwise_expand(
        d, args.init_threshold, args.max_iter, min_c, args.block_size
    )

    expand_compact = [asdict(s) for s in expand_recs if s.phase == "threshold_round_end"]
    payload: Dict[str, Any] = {
        "meta": {
            "min_candidates": min_c,
            "k_neighbors": k,
            "min_candidates_ratio": args.min_candidates_ratio,
            "init_threshold": args.init_threshold,
            "max_iter": args.max_iter,
            "block_size": args.block_size,
            "fps_center_step_t": t,
            "final_threshold_expand": thr_f,
            "final_M_expand": M_f,
            "note_spec": "spec: coarse stops when M >= min_candidates; fine = topk on masked candidates",
            "note_strict_cap": "bisect: find thr with k <= M <= min_candidates (bounded fine input)",
        },
        "expand_trace_round_ends": expand_compact,
    }
    if args.full_trace:
        payload["expand_trace_all_blocks"] = records_to_jsonable(expand_recs)

    shrink_path = None
    if args.plot_fine_cap_bisect:
        shrink_recs, thr_s, M_s = trace_bisect_for_fine_cap(
            d,
            min_candidates=min_c,
            k_neighbors=k,
            block_size=args.block_size,
        )
        payload["fine_cap_bisect_trace"] = records_to_jsonable(shrink_recs)
        payload["meta"]["final_threshold_fine_cap"] = thr_s
        payload["meta"]["final_M_fine_cap"] = M_s
        payload["meta"]["fine_cap_feasible"] = bool(k <= M_s <= min_c)
        shrink_path = plot_traces(
            expand_recs,
            shrink_recs,
            min_c,
            k,
            args.out_png,
            title=f"KITTI 1 frame | FPS center step t={t} | N={args.num_points} | block={args.block_size}",
        )
    else:
        plot_traces(
            expand_recs,
            None,
            min_c,
            k,
            args.out_png,
            title=f"KITTI 1 frame | FPS center step t={t} | N={args.num_points} | block={args.block_size}",
        )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("Wrote", args.out_png)
    print("Wrote", args.out_json)
    if shrink_path:
        print("Wrote", shrink_path)


if __name__ == "__main__":
    main()
