#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 fps_with_knn_coarse_fine_fused 的阈值迭代次数分布（是否耗尽 max_iter）。

说明：
- 这里复刻当前 ScanObjectNN fused 实现里的 coarse-fine 阈值循环逻辑（每个 center 一个 threshold loop）。
- 只统计 iters_used / final_threshold / final_candidate_count，不生成 knn_idx，降低开销。
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from ScanObjectNN import ScanObjectNN  # noqa: E402


@torch.no_grad()
def analyze_iters(
    X: torch.Tensor,  # [B, N, 3]
    K: int,
    *,
    k_neighbors: int,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    seed: str = "first",
    exclude_self: bool = False,
) -> Dict[str, torch.Tensor]:
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    K = min(int(K), int(N))
    k_actual = min(int(k_neighbors), int(N))
    min_candidates = max(k_actual + 1, int(k_actual * float(min_candidates_ratio)))

    x2 = (X * X).sum(dim=-1)  # [B, N]
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    iters_used = torch.zeros((B, K), dtype=torch.int32, device=device)
    final_threshold = torch.zeros((B, K), dtype=dtype, device=device)
    final_candidate_count = torch.zeros((B, K), dtype=torch.int32, device=device)

    for t in range(K):
        # center C: [B, D]
        C = X[batch_arange, farthest]
        d = torch.cdist(X, C.unsqueeze(1), p=2).squeeze(-1)  # [B, N]
        if exclude_self:
            d = d.clone()
            d[batch_arange, farthest] = float("inf")

        threshold = torch.full((B,), float(init_threshold), dtype=dtype, device=device)
        done = torch.zeros((B,), dtype=torch.bool, device=device)

        # Bidirectional threshold adjustment (matches current fused impl)
        gamma = 1.1
        mask = None
        for it in range(int(max_iter)):
            mask = d < threshold.unsqueeze(-1)
            count = mask.sum(dim=1)  # [B]
            too_few = count <= int(k_neighbors)
            too_many = count > int(min_candidates)
            done = ~(too_few | too_many)
            if done.all():
                iters_used[:, t] = int(it + 1)
                final_threshold[:, t] = threshold
                final_candidate_count[:, t] = count.to(torch.int32)
                break
            threshold = torch.where(too_few, threshold * gamma, threshold)
            threshold = torch.where(too_many, threshold / gamma, threshold)
            if it == int(max_iter) - 1:
                iters_used[:, t] = int(max_iter)
                final_threshold[:, t] = threshold
                if mask is None:
                    mask = d < threshold.unsqueeze(-1)
                final_candidate_count[:, t] = (d < threshold.unsqueeze(-1)).sum(dim=1).to(torch.int32)

        # FPS min_dist update (same as fused impl)
        torch.minimum(min_dist, d, out=min_dist)
        min_dist[batch_arange, farthest] = 0.0
        if t < K - 1:
            farthest = torch.argmax(min_dist, dim=1)

    return {
        "iters_used": iters_used,  # (B,K)
        "final_threshold": final_threshold,  # (B,K)
        "final_candidate_count": final_candidate_count,  # (B,K)
    }


def _hist_int(x: torch.Tensor, *, max_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # x is int tensor; returns (bins 1..max_k, counts)
    x = x.flatten().to(torch.int64)
    bins = torch.arange(1, max_k + 1, device=x.device, dtype=torch.int64)
    counts = torch.zeros_like(bins)
    for i, b in enumerate(bins):
        counts[i] = (x == b).sum()
    return bins, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze fused iters distribution on ScanObjectNN")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--k_neighbors", type=int, default=24)
    ap.add_argument("--init_threshold", type=float, default=0.6)
    ap.add_argument("--max_iter", type=int, default=10)
    ap.add_argument("--min_candidates_ratio", type=float, default=1.1)
    ap.add_argument("--seed", type=str, default="first", choices=("first", "random", "max_norm"))
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--max_batches", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else (torch.float16 if args.dtype == "fp16" else torch.bfloat16)

    ds = ScanObjectNN(partition="test", num_points=int(args.num_points))
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.workers), drop_last=False)

    all_iters = []
    all_counts = []
    all_thr = []

    for bi, (pts, _y) in enumerate(loader):
        if args.max_batches is not None and bi >= int(args.max_batches):
            break
        X = pts.to(device=device, dtype=dtype)  # [B,N,3]
        out = analyze_iters(
            X,
            int(args.K),
            k_neighbors=int(args.k_neighbors),
            init_threshold=float(args.init_threshold),
            max_iter=int(args.max_iter),
            min_candidates_ratio=float(args.min_candidates_ratio),
            seed=str(args.seed),
            exclude_self=bool(args.exclude_self),
        )
        all_iters.append(out["iters_used"].detach().to("cpu"))
        all_thr.append(out["final_threshold"].detach().to("cpu"))
        all_counts.append(out["final_candidate_count"].detach().to("cpu"))

    iters = torch.cat(all_iters, dim=0)  # [Btot,K]
    thr = torch.cat(all_thr, dim=0)
    cnt = torch.cat(all_counts, dim=0)

    Btot = int(iters.shape[0])
    K = int(iters.shape[1])
    total = Btot * K
    hit_max = int((iters == int(args.max_iter)).sum().item())
    frac_hit = hit_max / max(total, 1)

    bins, counts = _hist_int(iters, max_k=int(args.max_iter))

    # Per-step mean iters over batch dimension
    per_t_mean = iters.float().mean(dim=0)  # (K,)

    # Print
    print("=== Config ===")
    print(
        {
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "num_points": args.num_points,
            "K": args.K,
            "k_neighbors": args.k_neighbors,
            "init_threshold": args.init_threshold,
            "max_iter": args.max_iter,
            "min_candidates_ratio": args.min_candidates_ratio,
            "seed": args.seed,
            "exclude_self": bool(args.exclude_self),
            "max_batches": args.max_batches,
        }
    )
    print("\n=== Summary ===")
    print(f"B_total={Btot}  K={K}  samples(B*K)={total}")
    print(f"hit_max_iter={hit_max}  frac_hit_max_iter={frac_hit:.6f}")
    print(f"iters_mean={iters.float().mean().item():.4f}  iters_p50={iters.float().median().item():.1f}  iters_max={int(iters.max().item())}")

    print("\n=== Histogram (iters_used) ===")
    print("CSV,iters,count,frac")
    for b, c in zip(bins.tolist(), counts.tolist()):
        print(f"CSV,{b},{c},{c / max(total,1):.6f}")

    print("\n=== Per-step mean iters (head/tail) ===")
    head_n = min(10, K)
    tail_n = min(10, K)
    print("head:", [round(float(x), 3) for x in per_t_mean[:head_n].tolist()])
    print("tail:", [round(float(x), 3) for x in per_t_mean[-tail_n:].tolist()])

    # Candidate count sanity
    print("\n=== Candidate count summary ===")
    print(
        f"cand_mean={cnt.float().mean().item():.2f}  "
        f"cand_min={int(cnt.min().item())}  "
        f"cand_p50={int(cnt.float().median().item())}  "
        f"cand_max={int(cnt.max().item())}"
    )

    # Threshold sanity
    print("\n=== Threshold summary ===")
    print(
        f"thr_mean={thr.float().mean().item():.4f}  "
        f"thr_min={float(thr.min().item()):.4f}  "
        f"thr_p50={float(thr.float().median().item()):.4f}  "
        f"thr_max={float(thr.max().item()):.4f}"
    )


if __name__ == "__main__":
    main()

