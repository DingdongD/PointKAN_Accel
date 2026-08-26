#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from models.pointkanElite_fps_opt import (  # noqa: E402
    fps_with_knn_coarse_fine_fused_blockwise,
    fps_with_knn_coarse_fine_cuda,
)


def knn_neighbor_set_iou(idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    # idx: [B, K, k]
    B, K, _ = idx_a.shape
    out = torch.zeros((B, K), dtype=torch.float32, device=idx_a.device)
    for b in range(B):
        for t in range(K):
            a = set(idx_a[b, t].tolist())
            bb = set(idx_b[b, t].tolist())
            inter = len(a & bb)
            union = len(a | bb)
            out[b, t] = float(inter / union) if union > 0 else 1.0
    return out


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser("Verify CUDA coarse-fine vs Python coarse-fine")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_points", type=int, default=4096)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--k_neighbors", type=int, default=24)
    ap.add_argument("--init_threshold", type=float, default=0.1)
    ap.add_argument("--max_iter", type=int, default=10)
    ap.add_argument("--min_candidates_ratio", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--mmax_factor", type=int, default=4)
    ap.add_argument("--strict_semantic", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))
    x = (torch.rand((args.batch_size, args.num_points, 3), generator=g, device=device, dtype=torch.float32) * 2 - 1).contiguous()

    idx_py, knn_py = fps_with_knn_coarse_fine_fused_blockwise(
        x,
        int(args.K),
        k_neighbors=int(args.k_neighbors),
        init_threshold=float(args.init_threshold),
        max_iter=int(args.max_iter),
        min_candidates_ratio=float(args.min_candidates_ratio),
        seed="first",
        exclude_self=False,
        block_size=int(args.block_size),
    )
    idx_cu, knn_cu = fps_with_knn_coarse_fine_cuda(
        x,
        int(args.K),
        k_neighbors=int(args.k_neighbors),
        init_threshold=float(args.init_threshold),
        max_iter=int(args.max_iter),
        min_candidates_ratio=float(args.min_candidates_ratio),
        seed="first",
        exclude_self=False,
        block_size=int(args.block_size),
        mmax_factor=int(args.mmax_factor),
        strict_semantic=bool(args.strict_semantic),
    )

    same_fps = (idx_py == idx_cu).float().mean().item()
    same_knn = (knn_py == knn_cu).float().mean().item()
    iou = knn_neighbor_set_iou(knn_py.cpu(), knn_cu.cpu())
    print("=== verify cuda_cfknn ===")
    print(f"fps_idx_exact_ratio: {same_fps:.6f}")
    print(f"knn_idx_exact_ratio: {same_knn:.6f}")
    print(f"knn_set_iou_mean: {float(iou.mean().item()):.6f}")
    print(f"knn_set_iou_min : {float(iou.min().item()):.6f}")


if __name__ == "__main__":
    main()

