#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate distance_hi_cap for knn_coarse_fine_ball_mask (hi = min(hi_full, cap)).

Default (recommended): per sampled center, take the **fine KNN top-k radius** — the k-th smallest
Euclidean distance to other points (same as the outer neighbor distance in exact top-k), then
aggregate a high percentile across probes. This matches the distance scale of KNN rather than
the global diameter d_max.

Optional modes:
  - pool_hi: use the pool_hi-th smallest distance (pool_hi = max(k, floor(k*(1+tau))) aligned with
    coarse candidate upper count).
  - d_max: legacy per-center max distance to any point.

Usage:
  python calibrate_knn_coarse_distance_hi.py --root ./data --partition train \\
    --num_points 1024 --k_neighbors 24 --min_candidates_ratio 1.2 \\
    --max_clouds 500 --stat_mode topk --out_json ./runs/knn_coarse_hi_cap.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from spec_kan_fps_opt import ModelNet40Wrapper, collate_points


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate knn coarse bisection distance_hi_cap")
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--partition", type=str, default="train", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_clouds", type=int, default=0, help="0 = use full partition")
    p.add_argument("--centers_per_cloud", type=int, default=64, help="Random center indices per cloud")
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--margin", type=float, default=1.02, help="Multiply reported cap by this factor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--k_neighbors", type=int, default=24)
    p.add_argument("--min_candidates_ratio", type=float, default=1.2)
    p.add_argument(
        "--exclude_self",
        action="store_true",
        help="Match KNN path: exclude center from distances before sorting",
    )
    p.add_argument(
        "--stat_mode",
        type=str,
        default="topk",
        choices=("topk", "pool_hi", "d_max"),
        help=(
            "topk: k-th smallest distance (fine top-k outer radius). "
            "pool_hi: pool_hi-th smallest with pool_hi=max(k,floor(k*ratio)). "
            "d_max: max distance to any point (legacy)."
        ),
    )
    return p.parse_args()


def _pool_bounds(k: int, ratio: float) -> tuple[int, int]:
    pool_hi = int(math.floor(k * ratio))
    pool_hi = max(pool_hi, k)
    pool_lo = k + 1
    if pool_lo > pool_hi:
        pool_lo = k
    return pool_lo, pool_hi


@torch.no_grad()
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = ModelNet40Wrapper(
        partition=args.partition, num_points=args.num_points, root=args.root
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_points,
    )

    vals: List[float] = []
    skipped = 0
    clouds_seen = 0
    for pts, _ in loader:
        B, N, _ = pts.shape
        for b in range(B):
            if args.max_clouds > 0 and clouds_seen >= args.max_clouds:
                break
            X = pts[b].float()
            k = min(int(args.k_neighbors), N)
            _, pool_hi = _pool_bounds(k, float(args.min_candidates_ratio))
            idx = torch.randperm(N)[: min(args.centers_per_cloud, N)]
            for j in idx:
                c = X[int(j)]
                d = torch.norm(X - c, dim=-1)
                if args.exclude_self:
                    d = d.clone()
                    d[int(j)] = float("inf")
                d_fin = d[torch.isfinite(d)]
                n_fin = int(d_fin.numel())
                if n_fin < 1:
                    skipped += 1
                    continue

                d_sorted, _ = torch.sort(d_fin)

                if args.stat_mode == "d_max":
                    vals.append(float(d_sorted[-1].item()))
                    continue

                if args.stat_mode == "topk":
                    # Match eval: k_actual = min(k_neighbors, N); need enough finite distances for k_act-th order stat
                    k_act = min(int(args.k_neighbors), N)
                    if n_fin < k_act:
                        skipped += 1
                        continue
                    vals.append(float(d_sorted[k_act - 1].item()))
                    continue

                # pool_hi: distance to the pool_hi-th nearest (1-based rank pool_hi -> index pool_hi-1)
                rank = min(pool_hi, n_fin)
                ix = rank - 1
                vals.append(float(d_sorted[ix].item()))

            clouds_seen += 1
        if args.max_clouds > 0 and clouds_seen >= args.max_clouds:
            break

    if not vals:
        raise RuntimeError(
            "No samples collected; check data path, max_clouds, k_neighbors vs N, or stat_mode"
        )

    arr = np.array(vals, dtype=np.float64)
    raw_p = float(np.percentile(arr, args.percentile))
    distance_hi_cap = raw_p * float(args.margin)

    pool_lo_m, pool_hi_m = _pool_bounds(
        min(int(args.k_neighbors), int(args.num_points)),
        float(args.min_candidates_ratio),
    )

    out = {
        "distance_hi_cap": distance_hi_cap,
        "stat_mode": args.stat_mode,
        "percentile": float(args.percentile),
        "margin": float(args.margin),
        "num_probes": int(len(vals)),
        "skipped_centers": int(skipped),
        "num_clouds": int(clouds_seen),
        "centers_per_cloud": int(args.centers_per_cloud),
        "num_points": int(args.num_points),
        "partition": args.partition,
        "k_neighbors": int(args.k_neighbors),
        "min_candidates_ratio": float(args.min_candidates_ratio),
        "pool_lo_meta": int(pool_lo_m),
        "pool_hi_meta": int(pool_hi_m),
        "exclude_self": bool(args.exclude_self),
        "description": (
            "Use with spec_kan_fps_opt_kan_coarse_fine_fused_eval.py --knn_coarse_hi_json this_file. "
            "Coarse bisection hi = min(hi_full, distance_hi_cap). "
            "stat_mode=topk: cap from empirical distribution of the k-th nearest distance per center "
            "(same scale as fine KNN outer radius). stat_mode=pool_hi: pool_hi-th smallest distance. "
            "stat_mode=d_max: legacy max distance to farthest point."
        ),
    }
    _od = os.path.dirname(os.path.abspath(args.out_json))
    if _od:
        os.makedirs(_od, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=True)

    print(
        f"Wrote {args.out_json}: distance_hi_cap={distance_hi_cap:.6f} "
        f"stat_mode={args.stat_mode} (p{args.percentile}={raw_p:.6f} * {args.margin}, "
        f"n={len(vals)} probes, clouds={clouds_seen}, skipped_centers={skipped})"
    )


if __name__ == "__main__":
    main()
