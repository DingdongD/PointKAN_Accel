#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两种“纯 PyTorch”实现的 FPS+KNN 推理开销（CUDA event 计时）并做点数 N sweep：

1) baseline_pytorch:
   - FPS: farthest_point_sample (Python for-loop)
   - KNN: 对每个 FPS center 计算 square_distance(center, xyz) + topk 得邻居

2) fused_coarse_fine:
   - 使用 models/pointkanElite_fps_opt.py 里的 fps_with_knn_coarse_fine_fused（每步距离复用 + coarse-fine 阈值）

注意：
- 这是算子级 benchmark，不跑完整模型 forward。
- 计算量极大（O(B*K*N)），建议用小 batch/iters 做对比趋势。
"""

import argparse
import os
import sys
from typing import Callable, List, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from models.pointkanElite import farthest_point_sample, square_distance  # noqa: E402
from models.pointkanElite_fps_opt import fps_with_knn_coarse_fine_fused  # noqa: E402
from models.pointkanElite_fps_opt import fps_with_knn_coarse_fine_cuda  # noqa: E402


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _bench_ms(fn: Callable[[], None], *, device: torch.device, warmup: int, iters: int) -> float:
    _cuda_sync(device)
    for _ in range(warmup):
        fn()
    _cuda_sync(device)

    if device.type == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        times: List[float] = []
        for _ in range(iters):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(float(s.elapsed_time(e)))
        return float(torch.tensor(times).mean().item())
    else:
        import time

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return float(torch.tensor(times).mean().item())


@torch.no_grad()
def baseline_pytorch_fps_knn(xyz: torch.Tensor, K: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xyz: [B, N, 3]
    returns:
      idx: [B, K]
      knn_idx: [B, K, k]
    """
    B, N, _ = xyz.shape
    idx = farthest_point_sample(xyz, int(K)).long()  # [B,K]
    centers = xyz[torch.arange(B, device=xyz.device).unsqueeze(1), idx]  # [B,K,3]
    # For each center: compute dist to all xyz then topk
    # square_distance wants src=[B, S, 3], dst=[B, N, 3] -> [B,S,N]
    d2 = square_distance(centers, xyz)  # [B,K,N]
    _vals, knn_idx = torch.topk(d2, k=int(k), dim=-1, largest=False, sorted=False)
    return idx, knn_idx


@torch.no_grad()
def fused_coarse_fine(xyz: torch.Tensor, K: int, k: int, init_threshold: float, max_iter: int, min_candidates_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    idx, knn_idx = fps_with_knn_coarse_fine_fused(
        xyz,
        int(K),
        k_neighbors=int(k),
        init_threshold=float(init_threshold),
        max_iter=int(max_iter),
        min_candidates_ratio=float(min_candidates_ratio),
        seed="first",
        exclude_self=False,
    )
    return idx, knn_idx


@torch.no_grad()
def cuda_coarse_fine(
    xyz: torch.Tensor,
    K: int,
    k: int,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    block_size: int,
    mmax_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    idx, knn_idx = fps_with_knn_coarse_fine_cuda(
        xyz.float(),
        int(K),
        k_neighbors=int(k),
        init_threshold=float(init_threshold),
        max_iter=int(max_iter),
        min_candidates_ratio=float(min_candidates_ratio),
        seed="first",
        exclude_self=False,
        block_size=int(block_size),
        mmax_factor=int(mmax_factor),
    )
    return idx, knn_idx


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark PyTorch FPS+KNN baseline vs fused_coarse_fine")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--points_sweep", type=str, default="512,1024,2048", help="例如 '512,1024,2048,4096'")
    ap.add_argument("--K_ratio", type=float, default=0.5, help="K = int(K_ratio * N)")
    ap.add_argument("--k_neighbors", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    # fused hyperparams
    ap.add_argument("--init_threshold", type=float, default=0.6)
    ap.add_argument("--max_iter", type=int, default=10)
    ap.add_argument("--min_candidates_ratio", type=float, default=1.1)
    ap.add_argument("--impl", type=str, default="fused", choices=("fused", "cuda"), help="优化实现：fused=python blockwise, cuda=CUDA extension")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--mmax_factor", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else (torch.float16 if args.dtype == "fp16" else torch.bfloat16)
    points_list = [int(x) for x in str(args.points_sweep).split(",") if x.strip()]

    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.version.cuda}")
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")
    print(f"batch_size: {args.batch_size}")
    print(f"k_neighbors: {args.k_neighbors}")
    print(f"hparams: init_threshold={args.init_threshold}, max_iter={args.max_iter}, min_candidates_ratio={args.min_candidates_ratio}, impl={args.impl}, block_size={args.block_size}, mmax_factor={args.mmax_factor}")

    print("\nCSV,N,K,baseline_ms,opt_ms,speedup(baseline/opt)")
    for i, N in enumerate(points_list):
        K = max(1, min(N, int(round(float(args.K_ratio) * N))))
        k = min(int(args.k_neighbors), int(N))

        g = torch.Generator(device=device)
        g.manual_seed(int(args.seed) + i)
        xyz = (torch.rand((int(args.batch_size), int(N), 3), generator=g, device=device, dtype=dtype) * 2 - 1).contiguous()

        def fn_base():
            _ = baseline_pytorch_fps_knn(xyz, K, k)

        if args.impl == "cuda":
            def fn_opt():
                _ = cuda_coarse_fine(
                    xyz, K, k, args.init_threshold, args.max_iter, args.min_candidates_ratio,
                    args.block_size, args.mmax_factor
                )
        else:
            def fn_opt():
                _ = fused_coarse_fine(xyz, K, k, args.init_threshold, args.max_iter, args.min_candidates_ratio)

        ms_base = _bench_ms(fn_base, device=device, warmup=int(args.warmup), iters=int(args.iters))
        ms_opt = _bench_ms(fn_opt, device=device, warmup=int(args.warmup), iters=int(args.iters))
        sp = (ms_base / ms_opt) if ms_opt > 0 else 0.0
        print(f"CSV,{N},{K},{ms_base:.6f},{ms_opt:.6f},{sp:.6f}")


if __name__ == "__main__":
    main()

