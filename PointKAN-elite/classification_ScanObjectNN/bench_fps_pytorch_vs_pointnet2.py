#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 FPS 的两种实现推理速度（CUDA event 计时）：
- pointnet2_utils.furthest_point_sample（CUDA 扩展）
- PyTorch 版 farthest_point_sample（Python for-loop + 张量运算）
"""

import argparse
import os
import sys
from typing import Callable, List

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from pointnet2_ops import pointnet2_utils  # noqa: E402
from models.pointkanElite import farthest_point_sample as fps_pytorch  # noqa: E402


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _bench_ms(fn: Callable[[], torch.Tensor], *, device: torch.device, warmup: int, iters: int) -> float:
    _cuda_sync(device)
    for _ in range(warmup):
        _ = fn()
    _cuda_sync(device)

    if device.type == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        times: List[float] = []
        for _ in range(iters):
            s.record()
            _ = fn()
            e.record()
            torch.cuda.synchronize()
            times.append(float(s.elapsed_time(e)))
        return float(torch.tensor(times).mean().item())
    else:
        import time

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        return float(torch.tensor(times).mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description="FPS: pointnet2_utils vs PyTorch 版本（CUDA event 计时）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else (torch.float16 if args.dtype == "fp16" else torch.bfloat16)

    g = torch.Generator(device=device)
    g.manual_seed(int(args.seed))
    xyz = (torch.rand((args.batch_size, args.num_points, 3), generator=g, device=device, dtype=dtype) * 2 - 1).contiguous()

    def fn_pointnet2():
        return pointnet2_utils.furthest_point_sample(xyz, int(args.K)).long()

    def fn_pytorch():
        # our PyTorch implementation expects [B,N,3] and returns [B,K]
        return fps_pytorch(xyz, int(args.K))

    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.version.cuda}")
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")
    print(f"shape: B={args.batch_size}, N={args.num_points}, K={args.K}")

    ms_cuda = _bench_ms(fn_pointnet2, device=device, warmup=args.warmup, iters=args.iters)
    ms_pt = _bench_ms(fn_pytorch, device=device, warmup=args.warmup, iters=args.iters)
    speedup = (ms_pt / ms_cuda) if ms_cuda > 0 else 0.0

    print("\n=== Results (mean ms per call) ===")
    print(f"pointnet2_utils.furthest_point_sample: {ms_cuda:.6f} ms")
    print(f"pytorch farthest_point_sample:        {ms_pt:.6f} ms")
    print(f"ratio (pytorch / pointnet2):          {speedup:.3f}x")
    print("\nCSV,impl,ms")
    print(f"CSV,pointnet2,{ms_cuda:.9f}")
    print(f"CSV,pytorch,{ms_pt:.9f}")


if __name__ == "__main__":
    main()

