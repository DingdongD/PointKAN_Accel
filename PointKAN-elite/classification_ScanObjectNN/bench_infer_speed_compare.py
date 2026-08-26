#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScanObjectNN: 推理速度对比（两两对比，输出 speedup）。

默认对比：
- A: fused   = pointKANEliteFusedFPSKNNOptimized（融合 FPS+coarse-fine KNN）
- B: pytorch = pointKANElitePyTorchFPSOptimized（PyTorch FPS + 原 knn_point）

说明：
- 使用随机输入点云做纯 forward 计时（不跑 DataLoader），避免 I/O 干扰。
- 若提供 --checkpoint，则两边加载同一份权重，保证可比性（前提：结构兼容）。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from inference_and_kan_analysis import get_checkpoint_path, load_state_dict_for_model  # noqa: E402
import models as models  # noqa: E402


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _bench_forward_ms(net: torch.nn.Module, x: torch.Tensor, *, device: torch.device, warmup: int, iters: int) -> float:
    net.eval()
    _cuda_sync(device)
    for _ in range(int(warmup)):
        _ = net(x)
    _cuda_sync(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: List[float] = []
        for _ in range(int(iters)):
            start.record()
            _ = net(x)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        return float(torch.tensor(times).mean().item())

    import time

    times = []
    for _ in range(int(iters)):
        t0 = time.perf_counter()
        _ = net(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(torch.tensor(times).mean().item())


def _load_weights_if_any(net: torch.nn.Module, checkpoint: str | None, device: torch.device) -> str | None:
    if not checkpoint:
        return None
    ckpt_path = get_checkpoint_path(checkpoint)
    state = load_state_dict_for_model(ckpt_path, device)
    net.load_state_dict(state, strict=True)
    return ckpt_path


def _make_input(batch_size: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    # Model expects [B, 3, N]
    return (torch.rand((int(batch_size), int(npts), 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()


def _build_model(
    variant: str,
    *,
    num_classes: int,
    points: int,
    device: torch.device,
    dtype: torch.dtype,
    fused_init_threshold: float,
    fused_max_iter: int,
    fused_min_candidates_ratio: float,
    fused_seed: str,
) -> torch.nn.Module:
    v = str(variant).lower()
    if v == "vanilla":
        name = "pointKANElite"
        net = models.__dict__[name](num_classes=num_classes, points=points)
    elif v in ("spec_vanilla", "specvanilla"):
        name = "pointKANEliteSpecVanillaOptimized"
        net = models.__dict__[name](
            num_classes=num_classes,
            points=points,
            seed=str(fused_seed),
        )
    elif v in ("spec_cf", "speccf", "spec_coarse_fine", "spec_coarsefine"):
        name = "pointKANEliteSpecCoarseFineOptimized"
        net = models.__dict__[name](
            num_classes=num_classes,
            points=points,
            init_threshold_coarse_fine=float(fused_init_threshold),
            max_iter_coarse_fine=int(fused_max_iter),
            min_candidates_ratio=float(fused_min_candidates_ratio),
            seed=str(fused_seed),
        )
    elif v in ("cuda_cf", "cudacf"):
        name = "pointKANEliteCudaCoarseFineOptimized"
        net = models.__dict__[name](
            num_classes=num_classes,
            points=points,
            init_threshold_coarse_fine=float(fused_init_threshold),
            max_iter_coarse_fine=int(fused_max_iter),
            min_candidates_ratio=float(fused_min_candidates_ratio),
            seed=str(fused_seed),
        )
    elif v in ("cuda_cf_strict", "cudacf_strict", "strict"):
        name = "pointKANEliteCudaCoarseFineStrictOptimized"
        net = models.__dict__[name](
            num_classes=num_classes,
            points=points,
            init_threshold_coarse_fine=float(fused_init_threshold),
            max_iter_coarse_fine=int(fused_max_iter),
            min_candidates_ratio=float(fused_min_candidates_ratio),
            seed=str(fused_seed),
        )
    elif v == "pytorch":
        name = "pointKANElitePyTorchFPSOptimized"
        net = models.__dict__[name](num_classes=num_classes, points=points)
    elif v == "cascade":
        name = "pointKANEliteFPSOptimized"
        net = models.__dict__[name](num_classes=num_classes, points=points)
    elif v == "prefix":
        name = "pointKANEliteFPSPrefixOptimized"
        net = models.__dict__[name](num_classes=num_classes, points=points)
    elif v == "fused":
        name = "pointKANEliteFusedFPSKNNOptimized"
        net = models.__dict__[name](
            num_classes=num_classes,
            points=points,
            init_threshold_coarse_fine=float(fused_init_threshold),
            max_iter_coarse_fine=int(fused_max_iter),
            min_candidates_ratio=float(fused_min_candidates_ratio),
            seed=str(fused_seed),
        )
    elif v in ("pn2_cfknn", "pointnet2_cfknn"):
        name = "pointKANElitePointNet2FPSCoarseFineKNNOptimized"
        net = models.__dict__[name](
            num_classes=num_classes,
            points=points,
            init_threshold_coarse_fine=float(fused_init_threshold),
            max_iter_coarse_fine=int(fused_max_iter),
            min_candidates_ratio=float(fused_min_candidates_ratio),
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use one of: vanilla|pytorch|cascade|prefix|fused|pn2_cfknn|spec_vanilla|spec_cf|cuda_cf|cuda_cf_strict")
    return net.to(device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="ScanObjectNN: 两两对比推理速度（输出 speedup）")
    ap.add_argument("--a", type=str, default="fused", help="模型A：vanilla|pytorch|cascade|prefix|fused")
    ap.add_argument("--b", type=str, default="pytorch", help="模型B：vanilla|pytorch|cascade|prefix|fused")
    ap.add_argument("--checkpoint", type=str, default=None, help="检查点目录（含 best_checkpoint.pth）或 .pth/.pt；不提供则用随机权重")
    ap.add_argument("--num_classes", type=int, default=15)
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    # fused hyperparams (only used when variant includes fused)
    ap.add_argument("--fused_init_threshold", type=float, default=0.15)
    ap.add_argument("--fused_max_iter", type=int, default=10)
    ap.add_argument("--fused_min_candidates_ratio", type=float, default=1.1)
    ap.add_argument("--fused_seed", type=str, default="first", choices=("first", "random", "max_norm"))
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else (torch.float16 if args.dtype == "fp16" else torch.bfloat16)

    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.version.cuda}")
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")

    net_a = _build_model(
        args.a,
        num_classes=int(args.num_classes),
        points=int(args.points),
        device=device,
        dtype=dtype,
        fused_init_threshold=float(args.fused_init_threshold),
        fused_max_iter=int(args.fused_max_iter),
        fused_min_candidates_ratio=float(args.fused_min_candidates_ratio),
        fused_seed=str(args.fused_seed),
    )
    net_b = _build_model(
        args.b,
        num_classes=int(args.num_classes),
        points=int(args.points),
        device=device,
        dtype=dtype,
        fused_init_threshold=float(args.fused_init_threshold),
        fused_max_iter=int(args.fused_max_iter),
        fused_min_candidates_ratio=float(args.fused_min_candidates_ratio),
        fused_seed=str(args.fused_seed),
    )

    ckpt_used = None
    if args.checkpoint:
        ckpt_used = _load_weights_if_any(net_a, args.checkpoint, device)
        _ = _load_weights_if_any(net_b, args.checkpoint, device)

    x = _make_input(args.batch_size, args.points, device, dtype, seed=int(args.seed))

    ms_a = _bench_forward_ms(net_a, x, device=device, warmup=int(args.warmup), iters=int(args.iters))
    ms_b = _bench_forward_ms(net_b, x, device=device, warmup=int(args.warmup), iters=int(args.iters))
    sp = (ms_b / ms_a) if ms_a > 0 else 0.0

    print("\n=== Results ===")
    print(f"A: {args.a}  ms={ms_a:.6f}")
    print(f"B: {args.b}  ms={ms_b:.6f}")
    print(f"speedup (B/A): {sp:.6f}x")
    if ckpt_used:
        print(f"checkpoint_used: {ckpt_used}")
    else:
        print("checkpoint_used: (none, random weights)")


if __name__ == "__main__":
    main()

