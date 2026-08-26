#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScanObjectNN: 推理速度对比（vanilla vs FPS 级联预计算优化版），并在不同输入点数 N 下输出加速比。

说明：
- 使用随机输入点云做纯 forward 计时（不跑 DataLoader），避免 I/O 干扰。
- 两个模型加载同一个 checkpoint（若提供），保证权重一致。
- 对 CUDA：使用 cuda event + synchronize；包含 warmup。
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

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
    for _ in range(warmup):
        _ = net(x)
    _cuda_sync(device)

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: List[float] = []
        for _ in range(iters):
            start.record()
            _ = net(x)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        return float(torch.tensor(times).mean().item())
    else:
        import time

        times = []
        for _ in range(iters):
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
    return (torch.rand((batch_size, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description="ScanObjectNN: vanilla vs FPS-opt 推理速度 & 点数 N sweep")
    ap.add_argument("--checkpoint", type=str, default=None, help="检查点目录（含 best_checkpoint.pth）或 .pth/.pt；不提供则用随机权重")
    ap.add_argument("--num_classes", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--points_sweep", type=str, default="512,1024,2048,4096", help="例如 '512,1024,2048'")
    ap.add_argument(
        "--opt",
        type=str,
        default="cascade",
        choices=("cascade", "prefix", "fused", "pytorchfps"),
        help="优化版类型：cascade=级联FPS；prefix=仅stage0 FPS，其余取前缀；fused=GAM内融合FPS+KNN(探索版)；pytorchfps=用PyTorch版FPS替代pointnet2",
    )
    # fused hyperparams (only used when --opt fused)
    ap.add_argument("--fused_init_threshold", type=float, default=0.15, help="fused: init_threshold_coarse_fine")
    ap.add_argument("--fused_max_iter", type=int, default=10, help="fused: max_iter_coarse_fine")
    ap.add_argument("--fused_min_candidates_ratio", type=float, default=1.1, help="fused: min_candidates_ratio")
    ap.add_argument("--fused_seed", type=str, default="first", help="fused: seed (first/random/max_norm)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else (torch.float16 if args.dtype == "fp16" else torch.bfloat16)
    points_list = [int(x) for x in str(args.points_sweep).split(",") if x.strip()]
    if not points_list:
        raise SystemExit("points_sweep 为空。")

    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.version.cuda}")
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {args.dtype}")

    ckpt_used = None

    print("\n=== Results ===")
    print(f"opt_variant: {args.opt}")
    if args.opt == "fused":
        print(
            "fused_hparams:",
            {
                "init_threshold_coarse_fine": args.fused_init_threshold,
                "max_iter_coarse_fine": args.fused_max_iter,
                "min_candidates_ratio": args.fused_min_candidates_ratio,
                "seed": args.fused_seed,
            },
        )
    print("CSV,points,vanilla_ms,opt_ms,speedup")
    for i, npts in enumerate(points_list):
        if npts <= 0:
            continue

        # Build models with points=npts
        net_v = models.__dict__["pointKANElite"](num_classes=args.num_classes, points=npts).to(device=device, dtype=dtype)
        if args.opt == "cascade":
            opt_name = "pointKANEliteFPSOptimized"
        elif args.opt == "prefix":
            opt_name = "pointKANEliteFPSPrefixOptimized"
        elif args.opt == "fused":
            opt_name = "pointKANEliteFusedFPSKNNOptimized"
        else:
            opt_name = "pointKANElitePyTorchFPSOptimized"
        if args.opt == "fused":
            net_o = models.__dict__[opt_name](
                num_classes=args.num_classes,
                points=npts,
                init_threshold_coarse_fine=float(args.fused_init_threshold),
                max_iter_coarse_fine=int(args.fused_max_iter),
                min_candidates_ratio=float(args.fused_min_candidates_ratio),
                seed=str(args.fused_seed),
            ).to(device=device, dtype=dtype)
        else:
            net_o = models.__dict__[opt_name](num_classes=args.num_classes, points=npts).to(device=device, dtype=dtype)

        if ckpt_used is None:
            ckpt_used = _load_weights_if_any(net_v, args.checkpoint, device)
            _ = _load_weights_if_any(net_o, args.checkpoint, device)
        else:
            _ = _load_weights_if_any(net_v, args.checkpoint, device)
            _ = _load_weights_if_any(net_o, args.checkpoint, device)

        x = _make_input(args.batch_size, npts, device, dtype, seed=args.seed + i)

        ms_v = _bench_forward_ms(net_v, x, device=device, warmup=args.warmup, iters=args.iters)
        ms_o = _bench_forward_ms(net_o, x, device=device, warmup=args.warmup, iters=args.iters)
        sp = (ms_v / ms_o) if ms_o > 0 else 0.0

        print(f"CSV,{npts},{ms_v:.6f},{ms_o:.6f},{sp:.6f}")

    if ckpt_used:
        print(f"\ncheckpoint_used: {ckpt_used}")
    else:
        print("\ncheckpoint_used: (none, random weights)")


if __name__ == "__main__":
    main()

