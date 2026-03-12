#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独测量 KAT_Group 前向推理时间（单次调用，不含整模型其它部分）。

用法（在 PointKAN-elite 根目录或本目录执行）：
  python -m kat_rational.bench_kat_group
  python -m kat_rational.bench_kat_group --device cuda --warmup 5 --repeat 100
  python -m kat_rational.bench_kat_group --shape 1,1024,256 --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

# 确保可导入 kat_rational
ELITE_ROOT = Path(__file__).resolve().parent.parent
if str(ELITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ELITE_ROOT))

import torch
from kat_rational import KAT_Group


def parse_shape(s: str):
    return tuple(int(x) for x in s.strip().split(","))


def bench_forward(module, x, device, warmup=5, repeat=100):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = module(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / repeat * 1000.0  # ms per forward


def main():
    ap = argparse.ArgumentParser(description="KAT_Group 单独推理时间")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shape", default="1,1024,256", help="输入形状 B,L,D 或 B,D，例如 1,1024,256")
    ap.add_argument("--num_groups", type=int, default=8)
    ap.add_argument("--mode", default="gelu")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device)
    shape = parse_shape(args.shape)
    if len(shape) == 2:
        # (B, D)
        x = torch.randn(*shape, device=device, dtype=torch.float32)
    elif len(shape) == 3:
        # (B, L, D)
        x = torch.randn(*shape, device=device, dtype=torch.float32)
    else:
        print("shape 应为 2 维 (B,D) 或 3 维 (B,L,D)", file=sys.stderr)
        sys.exit(1)

    kat = KAT_Group(num_groups=args.num_groups, mode=args.mode, device=args.device)
    kat = kat.to(device)
    kat.eval()

    ms = bench_forward(kat, x, device, warmup=args.warmup, repeat=args.repeat)
    print(f"KAT_Group 推理时间: {ms:.4f} ms/次 (shape={shape}, device={args.device}, warmup={args.warmup}, repeat={args.repeat})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
