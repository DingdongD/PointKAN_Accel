#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对三种 PointKAN-Elite 任务做端到端性能对比：
  - Part Segmentation
  - Classification (ModelNet40)
  - Classification (ScanObjectNN)

对比项（同一任务、同一 checkpoint、同一输入 shape）：
  - Vanilla: Triton-GRF (KAT_Group)
  - PWL:    Triton-PWL (KAT_Group_PWL_Triton) via inject_grf_pwl_v2(..., impl="triton")

仅测在线推理（LUT 构建离线，不计入）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple, Optional, List

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (THIS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

# Task-specific helpers
from part_segmentation.inference_and_kan_analysis import (  # type: ignore
    get_checkpoint_path as get_ckpt_partseg,
    build_model_and_load_checkpoint as build_partseg,
)
from classification_ModelNet40.inference_and_kan_analysis import (  # type: ignore
    get_checkpoint_path as get_ckpt_mn40,
    build_model_and_load_checkpoint as build_mn40,
)
from classification_ScanObjectNN.inference_and_kan_analysis import (  # type: ignore
    get_checkpoint_path as get_ckpt_scan,
    build_model_and_load_checkpoint as build_scan,
)

from part_segmentation.inject_grf_pwl_v2 import inject_grf_pwl_v2  # type: ignore


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _bench_forward(fn, *, device: torch.device, warmup: int, iters: int) -> Dict[str, float]:
    _cuda_sync(device)
    for _ in range(warmup):
        _ = fn()
    _cuda_sync(device)

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    times_ms: List[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    t = torch.tensor(times_ms)
    mean_ms = float(t.mean().item())
    p50_ms = float(t.median().item())
    p90_ms = float(t.quantile(0.9).item()) if iters >= 10 else float(max(times_ms))
    ips = iters / (sum(times_ms) / 1000.0)
    return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p90_ms": p90_ms, "ips": float(ips)}


def _make_partseg_inputs(batch: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pts = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    nrm = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    cls_idx = torch.randint(0, 16, (batch,), generator=g, device=device)
    onehot = torch.eye(16, device=device, dtype=dtype)[cls_idx]
    return pts, nrm, onehot


def _make_cls_inputs(batch: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pts = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    return pts


def _default_partseg_ckpt_dir() -> str:
    return os.path.join(THIS_DIR, "part_segmentation", "checkpoints", "pointKANElite_part_exp")


def _default_partseg_pwl_json() -> str:
    return os.path.join(THIS_DIR, "part_segmentation", "grf_export_out", "pwl_v2.json")


def _default_mn40_ckpt_dir() -> str:
    return os.path.join(THIS_DIR, "classification_ModelNet40", "checkpoints", "pointKANElite-pointkan_elite_exp-42")


def _default_mn40_pwl_json() -> str:
    # Most common export location in this workspace (see my_pwl_compare_out/modelnet40_pwl_compare.json).
    return os.path.join(os.path.dirname(os.path.dirname(THIS_DIR)), "GQA-LUT", "my_out", "pwl_v2.json")


def _default_scan_ckpt_dir() -> str:
    return os.path.join(THIS_DIR, "classification_ScanObjectNN", "checkpoints", "pointKANElite-scan_exp-42")


def _default_scan_pwl_json() -> str:
    return os.path.join(THIS_DIR, "classification_ScanObjectNN", "grf_export_out", "pwl_v2.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="三任务：Triton-GRF vs Triton-PWL speedup（端到端）")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    # 不同任务的 pwl_v2.json 通常支持的 bits 档位不同（例如 partseg=8, mn40=6, scan=6）。
    # 因此默认每个任务都走 JSON meta 的 deploy_breakpoint_bits（即不传 bit_key）。
    ap.add_argument("--partseg_bit", type=str, default=None, help="partseg: layers[].bits[该键]；默认用 JSON meta")
    ap.add_argument("--mn40_bit", type=str, default=None, help="modelnet40: layers[].bits[该键]；默认用 JSON meta")
    ap.add_argument("--scan_bit", type=str, default=None, help="scanobjectnn: layers[].bits[该键]；默认用 JSON meta")

    # Task configs
    ap.add_argument("--partseg_ckpt", type=str, default=_default_partseg_ckpt_dir())
    ap.add_argument("--partseg_pwl", type=str, default=_default_partseg_pwl_json())
    ap.add_argument("--partseg_points", type=int, default=2048)

    ap.add_argument("--mn40_ckpt", type=str, default=_default_mn40_ckpt_dir())
    ap.add_argument(
        "--mn40_pwl",
        type=str,
        default=_default_mn40_pwl_json(),
        help="ModelNet40 的 pwl_v2.json；默认尝试 workspace/GQA-LUT/my_out/pwl_v2.json（若你导出在别处请覆盖）",
    )
    ap.add_argument("--mn40_points", type=int, default=1024)

    ap.add_argument("--scan_ckpt", type=str, default=_default_scan_ckpt_dir())
    ap.add_argument("--scan_pwl", type=str, default=_default_scan_pwl_json())
    ap.add_argument("--scan_points", type=int, default=1024)
    ap.add_argument("--scan_num_classes", type=int, default=15)

    args = ap.parse_args()
    device = torch.device(args.device)
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    print("=== Environment ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda: {torch.version.cuda}")
        try:
            print(f"gpu: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"dtype: {args.dtype}")
    print()

    rows = []

    # --- PartSeg ---
    ckpt_part = get_ckpt_partseg(args.partseg_ckpt)
    net_v = build_partseg(ckpt_part, device)
    net_p = build_partseg(ckpt_part, device)
    n_rep, bit_used = inject_grf_pwl_v2(net_p, args.partseg_pwl, bit_key=args.partseg_bit, device=device, impl="triton")
    pts, nrm, onehot = _make_partseg_inputs(args.batch_size, args.partseg_points, device, dtype, args.seed)
    v = _bench_forward(lambda: net_v(pts, nrm, onehot), device=device, warmup=args.warmup, iters=args.iters)
    p = _bench_forward(lambda: net_p(pts, nrm, onehot), device=device, warmup=args.warmup, iters=args.iters)
    rows.append(("partseg", n_rep, bit_used, v, p))

    # --- ModelNet40 ---
    if not args.mn40_pwl or not os.path.isfile(os.path.abspath(os.path.expanduser(args.mn40_pwl))):
        print("\n[warning] 未找到 mn40_pwl：请显式传 --mn40_pwl 指向 ModelNet40 对应 pwl_v2.json；将跳过 ModelNet40。")
    else:
        ckpt_mn40 = get_ckpt_mn40(args.mn40_ckpt)
        net_v = build_mn40(ckpt_mn40, device)
        net_p = build_mn40(ckpt_mn40, device)
        n_rep, bit_used = inject_grf_pwl_v2(net_p, args.mn40_pwl, bit_key=args.mn40_bit, device=device, impl="triton")
        x = _make_cls_inputs(args.batch_size, args.mn40_points, device, dtype, args.seed + 1)
        v = _bench_forward(lambda: net_v(x), device=device, warmup=args.warmup, iters=args.iters)
        p = _bench_forward(lambda: net_p(x), device=device, warmup=args.warmup, iters=args.iters)
        rows.append(("modelnet40", n_rep, bit_used, v, p))

    # --- ScanObjectNN ---
    ckpt_scan = get_ckpt_scan(args.scan_ckpt)
    net_v = build_scan(ckpt_scan, device, num_classes=args.scan_num_classes)
    net_p = build_scan(ckpt_scan, device, num_classes=args.scan_num_classes)
    n_rep, bit_used = inject_grf_pwl_v2(net_p, args.scan_pwl, bit_key=args.scan_bit, device=device, impl="triton")
    x = _make_cls_inputs(args.batch_size, args.scan_points, device, dtype, args.seed + 2)
    v = _bench_forward(lambda: net_v(x), device=device, warmup=args.warmup, iters=args.iters)
    p = _bench_forward(lambda: net_p(x), device=device, warmup=args.warmup, iters=args.iters)
    rows.append(("scanobjectnn", n_rep, bit_used, v, p))

    print("\n" + "=" * 88)
    print("Triton-GRF vs Triton-PWL speedup（端到端）")
    print("=" * 88)
    print(f"{'task':<14} {'replaced':>8} {'bit':>5}  {'van_mean':>10} {'pwl_mean':>10} {'speedup':>8}  {'van_p50':>9} {'pwl_p50':>9}")
    for task, n_rep, bit_used, v, p in rows:
        speed = v["mean_ms"] / p["mean_ms"]
        print(
            f"{task:<14} {n_rep:>8} {str(bit_used):>5}  "
            f"{v['mean_ms']:>10.4f} {p['mean_ms']:>10.4f} {speed:>8.3f}  "
            f"{v['p50_ms']:>9.4f} {p['p50_ms']:>9.4f}"
        )
    print("=" * 88)
    print("\n注：该脚本用随机输入测在线推理（不含数据加载），用于纯推理 kernel 对比。")


if __name__ == "__main__":
    main()

