#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量 profiling：用随机输入测一次 forward 时 KAT 子模块占比。

统计项：
  - Vanilla：Triton-GRF（KAT_Group）
  - PWL：Triton-PWL（KAT_Group_PWL_Triton）

输出：
  - total_forward_ms
  - kat_forward_ms (sum of all matched KAT module forward times)
  - kat_fraction
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple, List, Optional

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (THIS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

from part_segmentation.inference_and_kan_analysis import (  # type: ignore
    get_checkpoint_path as get_ckpt_partseg,
    build_model_and_load_checkpoint as build_partseg,
)
from part_segmentation.inject_grf_pwl_v2 import inject_grf_pwl_v2  # type: ignore

from classification_ModelNet40.inference_and_kan_analysis import (  # type: ignore
    get_checkpoint_path as get_ckpt_mn40,
    build_model_and_load_checkpoint as build_mn40,
)
from classification_ScanObjectNN.inference_and_kan_analysis import (  # type: ignore
    get_checkpoint_path as get_ckpt_scan,
    build_model_and_load_checkpoint as build_scan,
)

from kat_rational import KAT_Group, KAT_Group_PWL_Triton  # type: ignore


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def _time_forward_and_kat(
    net: torch.nn.Module,
    kat_types: Tuple[type, ...],
    inputs_fn,
    *,
    device: torch.device,
) -> Dict[str, float]:
    # Warmup: trigger Triton compilation & kernel cache before measuring.
    out = net(*inputs_fn())
    _cuda_sync(device)
    if out is None:
        pass

    kat_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    start_events: Dict[int, torch.cuda.Event] = {}

    def _pre_hook(m, _inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        start_events[id(m)] = e

    def _post_hook(m, _inp, _out):
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        e_start = start_events.pop(id(m), None)
        if e_start is not None:
            kat_pairs.append((e_start, e_end))

    hooks = []
    for _name, mod in net.named_modules():
        if isinstance(mod, kat_types):
            hooks.append(mod.register_forward_pre_hook(_pre_hook))
            hooks.append(mod.register_forward_hook(_post_hook))

    # Overall timing (only for the measured forward)
    if device.type == "cuda":
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        e_start.record()
    out = net(*inputs_fn())
    if device.type == "cuda":
        e_end.record()
        _cuda_sync(device)
        total_ms = float(e_start.elapsed_time(e_end))
    else:
        total_ms = 0.0
    # Sum kat module timings
    kat_ms = 0.0
    if kat_pairs:
        for e_s, e_e in kat_pairs:
            kat_ms += float(e_s.elapsed_time(e_e))

    for h in hooks:
        h.remove()

    return {"total_forward_ms": total_ms, "kat_forward_ms": kat_ms, "kat_fraction": kat_ms / max(total_ms, 1e-9)}


def _make_partseg_inputs(batch: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pts = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    nrm = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    cls_idx = torch.randint(0, 16, (batch,), generator=g, device=device)
    onehot = torch.eye(16, device=device, dtype=dtype)[cls_idx]
    return pts, nrm, onehot


def _make_cls_inputs(batch: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pts = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    return (pts,)


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile 三任务：KAT 子模块占比")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bit", type=str, default=None, help="若提供，则强制使用该 bits 键；否则用 JSON meta deploy bits")
    ap.add_argument("--partseg_points", type=int, default=2048)
    ap.add_argument("--mn40_points", type=int, default=1024)
    ap.add_argument("--scan_points", type=int, default=1024)

    ap.add_argument("--partseg_ckpt", type=str, default=os.path.join(THIS_DIR, "part_segmentation", "checkpoints", "pointKANElite_part_exp"))
    ap.add_argument("--partseg_pwl", type=str, default=os.path.join(THIS_DIR, "part_segmentation", "grf_export_out", "pwl_v2.json"))

    ap.add_argument("--mn40_ckpt", type=str, default=os.path.join(THIS_DIR, "classification_ModelNet40", "checkpoints", "pointKANElite-pointkan_elite_exp-42"))
    # ModelNet40 的 pwl_v2.json 来自 /workspace/GQA-LUT/my_out/pwl_v2.json
    ap.add_argument(
        "--mn40_pwl",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(THIS_DIR)), "GQA-LUT", "my_out", "pwl_v2.json"),
    )

    ap.add_argument("--scan_ckpt", type=str, default=os.path.join(THIS_DIR, "classification_ScanObjectNN", "checkpoints", "pointKANElite-scan_exp-42"))
    ap.add_argument("--scan_pwl", type=str, default=os.path.join(THIS_DIR, "classification_ScanObjectNN", "grf_export_out", "pwl_v2.json"))
    ap.add_argument("--scan_num_classes", type=int, default=15)

    args = ap.parse_args()
    device = torch.device(args.device)
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    def run_task(task: str, net_v, net_p, kat_types_v, kat_types_p, inputs_v_fn, inputs_p_fn):
        r_v = _time_forward_and_kat(net_v, kat_types_v, inputs_v_fn, device=device)
        r_p = _time_forward_and_kat(net_p, kat_types_p, inputs_p_fn, device=device)
        speed_total = r_v["total_forward_ms"] / max(r_p["total_forward_ms"], 1e-9)
        speed_kat = r_v["kat_forward_ms"] / max(r_p["kat_forward_ms"], 1e-9)
        return task, r_v, r_p, speed_total, speed_kat

    rows = []

    # PartSeg
    ckpt_part = get_ckpt_partseg(args.partseg_ckpt)
    net_v = build_partseg(ckpt_part, device)
    net_p = build_partseg(ckpt_part, device)
    inject_grf_pwl_v2(net_p, args.partseg_pwl, bit_key=args.bit, device=device, impl="triton")
    pts_v, nrm_v, onehot_v = _make_partseg_inputs(args.batch_size, args.partseg_points, device, dtype, args.seed)
    pts_p, nrm_p, onehot_p = _make_partseg_inputs(args.batch_size, args.partseg_points, device, dtype, args.seed)
    rows.append(
        run_task(
            "partseg",
            net_v,
            net_p,
            (KAT_Group,),
            (KAT_Group_PWL_Triton,),
            inputs_v_fn=lambda: (pts_v, nrm_v, onehot_v),
            inputs_p_fn=lambda: (pts_p, nrm_p, onehot_p),
        )
    )

    # ModelNet40
    ckpt_mn = get_ckpt_mn40(args.mn40_ckpt)
    net_v = build_mn40(ckpt_mn, device)
    net_p = build_mn40(ckpt_mn, device)
    inject_grf_pwl_v2(net_p, args.mn40_pwl, bit_key=args.bit, device=device, impl="triton")
    x_v = _make_cls_inputs(args.batch_size, args.mn40_points, device, dtype, args.seed + 1)[0]
    x_p = _make_cls_inputs(args.batch_size, args.mn40_points, device, dtype, args.seed + 1)[0]
    rows.append(
        run_task(
            "modelnet40",
            net_v,
            net_p,
            (KAT_Group,),
            (KAT_Group_PWL_Triton,),
            inputs_v_fn=lambda: (x_v,),
            inputs_p_fn=lambda: (x_p,),
        )
    )

    # ScanObjectNN
    ckpt_scan = get_ckpt_scan(args.scan_ckpt)
    net_v = build_scan(ckpt_scan, device, num_classes=args.scan_num_classes)
    net_p = build_scan(ckpt_scan, device, num_classes=args.scan_num_classes)
    inject_grf_pwl_v2(net_p, args.scan_pwl, bit_key=args.bit, device=device, impl="triton")
    x_v = _make_cls_inputs(args.batch_size, args.scan_points, device, dtype, args.seed + 2)[0]
    x_p = _make_cls_inputs(args.batch_size, args.scan_points, device, dtype, args.seed + 2)[0]
    rows.append(
        run_task(
            "scanobjectnn",
            net_v,
            net_p,
            (KAT_Group,),
            (KAT_Group_PWL_Triton,),
            inputs_v_fn=lambda: (x_v,),
            inputs_p_fn=lambda: (x_p,),
        )
    )

    print("\nKAT time fraction profiling (random inputs, 1 forward per model)")
    print("-" * 90)
    print(f"{'task':<12} {'van_total':>10} {'pwl_total':>10} {'speed_total':>11}  {'van_kat':>10} {'pwl_kat':>10} {'speed_kat':>11}")
    for task, r_v, r_p, speed_total, speed_kat in rows:
        print(
            f"{task:<12} {r_v['total_forward_ms']:>10.3f} {r_p['total_forward_ms']:>10.3f} {speed_total:>11.3f}  "
            f"{r_v['kat_forward_ms']:>10.3f} {r_p['kat_forward_ms']:>10.3f} {speed_kat:>11.3f}"
        )
    print("-" * 90)


if __name__ == "__main__":
    main()

