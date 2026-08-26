#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三任务 latency breakdown（随机输入，在线推理，CUDA/CPU）。

目标：把 vanilla pointKAN-elite 一次 forward 的耗时拆成你关心的几类：
  - FPS (pointnet2 furthest_point_sample)
  - KNN (aten::topk)
  - CONV
  - LINEAR（含 GEMM）
  - 完整 KAT（模块级 hook 计时：所有 KAT_Group 的 forward，总体上等价 EfficientKAN 中 GRF 占比）
  - Others（除上述外的所有 op）

注意：
  1) torch.profiler 的算子时间是按 op 聚合的（可近似代表 kernel 时间分布）。
  2) KAT 用 hook 计时（模块级）单独输出，不纳入 op-bucket 的 100%，避免重复计数。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (THIS_DIR,):
    if p not in sys.path:
        sys.path.insert(0, p)

# Task-specific builders
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

from kat_rational import KAT_Group  # type: ignore


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass(frozen=True)
class Bucket:
    name: str
    patterns: Tuple[str, ...]


BUCKETS: Tuple[Bucket, ...] = (
    # FPS 在 profiler 里常见名字包括 FurthestPointSampling / furthest_point_sample*
    Bucket("fps", ("FurthestPointSampling", "furthest_point_sample", "furthest_point_sampling", "farthest_point_sampling")),
    Bucket("knn_topk", ("aten::topk",)),
    Bucket(
        "conv",
        (
            "aten::conv1d",
            "aten::conv2d",
            "aten::conv3d",
            "aten::convolution",
            "aten::_convolution",
            "aten::cudnn_convolution",
        ),
    ),
    Bucket(
        "linear",
        (
            # 线性层本体与其常见下游 GEMM
            "aten::linear",
            "aten::addmm",
            "aten::mm",
            "aten::bmm",
            "aten::matmul",
            "aten::_scaled_mm",
        ),
    ),
)


def _bucket_for_op(op_name: str) -> str:
    for b in BUCKETS:
        for p in b.patterns:
            if p in op_name:
                return b.name
    return "other"

def _event_device_us(evt) -> float:
    """
    兼容不同 torch 版本 profiler 字段：
      - torch>=2.1: device_time_total (us)
      - torch<=2.0: cuda_time_total (us) 或 cuda_time (us, deprecated)
    """
    v = getattr(evt, "device_time_total", None)
    if v is not None:
        return float(v)
    v = getattr(evt, "cuda_time_total", None)
    if v is not None:
        return float(v)
    v = getattr(evt, "cuda_time", None)
    if v is not None:
        return float(v)
    return 0.0


@torch.no_grad()
def _bench_total_ms(fn, *, device: torch.device, warmup: int, iters: int) -> float:
    _cuda_sync(device)
    for _ in range(warmup):
        _ = fn()
    _cuda_sync(device)

    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    for _ in range(iters):
        if device.type == "cuda":
            assert start is not None and end is not None
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
        else:
            import time

            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
    return float(torch.tensor(times).mean().item())

def _patch_knn_point_for_timing(tracker: Dict[str, object]) -> List[Tuple[object, str, object]]:
    """
    Monkey-patch knn_point(...) in known model modules so we can time the full
    KNN path (square_distance + topk) as executed by the model.

    Returns a list of (module_obj, attr_name, original_fn) for restoration.
    """

    def _wrap(fn):
        def wrapped(*args, **kwargs):
            dev = tracker.get("device")
            if isinstance(dev, torch.device) and dev.type == "cuda":
                e_s = torch.cuda.Event(enable_timing=True)
                e_e = torch.cuda.Event(enable_timing=True)
                e_s.record()
                out = fn(*args, **kwargs)
                e_e.record()
                tracker.setdefault("pairs", []).append((e_s, e_e))  # type: ignore[arg-type]
                return out
            return fn(*args, **kwargs)

        return wrapped

    patched: List[Tuple[object, str, object]] = []

    # First, try patching known module paths.
    candidate_modules = [
        "part_segmentation.model.pointKANElite",
        "classification_ModelNet40.models.pointkanElite",
        "classification_ModelNet40.models.pointkanElite_fps_opt",
        "classification_ScanObjectNN.models.pointkanElite",
        "classification_ScanObjectNN.models.pointkanElite_fps_opt",
        "classification_ModelNet40_Fewshot.models.pointkanElite",
    ]
    for mod_name in candidate_modules:
        mod = sys.modules.get(mod_name, None)
        if mod is None:
            try:
                __import__(mod_name)
                mod = sys.modules.get(mod_name, None)
            except Exception:
                mod = None
        if mod is None:
            continue
        if hasattr(mod, "knn_point"):
            orig = getattr(mod, "knn_point")
            setattr(mod, "knn_point", _wrap(orig))
            patched.append((mod, "knn_point", orig))

    # Fallback: patch any already-imported module in this repo that defines knn_point.
    # This avoids missing cases due to different import paths/aliases.
    for mod in list(sys.modules.values()):
        if mod is None or not hasattr(mod, "__file__"):
            continue
        f = getattr(mod, "__file__", "") or ""
        if "PointKAN-elite" not in f:
            continue
        if "pointkanElite" not in f and "pointKANElite" not in f:
            continue
        if not hasattr(mod, "knn_point"):
            continue
        # Skip if already patched via candidate_modules
        if any(mod is m for (m, _a, _o) in patched):
            continue
        orig = getattr(mod, "knn_point")
        setattr(mod, "knn_point", _wrap(orig))
        patched.append((mod, "knn_point", orig))

    return patched


@torch.no_grad()
def _knn_full_hook_time_ms(net: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], *, device: torch.device) -> float:
    """
    Full KNN timing for knn_point (distance + topk) via temporary monkey-patch.
    """
    if device.type != "cuda":
        return 0.0

    tracker: Dict[str, object] = {"device": device, "pairs": []}
    patched = _patch_knn_point_for_timing(tracker)
    try:
        _ = net(*inputs)  # warm compile
        _cuda_sync(device)
        tracker["pairs"] = []
        _ = net(*inputs)
        _cuda_sync(device)
    finally:
        for mod, attr, orig in patched:
            setattr(mod, attr, orig)

    ms = 0.0
    for e_s, e_e in tracker.get("pairs", []):  # type: ignore[assignment]
        ms += float(e_s.elapsed_time(e_e))
    return ms


def _patch_square_distance_for_timing(tracker: Dict[str, object]) -> List[Tuple[object, str, object]]:
    """
    Patch square_distance(...) to time KNN distance computation only (excluding topk).
    Returns restoration list.
    """

    def _wrap(fn):
        def wrapped(*args, **kwargs):
            dev = tracker.get("device")
            if isinstance(dev, torch.device) and dev.type == "cuda":
                e_s = torch.cuda.Event(enable_timing=True)
                e_e = torch.cuda.Event(enable_timing=True)
                e_s.record()
                out = fn(*args, **kwargs)
                e_e.record()
                tracker.setdefault("pairs", []).append((e_s, e_e))  # type: ignore[arg-type]
                return out
            return fn(*args, **kwargs)

        return wrapped

    patched: List[Tuple[object, str, object]] = []
    # Known module paths where square_distance is defined
    candidate_modules = [
        "part_segmentation.model.pointKANElite",
        "classification_ModelNet40.models.pointkanElite",
        "classification_ModelNet40.models.pointkanElite_fps_opt",
        "classification_ScanObjectNN.models.pointkanElite",
        "classification_ScanObjectNN.models.pointkanElite_fps_opt",
        "classification_ModelNet40_Fewshot.models.pointkanElite",
    ]
    for mod_name in candidate_modules:
        mod = sys.modules.get(mod_name, None)
        if mod is None:
            try:
                __import__(mod_name)
                mod = sys.modules.get(mod_name, None)
            except Exception:
                mod = None
        if mod is None:
            continue
        if hasattr(mod, "square_distance"):
            orig = getattr(mod, "square_distance")
            setattr(mod, "square_distance", _wrap(orig))
            patched.append((mod, "square_distance", orig))

    # Fallback: patch any already-imported module in this repo that defines square_distance
    for mod in list(sys.modules.values()):
        if mod is None or not hasattr(mod, "__file__"):
            continue
        f = getattr(mod, "__file__", "") or ""
        if "PointKAN-elite" not in f:
            continue
        if "pointkanElite" not in f and "pointKANElite" not in f:
            continue
        if not hasattr(mod, "square_distance"):
            continue
        if any(mod is m for (m, _a, _o) in patched):
            continue
        orig = getattr(mod, "square_distance")
        setattr(mod, "square_distance", _wrap(orig))
        patched.append((mod, "square_distance", orig))

    return patched


@torch.no_grad()
def _knn_dist_hook_time_ms(net: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], *, device: torch.device) -> float:
    """
    Time KNN distance computation (square_distance only) via temporary patching.
    """
    if device.type != "cuda":
        return 0.0

    tracker: Dict[str, object] = {"device": device, "pairs": []}
    patched = _patch_square_distance_for_timing(tracker)
    try:
        _ = net(*inputs)  # warm compile
        _cuda_sync(device)
        tracker["pairs"] = []
        _ = net(*inputs)
        _cuda_sync(device)
    finally:
        for mod, attr, orig in patched:
            setattr(mod, attr, orig)

    ms = 0.0
    for e_s, e_e in tracker.get("pairs", []):  # type: ignore[assignment]
        ms += float(e_s.elapsed_time(e_e))
    return ms


@torch.no_grad()
def _kat_hook_time_ms(net: torch.nn.Module, kat_types: Tuple[type, ...], inputs: Tuple[torch.Tensor, ...], *, device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0

    _ = net(*inputs)  # warm compile
    _cuda_sync(device)

    starts: Dict[int, torch.cuda.Event] = {}
    pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def _pre(m, _inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        starts[id(m)] = e

    def _post(m, _inp, _out):
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        e_start = starts.pop(id(m), None)
        if e_start is not None:
            pairs.append((e_start, e_end))

    hooks = []
    for _name, mod in net.named_modules():
        if isinstance(mod, kat_types):
            hooks.append(mod.register_forward_pre_hook(_pre))
            hooks.append(mod.register_forward_hook(_post))

    _ = net(*inputs)
    _cuda_sync(device)

    for h in hooks:
        h.remove()

    ms = 0.0
    for s, e in pairs:
        ms += float(s.elapsed_time(e))
    return ms


@torch.no_grad()
def _kat_hook_time_and_elems(
    net: torch.nn.Module, kat_types: Tuple[type, ...], inputs: Tuple[torch.Tensor, ...], *, device: torch.device
) -> Tuple[float, int]:
    """
    Return (kat_ms, kat_input_elems) where kat_input_elems is the sum of input.numel()
    over all matched KAT modules for one forward. This enables simple roofline estimates.
    """
    if device.type != "cuda":
        return 0.0, 0

    _ = net(*inputs)  # warm compile
    _cuda_sync(device)

    starts: Dict[int, torch.cuda.Event] = {}
    pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    elems = 0

    def _pre(m, _inp):
        nonlocal elems
        # _inp is a tuple; KAT forward takes one tensor
        if _inp and isinstance(_inp[0], torch.Tensor):
            elems += int(_inp[0].numel())
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        starts[id(m)] = e

    def _post(m, _inp, _out):
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        e_start = starts.pop(id(m), None)
        if e_start is not None:
            pairs.append((e_start, e_end))

    hooks = []
    for _name, mod in net.named_modules():
        if isinstance(mod, kat_types):
            hooks.append(mod.register_forward_pre_hook(_pre))
            hooks.append(mod.register_forward_hook(_post))

    _ = net(*inputs)
    _cuda_sync(device)

    for h in hooks:
        h.remove()

    ms = 0.0
    for s, e in pairs:
        ms += float(s.elapsed_time(e))
    return ms, int(elems)


@torch.no_grad()
def _effkan_time_and_shapes(net: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], *, device: torch.device) -> Tuple[float, List[Tuple[int, int, int]]]:
    """
    Time all Efficient_KAN modules (includes linears + KAT) and collect their input shapes.
    Returns: (total_ms, [(B', L, D), ...]) for one forward.
    """
    if device.type != "cuda":
        return 0.0, []

    _ = net(*inputs)  # warm compile
    _cuda_sync(device)

    starts: Dict[int, torch.cuda.Event] = {}
    pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
    shapes: List[Tuple[int, int, int]] = []

    def _pre(m, _inp):
        if _inp and isinstance(_inp[0], torch.Tensor):
            x = _inp[0]
            if x.dim() == 3:
                shapes.append((int(x.shape[0]), int(x.shape[1]), int(x.shape[2])))
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        starts[id(m)] = e

    def _post(m, _inp, _out):
        e_end = torch.cuda.Event(enable_timing=True)
        e_end.record()
        e_start = starts.pop(id(m), None)
        if e_start is not None:
            pairs.append((e_start, e_end))

    hooks = []
    for _name, mod in net.named_modules():
        # Avoid importing model classes; match by class name.
        if mod.__class__.__name__ == "Efficient_KAN":
            hooks.append(mod.register_forward_pre_hook(_pre))
            hooks.append(mod.register_forward_hook(_post))

    _ = net(*inputs)
    _cuda_sync(device)

    for h in hooks:
        h.remove()

    ms = 0.0
    for s, e in pairs:
        ms += float(s.elapsed_time(e))
    return ms, shapes


@torch.no_grad()
def _profile_op_buckets(
    fn,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    """
    返回：各 bucket 的 CUDA 时间占比（0~1），以及 total_cuda_ms（用于 sanity check）。
    另外附带每个 bucket 的绝对 CUDA 时间（ms）：{bucket}_cuda_ms。
    """
    for _ in range(warmup):
        _ = fn()
    _cuda_sync(device)

    if device.type != "cuda":
        return {
            "fps": 0.0,
            "knn_topk": 0.0,
            "linear": 0.0,
            "conv": 0.0,
            "other": 1.0,
            "total_cuda_ms": 0.0,
        }

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(iters):
            _ = fn()
            prof.step()
    _cuda_sync(device)

    bucket_ms: Dict[str, float] = {
        "fps": 0.0,
        "knn_topk": 0.0,
        "conv": 0.0,
        "linear": 0.0,
        "other": 0.0,
    }
    total_cuda_ms = 0.0
    for evt in prof.key_averages():
        name = str(evt.key)
        dev_us = _event_device_us(evt)
        dev_ms = dev_us / 1000.0  # us -> ms
        if dev_ms <= 0:
            continue
        total_cuda_ms += dev_ms
        bucket = _bucket_for_op(name)
        bucket_ms[bucket] = bucket_ms.get(bucket, 0.0) + dev_ms

    if total_cuda_ms <= 0:
        out0 = {**{k: 0.0 for k in bucket_ms.keys()}, "total_cuda_ms": 0.0}
        for k in bucket_ms.keys():
            out0[f"{k}_cuda_ms"] = 0.0
        return out0

    out = {k: float(v / total_cuda_ms) for k, v in bucket_ms.items()}
    out["total_cuda_ms"] = float(total_cuda_ms)
    for k, v in bucket_ms.items():
        out[f"{k}_cuda_ms"] = float(v)
    return out

@torch.no_grad()
def _profile_other_top_ops(
    fn,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    topk: int,
) -> List[Tuple[str, float]]:
    """
    返回 other bucket 中占用 CUDA 时间最多的 op 列表: [(op_name, frac_of_total_cuda), ...]
    """
    if device.type != "cuda" or topk <= 0:
        return []
    for _ in range(warmup):
        _ = fn()
    _cuda_sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        for _ in range(iters):
            _ = fn()
            prof.step()
    _cuda_sync(device)

    op_ms: List[Tuple[str, float]] = []
    total_ms = 0.0
    for evt in prof.key_averages():
        name = str(evt.key)
        dev_ms = _event_device_us(evt) / 1000.0
        if dev_ms <= 0:
            continue
        total_ms += dev_ms
        if _bucket_for_op(name) == "other":
            op_ms.append((name, dev_ms))
    if total_ms <= 0:
        return []
    op_ms.sort(key=lambda x: x[1], reverse=True)
    return [(n, float(ms / total_ms)) for (n, ms) in op_ms[:topk]]

def _make_partseg_inputs(batch: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pts = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    nrm = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    cls_idx = torch.randint(0, 16, (batch,), generator=g, device=device)
    onehot = torch.eye(16, device=device, dtype=dtype)[cls_idx]
    return pts, nrm, onehot


def _make_cls_inputs(batch: int, npts: int, device: torch.device, dtype: torch.dtype, seed: int) -> Tuple[torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    pts = (torch.rand((batch, npts, 3), generator=g, device=device, dtype=dtype) * 2 - 1).transpose(2, 1).contiguous()
    return (pts,)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:6.2f}%"


def _print_breakdown(task: str, total_ms: float, op_buckets: Dict[str, float], kat_ms: float, knn_full_ms: float) -> None:
    print(f"\n[{task} | vanilla]")
    print(f"total_forward_mean_ms: {total_ms:.3f}")
    if "total_cuda_ms" in op_buckets:
        print(f"profiler_total_cuda_ms (sum key_averages): {op_buckets['total_cuda_ms']:.3f}")
    print("op-bucket fraction (CUDA):")
    for k, label in (
        ("fps", "FPS"),
        ("knn_topk", "KNN(topk)"),
        ("conv", "CONV"),
        ("linear", "LINEAR"),
        ("other", "Others"),
    ):
        print(f"  - {label:<14} {_fmt_pct(float(op_buckets.get(k, 0.0)))}")
    if kat_ms > 0:
        print(f"kat_hook_ms: {kat_ms:.3f}  (KAT_fraction_vs_total: {kat_ms / max(total_ms, 1e-9):.4f})")
        print("  注：KAT 为模块级 hook 计时，未计入上面的 op-bucket 100%。")
    if knn_full_ms > 0:
        print(f"knn_full_hook_ms: {knn_full_ms:.3f}  (KNN_full_fraction_vs_total: {knn_full_ms / max(total_ms, 1e-9):.4f})")
        print("  注：KNN(full) 为 knn_point(distance+topk) 模块级计时，会与 LINEAR/Others 的 profiler 桶存在重叠；因此不纳入 6-way。")

    # 6-way 视图：保持“主要桶来自 profiler + KAT(full) 来自 hook”的混合口径。
    # KNN(full) 会与 LINEAR/Others 重叠，因此这里仍使用 KNN(topk) 作为 KNN 桶。
    fps = float(op_buckets.get("fps", 0.0))
    knn = float(op_buckets.get("knn_topk", 0.0))
    conv = float(op_buckets.get("conv", 0.0))
    lin = float(op_buckets.get("linear", 0.0))
    kat_frac = (kat_ms / max(total_ms, 1e-9)) if kat_ms > 0 else 0.0
    others_excl_kat = max(0.0, 1.0 - (fps + knn + conv + lin + kat_frac))
    print("6-way breakdown (hybrid, sum=100%):")
    print(f"  - {'FPS':<14} {_fmt_pct(fps)}")
    print(f"  - {'KNN(topk)':<14} {_fmt_pct(knn)}")
    print(f"  - {'CONV':<14} {_fmt_pct(conv)}")
    print(f"  - {'LINEAR':<14} {_fmt_pct(lin)}")
    print(f"  - {'KAT(full)':<14} {_fmt_pct(kat_frac)}")
    print(f"  - {'Others':<14} {_fmt_pct(others_excl_kat)}")

def _print_csv_row(task: str, total_ms: float, op_buckets: Dict[str, float], kat_ms: float, knn_full_ms: float) -> None:
    # 便于直接粘到表格/画图工具
    fps = float(op_buckets.get("fps", 0.0))
    knn = float(op_buckets.get("knn_topk", 0.0))
    lin = float(op_buckets.get("linear", 0.0))
    conv = float(op_buckets.get("conv", 0.0))
    other = float(op_buckets.get("other", 0.0))
    kat_frac = (kat_ms / max(total_ms, 1e-9)) if kat_ms > 0 else 0.0
    print(
        "CSV,"
        f"{task},{total_ms:.6f},"
        f"{fps:.8f},{knn:.8f},{conv:.8f},{lin:.8f},{other:.8f},"
        f"{kat_ms:.6f},{kat_frac:.8f}"
    )

    # 另一行：6-way hybrid（直接可视化用）
    others_excl_kat = max(0.0, 1.0 - (fps + knn + conv + lin + kat_frac))
    print(
        "CSV6,"
        f"{task},{total_ms:.6f},"
        f"{fps:.8f},{knn:.8f},{conv:.8f},{lin:.8f},{kat_frac:.8f},{others_excl_kat:.8f}"
    )

    # 额外一行：把 KNN(full) 也导出（不参与 6-way 求和，只用于观察 KNN distance+topk 的整体占比）
    knn_full_frac = (knn_full_ms / max(total_ms, 1e-9)) if knn_full_ms > 0 else 0.0
    print(f"CSVKNNFULL,{task},{total_ms:.6f},{knn_full_ms:.6f},{knn_full_frac:.8f}")

def main() -> None:
    ap = argparse.ArgumentParser(description="PointKAN-elite (vanilla) 三任务 latency breakdown (FPS/KNN/KAT/Other)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--task", type=str, default="all", choices=("all", "partseg", "modelnet40", "scanobjectnn"))

    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20, help="profiling iters (prof.step per iter)")
    ap.add_argument("--bench_warmup", type=int, default=20)
    ap.add_argument("--bench_iters", type=int, default=200)
    ap.add_argument("--topk_other", type=int, default=0, help="若>0，额外打印 other bucket 的 TOP-K op（按 CUDA 时间占比）")

    ap.add_argument("--partseg_points", type=int, default=2048)
    ap.add_argument("--mn40_points", type=int, default=1024)
    ap.add_argument("--scan_points", type=int, default=1024)
    ap.add_argument(
        "--scan_points_sweep",
        type=str,
        default=None,
        help="例如 '512,1024,2048,4096'。若提供，则仅对 scanobjectnn 按 points sweep 输出 CSV6P 行。",
    )
    ap.add_argument("--scan_num_classes", type=int, default=15)

    ap.add_argument("--partseg_ckpt", type=str, default=os.path.join(THIS_DIR, "part_segmentation", "checkpoints", "pointKANElite_part_exp"))
    ap.add_argument("--mn40_ckpt", type=str, default=os.path.join(THIS_DIR, "classification_ModelNet40", "checkpoints", "pointKANElite-pointkan_elite_exp-42"))

    ap.add_argument("--scan_ckpt", type=str, default=os.path.join(THIS_DIR, "classification_ScanObjectNN", "checkpoints", "pointKANElite-scan_exp-42"))

    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "fp32" else (torch.float16 if args.dtype == "fp16" else torch.bfloat16)

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

    def run_one_task(
        *,
        task: str,
        build_fn,
        ckpt_path: str,
        inputs: Tuple[torch.Tensor, ...],
        kat_types_v: Tuple[type, ...] = (KAT_Group,),
    ) -> None:
        net_v = build_fn(ckpt_path, device)

        fn_v = lambda: net_v(*inputs)

        total_v = _bench_total_ms(fn_v, device=device, warmup=args.bench_warmup, iters=args.bench_iters)
        op_v = _profile_op_buckets(fn_v, device=device, warmup=args.warmup, iters=args.iters)
        kat_v = _kat_hook_time_ms(net_v, kat_types_v, inputs, device=device)
        knn_full_ms = _knn_full_hook_time_ms(net_v, inputs, device=device)

        _print_breakdown(task, total_v, op_v, kat_v, knn_full_ms)
        _print_csv_row(task, total_v, op_v, kat_v, knn_full_ms)
        if args.topk_other > 0 and device.type == "cuda":
            top_ops = _profile_other_top_ops(fn_v, device=device, warmup=args.warmup, iters=args.iters, topk=args.topk_other)
            if top_ops:
                print(f"[{task}] other bucket top-{args.topk_other} ops (frac_of_total_cuda):")
                for name, frac in top_ops:
                    print(f"  - {name:<35} {_fmt_pct(frac)}")

    if args.task in ("all", "partseg"):
        ckpt_part = get_ckpt_partseg(args.partseg_ckpt)
        inputs_part = _make_partseg_inputs(args.batch_size, args.partseg_points, device, dtype, args.seed)
        run_one_task(task="partseg", build_fn=build_partseg, ckpt_path=ckpt_part, inputs=inputs_part)

    if args.task in ("all", "modelnet40"):
        ckpt_mn40 = get_ckpt_mn40(args.mn40_ckpt)
        inputs_mn40 = _make_cls_inputs(args.batch_size, args.mn40_points, device, dtype, args.seed + 1)
        run_one_task(task="modelnet40", build_fn=build_mn40, ckpt_path=ckpt_mn40, inputs=inputs_mn40)

    if args.task in ("all", "scanobjectnn"):
        ckpt_scan = get_ckpt_scan(args.scan_ckpt)

        def _build_scan(pth: str, dev: torch.device):
            return build_scan(pth, dev, num_classes=args.scan_num_classes)

        def _extract_stage0_gk(net: torch.nn.Module) -> Tuple[int, int]:
            # Most PointKAN-elite variants expose GAM_list[stage].groups/kneighbors
            try:
                gam_list = getattr(net, "GAM_list")
                gam0 = gam_list[0]
                g = int(getattr(gam0, "groups"))
                k = int(getattr(gam0, "kneighbors"))
                return g, k
            except Exception:
                return -1, -1

        def _extract_all_gk(net: torch.nn.Module) -> Tuple[List[int], List[int]]:
            try:
                gam_list = getattr(net, "GAM_list")
                gs = [int(getattr(g, "groups")) for g in gam_list]
                ks = [int(getattr(g, "kneighbors")) for g in gam_list]
                return gs, ks
            except Exception:
                return [], []

        if args.scan_points_sweep:
            # points sweep mode: print a compact CSV6P for plotting
            pts_list = [int(x) for x in args.scan_points_sweep.split(",") if x.strip()]
            print("\n=== scanobjectnn points sweep ===")
            print("CSV6P,task,num_points,total_ms,fps,knn,conv,linear,kat,others")
            print("CSVBUCKETMSP,task,num_points,total_cuda_ms,fps_ms,knn_topk_ms,conv_ms,linear_ms,other_ms")
            print("CSVMETA,task,num_points,G0,K0")
            print("CSVKATMETAP,task,num_points,kat_ms,kat_frac,kat_elems")
            print("CSVKNNDISTP,task,num_points,knn_dist_ms,knn_dist_frac")
            print("CSVMETAALL,task,num_points,G_list,K_list")
            print("CSVKATEFFP,task,num_points,eff_ms,eff_frac,eff_flops,eff_bytes,eff_ai")
            for i, npts in enumerate(pts_list):
                inputs_scan = _make_cls_inputs(args.batch_size, int(npts), device, dtype, args.seed + 2 + i)[0]
                net_v = _build_scan(ckpt_scan, device)
                fn_v = lambda: net_v(inputs_scan)
                total_v = _bench_total_ms(fn_v, device=device, warmup=args.bench_warmup, iters=args.bench_iters)
                op_v = _profile_op_buckets(fn_v, device=device, warmup=args.warmup, iters=args.iters)
                kat_v, kat_elems = _kat_hook_time_and_elems(net_v, (KAT_Group,), (inputs_scan,), device=device)
                knn_full_ms = _knn_full_hook_time_ms(net_v, (inputs_scan,), device=device)
                knn_dist_ms = _knn_dist_hook_time_ms(net_v, (inputs_scan,), device=device)
                eff_ms, eff_shapes = _effkan_time_and_shapes(net_v, (inputs_scan,), device=device)

                fps = float(op_v.get("fps", 0.0))
                knn = float(op_v.get("knn_topk", 0.0))
                conv = float(op_v.get("conv", 0.0))
                linear = float(op_v.get("linear", 0.0))
                kat_frac = (kat_v / max(total_v, 1e-9)) if kat_v > 0 else 0.0
                others = max(0.0, 1.0 - (fps + knn + conv + linear + kat_frac))
                print(
                    f"CSV6P,scanobjectnn,{int(npts)},{total_v:.6f},"
                    f"{fps:.8f},{knn:.8f},{conv:.8f},{linear:.8f},{kat_frac:.8f},{others:.8f}"
                )
                knn_full_frac = (knn_full_ms / max(total_v, 1e-9)) if knn_full_ms > 0 else 0.0
                print(f"CSVKNNFULLP,scanobjectnn,{int(npts)},{knn_full_ms:.6f},{knn_full_frac:.8f}")
                knn_dist_frac = (knn_dist_ms / max(total_v, 1e-9)) if knn_dist_ms > 0 else 0.0
                print(f"CSVKNNDISTP,scanobjectnn,{int(npts)},{knn_dist_ms:.6f},{knn_dist_frac:.8f}")
                print(
                    f"CSVBUCKETMSP,scanobjectnn,{int(npts)},{float(op_v.get('total_cuda_ms', 0.0)):.6f},"
                    f"{float(op_v.get('fps_cuda_ms', 0.0)):.6f},"
                    f"{float(op_v.get('knn_topk_cuda_ms', 0.0)):.6f},"
                    f"{float(op_v.get('conv_cuda_ms', 0.0)):.6f},"
                    f"{float(op_v.get('linear_cuda_ms', 0.0)):.6f},"
                    f"{float(op_v.get('other_cuda_ms', 0.0)):.6f}"
                )
                g0, k0 = _extract_stage0_gk(net_v)
                print(f"CSVMETA,scanobjectnn,{int(npts)},{g0},{k0}")
                print(f"CSVKATMETAP,scanobjectnn,{int(npts)},{kat_v:.6f},{kat_frac:.8f},{int(kat_elems)}")
                gs, ks = _extract_all_gk(net_v)
                print(f"CSVMETAALL,scanobjectnn,{int(npts)},{'|'.join(map(str, gs))},{'|'.join(map(str, ks))}")

                # Strict-ish Efficient_KAN operator model (per call)
                # FLOPs per Efficient_KAN(B',L,D): 4*B'*L*D^2 + 38*B'*L*D
                # Bytes per call (simple model):
                #   2x KAT_Group: 96 * B'*L*D bytes
                #   2x Linear(D->D): 16 * B'*L*D + 8 * D^2 bytes
                # => 112 * B'*L*D + 8 * D^2
                eff_flops = 0.0
                eff_bytes = 0.0
                for b2, l2, d2 in eff_shapes:
                    eff_flops += float(4.0 * b2 * l2 * d2 * d2 + 38.0 * b2 * l2 * d2)
                    eff_bytes += float(112.0 * b2 * l2 * d2 + 8.0 * d2 * d2)
                eff_frac = (eff_ms / max(total_v, 1e-9)) if eff_ms > 0 else 0.0
                eff_ai = (eff_flops / eff_bytes) if eff_bytes > 0 else 0.0
                print(
                    f"CSVKATEFFP,scanobjectnn,{int(npts)},{eff_ms:.6f},{eff_frac:.8f},"
                    f"{eff_flops:.0f},{eff_bytes:.0f},{eff_ai:.6e}"
                )
            return

        inputs_scan = _make_cls_inputs(args.batch_size, args.scan_points, device, dtype, args.seed + 2)
        run_one_task(task="scanobjectnn", build_fn=_build_scan, ckpt_path=ckpt_scan, inputs=inputs_scan)


if __name__ == "__main__":
    main()

