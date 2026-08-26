#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark vanilla GRF (KAT_Group) vs PWL 注入后 (KAT_Group_PWL) 的在线推理时间。

默认只用随机输入（不依赖数据集），尽量隔离算子/内核耗时。

支持两种基准：
  1) end-to-end：完整 pointKANElite forward 的 latency/吞吐
  2) kat_modules：仅对 KAT_Group / KAT_Group_PWL 前向计时（按 stage/op/act 维度分组）

使用方式示例：
  python benchmark_grf_pwl_online.py \
    --checkpoint checkpoints/pointKANElite_part_exp \
    --pwl_v2_json grf_export_out/pwl_v2.json \
    --bit 6 \
    --device cuda \
    --batch_size 1 \
    --num_points 2048 \
    --benchmark_mode both
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # PointKAN-elite
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import model as models  # type: ignore
from inject_grf_pwl_v2 import inject_grf_pwl_v2  # type: ignore
from kat_rational import KAT_Group, KAT_Group_PWL, KAT_Group_PWL_Triton  # type: ignore
from kat_rational.kat_1dgroup_torch import Rational_CUDA_A_1DGroup  # type: ignore


class KAT_Group_ForceTorch(torch.nn.Module):
    """
    与 KAT_Group 参数一致，但在 CUDA 上也强制走纯 PyTorch 计算路径（不走 Triton kernel）。
    目的：与 PyTorch 实现的 PWL 做“同等级实现”的公平对比。
    """

    def __init__(self, num_groups: int, weight_numerator: torch.Tensor, weight_denominator: torch.Tensor):
        super().__init__()
        self.order = (5, 4)
        self.num_groups = int(num_groups)
        # 直接复制权重，保持与原 KAT_Group 完全一致
        self.weight_numerator = torch.nn.Parameter(weight_numerator.detach().clone(), requires_grad=False)
        self.weight_denominator = torch.nn.Parameter(weight_denominator.detach().clone(), requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() in (2, 3), "Input tensor must be 3D (B,L,D) or 2D (B,D)."
        x = input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) -> (B, 1, D)
        weight_numerator = self.weight_numerator.repeat(self.num_groups, 1)
        out = Rational_CUDA_A_1DGroup(x, weight_numerator, self.weight_denominator, self.num_groups)
        return out.squeeze(1) if input.dim() == 2 else out

    def extra_repr(self) -> str:
        return f"num_groups={self.num_groups}, order={self.order}, impl=torch"


def replace_kat_group_with_torch(net: torch.nn.Module) -> int:
    """
    将模型内所有 KAT_Group（排除 PWL）替换成 KAT_Group_ForceTorch。
    Returns: 替换的模块数
    """
    replaced = 0
    for name, mod in list(net.named_modules()):
        if isinstance(mod, KAT_Group) and not isinstance(mod, KAT_Group_PWL):
            # build replacement with identical weights
            new_mod = KAT_Group_ForceTorch(
                num_groups=int(mod.num_groups),
                weight_numerator=mod.weight_numerator,
                weight_denominator=mod.weight_denominator,
            ).to(next(mod.parameters()).device)
            # assign into parent
            parts = name.split(".")
            parent = net
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_mod)
            replaced += 1
    return replaced


def _resolve_checkpoint_path(checkpoint_arg: str) -> str:
    path = os.path.abspath(os.path.expanduser(checkpoint_arg))
    if os.path.isfile(path):
        if path.endswith((".pth", ".pt")):
            return path
        raise FileNotFoundError(f"checkpoint 需为 .pth/.pt 文件: {path}")
    if os.path.isdir(path):
        for name in ("best_insiou_model.pth", "best_checkpoint.pth", "best_model.pth"):
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"目录下未找到 checkpoint: {path}")
    raise FileNotFoundError(f"checkpoint 路径不存在: {path}")


def _load_state_dict_for_model(checkpoint_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state = None
        for key in ("net", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict) and ckpt[key]:
                state = ckpt[key]
                break
        if state is None:
            state = ckpt
    else:
        # unexpected, but keep compatibility
        state = ckpt

    if not isinstance(state, dict):
        raise TypeError(f"checkpoint 里未找到 state_dict: {checkpoint_path}")

    # strip DataParallel prefix
    first_key = next(iter(state.keys()))
    if isinstance(first_key, str) and first_key.startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state  # type: ignore[return-value]


def build_model_and_load_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    # PartSeg 固定 50 part / 16 class（与原 repo 脚本一致）
    NUM_PART = 50
    net = models.__dict__["pointKANElite"](num_classes=NUM_PART)
    state = _load_state_dict_for_model(checkpoint_path, device=device)
    net.load_state_dict(state, strict=True)
    net = net.to(device)
    net.eval()
    return net


def make_random_inputs(
    batch_size: int,
    num_points: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    # point cloud / normal 一般在 [-1, 1] 或附近，避免 rational 分母数值不稳定
    points = (torch.rand((batch_size, num_points, 3), generator=g, device=device, dtype=dtype) * 2.0 - 1.0).contiguous()
    norm = (torch.rand((batch_size, num_points, 3), generator=g, device=device, dtype=dtype) * 2.0 - 1.0).contiguous()
    cls_idx = torch.randint(0, 16, (batch_size,), generator=g, device=device)
    cls_label = torch.eye(16, device=device, dtype=dtype)[cls_idx]  # (B, 16)

    # match net(points, norm_plt, cls_label) 的输入约定:
    # points: [B, 3, N], norm_plt: [B, 3, N]
    return points.transpose(2, 1).contiguous(), norm.transpose(2, 1).contiguous(), cls_label


@torch.no_grad()
def benchmark_full_model(
    net: torch.nn.Module,
    points_b3n: torch.Tensor,
    norm_b3n: torch.Tensor,
    cls_label: torch.Tensor,
    *,
    num_warmup: int,
    num_iters: int,
    device: torch.device,
):
    # torch2: make sure kernels finished before starting
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup (includes potential triton compilation / caching)
    for _ in range(num_warmup):
        _ = net(points_b3n, norm_b3n, cls_label)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    times_ms: List[float] = []
    last_out = None
    for _ in range(num_iters):
        if device.type == "cuda":
            start.record()
        else:
            t0 = time.perf_counter()
        last_out = net(points_b3n, norm_b3n, cls_label)
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        else:
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    # latency stats
    times_tensor = torch.tensor(times_ms)
    p50 = float(times_tensor.median().item())
    p90 = float(times_tensor.quantile(0.9).item()) if len(times_ms) >= 10 else max(times_ms)
    mean_ms = float(times_tensor.mean().item())

    # throughput: point clouds per second
    batch_size = int(points_b3n.size(0))
    total_time_s = sum(times_ms) / 1000.0
    ips = num_iters / total_time_s
    pts_per_s = ips * int(points_b3n.size(-1))

    return {
        "mean_ms": mean_ms,
        "p50_ms": p50,
        "p90_ms": p90,
        "ips_pointcloud_s": ips,
        "pts_per_s": pts_per_s,
        "out_shape": tuple(last_out.shape) if last_out is not None else None,
    }


def _collect_kat_modules_records(
    net: torch.nn.Module,
    *,
    kat_type: type,
    batch_size: int,
):
    # PartKANElite 的 KAT 只出现在 LFP_blocks_list[i].operation1/operation2 里（Efficient_KAN.act1/act2）。
    # 利用网络结构直接推导 input shape，避免 name parsing。
    records = []
    if not hasattr(net, "LFP_blocks_list") or not hasattr(net, "GAM_list"):
        raise RuntimeError("unexpected net structure: missing LFP_blocks_list/GAM_list")

    lfp_blocks_list = getattr(net, "LFP_blocks_list")
    gam_list = getattr(net, "GAM_list")

    for stage_i in range(len(lfp_blocks_list)):
        lfp = lfp_blocks_list[stage_i]
        gam = gam_list[stage_i]

        groups_i = int(getattr(gam, "groups"))
        knei_i = int(getattr(gam, "kneighbors"))
        # transfer1 / transfer2 的 out_channels = out_channels(stage)
        out_channels_i = int(getattr(lfp.transfer1.net[0], "out_channels"))  # type: ignore[attr-defined]

        for op_name in ("operation1", "operation2"):
            seq = getattr(lfp, op_name)
            for blk_j in range(len(seq)):
                eff = seq[blk_j]
                for act_name in ("act1", "act2"):
                    mod = getattr(eff, act_name)
                    if isinstance(mod, kat_type):
                        if op_name == "operation1":
                            # Local_Feature_Processing: after transfer1 -> (B*batches?) reshape -> x: [b*groups, kneighbors, out_channels]
                            x_shape = (batch_size * groups_i, knei_i, out_channels_i)
                        else:
                            # operation2 sees center: [b, groups, out_channels]
                            x_shape = (batch_size, groups_i, out_channels_i)
                        records.append(
                            {
                                "stage_i": stage_i,
                                "op_name": op_name,
                                "blk_j": blk_j,
                                "act_name": act_name,
                                "module": mod,
                                "x_shape": x_shape,
                            }
                        )
    return records


@torch.no_grad()
def benchmark_kat_modules(
    net: torch.nn.Module,
    *,
    kat_type: type,
    batch_size: int,
    num_points: int,  # kept for signature consistency; not used in kat-only
    device: torch.device,
    num_warmup: int,
    num_iters: int,
    seed: int = 1234,
    dtype: torch.dtype = torch.float32,
):
    records = _collect_kat_modules_records(net, kat_type=kat_type, batch_size=batch_size)
    if not records:
        raise RuntimeError(f"no kat modules found for type={kat_type}")

    # Ensure all modules have been allocated / kernels are warmed up
    torch.manual_seed(seed)

    results: List[Dict[str, object]] = []
    for rec_i, rec in enumerate(records):
        mod = rec["module"]
        x_shape = rec["x_shape"]

        # Stable input range (avoid extreme values for rational/division)
        g = torch.Generator(device=device)
        g.manual_seed(seed + rec_i * 17)
        x = (torch.rand(x_shape, generator=g, device=device, dtype=dtype) * 2.0 - 1.0).contiguous()

        # warmup per module
        for _ in range(num_warmup):
            _ = mod(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        times_ms: List[float] = []
        last_out = None
        for _ in range(num_iters):
            if device.type == "cuda":
                start.record()
            else:
                t0 = time.perf_counter()
            last_out = mod(x)
            if device.type == "cuda":
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))
            else:
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        t = torch.tensor(times_ms)
        results.append(
            {
                "stage_i": int(rec["stage_i"]),
                "op_name": str(rec["op_name"]),
                "blk_j": int(rec["blk_j"]),
                "act_name": str(rec["act_name"]),
                "x_shape": tuple(rec["x_shape"]),
                "mean_ms": float(t.mean().item()),
                "p50_ms": float(t.median().item()),
                "p90_ms": float(t.quantile(0.9).item()) if len(times_ms) >= 10 else max(times_ms),
                "out_shape": tuple(last_out.shape) if last_out is not None else None,
            }
        )

    # Aggregate quick view (by stage/op)
    agg: Dict[Tuple[int, str], List[float]] = {}
    for r in results:
        key = (int(r["stage_i"]), str(r["op_name"]))
        agg.setdefault(key, []).append(float(r["mean_ms"]))  # per-module mean_ms

    # sum of means as rough "GRF-only" proxy per forward
    agg_summary = []
    for (stage_i, op_name), ms_list in sorted(agg.items()):
        agg_summary.append(
            {
                "stage_i": stage_i,
                "op_name": op_name,
                "num_modules": len(ms_list),
                "sum_mean_ms": float(sum(ms_list)),
                "avg_mean_ms": float(sum(ms_list) / max(1, len(ms_list))),
            }
        )

    return {"per_module": results, "stage_op_summary": agg_summary}


def _fmt_table(rows: List[Dict[str, object]], cols: List[str]) -> str:
    # Minimal formatter: keep ASCII alignment, no pandas dependency.
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    lines = []
    header = "  ".join([str(c).ljust(widths[c]) for c in cols])
    lines.append(header)
    lines.append("  ".join(["-" * widths[c] for c in cols]))
    for r in rows:
        line = "  ".join([str(r.get(c, "")).ljust(widths[c]) for c in cols])
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark vanilla GRF vs PWL注入在线推理耗时")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(SCRIPT_DIR, "checkpoints", "pointKANElite_part_exp"),
        help="checkpoint 目录或 .pth/.pt 文件",
    )
    parser.add_argument(
        "--pwl_v2_json",
        type=str,
        default=os.path.join(SCRIPT_DIR, "grf_export_out", "pwl_v2.json"),
        help="PWL v2 json（用于 inject_grf_pwl_v2）",
    )
    parser.add_argument("--bit", type=str, default=None, help="layers[].bits 中的键")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16", "bf16"))
    parser.add_argument(
        "--vanilla_impl",
        type=str,
        default="triton",
        choices=("triton", "torch"),
        help="vanilla GRF 的实现：triton=默认 KAT_Group；torch=强制纯 PyTorch（不走 Triton kernel）",
    )
    parser.add_argument(
        "--pwl_impl",
        type=str,
        default="torch",
        choices=("torch", "triton", "cpp"),
        help="PWL 的实现：torch=PyTorch；triton=Triton fused kernel；cpp=CPU C++ fused（仅 CPU 有意义）",
    )

    parser.add_argument("--benchmark_mode", type=str, default="both", choices=("end_to_end", "kat_modules", "both"))

    parser.add_argument("--warmup_full", type=int, default=20)
    parser.add_argument("--iters_full", type=int, default=50)

    parser.add_argument("--warmup_kat", type=int, default=5)
    parser.add_argument("--iters_kat", type=int, default=10)

    args = parser.parse_args()

    device = torch.device(args.device)
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    pwl_json_path = os.path.abspath(os.path.expanduser(args.pwl_v2_json))
    if not os.path.isfile(pwl_json_path):
        raise FileNotFoundError(pwl_json_path)

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

    # Build vanilla model
    print("=== Build vanilla (KAT_Group) ===")
    net_vanilla = build_model_and_load_checkpoint(checkpoint_path, device)
    if args.vanilla_impl == "torch":
        n = replace_kat_group_with_torch(net_vanilla)
        print(f"vanilla_impl=torch: replaced {n} KAT_Group -> KAT_Group_ForceTorch")
    print("vanilla ready.")

    # Build PWL model
    print("=== Build PWL (inject KAT_Group -> KAT_Group_PWL) ===")
    net_pwl = build_model_and_load_checkpoint(checkpoint_path, device)
    n_replaced, bit_used = inject_grf_pwl_v2(
        net_pwl,
        pwl_json_path,
        bit_key=args.bit,
        device=device,
        impl=args.pwl_impl,
    )
    print(f"replaced: {n_replaced}, bits[{bit_used}], pwl_impl={args.pwl_impl}")
    print("pwl ready.")

    # Prepare inputs (random, default)
    points_b3n, norm_b3n, cls_label = make_random_inputs(
        args.batch_size,
        args.num_points,
        device,
        dtype=dtype if device.type == "cuda" else torch.float32,
        seed=args.seed,
    )

    if args.benchmark_mode in ("end_to_end", "both"):
        print("\n=== End-to-end benchmark (full model) ===")
        print(f"batch_size={args.batch_size}, num_points={args.num_points}")

        res_v = benchmark_full_model(
            net_vanilla,
            points_b3n,
            norm_b3n,
            cls_label,
            num_warmup=args.warmup_full,
            num_iters=args.iters_full,
            device=device,
        )
        res_p = benchmark_full_model(
            net_pwl,
            points_b3n,
            norm_b3n,
            cls_label,
            num_warmup=args.warmup_full,
            num_iters=args.iters_full,
            device=device,
        )

        print("\n[Vanilla]")
        print(f"mean_ms: {res_v['mean_ms']:.4f}  p50_ms: {res_v['p50_ms']:.4f}  p90_ms: {res_v['p90_ms']:.4f}")
        print(f"throughput: {res_v['ips_pointcloud_s']:.3f} pointcloud/s, {res_v['pts_per_s']:.3e} pts/s")
        print(f"out_shape: {res_v['out_shape']}")

        print("\n[PWL v2]")
        print(f"mean_ms: {res_p['mean_ms']:.4f}  p50_ms: {res_p['p50_ms']:.4f}  p90_ms: {res_p['p90_ms']:.4f}")
        print(f"throughput: {res_p['ips_pointcloud_s']:.3f} pointcloud/s, {res_p['pts_per_s']:.3e} pts/s")
        print(f"out_shape: {res_p['out_shape']}")

        speedup = float(res_v["mean_ms"]) / float(res_p["mean_ms"])
        print(f"\nSpeedup (vanilla_mean_ms / pwl_mean_ms): {speedup:.3f}x")

    if args.benchmark_mode in ("kat_modules", "both"):
        print("\n=== KAT-only benchmark (GRF compute proxy) ===")
        print(f"batch_size={args.batch_size} (used in input shaping), iters_kat={args.iters_kat}")

        vanilla_kat_type = KAT_Group_ForceTorch if args.vanilla_impl == "torch" else KAT_Group
        pwl_kat_type = KAT_Group_PWL_Triton if args.pwl_impl == "triton" else KAT_Group_PWL
        # Vanilla
        kat_v = benchmark_kat_modules(
            net_vanilla,
            kat_type=vanilla_kat_type,
            batch_size=args.batch_size,
            num_points=args.num_points,
            device=device,
            num_warmup=args.warmup_kat,
            num_iters=args.iters_kat,
            seed=args.seed + 100,
            dtype=dtype,
        )
        # PWL
        kat_p = benchmark_kat_modules(
            net_pwl,
            kat_type=pwl_kat_type,
            batch_size=args.batch_size,
            num_points=args.num_points,
            device=device,
            num_warmup=args.warmup_kat,
            num_iters=args.iters_kat,
            seed=args.seed + 200,
            dtype=dtype,
        )

        # Print aggregated stage/op summary
        v_rows = []
        for s in kat_v["stage_op_summary"]:
            v_rows.append(
                {
                    "stage_i": s["stage_i"],
                    "op": s["op_name"],
                    "num_modules": s["num_modules"],
                    "sum_mean_ms_vanilla": s["sum_mean_ms"],
                }
            )
        p_rows = []
        for s in kat_p["stage_op_summary"]:
            p_rows.append(
                {
                    "stage_i": s["stage_i"],
                    "op": s["op_name"],
                    "num_modules": s["num_modules"],
                    "sum_mean_ms_pwl": s["sum_mean_ms"],
                }
            )

        # Merge by (stage_i, op)
        p_map = {(int(r["stage_i"]), str(r["op"])): float(r["sum_mean_ms_pwl"]) for r in p_rows}
        merged = []
        for vr in v_rows:
            key = (int(vr["stage_i"]), str(vr["op"]))
            pm = p_map.get(key, None)
            if pm is None:
                continue
            merged.append(
                {
                    "stage_i": vr["stage_i"],
                    "op": vr["op"],
                    "num_modules": vr["num_modules"],
                    "sum_mean_ms_vanilla": vr["sum_mean_ms_vanilla"],
                    "sum_mean_ms_pwl": pm,
                    "speedup": float(vr["sum_mean_ms_vanilla"]) / float(pm) if pm > 0 else float("nan"),
                }
            )

        print("\n[Stage/Op summary: sum of mean_ms across all KAT modules matching that group]")
        print(_fmt_table(merged, ["stage_i", "op", "num_modules", "sum_mean_ms_vanilla", "sum_mean_ms_pwl", "speedup"]))

        # Optional: tell where PWL is slower/faster
        # (只给出 coarse 结论，避免输出太长)
        best = max(merged, key=lambda r: float(r["speedup"]))
        worst = min(merged, key=lambda r: float(r["speedup"]))
        print(f"\nKAT-only speedup range across stage/op: best={best['speedup']:.3f}x, worst={worst['speedup']:.3f}x")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

