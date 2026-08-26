#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一随机推理入口：PointKAN-elite、ISEDA 移植网络、openpoints_mini（PointNeXt / PointVector）。
不依赖 checkpoint。在 PROFILE_RUN_DIR 下写出前向切片 JSON，供外部 profiler 将 NVML 曲线对齐到
model(x) 的 latency / energy（脚本本身只计时与 pyRAPL 包住的是纯 forward）。

任务 (--task)::
  modelnet40 | scanobjectnn | part_seg     — PointKAN（原行为不变）
  isda_<name>                             — ISEDA，name 见 isda_models.ISDA_MODEL_NAMES
  openpoints_cls | openpoints_part | openpoints_sem — 须配合 --backbone pointnext|pointvector

openpoints 需将 PointKAN-pytorch 根目录置于 sys.path（本脚本自动加入上级仓库根）。
首次 GPU 建议: bash openpoints_mini/install_pointnet2_cuda.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

_elite_next_to_demo = os.path.join(SCRIPT_DIR, "PointKAN-elite")
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
ELITE_ROOT = os.environ.get(
    "POINTKAN_ELITE_ROOT",
    _elite_next_to_demo if os.path.isdir(_elite_next_to_demo) else os.path.join(_REPO_ROOT, "PointKAN-elite"),
)
TASK_DIRS = {
    "modelnet40": os.path.join(ELITE_ROOT, "classification_ModelNet40"),
    "scanobjectnn": os.path.join(ELITE_ROOT, "classification_ScanObjectNN"),
    "part_seg": os.path.join(ELITE_ROOT, "part_segmentation"),
}

from isda_models import ISDA_MODEL_NAMES  # noqa: E402

ISDA_TASKS = tuple(f"isda_{m}" for m in ISDA_MODEL_NAMES)
OPENPOINTS_TASKS = ("openpoints_cls", "openpoints_part", "openpoints_sem")
POINTKAN_TASKS = ("modelnet40", "scanobjectnn", "part_seg")
TASK_CHOICES = POINTKAN_TASKS + ISDA_TASKS + OPENPOINTS_TASKS


def _emit_result(args, n: int, infer_only_s: float, cpu_energy: dict | None, extra: dict | None = None):
    tail = f"task={args.task} device={args.device} batches={n}"
    if getattr(args, "backbone", None):
        tail += f" backbone={args.backbone}"
    per = (infer_only_s / n) if n else 0.0
    print(
        f"run_infer_random: {tail} model_forward_total_s={infer_only_s:.6f} "
        f"model_forward_per_call_s={per:.6f} (sum/mean of {n}× model forward only)",
        file=sys.stderr,
    )
    if cpu_energy:
        pkg = cpu_energy.get("cpu_energy_pkg_j") or 0
        ram = cpu_energy.get("cpu_energy_ram_j") or 0
        total = cpu_energy.get("cpu_energy_total_j") or 0
        print(
            f"run_infer_random: cpu_energy_inference pkg={pkg:.4f}J ram={ram:.4f}J total={total:.4f}J",
            file=sys.stderr,
        )
    run_dir = os.environ.get("PROFILE_RUN_DIR")
    run_tag = os.environ.get("PROFILE_RUN_TAG")
    if run_dir and run_tag:
        path = os.path.join(run_dir, f"{run_tag}_inference_only.json")
        try:
            os.makedirs(run_dir, exist_ok=True)
            out = {
                "inference_only_s": infer_only_s,
                "inference_only_per_forward_s": per,
                "num_batches": n,
                "task": args.task,
            }
            if getattr(args, "backbone", None):
                out["backbone"] = args.backbone
            if extra:
                out.update(extra)
            if cpu_energy:
                out["cpu_energy_inference_pkg_j"] = cpu_energy.get("cpu_energy_pkg_j")
                out["cpu_energy_inference_ram_j"] = cpu_energy.get("cpu_energy_ram_j")
                out["cpu_energy_inference_total_j"] = cpu_energy.get("cpu_energy_total_j")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f)
        except Exception:
            pass


def _prepare_openpoints_repo():
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    try:
        from openpoints_mini.cpp import pointnet2_cuda  # noqa: F401
    except Exception:
        pass
    import openpoints_mini.models  # noqa: F401


def _openpoints_sem_batch(pts_b_n_9, in_channels: int):
    pos = pts_b_n_9[:, :, :3].contiguous()
    feat = pts_b_n_9[:, :, :in_channels].float().transpose(1, 2).contiguous()
    return {"pos": pos, "x": feat}


def run_modelnet40_random(args):
    task_dir = TASK_DIRS["modelnet40"]
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if ELITE_ROOT not in sys.path:
        sys.path.insert(0, ELITE_ROOT)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    import torch
    import models as models
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    net = models.__dict__["pointKANElite"]().to(device).eval()

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                x = torch.randn(args.batch_size, 3, args.num_points, device=device, dtype=torch.float32)
                infer_total_s += time_forward_seconds(device, lambda: net(x))
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)


def run_scanobjectnn_random(args):
    task_dir = TASK_DIRS["scanobjectnn"]
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if ELITE_ROOT not in sys.path:
        sys.path.insert(0, ELITE_ROOT)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    import torch
    import models as models
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    net = models.__dict__["pointKANElite"](num_classes=args.num_classes).to(device).eval()

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                x = torch.randn(args.batch_size, 3, args.num_points, device=device, dtype=torch.float32)
                infer_total_s += time_forward_seconds(device, lambda: net(x))
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)


def run_part_seg_random(args):
    task_dir = TASK_DIRS["part_seg"]
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if ELITE_ROOT not in sys.path:
        sys.path.insert(0, ELITE_ROOT)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    import torch
    import model as models
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    NUM_CLASSES = 16
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    net = models.__dict__["pointKANElite"](num_classes=50).to(device).eval()

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                points = torch.randn(args.batch_size, 3, args.num_points, device=device, dtype=torch.float32)
                norm_plt = torch.randn(args.batch_size, 3, args.num_points, device=device, dtype=torch.float32)
                onehot = torch.zeros(args.batch_size, NUM_CLASSES, device=device, dtype=torch.float32)
                onehot.scatter_(1, torch.randint(0, NUM_CLASSES, (args.batch_size, 1), device=device), 1.0)
                infer_total_s += time_forward_seconds(device, lambda: net(points, norm_plt, onehot))
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)


def run_isda(args):
    import torch
    from isda_models.benchmark import run_benchmark

    device = torch.device(args.device)
    model = args.task[len("isda_") :]
    return run_benchmark(
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        num_points=args.num_points,
        num_classes=args.num_classes,
        seed=args.seed,
    )


def run_openpoints_cls(args):
    import torch
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    _prepare_openpoints_repo()
    from openpoints_mini.scripts.model_builder import build_modelnet40_cls_model
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    net = build_modelnet40_cls_model(args.backbone, device).eval()
    B, N = args.batch_size, args.num_points

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                pos = torch.randn(B, N, 3, device=device, dtype=torch.float32)

                def _one():
                    net(pos)

                infer_total_s += time_forward_seconds(device, _one)
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)


def run_openpoints_part(args):
    import torch
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    _prepare_openpoints_repo()
    from openpoints_mini.scripts.model_builder import build_shapenet_part_model
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    in_ch = args.part_in_channels
    net = build_shapenet_part_model(args.backbone, device, in_channels=in_ch).eval()
    B, N = args.batch_size, args.num_points

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                pts = torch.randn(B, N, in_ch, device=device, dtype=torch.float32)
                pos = pts[:, :, :3].contiguous()
                x = pts.transpose(1, 2).contiguous()
                cls_l = torch.randint(0, 16, (B, 1), device=device)

                def _one():
                    net({"pos": pos, "x": x, "cls": cls_l})

                infer_total_s += time_forward_seconds(device, _one)
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)


def run_openpoints_sem(args):
    import torch
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    _prepare_openpoints_repo()
    from openpoints_mini.scripts.model_builder import build_s3dis_semseg_model
    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    net = build_s3dis_semseg_model(
        args.backbone, args.variant, args.in_channels, device
    ).eval()
    B, N = args.batch_size, args.num_points
    ic = args.in_channels

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                pts = torch.randn(B, N, 9, device=device, dtype=torch.float32)
                batch = _openpoints_sem_batch(pts, ic)

                def _one():
                    net(batch)

                infer_total_s += time_forward_seconds(device, _one)
                n_batches += 1
        return n_batches, infer_total_s

    return measure_cpu_energy_inference(do_inference)


def _resolve_num_points(args) -> int:
    if args.num_points is not None:
        return args.num_points
    if args.task.startswith("isda_"):
        from isda_models.benchmark import default_num_points

        return default_num_points(args.task[5:])
    if args.task in ("part_seg", "openpoints_part"):
        return 2048
    return 1024


def _resolve_num_classes_scan_and_isda(args):
    """scanobjectnn / isda 的 num_classes；其它任务忽略。"""
    if args.task == "scanobjectnn":
        return args.num_classes if args.num_classes is not None else 15
    if args.task.startswith("isda_"):
        from isda_models.benchmark import default_num_classes

        return (
            args.num_classes
            if args.num_classes is not None
            else default_num_classes(args.task[5:])
        )
    return args.num_classes


def main():
    ap = argparse.ArgumentParser(
        description="统一随机推理：PointKAN / ISEDA / openpoints_mini(PointNeXt|PointVector)"
    )
    ap.add_argument("--task", required=True, choices=TASK_CHOICES)
    ap.add_argument(
        "--backbone",
        default=None,
        choices=["pointnext", "pointvector"],
        help="openpoints_* 任务必选",
    )
    ap.add_argument(
        "--variant",
        default="l",
        choices=["l", "xl"],
        help="openpoints_sem + pointvector 时有效",
    )
    ap.add_argument(
        "--in-channels",
        type=int,
        default=6,
        dest="in_channels",
        help="openpoints_sem 每点特征维（≤9，与 S3DIS 块一致）",
    )
    ap.add_argument(
        "--part-in-channels",
        type=int,
        default=7,
        dest="part_in_channels",
        help="openpoints_part 点特征维",
    )
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    ap.add_argument("--max-batches", type=int, default=20, dest="max_batches")
    ap.add_argument("--num-points", type=int, default=None, dest="num_points")
    ap.add_argument(
        "--num-classes",
        type=int,
        default=None,
        dest="num_classes",
        help="scanobjectnn / isda_*；默认 15(scan) 或按 isda 模型",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task in OPENPOINTS_TASKS:
        if not args.backbone:
            ap.error("openpoints_* 任务必须指定 --backbone pointnext 或 pointvector")

    args.num_points = _resolve_num_points(args)
    if args.task in ("scanobjectnn",) or args.task.startswith("isda_"):
        args.num_classes = _resolve_num_classes_scan_and_isda(args)

    runners = {
        "modelnet40": run_modelnet40_random,
        "scanobjectnn": run_scanobjectnn_random,
        "part_seg": run_part_seg_random,
        "openpoints_cls": run_openpoints_cls,
        "openpoints_part": run_openpoints_part,
        "openpoints_sem": run_openpoints_sem,
    }
    for m in ISDA_MODEL_NAMES:
        runners[f"isda_{m}"] = run_isda

    inner, cpu_energy = runners[args.task](args)
    n, infer_only_s = inner
    _emit_result(args, n, infer_only_s, cpu_energy)


if __name__ == "__main__":
    main()
