#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointKAN 随机数据 + 随机权重推理：不依赖 checkpoint 与真实数据集，仍调用三种模型结构，
供 profiler 在 CPU/GPU 上采集 latency 与 energy。

计时与 CPU 能耗说明：
- 本脚本测量 model(…) 前向时间：GPU 用 torch.cuda.Event，CPU 用 time.perf_counter，只统计纯前向时间之和。
- CPU 能耗采用 pyRAPL：在进程内用 Measurement.begin/end 只包住推理循环（不测整进程），
  得到「仅模型推理」的 CPU 能耗，写入 inference_only.json 供 profiler/绘图使用。
用法示例：
  python run_infer_random.py --task modelnet40 --device cuda --max-batches 20
  python run_infer_random.py --task part_seg --device cpu --batch-size 2 --seed 42
  # 配合 profiler（与 run_infer.py 相同方式）：
  CMD='python run_infer_random.py --task modelnet40 --device cuda --max-batches 20' bash scripts/profile_gpu.sh
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
ELITE_ROOT = os.environ.get(
    "POINTKAN_ELITE_ROOT",
    os.path.join(WORKSPACE, "PointKAN-elite"),
)

TASK_DIRS = {
    "modelnet40": os.path.join(ELITE_ROOT, "classification_ModelNet40"),
    "scanobjectnn": os.path.join(ELITE_ROOT, "classification_ScanObjectNN"),
    "part_seg": os.path.join(ELITE_ROOT, "part_segmentation"),
}


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

    net = models.__dict__["pointKANElite"]()
    net = net.to(device)
    net.eval()

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                x = torch.randn(args.batch_size, 3, args.num_points, device=device, dtype=torch.float32)
                infer_total_s += time_forward_seconds(device, lambda: net(x))
                n_batches += 1
        return n_batches, infer_total_s

    (n_batches, infer_total_s), cpu_energy = measure_cpu_energy_inference(do_inference)
    return n_batches, infer_total_s, cpu_energy


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
    num_classes = getattr(args, "num_classes", 15)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    net = models.__dict__["pointKANElite"](num_classes=num_classes)
    net = net.to(device)
    net.eval()

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for _ in range(args.max_batches):
                x = torch.randn(args.batch_size, 3, args.num_points, device=device, dtype=torch.float32)
                infer_total_s += time_forward_seconds(device, lambda: net(x))
                n_batches += 1
        return n_batches, infer_total_s

    (n_batches, infer_total_s), cpu_energy = measure_cpu_energy_inference(do_inference)
    return n_batches, infer_total_s, cpu_energy


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

    net = models.__dict__["pointKANElite"](num_classes=50)
    net = net.to(device)
    net.eval()

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

    (n_batches, infer_total_s), cpu_energy = measure_cpu_energy_inference(do_inference)
    return n_batches, infer_total_s, cpu_energy


def main():
    ap = argparse.ArgumentParser(
        description="PointKAN 随机数据+随机权重推理，供 profiler 采 latency/energy"
    )
    ap.add_argument("--task", required=True, choices=["modelnet40", "scanobjectnn", "part_seg"])
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    ap.add_argument("--max-batches", type=int, default=20, dest="max_batches")
    ap.add_argument("--num-points", type=int, default=None, dest="num_points")
    ap.add_argument("--num-classes", type=int, default=15, dest="num_classes", help="仅 ScanObjectNN")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，可复现")
    args = ap.parse_args()

    if args.num_points is None:
        args.num_points = 1024 if args.task != "part_seg" else 2048

    runners = {
        "modelnet40": run_modelnet40_random,
        "scanobjectnn": run_scanobjectnn_random,
        "part_seg": run_part_seg_random,
    }
    ret = runners[args.task](args)
    if len(ret) == 3:
        n, infer_only_s, cpu_energy = ret
    else:
        n, infer_only_s = ret
        cpu_energy = None
    print(
        f"run_infer_random: task={args.task} device={args.device} batches={n} "
        f"inference_only_s={infer_only_s:.6f} (model forward only, sum of {n} batches)",
        file=sys.stderr,
    )
    if cpu_energy:
        pkg = cpu_energy.get("cpu_energy_pkg_j") or 0
        ram = cpu_energy.get("cpu_energy_ram_j") or 0
        total = cpu_energy.get("cpu_energy_total_j") or 0
        print(
            f"run_infer_random: cpu_energy_inference pkg={pkg:.4f}J ram={ram:.4f}J total={total:.4f}J (pyRAPL, inference only)",
            file=sys.stderr,
        )
    run_dir = os.environ.get("PROFILE_RUN_DIR")
    run_tag = os.environ.get("PROFILE_RUN_TAG")
    if run_dir and run_tag:
        path = os.path.join(run_dir, f"{run_tag}_inference_only.json")
        try:
            out = {"inference_only_s": infer_only_s, "num_batches": n}
            if cpu_energy:
                out["cpu_energy_inference_pkg_j"] = cpu_energy.get("cpu_energy_pkg_j")
                out["cpu_energy_inference_ram_j"] = cpu_energy.get("cpu_energy_ram_j")
                out["cpu_energy_inference_total_j"] = cpu_energy.get("cpu_energy_total_j")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out, f)
        except Exception:
            pass


if __name__ == "__main__":
    main()
