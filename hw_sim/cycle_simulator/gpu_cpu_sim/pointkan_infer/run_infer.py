#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointKAN 统一推理入口：供 profiler 通过 CMD 调用。

支持三个任务：modelnet40 / scanobjectnn / part_seg。
CPU 能耗采用 pyRAPL：仅包住推理循环（Measurement.begin/end），得到「仅模型推理」的 CPU 能耗。
用法示例：
  python run_infer.py --task modelnet40 --checkpoint /path/to/checkpoints/pointKANElite-xxx --max_batches 20
  python run_infer.py --task scanobjectnn --checkpoint /path/to/checkpoints/pointKANElite-ScanObjectNN --batch_size 4
  python run_infer.py --task part_seg --checkpoint /path/to/checkpoints/pointKANElite_part_exp --max_batches 10
"""

import argparse
import json
import os
import sys
import time

# 工程根：PointKAN-elite 所在目录（pointkan_infer -> gpu_cpu_sim -> cycle_simulator -> hw_sim -> PointKAN-pytorch）
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


def _resolve_checkpoint(path: str, task: str) -> str:
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
    if os.path.isdir(path):
        if task == "part_seg":
            for name in ("best_insiou_model.pth", "best_checkpoint.pth"):
                c = os.path.join(path, name)
                if os.path.isfile(c):
                    return c
        else:
            c = os.path.join(path, "best_checkpoint.pth")
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(f"目录下未找到 checkpoint 文件: {path}")
    raise FileNotFoundError(f"检查点路径不存在: {path}")


def run_modelnet40(args):
    task_dir = TASK_DIRS["modelnet40"]
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    from data import ModelNet40
    import models as models
    from utils import cal_loss
    import torch
    from torch.utils.data import DataLoader
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    device = torch.device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint, "modelnet40")

    def load_state(path, dev):
        ckpt = torch.load(path, map_location=dev)
        state = ckpt.get("net", ckpt)
        if next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    net = models.__dict__["pointKANElite"]()
    net.load_state_dict(load_state(ckpt_path, device), strict=True)
    net = net.to(device)
    net.eval()

    test_set = ModelNet40(num_points=args.num_points, partition="test")
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    criterion = cal_loss

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                data = data.to(device).permute(0, 2, 1)
                label = label.to(device).squeeze()
                infer_total_s += time_forward_seconds(device, lambda: net(data))
                n_batches += 1
        return n_batches, infer_total_s

    (n_batches, infer_total_s), cpu_energy = measure_cpu_energy_inference(do_inference)
    return n_batches, infer_total_s, cpu_energy


def run_scanobjectnn(args):
    task_dir = TASK_DIRS["scanobjectnn"]
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    from ScanObjectNN import ScanObjectNN
    import models as models
    from utils import cal_loss
    import torch
    from torch.utils.data import DataLoader
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    device = torch.device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint, "scanobjectnn")

    def load_state(path, dev):
        ckpt = torch.load(path, map_location=dev)
        state = ckpt.get("net", ckpt)
        if next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    num_classes = getattr(args, "num_classes", 15)
    net = models.__dict__["pointKANElite"](num_classes=num_classes)
    net.load_state_dict(load_state(ckpt_path, device), strict=True)
    net = net.to(device)
    net.eval()

    test_set = ScanObjectNN(num_points=args.num_points, partition="test")
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    criterion = cal_loss

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                data = data.to(device).permute(0, 2, 1)
                label = label.to(device).squeeze()
                infer_total_s += time_forward_seconds(device, lambda: net(data))
                n_batches += 1
        return n_batches, infer_total_s

    (n_batches, infer_total_s), cpu_energy = measure_cpu_energy_inference(do_inference)
    return n_batches, infer_total_s, cpu_energy


def run_part_seg(args):
    task_dir = TASK_DIRS["part_seg"]
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    from util.data_util import PartNormalDataset
    from util.util import to_categorical
    import model as models
    import torch
    from torch.utils.data import DataLoader
    from rapl_inference import measure_cpu_energy_inference, time_forward_seconds

    NUM_PART = 50
    NUM_CLASSES = 16
    device = torch.device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint, "part_seg")

    def load_state(path, dev):
        ckpt = torch.load(path, map_location=dev)
        state = ckpt.get("model", ckpt)
        if next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    net = models.__dict__["pointKANElite"](num_classes=NUM_PART)
    net.load_state_dict(load_state(ckpt_path, device), strict=True)
    net = net.to(device)
    net.eval()

    test_set = PartNormalDataset(npoints=args.num_points, split="test", normalize=False)
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    def do_inference():
        n_batches = 0
        infer_total_s = 0.0
        with torch.no_grad():
            for batch_idx, (points, label, target, norm_plt) in enumerate(loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                points = points.float().to(device).transpose(2, 1)
                norm_plt = norm_plt.float().to(device).transpose(2, 1)
                label = label.squeeze(1).long().to(device)
                onehot = to_categorical(label, NUM_CLASSES)
                infer_total_s += time_forward_seconds(device, lambda: net(points, norm_plt, onehot))
                n_batches += 1
        return n_batches, infer_total_s

    (n_batches, infer_total_s), cpu_energy = measure_cpu_energy_inference(do_inference)
    return n_batches, infer_total_s, cpu_energy


def main():
    ap = argparse.ArgumentParser(description="PointKAN 统一推理入口（供 profiler 调用）")
    ap.add_argument("--task", required=True, choices=["modelnet40", "scanobjectnn", "part_seg"])
    ap.add_argument("--checkpoint", required=True, help="检查点目录或 .pth 路径")
    ap.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    ap.add_argument("--max-batches", type=int, default=20, dest="max_batches", help="0 表示跑完整个测试集")
    ap.add_argument("--num-points", type=int, default=None, dest="num_points")
    ap.add_argument("--num-classes", type=int, default=15, dest="num_classes", help="仅 ScanObjectNN")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.num_points is None:
        args.num_points = 1024 if args.task != "part_seg" else 2048
    if args.max_batches == 0:
        args.max_batches = None

    runners = {
        "modelnet40": run_modelnet40,
        "scanobjectnn": run_scanobjectnn,
        "part_seg": run_part_seg,
    }
    ret = runners[args.task](args)
    if len(ret) == 3:
        n, infer_only_s, cpu_energy = ret
    else:
        n, infer_only_s = ret
        cpu_energy = None
    print(f"run_infer: task={args.task} batches={n} inference_only_s={infer_only_s:.6f}", file=sys.stderr)
    if cpu_energy:
        pkg = cpu_energy.get("cpu_energy_pkg_j") or 0
        ram = cpu_energy.get("cpu_energy_ram_j") or 0
        total = cpu_energy.get("cpu_energy_total_j") or 0
        print(
            f"run_infer: cpu_energy_inference pkg={pkg:.4f}J ram={ram:.4f}J total={total:.4f}J (pyRAPL, inference only)",
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
