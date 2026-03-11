#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointKAN 推理开销 Breakdown 分析：按阶段/模块统计前向时间。

输出：
- 各阶段耗时（embedding / GAM / LFP / GFP / classifier）
- 单 batch 与多 batch 平均
- 可选：PyTorch profiler 的 trace 文件，便于在 Chrome tracing 中查看算子级 breakdown

用法：
  python breakdown_profiler.py --task modelnet40 --checkpoint /path/to/checkpoints/... --max_batches 5 --out breakdown.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

# 复用 run_infer 的路径与 checkpoint 解析
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
ELITE_ROOT = os.environ.get("POINTKAN_ELITE_ROOT", os.path.join(WORKSPACE, "PointKAN-elite"))
TASK_DIRS = {
    "modelnet40": os.path.join(ELITE_ROOT, "classification_ModelNet40"),
    "scanobjectnn": os.path.join(ELITE_ROOT, "classification_ScanObjectNN"),
    "part_seg": os.path.join(ELITE_ROOT, "part_segmentation"),
}
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from rapl_inference import time_forward_seconds


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


def _get_stage_modules(net):
    """返回 (name, module) 列表，用于挂 hook。"""
    stages = []
    stages.append(("embedding", net.embedding))
    n_stages = getattr(net, "stages", len(net.GAM_list))
    for i in range(n_stages):
        stages.append((f"GAM_{i}", net.GAM_list[i]))
        stages.append((f"LFP_{i}", net.LFP_blocks_list[i]))
        stages.append((f"GFP_{i}", net.GFP_blocks_list[i]))
    stages.append(("classifier", net.classifier))
    return stages


def _run_breakdown_modelnet40(args):
    task_dir = TASK_DIRS["modelnet40"]
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    from data import ModelNet40
    import models as models
    from torch.utils.data import DataLoader

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
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    return net, loader, device, None


def _run_breakdown_scanobjectnn(args):
    task_dir = TASK_DIRS["scanobjectnn"]
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    from ScanObjectNN import ScanObjectNN
    import models as models
    from torch.utils.data import DataLoader

    device = torch.device(args.device)
    ckpt_path = _resolve_checkpoint(args.checkpoint, "scanobjectnn")
    num_classes = getattr(args, "num_classes", 15)

    def load_state(path, dev):
        ckpt = torch.load(path, map_location=dev)
        state = ckpt.get("net", ckpt)
        if next(iter(state.keys())).startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        return state

    net = models.__dict__["pointKANElite"](num_classes=num_classes)
    net.load_state_dict(load_state(ckpt_path, device), strict=True)
    net = net.to(device)
    net.eval()

    test_set = ScanObjectNN(num_points=args.num_points, partition="test")
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    return net, loader, device, None


def _run_breakdown_part_seg(args):
    task_dir = TASK_DIRS["part_seg"]
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
    os.chdir(task_dir)

    from util.data_util import PartNormalDataset
    from util.util import to_categorical
    import model as models
    from torch.utils.data import DataLoader

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
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    return net, loader, device, {"num_classes": NUM_CLASSES, "to_categorical": to_categorical}


def run_breakdown(args):
    loaders = {
        "modelnet40": _run_breakdown_modelnet40,
        "scanobjectnn": _run_breakdown_scanobjectnn,
        "part_seg": _run_breakdown_part_seg,
    }
    net, loader, device, extra = loaders[args.task](args)
    stage_modules = _get_stage_modules(net)

    # 每阶段累计时间 (ms)
    stage_times_ms = defaultdict(float)
    data_load_times_ms = []
    forward_times_ms = []
    n_runs = 0

    def make_hooks():
        current = {}
        start_t = {}

        def pre_hook(name):
            def hook(module, inp):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_t[name] = time.perf_counter()
            return hook

        def post_hook(name):
            def hook(module, inp, out):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start_t[name]) * 1000.0
                current[name] = elapsed
            return hook

        return start_t, current, pre_hook, post_hook

    hooks_handles = []
    start_t, current, pre_hook, post_hook = make_hooks()
    for name, mod in stage_modules:
        hooks_handles.append(mod.register_forward_pre_hook(pre_hook(name)))
        hooks_handles.append(mod.register_forward_hook(post_hook(name)))

    max_batches = args.max_batches or len(loader)
    if args.task == "part_seg":
        to_categorical = extra["to_categorical"]
        num_classes = extra["num_classes"]

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if args.task == "part_seg":
            points, label, target, norm_plt = batch
            t0 = time.perf_counter()
            points = points.float().to(device).transpose(2, 1)
            norm_plt = norm_plt.float().to(device).transpose(2, 1)
            label = label.squeeze(1).long().to(device)
            onehot = to_categorical(label, num_classes)
            if device.type == "cuda":
                torch.cuda.synchronize()
            data_load_times_ms.append((time.perf_counter() - t0) * 1000.0)
            with torch.no_grad():
                forward_times_ms.append(time_forward_seconds(device, lambda: net(points, norm_plt, onehot)) * 1000.0)
        else:
            data, label = batch
            t0 = time.perf_counter()
            data = data.to(device).permute(0, 2, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            data_load_times_ms.append((time.perf_counter() - t0) * 1000.0)
            with torch.no_grad():
                forward_times_ms.append(time_forward_seconds(device, lambda: net(data)) * 1000.0)  # GPU: cuda.Event

        for name in current:
            stage_times_ms[name] += current[name]
        n_runs += 1

    for h in hooks_handles:
        h.remove()

    # 汇总
    n = max(n_runs, 1)
    result = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "num_batches": n_runs,
        "num_points": args.num_points,
        "device": args.device,
        "stages_ms": {k: round(v, 4) for k, v in stage_times_ms.items()},
        "stages_ms_mean": {k: round(v / n, 4) for k, v in stage_times_ms.items()},
        "stage_pct": {},
        "data_load_ms_total": round(sum(data_load_times_ms), 4),
        "data_load_ms_mean": round(sum(data_load_times_ms) / n, 4) if data_load_times_ms else 0,
        "forward_ms_total": round(sum(forward_times_ms), 4),
        "forward_ms_mean": round(sum(forward_times_ms) / n, 4) if forward_times_ms else 0,
    }
    total_stage = sum(stage_times_ms.values())
    if total_stage > 0:
        result["stage_pct"] = {k: round(100.0 * v / total_stage, 2) for k, v in stage_times_ms.items()}
    return result


def main():
    ap = argparse.ArgumentParser(description="PointKAN 推理 Breakdown 分析")
    ap.add_argument("--task", required=True, choices=["modelnet40", "scanobjectnn", "part_seg"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=1, dest="batch_size")
    ap.add_argument("--max-batches", type=int, default=5, dest="max_batches", help="0=全部")
    ap.add_argument("--num-points", type=int, default=None, dest="num_points")
    ap.add_argument("--num-classes", type=int, default=15, dest="num_classes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="breakdown.json", help="输出 JSON 路径")
    ap.add_argument("--trace", default=None, help="可选：PyTorch profiler trace 输出路径 (.json)")
    args = ap.parse_args()

    if args.num_points is None:
        args.num_points = 1024 if args.task != "part_seg" else 2048

    result = run_breakdown(args)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Breakdown 已写入: {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
