#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
适配自定义 PyTorch network + dataset 的推理分析入口
"""

import argparse
import importlib
import json
import time
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

from profiler import (
    GPUSampler,
    CPUSampler,
    trapezoid_energy_j,
    write_csv,
    parse_perf_stat_csv,
    command_exists,
    discover_rapl_energy_nodes,
    read_rapl_energy_uj,
    diff_rapl_j,
)


def import_from_string(module_name: str, obj_name: str):
    """动态导入模块中的类/函数"""
    mod = importlib.import_module(module_name)
    return getattr(mod, obj_name)


def move_to_device(batch, device):
    """递归将 batch 移动到 device"""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


@torch.no_grad()
def run_infer_loop(model, loader, device, max_batches: int):
    """
    执行若干 batch 推理，返回每个 batch latency
    """
    model.eval()
    latencies = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        batch = move_to_device(batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 这里假设 model(batch) 可直接调用
        # 如果你的接口是 model(xyz, feat) 这种，可以改这里
        _ = model(batch)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append(t1 - t0)

    return latencies


def main():
    ap = argparse.ArgumentParser()

    # 模型导入参数
    ap.add_argument("--model-module", required=True)
    ap.add_argument("--model-class", required=True)

    # 数据集导入参数
    ap.add_argument("--dataset-module", required=True)
    ap.add_argument("--dataset-class", required=True)

    # checkpoint
    ap.add_argument("--checkpoint", default="")

    # device
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--gpu-index", type=int, default=0)

    # dataloader
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)

    # profiling
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--gpu-sample-ms", type=int, default=50)
    ap.add_argument("--cpu-sample-ms", type=int, default=50)

    # 输出
    ap.add_argument("--out-dir", default="outputs_torch")

    args = ap.parse_args()

    out_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 导入模型和数据集类
    ModelClass = import_from_string(args.model_module, args.model_class)
    DatasetClass = import_from_string(args.dataset_module, args.dataset_class)

    # 构建模型
    model = ModelClass()

    # 加载权重
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(args.gpu_index)

    model.to(device)

    # 构建数据集
    dataset = DatasetClass()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # warmup
    for _ in range(args.warmup):
        _ = run_infer_loop(model, loader, device, max_batches=min(5, args.max_batches))

    # 系统采样器
    gpu_sampler = None
    cpu_sampler = None
    gpu_backend = None

    if device.type == "cuda":
        gpu_sampler = GPUSampler(args.gpu_sample_ms / 1000.0, args.gpu_index)
        if gpu_sampler.setup():
            gpu_sampler.start()
            gpu_backend = "pynvml"

    cpu_sampler = CPUSampler(args.cpu_sample_ms / 1000.0, os.getpid())
    cpu_sampler.start()

    # RAPL
    rapl_nodes = discover_rapl_energy_nodes()
    rapl_before = read_rapl_energy_uj(rapl_nodes) if rapl_nodes else {}

    # repeat
    run_results = []
    for r in range(args.repeat):
        latencies = run_infer_loop(model, loader, device, max_batches=args.max_batches)
        run_results.append({
            "run_idx": r,
            "batch_latency_s_mean": mean(latencies) if latencies else None,
            "batch_latency_s_min": min(latencies) if latencies else None,
            "batch_latency_s_max": max(latencies) if latencies else None,
            "num_batches": len(latencies),
        })

    rapl_after = read_rapl_energy_uj(rapl_nodes) if rapl_nodes else {}
    cpu_energy_rapl = diff_rapl_j(rapl_before, rapl_after) if rapl_before and rapl_after else {}

    if gpu_sampler:
        gpu_sampler.stop()
        write_csv(out_dir / "gpu_samples.csv", gpu_sampler.samples)

    if cpu_sampler:
        cpu_sampler.stop()
        write_csv(out_dir / "cpu_samples.csv", cpu_sampler.samples)

    gpu_energy_j = trapezoid_energy_j(gpu_sampler.samples) if gpu_sampler and gpu_sampler.samples else None

    summary = {
        "gpu_backend": gpu_backend,
        "gpu_energy_j": gpu_energy_j,
        "cpu_energy_rapl_j": cpu_energy_rapl,
        "runs": run_results,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import os
    main()