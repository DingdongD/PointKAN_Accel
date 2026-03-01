"""
PointKAN 在 ModelNet40 上的推理 latency 与模块/算子级 latency breakdown。
参考 main.py 加载模型与数据，逐模块计时（torch.cuda.synchronize）并输出 JSON。
"""
import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

import models as models
from data import ModelNet40


def load_model_and_data(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = models.__dict__[args.model]()
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("net", ckpt)
        net.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint loaded, using random init.")
    net.eval()

    loader = DataLoader(
        ModelNet40(partition="test", num_points=args.num_points),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return net, loader, device


def forward_with_breakdown(net, x, device):
    """
    逐模块执行 forward 并在每步后 cuda.synchronize，返回各段耗时（ms）。
    net 可能是 DataParallel，内部是 pointkan.Model。
    """
    if device == "cpu":
        timings = {}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = net(x)
        timings["full_forward"] = (time.perf_counter() - t0) * 1000
        return timings, out

    model = net.module if hasattr(net, "module") else net
    timings = {}
    with torch.no_grad():
        x = x.to(device)
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        xyz = x.permute(0, 2, 1)
        x = model.embedding(x)
        torch.cuda.synchronize()
        timings["embedding"] = (time.perf_counter() - t_start) * 1000

        for i in range(model.stages):
            t0 = time.perf_counter()
            xyz, x_fea, center = model.GAM_list[i](xyz, x.permute(0, 2, 1))
            torch.cuda.synchronize()
            timings[f"stage{i}_GAM"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            x = model.LFP_blocks_list[i](x_fea, center)
            torch.cuda.synchronize()
            timings[f"stage{i}_LFP"] = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            x = model.GFP_blocks_list[i](x)
            torch.cuda.synchronize()
            timings[f"stage{i}_GFP"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        x = torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        torch.cuda.synchronize()
        timings["pool"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        x = model.classifier(x)
        torch.cuda.synchronize()
        timings["classifier"] = (time.perf_counter() - t0) * 1000

    full = sum(v for k, v in timings.items() if k != "full_forward")
    timings["full_forward"] = full
    out = x
    return timings, out


def run_benchmark(net, loader, device, warmup=20, repeat=100):
    it = iter(loader)
    sample = next(it)
    x = sample[0].permute(0, 2, 1)  # [B, 3, N] as in main
    x = x.to(device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = net(x)
        if device == "cuda":
            torch.cuda.synchronize()

    # Full forward latency (no breakdown)
    latencies = []
    for _ in range(repeat):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = net(x)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
    full_mean = float(np.mean(latencies))
    full_std = float(np.std(latencies))

    # One run with breakdown
    timings, _ = forward_with_breakdown(net, x, device)

    return {
        "full_forward_ms_mean": full_mean,
        "full_forward_ms_std": full_std,
        "full_forward_ms_list": [round(x, 4) for x in latencies],
        "breakdown_ms": {k: round(v, 4) for k, v in timings.items()},
        "warmup": warmup,
        "repeat": repeat,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pointKAN")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str, default=None, help=".pth 或 last_checkpoint.pth 路径")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--out_json", type=str, default=None, help="结果保存为 JSON")
    args = parser.parse_args()

    net, loader, device = load_model_and_data(args)
    print(f"Running latency benchmark (device={device}, warmup={args.warmup}, repeat={args.repeat})...")
    result = run_benchmark(net, loader, device, warmup=args.warmup, repeat=args.repeat)

    print("\n=== Full forward (ms) ===")
    print(f"  mean: {result['full_forward_ms_mean']:.4f} ± {result['full_forward_ms_std']:.4f}")
    print("\n=== Breakdown (ms) ===")
    for k, v in result["breakdown_ms"].items():
        print(f"  {k}: {v:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        out = {
            "config": {
                "model": args.model,
                "num_points": args.num_points,
                "checkpoint": args.checkpoint,
                "warmup": args.warmup,
                "repeat": args.repeat,
            },
            "full_forward_ms_mean": result["full_forward_ms_mean"],
            "full_forward_ms_std": result["full_forward_ms_std"],
            "breakdown_ms": result["breakdown_ms"],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nWritten to {args.out_json}")


if __name__ == "__main__":
    main()
