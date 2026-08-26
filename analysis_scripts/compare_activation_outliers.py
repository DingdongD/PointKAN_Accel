#!/usr/bin/env python3
from __future__ import absolute_import

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTKAN_ELITE_ROOT = os.path.join(REPO_ROOT, "PointKAN-elite")
for p in (REPO_ROOT, POINTKAN_ELITE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from quantization import QuantConfig, quantize_model  # noqa: E402
from quantization.fake_quant import QuantizeBase  # noqa: E402
from quantization import evaluate_quant_pointkan as eq  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Compare activation outlier statistics across PointKAN tasks.")
    p.add_argument("--tasks", nargs="+", default=["modelnet40", "scan", "partseg"], choices=["modelnet40", "scan", "partseg"])
    p.add_argument("--modelnet_checkpoint", default="PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42/best_checkpoint.pth")
    p.add_argument("--scan_checkpoint", default="PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42/best_checkpoint.pth")
    p.add_argument("--partseg_checkpoint", default="PointKAN-elite/part_segmentation/checkpoints/pointKANElite_part_exp/best_insiou_model.pth")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--partseg_num_points", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--calibration_size", type=int, default=512)
    p.add_argument("--max_values_per_layer", type=int, default=200000)
    p.add_argument("--sample_per_batch_per_layer", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="my_progressive_out/activation_outlier_compare")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def purge_task_modules():
    for key in list(sys.modules.keys()):
        if key == "models" or key.startswith("models.") or key == "model" or key.startswith("model."):
            del sys.modules[key]


def _sample_flat_abs(x, max_items):
    flat = x.detach().float().abs().reshape(-1)
    if flat.numel() > max_items:
        idx = torch.randperm(flat.numel(), device=flat.device)[:max_items]
        flat = flat[idx]
    return flat.cpu()


class LayerCollector:
    def __init__(self, max_values_per_layer, sample_per_batch_per_layer):
        self.max_values = int(max_values_per_layer)
        self.sample_per_batch = int(sample_per_batch_per_layer)
        self.samples = defaultdict(list)
        self.full_count = defaultdict(int)
        self.batch_absmax = defaultdict(list)

    def hook(self, name):
        def _hook(_module, inputs, _output):
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            self.full_count[name] += int(x.numel())
            self.batch_absmax[name].append(float(x.detach().float().abs().max().cpu().item()))
            current = sum(t.numel() for t in self.samples[name])
            if current >= self.max_values:
                return
            take = min(self.sample_per_batch, self.max_values - current)
            self.samples[name].append(_sample_flat_abs(x, take))
        return _hook

    def summarize(self, task):
        rows = []
        for name in sorted(self.samples):
            vals = torch.cat(self.samples[name]).double()
            if vals.numel() == 0:
                continue
            p50 = float(torch.quantile(vals, 0.50).item())
            p90 = float(torch.quantile(vals, 0.90).item())
            p99 = float(torch.quantile(vals, 0.99).item())
            p999 = float(torch.quantile(vals, 0.999).item())
            p9999 = float(torch.quantile(vals, 0.9999).item()) if vals.numel() >= 10000 else float("nan")
            sample_max = float(vals.max().item())
            batch_max = max(self.batch_absmax[name]) if self.batch_absmax[name] else sample_max
            mean = float(vals.mean().item())
            std = float(vals.std(unbiased=False).item())
            centered = vals - vals.mean()
            kurt = float((centered.pow(4).mean() / torch.clamp(vals.var(unbiased=False).pow(2), min=1e-24)).item())
            energy = vals.pow(2).sum()
            tail99_energy = float(vals[vals > p99].pow(2).sum().div(torch.clamp(energy, min=1e-24)).item())
            tail999_energy = float(vals[vals > p999].pow(2).sum().div(torch.clamp(energy, min=1e-24)).item())
            rows.append({
                "task": task,
                "layer": name,
                "sample_count": int(vals.numel()),
                "full_value_count": int(self.full_count[name]),
                "mean_abs": mean,
                "std_abs": std,
                "p50_abs": p50,
                "p90_abs": p90,
                "p99_abs": p99,
                "p999_abs": p999,
                "p9999_abs": p9999,
                "sample_absmax": sample_max,
                "batch_absmax": batch_max,
                "absmax_over_p99": batch_max / max(p99, 1e-12),
                "absmax_over_p999": batch_max / max(p999, 1e-12),
                "used_int8_levels_p99": 127.0 * p99 / max(batch_max, 1e-12),
                "used_int8_levels_p999": 127.0 * p999 / max(batch_max, 1e-12),
                "tail99_energy_frac": tail99_energy,
                "tail999_energy_frac": tail999_energy,
                "kurtosis_abs": kurt,
            })
        return rows


def register_activation_hooks(model, collector):
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and "act_fake_quant" in name:
            handles.append(module.register_forward_hook(collector.hook(name)))
    return handles


def build_task(task, args, device):
    purge_task_modules()
    if task == "modelnet40":
        model = eq.build_model_modelnet40(checkpoint_path=args.modelnet_checkpoint, device=device)
        loader, _ = eq.build_dataloaders_modelnet40(args.num_points, args.batch_size, args.workers, calibration_size=args.calibration_size)

        def forward(batch):
            x = batch[0].to(device).permute(0, 2, 1)
            return model(x)

        return model, loader, forward
    if task == "scan":
        model = eq.build_model_scan(checkpoint_path=args.scan_checkpoint, device=device)
        loader, _ = eq.build_dataloaders_scan(args.num_points, args.batch_size, args.workers, calibration_size=args.calibration_size, seed=args.seed)

        def forward(batch):
            x = batch[0].to(device)
            if x.dim() == 3 and x.shape[-1] == 3:
                x = x.permute(0, 2, 1)
            return model(x)

        return model, loader, forward
    if task == "partseg":
        model = eq.build_model_partseg(checkpoint_path=args.partseg_checkpoint, device=device)
        loader, _, to_categorical, _ = eq.build_dataloaders_partseg(args.batch_size, args.workers, calibration_size=args.calibration_size)

        def forward(batch):
            data, label = batch[0], batch[1]
            data = data.to(device).permute(0, 2, 1)
            norm = batch[3].to(device).permute(0, 2, 1) if len(batch) >= 4 else data
            label_one_hot = to_categorical(label.squeeze(-1), 16).to(device)
            return model(data, norm, label_one_hot)

        return model, loader, forward
    raise ValueError(task)


def collect_task(task, args, device):
    model, loader, forward = build_task(task, args, device)
    quant_config = QuantConfig(model_name=f"{task}_activation_probe")
    quantize_model(model, quant_config, use_input_quant=True, verbose=False)
    collector = LayerCollector(args.max_values_per_layer, args.sample_per_batch_per_layer)
    handles = register_activation_hooks(model, collector)
    model.eval()
    seen = 0
    with torch.no_grad():
        for batch in loader:
            forward(batch)
            seen += int(batch[0].size(0))
            if seen >= args.calibration_size:
                break
    for h in handles:
        h.remove()
    rows = collector.summarize(task)
    for r in rows:
        r["calibration_samples"] = seen
    return rows


def write_outputs(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "activation_outlier_per_layer.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    summary = []
    for task, task_rows in sorted(by_task.items()):
        for metric in ("absmax_over_p99", "absmax_over_p999", "used_int8_levels_p99", "tail99_energy_frac", "kurtosis_abs"):
            vals = np.asarray([float(r[metric]) for r in task_rows if np.isfinite(float(r[metric]))], dtype=np.float64)
            summary.append({
                "task": task,
                "metric": metric,
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "max": float(vals.max()),
            })
    summary_path = os.path.join(out_dir, "activation_outlier_task_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "metric", "mean", "median", "max"])
        writer.writeheader()
        writer.writerows(summary)
    json_path = os.path.join(out_dir, "activation_outlier_compare.json")
    with open(json_path, "w") as f:
        json.dump({"per_layer": rows, "summary": summary}, f, indent=2)
    return csv_path, summary_path, json_path


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for task in args.tasks:
        print(f"Collecting {task} activation outlier stats...")
        rows.extend(collect_task(task, args, device))
    paths = write_outputs(rows, args.out_dir)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
