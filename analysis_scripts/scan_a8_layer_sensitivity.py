#!/usr/bin/env python3
from __future__ import absolute_import

import argparse
import csv
import json
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTKAN_ELITE_ROOT = os.path.join(REPO_ROOT, "PointKAN-elite")
for p in (REPO_ROOT, POINTKAN_ELITE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from quantization import QuantConfig, QConfig, quantize_model, calibrate, enable_quantization  # noqa: E402
from quantization.fake_quant import QuantizeBase  # noqa: E402
from quantization import evaluate_quant_pointkan as eq  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="ScanObjectNN A8 layer sensitivity by leave-one-activation-FP evaluation.")
    p.add_argument("--checkpoint", default="PointKAN-elite/classification_ScanObjectNN/checkpoints/pointKANElite-scan_exp-42/best_checkpoint.pth")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--calibration_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--a_percentile_low", type=float, default=0.01)
    p.add_argument("--a_percentile_high", type=float, default=0.99)
    p.add_argument("--weight_ch_axis", type=int, default=-1, help="-1: per-tensor, 0: per-output-channel")
    p.add_argument("--max_eval_batches", type=int, default=30, help="Coarse ranking batches; <=0 means full test")
    p.add_argument("--full_top_k", type=int, default=8, help="Run full-test confirmation for top-K coarse layers")
    p.add_argument("--out_dir", default="my_progressive_out/scan_a8_layer_sensitivity")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_scan_limited(model, test_loader, device, max_batches=None):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for bi, (data, label) in enumerate(test_loader):
            if max_batches is not None and max_batches > 0 and bi >= max_batches:
                break
            data = data.to(device)
            if data.dim() == 3 and data.shape[-1] == 3:
                data = data.permute(0, 2, 1)
            out = model(data)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            pred = logits.max(dim=1)[1]
            all_pred.append(pred.cpu().numpy())
            all_true.append(label.view(-1).cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    acc = 100.0 * np.mean(all_pred == all_true)
    try:
        from sklearn.metrics import balanced_accuracy_score
        bal = 100.0 * balanced_accuracy_score(all_true, all_pred)
    except Exception:
        bal = acc
    return {"accuracy": float(acc), "balanced_accuracy": float(bal), "num_samples": int(all_true.shape[0])}


def build_quantized_scan(args, device):
    model = eq.build_model_scan(num_classes=15, checkpoint_path=args.checkpoint, device=device)
    calib_loader, test_loader = eq.build_dataloaders_scan(
        args.num_points,
        args.batch_size,
        args.workers,
        calibration_size=args.calibration_size,
        calib_sampling="random",
        seed=args.seed,
    )
    wq = QConfig(bit=8, symmetric=True, ch_axis=args.weight_ch_axis, quantizer="FixedFakeQuantize", observer="MinMaxObserver")
    aq = QConfig(
        bit=8,
        symmetric=True,
        ch_axis=-1,
        quantizer="FixedFakeQuantize",
        observer="PercentileObserver",
        observer_kwargs={"percentile_low": args.a_percentile_low, "percentile_high": args.a_percentile_high},
    )
    quant_config = QuantConfig(model_name="pointKANElite_scan_a8_sensitivity", w_qconfig=wq, a_qconfig=aq)
    quantize_model(model, quant_config, use_input_quant=True, verbose=True)
    calibrate(model, calib_loader, args.calibration_size, device, verbose=True)
    enable_quantization(model, quantizer_type="fake_quant")
    return model, test_loader


def collect_act_quantizers(model):
    out = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase) and "act_fake_quant" in name:
            out.append((name, module))
    return out


def qparams(module):
    scale = getattr(module, "scale", None)
    zp = getattr(module, "zero_point", None)
    scale_val = float(scale.detach().float().mean().cpu().item()) if scale is not None else None
    zp_val = float(zp.detach().float().mean().cpu().item()) if zp is not None else None
    return scale_val, zp_val


def run_leave_one(model, test_loader, device, act_modules, baseline, max_batches):
    rows = []
    for idx, (name, module) in enumerate(act_modules):
        module.disable_fake_quant()
        metrics = evaluate_scan_limited(model, test_loader, device, max_batches=max_batches)
        module.enable_fake_quant()
        scale, zp = qparams(module)
        rows.append({
            "rank_input_order": idx,
            "layer": name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "num_samples": metrics["num_samples"],
            "delta_accuracy_vs_all_a8": metrics["accuracy"] - baseline["accuracy"],
            "delta_balanced_vs_all_a8": metrics["balanced_accuracy"] - baseline["balanced_accuracy"],
            "act_scale_mean": scale,
            "act_zero_point_mean": zp,
        })
        print(
            f"[{idx+1:02d}/{len(act_modules)}] {name}: "
            f"acc {metrics['accuracy']:.3f} ({metrics['accuracy'] - baseline['accuracy']:+.3f})"
        )
    rows.sort(key=lambda r: (r["delta_accuracy_vs_all_a8"], r["delta_balanced_vs_all_a8"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["sensitivity_rank"] = rank
    return rows


def run_disable_set(model, test_loader, device, selected, max_batches=None):
    for _, module in selected:
        module.disable_fake_quant()
    metrics = evaluate_scan_limited(model, test_loader, device, max_batches=max_batches)
    for _, module in selected:
        module.enable_fake_quant()
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, test_loader = build_quantized_scan(args, device)
    act_modules = collect_act_quantizers(model)
    print(f"Found {len(act_modules)} activation fake-quant modules.")

    coarse_batches = None if args.max_eval_batches <= 0 else args.max_eval_batches
    baseline_coarse = evaluate_scan_limited(model, test_loader, device, max_batches=coarse_batches)
    print("Baseline all W8A8 coarse:", baseline_coarse)
    rows = run_leave_one(model, test_loader, device, act_modules, baseline_coarse, coarse_batches)
    coarse_csv = os.path.join(args.out_dir, "scan_a8_leave_one_coarse.csv")
    write_csv(coarse_csv, rows)

    full_results = []
    top_names = [r["layer"] for r in rows[: args.full_top_k]]
    name_to_module = dict(act_modules)
    baseline_full = evaluate_scan_limited(model, test_loader, device, max_batches=None)
    full_results.append({"case": "all_w8a8", **baseline_full, "delta_accuracy_vs_all_a8": 0.0, "layers_disabled": ""})
    for name in top_names:
        module = name_to_module[name]
        module.disable_fake_quant()
        metrics = evaluate_scan_limited(model, test_loader, device, max_batches=None)
        module.enable_fake_quant()
        full_results.append({
            "case": "disable_one",
            **metrics,
            "delta_accuracy_vs_all_a8": metrics["accuracy"] - baseline_full["accuracy"],
            "layers_disabled": name,
        })

    for k in (2, 4, min(args.full_top_k, len(top_names))):
        selected = [(name, name_to_module[name]) for name in top_names[:k]]
        metrics = run_disable_set(model, test_loader, device, selected, max_batches=None)
        full_results.append({
            "case": f"disable_top{k}",
            **metrics,
            "delta_accuracy_vs_all_a8": metrics["accuracy"] - baseline_full["accuracy"],
            "layers_disabled": ";".join(name for name, _ in selected),
        })
    full_csv = os.path.join(args.out_dir, "scan_a8_top_full_confirm.csv")
    write_csv(full_csv, full_results)

    out = {
        "config": vars(args),
        "baseline_coarse": baseline_coarse,
        "baseline_full": baseline_full,
        "coarse_leave_one": rows,
        "full_confirm": full_results,
    }
    json_path = os.path.join(args.out_dir, "scan_a8_layer_sensitivity.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(coarse_csv)
    print(full_csv)
    print(json_path)


if __name__ == "__main__":
    main()
