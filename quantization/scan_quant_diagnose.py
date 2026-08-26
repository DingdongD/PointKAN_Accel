#!/usr/bin/env python3
"""
ScanObjectNN quantization diagnosis:
1) Float vs W8-only vs A8-only vs W8A8 (at a reference calibration size)
2) W8A8 calibration-size sweep
"""
from __future__ import absolute_import
import argparse
import json
import os
import random
import sys
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINTKAN_ELITE_ROOT = os.path.join(REPO_ROOT, "PointKAN-elite")
if POINTKAN_ELITE_ROOT not in sys.path:
    sys.path.insert(0, POINTKAN_ELITE_ROOT)

from quantization import quantize_model, calibrate, enable_quantization, QuantConfig
from quantization import evaluate_quant_pointkan as eq


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose ScanObjectNN quantization drop")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--calib_sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--a_observer", type=str, default="MinMaxObserver", help="Activation observer")
    p.add_argument("--a_percentile_low", type=float, default=0.001)
    p.add_argument("--a_percentile_high", type=float, default=0.999)
    p.add_argument("--a_hist_method", type=str, default="mse", choices=["mse", "kl"])
    p.add_argument("--a_hist_bins", type=int, default=2048)
    p.add_argument("--a_hist_start_bin", type=int, default=128)
    p.add_argument("--a_hist_max_samples", type=int, default=262144)
    p.add_argument("--calib_sampling", type=str, default="random", choices=["random", "class_balanced"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scan(args, calibration_size=None):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = eq.build_model_scan(num_classes=15, checkpoint_path=args.checkpoint, device=device)
    calib_loader, test_loader = eq.build_dataloaders_scan(
        num_points=args.num_points,
        batch_size=args.batch_size,
        workers=args.workers,
        calibration_size=calibration_size or max(args.calib_sizes),
        calib_sampling=args.calib_sampling,
        seed=args.seed,
    )
    return model, calib_loader, test_loader, device


def run_float(args):
    model, _, test_loader, device = build_scan(args)
    return eq.evaluate_scan(model, test_loader, device)


def run_quant(args, calib_size, mode):
    model, calib_loader, test_loader, device = build_scan(args, calibration_size=calib_size)
    quant_config = QuantConfig(
        model_name="pointKANElite_scan",
        a_observer=args.a_observer,
        a_percentile_low=args.a_percentile_low,
        a_percentile_high=args.a_percentile_high,
        a_hist_method=args.a_hist_method,
        a_hist_bins=args.a_hist_bins,
        a_hist_start_bin=args.a_hist_start_bin,
        a_hist_max_samples=args.a_hist_max_samples,
    )
    quantize_model(model, quant_config, use_input_quant=True, verbose=False)
    calibrate(model, calib_loader, calib_size, device, verbose=False)

    if mode == "w8a8":
        enable_quantization(model, quantizer_type="fake_quant")
    elif mode == "w8_only":
        enable_quantization(model, quantizer_type="weight_fake_quant")
    elif mode == "a8_only":
        enable_quantization(model, quantizer_type="act_fake_quant")
    else:
        raise ValueError("Unknown mode: %s" % mode)
    return eq.evaluate_scan(model, test_loader, device)


def fmt(name, m, base):
    d_acc = m["accuracy"] - base["accuracy"]
    d_bal = m["balanced_accuracy"] - base["balanced_accuracy"]
    return (
        f"{name:20s} | "
        f"Acc {m['accuracy']:6.2f}% ({d_acc:+6.2f}) | "
        f"Bal {m['balanced_accuracy']:6.2f}% ({d_bal:+6.2f})"
    )


def run_one_seed(args, float_m, include_ablation=True):
    calib_ref = max(args.calib_sizes)

    w8_only = run_quant(args, calib_ref, "w8_only") if include_ablation else None
    a8_only = run_quant(args, calib_ref, "a8_only") if include_ablation else None
    w8a8_ref = run_quant(args, calib_ref, "w8a8") if include_ablation else None

    sweep = {}
    for cs in args.calib_sizes:
        sweep[str(cs)] = run_quant(args, cs, "w8a8")
    return {
        "ablation_at_calib_ref": {
            "calib_ref": calib_ref,
            "w8_only": w8_only,
            "a8_only": a8_only,
            "w8a8": w8a8_ref,
        },
        "w8a8_calib_sweep": sweep,
    }


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "best": float(arr.max())}


def main():
    args = parse_args()
    seed_list = args.seeds if args.seeds else [args.seed]
    args.seed = int(seed_list[0])
    set_seed(args.seed)

    float_m = run_float(args)
    first = run_one_seed(args, float_m, include_ablation=True)
    sweep = first["w8a8_calib_sweep"]
    w8_only = first["ablation_at_calib_ref"]["w8_only"]
    a8_only = first["ablation_at_calib_ref"]["a8_only"]
    w8a8_ref = first["ablation_at_calib_ref"]["w8a8"]
    calib_ref = first["ablation_at_calib_ref"]["calib_ref"]

    print("=== ScanObjectNN Quant Diagnosis ===")
    print(fmt("float", float_m, float_m))
    print(fmt(f"w8_only@{calib_ref}", w8_only, float_m))
    print(fmt(f"a8_only@{calib_ref}", a8_only, float_m))
    print(fmt(f"w8a8@{calib_ref}", w8a8_ref, float_m))
    print("")
    print("=== W8A8 Calibration Sweep ===")
    for cs in args.calib_sizes:
        m = sweep[str(cs)]
        print(fmt(f"w8a8@{cs}", m, float_m))

    multi_seed = {}
    if len(seed_list) > 1:
        per_seed = {str(args.seed): first}
        for seed in seed_list[1:]:
            args.seed = int(seed)
            set_seed(args.seed)
            per_seed[str(args.seed)] = run_one_seed(args, float_m, include_ablation=False)
        aggregate = {}
        for cs in args.calib_sizes:
            accs = [per_seed[str(seed)]["w8a8_calib_sweep"][str(cs)]["accuracy"] for seed in seed_list]
            bals = [per_seed[str(seed)]["w8a8_calib_sweep"][str(cs)]["balanced_accuracy"] for seed in seed_list]
            aggregate[str(cs)] = {
                "accuracy": _mean_std(accs),
                "balanced_accuracy": _mean_std(bals),
            }
        multi_seed = {"seeds": seed_list, "per_seed": per_seed, "aggregate": aggregate}
        print("")
        print("=== Multi-seed W8A8 Sweep Aggregate ===")
        for cs in args.calib_sizes:
            a = aggregate[str(cs)]["accuracy"]
            b = aggregate[str(cs)]["balanced_accuracy"]
            print(
                f"w8a8@{cs:4d} | "
                f"Acc mean {a['mean']:6.2f}% std {a['std']:.2f} best {a['best']:6.2f}% | "
                f"Bal mean {b['mean']:6.2f}% std {b['std']:.2f} best {b['best']:6.2f}%"
            )

    out = {
        "calib_sampling": args.calib_sampling,
        "a_observer": args.a_observer,
        "a_percentile_low": args.a_percentile_low,
        "a_percentile_high": args.a_percentile_high,
        "a_hist_method": args.a_hist_method,
        "a_hist_bins": args.a_hist_bins,
        "a_hist_start_bin": args.a_hist_start_bin,
        "float": float_m,
        "ablation_at_calib_ref": {
            "calib_ref": calib_ref,
            "w8_only": w8_only,
            "a8_only": a8_only,
            "w8a8": w8a8_ref,
        },
        "w8a8_calib_sweep": sweep,
        "multi_seed": multi_seed,
    }
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved json: {args.out_json}")


if __name__ == "__main__":
    main()
