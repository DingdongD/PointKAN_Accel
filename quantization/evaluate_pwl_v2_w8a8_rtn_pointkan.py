#!/usr/bin/env python3
"""
对比：baseline(W8A8+RTN, vanilla GRF-KAN) vs PWL v2 + W8A8+RTN

流程：
1) 构建模型并加载 checkpoint
2) baseline 分支：直接 W8A8+RTN 量化 + 评测
3) PWL 分支：先注入 PWL v2（KAT_Group -> KAT_Group_PWL），再 W8A8+RTN 量化 + 评测

任务支持：
  --task modelnet40 | scan | partseg
"""
from __future__ import absolute_import

import argparse
import json
import os
import sys
import copy

import torch

from quantization import quantize_model, calibrate, calibrate_partseg, enable_quantization, QuantConfig

# 在 main() 里动态导入 evaluate_quant_pointkan（避免 sys.path 顺序导致 kat_rational 从错误目录被加载）
eq = None


def parse_args():
    p = argparse.ArgumentParser(description="PWL v2 + W8A8 RTN vs baseline")
    p.add_argument("--task", type=str, required=True, choices=["modelnet40", "scan", "partseg"])
    p.add_argument("--checkpoint", type=str, required=True, help="float checkpoint (.pth)")
    p.add_argument(
        "--pwl_v2_json",
        type=str,
        default="",
        help="pwl_v2.json 路径；不填则使用各任务默认候选（需与 checkpoint 对应）。",
    )
    p.add_argument("--pwl_bit", type=str, default=None, help="inject 时传入 bit_key；默认使用 JSON 的 meta 部署 bit")

    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--calibration_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")

    # observer 配置（可用于鲁棒校准）
    p.add_argument("--a_observer", type=str, default="MinMaxObserver")
    p.add_argument("--a_percentile_low", type=float, default=0.01)
    p.add_argument("--a_percentile_high", type=float, default=0.99)
    p.add_argument("--a_hist_method", type=str, default="mse", choices=["mse", "kl"])
    p.add_argument("--a_hist_bins", type=int, default=2048)
    p.add_argument("--a_hist_start_bin", type=int, default=128)
    p.add_argument("--a_hist_max_samples", type=int, default=262144)
    p.add_argument("--calib_sampling", type=str, default="random", choices=["random", "class_balanced"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out_json", type=str, default=None)
    return p.parse_args()


def default_pwl_json_for_task(task: str) -> str:
    # 当前脚本在 PointKAN-pytorch/quantization
    pointkan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if task == "scan":
        return os.path.join(pointkan_root, "PointKAN-elite", "classification_ScanObjectNN", "grf_export_out", "pwl_v2.json")
    if task == "partseg":
        return os.path.join(pointkan_root, "PointKAN-elite", "part_segmentation", "grf_export_out", "pwl_v2.json")
    if task == "modelnet40":
        # ModelNet40 的 pwl json 默认候选来自 GQA-LUT（与 run_inference_pwl_compare 对齐）
        cand1 = os.path.join(workspace_root, "GQA-LUT", "my_out", "pwl_v2.json")
        cand2 = os.path.join(workspace_root, "GQA-LUT", "my_compare_out", "pwl_v2.json")
        if os.path.isfile(cand1):
            return cand1
        return cand2
    raise ValueError(f"Unknown task {task}")


def get_inject_module(task: str):
    # 为了避免同名模块冲突，显式从任务目录导入 inject_grf_pwl_v2
    if task == "scan":
        task_dir = os.path.join(os.path.dirname(__file__), "..", "PointKAN-elite", "classification_ScanObjectNN")
    elif task == "modelnet40":
        task_dir = os.path.join(os.path.dirname(__file__), "..", "PointKAN-elite", "classification_ModelNet40")
    elif task == "partseg":
        task_dir = os.path.join(os.path.dirname(__file__), "..", "PointKAN-elite", "part_segmentation")
    else:
        raise ValueError(f"Unknown task {task}")

    task_dir = os.path.abspath(task_dir)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)

    from inject_grf_pwl_v2 import inject_grf_pwl_v2

    return inject_grf_pwl_v2


def build_for_task(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.task == "modelnet40":
        model = eq.build_model_modelnet40(num_classes=40, num_points=args.num_points, checkpoint_path=args.checkpoint, device=device)
        calib_loader, test_loader = eq.build_dataloaders_modelnet40(
            args.num_points, args.batch_size, args.workers, data_root=None, calibration_size=args.calibration_size
        )
        return model, calib_loader, test_loader, device
    if args.task == "scan":
        model = eq.build_model_scan(num_classes=15, checkpoint_path=args.checkpoint, device=device)
        calib_loader, test_loader = eq.build_dataloaders_scan(
            args.num_points,
            args.batch_size,
            args.workers,
            calibration_size=args.calibration_size,
            calib_sampling=args.calib_sampling,
            seed=args.seed,
        )
        return model, calib_loader, test_loader, device
    if args.task == "partseg":
        model = eq.build_model_partseg(num_part=50, checkpoint_path=args.checkpoint, device=device)
        calib_loader, test_loader, to_categorical, compute_overall_iou = eq.build_dataloaders_partseg(
            args.batch_size, args.workers, args.calibration_size
        )
        return model, calib_loader, test_loader, to_categorical, compute_overall_iou, device
    raise ValueError(f"Unknown task {args.task}")


def quantize_and_eval(model, calib_loader, test_loader, device, args):
    # QuantConfig：8-bit weight & activation + RTN 两阶段校准
    if args.task == "modelnet40":
        quant_config = QuantConfig(
            model_name="pointKANElite",
            a_observer=args.a_observer,
            a_percentile_low=args.a_percentile_low,
            a_percentile_high=args.a_percentile_high,
            a_hist_method=args.a_hist_method,
            a_hist_bins=args.a_hist_bins,
            a_hist_start_bin=args.a_hist_start_bin,
            a_hist_max_samples=args.a_hist_max_samples,
        )
        quantize_model(model, quant_config, use_input_quant=True, verbose=False)
        calibrate(model, calib_loader, args.calibration_size, device, verbose=False)
        enable_quantization(model, quantizer_type="fake_quant")
        metrics = eq.evaluate_modelnet40(model, test_loader, device)
        return metrics

    if args.task == "scan":
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
        calibrate(model, calib_loader, args.calibration_size, device, verbose=False)
        enable_quantization(model, quantizer_type="fake_quant")
        metrics = eq.evaluate_scan(model, test_loader, device)
        return metrics

    if args.task == "partseg":
        # partseg 需要专用校准
        to_categorical = args._to_categorical
        compute_overall_iou = args._compute_overall_iou
        quant_config = QuantConfig(
            model_name="pointKANElite_partseg",
            a_observer=args.a_observer,
            a_percentile_low=args.a_percentile_low,
            a_percentile_high=args.a_percentile_high,
            a_hist_method=args.a_hist_method,
            a_hist_bins=args.a_hist_bins,
            a_hist_start_bin=args.a_hist_start_bin,
            a_hist_max_samples=args.a_hist_max_samples,
        )
        quantize_model(model, quant_config, use_input_quant=True, verbose=False)
        calibrate_partseg(model, calib_loader, args.calibration_size, device, num_classes=16, verbose=False)
        enable_quantization(model, quantizer_type="fake_quant")
        metrics = eq.evaluate_partseg(model, test_loader, to_categorical, compute_overall_iou, num_part=50, num_classes=16, device=device)
        return metrics

    raise ValueError(f"Unknown task {args.task}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 强制 sys.path 优先级：让 kat_rational 来自 PointKAN-elite/kat_rational
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pointkan_elite_root = os.path.join(repo_root, "PointKAN-elite")
    if pointkan_elite_root not in sys.path:
        sys.path.insert(0, pointkan_elite_root)
    for k in list(sys.modules.keys()):
        if k == "kat_rational" or k.startswith("kat_rational."):
            del sys.modules[k]

    global eq
    from quantization import evaluate_quant_pointkan as eq  # noqa: F401

    pwl_v2_json = args.pwl_v2_json.strip()
    if not pwl_v2_json:
        pwl_v2_json = default_pwl_json_for_task(args.task)
    if not os.path.isfile(pwl_v2_json):
        raise FileNotFoundError(f"pwl_v2_json not found: {pwl_v2_json}")

    # baseline（vanilla GRF-KAN + W8A8）
    if args.task == "partseg":
        model, calib_loader, test_loader, to_categorical, compute_overall_iou, device = build_for_task(args)
        # 暂存引用，给 quantize_and_eval 使用
        args._to_categorical = to_categorical
        args._compute_overall_iou = compute_overall_iou
    else:
        model, calib_loader, test_loader, device = build_for_task(args)

    model_baseline = model  # 用这个对象跑 baseline（之后不再用它）
    model_baseline.eval()
    baseline_metrics = quantize_and_eval(model_baseline, calib_loader, test_loader, device, args)

    # PWL 分支：重新构建模型，注入 PWL v2，再跑 W8A8
    if args.task == "partseg":
        model2, calib_loader2, test_loader2, to_categorical2, compute_overall_iou2, device2 = build_for_task(args)
        args._to_categorical = to_categorical2
        args._compute_overall_iou = compute_overall_iou2
        calib_loader_use, test_loader_use = calib_loader2, test_loader2
    else:
        model2, calib_loader_use, test_loader_use, device2 = build_for_task(args)

    inject_grf_pwl_v2 = get_inject_module(args.task)
    model2.eval()
    replaced, bit_used = inject_grf_pwl_v2(model2, pwl_v2_json, bit_key=args.pwl_bit, device=device2)

    pwl_metrics = quantize_and_eval(model2, calib_loader_use, test_loader_use, device2, args)

    out = {
        "task": args.task,
        "checkpoint": args.checkpoint,
        "pwl_v2_json": os.path.abspath(pwl_v2_json),
        "pwl_bit_used": bit_used,
        "pwl_replaced_kat_groups": replaced,
        "calib_sampling": args.calib_sampling,
        "seed": args.seed,
        "a_observer": args.a_observer,
        "a_percentile_low": args.a_percentile_low,
        "a_percentile_high": args.a_percentile_high,
        "a_hist_method": args.a_hist_method,
        "a_hist_bins": args.a_hist_bins,
        "a_hist_start_bin": args.a_hist_start_bin,
        "baseline_w8a8": baseline_metrics,
        "pwl_w8a8": pwl_metrics,
    }

    # 输出差异（按 key 尝试计算 delta；不同任务 key 不同，故只做部分统计）
    def _delta(a, b, k):
        if k in a and k in b:
            return b[k] - a[k]
        return None

    if args.task in ("modelnet40", "scan"):
        for k in ("accuracy", "balanced_accuracy"):
            out[f"delta_{k}(pwl_minus_baseline)"] = _delta(baseline_metrics, pwl_metrics, k)
    else:
        for k in ("point_accuracy", "shape_accuracy", "mean_iou"):
            out[f"delta_{k}(pwl_minus_baseline)"] = _delta(baseline_metrics, pwl_metrics, k)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Result saved to: {args.out_json}")

    # 直接打印摘要
    print("===== Baseline (vanilla GRF-KAN + W8A8 RTN) =====")
    print(baseline_metrics)
    print("===== PWL v2 + W8A8 RTN =====")
    print(pwl_metrics)
    print("===== Delta (PWL - baseline) =====")
    for k, v in out.items():
        if k.startswith("delta_"):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
