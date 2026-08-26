#!/usr/bin/env python3
"""
评估 PointKAN-Elite 校准后 W8A8 RTN 量化精度。
用法:
  python evaluate_quant_pointkan.py --task modelnet40 --checkpoint <path> [--num_points 1024] [--calibration_size 512]
  python evaluate_quant_pointkan.py --task partseg --checkpoint <path> [--calibration_size 512]
  python evaluate_quant_pointkan.py --task scan --checkpoint <path> [--calibration_size 512]
"""
from __future__ import absolute_import
import argparse
import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

# 使用 PointKAN-pytorch 根目录
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from quantization import quantize_model, calibrate, enable_quantization, QuantConfig

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("pointkan_quant")


def parse_args():
    p = argparse.ArgumentParser(description="PointKAN-Elite W8A8 RTN evaluation")
    p.add_argument("--task", type=str, required=True, choices=["modelnet40", "partseg", "scan"], help="Task")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to float checkpoint (.pth)")
    p.add_argument("--num_points", type=int, default=1024, help="Point number (modelnet40/scan)")
    p.add_argument("--calibration_size", type=int, default=512, help="Calibration samples")
    p.add_argument("--a_observer", type=str, default="MinMaxObserver", help="Activation observer")
    p.add_argument("--a_percentile_low", type=float, default=0.001, help="Activation percentile low (PercentileObserver)")
    p.add_argument("--a_percentile_high", type=float, default=0.999, help="Activation percentile high (PercentileObserver)")
    p.add_argument("--a_hist_method", type=str, default="mse", choices=["mse", "kl"], help="Activation histogram clipping objective")
    p.add_argument("--a_hist_bins", type=int, default=2048, help="Activation histogram bins")
    p.add_argument("--a_hist_start_bin", type=int, default=128, help="First histogram threshold bin to search")
    p.add_argument("--a_hist_max_samples", type=int, default=262144, help="Max sampled activations per observer")
    p.add_argument("--calib_sampling", type=str, default="random", choices=["random", "class_balanced"], help="Calibration subset sampling")
    p.add_argument("--seed", type=int, default=42, help="Calibration sampling seed")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_root", type=str, default=None, help="Override data dir (e.g. ModelNet40/Scan)")
    return p.parse_args()


def build_model_modelnet40(num_classes=40, num_points=1024, checkpoint_path=None, device="cuda"):
    sys.path.insert(0, os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40"))
    import models as models_cls
    model = models_cls.pointKANElite(num_classes=num_classes)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # 不同训练脚本保存字段可能不同（例如 model / net / state_dict）
        state = ckpt.get("model") or ckpt.get("net") or ckpt.get("state_dict") or ckpt
        if state is None:
            state = ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        incompatible = model.load_state_dict(new_state, strict=False)
        logger.info(
            "Loaded ModelNet40 checkpoint: %s (missing=%d, unexpected=%d)",
            checkpoint_path,
            len(getattr(incompatible, "missing_keys", [])),
            len(getattr(incompatible, "unexpected_keys", [])),
        )
        if getattr(incompatible, "missing_keys", None):
            logger.info("Sample missing keys: %s", getattr(incompatible, "missing_keys", [])[:10])
        if getattr(incompatible, "unexpected_keys", None):
            logger.info("Sample unexpected keys: %s", getattr(incompatible, "unexpected_keys", [])[:10])
    model = model.to(device)
    model.eval()
    return model


def build_dataloaders_modelnet40(num_points, batch_size, workers, data_root=None, calibration_size=512):
    sys.path.insert(0, os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40"))
    from data import ModelNet40
    if data_root:
        import data as data_mod
        data_mod.load_data = lambda partition: _load_data_root(data_root, partition)
    train_set = ModelNet40(num_points=num_points, partition="train")
    test_set = ModelNet40(num_points=num_points, partition="test")
    calib_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return calib_loader, test_loader


def _load_data_root(data_root, partition):
    import glob
    import h5py
    all_data, all_label = [], []
    for h5_name in glob.glob(os.path.join(data_root, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition)):
        with h5py.File(h5_name, "r") as f:
            all_data.append(f["data"][:].astype("float32"))
            all_label.append(f["label"][:].astype("int64"))
    return np.concatenate(all_data, axis=0), np.concatenate(all_label, axis=0)


def evaluate_modelnet40(model, test_loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device).permute(0, 2, 1)
            out = model(data)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            pred = logits.max(dim=1)[1]
            all_pred.append(pred.cpu().numpy())
            # ModelNet40 的 dataloader 里 label 通常是 (B, 1)，需要 squeeze 成 (B,)
            all_true.append(label.view(-1).cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    acc = 100.0 * np.mean(all_pred == all_true)
    try:
        from sklearn.metrics import balanced_accuracy_score
        acc_avg = 100.0 * balanced_accuracy_score(all_true, all_pred)
    except Exception:
        acc_avg = acc
    return {"accuracy": acc, "balanced_accuracy": acc_avg}


def run_modelnet40(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model_modelnet40(num_classes=40, num_points=args.num_points, checkpoint_path=args.checkpoint, device=device)
    calib_loader, test_loader = build_dataloaders_modelnet40(
        args.num_points, args.batch_size, args.workers, args.data_root, args.calibration_size
    )
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
    quantize_model(model, quant_config, use_input_quant=True, verbose=True)
    calibrate(model, calib_loader, args.calibration_size, device, verbose=True)
    # 一次性同时启用 act/weight 的 fake quant：
    # 之前分别启用 act 后再启用 weight，会导致后一次把前一次启用的 act fake-quant 又关闭，
    # 最终推理实际并不是完整的 A8W8。
    enable_quantization(model, quantizer_type="fake_quant")
    metrics = evaluate_modelnet40(model, test_loader, device)
    logger.info("ModelNet40 W8A8 RTN — Accuracy: %.2f%%, Balanced: %.2f%%", metrics["accuracy"], metrics["balanced_accuracy"])
    return metrics


def build_model_partseg(num_part=50, checkpoint_path=None, device="cuda"):
    sys.path.insert(0, os.path.join(REPO_ROOT, "PointKAN-elite", "part_segmentation"))
    import model as models_seg
    model = models_seg.__dict__["pointKANElite"](num_part)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model") or ckpt.get("net") or ckpt.get("state_dict") or ckpt
        if state is None:
            state = ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        incompatible = model.load_state_dict(new_state, strict=False)
        logger.info(
            "Loaded PartSeg checkpoint: %s (missing=%d, unexpected=%d)",
            checkpoint_path,
            len(getattr(incompatible, "missing_keys", [])),
            len(getattr(incompatible, "unexpected_keys", [])),
        )
        if getattr(incompatible, "missing_keys", None):
            logger.info("Sample missing keys: %s", getattr(incompatible, "missing_keys", [])[:10])
        if getattr(incompatible, "unexpected_keys", None):
            logger.info("Sample unexpected keys: %s", getattr(incompatible, "unexpected_keys", [])[:10])
    model = model.to(device)
    model.eval()
    return model


def build_dataloaders_partseg(batch_size, workers, calibration_size=512):
    sys.path.insert(0, os.path.join(REPO_ROOT, "PointKAN-elite", "part_segmentation"))
    from util.data_util import PartNormalDataset
    from util.util import to_categorical, compute_overall_iou
    train_set = PartNormalDataset(npoints=2048, split="trainval", normalize=False)
    test_set = PartNormalDataset(npoints=2048, split="test", normalize=False)
    calib_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return calib_loader, test_loader, to_categorical, compute_overall_iou


def evaluate_partseg(model, test_loader, to_categorical, compute_overall_iou, num_part=50, num_classes=16, device="cuda"):
    model.eval()
    all_ious = []
    all_point_acc = []
    all_shape_acc = []
    with torch.no_grad():
        for batch in test_loader:
            points, label, seg, normal = batch[0], batch[1], batch[2], batch[3]
            points = points.to(device).permute(0, 2, 1)
            norm_plt = normal.to(device).permute(0, 2, 1)
            label_one_hot = to_categorical(label.squeeze(-1), num_classes).to(device)
            seg_pred = model(points, norm_plt, label_one_hot)
            seg_pred = seg_pred[0] if isinstance(seg_pred, (list, tuple)) else seg_pred
            target = seg.to(device)
            # seg_pred: (B, N, num_part) -> pred label: (B, N)
            pred = seg_pred.max(dim=2)[1]
            all_point_acc.append((pred == target).float().mean().item())  # point-wise
            all_shape_acc.append((pred == target).float().mean(dim=1).mean().item())  # per-shape
            batch_ious = compute_overall_iou(seg_pred, target, num_part)
            all_ious.extend(batch_ious)
    mean_iou = np.mean(all_ious) * 100.0
    point_accuracy = np.mean(all_point_acc) * 100.0
    shape_accuracy = np.mean(all_shape_acc) * 100.0
    return {"point_accuracy": point_accuracy, "shape_accuracy": shape_accuracy, "mean_iou": mean_iou}


def run_partseg(args):
    from quantization import calibrate_partseg
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model_partseg(num_part=50, checkpoint_path=args.checkpoint, device=device)
    calib_loader, test_loader, to_categorical, compute_overall_iou = build_dataloaders_partseg(args.batch_size, args.workers, args.calibration_size)
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
    quantize_model(model, quant_config, use_input_quant=True, verbose=True)
    calibrate_partseg(model, calib_loader, args.calibration_size, device, num_classes=16, verbose=True)
    enable_quantization(model, quantizer_type="fake_quant")
    metrics = evaluate_partseg(model, test_loader, to_categorical, compute_overall_iou, num_part=50, num_classes=16, device=device)
    logger.info(
        "PartSeg W8A8 RTN — PointAcc: %.2f%%, ShapeAcc: %.2f%%, Mean IoU: %.2f%%",
        metrics["point_accuracy"],
        metrics["shape_accuracy"],
        metrics["mean_iou"],
    )
    return metrics


def build_model_scan(num_classes=15, checkpoint_path=None, device="cuda"):
    sys.path.insert(0, os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ScanObjectNN"))
    import models as models_scan
    model = models_scan.pointKANElite(num_classes=num_classes)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model") or ckpt.get("net") or ckpt.get("state_dict") or ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        incompatible = model.load_state_dict(new_state, strict=False)
        logger.info(
            "Loaded Scan checkpoint: %s (missing=%d, unexpected=%d)",
            checkpoint_path,
            len(getattr(incompatible, "missing_keys", [])),
            len(getattr(incompatible, "unexpected_keys", [])),
        )
        if getattr(incompatible, "missing_keys", None):
            logger.info("Sample missing keys: %s", getattr(incompatible, "missing_keys", [])[:10])
        if getattr(incompatible, "unexpected_keys", None):
            logger.info("Sample unexpected keys: %s", getattr(incompatible, "unexpected_keys", [])[:10])
    model = model.to(device)
    model.eval()
    return model


def build_dataloaders_scan(num_points, batch_size, workers, calibration_size=512, calib_sampling="random", seed=42):
    sys.path.insert(0, os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ScanObjectNN"))
    from ScanObjectNN import ScanObjectNN
    from quantization.calibration_sampling import build_calibration_loader
    train_set = ScanObjectNN(partition="training", num_points=num_points)
    test_set = ScanObjectNN(partition="test", num_points=num_points)
    calib_loader = build_calibration_loader(
        train_set,
        batch_size=batch_size,
        workers=workers,
        calibration_size=calibration_size,
        sampling=calib_sampling,
        seed=seed,
        drop_last=True,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    return calib_loader, test_loader


def evaluate_scan(model, test_loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            if data.dim() == 3 and data.shape[-1] == 3:
                data = data.permute(0, 2, 1)
            out = model(data)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            pred = logits.max(dim=1)[1]
            all_pred.append(pred.cpu().numpy())
            # ScanObjectNN 的 dataloader 里 label 也可能是 (B, 1)
            all_true.append(label.view(-1).cpu().numpy())
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    acc = 100.0 * np.mean(all_pred == all_true)
    try:
        from sklearn.metrics import balanced_accuracy_score
        acc_avg = 100.0 * balanced_accuracy_score(all_true, all_pred)
    except Exception:
        acc_avg = acc
    return {"accuracy": acc, "balanced_accuracy": acc_avg}


def run_scan(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model_scan(num_classes=15, checkpoint_path=args.checkpoint, device=device)
    calib_loader, test_loader = build_dataloaders_scan(
        args.num_points,
        args.batch_size,
        args.workers,
        args.calibration_size,
        calib_sampling=args.calib_sampling,
        seed=args.seed,
    )
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
    quantize_model(model, quant_config, use_input_quant=True, verbose=True)
    calibrate(model, calib_loader, args.calibration_size, device, verbose=True)
    enable_quantization(model, quantizer_type="fake_quant")
    metrics = evaluate_scan(model, test_loader, device)
    logger.info(
        "ScanObjectNN W8A8 RTN — Accuracy: %.2f%%, Balanced: %.2f%%",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
    )
    return metrics


def main():
    args = parse_args()
    if args.task == "modelnet40":
        run_modelnet40(args)
    elif args.task == "partseg":
        run_partseg(args)
    elif args.task == "scan":
        run_scan(args)
    else:
        raise ValueError("Unknown task: %s" % args.task)


if __name__ == "__main__":
    main()
