#!/usr/bin/env python3
"""
加载检查点并在测试集上评估（ModelNet40 / ShapeNet-Part / S3DIS HDF5）。

示例::

    python -m openpoints_mini.scripts.infer --task cls --backbone pointnext --weights PATH
    python -m openpoints_mini.scripts.infer --task partseg --backbone pointvector --weights PATH
    python -m openpoints_mini.scripts.infer --task semseg --backbone pointnext --weights PATH \\
        --test_area 5 --in_channels 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT_PK = Path(__file__).resolve().parents[2]
if str(_ROOT_PK) not in sys.path:
    sys.path.insert(0, str(_ROOT_PK))

_ISEDA = Path(os.environ.get("ISEDA26_ROOT", "/workspace/ISEDA26"))

import openpoints_mini.models  # noqa: F401

from openpoints_mini.scripts.model_builder import (
    build_modelnet40_cls_model,
    build_s3dis_semseg_model,
    build_shapenet_part_model,
)


def _load_weights(model: nn.Module, path: Path, device):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    print(f"=> loaded weights from {path}")


def eval_cls(args, device):
    if not _ISEDA.is_dir():
        raise FileNotFoundError("需要 ISEDA26 / data_config")
    sys.path.insert(0, str(_ISEDA))
    from data_config import ModelNet40

    test_loader = DataLoader(
        ModelNet40(args.num_points, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_modelnet40_cls_model(args.backbone, device)
    _load_weights(model, Path(args.weights).expanduser(), device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for pts, label in test_loader:
            pts = pts.float().to(device)
            label = label.squeeze(-1).long().to(device)
            pred = model(pts).argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.numel()
    print(f"ModelNet40 test accuracy: {correct / max(total, 1):.4f}")


def eval_part(args, device):
    if not _ISEDA.is_dir():
        raise FileNotFoundError("需要 ISEDA26 / data_config")
    sys.path.insert(0, str(_ISEDA))
    from data_config import ShapeNetPart

    in_ch = args.part_in_channels
    test_loader = DataLoader(
        ShapeNetPart(args.num_points, "test"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = build_shapenet_part_model(args.backbone, device, in_channels=in_ch)
    _load_weights(model, Path(args.weights).expanduser(), device)
    model.eval()

    def _batch_miou(pred, seg, num_parts=50):
        pred = pred.view(-1).cpu().numpy()
        seg = seg.view(-1).cpu().numpy()
        ious = []
        for c in range(num_parts):
            inter = np.sum((pred == c) & (seg == c))
            union = np.sum((pred == c) | (seg == c))
            if union > 0:
                ious.append(inter / union)
        return float(np.mean(ious)) if ious else 0.0

    mious = []
    with torch.no_grad():
        for pts, cls_l, seg in test_loader:
            pts = pts.float().to(device)
            if pts.shape[-1] < in_ch:
                pad = in_ch - pts.shape[-1]
                pts = torch.cat(
                    [pts, torch.zeros(pts.shape[0], pts.shape[1], pad, device=device, dtype=pts.dtype)],
                    dim=-1,
                )
            elif pts.shape[-1] > in_ch:
                pts = pts[:, :, :in_ch]
            pos = pts[:, :, :3].contiguous()
            x = pts.transpose(1, 2).contiguous()
            cls_l = cls_l.long().to(device)
            if cls_l.dim() == 1:
                cls_l = cls_l.unsqueeze(1)
            seg = seg.long().to(device)
            logits = model({"pos": pos, "x": x, "cls": cls_l})
            pred = logits.argmax(dim=1)
            mious.append(_batch_miou(pred, seg))
    print(f"ShapeNet-Part test mIoU (approx per-batch mean): {float(np.mean(mious)):.4f}")


def eval_semseg(args, device):
    if not _ISEDA.is_dir():
        raise FileNotFoundError("需要 ISEDA26 / data_config")
    sys.path.insert(0, str(_ISEDA))
    from data_config import S3DIS

    from openpoints_mini.train_s3dis_pointnext_iseda26_data import (
        batch_to_openpoints,
        calculate_sem_iou,
    )

    test_loader = DataLoader(
        S3DIS(partition="test", num_points=args.num_points, test_area=args.test_area),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    model = build_s3dis_semseg_model(args.backbone, args.variant, args.in_channels, device)
    _load_weights(model, Path(args.weights).expanduser(), device)
    model.eval()
    pred_all, seg_all = [], []
    with torch.no_grad():
        for pts, seg in test_loader:
            pts = pts.float().to(device)
            seg = seg.to(device)
            inp = batch_to_openpoints(pts, args.in_channels)
            logits = model(inp)
            pred_all.append(logits.argmax(dim=1).cpu().numpy())
            seg_all.append(seg.cpu().numpy())
    pred_np = np.concatenate(pred_all, axis=0)
    seg_np = np.concatenate(seg_all, axis=0)
    ious = calculate_sem_iou(pred_np, seg_np)
    print(f"S3DIS Area {args.test_area} mIoU mean: {float(np.mean(ious)):.4f}  per-class IoU: {np.round(ious, 4)}")


def main():
    p = argparse.ArgumentParser(description="openpoints_mini 推理 / 评估")
    p.add_argument("--task", type=str, required=True, choices=["cls", "partseg", "semseg"])
    p.add_argument("--backbone", type=str, default="pointnext", choices=["pointnext", "pointvector"])
    p.add_argument("--weights", type=str, required=True, help="checkpoint_last.pth 或 checkpoint_best.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--part_in_channels", type=int, default=7)
    p.add_argument("--test_area", type=str, default="5")
    p.add_argument("--in_channels", type=int, default=6, help="S3DIS HDF5 特征维（与训练一致）")
    p.add_argument("--variant", type=str, default="l", choices=["l", "xl"])
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.task == "cls":
        eval_cls(args, device)
    elif args.task == "partseg":
        if args.num_points == 1024:
            args.num_points = 2048
        eval_part(args, device)
    else:
        eval_semseg(args, device)


if __name__ == "__main__":
    main()
