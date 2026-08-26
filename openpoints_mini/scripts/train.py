#!/usr/bin/env python3
"""
统一训练入口（ModelNet40 分类、ShapeNet-Part 部件分割）。

S3DIS 室内 HDF5 语义分割请使用（支持 PointNeXt / PointVector）::

    python -m openpoints_mini.train_s3dis_pointnext_iseda26_data --backbone pointnext --test_area 5

在 PointKAN-pytorch 根目录执行；需已 ``conda activate pointkan`` 并编译 pointnet2_cuda。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_ROOT_PK = Path(__file__).resolve().parents[2]
if str(_ROOT_PK) not in sys.path:
    sys.path.insert(0, str(_ROOT_PK))

_ISEDA = Path(os.environ.get("ISEDA26_ROOT", "/workspace/ISEDA26"))
if _ISEDA.is_dir() and str(_ISEDA) not in sys.path:
    sys.path.insert(0, str(_ISEDA))

import openpoints_mini.models  # noqa: F401

from openpoints_mini.scripts.model_builder import (
    build_modelnet40_cls_model,
    build_shapenet_part_model,
)


def _save_ckpt(path: Path, epoch: int, model, opt, scheduler, best_metric: float, key: str = "best_acc"):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            key: best_metric,
        },
        path,
    )


def _load_ckpt(path: Path, model, opt, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    opt.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["epoch"]) + 1, ckpt


def train_modelnet40(args, device):
    if not _ISEDA.is_dir():
        raise FileNotFoundError("需要 ISEDA26 下的 data_config（ModelNet40），请设置 ISEDA26_ROOT")
    from data_config import ModelNet40

    train_set = ModelNet40(args.num_points, "train")
    test_set = ModelNet40(args.num_points, "test")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_modelnet40_cls_model(args.backbone, device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(args.ckpt_dir) / args.exp_name / "modelnet40" / args.backbone
    last_p = ckpt_dir / "checkpoint_last.pth"
    best_p = ckpt_dir / "checkpoint_best.pth"

    start_epoch = 0
    best_acc = 0.0
    if args.auto_resume and last_p.is_file():
        start_epoch, ckpt = _load_ckpt(last_p, model, opt, scheduler, device)
        best_acc = float(ckpt.get("best_acc", 0.0))
        print(f"=> resume ModelNet40 from epoch {start_epoch}, best_acc={best_acc:.4f}")
    elif str(args.resume).strip():
        start_epoch, ckpt = _load_ckpt(Path(args.resume).expanduser(), model, opt, scheduler, device)
        best_acc = float(ckpt.get("best_acc", 0.0))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        correct = total = 0
        total_loss = 0.0
        for pts, label in train_loader:
            pts = pts.float().to(device)
            label = label.squeeze(-1).long().to(device)
            opt.zero_grad()
            logits = model(pts)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.numel()
            total_loss += loss.item() * label.size(0)
        scheduler.step()

        train_loss = total_loss / max(total, 1)
        tr_acc = correct / max(total, 1)
        lr = scheduler.get_last_lr()[0]
        do_val = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if do_val:
            model.eval()
            t_correct = t_total = 0
            with torch.no_grad():
                for pts, label in test_loader:
                    pts = pts.float().to(device)
                    label = label.squeeze(-1).long().to(device)
                    logits = model(pts)
                    pred = logits.argmax(dim=1)
                    t_correct += (pred == label).sum().item()
                    t_total += label.numel()
            te_acc = t_correct / max(t_total, 1)
            prev_best = best_acc
            best_acc = max(best_acc, te_acc)
            print(
                f"epoch {epoch} train_loss={train_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_acc={te_acc:.4f} best={best_acc:.4f} lr={lr:.6f}"
            )
            _save_ckpt(last_p, epoch, model, opt, scheduler, best_acc, "best_acc")
            if te_acc > prev_best + 1e-9:
                _save_ckpt(best_p, epoch, model, opt, scheduler, best_acc, "best_acc")
        else:
            print(
                f"epoch {epoch} train_loss={train_loss:.4f} train_acc={tr_acc:.4f} "
                f"val=skipped (每 {args.eval_every} epoch) best={best_acc:.4f} lr={lr:.6f}"
            )
            _save_ckpt(last_p, epoch, model, opt, scheduler, best_acc, "best_acc")


def train_shapenet_part(args, device):
    if not _ISEDA.is_dir():
        raise FileNotFoundError("需要 ISEDA26 下的 data_config（ShapeNetPart），请设置 ISEDA26_ROOT")
    from data_config import ShapeNetPart

    train_set = ShapeNetPart(args.num_points, "trainval")
    test_set = ShapeNetPart(args.num_points, "test")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    in_ch = args.part_in_channels
    model = build_shapenet_part_model(args.backbone, device, in_channels=in_ch)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(args.ckpt_dir) / args.exp_name / "shapenet_part" / args.backbone
    last_p = ckpt_dir / "checkpoint_last.pth"
    best_p = ckpt_dir / "checkpoint_best.pth"

    start_epoch = 0
    best_miou = 0.0
    if args.auto_resume and last_p.is_file():
        start_epoch, ckpt = _load_ckpt(last_p, model, opt, scheduler, device)
        best_miou = float(ckpt.get("best_miou", 0.0))
        print(f"=> resume ShapeNet-Part from epoch {start_epoch}, best_mIoU={best_miou:.4f}")
    elif str(args.resume).strip():
        start_epoch, ckpt = _load_ckpt(Path(args.resume).expanduser(), model, opt, scheduler, device)
        best_miou = float(ckpt.get("best_miou", 0.0))

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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        n = 0
        total_loss = 0.0
        for pts, cls_l, seg in train_loader:
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
            opt.zero_grad()
            logits = model({"pos": pos, "x": x, "cls": cls_l})
            loss = criterion(logits, seg)
            loss.backward()
            opt.step()
            total_loss += loss.item() * seg.size(0)
            n += seg.size(0)
        scheduler.step()

        train_loss = total_loss / max(n, 1)
        lr = scheduler.get_last_lr()[0]
        do_val = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if do_val:
            model.eval()
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
            te_miou = float(np.mean(mious)) if mious else 0.0
            prev_bm = best_miou
            best_miou = max(best_miou, te_miou)
            print(
                f"epoch {epoch} train_loss={train_loss:.4f} val_mIoU~={te_miou:.4f} "
                f"best={best_miou:.4f} lr={lr:.6f}"
            )
            _save_ckpt(last_p, epoch, model, opt, scheduler, best_miou, "best_miou")
            if te_miou > prev_bm + 1e-9:
                _save_ckpt(best_p, epoch, model, opt, scheduler, best_miou, "best_miou")
        else:
            print(
                f"epoch {epoch} train_loss={train_loss:.4f} val=skipped (每 {args.eval_every} epoch) "
                f"best={best_miou:.4f} lr={lr:.6f}"
            )
            _save_ckpt(last_p, epoch, model, opt, scheduler, best_miou, "best_miou")


def main():
    p = argparse.ArgumentParser(description="openpoints_mini：ModelNet40 / ShapeNet-Part 训练")
    p.add_argument("--task", type=str, required=True, choices=["cls", "partseg"])
    p.add_argument(
        "--backbone",
        type=str,
        default="pointnext",
        choices=["pointnext", "pointvector"],
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="每隔多少 epoch 在验证集上评估；最后一轮总会评估；1 表示每 epoch 验证",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--test_batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_points", type=int, default=1024, help="ModelNet 默认 1024；ShapeNet 常用 2048")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--exp_name", type=str, default="openpoints_exp")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--auto_resume", action="store_true")
    p.add_argument(
        "--part_in_channels",
        type=int,
        default=7,
        help="ShapeNet 点特征维（HDF5 常见 6 或 7；若实际更小会自动补零到该维）",
    )
    args = p.parse_args()
    if args.eval_every < 1:
        raise SystemExit("错误: --eval_every 须 >= 1")
    if args.ckpt_dir is None:
        args.ckpt_dir = str(_ROOT_PK / "checkpoints_s3dis_pointnext")
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.task == "cls":
        train_modelnet40(args, device)
    else:
        if args.num_points == 1024:
            args.num_points = 2048
            print("提示: ShapeNet-Part 已将 --num_points 从 1024 改为 2048（可用参数覆盖）")
        train_shapenet_part(args, device)


if __name__ == "__main__":
    main()
