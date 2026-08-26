#!/usr/bin/env python3
"""
使用与 ISEDA26 ``pointnet_seg.py`` **完全相同**的 S3DIS 数据入口：
``from data_config import S3DIS``，以及 ``train_s3dis`` 里同款 ``DataLoader`` 参数
（``partition`` / ``num_points`` / ``test_area``、``batch_size``、``test_batch_size``、
``num_workers``、``shuffle`` / ``drop_last``）。

数据与 ``pointnet_seg.py`` 一致：由 ``data_config.load_data_semseg`` 读
``PC_DATA_DIR`` 下 ``indoor3d_sem_seg_hdf5_data*``；实例化 ``S3DIS`` 时会走
``download_S3DIS`` / ``prepare_test_data_semseg``（与 ISEDA 一致）。

**说明：** ``--epochs N`` 表示正式训练 N 个 epoch；若只想冒烟测试可用 ``--epochs 1``。

**6 折：** 加 ``--all_folds`` 会依次对 ``test_area=1…6`` 各训一整轮模型（每折从随机初始化开始），
并打印各折 **验证集最佳 mIoU** 及六折平均（S3DIS 常用协议；耗时约为单折的 6 倍）。

**断点续训：** 每 epoch 结束写入 ``{ckpt_dir}/{exp_name}/area_{N}/checkpoint_last.pth``（含
model / optimizer / scheduler / epoch / best_mIoU）。可用 ``--resume PATH`` 指定文件恢复；
或 ``--auto_resume``：若对应折的 ``checkpoint_last.pth`` 存在则自动接着训。

**验证频率：** ``--eval_every K``（默认 10）表示每训练 K 个 epoch 在验证集上算一次 mIoU；
最后一轮 epoch 总会验证。``checkpoint_best.pth`` 仅在验证时若刷新 best 才更新。

``prepare_s3dis_openpoints.sh`` 面向 OpenPoints 整房 **.npy**，与本脚本的 HDF5 管线无关。

**骨干：** ``--backbone pointnext``（默认）或 ``pointvector``；PointVector 可用 ``--variant l|xl``。
也可使用包装脚本 ``openpoints_mini/scripts/train_s3dis_semseg.sh``。

环境变量（与 ``data_config.py`` 一致）：
- ``PC_DATA_DIR``：默认 ``/workspace/VoxelNet/pc_data``
- ``ISEDA26_DATA_DIR``：仅影响部分 legacy 下载路径；S3DIS 主路径为 ``PC_DATA_DIR``
- ``ISEDA26_ROOT``：本脚本将此前缀加入 ``sys.path`` 以导入 ``data_config``（默认 ``/workspace/ISEDA26``）
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

# 注册 MODELS
_ROOT_PK = Path(__file__).resolve().parents[1]
if str(_ROOT_PK) not in sys.path:
    sys.path.insert(0, str(_ROOT_PK))

_ISEDA = Path(os.environ.get("ISEDA26_ROOT", "/workspace/ISEDA26"))
if not _ISEDA.is_dir():
    raise FileNotFoundError(
        f"未找到 ISEDA26 目录 {_ISEDA}，无法导入 data_config；请设置环境变量 ISEDA26_ROOT"
    )
if str(_ISEDA) not in sys.path:
    sys.path.insert(0, str(_ISEDA))

# 与 pointnet_seg.py 相同：from data_config import S3DIS（依赖上面对 ISEDA26 的 sys.path）
from data_config import S3DIS  # noqa: E402

import openpoints_mini.models  # noqa: F401

from openpoints_mini.scripts.model_builder import build_s3dis_semseg_model


def calculate_sem_iou(pred_np: np.ndarray, seg_np: np.ndarray, num_classes: int = 13):
    """与 ISEDA26 ``pointnet_seg.calculate_sem_IoU`` 相同逻辑。"""
    i_all = np.zeros(num_classes)
    u_all = np.zeros(num_classes)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(num_classes):
            i = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            u = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            i_all[sem] += i
            u_all[sem] += u
    return i_all / np.maximum(u_all, 1e-6)


def _fold_ckpt_dir(args, test_area: str) -> Path:
    return Path(args.ckpt_dir) / args.exp_name / f"area_{test_area}"


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    opt: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    best_miou: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    opt: optim.Optimizer,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    opt.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = int(ckpt["epoch"]) + 1
    best_miou = float(ckpt.get("best_miou", 0.0))
    print(f"=> loaded checkpoint '{path}' (resume from epoch {start_epoch}, best_mIoU={best_miou:.4f})")
    return start_epoch, best_miou


def _make_s3dis_loaders(test_area: str, args):
    train_loader = DataLoader(
        S3DIS(partition="train", num_points=args.num_points, test_area=test_area),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        S3DIS(partition="test", num_points=args.num_points, test_area=test_area),
        num_workers=args.num_workers,
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False,
    )
    return train_loader, test_loader


def train_one_fold(args, device: torch.device, test_area: str, log_prefix: str = "") -> float:
    """
    训练单个 test_area 折，返回该折在验证集上的最佳 mean IoU（跨 epoch 取 max）。
    """
    train_loader, test_loader = _make_s3dis_loaders(test_area, args)
    model = build_s3dis_semseg_model(args.backbone, args.variant, args.in_channels, device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    fold_dir = _fold_ckpt_dir(args, test_area)
    last_path = fold_dir / "checkpoint_last.pth"
    best_path = fold_dir / "checkpoint_best.pth"

    start_epoch = 0
    best_miou = 0.0
    resume_path: Path | None = None
    if getattr(args, "resume", None) and str(args.resume).strip():
        resume_path = Path(args.resume).expanduser()
    elif getattr(args, "auto_resume", False) and last_path.is_file():
        resume_path = last_path
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {resume_path}")
        start_epoch, best_miou = _load_checkpoint(resume_path, model, opt, scheduler, device)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for pts, seg in train_loader:
            pts = pts.float().to(device)
            seg = seg.to(device)
            inp = batch_to_openpoints(pts, args.in_channels)
            opt.zero_grad()
            logits = model(inp)
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
            pred_all, seg_all = [], []
            with torch.no_grad():
                for pts, seg in test_loader:
                    pts = pts.float().to(device)
                    seg = seg.to(device)
                    inp = batch_to_openpoints(pts, args.in_channels)
                    logits = model(inp)
                    pred = logits.argmax(dim=1)
                    pred_all.append(pred.cpu().numpy())
                    seg_all.append(seg.cpu().numpy())
            pred_np = np.concatenate(pred_all, axis=0)
            seg_np = np.concatenate(seg_all, axis=0)
            ious = calculate_sem_iou(pred_np, seg_np)
            miou = float(np.mean(ious))
            prev_best = best_miou
            best_miou = max(best_miou, miou)
            print(
                f"{log_prefix}epoch {epoch} train_loss={train_loss:.4f} "
                f"val_mIoU={miou:.4f} best={best_miou:.4f} lr={lr:.6f}"
            )
            _save_checkpoint(last_path, epoch, model, opt, scheduler, best_miou)
            if miou > prev_best + 1e-8:
                _save_checkpoint(best_path, epoch, model, opt, scheduler, best_miou)
        else:
            print(
                f"{log_prefix}epoch {epoch} train_loss={train_loss:.4f} "
                f"val=skipped (每 {args.eval_every} epoch 验证) best={best_miou:.4f} lr={lr:.6f}"
            )
            _save_checkpoint(last_path, epoch, model, opt, scheduler, best_miou)
    return best_miou


def batch_to_openpoints(pts: torch.Tensor, in_channels: int):
    """
    pts: (B, N, 9) Indoor3D 块 — 注释为 xyz + 法向 + rgb 等归一化特征。
    OpenPoints PointNeXt S3DIS 常用 in_channels=4（xyz+强度）或 6（xyz+rgb）。
    这里默认取前 in_channels 维作为 per-point 特征，坐标仍用前三维做几何。
    """
    pos = pts[:, :, :3].contiguous()
    feat = pts[:, :, :in_channels].float().transpose(1, 2).contiguous()
    return {"pos": pos, "x": feat}


def main():
    parser = argparse.ArgumentParser(
        description="S3DIS 语义分割训练（openpoints_mini），数据与 ISEDA26 pointnet_seg.train_s3dis 一致"
    )
    parser.add_argument("--test_area", type=str, default="5", help="单折：留出该 Area 做验证，其余训练（1–6）")
    parser.add_argument(
        "--all_folds",
        action="store_true",
        help="依次跑 test_area=1…6（六折各训一个新模型），最后打印各折最佳 mIoU 与均值",
    )
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32, help="训练 batch，与 pointnet_seg 默认一致")
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="测试 batch；默认与 --batch_size 相同（pointnet_seg 默认为 32）",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers，与 pointnet_seg 一致")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数；pointnet_seg 默认 200")
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="每隔多少个 epoch 在验证集上评估一次（最后一个 epoch 总会评估）；设为 1 表示每 epoch 验证",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--in_channels", type=int, default=6, help="encoder 输入通道（≤9，与 HDF5 块特征维一致）")
    parser.add_argument(
        "--backbone",
        type=str,
        default="pointnext",
        choices=["pointnext", "pointvector"],
        help="分割骨干：PointNeXt 或 PointVector",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="l",
        choices=["l", "xl"],
        help="PointVector-S3DIS 规模：l 或 xl（pointnext 仅实现 l，此参数对其无效）",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="检查点根目录；默认 PointKAN-pytorch/checkpoints_s3dis_pointnext",
    )
    parser.add_argument("--exp_name", type=str, default="s3dis_pointnext", help="实验子目录名")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="从指定 .pth 恢复（含优化器与调度器）；与 --auto_resume 二选一优先用本项",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="若本折 checkpoint_last.pth 已存在则自动续训（路径见文档）",
    )
    args = parser.parse_args()

    if args.eval_every < 1:
        raise SystemExit("错误: --eval_every 须为 >= 1 的整数")

    if args.ckpt_dir is None:
        args.ckpt_dir = str(_ROOT_PK / "checkpoints_s3dis_pointnext")

    if args.all_folds and str(args.resume).strip():
        raise SystemExit(
            "错误：--all_folds 时不要使用 --resume（同一文件无法对应 6 个 area）。"
            "六折续训请使用 --auto_resume，将按 area_1 … area_6 分别读取各自的 checkpoint_last.pth。"
        )

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    import data_config as _dc  # 与 ISEDA 同源，打印实际数据根目录

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"data_config.PC_DATA_DIR={_dc.PC_DATA_DIR}（S3DIS HDF5 根目录，与 pointnet_seg 相同）")
    print(f"检查点目录: {args.ckpt_dir}/{args.exp_name}/area_<N>/checkpoint_last.pth")
    print(f"验证频率: 每 {args.eval_every} 个 epoch（且最后一轮必验证）")

    if args.all_folds:
        areas = [str(i) for i in range(1, 7)]
        fold_best: list[float] = []
        for a in areas:
            print(f"\n========== S3DIS 6-fold: test_area={a} ({areas.index(a)+1}/6) ==========")
            b = train_one_fold(args, device, a, log_prefix=f"[Area{a}] ")
            fold_best.append(b)
            print(f"[Area{a}] 本折最佳 test_mIoU={b:.4f}")
        arr = np.array(fold_best, dtype=np.float64)
        print(
            "\n========== 6-fold 汇总（每折验证集最佳 mIoU）==========\n"
            f"  per-fold: {[round(x, 4) for x in fold_best]}\n"
            f"  mean={arr.mean():.4f}  std={arr.std():.4f}"
        )
    else:
        train_one_fold(args, device, args.test_area, log_prefix="")


if __name__ == "__main__":
    main()
