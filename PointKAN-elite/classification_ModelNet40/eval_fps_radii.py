#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估 FPS radii MLP：在测试集上计算预测损失，并对若干样本做 GT vs 预测 曲线可视化。
需先运行 spec_kan_fps.py 生成 best.pt 与 radii_test_K*.npy。
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from spec_kan_fps import (
    Config,
    build_model,
    build_prefix_suffix_io,
    lin_resample_1d,
    eval_mse,
    RadiiIOTensorDataset,
    _radii_paths,
)


def parse_args():
    p = argparse.ArgumentParser(description="评估 FPS radii 模型（MLP/KAN）并可视化若干样本")
    p.add_argument("--save_dir", type=str, default="./runs/fps_radii_mlp", help="与训练时一致的保存目录")
    p.add_argument("--checkpoint", type=str, default="", help="权重文件，默认 save_dir/best.pt")
    p.add_argument("--radii_dir", type=str, default="", help="radii 缓存目录，默认用 save_dir；与训练时 --radii_dir 一致")
    p.add_argument("--num_vis", type=int, default=6, help="可视化样本数量")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def predict_full_curve(model, r: torch.Tensor, prefix_ratio: float, in_dim: int, out_dim: int, device):
    """给定单条 radii [K]，返回预测的完整曲线 [K]（prefix 真值 + suffix 预测插值）。"""
    K = r.shape[0]
    r = r.to(device).float().unsqueeze(0)  # [1, K]
    x, y = build_prefix_suffix_io(r, prefix_ratio, in_dim, out_dim)
    yhat = model(x)  # [1, out_dim]
    prefix_len = max(2, min(int(round(K * prefix_ratio)), K - 2))
    suffix_len = K - prefix_len
    yhat_suffix = lin_resample_1d(yhat, suffix_len)
    prefix_true = r[:, :prefix_len]
    full_pred = torch.cat([prefix_true, yhat_suffix], dim=1)
    return full_pred.squeeze(0).cpu().numpy()


def plot_sample(
    gt: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    title: str = "GT vs Pred",
    prefix_len: int = 0,
):
    """绘制 GT vs 预测曲线，并用竖线与背景区分输入段(prefix)与预测段(suffix)。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过保存图片。")
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    K = len(gt)
    x_axis = np.arange(K)
    ax.plot(x_axis, gt, "b-", label="GT", alpha=0.9, linewidth=1.5)
    ax.plot(x_axis, pred, "r--", label="Pred", alpha=0.9, linewidth=1.5)

    # 区分输入段与预测段：竖线 + 背景色
    if prefix_len > 0 and prefix_len < K:
        ax.axvline(x=prefix_len - 0.5, color="gray", linestyle="-", linewidth=1.2, alpha=0.8)
        ax.axvspan(0, prefix_len - 0.5, alpha=0.12, color="blue", label="输入段 (prefix)")
        ax.axvspan(prefix_len - 0.5, K, alpha=0.12, color="red", label="预测段 (suffix)")
        ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc="best")

    ax.set_xlabel("FPS step")
    ax.set_ylabel("radius")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or os.path.join(args.save_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到权重文件: {ckpt_path}，请先运行 spec_kan_fps.py 完成训练。")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config()
    for k, v in ckpt.get("cfg", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model_type = getattr(cfg, "model_type", "mlp")
    radii_dir = (args.radii_dir or getattr(cfg, "radii_dir", "") or "").strip() or args.save_dir

    _, test_radii_path = _radii_paths(radii_dir, cfg.fps_steps)
    if not os.path.isfile(test_radii_path):
        raise FileNotFoundError(
            f"未找到测试集 radii: {test_radii_path}，请先运行 spec_kan_fps.py 生成 radii_test_K{cfg.fps_steps}.npy"
        )

    test_radii = np.load(test_radii_path)
    ds_test = RadiiIOTensorDataset(
        test_radii, cfg.prefix_ratio, cfg.in_dim, cfg.out_dim
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=cfg.batch_size_train,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_mse = eval_mse(model, test_loader, device)
    print("=" * 50)
    print("测试集评估 (FPS Radii model_type=%s)" % model_type)
    print("=" * 50)
    print(f"  权重: {ckpt_path}")
    print(f"  测试样本数: {len(test_radii)}")
    print(f"  Test MSE: {test_mse:.6e}")
    print("=" * 50)

    # 选取若干样本做 GT vs 预测 曲线可视化（图中区分输入段 prefix 与预测段 suffix）
    num_vis = min(args.num_vis, len(test_radii))
    indices = np.linspace(0, len(test_radii) - 1, num_vis, dtype=int)
    K = test_radii.shape[1]
    prefix_len = max(2, min(int(round(K * cfg.prefix_ratio)), K - 2))

    vis_dir = os.path.join(args.save_dir, "vis")
    for i, idx in enumerate(indices):
        r = torch.from_numpy(test_radii[idx]).float()
        gt_curve = test_radii[idx]
        pred_curve = predict_full_curve(
            model, r, cfg.prefix_ratio, cfg.in_dim, cfg.out_dim, device
        )
        out_path = os.path.join(vis_dir, f"sample_{i:02d}_idx{idx}.png")
        plot_sample(
            gt_curve,
            pred_curve,
            out_path,
            title=f"Sample {i} (idx={idx})  prefix_len={prefix_len}",
            prefix_len=prefix_len,
        )
        print(f"  可视化已保存: {out_path}")

    print(f"\n可视化目录: {vis_dir}")


if __name__ == "__main__":
    main()
