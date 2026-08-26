#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN 粗筛+细筛加强版：IoU 分析与 ModelNet40 推理评估

1. 在 FPS 基础上对比三种 KNN：
   - vanilla：精确 topk
   - adaptive：原 fps_with_knn_adaptive（阈值迭代，候选数在 [k*(1-tol), k*(1+tol)]）
   - coarse_fine：粗筛（候选数 >= min_candidates_ratio*k）+ 细筛 topk

2. 计算 coarse_fine / adaptive 相对 vanilla 的邻居集合 IoU。

3. 可选：调用与 inference_and_kan_analysis 一致的推理流程，汇报 ModelNet40 测试集准确率
   （当前分类模型内部仍为 vanilla KNN；本脚本主要验证粗筛+细筛与 vanilla 的 IoU 及 radii 差异）。

用法：
  python run_knn_coarse_fine_eval.py --k_neighbors 24 --fps_steps 512
  python run_knn_coarse_fine_eval.py --k_neighbors 24 --max_samples 500 --run_inference --checkpoint checkpoints/pointKANElite-pointkan_elite_exp-42
  python run_knn_coarse_fine_eval.py --no_inference --out_json ./knn_iou_results.json
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None  # type: ignore

from spec_kan_fps import (
    fps_with_knn_adaptive,
    fps_with_knn_coarse_fine,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
    ModelNet40Wrapper,
    collate_points,
)

# 有的分支/版本里 fused 版本可能不存在：存在则优先用 fused（更省一次遍历），否则自动回退
try:
    from spec_kan_fps import fps_with_knn_coarse_fine_fused  # type: ignore
except Exception:
    fps_with_knn_coarse_fine_fused = None  # type: ignore

# blockwise fused 版本：存在则可用于统计每个 block 的候选数量
try:
    from spec_kan_fps import fps_with_knn_coarse_fine_fused_blockwise  # type: ignore
except Exception:
    fps_with_knn_coarse_fine_fused_blockwise = None  # type: ignore

@torch.no_grad()
def fps_with_knn_coarse_fine_fused_blockwise_instrumented(
    X: torch.Tensor,
    K: int,
    k_neighbors: int = 24,
    init_threshold: float = 0.1,
    max_iter: int = 10,
    min_candidates_ratio: float = 1.2,
    tol: float = 0.1,
    seed: str = "first",
    return_distances_matrix: bool = True,
    exclude_self: bool = False,
    block_size: int = 128,
):
    """
    融合版 FPS + blockwise coarse-fine KNN（带统计版）。

    返回：
      idx, radii, knn_idx, distances_matrix, knn_radii, thresholds, iters_used
    """
    assert X.ndim == 3
    assert block_size > 0

    B, N, D = X.shape
    device, dtype = X.device, X.dtype

    K = min(K, N)
    k_actual = min(k_neighbors, N)
    min_candidates = max(k_actual + 1, int(k_actual * min_candidates_ratio))

    x2 = (X * X).sum(dim=-1)  # [B, N]

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=dtype, device=device)
    thresholds = torch.zeros((B, K), dtype=dtype, device=device)
    knn_idx = torch.zeros((B, K, k_actual), dtype=torch.long, device=device)
    knn_radii = torch.zeros((B, K), dtype=dtype, device=device)
    iters_used = torch.zeros((B, K), dtype=torch.int16, device=device)

    if return_distances_matrix:
        distances_matrix = torch.empty((B, K, N), dtype=dtype, device=device)
    else:
        distances_matrix = torch.empty((0,), dtype=dtype, device=device)

    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_arange = torch.arange(B, device=device)

    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        farthest = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    for t in range(K):
        idx[:, t] = farthest

        C = X[batch_arange, farthest]                          # [B, D]
        c2 = (C * C).sum(dim=-1, keepdim=True)                # [B, 1]
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)               # [B, N]
        d = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()      # [B, N]

        if return_distances_matrix:
            distances_matrix[:, t, :] = d

        if exclude_self:
            d_knn = d.clone()
            d_knn[batch_arange, farthest] = float("inf")
        else:
            d_knn = d

        threshold = torch.full((B,), init_threshold, dtype=dtype, device=device)
        final_candidate_mask = None
        used = 0
        for i in range(max_iter):
            used = i + 1
            candidate_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            for start in range(0, N, block_size):
                end = min(start + block_size, N)
                d_blk = d_knn[:, start:end]
                blk_mask = d_blk < threshold.unsqueeze(-1)
                candidate_mask[:, start:end] |= blk_mask
            count = candidate_mask.sum(dim=1)
            final_candidate_mask = candidate_mask
            if (count >= min_candidates).all():
                break
            threshold = torch.where(count >= min_candidates, threshold, threshold * (1 + tol))

        thresholds[:, t] = threshold
        iters_used[:, t] = used

        masked_d = d_knn.masked_fill(~final_candidate_mask, float("inf"))
        knn_vals, knn_idx_t = torch.topk(masked_d, k=k_actual, dim=1, largest=False)
        knn_idx[:, t, :] = knn_idx_t
        knn_radii[:, t] = knn_vals[:, -1]

        torch.minimum(min_dist, d, out=min_dist)
        min_dist[batch_arange, farthest] = 0.0
        radii[:, t], _ = torch.max(min_dist, dim=1)
        if t < K - 1:
            farthest = torch.argmax(min_dist, dim=1)

    return idx, radii, knn_idx, distances_matrix, knn_radii, thresholds, iters_used


if fps_with_knn_coarse_fine_fused_blockwise is None:
    @torch.no_grad()
    def fps_with_knn_coarse_fine_fused_blockwise(  # type: ignore
        X: torch.Tensor,
        K: int,
        k_neighbors: int = 24,
        init_threshold: float = 0.1,
        max_iter: int = 10,
        min_candidates_ratio: float = 1.2,
        tol: float = 0.1,
        seed: str = "first",
        return_distances_matrix: bool = True,
        exclude_self: bool = False,
        block_size: int = 256,
    ):
        """
        本地兜底实现（无 iters 输出）：保持与原函数 6 返回一致，供纯性能/IoU 测试使用。
        """
        assert X.ndim == 3
        assert block_size > 0

        idx, radii, knn_idx, distances_matrix, knn_radii, thresholds, _iters = (
            fps_with_knn_coarse_fine_fused_blockwise_instrumented(
                X, K,
                k_neighbors=k_neighbors,
                init_threshold=init_threshold,
                max_iter=max_iter,
                min_candidates_ratio=min_candidates_ratio,
                tol=tol,
                seed=seed,
                return_distances_matrix=return_distances_matrix,
                exclude_self=exclude_self,
                block_size=block_size,
            )
        )
        return idx, radii, knn_idx, distances_matrix, knn_radii, thresholds


def parse_args():
    p = argparse.ArgumentParser(description="KNN 粗筛+细筛 IoU 分析与 ModelNet40 推理评估")
    p.add_argument("--num_points", type=int, default=1024, help="点云点数")
    p.add_argument("--fps_steps", type=int, default=512, help="FPS 采样数 K")
    p.add_argument("--k_neighbors", type=int, default=24, help="KNN 邻居数 k")
    p.add_argument("--batch_size", type=int, default=16, help="FPS/KNN 批大小")
    p.add_argument("--max_samples", type=int, default=0, help="最多评估样本数，0 表示全部测试集")
    p.add_argument("--min_candidates_ratio", type=float, default=1.2,
                   help="粗筛+细筛：候选数 >= min_candidates_ratio * k_neighbors")
    p.add_argument("--tol", type=float, default=0.05, help="adaptive 的 tol")
    p.add_argument("--init_threshold_adaptive", type=float, default=0.15)
    p.add_argument("--max_iter_adaptive", type=int, default=10)
    p.add_argument("--init_threshold_coarse_fine", type=float, default=0.15)
    p.add_argument("--max_iter_coarse_fine", type=int, default=10)
    p.add_argument("--init_threshold_blockwise", type=float, default=0.2)
    p.add_argument("--max_iter_blockwise", type=int, default=10)
    p.add_argument("--run_blockwise", action="store_true",
                   help="额外评估 blockwise coarse-fine（fused_blockwise）并统计候选数/加速比")
    p.add_argument("--block_size", type=int, default=256, help="blockwise 粗筛的 block_size")
    p.add_argument("--collect_block_stats", action="store_true",
                   help="统计每个 block 的候选数与最终 masked 数（会返回 distances_matrix，显存开销更大）")
    p.add_argument("--sweep_blockwise", action="store_true",
                   help="扫描 init_threshold/tol/max_iter 并输出 blockwise-vs-vanilla 的 IoU/速度热力图")
    p.add_argument("--sweep_init_thresholds", type=str, default="0.05,0.1,0.15,0.2,0.25",
                   help="逗号分隔的 init_threshold 列表（sweep；作用于 blockwise）")
    p.add_argument("--sweep_tols", type=str, default="0.01,0.03,0.05,0.08",
                   help="逗号分隔的 tol 列表（sweep；每个 tol 输出一张热力图）")
    p.add_argument("--sweep_max_iters", type=str, default="2,4,6,8,10",
                   help="逗号分隔的 max_iter 列表（sweep；作用于 blockwise）")
    p.add_argument("--sweep_out_prefix", type=str, default="./sweep_blockwise",
                   help="sweep 输出前缀（会写 *_iou_tol=...png 与 *_speedup_tol=...png）")
    p.add_argument("--sweep_plot", type=str, default="heatmap", choices=["heatmap", "surface", "both"],
                   help="sweep 输出图类型：heatmap/surface/both")
    p.add_argument("--sweep_fixed_threshold", type=float, default=None,
                   help="若设置（如 0.15），则固定 init_threshold，扫描 tol × max_iter（adaptive vs vanilla），输出 surface")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root", type=str, default="./data", help="ModelNet40 数据根目录")
    p.add_argument("--out_json", type=str, default="", help="结果写入 JSON 路径")
    p.add_argument("--run_inference", action="store_true", help="是否运行分类推理（与 inference_and_kan_analysis 一致）")
    p.add_argument("--no_inference", action="store_true", help="仅做 IoU/radii 分析，不跑推理")
    p.add_argument("--checkpoint", type=str, default="checkpoints/pointKANElite-pointkan_elite_exp-42",
                   help="推理用检查点目录或 best_checkpoint.pth 路径")
    p.add_argument("--inference_batch_size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _parse_csv_floats(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def plot_iou_heatmap(
    threshold_list,
    max_iter_list,
    Z,
    save_path="iou_heatmap.png",
    title="IoU Heatmap",
    cbar_label="IoU",
    xlabel=None,
    ylabel=None,
    cmap="viridis",
    font_family="Arial",
    label_fontsize=20,
    tick_fontsize=18,
    cbar_fontsize=20,
):
    """与 plot_iou_surface 一致的论文风格：白底黑字、Arial、统一字号与 colorbar。"""
    if plt is None:
        raise ImportError("matplotlib 不可用，无法绘图；请先安装 matplotlib。")
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    threshold_list = np.asarray(threshold_list, dtype=float)
    max_iter_list = np.asarray(max_iter_list, dtype=float)
    Z = np.asarray(Z, dtype=float)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "Helvetica", "Arial", "DejaVu Sans"],
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.edgecolor": "black",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    ax.set_facecolor("white")

    norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    cmap_obj = mpl.colormaps[cmap]
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap_obj, norm=norm)

    ax.set_xticks(np.arange(len(threshold_list)))
    ax.set_xticklabels([f"{x:.3f}" for x in threshold_list], rotation=45, fontsize=tick_fontsize, color="black")
    ax.set_yticks(np.arange(len(max_iter_list)))
    ax.set_yticklabels([str(int(x)) if float(x) == int(x) else str(x) for x in max_iter_list], fontsize=tick_fontsize, color="black")

    ax.set_xlabel(xlabel if xlabel is not None else "init_threshold", fontsize=label_fontsize, color="black")
    ax.set_ylabel(ylabel if ylabel is not None else "max_iter", fontsize=label_fontsize, color="black")
    if title:
        ax.set_title(title, fontsize=label_fontsize, color="black")
    ax.tick_params(axis="both", labelsize=tick_fontsize, colors="black")

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(cbar_label, fontsize=cbar_fontsize, color="black")
    cbar.ax.tick_params(labelsize=tick_fontsize, colors="black")
    cbar.outline.set_edgecolor("black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.cm import ScalarMappable


def plot_iou_surface(
    threshold_list,
    max_iter_list,
    Z,
    save_path="iou_surface.png",
    zlabel="IoU",
    xlabel=None,
    ylabel=None,
    title=None,
    cmap="viridis",
    elev=22,
    azim=-50,
    font_family="Arial",
    label_fontsize=20,
    tick_fontsize=18,
    cbar_fontsize=20,
    contour_on_floor=True,
    cbar_pad=0.01,
    cbar_width=0.025,
    cbar_shrink=0.5,
    show_title: bool = True,
    show_axis_labels: bool = True,
    show_tick_labels: bool = True,
    dpi: int = 600,
):
    """
    论文风格 3D surface:
      - 白底黑字
      - 浅灰网格
      - 可选标题 title；x/y/z 轴标签由 xlabel/ylabel/zlabel 控制（None 则该轴标题为空）
      - 默认显示 x/y/z 刻度数值
      - 底部 contour 投影

    ``show_title`` / ``show_axis_labels`` / ``show_tick_labels``：设为 False 可去掉标题、轴标题、刻度数字
    （仍保留刻度线与网格，便于版式留白）。
    """
    if plt is None:
        raise ImportError("matplotlib 不可用，无法绘图；请先安装 matplotlib。")

    threshold_list = np.asarray(threshold_list, dtype=float)
    max_iter_list = np.asarray(max_iter_list, dtype=float)
    Z = np.asarray(Z, dtype=float)

    T, M = np.meshgrid(threshold_list, max_iter_list)

    # -----------------------------
    # 全局论文风格
    # -----------------------------
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "Helvetica", "Arial", "DejaVu Sans"],
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.edgecolor": "black",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(10, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    # plt.subplots_adjust(left=0.08, right=0.80, bottom=0.08, top=0.95)
    # -----------------------------
    # colormap / norm / 光照
    # -----------------------------
    cmap_obj = mpl.colormaps[cmap]
    norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))

    ls = LightSource(azdeg=315, altdeg=42)
    # hillshade 内部 np.gradient 要求各维长度 ≥2；1×1 / 1×N 扫参会触发 ValueError
    try:
        rgb = ls.shade(
            Z,
            cmap=cmap_obj,
            norm=norm,
            vert_exag=0.6,
            blend_mode="soft",
        )
    except ValueError:
        rgba = np.asarray(cmap_obj(norm(Z)))
        rgb = rgba[..., :3]

    _grid_ok_for_contour = Z.shape[0] >= 2 and Z.shape[1] >= 2

    # -----------------------------
    # 主 surface
    # -----------------------------
    surf = ax.plot_surface(
        T,
        M,
        Z,
        facecolors=rgb,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )

    # -----------------------------
    # 底部 contour 投影
    # -----------------------------
    zmin = float(np.nanmin(Z))
    zmax = float(np.nanmax(Z))
    zrng = zmax - zmin
    if zrng < 1e-12:
        zrng = 1e-12

    floor_offset = zmin - 0.12 * zrng

    if contour_on_floor and _grid_ok_for_contour:
        ax.contourf(
            T, M, Z,
            zdir="z",
            offset=floor_offset,
            levels=16,
            cmap=cmap_obj,
            alpha=0.95,
        )
        ax.contour(
            T, M, Z,
            zdir="z",
            offset=floor_offset,
            levels=8,
            colors="0.45",
            linewidths=0.6,
        )

    # -----------------------------
    # colorbar：位置由 cbar_pad / cbar_width / cbar_shrink 控制
    # -----------------------------
    # 替换原来的 colorbar 相关代码块

    mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(Z)
    # cbar = fig.colorbar(
    #     mappable,
    #     ax=ax,
    #     shrink=0.4, # 改变高度
    #     aspect=20,
    #     pad=0.15,
    #     anchor=(0.0, 0.5),    # colorbar 自身锚点，不影响 axes 缩放
    #     panchor=False, # 禁止colorbar影响 parent axes
    # )
    # # cbar.set_label(zlabel, fontsize=cbar_fontsize, color="black")
    # cbar.ax.tick_params(labelsize=tick_fontsize, colors="black")
    # cbar.outline.set_edgecolor("black")
    
    # -----------------------------
    # 坐标轴标签与标题
    # -----------------------------
    xl = "" if xlabel is None else str(xlabel)
    yl = "" if ylabel is None else str(ylabel)
    zl = "" if zlabel is None else str(zlabel)
    if show_axis_labels:
        ax.set_xlabel(xl, fontsize=label_fontsize, color="black", labelpad=12)
        ax.set_ylabel(yl, fontsize=label_fontsize, color="black", labelpad=12)
        ax.set_zlabel(zl, fontsize=label_fontsize, color="black", labelpad=14)
    else:
        ax.set_xlabel("", fontsize=1, labelpad=0)
        ax.set_ylabel("", fontsize=1, labelpad=0)
        ax.set_zlabel("", fontsize=1, labelpad=0)
        try:
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
            ax.zaxis.label.set_visible(False)
        except Exception:
            pass
    if show_title and title:
        ax.set_title(str(title), fontsize=label_fontsize - 2, color="black", pad=10)
    else:
        ax.set_title("")

    # -----------------------------
    # 视角
    # -----------------------------
    ax.view_init(elev=elev, azim=azim)

    # -----------------------------
    # 轴范围
    # -----------------------------
    ax.set_zlim(floor_offset, zmax + 0.03 * zrng)
    zticks = ax.get_zticks()
    if show_tick_labels:
        ztick_labels = []
        for zt in zticks:
            if not np.isfinite(zt):
                ztick_labels.append("")
            else:
                ztick_labels.append(f"{float(zt):.3f}")
    else:
        ztick_labels = [""] * len(zticks)
    ax.set_zticks(zticks)
    ax.set_zticklabels(ztick_labels, fontsize=tick_fontsize)

    # for i in range(len(max_iter_list)):
    #     for j in range(len(threshold_list)):
    #         ax.text(
    #             threshold_list[j],
    #             max_iter_list[i],
    #             Z[i, j] + 0.01 * (np.nanmax(Z) - np.nanmin(Z)),  # 稍微抬高一点，避免压在 surface 上
    #             f"{Z[i, j]:.2f}",
    #             fontsize=10,
    #             color="black",
    #             ha="center",
    #             va="bottom",
    #         )

    # -----------------------------
    # 刻度字体
    # -----------------------------
    ax.tick_params(axis="x", labelsize=tick_fontsize, colors="black", pad=2)
    ax.tick_params(axis="y", labelsize=tick_fontsize, colors="black", pad=2)
    ax.tick_params(axis="z", labelsize=tick_fontsize, colors="black", pad=6)

    ax.set_xticks(threshold_list)
    if show_tick_labels:
        ax.set_xticklabels(
            [f"{float(x):g}" for x in threshold_list], fontsize=tick_fontsize
        )
    else:
        ax.set_xticklabels([""] * len(threshold_list), fontsize=tick_fontsize)
    # 如果 max_iter 本身是整数，强制整数刻度更好看
    if np.allclose(max_iter_list, np.round(max_iter_list)):
        yticks = max_iter_list.astype(int)
        ax.set_yticks(yticks)
        if show_tick_labels:
            ax.set_yticklabels([str(int(y)) for y in yticks], fontsize=tick_fontsize)
        else:
            ax.set_yticklabels([""] * len(yticks), fontsize=tick_fontsize)
    else:
        ax.set_yticks(max_iter_list)
        if show_tick_labels:
            ax.set_yticklabels(
                [f"{float(y):g}" for y in max_iter_list], fontsize=tick_fontsize
            )
        else:
            ax.set_yticklabels(
                [""] * len(max_iter_list), fontsize=tick_fontsize
            )

    # -----------------------------
    # 淡化 3D 墙面和网格
    # -----------------------------
    # pane 纯白
    try:
        ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

        ax.xaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        ax.yaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        ax.zaxis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
    except Exception:
        pass

    # 网格线浅灰
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["grid"]["color"] = (0.86, 0.86, 0.86, 1.0)
            axis._axinfo["grid"]["linewidth"] = 0.7
            axis._axinfo["grid"]["linestyle"] = "-"
        except Exception:
            pass

    # 坐标轴线条不要太重
    try:
        ax.xaxis.line.set_color("black")
        ax.yaxis.line.set_color("black")
        ax.zaxis.line.set_color("black")
        ax.xaxis.line.set_linewidth(0.8)
        ax.yaxis.line.set_linewidth(0.8)
        ax.zaxis.line.set_linewidth(0.8)
    except Exception:
        pass

    # -----------------------------
    # 布局保存
    # -----------------------------
    plt.tight_layout()
    # pad_inches 在 tight 外留白，避免 z 轴标题被裁掉
    plt.savefig(save_path, dpi=int(dpi), facecolor="white", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def blockwise_knn_from_distances(
    dist: torch.Tensor,
    k_neighbors: int,
    init_threshold: float,
    max_iter: int,
    min_candidates_ratio: float,
    block_size: int,
    tol: float = 0.05,
):
    """
    仅在已知距离矩阵 dist 上做 blockwise coarse screening + topk fine screening。
    未收敛时阈值放大：thresholds = thresholds * (1 + tol)。
    dist: [B, K, N] （一般来自 vanilla 的 distances_matrix）
    返回：knn_idx [B,K,k], thresholds [B,K], iters_used [B,K], final_candidates [B,K]
    """
    assert dist.ndim == 3
    B, K, N = dist.shape
    device, dtype = dist.device, dist.dtype
    k_actual = min(int(k_neighbors), N)
    min_candidates = max(k_actual + 1, int(k_actual * float(min_candidates_ratio)))

    BK = B * K
    d = dist.reshape(BK, N)
    thresholds = torch.full((BK,), float(init_threshold), dtype=dtype, device=device)
    iters_used = torch.zeros((BK,), dtype=torch.int16, device=device)

    final_mask = None
    for i in range(int(max_iter)):
        iters_used[:] = i + 1
        cand = torch.zeros((BK, N), dtype=torch.bool, device=device)
        for start in range(0, N, int(block_size)):
            end = min(start + int(block_size), N)
            cand[:, start:end] |= d[:, start:end] < thresholds.unsqueeze(-1)
        final_mask = cand
        cnt = cand.sum(dim=1)
        done = cnt >= min_candidates
        if bool(done.all()):
            break
        thresholds = torch.where(done, thresholds, thresholds * (1 + tol))

    masked_d = d.masked_fill(~final_mask, float("inf"))
    _, knn_idx = torch.topk(masked_d, k=k_actual, dim=1, largest=False)
    final_cnt = final_mask.sum(dim=1)

    return (
        knn_idx.reshape(B, K, k_actual),
        thresholds.reshape(B, K),
        iters_used.reshape(B, K),
        final_cnt.reshape(B, K),
    )


def fast_knn_iou(
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """
    Vectorised IoU for (B, K, k) KNN index tensors (-1 = invalid).

    Uses one-hot scatter_add so no Python loops over B or K.
    ~100× faster than the Python-set knn_neighbor_set_iou for typical B/K.

    Memory: 2 × B×K×N floats (e.g. 64 MB for B=16, K=512, N=1024).
    """
    B, K, k = idx_a.shape
    BK = B * K
    device = idx_a.device

    a  = idx_a.view(BK, k)
    b  = idx_b.view(BK, k)
    va = (a >= 0).float()
    vb = (b >= 0).float()

    # scatter_add_ with 0.0 for invalid indices is safe: it adds nothing.
    oh_a = idx_a.new_zeros(BK, N, dtype=torch.float32)
    oh_b = idx_a.new_zeros(BK, N, dtype=torch.float32)
    oh_a.scatter_add_(1, a.clamp(min=0), va)
    oh_b.scatter_add_(1, b.clamp(min=0), vb)

    inter = (oh_a * oh_b).sum(-1)          # (BK,)
    union = va.sum(1) + vb.sum(1) - inter
    iou   = torch.where(union > 0, inter / union, torch.ones_like(inter))
    return iou.view(B, K)



@torch.no_grad()
def run_iou_analysis(args):
    device = torch.device(args.device)
    ds = ModelNet40Wrapper(partition="test", num_points=args.num_points, root=args.root)
    n_total   = len(ds)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=collate_points,
        pin_memory=(device.type == "cuda"),
    )

    k = args.k_neighbors
    K = min(args.fps_steps, args.num_points)
    N = args.num_points
    min_cand = int(np.ceil(args.min_candidates_ratio * k))

    iou_cf_v_list       = []
    iou_ad_v_list       = []
    iou_bw_v_list       = []
    radii_mse_cf_v_list = []
    radii_mse_ad_v_list = []
    t_v_total = t_ad_total = t_cf_total = 0.0
    t_bw_total = 0.0

    bw_block_sum = None  # np.ndarray[n_blocks]
    bw_block_cnt = 0     # number of (B*K) observations
    bw_final_candidates = []
    bw_iters_used = []

    n_pts_processed = 0

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    pbar = tqdm(loader, desc="KNN IoU & radii", ncols=100)
    for pts, _ in pbar:
        if args.max_samples > 0 and n_pts_processed >= args.max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if args.max_samples > 0 and n_pts_processed + B > args.max_samples:
            pts = pts[: args.max_samples - n_pts_processed]
            B = pts.size(0)
        n_pts_processed += B

        # ── Vanilla KNN ── also yields the (B,K,N) dist matrix reused below
        sync(); t0 = time.perf_counter()
        fps_idx, _, knn_vanilla, dist_vanilla, radii_vanilla = fps_with_knn_vanilla(
            pts, K, k_neighbors=k, seed="first"
        )
        sync(); t_v_total += time.perf_counter() - t0

        # ── Adaptive KNN ─────────────────────────────────────────────────
        sync(); t0 = time.perf_counter()
        _, _, knn_adaptive, _, radii_adaptive, _, _ = fps_with_knn_adaptive(
            pts, K, k_neighbors=k, init_threshold=args.init_threshold_adaptive, max_iter=args.max_iter_adaptive,
            tol=args.tol, seed="first"
        )
        sync(); t_ad_total += time.perf_counter() - t0

        # ── Coarse-Fine KNN ───────────────────────────────────────────────
        sync(); t0 = time.perf_counter()
        if fps_with_knn_coarse_fine_fused is not None:
            _, _, knn_cf, _, radii_cf, _ = fps_with_knn_coarse_fine_fused(
                pts, K, k_neighbors=k, init_threshold=args.init_threshold_coarse_fine, max_iter=args.max_iter_coarse_fine,
                min_candidates_ratio=args.min_candidates_ratio, seed="first"
            )
        else:
            _, _, knn_cf, _, radii_cf, _ = fps_with_knn_coarse_fine(
                pts, K, k_neighbors=k, init_threshold=args.init_threshold_coarse_fine, max_iter=args.max_iter_coarse_fine,
                min_candidates_ratio=args.min_candidates_ratio, seed="first"
            )
        sync(); t_cf_total += time.perf_counter() - t0

        # ── Blockwise Coarse-Fine（可选） ─────────────────────────────────
        knn_bw = dist_bw = thr_bw = None
        if args.run_blockwise:
            sync(); t0 = time.perf_counter()
            return_dm = bool(args.collect_block_stats)
            if args.collect_block_stats:
                _, _, knn_bw, dist_bw, _, thr_bw, it_used = fps_with_knn_coarse_fine_fused_blockwise_instrumented(
                    pts, K, k_neighbors=k, init_threshold=args.init_threshold_blockwise, max_iter=args.max_iter_blockwise,
                    min_candidates_ratio=args.min_candidates_ratio, seed="first",
                    return_distances_matrix=return_dm,
                    exclude_self=False,
                    block_size=int(args.block_size),
                )
                bw_iters_used.append(it_used.reshape(-1).cpu().numpy())
            else:
                _, _, knn_bw, dist_bw, _, thr_bw = fps_with_knn_coarse_fine_fused_blockwise(
                    pts, K, k_neighbors=k, init_threshold=args.init_threshold_blockwise, max_iter=args.max_iter_blockwise,
                    min_candidates_ratio=args.min_candidates_ratio, seed="first",
                    return_distances_matrix=return_dm,
                    exclude_self=False,
                    block_size=int(args.block_size),
                )
            sync(); t_bw_total += time.perf_counter() - t0

        # ── IoU (相对 Vanilla) ────────────────────────────────────────────
        # (B, K)；注意：这里比较的是邻居集合，而不是距离值
        iou_cf_v = fast_knn_iou(knn_cf, knn_vanilla, N).cpu().numpy()
        iou_ad_v = fast_knn_iou(knn_adaptive, knn_vanilla, N).cpu().numpy()
        iou_cf_v_list.append(iou_cf_v)
        iou_ad_v_list.append(iou_ad_v)
        if args.run_blockwise and knn_bw is not None:
            iou_bw_v = fast_knn_iou(knn_bw, knn_vanilla, N).cpu().numpy()
            iou_bw_v_list.append(iou_bw_v)

            if args.collect_block_stats and dist_bw is not None and thr_bw is not None:
                cand_mask = dist_bw < thr_bw.unsqueeze(-1)  # [B, K, N]
                bw_final_candidates.append(cand_mask.sum(dim=-1).reshape(-1).cpu().numpy())

                bs = int(args.block_size)
                n_blocks = int((N + bs - 1) // bs)
                if bw_block_sum is None:
                    bw_block_sum = np.zeros((n_blocks,), dtype=np.float64)

                # 统计：每个 block 的候选命中数（对所有 B*K 汇总）
                for bi in range(n_blocks):
                    s = bi * bs
                    e = min(s + bs, N)
                    bw_block_sum[bi] += float(cand_mask[:, :, s:e].sum().item())

                bw_block_cnt += int(cand_mask.shape[0] * cand_mask.shape[1])

        # ── Radii MSE ─────────────────────────────────────────────────────
        ok = (torch.isfinite(radii_vanilla)
              & torch.isfinite(radii_cf)
              & torch.isfinite(radii_adaptive))
        if ok.any():
            mse_cf_v = ((radii_cf       - radii_vanilla).float() ** 2).masked_select(ok).mean().item()
            mse_ad_v = ((radii_adaptive - radii_vanilla).float() ** 2).masked_select(ok).mean().item()
        else:
            mse_cf_v = mse_ad_v = float("nan")
        radii_mse_cf_v_list.append(mse_cf_v)
        radii_mse_ad_v_list.append(mse_ad_v)

    if not iou_cf_v_list:
        raise ValueError(
            "未处理到任何样本，无法计算 IoU。"
            f" dataset_len={n_total}, max_samples={args.max_samples}, "
            f"batch_size={args.batch_size}, root={args.root!r}"
        )

    iou_cf_v_all = np.concatenate(iou_cf_v_list, axis=0)
    iou_ad_v_all = np.concatenate(iou_ad_v_list, axis=0)
    iou_bw_v_all = np.concatenate(iou_bw_v_list, axis=0) if (args.run_blockwise and iou_bw_v_list) else None
    # Per-sample mean over K anchors
    iou_cf_v_ps = iou_cf_v_all.mean(axis=1)
    iou_ad_v_ps = iou_ad_v_all.mean(axis=1)
    iou_bw_v_ps = iou_bw_v_all.mean(axis=1) if iou_bw_v_all is not None else None

    results = {
        "n_samples": int(iou_cf_v_ps.size),
        "n_total_test": n_total,
        "fps_steps": K,
        "k_neighbors": k,
        "min_candidates_ratio": args.min_candidates_ratio,
        "tol_adaptive": args.tol,
        "iou_coarse_fine_vs_vanilla": {
            "mean": float(np.mean(iou_cf_v_ps)),
            "std":  float(np.std(iou_cf_v_ps)),
            "min":  float(np.min(iou_cf_v_ps)),
            "max":  float(np.max(iou_cf_v_ps)),
        },
        "iou_adaptive_vs_vanilla": {
            "mean": float(np.mean(iou_ad_v_ps)),
            "std":  float(np.std(iou_ad_v_ps)),
            "min":  float(np.min(iou_ad_v_ps)),
            "max":  float(np.max(iou_ad_v_ps)),
        },
        "iou_blockwise_vs_vanilla": None,
        "radii_mse_coarse_fine_vs_vanilla": float(np.nanmean(radii_mse_cf_v_list)),
        "radii_mse_adaptive_vs_vanilla":    float(np.nanmean(radii_mse_ad_v_list)),
        "timing_sec": {
            "vanilla":     t_v_total,
            "adaptive":    t_ad_total,
            "coarse_fine": t_cf_total,
            "blockwise":   t_bw_total if args.run_blockwise else 0.0,
        },
        "speedup_vs_vanilla": {
            "adaptive":    float(t_v_total / t_ad_total)  if t_ad_total > 0 else 0.0,
            "coarse_fine": float(t_v_total / t_cf_total)  if t_cf_total > 0 else 0.0,
            "blockwise":   float(t_v_total / t_bw_total) if (args.run_blockwise and t_bw_total > 0) else 0.0,
        },

    }
    if iou_bw_v_ps is not None:
        results["iou_blockwise_vs_vanilla"] = {
            "mean": float(np.mean(iou_bw_v_ps)),
            "std":  float(np.std(iou_bw_v_ps)),
            "min":  float(np.min(iou_bw_v_ps)),
            "max":  float(np.max(iou_bw_v_ps)),
        }

    if args.run_blockwise and args.collect_block_stats and bw_block_sum is not None and bw_block_cnt > 0:
        bw_final = np.concatenate(bw_final_candidates, axis=0) if bw_final_candidates else np.array([], dtype=np.int64)
        bw_block_mean = (bw_block_sum / float(bw_block_cnt)).tolist()
        it_all = np.concatenate(bw_iters_used, axis=0) if bw_iters_used else np.array([], dtype=np.int16)
        max_iter_bw = int(args.max_iter_blockwise)
        hist = {str(i): int((it_all == i).sum()) for i in range(1, max_iter_bw + 1)} if it_all.size else {}
        results["blockwise_stats"] = {
            "block_size": int(args.block_size),
            "n_blocks": int(len(bw_block_mean)),
            "avg_candidates_per_block": bw_block_mean,
            "iters_used": {
                "max_iter": int(max_iter_bw),
                "mean": float(it_all.mean()) if it_all.size else 0.0,
                "std":  float(it_all.std())  if it_all.size else 0.0,
                "min":  int(it_all.min())    if it_all.size else 0,
                "max":  int(it_all.max())    if it_all.size else 0,
                "pct_hit_max_iter": float((it_all == max_iter_bw).mean()) if it_all.size else 0.0,
                "hist": hist,
            },
            "final_candidates": {
                "mean": float(bw_final.mean()) if bw_final.size else 0.0,
                "std":  float(bw_final.std())  if bw_final.size else 0.0,
                "min":  int(bw_final.min())    if bw_final.size else 0,
                "max":  int(bw_final.max())    if bw_final.size else 0,
            },
            "masked_d_finite": {
                "mean": float(bw_final.mean()) if bw_final.size else 0.0,
                "std":  float(bw_final.std())  if bw_final.size else 0.0,
                "min":  int(bw_final.min())    if bw_final.size else 0,
                "max":  int(bw_final.max())    if bw_final.size else 0,
            },
        }
    return results


def run_inference_evaluation(args):
    """与 inference_and_kan_analysis 一致的推理评估，返回 eval_result dict。"""
    from inference_and_kan_analysis import (
        get_checkpoint_path,
        build_model_and_load_checkpoint,
        run_evaluation,
    )
    from data import ModelNet40
    from utils import cal_loss

    device = torch.device(args.device)
    checkpoint_path = get_checkpoint_path(args.checkpoint)
    net = build_model_and_load_checkpoint(checkpoint_path, device)
    test_dataset = ModelNet40(partition="test", num_points=args.num_points)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.inference_batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    eval_result = run_evaluation(net, test_loader, cal_loss, device)
    return eval_result


@torch.no_grad()
def run_blockwise_sweep_and_plot(args):
    """
    扫描 blockwise 的 init_threshold / max_iter，并按 tol 分组输出 heatmap。
    - IoU: blockwise vs vanilla (mean over samples)
    - Speedup: vanilla_time / blockwise_time
    """
    if plt is None:
        raise ImportError("matplotlib 不可用，无法 sweep 绘图；请先安装 matplotlib。")

    threshold_list = _parse_csv_floats(args.sweep_init_thresholds)
    max_iter_list = _parse_csv_ints(args.sweep_max_iters)
    tol_list = _parse_csv_floats(args.sweep_tols)

    # 为了速度：sweep 强制走 blockwise，且建议配合 max_samples
    base_tol = args.tol
    base_init_bw = args.init_threshold_blockwise
    base_max_bw = args.max_iter_blockwise

    # --- 预取固定的一批点云（避免 sweep 每个组合都跑 DataLoader） ---
    device = torch.device(args.device)
    ds = ModelNet40Wrapper(partition="test", num_points=args.num_points, root=args.root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=collate_points,
        pin_memory=(device.type == "cuda"),
    )

    pts_batches = []
    n = 0
    need = args.max_samples if args.max_samples > 0 else min(len(ds), args.batch_size)
    for pts, _ in loader:
        pts_batches.append(pts.to(device, dtype=torch.float32))
        n += pts.shape[0]
        if n >= need:
            break

    K = min(args.fps_steps, args.num_points)
    k = args.k_neighbors

    # vanilla 只算一次：拿 knn_vanilla + distances_matrix（用于后续 blockwise 或 adaptive 扫描）
    t_v_total = 0.0
    vanilla_knn = []
    vanilla_dist = []
    for pts in pts_batches:
        sync = (device.type == "cuda")
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _, knn_v, dist_v, _ = fps_with_knn_vanilla(pts, K, k_neighbors=k, seed="first")
        if sync:
            torch.cuda.synchronize()
        t_v_total += time.perf_counter() - t0
        vanilla_knn.append(knn_v)
        vanilla_dist.append(dist_v)

    sweep_records = []
    fixed_th = getattr(args, "sweep_fixed_threshold", None)

    if fixed_th is not None:
        # 固定 init_threshold，扫描 tol × max_iter（blockwise vs vanilla）；blockwise 不用 tol，故沿 tol 方向结果相同
        tol_list_ax = _parse_csv_floats(args.sweep_tols)
        max_iter_list_ax = _parse_csv_ints(args.sweep_max_iters)
        Z_iou = np.zeros((len(max_iter_list_ax), len(tol_list_ax)), dtype=np.float32)
        Z_spd = np.zeros((len(max_iter_list_ax), len(tol_list_ax)), dtype=np.float32)
        for yi, max_it in enumerate(max_iter_list_ax):
            for xi, tol_val in enumerate(tol_list_ax):
                iou_vals = []
                t_bw_total = 0.0
                for knn_v, dist_v in zip(vanilla_knn, vanilla_dist):
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    knn_bw, _thr, _it, _cnt = blockwise_knn_from_distances(
                        dist_v, k_neighbors=k, init_threshold=float(fixed_th), max_iter=int(max_it),
                        min_candidates_ratio=args.min_candidates_ratio, block_size=int(args.block_size),
                        tol=float(tol_val),
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_bw_total += time.perf_counter() - t0
                    iou_vals.append(fast_knn_iou(knn_bw, knn_v, args.num_points).mean().item())
                Z_iou[yi, xi] = float(np.mean(iou_vals)) if iou_vals else np.nan
                Z_spd[yi, xi] = float(t_v_total / t_bw_total) if t_bw_total > 0 else 0.0

        x_axis = tol_list_ax
        y_axis = max_iter_list_ax
        xlabel, ylabel = "Tolerance", "Maximum Iters"
        if args.sweep_plot in ("heatmap", "both"):
            plot_iou_heatmap(x_axis, y_axis, Z_iou, f"{args.sweep_out_prefix}_blockwise_iou_heatmap.png",
                             title=f"Blockwise vs Vanilla IoU (init_threshold={fixed_th})", cbar_label="IoU")
            plot_iou_heatmap(x_axis, y_axis, Z_spd, f"{args.sweep_out_prefix}_blockwise_speedup_heatmap.png",
                             title=f"Blockwise speedup (init_threshold={fixed_th})", cbar_label="Speedup (x)")
            print(f"[sweep] wrote {args.sweep_out_prefix}_blockwise_iou_heatmap.png, ..._speedup_heatmap.png")
        if args.sweep_plot in ("surface", "both"):
            plot_iou_surface(x_axis, y_axis, Z_iou, f"{args.sweep_out_prefix}_blockwise_iou_surface.png",
                             zlabel="IoU", xlabel=xlabel, ylabel=ylabel)
            plot_iou_surface(x_axis, y_axis, Z_spd, f"{args.sweep_out_prefix}_blockwise_speedup_surface.png",
                             zlabel="Speedup (x)", xlabel=xlabel, ylabel=ylabel)
            print(f"[sweep] wrote {args.sweep_out_prefix}_blockwise_iou_surface.png, ..._speedup_surface.png")
        sweep_records.append({
            "mode": "blockwise_tol_maxiter",
            "fixed_init_threshold": float(fixed_th),
            "tol_list": [float(x) for x in x_axis],
            "max_iter_list": [int(x) for x in y_axis],
            "Z_iou": Z_iou.tolist(),
            "Z_speedup": Z_spd.tolist(),
        })
    else:
        for tol in tol_list:
            Z_iou = np.zeros((len(max_iter_list), len(threshold_list)), dtype=np.float32)
            Z_spd = np.zeros((len(max_iter_list), len(threshold_list)), dtype=np.float32)

            for yi, max_it in enumerate(max_iter_list):
                for xi, th in enumerate(threshold_list):
                    # blockwise 阈值步长使用 tol：thresholds = thresholds * (1 + tol)
                    iou_vals = []
                    t_bw_total = 0.0
                    for knn_v, dist_v in zip(vanilla_knn, vanilla_dist):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        knn_bw, _thr, _it, _cnt = blockwise_knn_from_distances(
                            dist_v, k_neighbors=k, init_threshold=float(th), max_iter=int(max_it),
                            min_candidates_ratio=args.min_candidates_ratio, block_size=int(args.block_size),
                            tol=float(tol),
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t_bw_total += time.perf_counter() - t0
                        iou_vals.append(fast_knn_iou(knn_bw, knn_v, args.num_points).mean().item())

                    Z_iou[yi, xi] = float(np.mean(iou_vals)) if iou_vals else np.nan
                    Z_spd[yi, xi] = float(t_v_total / t_bw_total) if t_bw_total > 0 else 0.0

            if args.sweep_plot in ("heatmap", "both"):
                out_iou = f"{args.sweep_out_prefix}_iou_tol={tol:.3f}.png"
                out_spd = f"{args.sweep_out_prefix}_speedup_tol={tol:.3f}.png"
                plot_iou_heatmap(
                    threshold_list=threshold_list,
                    max_iter_list=max_iter_list,
                    Z=Z_iou,
                    save_path=out_iou,
                    title=f"Blockwise vs Vanilla IoU (tol={tol:.3f})",
                    cbar_label="IoU",
                )
                plot_iou_heatmap(
                    threshold_list=threshold_list,
                    max_iter_list=max_iter_list,
                    Z=Z_spd,
                    save_path=out_spd,
                    title=f"Blockwise speedup vs Vanilla (tol={tol:.3f})",
                    cbar_label="Speedup (x)",
                )
                print(f"[sweep] wrote {out_iou}")
                print(f"[sweep] wrote {out_spd}")
            else:
                out_iou = None
                out_spd = None

            if args.sweep_plot in ("surface", "both"):
                out_iou_s = f"{args.sweep_out_prefix}_iou_surface_tol={tol:.3f}.png"
                out_spd_s = f"{args.sweep_out_prefix}_speedup_surface_tol={tol:.3f}.png"
                plot_iou_surface(
                    threshold_list=threshold_list,
                    max_iter_list=max_iter_list,
                    Z=Z_iou,
                    save_path=out_iou_s,
                    zlabel="IoU",
                    xlabel="init_threshold",
                    ylabel="max_iter",
                    title=f"tol={tol:.3f} (fixed per figure)",
                )
                plot_iou_surface(
                    threshold_list=threshold_list,
                    max_iter_list=max_iter_list,
                    Z=Z_spd,
                    save_path=out_spd_s,
                    zlabel="Speedup (x)",
                    xlabel="init_threshold",
                    ylabel="max_iter",
                    title=f"tol={tol:.3f} (fixed per figure)",
                )
                print(f"[sweep] wrote {out_iou_s}")
                print(f"[sweep] wrote {out_spd_s}")
            else:
                out_iou_s = None
                out_spd_s = None

            sweep_records.append({
                "tol": float(tol),
                "threshold_list": [float(x) for x in threshold_list],
                "max_iter_list": [int(x) for x in max_iter_list],
                "Z_iou": Z_iou.tolist(),
                "Z_speedup": Z_spd.tolist(),
                "images": {
                    "heatmap_iou": out_iou,
                    "heatmap_speedup": out_spd,
                    "surface_iou": out_iou_s,
                    "surface_speedup": out_spd_s,
                },
            })

    out_json = f"{args.sweep_out_prefix}_sweep.json"
    payload = {
        "mode": "blockwise_tol_maxiter" if fixed_th is not None else "fast_cached_vanilla_distances",
        "dataset": {
            "root": args.root,
            "partition": "test",
            "num_points": int(args.num_points),
            "dataset_len": int(len(ds)),
            "max_samples": int(args.max_samples),
            "effective_samples": int(sum(int(p.shape[0]) for p in pts_batches)),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
        },
        "knn": {
            "fps_steps": int(K),
            "k_neighbors": int(k),
            "min_candidates_ratio": float(args.min_candidates_ratio),
            "block_size": int(args.block_size),
        },
        "sweep": {
            "sweep_plot": str(args.sweep_plot),
            "threshold_list": [float(x) for x in threshold_list],
            "max_iter_list": [int(x) for x in max_iter_list],
            "tol_list": [float(x) for x in tol_list],
        },
        "results_by_tol": sweep_records,
    }
    if fixed_th is not None:
        payload["sweep"]["sweep_fixed_threshold"] = float(fixed_th)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[sweep] wrote {out_json}")

    # 还原参数
    args.tol = base_tol
    args.init_threshold_blockwise = base_init_bw
    args.max_iter_blockwise = base_max_bw


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    do_inference = args.run_inference and not args.no_inference

    if args.sweep_blockwise:
        run_blockwise_sweep_and_plot(args)
        return

    # ---------- 1) IoU 与 radii 分析 ----------
    print("=" * 60)
    print("KNN 粗筛+细筛 与 Vanilla KNN IoU / radii 分析")
    print("=" * 60)
    t0 = time.perf_counter()
    iou_results = run_iou_analysis(args)
    t1 = time.perf_counter()
    print(f"  分析样本数: {iou_results['n_samples']} (测试集总数: {iou_results['n_total_test']})")
    print(f"  FPS 步数 K: {iou_results['fps_steps']}, 邻居数 k: {iou_results['k_neighbors']}")
    print(f"  粗筛+细筛 min_candidates_ratio: {iou_results['min_candidates_ratio']}")
    print("  --- IoU (相对 Vanilla) ---")
    print(f"  Coarse-Fine vs Vanilla: mean={iou_results['iou_coarse_fine_vs_vanilla']['mean']:.4f} "
          f"std={iou_results['iou_coarse_fine_vs_vanilla']['std']:.4f} "
          f"[{iou_results['iou_coarse_fine_vs_vanilla']['min']:.4f}, {iou_results['iou_coarse_fine_vs_vanilla']['max']:.4f}]")
    print(f"  Adaptive vs Vanilla:    mean={iou_results['iou_adaptive_vs_vanilla']['mean']:.4f} "
          f"std={iou_results['iou_adaptive_vs_vanilla']['std']:.4f} "
          f"[{iou_results['iou_adaptive_vs_vanilla']['min']:.4f}, {iou_results['iou_adaptive_vs_vanilla']['max']:.4f}]")
    if iou_results.get("iou_blockwise_vs_vanilla"):
        print(f"  Blockwise vs Vanilla:   mean={iou_results['iou_blockwise_vs_vanilla']['mean']:.4f} "
              f"std={iou_results['iou_blockwise_vs_vanilla']['std']:.4f} "
              f"[{iou_results['iou_blockwise_vs_vanilla']['min']:.4f}, {iou_results['iou_blockwise_vs_vanilla']['max']:.4f}]")
   
    print("  --- Radii MSE (相对 Vanilla) ---")
    print(f"  Coarse-Fine vs Vanilla: {iou_results['radii_mse_coarse_fine_vs_vanilla']:.6e}")
    print(f"  Adaptive vs Vanilla:    {iou_results['radii_mse_adaptive_vs_vanilla']:.6e}")

    print(f"  耗时: {t1 - t0:.2f}s")
    if iou_results.get("timing_sec") and iou_results.get("speedup_vs_vanilla"):
        ts = iou_results["timing_sec"]
        sp = iou_results["speedup_vs_vanilla"]
        if args.run_blockwise:
            print(f"  --- Timing (sec) --- vanilla={ts['vanilla']:.2f} adaptive={ts['adaptive']:.2f} "
                  f"coarse_fine={ts['coarse_fine']:.2f} blockwise={ts['blockwise']:.2f}")
            print(f"  --- Speedup vs vanilla --- adaptive={sp['adaptive']:.2f} coarse_fine={sp['coarse_fine']:.2f} "
                  f"blockwise={sp['blockwise']:.2f}")
        else:
            print(f"  --- Timing (sec) --- vanilla={ts['vanilla']:.2f} adaptive={ts['adaptive']:.2f} coarse_fine={ts['coarse_fine']:.2f}")
            print(f"  --- Speedup vs vanilla --- adaptive={sp['adaptive']:.2f} coarse_fine={sp['coarse_fine']:.2f}")

    if args.run_blockwise and args.collect_block_stats and iou_results.get("blockwise_stats"):
        bs = iou_results["blockwise_stats"]
        fc = bs["final_candidates"]
        it = bs.get("iters_used")
        print("  --- Blockwise 统计 ---")
        print(f"  block_size={bs['block_size']} n_blocks={bs['n_blocks']}")
        if it:
            print(f"  iters_used: mean={it['mean']:.2f} std={it['std']:.2f} min={it['min']} max={it['max']} "
                  f"hit_max_iter={it['pct_hit_max_iter']*100:.1f}%")
            print(f"  iters_used_hist(1..{it['max_iter']}): {it['hist']}")
        print(f"  final_candidates: mean={fc['mean']:.2f} std={fc['std']:.2f} min={fc['min']} max={fc['max']}")
        # 每个 block 的平均候选数过长，默认只打印前后若干个
        avg_blk = bs["avg_candidates_per_block"]
        head = avg_blk[: min(6, len(avg_blk))]
        tail = avg_blk[max(0, len(avg_blk) - 6):]
        if len(avg_blk) <= 12:
            print(f"  avg_candidates_per_block={avg_blk}")
        else:
            print(f"  avg_candidates_per_block(head)={head}")
            print(f"  avg_candidates_per_block(tail)={tail}")
    print("=" * 60)

    # ---------- 2) 可选：分类推理 ----------
    if do_inference:
        print("\n" + "=" * 60)
        print("ModelNet40 分类推理 (Vanilla 模型，与 inference_and_kan_analysis 一致)")
        print("=" * 60)
        try:
            eval_result = run_inference_evaluation(args)
            print(f"  测试样本数: {eval_result['num_samples']}")
            print(f"  Test Loss: {eval_result['loss']:.4f}")
            print(f"  Test Acc (accuracy): {eval_result['acc']:.2f}%")
            print(f"  Test Acc (balanced): {eval_result['acc_avg']:.2f}%")
            iou_results["inference"] = {
                "num_samples": eval_result["num_samples"],
                "loss": eval_result["loss"],
                "acc": eval_result["acc"],
                "acc_avg": eval_result["acc_avg"],
            }
        except FileNotFoundError as e:
            print(f"  推理跳过（未找到检查点或数据）: {e}")
            iou_results["inference"] = {"error": str(e)}
        except Exception as e:
            print(f"  推理异常: {e}")
            iou_results["inference"] = {"error": str(e)}
        print("=" * 60)

    # ---------- 3) 写出 JSON ----------
    out_path = args.out_json.strip()
    if not out_path:
        out_path = os.path.join(SCRIPT_DIR, "knn_coarse_fine_iou_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(iou_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
