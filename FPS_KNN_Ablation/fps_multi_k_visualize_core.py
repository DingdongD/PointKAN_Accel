# -*- coding: utf-8 -*-
"""三次 FPS 3D 叠加可视化的共用逻辑（S3DIS / KITTI）。"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Set, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def _get_cmap(name: str):
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return matplotlib.cm.get_cmap(name)

from kitti_fps_exp import (
    farthest_point_sampling,
    find_bin_files,
    load_kitti_bin,
    roi_filter,
)

FIG_FONTSIZE = 25


def _try_register_liberation_sans() -> None:
    """若系统已装 Liberation（与 Arial 度量接近），注册到 matplotlib 字体表。"""
    patterns = (
        "/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf",
        "/usr/share/fonts/truetype/ttf-liberation/LiberationSans-*.ttf",
    )
    from matplotlib import font_manager

    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            try:
                font_manager.fontManager.addfont(path)
            except (OSError, ValueError, RuntimeError):
                pass


def _resolve_fig_font_family() -> str:
    """
    优先 Arial；无则 Liberation Sans / Nimbus / Helvetica；再退 DejaVu Sans。
    仅使用 matplotlib 已登记字体名，避免 findfont 对缺失 Arial 反复告警。
    """
    _try_register_liberation_sans()
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in (
        "Arial",
        "Liberation Sans",
        "Nimbus Sans",
        "Nimbus Sans L",
        "Helvetica",
        "DejaVu Sans",
    ):
        if fam in available:
            return fam
    return "sans-serif"


FIG_FONT_FAMILY = _resolve_fig_font_family()
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    FIG_FONT_FAMILY,
    "Liberation Sans",
    "DejaVu Sans",
    "Helvetica",
    "sans-serif",
]


def _set_3d_axis_labels_and_ticks(
    ax,
    *,
    show_tick_labels: bool = True,
    show_axis_labels: bool = True,
) -> None:
    """坐标轴名与刻度；可单独关闭轴标题或刻度数字。"""
    if show_axis_labels:
        ax.set_xlabel("x", fontsize=FIG_FONTSIZE, fontfamily=FIG_FONT_FAMILY)
        ax.set_ylabel("y", fontsize=FIG_FONTSIZE, fontfamily=FIG_FONT_FAMILY)
        ax.set_zlabel("z", fontsize=FIG_FONTSIZE, fontfamily=FIG_FONT_FAMILY)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ax.zaxis.label.set_visible(False)
    if not show_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.tick_params(axis="x", labelsize=0, pad=-2)
        ax.tick_params(axis="y", labelsize=0, pad=-2)
        ax.tick_params(axis="z", labelsize=0, pad=-2)
        return
    ax.tick_params(axis="x", labelsize=FIG_FONTSIZE)
    ax.tick_params(axis="y", labelsize=FIG_FONTSIZE)
    ax.tick_params(axis="z", labelsize=FIG_FONTSIZE)
    for labels in (ax.get_xticklabels(), ax.get_yticklabels(), ax.get_zticklabels()):
        for t in labels:
            t.set_fontfamily(FIG_FONT_FAMILY)


def subsample_xyz(xyz: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    n = xyz.shape[0]
    if max_points <= 0 or n <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx]


def _as_set(idx: np.ndarray) -> Set[int]:
    return set(int(x) for x in idx.tolist())


def overlap_report(
    idx_s: np.ndarray,
    idx_m: np.ndarray,
    idx_l: np.ndarray,
    k_s: int,
    k_m: int,
    k_l: int,
) -> Tuple[str, Dict[str, int]]:
    s_s, s_m, s_l = _as_set(idx_s), _as_set(idx_m), _as_set(idx_l)
    lines = [
        f"|S_{k_s}|={len(s_s)}, |S_{k_m}|={len(s_m)}, |S_{k_l}|={len(s_l)}",
        f"S_{k_s} ⊆ S_{k_m}: {s_s <= s_m}  |  S_{k_m} ⊆ S_{k_l}: {s_m <= s_l}",
        f"|S_{k_s} ∩ S_{k_m}|={len(s_s & s_m)}  |  |S_{k_s} ∩ S_{k_l}|={len(s_s & s_l)}  |  |S_{k_m} ∩ S_{k_l}|={len(s_m & s_l)}",
    ]
    stats = {
        f"n_{k_s}": len(s_s),
        f"n_{k_m}": len(s_m),
        f"n_{k_l}": len(s_l),
        f"subset_{k_s}_in_{k_m}": int(s_s <= s_m),
        f"subset_{k_m}_in_{k_l}": int(s_m <= s_l),
    }
    return "\n".join(lines), stats


def _kde_pdf_1d(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """一维 KDE 概率密度；方差过小或 scipy 不可用时退回归一化直方图。"""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2 or float(np.ptp(v)) < 1e-12:
        return np.zeros_like(x_grid, dtype=np.float64)
    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(v)
        return np.asarray(kde(x_grid), dtype=np.float64)
    except Exception:
        counts, edges = np.histogram(v, bins=min(50, max(10, v.size // 5)), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.interp(x_grid, centers, counts, left=0.0, right=0.0)


def plot_fps_marginal_pdf_compare(
    xyz: np.ndarray,
    idx_small: np.ndarray,
    idx_mid: np.ndarray,
    idx_large: np.ndarray,
    ks: Tuple[int, int, int],
    colors: Tuple[str, str, str],
    save_path: str,
    *,
    show_full_reference: bool = True,
    full_ref_max_points: int = 12000,
    grid_points: int = 512,
    fill_alpha: float = 0.18,
) -> None:
    """
    三种 FPS 采样点在 x / y / z 上的边际 PDF（KDE）叠在同一张图的三个子图中。
    colors 顺序对应 (K_small, K_mid, K_large)，与三色叠加图一致。
    """
    k_s, k_m, k_l = ks
    sets = [
        (xyz[idx_small], f"FPS K={k_s}", colors[0]),
        (xyz[idx_mid], f"FPS K={k_m}", colors[1]),
        (xyz[idx_large], f"FPS K={k_l}", colors[2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    fig.patch.set_alpha(0.0)

    for dim, ax in zip((0, 1, 2), axes):
        vmin = min(float(np.min(pts[:, dim])) for pts, _, _ in sets)
        vmax = max(float(np.max(pts[:, dim])) for pts, _, _ in sets)
        pad = 0.02 * (vmax - vmin + 1e-9)
        lo, hi = vmin - pad, vmax + pad
        xs = np.linspace(lo, hi, int(grid_points))

        if show_full_reference and xyz.shape[0] > 0:
            n = xyz.shape[0]
            if n > full_ref_max_points:
                rng = np.random.default_rng(0)
                ref_idx = rng.choice(n, size=full_ref_max_points, replace=False)
                ref_v = xyz[ref_idx, dim]
            else:
                ref_v = xyz[:, dim]
            ref_pdf = _kde_pdf_1d(ref_v, xs)
            ax.fill_between(xs, ref_pdf, color="0.75", alpha=0.25, linewidth=0)
            ax.plot(
                xs,
                ref_pdf,
                color="0.45",
                linewidth=1.4,
                linestyle="--",
                zorder=1,
            )

        for pts, _lab, c in sets:
            pdf = _kde_pdf_1d(pts[:, dim], xs)
            ax.fill_between(xs, pdf, color=c, alpha=fill_alpha, linewidth=0, zorder=2)
            ax.plot(xs, pdf, color=c, linewidth=2.2, zorder=3)

        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.patch.set_alpha(0.0)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.42, zorder=0)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.28, zorder=0)

        for s in ax.spines.values():
            s.set_visible(False)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.add_patch(
            FancyArrowPatch(
                (xmin, ymin),
                (xmax, ymin),
                transform=ax.transData,
                arrowstyle="-|>",
                mutation_scale=16,
                color="black",
                linewidth=1.35,
                clip_on=False,
                zorder=20,
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (xmin, ymin),
                (xmin, ymax),
                transform=ax.transData,
                arrowstyle="-|>",
                mutation_scale=16,
                color="black",
                linewidth=1.35,
                clip_on=False,
                zorder=20,
            )
        )

        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            direction="out",
            length=7,
            width=1.15,
            colors="black",
            labelbottom=False,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            bottom=True,
            direction="out",
            length=4,
            width=0.85,
            colors="black",
            labelbottom=False,
        )
        ax.tick_params(
            axis="y",
            which="major",
            left=True,
            right=False,
            direction="out",
            length=7,
            width=1.15,
            colors="black",
            labelleft=False,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            left=True,
            direction="out",
            length=4,
            width=0.85,
            colors="black",
            labelleft=False,
        )

    plt.tight_layout(pad=0.4, w_pad=0.35)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
        facecolor="none",
        edgecolor="none",
    )
    plt.close()


def _set_equal_3d_aspect(ax, xyz_min: np.ndarray, xyz_max: np.ndarray) -> None:
    ctr = (xyz_min + xyz_max) / 2.0
    span = float(np.max(xyz_max - xyz_min)) or 1.0
    half = span / 2.0
    ax.set_xlim(ctr[0] - half, ctr[0] + half)
    ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_zlim(ctr[2] - half, ctr[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_fps_three_runs_3d(
    xyz: np.ndarray,
    idx_large: np.ndarray,
    idx_mid: np.ndarray,
    idx_small: np.ndarray,
    save_path: str,
    title_prefix: str,
    color_a: str,
    color_b: str,
    color_c: str,
    start_idx: int,
    bg_subsample: int = 12000,
    bg_alpha: float = 0.06,
    elev: float = 22.0,
    azim: float = -58.0,
) -> str:
    """先画最大 K（颜色 A），再中 K（B），最后最小 K（C）。"""
    p_l = xyz[idx_large]
    p_m = xyz[idx_mid]
    p_s = xyz[idx_small]

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    if bg_subsample > 0 and xyz.shape[0] > bg_subsample:
        rng = np.random.default_rng(0)
        bg = xyz[rng.choice(xyz.shape[0], bg_subsample, replace=False)]
    else:
        bg = xyz

    ax.scatter(
        bg[:, 0], bg[:, 1], bg[:, 2],
        s=0.15, c="0.55", alpha=bg_alpha, linewidths=0,
        depthshade=False,
    )

    ax.scatter(
        p_l[:, 0], p_l[:, 1], p_l[:, 2],
        s=6, c=color_a, alpha=0.38, linewidths=0, depthshade=True,
    )
    ax.scatter(
        p_m[:, 0], p_m[:, 1], p_m[:, 2],
        s=10, c=color_b, alpha=0.52, linewidths=0, depthshade=True,
    )
    ax.scatter(
        p_s[:, 0], p_s[:, 1], p_s[:, 2],
        s=16, c=color_c, alpha=0.85, edgecolors="0.2", linewidths=0.2,
        depthshade=True,
    )

    anchor = xyz[int(np.clip(start_idx, 0, xyz.shape[0] - 1))]
    ax.scatter(
        [anchor[0]], [anchor[1]], [anchor[2]],
        s=200, c="gold", marker="*", edgecolors="black", linewidths=0.6,
        depthshade=False, zorder=20,
    )

    rep, _ = overlap_report(
        idx_small, idx_mid, idx_large,
        len(idx_small), len(idx_mid), len(idx_large),
    )

    _set_3d_axis_labels_and_ticks(ax)
    ax.view_init(elev=elev, azim=azim)

    all_pts = np.vstack([p_l, p_m, p_s, anchor[None, :]])
    mn = np.min(all_pts, axis=0)
    mx = np.max(all_pts, axis=0)
    pad = 0.02 * (mx - mn + 1e-6)
    _set_equal_3d_aspect(ax, mn - pad, mx + pad)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    return rep


def _add_fps_polyline_gradient(
    ax,
    path: np.ndarray,
    cmap_name: str = "plasma",
    linewidth: float = 1.15,
    alpha: float = 0.9,
) -> None:
    """按 FPS 顺序将相邻采样点连成折线，线段颜色沿步数渐变（早→晚）。"""
    k = path.shape[0]
    if k < 2:
        return
    segs = np.stack([path[:-1], path[1:]], axis=1)
    cmap = _get_cmap(cmap_name)
    colors = cmap(np.linspace(0.15, 0.95, segs.shape[0]))
    lc = Line3DCollection(
        segs,
        colors=colors,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection3d(lc)


def _style_3d_axes(ax) -> None:
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.85")
    ax.yaxis.pane.set_edgecolor("0.85")
    ax.zaxis.pane.set_edgecolor("0.85")
    ax.grid(True, alpha=0.35)


def _milestone_indices(k: int, max_markers: int) -> np.ndarray:
    """包含 0 与 k-1，沿路径均匀取少量里程碑索引。"""
    if k <= 0:
        return np.array([], dtype=np.int64)
    if k == 1:
        return np.array([0], dtype=np.int64)
    n = max(2, min(max_markers, k))
    idx = np.linspace(0, k - 1, num=n)
    return np.unique(np.round(idx).astype(np.int64))


def _add_fps_direction_arrows(
    ax,
    path: np.ndarray,
    stride: int,
    *,
    length_scale: float = 0.04,
    color: str = "0.2",
    alpha: float = 0.55,
) -> None:
    """在 path[i]→path[i+1] 方向画短箭头（长度按场景尺度归一，便于读方向）。"""
    k = path.shape[0]
    if k < 2 or stride <= 0:
        return
    span = float(np.max(np.ptp(path, axis=0)))
    L = max(span * length_scale, 0.25)
    i0 = np.arange(0, k - 1, stride, dtype=np.int64)
    if i0.size == 0 or int(i0[-1]) != k - 2:
        i0 = np.unique(np.append(i0, k - 2))
    U = path[i0 + 1] - path[i0]
    ax.quiver(
        path[i0, 0],
        path[i0, 1],
        path[i0, 2],
        U[:, 0],
        U[:, 1],
        U[:, 2],
        length=L,
        normalize=True,
        color=color,
        alpha=alpha,
        arrow_length_ratio=0.32,
        linewidth=0.8,
    )


def plot_fps_trajectories_row(
    xyz: np.ndarray,
    trajectories: Tuple[Tuple[np.ndarray, str, str], ...],
    save_path: str,
    title_prefix: str,
    start_idx: int,
    bg_subsample: int = 0,
    bg_alpha: float = 0.04,
    elev: float = 22.0,
    azim: float = -58.0,
    *,
    milestone_max: int = 14,
    arrow_stride: int = -1,
    line_width: float = 2.4,
    dense_scatter_stride: int = 0,
    use_shared_colorbar: bool = True,
) -> None:
    """
    一行多子图：每个子图一种 K 的 FPS 空间轨迹。

    默认「清爽」模式：无背景点云（bg_subsample=0）、粗渐变折线、稀疏方向箭头、
    少量里程碑散点；dense_scatter_stride>0 时恢复沿路径密散点（旧风格）。
    arrow_stride: -1 表示按 K 自动；-2 表示不画箭头。
    """
    n_plots = len(trajectories)
    fig = plt.figure(figsize=(6.0 * n_plots + 0.8, 5.85))
    fig.patch.set_facecolor("#fafafa")
    anchor = xyz[int(np.clip(start_idx, 0, xyz.shape[0] - 1))]

    bg = None
    if bg_subsample > 0 and xyz.shape[0] > 0:
        if xyz.shape[0] > bg_subsample:
            rng = np.random.default_rng(0)
            bg = xyz[rng.choice(xyz.shape[0], bg_subsample, replace=False)]
        else:
            bg = xyz

    all_pts_list = [anchor]
    axes_row: list = []
    scatter_for_cbar = None

    for j, (sampled_idx, _, cmap_name) in enumerate(trajectories):
        ax = fig.add_subplot(1, n_plots, j + 1, projection="3d")
        axes_row.append(ax)
        _style_3d_axes(ax)

        if bg is not None:
            ax.scatter(
                bg[:, 0], bg[:, 1], bg[:, 2],
                s=0.08, c="#9aa0a6", alpha=bg_alpha, linewidths=0, depthshade=False,
            )

        path = xyz[sampled_idx]
        k = path.shape[0]
        _add_fps_polyline_gradient(
            ax, path, cmap_name=cmap_name, linewidth=line_width, alpha=0.92
        )

        if arrow_stride == -2:
            stride_ar = 0
        elif arrow_stride < 0:
            stride_ar = max(10, k // 18)
        else:
            stride_ar = arrow_stride
        _add_fps_direction_arrows(ax, path, stride_ar)

        if dense_scatter_stride > 0:
            step = max(1, int(dense_scatter_stride))
            sel = np.arange(0, k, step)
            pts = path[sel]
            t_norm = sel.astype(np.float64) / max(k - 1, 1)
            sc = ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=9.0,
                c=t_norm,
                cmap=cmap_name,
                alpha=0.72,
                edgecolors="0.25",
                linewidths=0.12,
                depthshade=True,
            )
            scatter_for_cbar = sc
        else:
            mil = _milestone_indices(k, milestone_max)
            pts_m = path[mil]
            t_m = mil.astype(np.float64) / max(k - 1, 1)
            sc = ax.scatter(
                pts_m[:, 0], pts_m[:, 1], pts_m[:, 2],
                s=55,
                c=t_m,
                cmap=cmap_name,
                alpha=0.95,
                edgecolors="0.15",
                linewidths=0.35,
                depthshade=True,
                zorder=5,
            )
            scatter_for_cbar = sc

        ax.scatter(
            [path[-1, 0]], [path[-1, 1]], [path[-1, 2]],
            s=120, c="#2ca02c", marker="D", edgecolors="0.15", linewidths=0.4,
            depthshade=False, zorder=8,
        )
        ax.scatter(
            [anchor[0]], [anchor[1]], [anchor[2]],
            s=200, c="#f5c518", marker="*", edgecolors="0.1", linewidths=0.55,
            depthshade=False, zorder=9,
        )

        ax.view_init(elev=elev, azim=azim)
        all_pts_list.append(path)

    all_pts = np.vstack(all_pts_list)
    mn = np.min(all_pts, axis=0)
    mx = np.max(all_pts, axis=0)
    pad = 0.02 * (mx - mn + 1e-6)
    for ax in axes_row:
        _set_equal_3d_aspect(ax, mn - pad, mx + pad)
    for ax in axes_row:
        _set_3d_axis_labels_and_ticks(
            ax, show_tick_labels=False, show_axis_labels=False
        )

    if use_shared_colorbar and scatter_for_cbar is not None and n_plots > 0:
        # 压缩子图间距；colorbar 略左移以适配更窄的留白
        fig.subplots_adjust(left=0.02, right=0.91, top=0.96, bottom=0.06, wspace=0.02)
        cax = fig.add_axes([0.925, 0.22, 0.014, 0.55])
        cb = fig.colorbar(scatter_for_cbar, cax=cax)
        cb.ax.set_yticklabels([])
        cb.ax.tick_params(axis="y", length=0, labelleft=False, labelright=False)
    else:
        plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="S3DIS 或 KITTI：同一起点三次 FPS，3D 三色叠加看重叠",
    )
    p.add_argument(
        "--dataset",
        type=str,
        choices=["s3dis", "kitti"],
        default="s3dis",
        help="数据源：s3dis 为 .npy；kitti 为 .bin（默认可做 ROI）",
    )
    p.add_argument("--data_dir", type=str, default=None, help="点云目录（默认随 dataset）")
    p.add_argument(
        "--cloud_path",
        type=str,
        default=None,
        help="指定单个点云文件（.npy 或 .bin）",
    )
    p.add_argument("--file_index", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--max_points", type=int, default=-1)
    p.add_argument("--subsample_seed", type=int, default=0)
    p.add_argument("--k_small", type=int, default=None)
    p.add_argument("--k_mid", type=int, default=None)
    p.add_argument("--k_large", type=int, default=None)
    p.add_argument("--color_a", type=str, default="#e41a1c", help="最大 K（颜色 A）")
    p.add_argument("--color_b", type=str, default="#4daf4a")
    p.add_argument("--color_c", type=str, default="#377eb8")
    p.add_argument("--bg_subsample", type=int, default=12000)
    p.add_argument("--elev", type=float, default=22.0)
    p.add_argument("--azim", type=float, default=-58.0)
    p.add_argument(
        "--no_roi",
        action="store_true",
        help="KITTI 时关闭 ROI 过滤",
    )
    p.add_argument(
        "--no_trajectory",
        action="store_true",
        help="不保存 FPS 顺序空间轨迹图（默认可视化开启）",
    )
    p.add_argument(
        "--trajectory_point_stride",
        type=int,
        default=0,
        help=">0 时沿路径画密散点（步长=该值，旧风格）；0 则仅用少量里程碑+折线+箭头",
    )
    p.add_argument(
        "--trajectory_bg_points",
        type=int,
        default=0,
        help="轨迹图里背景原始点数量；0=不画背景（推荐），需要上下文时可设 3000~8000",
    )
    p.add_argument(
        "--trajectory_milestone_max",
        type=int,
        default=14,
        help="清爽模式下沿路径最多几个里程碑点（含首尾）",
    )
    p.add_argument(
        "--trajectory_arrow_stride",
        type=int,
        default=-1,
        help="方向箭头间隔：-1 自动；-2 关闭；正整数为固定步长",
    )
    p.add_argument(
        "--trajectory_line_width",
        type=float,
        default=2.4,
        help="轨迹折线线宽（3D 线段集合）",
    )
    p.add_argument(
        "--no_marginal_pdf",
        action="store_true",
        help="不保存三种 FPS 的 x/y/z 边际 PDF（KDE）对比图",
    )
    p.add_argument(
        "--marginal_no_full_ref",
        action="store_true",
        help="边际 PDF 图中不叠加「全点云」参考曲线",
    )
    # 兼容旧参数名
    p.add_argument("--npy_path", type=str, default=None, help="同 --cloud_path（S3DIS）")
    return p


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "kitti":
        if args.data_dir is None:
            args.data_dir = "/workspace/VoxelNet/KITTI_Mini/training/velodyne"
        if args.out_dir is None:
            args.out_dir = "./kitti_fps_multi_k_outputs"
        if args.k_small is None:
            args.k_small = 512
        if args.k_mid is None:
            args.k_mid = 1024
        if args.k_large is None:
            args.k_large = 2048
    else:
        if args.data_dir is None:
            args.data_dir = "/workspace/VoxelNet/pc_data/s3dis_trainval"
        if args.out_dir is None:
            args.out_dir = "./s3dis_fps_multi_k_outputs"
        if args.k_small is None:
            args.k_small = 1024
        if args.k_mid is None:
            args.k_mid = 2048
        if args.k_large is None:
            args.k_large = 4096


def run(args: argparse.Namespace) -> None:
    from s3dis_fps_exp import find_npy_files, load_s3dis_npy

    apply_dataset_defaults(args)

    ks = (args.k_small, args.k_mid, args.k_large)
    if not (ks[0] < ks[1] < ks[2]):
        raise SystemExit("要求 k_small < k_mid < k_large")

    cloud_path = args.cloud_path or args.npy_path
    if cloud_path:
        path = cloud_path
        if not os.path.isfile(path):
            raise SystemExit(f"文件不存在: {path}")
    else:
        if args.dataset == "s3dis":
            files = find_npy_files(args.data_dir)
        else:
            files = find_bin_files(args.data_dir)
        if not files:
            raise SystemExit(f"目录下无点云文件: {args.data_dir}")
        if args.file_index < 0 or args.file_index >= len(files):
            raise SystemExit(f"file_index 越界 [0,{len(files)-1}]")
        path = files[args.file_index]

    if args.dataset == "s3dis":
        xyz_full = load_s3dis_npy(path)
    else:
        pts = load_kitti_bin(path)
        if not args.no_roi:
            pts = roi_filter(pts)
        xyz_full = pts[:, :3].astype(np.float32)

    n_full = xyz_full.shape[0]
    xyz = subsample_xyz(xyz_full, args.max_points, args.subsample_seed)
    n = xyz.shape[0]
    if n < ks[2]:
        raise SystemExit(f"点数 n={n} 小于 k_large={ks[2]}")

    si = int(np.clip(args.start_idx, 0, n - 1))

    idx_s, _, _, _ = farthest_point_sampling(xyz, k=ks[0], start_idx=si)
    idx_m, _, _, _ = farthest_point_sampling(xyz, k=ks[1], start_idx=si)
    idx_l, _, _, _ = farthest_point_sampling(xyz, k=ks[2], start_idx=si)

    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith(".pcd"):
        stem = os.path.splitext(stem)[0]

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.dataset}_{stem}"
    out_png = os.path.join(
        out_dir,
        f"{tag}_fps3d_{ks[0]}_{ks[1]}_{ks[2]}_start{si}.png",
    )

    rep = plot_fps_three_runs_3d(
        xyz, idx_l, idx_m, idx_s, out_png,
        title_prefix=f"{args.dataset.upper()} {stem}",
        color_a=args.color_a,
        color_b=args.color_b,
        color_c=args.color_c,
        start_idx=si,
        bg_subsample=args.bg_subsample,
        elev=args.elev,
        azim=args.azim,
    )

    np.save(os.path.join(out_dir, f"{tag}_fps_idx_k{ks[0]}_start{si}.npy"), idx_s)
    np.save(os.path.join(out_dir, f"{tag}_fps_idx_k{ks[1]}_start{si}.npy"), idx_m)
    np.save(os.path.join(out_dir, f"{tag}_fps_idx_k{ks[2]}_start{si}.npy"), idx_l)

    print(f"dataset={args.dataset}")
    print(f"输入: {path}")
    roi_info = "n/a"
    if args.dataset == "kitti":
        roi_info = "off" if args.no_roi else "on"
    print(f"点数 N={n} (full N={n_full}), ROI={roi_info}, start_idx={si}")
    print("三次 FPS 重叠统计:")
    print(rep)
    print(f"已保存: {out_png}")

    if not args.no_marginal_pdf:
        pdf_png = os.path.join(
            out_dir,
            f"{tag}_fps_marginal_pdf_{ks[0]}_{ks[1]}_{ks[2]}_start{si}.png",
        )
        plot_fps_marginal_pdf_compare(
            xyz,
            idx_s,
            idx_m,
            idx_l,
            ks,
            (args.color_c, args.color_b, args.color_a),
            pdf_png,
            show_full_reference=not args.marginal_no_full_ref,
        )
        print(f"已保存边际 PDF 对比: {pdf_png}")

    if not args.no_trajectory:
        traj_png = os.path.join(
            out_dir,
            f"{tag}_fps_trajectory_{ks[0]}_{ks[1]}_{ks[2]}_start{si}.png",
        )
        plot_fps_trajectories_row(
            xyz,
            (
                (idx_s, f"K={ks[0]} (smallest)", "turbo"),
                (idx_m, f"K={ks[1]}", "turbo"),
                (idx_l, f"K={ks[2]} (largest)", "turbo"),
            ),
            traj_png,
            title_prefix=f"{args.dataset.upper()} {stem}",
            start_idx=si,
            bg_subsample=args.trajectory_bg_points,
            elev=args.elev,
            azim=args.azim,
            milestone_max=args.trajectory_milestone_max,
            arrow_stride=args.trajectory_arrow_stride,
            line_width=args.trajectory_line_width,
            dense_scatter_stride=max(0, args.trajectory_point_stride),
        )
        print(f"已保存轨迹图: {traj_png}")


def main():
    p = build_parser()
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
