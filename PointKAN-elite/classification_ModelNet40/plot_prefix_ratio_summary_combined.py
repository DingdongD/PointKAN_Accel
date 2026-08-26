#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将各数据集单独的 prefix_ratio → Test MSE 汇总曲线画在**同一张图**中。

数据来源：与 collect_prefix_ratio_results.py 生成的 JSON 一致，例如：
  runs/fps_radii_oneclick_all/prefix_ratio_summary_kan_shapenet_part_K512.json

**Test MSE 含义**：训练时 target 为「真实 FPS radii 曲线」经 prefix/suffix 重采样后的后缀段；
因此该曲线刻画的是 **KAN（或 MLP）相对真实 radii（可理解为 vanilla / GT 监督）的拟合误差**
随 prefix 比例的变化，**不是**在一张图里画「半径随 FPS 步 index 的两条空间曲线」。

若要看 **逐步 radii：GT 曲线 vs KAN 预测曲线**，请用：
  visualize_fps_radii_curves.py（--auto_from / --entry）

可选 ``--with_mlp_baseline``：同色同 marker 形状下，**KAN 与 MLP** 用不同线型区分。

样式：四边 **spines** 线宽 **5**；主曲线线宽 **4**；**加粗 ``--`` 虚线**，**各数据集不同 marker**；**major + minor** 网格，
**minor 网格极低透明度**；无图例、无标题；刻度数字默认关闭，可用 ``--show_tick_labels`` 打开。默认图幅 **1:1**。

对数 Y 默认时 **1×2 左右双子图**（共享 X）：左 Outdoor ``[1e-3,1e0]``；右 Indoor/Object ``[1e-7,1e-3]``。
各子图坐标轴 **框** 为 1:1（``set_box_aspect(1)``）；分屏时 ``--figsize`` 取**首项**为单面板边长（英寸），整图约为 ``(2w, w)``。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 左栏 / 较大 MSE 量级
OUTDOOR_DATASETS = frozenset({"kitti", "nuscenes", "waymo"})
# 右栏：modelnet40 / scanobjectnn / shapenet_part / s3dis 等

YLIM_OUTDOOR_LOG = (1e-3, 1e0)
YLIM_INDOOR_LOG = (1e-7, 1e-3)

SPINE_LINEWIDTH = 5.0
GRID_MAJOR_LW = 0.65
GRID_MAJOR_ALPHA = 0.38
GRID_MINOR_LW = 0.55
GRID_MINOR_ALPHA = 0.07

# 主曲线：加粗虚线；各数据集按序循环 marker
LINE_KAN_LS = "--"
LINE_KAN_LW = 4.0
MARKER_SIZE_KAN = 15.0
LINE_MLP_LS = "-."
LINE_MLP_LW = 3.2
MARKER_SIZE_MLP = 5.5

MARKERS_CYCLE = (
    "o",
    "s",
    "^",
    "v",
    "D",
    "P",
    "X",
    "*",
    "h",
    "8",
    "p",
    "<",
    ">",
)


def dataset_panel(ds: str) -> str:
    """返回 'outdoor' 或 'indoor'。"""
    key = ds.strip().lower()
    if key in OUTDOOR_DATASETS:
        return "outdoor"
    return "indoor"


def parse_summary_filename(name: str) -> Optional[Tuple[str, str, int]]:
    """
    prefix_ratio_summary_{model_type}_{dataset}_K{k}.json
    dataset 可含下划线（如 shapenet_part）。
    """
    m = re.match(r"^prefix_ratio_summary_(kan|mlp)_(.+)_K(\d+)\.json$", name, re.I)
    if not m:
        return None
    mt, ds, k = m.group(1).lower(), m.group(2), int(m.group(3))
    return mt, ds, k


def load_runs(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("runs") or []
    rows = sorted(rows, key=lambda r: float(r["prefix_ratio"]))
    return rows


def apply_spines_and_grid(ax, *, show_tick_labels: bool = False) -> None:
    """四边 spine；major + 高透明 minor 网格；刻度数字由 show_tick_labels 控制。"""
    for side in ("top", "right", "left", "bottom"):
        sp = ax.spines[side]
        sp.set_visible(True)
        sp.set_linewidth(SPINE_LINEWIDTH)
    ax.grid(
        True,
        which="major",
        linestyle="-",
        linewidth=GRID_MAJOR_LW,
        alpha=GRID_MAJOR_ALPHA,
        color="0.4",
    )
    ax.minorticks_on()
    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=GRID_MINOR_LW,
        alpha=GRID_MINOR_ALPHA,
        color="0.5",
    )
    ax.tick_params(
        axis="both",
        which="major",
        width=SPINE_LINEWIDTH,
        length=5,
        labelleft=show_tick_labels,
        labelbottom=show_tick_labels,
        labelright=False,
        labeltop=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=SPINE_LINEWIDTH * 0.45,
        length=2.5,
        labelleft=False,
        labelbottom=False,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="多数据集 prefix_ratio–Test MSE 曲线合一图")
    p.add_argument(
        "--base_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "runs", "fps_radii_oneclick_all"),
        help="汇总 JSON 所在目录",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="kan",
        choices=("kan", "mlp"),
        help="只纳入该模型类型的汇总文件",
    )
    p.add_argument("--k", type=int, default=512, help="只纳入文件名中 K 与此相同的汇总")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 PNG 路径；默认 base_dir/prefix_ratio_summary_combined_{mt}_K{k}.png",
    )
    p.add_argument(
        "--scale_y",
        type=str,
        default="log",
        choices=("log", "linear"),
        help="Y 轴刻度：log 便于跨数据集比较量级；linear 为线性",
    )
    p.add_argument(
        "--figsize",
        type=str,
        default="8,8",
        help="单图：宽,高（英寸）。双子图：首项 w 为单面板方形边长，整图 (2w,w)，次项可忽略",
    )
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="逗号分隔只画这些数据集；留空表示目录内全部",
    )
    p.add_argument(
        "--with_mlp_baseline",
        action="store_true",
        help="主曲线为 KAN 时，叠加同数据集的 MLP 汇总（需存在 prefix_ratio_summary_mlp_{dataset}_K{k}.json）",
    )
    p.add_argument(
        "--split_panels",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="auto: log 轴且同时含 outdoor+indoor 时 1×2 双子图；on 强制（仅一类数据时另一侧空）；off 单图",
    )
    p.add_argument(
        "--wspace",
        type=float,
        default=0.06,
        help="1×2 双子图水平间距（subplots_adjust wspace），越小越紧凑",
    )
    p.add_argument(
        "--show_tick_labels",
        action="store_true",
        help="显示坐标轴刻度数字（默认关闭，仅保留刻度线）",
    )
    args = p.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.isdir(base_dir):
        print(f"目录不存在: {base_dir}", file=sys.stderr)
        sys.exit(1)

    filter_ds = {s.strip() for s in args.datasets.split(",") if s.strip()}

    files: List[Tuple[str, str, int, str]] = []
    for fn in sorted(os.listdir(base_dir)):
        if not fn.endswith(".json"):
            continue
        parsed = parse_summary_filename(fn)
        if parsed is None:
            continue
        mt, ds, k = parsed
        if mt != args.model_type.lower() or k != args.k:
            continue
        if filter_ds and ds not in filter_ds:
            continue
        files.append((fn, ds, k, os.path.join(base_dir, fn)))

    if not files:
        print(
            f"未找到匹配的 prefix_ratio_summary_{args.model_type}_*_K{args.k}.json @ {base_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    with_mlp = bool(args.with_mlp_baseline) and args.model_type.lower() == "kan"
    if args.with_mlp_baseline and args.model_type.lower() != "kan":
        print("--with_mlp_baseline 仅在与 --model_type kan 联用时有效，已忽略。", file=sys.stderr)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print("需要 matplotlib 与 numpy", file=sys.stderr)
        raise SystemExit(1) from e

    w, h = [float(x.strip()) for x in args.figsize.split(",")[:2]]
    use_log = args.scale_y == "log"

    all_ds_sorted = sorted({ds for _, ds, _, _ in files})
    n_ds = len(all_ds_sorted)
    if n_ds <= 10:
        cmap = plt.cm.tab10(np.linspace(0, 1, max(3, n_ds)))
    else:
        cmap = plt.cm.tab20(np.linspace(0, 1, n_ds))
    color_map = {d: cmap[i] for i, d in enumerate(all_ds_sorted)}
    marker_map = {
        d: MARKERS_CYCLE[i % len(MARKERS_CYCLE)] for i, d in enumerate(all_ds_sorted)
    }

    outdoor_files = [t for t in files if dataset_panel(t[1]) == "outdoor"]
    indoor_files = [t for t in files if dataset_panel(t[1]) == "indoor"]

    use_split = args.split_panels == "on" or (
        args.split_panels == "auto"
        and use_log
        and len(outdoor_files) > 0
        and len(indoor_files) > 0
    )
    if args.split_panels == "off":
        use_split = False

    def plot_one_file_bundle(ax_plot, item: Tuple[str, str, int, str]) -> int:
        _fn, ds, _k, path = item
        runs = load_runs(path)
        if not runs:
            print(f"[skip] 无 runs: {path}", file=sys.stderr)
            return 0
        xs = [float(r["prefix_ratio"]) for r in runs]
        ys = [float(r["best_test_mse"]) for r in runs]
        color = color_map[ds]
        mk = marker_map[ds]
        n_add = 0
        if with_mlp:
            lab_kan = f"{ds} (KAN)"
        else:
            lab_kan = ds
        ax_plot.plot(
            xs,
            ys,
            linestyle=LINE_KAN_LS,
            linewidth=LINE_KAN_LW,
            color=color,
            marker=mk,
            markersize=MARKER_SIZE_KAN,
            markerfacecolor=color,
            # markeredgecolor="0.15",
            # markeredgewidth=0.6,
            label=lab_kan,
            alpha=0.95,
        )
        n_add += 1
        if with_mlp:
            mlp_json = os.path.join(base_dir, f"prefix_ratio_summary_mlp_{ds}_K{args.k}.json")
            if not os.path.isfile(mlp_json):
                print(f"[mlp skip] 无 {mlp_json}", file=sys.stderr)
            else:
                mruns = load_runs(mlp_json)
                if not mruns:
                    print(f"[mlp skip] 无 runs: {mlp_json}", file=sys.stderr)
                else:
                    mx = [float(r["prefix_ratio"]) for r in mruns]
                    my = [float(r["best_test_mse"]) for r in mruns]
                    ax_plot.plot(
                        mx,
                        my,
                        linestyle=LINE_MLP_LS,
                        linewidth=LINE_MLP_LW,
                        color=color,
                        marker=mk,
                        markersize=MARKER_SIZE_MLP,
                        markerfacecolor=color,
                        # markeredgecolor="0.15",
                        # markeredgewidth=0.5,
                        label=f"{ds} (MLP)",
                        alpha=0.88,
                    )
                    n_add += 1
        return n_add

    if use_split:
        panel_side = w
        fig_w, fig_h = 2 * panel_side, panel_side
        fig, (ax_left, ax_right) = plt.subplots(
            1,
            2,
            figsize=(fig_w, fig_h),
            dpi=120,
            sharex=True,
        )
        for item in outdoor_files:
            plot_one_file_bundle(ax_left, item)
        for item in indoor_files:
            plot_one_file_bundle(ax_right, item)
        if not outdoor_files:
            ax_left.text(
                0.5,
                0.5,
                "(无 Outdoor 数据集)",
                ha="center",
                va="center",
                transform=ax_left.transAxes,
                fontsize=10,
                color="0.5",
            )
        if not indoor_files:
            ax_right.text(
                0.5,
                0.5,
                "(无 Indoor/Object 数据集)",
                ha="center",
                va="center",
                transform=ax_right.transAxes,
                fontsize=10,
                color="0.5",
            )

        if use_log:
            ax_left.set_yscale("log")
            ax_right.set_yscale("log")
            ax_left.set_ylim(*YLIM_OUTDOOR_LOG)
            ax_right.set_ylim(*YLIM_INDOOR_LOG)

        apply_spines_and_grid(ax_left, show_tick_labels=args.show_tick_labels)
        apply_spines_and_grid(ax_right, show_tick_labels=args.show_tick_labels)
        ax_left.set_box_aspect(1)
        ax_right.set_box_aspect(1)

        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            top=0.97,
            bottom=0.10,
            wspace=args.wspace,
        )
    else:
        fig, ax = plt.subplots(figsize=(w, h), dpi=120)
        for item in files:
            plot_one_file_bundle(ax, item)
        if use_log:
            ax.set_yscale("log")
        apply_spines_and_grid(ax, show_tick_labels=args.show_tick_labels)
        ax.set_box_aspect(1)
        fig.subplots_adjust(left=0.10, right=0.96, top=0.97, bottom=0.08)

    out_path = (args.out or "").strip()
    if not out_path:
        suf = f"combined_{args.model_type}_K{args.k}"
        if with_mlp:
            suf = f"combined_kan_mlp_K{args.k}"
        out_path = os.path.join(base_dir, f"prefix_ratio_summary_{suf}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
