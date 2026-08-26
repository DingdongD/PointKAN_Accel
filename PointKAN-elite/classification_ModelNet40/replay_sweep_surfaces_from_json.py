#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ``*_sweep.json`` 重绘扫参曲面（无需重算实验）。

默认：无标题、无 xyz 轴标题、无刻度数字（曲面顶点数值标注本就未开启）。

示例::

  python replay_sweep_surfaces_from_json.py \\
    --json ./runs/fused_knn_sweep_sweep.json \\
    --name_suffix _notext

输出：与 JSON 里 ``png`` 同目录，``原文件名 + name_suffix + .png``。

若需要与扫参时完全相同的标题与刻度，加 ``--with_labels``。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_knn_coarse_fine_eval import plot_iou_surface


def _out_path(json_path: str, orig_png: str, name_suffix: str) -> str:
    jdir = os.path.dirname(os.path.abspath(json_path))
    stem, _ext = os.path.splitext(os.path.basename(orig_png))
    return os.path.join(jdir, f"{stem}{name_suffix}.png")


def _plot_surface(
    x_list,
    y_list,
    Z: np.ndarray,
    save_path: str,
    *,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    minimal: bool,
    dpi: int,
) -> None:
    if minimal:
        plot_iou_surface(
            x_list,
            y_list,
            Z,
            save_path=save_path,
            xlabel=None,
            ylabel=None,
            zlabel=None,
            title=None,
            show_title=False,
            show_axis_labels=False,
            show_tick_labels=False,
            dpi=dpi,
        )
    else:
        plot_iou_surface(
            x_list,
            y_list,
            Z,
            save_path=save_path,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            title=title,
            dpi=dpi,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="从 fused_knn sweep JSON 重绘 init×max_iter / clip×max_iter 曲面"
    )
    ap.add_argument("--json", type=str, required=True, help="sweep 结果 JSON 路径")
    ap.add_argument(
        "--name_suffix",
        type=str,
        default="_notext",
        help="插在输出文件名 .png 之前（默认 _notext）",
    )
    ap.add_argument("--dpi", type=int, default=600, help="输出 PNG 分辨率")
    ap.add_argument(
        "--with_labels",
        action="store_true",
        help="保留标题、轴标题与刻度数字（默认全部关闭）",
    )
    args = ap.parse_args()

    jpath = os.path.abspath(args.json)
    if not os.path.isfile(jpath):
        raise SystemExit(f"找不到 JSON: {jpath}")

    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    init_list = data["init_thresholds"]
    max_iter_list = data["max_iters"]
    clip_list = data["mult_alpha_clips"]
    minimal = not args.with_labels
    dpi = int(args.dpi)

    for rec in data.get("surfaces", []):
        Z = np.asarray(rec["Z"], dtype=np.float64)
        clip = float(rec["mult_alpha_clip"])
        out = _out_path(jpath, rec["png"], args.name_suffix)
        _plot_surface(
            init_list,
            max_iter_list,
            Z,
            out,
            xlabel="init_threshold",
            ylabel="max_iter",
            zlabel="KNN IoU",
            title=f"mult_alpha_clip={clip:g} (fixed per figure)",
            minimal=minimal,
            dpi=dpi,
        )
        print(f"  wrote {out}")

        if "Z_kan" in rec and "png_kan" in rec:
            Zk = np.asarray(rec["Z_kan"], dtype=np.float64)
            out_k = _out_path(jpath, rec["png_kan"], args.name_suffix)
            _plot_surface(
                init_list,
                max_iter_list,
                Zk,
                out_k,
                xlabel="init_threshold",
                ylabel="max_iter",
                zlabel="KNN IoU",
                title=f"mult_alpha_clip={clip:g} (fixed per figure)",
                minimal=minimal,
                dpi=dpi,
            )
            print(f"  wrote {out_k}")

    for rec in data.get("surfaces_clip_by_maxiter", []):
        Z = np.asarray(rec["Z"], dtype=np.float64)
        th = float(rec["init_threshold"])
        out = _out_path(jpath, rec["png"], args.name_suffix)
        _plot_surface(
            clip_list,
            max_iter_list,
            Z,
            out,
            xlabel="mult_alpha_clip",
            ylabel="max_iter",
            zlabel="KNN IoU",
            title=f"init_threshold={th:g} (fixed per figure)",
            minimal=minimal,
            dpi=dpi,
        )
        print(f"  wrote {out}")

        if "Z_kan" in rec and "png_kan" in rec:
            Zk = np.asarray(rec["Z_kan"], dtype=np.float64)
            out_k = _out_path(jpath, rec["png_kan"], args.name_suffix)
            _plot_surface(
                clip_list,
                max_iter_list,
                Zk,
                out_k,
                xlabel="mult_alpha_clip",
                ylabel="max_iter",
                zlabel="KNN IoU",
                title=f"init_threshold={th:g} (fixed per figure)",
                minimal=minimal,
                dpi=dpi,
            )
            print(f"  wrote {out_k}")


if __name__ == "__main__":
    main()
