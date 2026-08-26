#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR), os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scanobjnn_specfps_scale_sweep import (  # type: ignore
    exact_fps_radii_sq,
    segfilter_stats_remaining_denominator,
    set_iou,
)
from spec_kan_fps import Config  # type: ignore
from spec_kan_fps_multidataset import build_dataset_pair, finalize_preset_paths  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


DATASET_LABELS = {
    "modelnet40": "ModelNet40",
    "shapenet_part": "ShapeNet",
    "scanobjectnn": "ScanObjectNN",
}

SCANOBJECTNN_LABELS = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


class _PrintLogger:
    def info(self, msg: str, *args) -> None:
        if args:
            msg = msg % args
        print(msg)


@dataclass
class SampleResult:
    dataset: str
    sample_index: int
    label: int
    category_name: str
    num_points: int
    K: int
    segment_len: int
    prefix_ratio: float
    seg_filter_start_t: int
    iou: float
    matched_points: int
    union_points: int
    fallback_rate: float
    raw_candidates_mean: float
    actual_candidates_mean: float
    pts: np.ndarray
    idx_exact: np.ndarray
    idx_spec: np.ndarray


def _register_arial() -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            arial_name = font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams.update(
                {
                    "font.family": arial_name,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                    "axes.unicode_minus": False,
                }
            )
            return


def _auto_kan_save_dir(dataset: str, K: int, prefix_ratio: float) -> str:
    if dataset == "modelnet40":
        return os.path.join(MN40_DIR, "runs", f"fps_radii_kan_K{K}_pr{prefix_ratio:.2f}")
    return os.path.join(
        MN40_DIR,
        "runs",
        "fps_radii_oneclick_all",
        f"fps_radii_kan_K{K}_{dataset}_pr{prefix_ratio:.2f}",
    )


def _make_dataset_args(dataset: str, num_points: int) -> argparse.Namespace:
    args = argparse.Namespace(
        dataset=dataset,
        partition="test",
        preset="voxelnet",
        data_root="",
        shapenet_root="",
        shapenet_normalize=False,
        scanobjectnn_train_h5="",
        scanobjectnn_test_h5="",
        num_points=num_points,
        K=512,
        kitti_use_roi=False,
        roi_x0=0.0,
        roi_x1=70.4,
        roi_y0=-40.0,
        roi_y1=40.0,
        roi_z0=-3.0,
        roi_z1=1.0,
        outdoor_train_ratio=0.8,
        outdoor_eval_seed=12345,
        max_list_files=0,
        max_train_files=0,
        max_test_files=0,
    )
    finalize_preset_paths(args)
    return args


def _modelnet_label_name(label: int) -> str:
    shape_names_path = "/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048/shape_names.txt"
    if os.path.isfile(shape_names_path):
        with open(shape_names_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if 0 <= label < len(names):
            return names[label]
    return f"label_{label}"


def _modelnet_label_index(category_name: str) -> int:
    shape_names_path = "/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048/shape_names.txt"
    if not os.path.isfile(shape_names_path):
        raise FileNotFoundError(f"Missing ModelNet40 shape names: {shape_names_path}")
    with open(shape_names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    normalized = category_name.strip().lower()
    if normalized not in names:
        raise ValueError(f"ModelNet40 category {category_name!r} not found in {shape_names_path}")
    return int(names.index(normalized))


def _dataset_category_name(dataset: str, test_ds, sample_index: int, label: int) -> str:
    if dataset == "modelnet40":
        return _modelnet_label_name(label)
    if dataset == "shapenet_part" and hasattr(test_ds, "datapath"):
        cat, _path = test_ds.datapath[sample_index]
        return str(cat)
    if dataset == "scanobjectnn" and 0 <= label < len(SCANOBJECTNN_LABELS):
        return SCANOBJECTNN_LABELS[label]
    return f"label_{label}"


def _select_modelnet_category_index(test_ds, category_name: str) -> int:
    target_label = _modelnet_label_index(category_name)
    labels = None
    if hasattr(test_ds, "ds") and hasattr(test_ds.ds, "label"):
        labels = np.asarray(test_ds.ds.label).reshape(-1)
    elif hasattr(test_ds, "label"):
        labels = np.asarray(test_ds.label).reshape(-1)
    if labels is None:
        raise RuntimeError("Cannot inspect ModelNet40 labels for category-based sample selection.")
    hits = np.where(labels == target_label)[0]
    if hits.size == 0:
        raise RuntimeError(f"No ModelNet40 test sample found for category {category_name!r}")
    return int(hits[0])


def _load_test_sample(
    dataset: str,
    num_points: int,
    sample_index: int,
    modelnet_category: str = "",
) -> Tuple[torch.Tensor, int, str, str, int]:
    args = _make_dataset_args(dataset, num_points)
    cfg = Config()
    cfg.num_points = num_points
    _, test_ds, dataset_tag = build_dataset_pair(args, cfg, _PrintLogger())
    if dataset_tag == "modelnet40" and modelnet_category.strip():
        sample_index = _select_modelnet_category_index(test_ds, modelnet_category)
    sample_index = max(0, min(int(sample_index), len(test_ds) - 1))
    pts, label = test_ds[sample_index]
    label = int(np.asarray(label).reshape(-1)[0])
    category_name = _dataset_category_name(dataset_tag, test_ds, sample_index, label)
    if not torch.is_tensor(pts):
        pts = torch.as_tensor(pts)
    pts = pts[:, :3].to(dtype=torch.float32).contiguous()
    return pts, label, dataset_tag, category_name, sample_index


@torch.no_grad()
def _evaluate_sample(
    dataset: str,
    sample_index: int,
    num_points: int,
    K: int,
    segment_len: int,
    prefix_ratio: float,
    seed: str,
    device: torch.device,
    modelnet_category: str = "",
) -> SampleResult:
    pts_cpu, label, dataset_tag, category_name, resolved_sample_index = _load_test_sample(
        dataset, num_points, sample_index, modelnet_category=modelnet_category
    )
    pts = pts_cpu.unsqueeze(0).to(device)
    K_eff = min(int(K), int(pts.size(1)))
    idx_exact, radii_sq_exact = exact_fps_radii_sq(pts, K_eff, seed=seed)

    kan_dir = _auto_kan_save_dir(dataset_tag, K_eff, prefix_ratio)
    model, cfg = load_kan_radii_model(kan_dir, device)
    cfg.prefix_ratio = float(prefix_ratio)
    s0 = kan_prefix_len_from_K(K_eff, prefix_ratio)
    radii_sq_kan = predict_full_radii_squared_from_exact(radii_sq_exact, model, cfg, device)
    idx_spec, stats = segfilter_stats_remaining_denominator(
        pts, K_eff, radii_sq_kan, segment_len, s0, seed=seed
    )

    exact_np = idx_exact[0].detach().cpu().numpy()
    spec_np = idx_spec[0].detach().cpu().numpy()
    exact_set = set(exact_np.tolist())
    spec_set = set(spec_np.tolist())
    matched = len(exact_set & spec_set)
    union = len(exact_set | spec_set)
    iou = set_iou(idx_exact, idx_spec)[0]

    return SampleResult(
        dataset=dataset_tag,
        sample_index=int(resolved_sample_index),
        label=int(label),
        category_name=category_name,
        num_points=int(pts.size(1)),
        K=int(K_eff),
        segment_len=int(segment_len),
        prefix_ratio=float(prefix_ratio),
        seg_filter_start_t=int(s0),
        iou=float(iou),
        matched_points=int(matched),
        union_points=int(union),
        fallback_rate=float(stats["fallback_rate"]),
        raw_candidates_mean=float(stats["raw_candidates_mean"]),
        actual_candidates_mean=float(stats["actual_candidates_mean"]),
        pts=pts_cpu.detach().cpu().numpy(),
        idx_exact=exact_np,
        idx_spec=spec_np,
    )


def _set_equal_3d_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _style_3d_axis(ax, pts: np.ndarray) -> None:
    _set_equal_3d_axes(ax, pts)
    ax.view_init(elev=18, azim=-58)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_linewidth(1.8)


def _mark_ordered_points(ax, pts: np.ndarray, indices: np.ndarray, count: int) -> None:
    chosen = pts[np.asarray(indices[:count], dtype=np.int64)]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    offsets = np.array(
        [
            [0.12, 0.06, 0.10],
            [-0.12, 0.08, 0.10],
            [0.11, -0.11, 0.10],
            [-0.11, -0.09, 0.10],
            [0.13, 0.00, -0.09],
            [-0.13, 0.04, -0.08],
            [0.04, 0.14, 0.10],
            [0.00, -0.15, 0.10],
        ],
        dtype=np.float32,
    )
    ax.scatter(
        chosen[:, 0], chosen[:, 1], chosen[:, 2],
        s=130, c="#000000", edgecolors="#ffffff", linewidths=1.6,
        depthshade=False, zorder=24,
    )
    for start, end in zip(chosen[:-1], chosen[1:]):
        vec = end - start
        ax.quiver(
            start[0], start[1], start[2], vec[0], vec[1], vec[2],
            color="#000000", linewidth=2.6, arrow_length_ratio=0.16,
            normalize=False, zorder=22,
        )
    for order, pnt in ((1, chosen[0]), (len(chosen), chosen[-1])):
        label_pos = pnt + offsets[(order - 1) % len(offsets)] * span
        ax.text(
            label_pos[0], label_pos[1], label_pos[2], str(order),
            fontsize=30, color="#000000", fontweight="bold",
            ha="center", va="center", zorder=25,
        )


def _plot_results(results: List[SampleResult], out_stem: str) -> None:
    _register_arial()
    fig = plt.figure(figsize=(15.6, 12.4))
    grid = fig.add_gridspec(
        len(results), 3,
        left=0.018, right=0.996, bottom=0.018, top=0.905,
        wspace=-0.32, hspace=-0.20,
    )
    order_mark_count = 8
    columns = ["Original", "Vanilla FPS", "Speculative FPS"]
    grey = "#cfd3d7"
    original_black = "#111111"
    vanilla = "#ffc000"
    spec = "#00b0f0"

    for row, res in enumerate(results):
        pts = res.pts
        exact_pts = pts[res.idx_exact]
        spec_pts = pts[res.idx_spec]
        for col in range(3):
            ax = fig.add_subplot(grid[row, col], projection="3d")
            if col == 0:
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=5, c=original_black, alpha=0.58, linewidths=0, depthshade=False,
                )
            else:
                ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    s=4, c=grey, alpha=0.22, linewidths=0, depthshade=False,
                )
            if col == 1:
                ax.scatter(
                    exact_pts[:, 0], exact_pts[:, 1], exact_pts[:, 2],
                    s=14, c=vanilla, alpha=0.92, linewidths=0, depthshade=False,
                )
                _mark_ordered_points(ax, pts, res.idx_exact, order_mark_count)
            elif col == 2:
                ax.scatter(
                    spec_pts[:, 0], spec_pts[:, 1], spec_pts[:, 2],
                    s=14, c=spec, alpha=0.92, linewidths=0, depthshade=False,
                )
                _mark_ordered_points(ax, pts, res.idx_spec, order_mark_count)
            _style_3d_axis(ax, pts)
            ax.margins(0.0)
            if row == 0:
                ax.set_title(columns[col], fontsize=36, pad=2)
            if col == 0:
                ax.text2D(
                    -0.055, 0.50, DATASET_LABELS.get(res.dataset, res.dataset),
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=36, fontweight="bold",
                )
    png_path = out_stem + ".png"
    pdf_path = out_stem + ".pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    with Image.open(png_path) as im:
        im.convert("RGB").save(pdf_path, "PDF", resolution=300.0)


def _write_csv(results: List[SampleResult], csv_path: str) -> None:
    fields = [
        "dataset_name",
        "dataset",
        "sample_index",
        "label",
        "category_name",
        "num_points",
        "K",
        "segment_len",
        "prefix_ratio",
        "seg_filter_start_t",
        "vanilla_vs_speculative_iou",
        "matched_points",
        "union_points",
        "fallback_rate",
        "fallback_rate_pct",
        "raw_candidates_mean",
        "actual_candidates_mean",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "dataset_name": DATASET_LABELS.get(res.dataset, res.dataset),
                    "dataset": res.dataset,
                    "sample_index": res.sample_index,
                    "label": res.label,
                    "category_name": res.category_name,
                    "num_points": res.num_points,
                    "K": res.K,
                    "segment_len": res.segment_len,
                    "prefix_ratio": res.prefix_ratio,
                    "seg_filter_start_t": res.seg_filter_start_t,
                    "vanilla_vs_speculative_iou": res.iou,
                    "matched_points": res.matched_points,
                    "union_points": res.union_points,
                    "fallback_rate": res.fallback_rate,
                    "fallback_rate_pct": 100.0 * res.fallback_rate,
                    "raw_candidates_mean": res.raw_candidates_mean,
                    "actual_candidates_mean": res.actual_candidates_mean,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize vanilla FPS vs KAN-assisted speculative FPS on three datasets."
    )
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--segment_len", type=int, default=32)
    parser.add_argument("--prefix_ratio", type=float, default=0.10)
    parser.add_argument("--seed", choices=("first", "max_norm"), default="first")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--modelnet_category", default="chair")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(REPO_ROOT, "my_progressive_out", "speculative_fps_multidataset_visual"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    datasets = ["modelnet40", "shapenet_part", "scanobjectnn"]
    results = [
        _evaluate_sample(
            dataset=dataset,
            sample_index=args.sample_index,
            num_points=args.num_points,
            K=args.K,
            segment_len=args.segment_len,
            prefix_ratio=args.prefix_ratio,
            seed=args.seed,
            device=device,
            modelnet_category=args.modelnet_category,
        )
        for dataset in datasets
    ]

    stem = os.path.join(
        args.out_dir,
        f"vanilla_vs_speculative_fps_iou_three_datasets_N{args.num_points}_K{args.K}_L{args.segment_len}",
    )
    _write_csv(results, stem + ".csv")
    _plot_results(results, stem)
    for res in results:
        print(
            f"{DATASET_LABELS.get(res.dataset, res.dataset)}: "
            f"sample={res.sample_index}, category={res.category_name}, label={res.label}, "
            f"IoU={res.iou:.6f}, matched={res.matched_points}/{res.K}, "
            f"fallback={100.0 * res.fallback_rate:.2f}%, "
            f"raw_cand={res.raw_candidates_mean:.2f}, actual_cand={res.actual_candidates_mean:.2f}"
        )
    print("WROTE", stem + ".csv")
    print("WROTE", stem + ".pdf")
    print("WROTE", stem + ".png")


if __name__ == "__main__":
    main()
