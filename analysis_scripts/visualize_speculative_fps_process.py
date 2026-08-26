#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from torch.utils.data import DataLoader


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from spec_kan_fps import lin_resample_1d  # type: ignore
from spec_kan_fps_opt import ModelNet40Wrapper, collate_points, compute_iou_batch, fps_adaptive_argmax  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
    predict_full_radii_squared_from_fps_prefix_points,
)
from spec_kan_fps_opt_kan_coarse_fine_fused_eval import (  # type: ignore
    fps_segfilter_knn_coarse_fine_fused,
    knn_exact_topk_at_centers,
    knn_neighbor_set_iou,
)


def _setup_font(font_size: int) -> None:
    candidates = [
        "/root/.fonts/Arial.ttf",
        "/root/.fonts/arial.ttf",
        "/root/.fonts/ARIALN.TTF",
        "/root/.fonts/arialbd.ttf",
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    family = "Arial"
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _style_ax(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    ax.tick_params(width=2.2, length=7)
    ax.grid(True, linestyle="--", linewidth=1.1, alpha=0.35)
    ax.set_axisbelow(True)


def _step_threshold(radii_ref_sq: torch.Tensor, K: int, segment_len: int) -> torch.Tensor:
    t = torch.arange(K, device=radii_ref_sq.device)
    L = max(1, int(segment_len))
    e = ((t // L + 1) * L - 1).clamp(max=K - 1)
    return radii_ref_sq[:, e]


@torch.no_grad()
def _kan_process_arrays(
    radii_sq: torch.Tensor,
    points: torch.Tensor,
    idx_exact: torch.Tensor,
    model: torch.nn.Module,
    cfg,
    kan_input: str,
    K: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    prefix_ratio = float(getattr(cfg, "prefix_ratio", 0.10))
    in_dim = int(getattr(cfg, "in_dim", 32))
    out_dim = int(getattr(cfg, "out_dim", 64))
    prefix_len = kan_prefix_len_from_K(K, prefix_ratio)

    if kan_input == "fps_points":
        full_sq, _ = predict_full_radii_squared_from_fps_prefix_points(
            points, idx_exact, K, model, cfg, device
        )
        idx_p = idx_exact[:, :prefix_len]
        prefix_pts = points[torch.arange(points.size(0), device=device).unsqueeze(1), idx_p]
        flat = prefix_pts.reshape(points.size(0), -1)
        model_input = lin_resample_1d(flat, in_dim)
        model_output = model(model_input)
        pred_eucl = lin_resample_1d(model_output, K).clamp(min=0.0)
        arrays = {
            "model_input": model_input[0].detach().cpu().numpy(),
            "model_output": model_output[0].detach().cpu().numpy(),
            "pred_eucl": pred_eucl[0].detach().cpu().numpy(),
            "prefix_eucl": torch.sqrt(radii_sq[:, :prefix_len].clamp(min=0.0))[0].detach().cpu().numpy(),
        }
        return full_sq, arrays

    full_sq = predict_full_radii_squared_from_exact(radii_sq, model, cfg, device)
    prefix_sq = radii_sq[:, :prefix_len].clamp(min=0.0)
    prefix_eucl = torch.sqrt(prefix_sq)
    suffix_len = K - prefix_len
    model_input = lin_resample_1d(prefix_eucl, in_dim)
    model_output = model(model_input)
    suffix_pred = lin_resample_1d(model_output, suffix_len)
    pred_eucl = torch.cat([prefix_eucl, suffix_pred], dim=1).clamp(min=0.0)
    arrays = {
        "model_input": model_input[0].detach().cpu().numpy(),
        "model_output": model_output[0].detach().cpu().numpy(),
        "pred_eucl": pred_eucl[0].detach().cpu().numpy(),
        "prefix_eucl": prefix_eucl[0].detach().cpu().numpy(),
    }
    return full_sq, arrays


def _actual_fps_candidates(stats: Dict[str, torch.Tensor], num_points: int) -> np.ndarray:
    raw = stats["candidate_ratio"][0].detach().cpu().numpy() * float(num_points)
    fallback = stats["used_fallback"][0].detach().cpu().numpy().astype(bool)
    actual = raw.copy()
    actual[fallback] = float(num_points)
    return actual


@torch.no_grad()
def collect_one(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ModelNet40Wrapper(args.partition, args.num_points, root=args.root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )
    sample = None
    label = None
    for i, (pts, lab) in enumerate(loader):
        if i == args.sample_idx:
            sample = pts
            label = lab
            break
    if sample is None:
        raise IndexError(f"sample_idx={args.sample_idx} out of range")

    points = sample.to(device, dtype=torch.float32)
    K = min(int(args.K), points.size(1))
    model, cfg = load_kan_radii_model(args.kan_save_dir, device)
    cfg.prefix_ratio = float(args.prefix_ratio)
    prefix_len = kan_prefix_len_from_K(K, float(args.prefix_ratio))
    seg_start = prefix_len if args.defer_seg_filter else 1

    idx_exact, radii_exact_sq, _, _ = fps_adaptive_argmax(points, K, seed=args.seed, mode="exact")
    radii_kan_sq, kan_arrays = _kan_process_arrays(
        radii_exact_sq, points, idx_exact, model, cfg, args.kan_input, K, device
    )

    def run_branch(name: str, radii_ref_sq: torch.Tensor) -> Dict:
        idx, radii_fps_sq, knn_idx, _knn_r, knn_th, _dm, stats = fps_segfilter_knn_coarse_fine_fused(
            points,
            K,
            args.k_neighbors,
            radii_ref_sq,
            args.segment_len,
            init_threshold=args.init_threshold,
            max_iter=args.max_iter,
            min_candidates_ratio=args.min_candidates_ratio,
            seed=args.seed,
            exclude_self=args.exclude_self,
            return_distances_matrix=False,
            seg_filter_start_t=seg_start,
            coarse_schedule=args.knn_coarse_schedule,
            mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
            knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
        )
        iou_fps = compute_iou_batch(idx_exact, idx)[0][0].item()
        knn_ref = knn_exact_topk_at_centers(points, idx, args.k_neighbors, exclude_self=args.exclude_self)
        knn_iou = knn_neighbor_set_iou(knn_idx, knn_ref)[0].mean().item()
        return {
            "idx": idx[0].detach().cpu().numpy(),
            "radii_sq": radii_fps_sq[0].detach().cpu().numpy(),
            "knn_threshold": knn_th[0].detach().cpu().numpy(),
            "fps_candidates_actual": _actual_fps_candidates(stats, args.num_points),
            "fps_candidates_raw": (stats["candidate_ratio"][0].detach().cpu().numpy() * float(args.num_points)),
            "fps_fallback": stats["used_fallback"][0].detach().cpu().numpy().astype(bool),
            "knn_candidates": stats["knn_coarse_candidate_count"][0].detach().cpu().numpy(),
            "knn_iters": stats["knn_coarse_threshold_iters"][0].detach().cpu().numpy(),
            "iou_fps": float(iou_fps),
            "knn_iou": float(knn_iou),
        }

    return {
        "points": points[0].detach().cpu().numpy(),
        "label": int(label.reshape(-1)[0].item()) if label is not None else None,
        "idx_exact": idx_exact[0].detach().cpu().numpy(),
        "radii_exact_sq": radii_exact_sq[0].detach().cpu().numpy(),
        "tau_exact_sq": _step_threshold(radii_exact_sq, K, args.segment_len)[0].detach().cpu().numpy(),
        "radii_kan_sq": radii_kan_sq[0].detach().cpu().numpy(),
        "tau_kan_sq": _step_threshold(radii_kan_sq, K, args.segment_len)[0].detach().cpu().numpy(),
        "kan_arrays": kan_arrays,
        "prefix_len": int(prefix_len),
        "seg_start": int(seg_start),
        "K": int(K),
        "exact_branch": run_branch("exact", radii_exact_sq),
        "kan_branch": run_branch("kan", radii_kan_sq),
    }


def plot_process(data: Dict, args: argparse.Namespace) -> Dict[str, str]:
    _setup_font(args.fontsize)
    os.makedirs(args.out_dir, exist_ok=True)
    t = np.arange(data["K"])
    pts = data["points"]
    idx_exact = data["idx_exact"]
    prefix_len = data["prefix_len"]
    seg_start = data["seg_start"]
    exact = data["exact_branch"]
    kan = data["kan_branch"]

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    ax0, ax1, ax2, ax3 = axes.flatten()

    ax0.scatter(pts[:, 0], pts[:, 1], s=16, color="#cfcfcf", edgecolor="none", label="Input points")
    warm_idx = idx_exact[:prefix_len]
    post_idx = idx_exact[prefix_len : min(prefix_len + args.highlight_post_points, data["K"])]
    ax0.scatter(pts[warm_idx, 0], pts[warm_idx, 1], s=36, color="#ffc000", edgecolor="black", linewidth=0.4, label="Warmup exact FPS")
    if len(post_idx):
        ax0.scatter(pts[post_idx, 0], pts[post_idx, 1], s=34, color="#00b0f0", edgecolor="black", linewidth=0.4, label="Post-KAN FPS")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_title(f"Sample {args.sample_idx}: warmup prefix P={prefix_len}")
    ax0.legend(frameon=False, loc="best")
    _style_ax(ax0)

    ax1.plot(t, np.sqrt(np.maximum(data["radii_exact_sq"], 0)), color="#00b0f0", lw=3.0, label="Exact FPS radius")
    ax1.plot(t, np.sqrt(np.maximum(data["radii_kan_sq"], 0)), color="#ffc000", lw=2.6, linestyle="--", label="KAN predicted radius")
    ax1.plot(t, np.sqrt(np.maximum(data["tau_kan_sq"], 0)), color="#c08000", lw=2.0, linestyle=":", label="Segment threshold")
    ax1.axvspan(0, prefix_len - 1, color="#eeeeee", alpha=0.75, label="warmup")
    ax1.axvline(seg_start, color="#444444", lw=2.0, linestyle="--")
    ax1.set_xlabel("FPS iteration")
    ax1.set_ylabel("Radius")
    ax1.set_title("Warmup exact FPS -> KAN threshold curve")
    ax1.legend(frameon=False, loc="best")
    _style_ax(ax1)

    ax2.plot(t, np.full_like(t, args.num_points, dtype=float), color="#777777", lw=2.0, linestyle=":", label="Vanilla FPS")
    ax2.plot(t, exact["fps_candidates_actual"], color="#00b0f0", lw=2.8, label="Spec-FPS oracle threshold")
    ax2.plot(t, kan["fps_candidates_actual"], color="#c08000", lw=2.4, linestyle="--", label="Spec-FPS KAN threshold")
    fallback_t = np.where(kan["fps_fallback"])[0]
    if fallback_t.size:
        ax2.scatter(fallback_t, kan["fps_candidates_actual"][fallback_t], marker="x", s=42, color="#d62728", label="KAN fallback")
    ax2.axvspan(0, prefix_len - 1, color="#eeeeee", alpha=0.75)
    ax2.axvline(seg_start, color="#444444", lw=2.0, linestyle="--")
    ax2.set_xlabel("FPS iteration")
    ax2.set_ylabel("FPS argmax candidates")
    ax2.set_title("Distance comparisons for FPS selection")
    ax2.legend(frameon=False, loc="best")
    _style_ax(ax2)

    ax3.plot(t, kan["knn_candidates"], color="#00b0f0", lw=2.6, label="KNN coarse candidates")
    ax3b = ax3.twinx()
    ax3b.plot(t, kan["knn_iters"], color="#ffc000", lw=2.2, linestyle="--", label="Coarse threshold iterations")
    ax3.axvspan(0, prefix_len - 1, color="#eeeeee", alpha=0.75)
    ax3.axvline(seg_start, color="#444444", lw=2.0, linestyle="--")
    ax3.set_xlabel("FPS iteration")
    ax3.set_ylabel("KNN candidates")
    ax3b.set_ylabel("Iterations")
    ax3.set_title("Coarse-to-fine KNN filtering after each FPS center")
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
    _style_ax(ax3)
    for spine in ax3b.spines.values():
        spine.set_linewidth(2.5)
    ax3b.tick_params(width=2.2, length=7)

    fig.tight_layout()
    base = os.path.join(args.out_dir, f"speculative_fps_process_sample{args.sample_idx}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": base + ".pdf", "png": base + ".png"}


def write_summary(data: Dict, paths: Dict[str, str], args: argparse.Namespace) -> str:
    out = {
        "sample_idx": args.sample_idx,
        "label": data["label"],
        "K": data["K"],
        "num_points": args.num_points,
        "segment_len": args.segment_len,
        "prefix_len": data["prefix_len"],
        "seg_filter_start_t": data["seg_start"],
        "kan_save_dir": os.path.abspath(args.kan_save_dir),
        "kan_input": args.kan_input,
        "exact_branch": {
            "fps_iou_vs_exact": data["exact_branch"]["iou_fps"],
            "knn_iou_vs_exact_topk_at_same_centers": data["exact_branch"]["knn_iou"],
            "mean_fps_candidates": float(np.mean(data["exact_branch"]["fps_candidates_actual"][data["seg_start"] :])),
            "num_fallback_steps": int(np.sum(data["exact_branch"]["fps_fallback"])),
        },
        "kan_branch": {
            "fps_iou_vs_exact": data["kan_branch"]["iou_fps"],
            "knn_iou_vs_exact_topk_at_same_centers": data["kan_branch"]["knn_iou"],
            "mean_fps_candidates": float(np.mean(data["kan_branch"]["fps_candidates_actual"][data["seg_start"] :])),
            "raw_pass_candidates_mean": float(np.mean(data["kan_branch"]["fps_candidates_raw"][data["seg_start"] :])),
            "num_fallback_steps": int(np.sum(data["kan_branch"]["fps_fallback"][data["seg_start"] :])),
            "mean_knn_candidates": float(np.mean(data["kan_branch"]["knn_candidates"])),
            "mean_knn_threshold_iters": float(np.mean(data["kan_branch"]["knn_iters"])),
        },
        "plots": paths,
    }
    path = os.path.join(args.out_dir, f"speculative_fps_process_sample{args.sample_idx}_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def write_per_step_csv(data: Dict, args: argparse.Namespace) -> str:
    path = os.path.join(args.out_dir, f"speculative_fps_process_sample{args.sample_idx}_per_step.csv")
    exact = data["exact_branch"]
    kan = data["kan_branch"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t",
                "phase",
                "exact_radius",
                "kan_pred_radius",
                "exact_segment_threshold",
                "kan_segment_threshold",
                "oracle_fps_candidates_actual",
                "kan_fps_candidates_actual",
                "kan_fps_candidates_raw_pass",
                "kan_fallback",
                "kan_knn_coarse_candidates",
                "kan_knn_threshold_iters",
                "kan_knn_threshold",
            ],
        )
        writer.writeheader()
        for t in range(data["K"]):
            phase = "warmup_exact" if t < data["seg_start"] else "kan_seg_filter"
            writer.writerow(
                {
                    "t": t,
                    "phase": phase,
                    "exact_radius": float(np.sqrt(max(data["radii_exact_sq"][t], 0.0))),
                    "kan_pred_radius": float(np.sqrt(max(data["radii_kan_sq"][t], 0.0))),
                    "exact_segment_threshold": float(np.sqrt(max(data["tau_exact_sq"][t], 0.0))),
                    "kan_segment_threshold": float(np.sqrt(max(data["tau_kan_sq"][t], 0.0))),
                    "oracle_fps_candidates_actual": float(exact["fps_candidates_actual"][t]),
                    "kan_fps_candidates_actual": float(kan["fps_candidates_actual"][t]),
                    "kan_fps_candidates_raw_pass": float(kan["fps_candidates_raw"][t]),
                    "kan_fallback": int(bool(kan["fps_fallback"][t])),
                    "kan_knn_coarse_candidates": float(kan["knn_candidates"][t]),
                    "kan_knn_threshold_iters": int(kan["knn_iters"][t]),
                    "kan_knn_threshold": float(kan["knn_threshold"][t]),
                }
            )
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize one-sample speculative FPS + KAN threshold + coarse-fine KNN process.")
    p.add_argument("--root", default="./data")
    p.add_argument("--partition", default="test", choices=("train", "test"))
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--k_neighbors", type=int, default=24)
    p.add_argument("--init_threshold", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=10)
    p.add_argument("--min_candidates_ratio", type=float, default=1.2)
    p.add_argument("--seed", default="first")
    p.add_argument("--exclude_self", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--kan_save_dir", default="/workspace/PointKAN-pytorch/PointKAN-elite/classification_ModelNet40/runs/fps_radii_kan_K512_pr0.10")
    p.add_argument("--kan_input", choices=("radii", "fps_points"), default="radii")
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--defer_seg_filter", action="store_true", default=True)
    p.add_argument("--no_defer_seg_filter", dest="defer_seg_filter", action="store_false")
    p.add_argument("--knn_coarse_schedule", choices=("bisect", "mult_then_bisect"), default="bisect")
    p.add_argument("--knn_coarse_mult_alpha_clip", type=float, default=0.25)
    p.add_argument("--knn_coarse_pool_shortage", choices=("keep_mid", "search_hi"), default="keep_mid")
    p.add_argument("--highlight_post_points", type=int, default=64)
    p.add_argument("--fontsize", type=int, default=28)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_process")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = collect_one(args)
    paths = plot_process(data, args)
    paths["per_step_csv"] = write_per_step_csv(data, args)
    summary = write_summary(data, paths, args)
    print(paths["pdf"])
    print(paths["png"])
    print(summary)
    print(
        json.dumps(
            {
                "prefix_len": data["prefix_len"],
                "fps_iou_kan": data["kan_branch"]["iou_fps"],
                "knn_iou_kan": data["kan_branch"]["knn_iou"],
                "kan_fallback_steps": int(np.sum(data["kan_branch"]["fps_fallback"][data["seg_start"] :])),
                "kan_mean_fps_candidates": float(np.mean(data["kan_branch"]["fps_candidates_actual"][data["seg_start"] :])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
