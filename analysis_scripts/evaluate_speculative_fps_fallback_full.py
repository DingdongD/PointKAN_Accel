#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from spec_kan_fps import Config  # type: ignore
from spec_kan_fps_opt import ModelNet40Wrapper, collate_points, compute_iou_batch, fps_adaptive_argmax  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
    predict_full_radii_squared_from_fps_prefix_points,
)
from spec_kan_fps_opt_kan_coarse_fine_fused_eval import fps_segfilter_knn_coarse_fine_fused  # type: ignore
from spec_kan_fps_multidataset import build_dataset_pair, finalize_preset_paths  # type: ignore


class _PrintLogger:
    def info(self, msg: str, *args) -> None:
        if args:
            msg = msg % args
        print(msg)


def _auto_kan_save_dir(dataset: str, K: int, prefix_ratio: float) -> str:
    tag = dataset.strip().lower()
    if tag == "modelnet40":
        legacy = os.path.join(MN40_DIR, "runs", f"fps_radii_kan_K{K}_pr{prefix_ratio:.2f}")
        if os.path.isfile(os.path.join(legacy, "best.pt")):
            return legacy
    return os.path.join(
        MN40_DIR,
        "runs",
        "fps_radii_oneclick_all",
        f"fps_radii_kan_K{K}_{tag}_pr{prefix_ratio:.2f}",
    )


def _stats_from_lists(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    t = torch.tensor(values, dtype=torch.float64)
    return {"mean": float(t.mean().item()), "min": float(t.min().item()), "max": float(t.max().item())}


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not getattr(args, "data_root", "") and args.dataset == "modelnet40":
        args.data_root = args.root
    finalize_preset_paths(args)
    cfg_ds = Config()
    cfg_ds.num_points = int(args.num_points)
    cfg_ds.fps_steps = int(args.K)
    train_ds, test_ds, dataset_tag = build_dataset_pair(args, cfg_ds, _PrintLogger())
    dataset = train_ds if args.partition == "train" else test_ds
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_points,
    )
    kan_save_dir = args.kan_save_dir
    if not kan_save_dir or kan_save_dir == "auto":
        kan_save_dir = _auto_kan_save_dir(dataset_tag, args.K, args.prefix_ratio)
    model, cfg = load_kan_radii_model(kan_save_dir, device)
    cfg.prefix_ratio = float(args.prefix_ratio)
    s0 = kan_prefix_len_from_K(args.K, args.prefix_ratio) if args.defer_seg_filter else 1
    alpha_values = [float(x) for x in str(args.kan_alpha_list).split(",") if str(x).strip()]
    if not alpha_values:
        alpha_values = [1.0]

    branches = {
        "exact_ref": {
            "fallback_steps": 0,
            "active_steps": 0,
            "raw_candidate_sum": 0.0,
            "actual_candidate_sum": 0.0,
            "iou_values": [],
            "per_sample_fallback_rate": [],
        },
    }
    for alpha in alpha_values:
        branches[f"kan_alpha_{alpha:g}"] = {
            "fallback_steps": 0,
            "active_steps": 0,
            "raw_candidate_sum": 0.0,
            "actual_candidate_sum": 0.0,
            "iou_values": [],
            "per_sample_fallback_rate": [],
            "alpha": alpha,
        }
    num_samples = 0

    for batch_idx, (pts, _labels) in enumerate(tqdm(loader, desc="fallback full", ncols=100)):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        pts = pts.to(device, dtype=torch.float32)
        bsz, npts, _ = pts.shape
        k_eff = min(int(args.K), int(npts))
        num_samples += int(bsz)
        idx_exact, radii_sq_exact, _, _ = fps_adaptive_argmax(pts, k_eff, seed=args.seed, mode="exact")
        if args.kan_input == "fps_points":
            radii_sq_kan, _ = predict_full_radii_squared_from_fps_prefix_points(
                pts, idx_exact, k_eff, model, cfg, device
            )
        else:
            radii_sq_kan = predict_full_radii_squared_from_exact(radii_sq_exact, model, cfg, device)

        refs = [("exact_ref", radii_sq_exact)]
        refs.extend((f"kan_alpha_{alpha:g}", radii_sq_kan * alpha) for alpha in alpha_values)
        for name, radii_ref in refs:
            idx_f, _r, _knn, _kr, _kt, _dm, stats = fps_segfilter_knn_coarse_fine_fused(
                pts,
                k_eff,
                args.k_neighbors,
                radii_ref,
                args.segment_len,
                init_threshold=args.init_threshold,
                max_iter=args.max_iter,
                min_candidates_ratio=args.min_candidates_ratio,
                seed=args.seed,
                exclude_self=args.exclude_self,
                return_distances_matrix=False,
                seg_filter_start_t=s0,
                coarse_schedule=args.knn_coarse_schedule,
                mult_alpha_clip=args.knn_coarse_mult_alpha_clip,
                knn_coarse_pool_shortage=args.knn_coarse_pool_shortage,
            )
            iou = compute_iou_batch(idx_exact, idx_f)[0].detach().cpu()
            branch = branches[name]
            branch["iou_values"].extend(float(x) for x in iou.tolist())

            fallback = stats["used_fallback"][:, s0:]
            raw = stats["candidate_ratio"][:, s0:] * float(npts)
            actual = raw.clone()
            actual[fallback] = float(npts)
            branch["fallback_steps"] += int(fallback.sum().item())
            branch["active_steps"] += int(fallback.numel())
            branch["raw_candidate_sum"] += float(raw.sum().item())
            branch["actual_candidate_sum"] += float(actual.sum().item())
            per_sample = fallback.float().mean(dim=1).detach().cpu()
            branch["per_sample_fallback_rate"].extend(float(x) for x in per_sample.tolist())

    rows = []
    for name, branch in branches.items():
        active = max(int(branch["active_steps"]), 1)
        row = {
            "branch": name,
            "kan_alpha": float(branch.get("alpha", 1.0 if name == "exact_ref" else 0.0)),
            "num_samples": num_samples,
            "active_spec_steps": int(branch["active_steps"]),
            "fallback_steps": int(branch["fallback_steps"]),
            "fallback_rate": float(branch["fallback_steps"] / active),
            "raw_pass_candidates_mean": float(branch["raw_candidate_sum"] / active),
            "actual_candidates_mean_with_fallback": float(branch["actual_candidate_sum"] / active),
            "fps_iou_mean": float(sum(branch["iou_values"]) / max(len(branch["iou_values"]), 1)),
            "per_sample_fallback_rate_mean": _stats_from_lists(branch["per_sample_fallback_rate"])["mean"],
            "per_sample_fallback_rate_min": _stats_from_lists(branch["per_sample_fallback_rate"])["min"],
            "per_sample_fallback_rate_max": _stats_from_lists(branch["per_sample_fallback_rate"])["max"],
        }
        rows.append(row)

    return {
        "partition": args.partition,
        "dataset": dataset_tag,
        "num_points": args.num_points,
        "K": args.K,
        "segment_len": args.segment_len,
        "prefix_ratio": args.prefix_ratio,
        "seg_filter_start_t": s0,
        "defer_seg_filter": bool(args.defer_seg_filter),
        "kan_save_dir": os.path.abspath(kan_save_dir),
        "kan_input": args.kan_input,
        "kan_alpha_list": alpha_values,
        "num_samples": num_samples,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate speculative FPS fallback rate on a full dataset.")
    p.add_argument(
        "--dataset",
        default="modelnet40",
        choices=("modelnet40", "scanobjectnn", "shapenet_part", "kitti", "nuscenes", "s3dis", "waymo"),
    )
    p.add_argument("--root", default="./data")
    p.add_argument("--data_root", default="")
    p.add_argument("--preset", default="none", choices=("none", "voxelnet"))
    p.add_argument("--shapenet_root", default="")
    p.add_argument("--shapenet_normalize", action="store_true")
    p.add_argument("--scanobjectnn_train_h5", default="")
    p.add_argument("--scanobjectnn_test_h5", default="")
    p.add_argument("--outdoor_train_ratio", type=float, default=0.8)
    p.add_argument("--outdoor_eval_seed", type=int, default=0)
    p.add_argument("--kitti_use_roi", action="store_true")
    p.add_argument("--roi_x0", type=float, default=0.0)
    p.add_argument("--roi_x1", type=float, default=70.4)
    p.add_argument("--roi_y0", type=float, default=-40.0)
    p.add_argument("--roi_y1", type=float, default=40.0)
    p.add_argument("--roi_z0", type=float, default=-3.0)
    p.add_argument("--roi_z1", type=float, default=1.0)
    p.add_argument("--max_list_files", type=int, default=0)
    p.add_argument("--max_train_files", type=int, default=0)
    p.add_argument("--max_test_files", type=int, default=0)
    p.add_argument("--partition", default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--k_neighbors", type=int, default=24)
    p.add_argument("--init_threshold", type=float, default=0.1)
    p.add_argument("--max_iter", type=int, default=10)
    p.add_argument("--min_candidates_ratio", type=float, default=1.2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--seed", default="first")
    p.add_argument("--exclude_self", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--kan_save_dir", default="auto")
    p.add_argument("--kan_input", choices=("radii", "fps_points"), default="radii")
    p.add_argument("--kan_alpha_list", default="1.0", help="Comma-separated squared-threshold scale factors for KAN tau_sq.")
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--defer_seg_filter", action="store_true", default=True)
    p.add_argument("--no_defer_seg_filter", dest="defer_seg_filter", action="store_false")
    p.add_argument("--knn_coarse_schedule", choices=("bisect", "mult_then_bisect"), default="bisect")
    p.add_argument("--knn_coarse_mult_alpha_clip", type=float, default=0.25)
    p.add_argument("--knn_coarse_pool_shortage", choices=("keep_mid", "search_hi"), default="keep_mid")
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_process_full")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"fallback_full_{result['dataset']}_{args.partition}_K{args.K}_seg{args.segment_len}_pr{args.prefix_ratio:.2f}_{args.kan_input}"
    json_path = os.path.join(args.out_dir, stem + ".json")
    csv_path = os.path.join(args.out_dir, stem + ".csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("WROTE", json_path)
    print("WROTE", csv_path)


if __name__ == "__main__":
    main()
