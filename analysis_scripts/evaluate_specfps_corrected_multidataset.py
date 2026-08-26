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
for path in (MN40_DIR, os.path.dirname(MN40_DIR), os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scanobjnn_specfps_scale_sweep import (  # type: ignore
    exact_fps_radii_sq,
    mean,
    segfilter_stats_remaining_denominator,
    set_iou,
)
from spec_kan_fps import Config, collate_points  # type: ignore
from spec_kan_fps_multidataset import build_dataset_pair, finalize_preset_paths  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


class _PrintLogger:
    def info(self, msg: str, *args) -> None:
        if args:
            msg = msg % args
        print(msg)


def _auto_kan_save_dir(dataset: str, K: int, prefix_ratio: float) -> str:
    if dataset == "modelnet40":
        return os.path.join(MN40_DIR, "runs", f"fps_radii_kan_K{K}_pr{prefix_ratio:.2f}")
    return os.path.join(
        MN40_DIR,
        "runs",
        "fps_radii_oneclick_all",
        f"fps_radii_kan_K{K}_{dataset}_pr{prefix_ratio:.2f}",
    )


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict:
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    kan_save_dir = args.kan_save_dir if args.kan_save_dir != "auto" else _auto_kan_save_dir(dataset_tag, args.K, args.prefix_ratio)
    model, cfg = load_kan_radii_model(kan_save_dir, device)
    cfg.prefix_ratio = float(args.prefix_ratio)
    s0 = kan_prefix_len_from_K(args.K, args.prefix_ratio)

    accum = {
        "num_samples": 0,
        "exact_ref_active_spec_steps": 0,
        "exact_ref_fallback_steps": 0,
        "exact_ref_raw_candidates_sum": 0.0,
        "exact_ref_actual_candidates_sum": 0.0,
        "exact_ref_raw_rate_sum": 0.0,
        "exact_ref_actual_rate_sum": 0.0,
        "kan_active_spec_steps": 0,
        "kan_fallback_steps": 0,
        "kan_raw_candidates_sum": 0.0,
        "kan_actual_candidates_sum": 0.0,
        "kan_raw_rate_sum": 0.0,
        "kan_actual_rate_sum": 0.0,
    }
    ious: Dict[str, List[float]] = {"exact_ref": [], "kan": []}
    for batch_idx, (pts, _labels) in enumerate(tqdm(loader, desc=f"{dataset_tag} N={args.num_points}", ncols=100)):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        pts = pts.to(device=device, dtype=torch.float32)
        accum["num_samples"] += int(pts.size(0))
        idx_exact, radii_sq_exact = exact_fps_radii_sq(pts, args.K, seed=args.seed)
        radii_sq_kan = predict_full_radii_squared_from_exact(radii_sq_exact, model, cfg, device)
        for prefix, radii in (("exact_ref", radii_sq_exact), ("kan", radii_sq_kan)):
            idx_f, st = segfilter_stats_remaining_denominator(
                pts, args.K, radii, args.segment_len, s0, seed=args.seed
            )
            ious[prefix].extend(set_iou(idx_exact, idx_f))
            active = int(st["active_spec_steps"])
            accum[f"{prefix}_active_spec_steps"] += active
            accum[f"{prefix}_fallback_steps"] += int(st["fallback_steps"])
            accum[f"{prefix}_raw_candidates_sum"] += st["raw_candidates_mean"] * active
            accum[f"{prefix}_actual_candidates_sum"] += st["actual_candidates_mean"] * active
            accum[f"{prefix}_raw_rate_sum"] += st["raw_candidate_rate_remaining"] * active
            accum[f"{prefix}_actual_rate_sum"] += st["actual_candidate_rate_remaining"] * active

    rows = []
    for prefix in ("exact_ref", "kan"):
        active = max(int(accum[f"{prefix}_active_spec_steps"]), 1)
        rows.append({
            "dataset": dataset_tag,
            "partition": args.partition,
            "num_points": int(args.num_points),
            "branch": prefix,
            "num_samples": int(accum["num_samples"]),
            "K": int(args.K),
            "segment_len": int(args.segment_len),
            "seg_filter_start_t": int(s0),
            "active_spec_steps": active,
            "fallback_steps": int(accum[f"{prefix}_fallback_steps"]),
            "fallback_rate": float(accum[f"{prefix}_fallback_steps"] / active),
            "raw_candidates_mean": float(accum[f"{prefix}_raw_candidates_sum"] / active),
            "actual_candidates_mean": float(accum[f"{prefix}_actual_candidates_sum"] / active),
            "raw_candidate_rate_remaining": float(accum[f"{prefix}_raw_rate_sum"] / active),
            "actual_candidate_rate_remaining": float(accum[f"{prefix}_actual_rate_sum"] / active),
            "fps_iou_mean": mean(ious[prefix]),
            "kan_save_dir": os.path.abspath(kan_save_dir),
        })
    return {"dataset": dataset_tag, "rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corrected speculative FPS fallback stats.")
    p.add_argument("--dataset", required=True, choices=("modelnet40", "shapenet_part", "scanobjectnn"))
    p.add_argument("--partition", default="test", choices=("train", "test"))
    p.add_argument("--preset", default="voxelnet", choices=("none", "voxelnet"))
    p.add_argument("--data_root", default="")
    p.add_argument("--shapenet_root", default="")
    p.add_argument("--shapenet_normalize", action="store_true")
    p.add_argument("--scanobjectnn_train_h5", default="")
    p.add_argument("--scanobjectnn_test_h5", default="")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--seed", choices=("first", "max_norm"), default="first")
    p.add_argument("--device", default="cuda")
    p.add_argument("--kan_save_dir", default="auto")
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/specfps_corrected_multidataset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    result = evaluate(args)
    stem = f"corrected_{result['dataset']}_{args.partition}_N{args.num_points}_K{args.K}_seg{args.segment_len}"
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
