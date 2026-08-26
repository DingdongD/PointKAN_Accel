#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from spec_kan_fps_opt import ModelNet40Wrapper, collate_points, fps_adaptive_argmax  # type: ignore
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
    predict_full_radii_squared_from_fps_prefix_points,
)


@torch.no_grad()
def _exact_preselect_max_curve(points: torch.Tensor, idx_exact: torch.Tensor, K: int) -> torch.Tensor:
    """Return M_t=max_i min_dist_i immediately before selecting FPS step t."""
    bsz, npts, _ = points.shape
    device, dtype = points.device, points.dtype
    x2 = (points * points).sum(dim=-1)
    min_dist = torch.full((bsz, npts), float("inf"), dtype=dtype, device=device)
    out = torch.zeros((bsz, K), dtype=dtype, device=device)
    batch = torch.arange(bsz, device=device)

    for t in range(K):
        out[:, t], _ = min_dist.max(dim=1)
        center = points[batch, idx_exact[:, t]]
        c2 = (center * center).sum(dim=-1, keepdim=True)
        dot = (points @ center.unsqueeze(-1)).squeeze(-1)
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0)
        torch.minimum(min_dist, d2, out=min_dist)
        min_dist[batch, idx_exact[:, t]] = 0.0
    return out


def _threshold_curve_for_steps(radii_ref_sq: torch.Tensor, K: int, segment_len: int) -> torch.Tensor:
    steps = torch.arange(K, device=radii_ref_sq.device)
    seg_end = ((steps // max(1, int(segment_len)) + 1) * max(1, int(segment_len)) - 1).clamp(max=K - 1)
    return radii_ref_sq[:, seg_end]


@torch.no_grad()
def verify_loader(
    loader: DataLoader,
    *,
    K: int,
    segment_len: int,
    seed: str,
    device: torch.device,
    threshold_source: str,
    kan_save_dir: str,
    kan_input: str,
    prefix_ratio: float,
    defer_seg_filter: bool,
    tolerance: float,
) -> Dict:
    model_kan = None
    cfg_kan = None
    if threshold_source == "kan":
        model_kan, cfg_kan = load_kan_radii_model(kan_save_dir, device)
        cfg_kan.prefix_ratio = prefix_ratio

    total_samples = 0
    total_steps = 0
    threshold_violations = 0
    idx_mismatches = 0
    worst_margin = float("inf")
    fallback_steps = 0

    for points, _labels in tqdm(loader, desc="verify speculative FPS", ncols=100):
        points = points.to(device, dtype=torch.float32)
        bsz = points.size(0)
        k_eff = min(int(K), points.size(1))
        idx_exact, radii_exact_sq, _, _ = fps_adaptive_argmax(points, k_eff, seed=seed, mode="exact")

        if threshold_source == "exact":
            radii_ref_sq = radii_exact_sq
        elif kan_input == "fps_points":
            radii_ref_sq, _ = predict_full_radii_squared_from_fps_prefix_points(
                points, idx_exact, k_eff, model_kan, cfg_kan, device
            )
        else:
            radii_ref_sq = predict_full_radii_squared_from_exact(radii_exact_sq, model_kan, cfg_kan, device)

        seg_filter_start_t = kan_prefix_len_from_K(k_eff, prefix_ratio) if defer_seg_filter else 1
        idx_spec, _, _, stats = fps_adaptive_argmax(
            points,
            k_eff,
            seed=seed,
            mode="seg_filter",
            radii_ref=radii_ref_sq,
            segment_len=segment_len,
        )
        if defer_seg_filter and seg_filter_start_t > 1:
            # fps_adaptive_argmax has no defer knob, so use the formal threshold check for proof,
            # and count index mismatches only on its native all-seg-filter behavior.
            compare_start = 1
        else:
            compare_start = seg_filter_start_t

        preselect_max = _exact_preselect_max_curve(points, idx_exact, k_eff)
        tau_by_step = _threshold_curve_for_steps(radii_ref_sq, k_eff, segment_len)
        active = torch.arange(k_eff, device=device).view(1, -1) >= int(seg_filter_start_t)
        margin = preselect_max - tau_by_step
        violations = (margin < -float(tolerance)) & active

        idx_diff = (idx_exact[:, compare_start:] != idx_spec[:, compare_start:]).sum().item()
        idx_mismatches += int(idx_diff)
        threshold_violations += int(violations.sum().item())
        fallback_steps += int(stats["used_fallback"][:, compare_start:].sum().item())
        total_samples += int(bsz)
        total_steps += int(active.sum().item() * bsz)
        if active.any():
            worst_margin = min(worst_margin, float(margin[active.expand_as(margin)].min().item()))

    exact = idx_mismatches == 0
    return {
        "threshold_source": threshold_source,
        "kan_save_dir": kan_save_dir if threshold_source == "kan" else None,
        "kan_input": kan_input if threshold_source == "kan" else None,
        "K": int(K),
        "segment_len": int(segment_len),
        "seed": seed,
        "prefix_ratio": float(prefix_ratio),
        "defer_seg_filter": bool(defer_seg_filter),
        "seg_filter_start_t": int(kan_prefix_len_from_K(int(K), prefix_ratio) if defer_seg_filter else 1),
        "num_samples": int(total_samples),
        "checked_active_steps": int(total_steps),
        "threshold_violations": int(threshold_violations),
        "idx_mismatches_native_seg_filter": int(idx_mismatches),
        "fallback_steps_native_seg_filter": int(fallback_steps),
        "worst_margin_preselect_minus_tau": float(worst_margin),
        "condition_tau_le_exact_preselect_max_holds": bool(threshold_violations == 0),
        "fallback_free_filtering_for_checked_thresholds": bool(threshold_violations == 0),
        "proves_exact_fps_for_checked_thresholds": bool(exact),
        "note": (
            "Point-wise threshold filtering preserves exact FPS for any tau when empty_fallback=global: "
            "if tau <= max(min_dist), the global argmax is still in the candidate set; if tau > max(min_dist), "
            "the candidate set is empty and the implementation falls back to global argmax. "
            "The tau<=max condition is therefore a fallback-free efficiency condition, not a correctness requirement."
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify exactness condition for speculative/segment-filter FPS.")
    p.add_argument("--root", default="./data")
    p.add_argument("--partition", default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_batches", type=int, default=4)
    p.add_argument("--seed", default="first")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threshold_source", choices=("exact", "kan"), default="exact")
    p.add_argument("--kan_save_dir", default="")
    p.add_argument("--kan_input", choices=("radii", "fps_points"), default="radii")
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--defer_seg_filter", action="store_true")
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--out_json", default="/workspace/PointKAN-pytorch/my_progressive_out/speculative_fps_exactness_verify.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = ModelNet40Wrapper(args.partition, args.num_points, root=args.root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=collate_points,
    )
    if args.max_batches > 0:
        loader = list(loader)[: args.max_batches]
    result = verify_loader(
        loader,
        K=args.K,
        segment_len=args.segment_len,
        seed=args.seed,
        device=device,
        threshold_source=args.threshold_source,
        kan_save_dir=args.kan_save_dir,
        kan_input=args.kan_input,
        prefix_ratio=args.prefix_ratio,
        defer_seg_filter=args.defer_seg_filter,
        tolerance=args.tolerance,
    )
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("WROTE", args.out_json)


if __name__ == "__main__":
    main()
