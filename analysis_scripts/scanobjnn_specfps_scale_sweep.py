#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


class ScanObjectNNScaled(Dataset):
    def __init__(
        self,
        h5_path: str,
        num_points: int,
        max_samples: int = 0,
        sample_mode: str = "head",
        jitter_std: float = 1.0e-4,
        seed: int = 123,
    ):
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:].astype(np.float32)
            label = f["label"][:].astype(np.int64).reshape(-1)
        if max_samples > 0:
            if sample_mode == "head":
                pick = np.arange(min(max_samples, len(label)))
            elif sample_mode == "random":
                rng = np.random.default_rng(seed)
                pick = rng.choice(len(label), size=min(max_samples, len(label)), replace=False)
                pick.sort()
            elif sample_mode == "class_balanced":
                rng = np.random.default_rng(seed)
                classes = np.unique(label)
                quota = int(np.ceil(max_samples / max(len(classes), 1)))
                chunks = []
                for c in classes:
                    idx = np.flatnonzero(label == c)
                    take = min(quota, len(idx))
                    chunks.append(rng.choice(idx, size=take, replace=False))
                pick = np.concatenate(chunks)
                if len(pick) > max_samples:
                    pick = rng.choice(pick, size=max_samples, replace=False)
                pick.sort()
            else:
                raise ValueError(f"Unknown sample_mode: {sample_mode}")
            data = data[pick]
            label = label[pick]
        self.data = data
        self.label = label
        self.num_points = int(num_points)
        self.base_points = int(self.data.shape[1])
        self.jitter_std = float(jitter_std)
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        base = self.data[index][:, :3]
        n = self.num_points
        if n <= self.base_points:
            pts = base[:n].copy()
        else:
            reps = n // self.base_points
            rem = n % self.base_points
            chunks = [base.copy() for _ in range(reps)]
            if rem:
                chunks.append(base[:rem].copy())
            pts = np.concatenate(chunks, axis=0)
            if self.jitter_std > 0:
                rng = np.random.default_rng(self.seed + index * 1009 + n)
                noise = rng.normal(0.0, self.jitter_std, size=pts.shape).astype(np.float32)
                noise[: self.base_points] = 0.0
                pts = pts + noise
        return torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float32)), int(self.label[index])


def collate_points(batch):
    pts, labels = zip(*batch)
    return torch.stack(list(pts), dim=0), torch.as_tensor(labels, dtype=torch.long)


@torch.no_grad()
def exact_fps_radii_sq(X: torch.Tensor, K: int, seed: str = "first") -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = X.shape
    K = min(int(K), int(N))
    device, dtype = X.device, X.dtype
    x2 = (X * X).sum(dim=-1)
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=dtype, device=device)
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    selected = torch.zeros((B, N), dtype=torch.bool, device=device)
    batch = torch.arange(B, device=device)
    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unsupported seed for deterministic sweep: {seed}")

    for t in range(K):
        if t > 0:
            cand = min_dist.masked_fill(selected, -float("inf"))
            farthest = torch.argmax(cand, dim=1)
        idx[:, t] = farthest
        selected[batch, farthest] = True
        C = X[batch, farthest]
        c2 = (C * C).sum(dim=-1, keepdim=True)
        d2 = (x2 + c2 - 2.0 * (X @ C.unsqueeze(-1)).squeeze(-1)).clamp_min_(0.0)
        min_dist = torch.minimum(min_dist, d2)
        min_dist[selected] = 0.0
        radii[:, t], _ = torch.max(min_dist, dim=1)
    return idx, radii


@torch.no_grad()
def segfilter_stats_remaining_denominator(
    X: torch.Tensor,
    K: int,
    radii_ref_sq: torch.Tensor,
    segment_len: int,
    seg_filter_start_t: int,
    seed: str = "first",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B, N, _ = X.shape
    K = min(int(K), int(N))
    device, dtype = X.device, X.dtype
    x2 = (X * X).sum(dim=-1)
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    selected = torch.zeros((B, N), dtype=torch.bool, device=device)
    batch = torch.arange(B, device=device)
    L = max(1, int(segment_len))
    s0 = max(1, min(int(seg_filter_start_t), K))
    if seed == "first":
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "max_norm":
        farthest = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unsupported seed for deterministic sweep: {seed}")

    fallback_steps = 0
    active_steps = 0
    raw_count_sum = 0.0
    actual_count_sum = 0.0
    raw_rate_sum = 0.0
    actual_rate_sum = 0.0
    remaining_sum = 0.0

    for t in range(K):
        if t == 0:
            pass
        elif t < s0:
            cand = min_dist.masked_fill(selected, -float("inf"))
            farthest = torch.argmax(cand, dim=1)
        else:
            j = t // L
            seg_end = min((j + 1) * L - 1, K - 1)
            tau = radii_ref_sq[:, seg_end].to(device=device, dtype=dtype)
            valid = ~selected
            mask = (min_dist >= tau.unsqueeze(1)) & valid
            raw_count = mask.sum(dim=1).to(dtype)
            remaining = valid.sum(dim=1).to(dtype).clamp_min(1.0)
            empty = raw_count <= 0
            fallback_steps += int(empty.sum().item())
            active_steps += int(B)
            actual_count = raw_count.clone()
            actual_count[empty] = remaining[empty]
            raw_count_sum += float(raw_count.sum().item())
            actual_count_sum += float(actual_count.sum().item())
            raw_rate_sum += float((raw_count / remaining).sum().item())
            actual_rate_sum += float((actual_count / remaining).sum().item())
            remaining_sum += float(remaining.sum().item())
            if empty.any():
                mask = mask.clone()
                mask[empty] = valid[empty]
            cand = min_dist.masked_fill(~mask, -float("inf"))
            farthest = torch.argmax(cand, dim=1)

        idx[:, t] = farthest
        selected[batch, farthest] = True
        C = X[batch, farthest]
        c2 = (C * C).sum(dim=-1, keepdim=True)
        d2 = (x2 + c2 - 2.0 * (X @ C.unsqueeze(-1)).squeeze(-1)).clamp_min_(0.0)
        min_dist = torch.minimum(min_dist, d2)
        min_dist[selected] = 0.0

    denom = max(active_steps, 1)
    return idx, {
        "active_spec_steps": int(active_steps),
        "fallback_steps": int(fallback_steps),
        "fallback_rate": float(fallback_steps / denom),
        "raw_candidates_mean": float(raw_count_sum / denom),
        "actual_candidates_mean": float(actual_count_sum / denom),
        "raw_candidate_rate_remaining": float(raw_rate_sum / denom),
        "actual_candidate_rate_remaining": float(actual_rate_sum / denom),
        "remaining_points_mean": float(remaining_sum / denom),
    }


def set_iou(idx_a: torch.Tensor, idx_b: torch.Tensor) -> List[float]:
    out: List[float] = []
    for a, b in zip(idx_a.detach().cpu().tolist(), idx_b.detach().cpu().tolist()):
        sa, sb = set(a), set(b)
        out.append(len(sa & sb) / max(len(sa | sb), 1))
    return out


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / max(len(xs), 1))


@torch.no_grad()
def evaluate_n(args: argparse.Namespace, npoints: int, model: torch.nn.Module, cfg, device) -> Dict:
    ds = ScanObjectNNScaled(
        args.test_h5,
        npoints,
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        jitter_std=args.jitter_std,
        seed=args.upsample_seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_points,
    )
    s0 = kan_prefix_len_from_K(args.K, args.prefix_ratio)
    accum: Dict[str, float] = {
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
    ious_exact: List[float] = []
    ious_kan: List[float] = []
    for pts, _labels in tqdm(loader, desc=f"ScanObjectNN N={npoints}", ncols=100):
        pts = pts.to(device=device, dtype=torch.float32)
        bsz = int(pts.size(0))
        accum["num_samples"] += bsz
        idx_exact, radii_sq_exact = exact_fps_radii_sq(pts, args.K, seed=args.seed)
        radii_sq_kan = predict_full_radii_squared_from_exact(radii_sq_exact, model, cfg, device)
        for prefix, radii in (("exact_ref", radii_sq_exact), ("kan", radii_sq_kan)):
            idx_f, st = segfilter_stats_remaining_denominator(
                pts,
                args.K,
                radii,
                args.segment_len,
                s0,
                seed=args.seed,
            )
            if prefix == "exact_ref":
                ious_exact.extend(set_iou(idx_exact, idx_f))
            else:
                ious_kan.extend(set_iou(idx_exact, idx_f))
            accum[f"{prefix}_active_spec_steps"] += st["active_spec_steps"]
            accum[f"{prefix}_fallback_steps"] += st["fallback_steps"]
            accum[f"{prefix}_raw_candidates_sum"] += st["raw_candidates_mean"] * st["active_spec_steps"]
            accum[f"{prefix}_actual_candidates_sum"] += st["actual_candidates_mean"] * st["active_spec_steps"]
            accum[f"{prefix}_raw_rate_sum"] += st["raw_candidate_rate_remaining"] * st["active_spec_steps"]
            accum[f"{prefix}_actual_rate_sum"] += st["actual_candidate_rate_remaining"] * st["active_spec_steps"]

    rows = []
    for prefix, ious in (("exact_ref", ious_exact), ("kan", ious_kan)):
        active = max(int(accum[f"{prefix}_active_spec_steps"]), 1)
        rows.append({
            "dataset": "scanobjectnn",
            "num_points": npoints,
            "base_points": ds.base_points,
            "synthetic_upsample": bool(npoints > ds.base_points),
            "jitter_std": args.jitter_std if npoints > ds.base_points else 0.0,
            "branch": prefix,
            "num_samples": int(accum["num_samples"]),
            "sample_mode": args.sample_mode,
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
            "fps_iou_mean": mean(ious),
        })
    return {"num_points": npoints, "rows": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ScanObjectNN speculative FPS fallback scale sweep.")
    p.add_argument("--test_h5", default="/workspace/VoxelNet/scannet_kan/main_split/test_objectdataset_augmentedrot_scale75.h5")
    p.add_argument("--kan_save_dir", default="/workspace/PointKAN-pytorch/PointKAN-elite/classification_ModelNet40/runs/fps_radii_oneclick_all/fps_radii_kan_K512_scanobjectnn_pr0.10")
    p.add_argument("--n_list", default="1024,2048,4096,8192,16384,32768")
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--sample_mode", choices=("head", "random", "class_balanced"), default="head")
    p.add_argument("--seed", choices=("first", "max_norm"), default="first")
    p.add_argument("--device", default="cuda")
    p.add_argument("--jitter_std", type=float, default=1.0e-4)
    p.add_argument("--upsample_seed", type=int, default=123)
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/scanobjnn_specfps_scale_sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_kan_radii_model(args.kan_save_dir, device)
    cfg.prefix_ratio = float(args.prefix_ratio)
    n_values = [int(x) for x in args.n_list.split(",") if x.strip()]
    results = {
        "test_h5": os.path.abspath(args.test_h5),
        "kan_save_dir": os.path.abspath(args.kan_save_dir),
        "n_list": n_values,
        "max_samples": int(args.max_samples),
        "sample_mode": args.sample_mode,
        "note": "N > base_points uses deterministic tiling plus Gaussian jitter; rates use remaining unselected points as denominator.",
        "runs": [],
    }
    rows = []
    for n in n_values:
        res = evaluate_n(args, n, model, cfg, device)
        results["runs"].append(res)
        rows.extend(res["rows"])
    stem = f"scanobjnn_specfps_scale_K{args.K}_seg{args.segment_len}_samples{args.max_samples}"
    json_path = os.path.join(args.out_dir, stem + ".json")
    csv_path = os.path.join(args.out_dir, stem + ".csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("WROTE", json_path)
    print("WROTE", csv_path)


if __name__ == "__main__":
    main()
