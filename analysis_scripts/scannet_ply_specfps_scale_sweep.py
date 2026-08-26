#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MN40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for path in (MN40_DIR, os.path.dirname(MN40_DIR), os.path.dirname(__file__)):
    if path not in sys.path:
        sys.path.insert(0, path)

from scanobjnn_specfps_scale_sweep import (  # type: ignore
    collate_points,
    exact_fps_radii_sq,
    mean,
    segfilter_stats_remaining_denominator,
    set_iou,
)
from spec_kan_fps_opt_kan import (  # type: ignore
    kan_prefix_len_from_K,
    load_kan_radii_model,
    predict_full_radii_squared_from_exact,
)


def pc_normalize_np(pc: np.ndarray) -> np.ndarray:
    pc = pc.astype(np.float32, copy=False)
    pc = pc - pc.mean(axis=0, keepdims=True)
    radius = np.sqrt((pc * pc).sum(axis=1)).max()
    return pc / (radius + 1e-8)


def read_scannet_xyz_ply(path: str) -> np.ndarray:
    vertex_count = None
    header_len = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Malformed PLY header: {path}")
            header_len += len(line)
            s = line.decode("latin1").strip()
            if s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
            if s == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"No vertex count in PLY: {path}")
    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("alpha", "u1"),
    ])
    with open(path, "rb") as f:
        f.seek(header_len)
        arr = np.fromfile(f, dtype=dtype, count=vertex_count)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32, copy=False)


class ScanNetPlyDataset(Dataset):
    def __init__(self, paths: List[str], num_points: int, seed: int = 123, normalize: bool = True):
        self.paths = list(paths)
        self.num_points = int(num_points)
        self.seed = int(seed)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        xyz = read_scannet_xyz_ply(self.paths[index])
        if self.normalize:
            xyz = pc_normalize_np(xyz)
        n = xyz.shape[0]
        rng = np.random.default_rng(self.seed + index * 1009 + self.num_points)
        replace = n < self.num_points
        choice = rng.choice(n, size=self.num_points, replace=replace)
        pts = xyz[choice].astype(np.float32, copy=True)
        return torch.from_numpy(pts), 0


@torch.no_grad()
def evaluate_n(args, paths: List[str], npoints: int, model, cfg, device):
    ds = ScanNetPlyDataset(paths, npoints, seed=args.sample_seed, normalize=not args.no_normalize)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_points)
    s0 = kan_prefix_len_from_K(args.K, args.prefix_ratio)
    accum = {k: 0.0 for k in [
        "num_samples",
        "exact_ref_active_spec_steps", "exact_ref_fallback_steps", "exact_ref_raw_candidates_sum",
        "exact_ref_actual_candidates_sum", "exact_ref_raw_rate_sum", "exact_ref_actual_rate_sum",
        "kan_active_spec_steps", "kan_fallback_steps", "kan_raw_candidates_sum",
        "kan_actual_candidates_sum", "kan_raw_rate_sum", "kan_actual_rate_sum",
    ]}
    ious = {"exact_ref": [], "kan": []}
    for pts, _labels in tqdm(loader, desc=f"ScanNet PLY N={npoints}", ncols=100):
        pts = pts.to(device=device, dtype=torch.float32)
        bsz = int(pts.size(0))
        accum["num_samples"] += bsz
        idx_exact, radii_sq_exact = exact_fps_radii_sq(pts, args.K, seed=args.seed)
        radii_sq_kan = predict_full_radii_squared_from_exact(radii_sq_exact, model, cfg, device)
        for prefix, radii in (("exact_ref", radii_sq_exact), ("kan", radii_sq_kan)):
            idx_f, st = segfilter_stats_remaining_denominator(pts, args.K, radii, args.segment_len, s0, seed=args.seed)
            ious[prefix].extend(set_iou(idx_exact, idx_f))
            active = st["active_spec_steps"]
            accum[f"{prefix}_active_spec_steps"] += active
            accum[f"{prefix}_fallback_steps"] += st["fallback_steps"]
            accum[f"{prefix}_raw_candidates_sum"] += st["raw_candidates_mean"] * active
            accum[f"{prefix}_actual_candidates_sum"] += st["actual_candidates_mean"] * active
            accum[f"{prefix}_raw_rate_sum"] += st["raw_candidate_rate_remaining"] * active
            accum[f"{prefix}_actual_rate_sum"] += st["actual_candidate_rate_remaining"] * active
    rows = []
    for prefix in ("exact_ref", "kan"):
        active = max(int(accum[f"{prefix}_active_spec_steps"]), 1)
        rows.append({
            "dataset": "scannet_ply_real",
            "num_points": int(npoints),
            "branch": prefix,
            "num_samples": int(accum["num_samples"]),
            "K": int(args.K),
            "segment_len": int(args.segment_len),
            "seg_filter_start_t": int(s0),
            "normalized": not args.no_normalize,
            "active_spec_steps": active,
            "fallback_steps": int(accum[f"{prefix}_fallback_steps"]),
            "fallback_rate": float(accum[f"{prefix}_fallback_steps"] / active),
            "raw_candidates_mean": float(accum[f"{prefix}_raw_candidates_sum"] / active),
            "actual_candidates_mean": float(accum[f"{prefix}_actual_candidates_sum"] / active),
            "raw_candidate_rate_remaining": float(accum[f"{prefix}_raw_rate_sum"] / active),
            "actual_candidate_rate_remaining": float(accum[f"{prefix}_actual_rate_sum"] / active),
            "fps_iou_mean": mean(ious[prefix]),
        })
    return {"num_points": npoints, "rows": rows}


def parse_args():
    p = argparse.ArgumentParser(description="Speculative FPS fallback scale sweep on real ScanNet PLY scenes.")
    p.add_argument("--ply_root", default="/workspace/VoxelNet/scannet/scans_test")
    p.add_argument("--pattern", default="*/scene*_vh_clean_2.ply")
    p.add_argument("--kan_save_dir", default="/workspace/PointKAN-pytorch/PointKAN-elite/classification_ModelNet40/runs/fps_radii_oneclick_all/fps_radii_kan_K512_scanobjectnn_pr0.10")
    p.add_argument("--n_list", default="1024,2048,4096,8192,16384,32768")
    p.add_argument("--max_files", type=int, default=64)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", choices=("first", "max_norm"), default="first")
    p.add_argument("--sample_seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--out_dir", default="/workspace/PointKAN-pytorch/my_progressive_out/scannet_ply_specfps_scale_sweep")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.ply_root, args.pattern)))
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise FileNotFoundError(f"No PLY files under {args.ply_root}/{args.pattern}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_kan_radii_model(args.kan_save_dir, device)
    cfg.prefix_ratio = float(args.prefix_ratio)
    n_values = [int(x) for x in args.n_list.split(",") if x.strip()]
    results = {
        "ply_root": os.path.abspath(args.ply_root),
        "num_files": len(paths),
        "paths_head": paths[:5],
        "kan_save_dir": os.path.abspath(args.kan_save_dir),
        "n_list": n_values,
        "note": "Uses real ScanNet PLY scene points sampled without replacement when possible; rates use remaining unselected points as denominator.",
        "runs": [],
    }
    rows = []
    for n in n_values:
        res = evaluate_n(args, paths, n, model, cfg, device)
        results["runs"].append(res)
        rows.extend(res["rows"])
    stem = f"scannet_ply_specfps_scale_K{args.K}_seg{args.segment_len}_files{len(paths)}"
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
