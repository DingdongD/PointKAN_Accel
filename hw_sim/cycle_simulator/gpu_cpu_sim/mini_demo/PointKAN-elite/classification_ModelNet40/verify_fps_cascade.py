#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 PointKANElite 各层 GAM 的 FPS 索引层级关系：后面层的 FPS 采样点是否为前一层的子集。

结论预期：第 L 层 FPS 的输入是第 L-1 层的输出点，因此将第 L 层索引映射回原始 1024 空间后，
应为第 L-1 层映射到 1024 空间索引的子集。

用法：
    python verify_fps_cascade.py
    python verify_fps_cascade.py --num_samples 20 --seed 42
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# 项目根目录（PointKAN-pytorch），含 pointnet2_ops_lib
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "pointnet2_ops_lib")):
    if d not in sys.path:
        sys.path.insert(0, d)

from pointnet2_ops import pointnet2_utils
from models.pointkanElite import index_points, pointKANElite


def get_stage_groups():
    """与 Model 构造一致：reducers=[2,2,2,2], points=1024 -> [512, 256, 128, 64]"""
    points = 1024
    reducers = [2, 2, 2, 2]
    anchor_points = points
    groups_list = []
    for r in reducers:
        anchor_points = anchor_points // r
        groups_list.append(anchor_points)
    return groups_list


def run_cascade_fps(xyz, groups_list):
    """
    模拟 Model 中四层 GAM 的 FPS 级联。
    xyz: [B, N, 3]
    groups_list: [512, 256, 128, 64]
    返回: list of (fps_idx_in_current_space, fps_idx_in_1024_space)
          each fps_idx: [B, G]
    """
    B, N, _ = xyz.shape
    device = xyz.device
    # 当前层的点（在原始 1024 中的索引）
    # 第 0 层：当前点集 = 0..N-1，用 idx 表示即 arange
    current_idx_in_1024 = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)  # [B, N]
    results = []  # 每层: (fps_idx 当前空间, fps_idx 映射到 1024)

    for g in groups_list:
        # 当前 xyz 与当前点集在 1024 中的索引
        xyz_current = index_points(xyz, current_idx_in_1024)  # [B, num_current, 3]
        num_current = xyz_current.shape[1]
        fps_idx_current = pointnet2_utils.furthest_point_sample(xyz_current, g).long()  # [B, g]
        # 映射到 1024 空间: fps_idx_in_1024[b, j] = current_idx_in_1024[b, fps_idx_current[b, j]]
        fps_idx_in_1024 = torch.gather(current_idx_in_1024, 1, fps_idx_current)  # [B, g]
        results.append((fps_idx_current, fps_idx_in_1024))
        # 下一层的“当前点集”就是本层 FPS 得到的 g 个点
        current_idx_in_1024 = fps_idx_in_1024  # [B, g]

    return results


def verify_subset_relationship(results, verbose=True):
    """检查：层 L 在 1024 空间的索引集合 是 层 L-1 在 1024 空间索引集合 的子集。"""
    all_ok = True
    for L in range(1, len(results)):
        _, prev_1024 = results[L - 1]   # [B, G_prev]
        _, curr_1024 = results[L]       # [B, G_curr]
        B = prev_1024.shape[0]
        print(f"prev_1024: {prev_1024} with shape {prev_1024.shape}")
        print(f"curr_1024: {curr_1024} with shape {curr_1024.shape}")
        layer_ok = True
        for b in range(B):
            set_prev = set(prev_1024[b].cpu().tolist())
            set_curr = set(curr_1024[b].cpu().tolist())
            if not set_curr.issubset(set_prev):
                all_ok = False
                layer_ok = False
                if verbose:
                    print(f"  [FAIL] layer {L} batch {b}: set(curr) not subset of set(prev). "
                          f"len(prev)={len(set_prev)} len(curr)={len(set_curr)} "
                          f"len(curr-prev)={len(set_curr - set_prev)}")
        if layer_ok and verbose and B > 0:
            print(f"  层 {L} 的 FPS 索引（映射到 1024）是层 {L-1} 的子集 ✓")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="验证 GAM 各层 FPS 索引的层级子集关系")
    parser.add_argument("--num_samples", type=int, default=10, help="验证的 batch 数（每 batch 一个点云）")
    parser.add_argument("--batch_size", type=int, default=4, help="每个 batch 的样本数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    groups_list = get_stage_groups()
    print(f"Stage groups (与 pointKANElite 一致): {groups_list}")

    all_pass = True
    for sample in range(args.num_samples):
        xyz = torch.rand(args.batch_size, 1024, 3, device=args.device)
        results = run_cascade_fps(xyz, groups_list)
        verbose = sample == 0  # 仅第一个 sample 打印每层子集结论
        ok = verify_subset_relationship(results, verbose=verbose)
        if not ok:
            all_pass = False
        if args.num_samples <= 5 or not ok:
            print(f"Sample {sample}: {'PASS' if ok else 'FAIL'}")

    print()
    if all_pass:
        print("结论: 各层 FPS 在 1024 空间上的索引均满足「后层是前层的子集」，可安全用「首层 FPS + 后续索引」优化。")
    else:
        print("结论: 存在不满足子集关系的情况（不应发生），请检查。")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
