#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化后做 FPS + Vanilla KNN，与全精度 vanilla 对比（KNN / FPS 指标 + 坐标 MSE）。

量化方案由 datasets.yaml 的 quant.scheme 指定：
  - per_cloud_minmax：逐帧 AABB，每轴 2^bits 档（原默认）
  - symmetric_neg1_1：假定预处理后 xyz∈[-1,1]，对称整数、zero_point=0，
    qmax=2^(bits-1)-1，scale=1/qmax（bits=8 即 INT8 常用 scale=1/127）
  - global_per_axis：固定每轴 [lo,hi]，按 bits 均匀量化；
    centered=true 时先减轴心中点再对称量化（zp=0）；false 时仿射到 [0,2^bits-1]。
    可设 bounds_from_stats: true 或 json 路径，从 stats_dataset_xyz_bounds.py 输出读 xyz_min/max。
  - quant.bits_list：每数据集单独 bit 档位列表，覆盖命令行 --bits。

配置：datasets.yaml + 命令行。
依赖：spec_kan_fps、FPS_KNN_Ablation 数据加载。

断点续跑：``--resume`` + 同一 ``--out_json``；跳过 JSON 中已完整写入的 (数据集, bit)；每完成一个 bit **原子写入**，进程被 kill 后不易整文件丢失。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BENCH_DIR, "..", "..", ".."))
ABLATION_DIR = os.path.join(REPO_ROOT, "FPS_KNN_Ablation")
MODELNET40_DIR = os.path.join(REPO_ROOT, "PointKAN-elite", "classification_ModelNet40")
for d in (BENCH_DIR, ABLATION_DIR, REPO_ROOT, MODELNET40_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from spec_kan_fps import (  # noqa: E402
    collate_points,
    fps_with_knn_vanilla,
    knn_neighbor_set_iou,
)

import run_knn_coarse_fine_eval_multidataset as mds  # noqa: E402

from run_knn_coarse_fine_eval_kitti import KITTIPointCloudWrapper  # noqa: E402
from run_knn_coarse_fine_eval_waymo import WaymoPointCloudWrapper  # noqa: E402
from run_knn_coarse_fine_eval_nuscenes import NuScenesPointCloudWrapper  # noqa: E402


def quantize_dequantize_xyz(X: torch.Tensor, bits_per_axis: int) -> torch.Tensor:
    """
    X: (B, N, 3) 逐 batch 轴对齐 bbox，每轴独立量化到 2^b 档再反量化回 float。
    """
    if bits_per_axis < 1:
        raise ValueError("bits_per_axis must be >= 1")
    B, _, _ = X.shape
    levels = float((1 << bits_per_axis) - 1)
    out = torch.empty_like(X)
    for b in range(B):
        x = X[b]
        lo = x.min(dim=0).values
        hi = x.max(dim=0).values
        span = (hi - lo).clamp_min(1e-8)
        norm = (x - lo) / span * levels
        q = torch.round(norm).clamp(0.0, levels)
        out[b] = lo + (q / levels) * span
    return out


def quant_symmetric_neg1_1(X: torch.Tensor, bits: int, clip_to_unit: bool = True) -> torch.Tensor:
    """
    假定 xyz 落在 [-1, 1]（可选 clip）；对称量化 zero_point=0：
    qmax = 2^(bits-1)-1, scale = 1/qmax, q = round(x/scale) clamp ±qmax, x' = q*scale。
    bits=8 → qmax=127, scale=1/127（INT8 对称幅度常用设定）。
    """
    if bits < 2:
        raise ValueError("symmetric_neg1_1 需要 bits >= 2（对称幅度至少 1 档）")
    qmax = float((1 << (bits - 1)) - 1)
    scale = 1.0 / qmax
    x = X
    if clip_to_unit:
        x = x.clamp(-1.0, 1.0)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale


def quant_global_per_axis(X: torch.Tensor, bits: int, cfg: Dict[str, Any]) -> torch.Tensor:
    """
    每轴独立全局 [lo,hi]（与 ROI / 标定一致），按 bits 均匀反量化回 float。
    centered=False：q ∈ [0, 2^bits-1]，仿射覆盖 [lo,hi]（zero 由 lo 吸收）。
    centered=True：先 x' = x - (lo+hi)/2，对称 q∈[±(2^(bits-1)-1)]，scale = R/qmax，R=(hi-lo)/2。
    """
    if bits < 1:
        raise ValueError("global_per_axis 需要 bits >= 1")
    device, dtype = X.device, X.dtype
    lo = torch.tensor(
        [float(cfg["x_range"][0]), float(cfg["y_range"][0]), float(cfg["z_range"][0])],
        device=device,
        dtype=dtype,
    )
    hi = torch.tensor(
        [float(cfg["x_range"][1]), float(cfg["y_range"][1]), float(cfg["z_range"][1])],
        device=device,
        dtype=dtype,
    )
    centered = bool(cfg.get("centered", False))
    out = torch.empty_like(X)
    if centered:
        if bits < 2:
            raise ValueError("global_per_axis + centered 需要 bits >= 2")
        qmax = float((1 << (bits - 1)) - 1)
        c = (lo + hi) / 2.0
        R = (hi - lo) / 2.0
        R = R.clamp_min(1e-8)
        scale = R / qmax
        for d in range(3):
            xc = X[..., d] - c[d]
            q = torch.round(xc / scale[d]).clamp(-qmax, qmax)
            out[..., d] = q * scale[d] + c[d]
    else:
        L = float((1 << bits) - 1)
        span = (hi - lo).clamp_min(1e-8)
        for d in range(3):
            t = (X[..., d] - lo[d]) / span[d] * L
            q = torch.round(t).clamp(0.0, L)
            out[..., d] = lo[d] + (q / L) * span[d]
    return out


def resolve_quant_cfg(global_defaults: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
    gq = dict(global_defaults.get("quant") or {})
    eq = entry.get("quant") or {}
    if isinstance(eq, dict):
        gq.update(eq)
    if not gq.get("scheme"):
        gq["scheme"] = "per_cloud_minmax"
    return gq


def _stats_json_path(bfs: Any, global_defaults: Dict[str, Any], bench_dir: str) -> str:
    """bounds_from_stats: True | 相对/绝对路径 → 绝对路径。"""
    gq = global_defaults.get("quant") or {}
    default_name = str(gq.get("stats_json") or "dataset_xyz_bounds.json")
    if bfs is True:
        p = default_name
    else:
        p = str(bfs)
    return p if os.path.isabs(p) else os.path.join(bench_dir, p)


def finalize_quant_cfg(
    quant_cfg: Dict[str, Any],
    global_defaults: Dict[str, Any],
    dataset_key: str,
    bench_dir: str,
) -> Dict[str, Any]:
    """
    若 quant.bounds_from_stats 已设置且 scheme 为 global_per_axis：
    从 JSON（stats_dataset_xyz_bounds.py 输出）读取 datasets[stats_key].xyz_min/max，写入 x/y/z_range。
    stats_key 默认等于 yaml 数据集 key。
    """
    qc = dict(quant_cfg)
    bfs = qc.get("bounds_from_stats")
    if bfs is None or bfs is False:
        return qc
    if str(qc.get("scheme")) != "global_per_axis":
        print(f"[warn] {dataset_key}: bounds_from_stats 仅用于 scheme=global_per_axis，已忽略")
        qc.pop("bounds_from_stats", None)
        return qc

    path = _stats_json_path(bfs, global_defaults, bench_dir)
    stats_key = str(qc.get("stats_key") or dataset_key)
    with open(path, "r", encoding="utf-8") as f:
        stats_root = json.load(f)
    block = (stats_root.get("datasets") or {}).get(stats_key)
    if not block:
        raise KeyError(f"统计 JSON 中无 datasets[{stats_key}]，请检查 stats_key 或重新运行 stats_dataset_xyz_bounds.py: {path}")
    lo = block["xyz_min"]
    hi = block["xyz_max"]
    qc["x_range"] = [float(lo[0]), float(hi[0])]
    qc["y_range"] = [float(lo[1]), float(hi[1])]
    qc["z_range"] = [float(lo[2]), float(hi[2])]
    qc["stats_bounds_json"] = os.path.abspath(path)
    qc["stats_key_resolved"] = stats_key
    return qc


def dataset_bits_list(cli_bits: List[int], quant_cfg: Dict[str, Any]) -> List[int]:
    """quant.bits_list 存在则优先，否则用命令行解析结果。"""
    raw = quant_cfg.get("bits_list")
    if raw is None:
        return list(cli_bits)
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    raise TypeError(f"quant.bits_list 须为列表，当前: {type(raw)}")


def apply_quant_dequant(X: torch.Tensor, bits: int, quant_cfg: Dict[str, Any]) -> torch.Tensor:
    scheme = str(quant_cfg.get("scheme", "per_cloud_minmax"))
    if scheme == "per_cloud_minmax":
        return quantize_dequantize_xyz(X, bits)
    if scheme == "symmetric_neg1_1":
        return quant_symmetric_neg1_1(X, bits, clip_to_unit=bool(quant_cfg.get("clip_to_unit", True)))
    if scheme == "global_per_axis":
        return quant_global_per_axis(X, bits, quant_cfg)
    raise ValueError(f"未知 quant.scheme: {scheme}")


def quant_cfg_for_json(quant_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """可 JSON 序列化的量化配置副本（去掉 YAML 里的 True 占位）。"""
    out = dict(quant_cfg)
    out.pop("bounds_from_stats", None)
    return out


def effective_bits_list(bits_list: List[int], quant_cfg: Dict[str, Any]) -> List[int]:
    """按 scheme 去掉不合法的 bits（对称类需 bits>=2）。"""
    sch = str(quant_cfg.get("scheme", "per_cloud_minmax"))
    out: List[int] = []
    for b in bits_list:
        if b < 1:
            continue
        if sch == "symmetric_neg1_1" and b < 2:
            continue
        if sch == "global_per_axis" and bool(quant_cfg.get("centered")) and b < 2:
            continue
        out.append(b)
    return out


def fps_index_set_iou(idx_a: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    """idx_* : (B, K) 下标。返回每 batch 一个 IoU（将每行视为集合）。"""
    B, K = idx_a.shape
    device = idx_a.device
    ious = []
    for b in range(B):
        sa = set(idx_a[b].detach().cpu().tolist())
        sb = set(idx_b[b].detach().cpu().tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        ious.append(float(inter) / float(max(union, 1)))
    return torch.tensor(ious, device=device, dtype=torch.float32)


def fps_index_recall_vanilla(idx_vanilla: torch.Tensor, idx_other: torch.Tensor) -> torch.Tensor:
    """
    以全精度 vanilla 的 FPS 下标集合 S_v 为参考：Recall = |S_v ∩ S_o| / K。
    idx_* : (B, K)；K 为 FPS 步数。若存在重复下标，按集合语义计数。
    """
    B, K = idx_vanilla.shape
    device = idx_vanilla.device
    recalls: List[float] = []
    for b in range(B):
        sv = set(idx_vanilla[b].detach().cpu().tolist())
        so = set(idx_other[b].detach().cpu().tolist())
        inter = len(sv & so)
        recalls.append(float(inter) / float(max(K, 1)))
    return torch.tensor(recalls, device=device, dtype=torch.float32)


def knn_recall_vs_a(knn_vanilla: torch.Tensor, knn_other: torch.Tensor) -> torch.Tensor:
    """
    以 vanilla 为参考：对每个 (B,K) 中心，Recall = |N_v ∩ N_o| / k。
    knn_* : (B, K, k)
    """
    eq = knn_vanilla.unsqueeze(-1) == knn_other.unsqueeze(-2)
    inter = eq.any(dim=-1).sum(dim=-1).float()
    k = float(knn_vanilla.shape[-1])
    return inter / k


def build_torch_dataset(
    key: str,
    entry: Dict[str, Any],
    global_defaults: Dict[str, Any],
    seed: int,
    max_samples: int,
) -> Tuple[torch.utils.data.Dataset, str]:
    t = entry["type"]
    root = str(entry.get("data_root") or "")
    num_points = int(entry.get("num_points", global_defaults.get("num_points", 2048)))
    partseg_norm = bool(entry.get("partseg_normalize", False))

    if t == "s3dis" and bool(entry.get("indoor3d_hdf5", False)):
        from s3dis_indoor3d_h5 import S3DISIndoor3DH5Dataset

        pc_root = root or os.environ.get("PC_DATA_ROOT", "/workspace/VoxelNet/pc_data")
        partition = str(entry.get("indoor3d_partition", "all"))
        test_area = str(entry.get("test_area", "5"))
        max_blocks = int(entry.get("max_blocks", -1))
        return (
            S3DISIndoor3DH5Dataset(
                pc_root,
                partition=partition,
                test_area=test_area,
                num_points=num_points,
                max_blocks=max_blocks,
            ),
            "S3DIS",
        )

    if t in ("modelnet40", "s3dis", "partseg", "shapenet_part", "scannet", "scanobjectnn"):
        ds_arg = "shapenet_part" if t == "partseg" else t
        args = types.SimpleNamespace(
            dataset=ds_arg,
            data_root=root,
            num_points=num_points,
            partseg_normalize=partseg_norm,
            seed=seed,
        )
        args = mds.resolve_defaults(args)
        ds, name = mds.build_dataset(args)
        return ds, name

    max_frames = max_samples if max_samples > 0 else -1
    do_roi = bool(entry.get("use_roi", True))

    if t == "kitti":
        if not root:
            root = "/workspace/VoxelNet/KITTI_Mini/training/velodyne"
        return (
            KITTIPointCloudWrapper(
                root,
                num_points=num_points,
                do_roi=do_roi,
                max_frames=max_frames,
                seed=seed,
            ),
            "KITTI",
        )
    if t == "waymo":
        if not root:
            root = "/workspace/VoxelNet/Waymo_Mini/Waymo_mini/kitti_format/training/velodyne"
        return (
            WaymoPointCloudWrapper(
                root,
                num_points=num_points,
                do_roi=do_roi,
                max_frames=max_frames,
                seed=seed,
            ),
            "Waymo",
        )
    if t == "nuscenes":
        if not root:
            root = "/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP"
        return (
            NuScenesPointCloudWrapper(
                root,
                num_points=num_points,
                do_roi=do_roi,
                max_frames=max_frames,
                seed=seed,
            ),
            "NuScenes",
        )

    raise ValueError(f"未知数据集类型: {t} (key={key})")


@torch.no_grad()
def eval_bits_on_loader(
    loader: DataLoader,
    device: torch.device,
    K: int,
    k_neighbors: int,
    bits: int,
    max_batches: int,
    quant_cfg: Dict[str, Any],
) -> Dict[str, float]:
    knn_iou_list: List[np.ndarray] = []
    knn_recall_list: List[np.ndarray] = []
    fps_iou_list: List[np.ndarray] = []
    fps_recall_list: List[np.ndarray] = []
    coord_mse_batch: List[float] = []
    n_batches = 0

    qscheme = str(quant_cfg.get("scheme", "per_cloud_minmax"))
    for pts, _ in tqdm(loader, desc=f"{qscheme} bits={bits}", leave=False):
        if max_batches > 0 and n_batches >= max_batches:
            break
        pts = pts.to(device, dtype=torch.float32)
        n_batches += 1

        pts_q = apply_quant_dequant(pts, bits, quant_cfg)
        # 反量化坐标相对原 float 坐标：全点、xyz 三通道均值的 MSE（每个 batch 一个标量，再对 batch 求统计）
        err2 = (pts - pts_q).pow(2)
        mse_batch = err2.mean(dim=(1, 2))  # (B,)
        coord_mse_batch.extend(mse_batch.detach().cpu().tolist())

        # 与全精度 vanilla 公平对齐：两路均 seed="first" → FPS 第一个中心为 batch 内点云张量的下标 0（同序、同起点）
        # 返回顺序: idx, radii, knn_idx, distances_matrix, knn_radii（勿把 radii 当成 idx）
        idx_v, _, knn_v, _, _ = fps_with_knn_vanilla(pts, K, k_neighbors=k_neighbors, seed="first")
        idx_q, _, knn_q, _, _ = fps_with_knn_vanilla(pts_q, K, k_neighbors=k_neighbors, seed="first")

        kiou = knn_neighbor_set_iou(knn_v, knn_q)
        krec = knn_recall_vs_a(knn_v, knn_q)
        fiou = fps_index_set_iou(idx_v, idx_q)
        frec = fps_index_recall_vanilla(idx_v, idx_q)

        knn_iou_list.append(kiou.cpu().numpy())
        knn_recall_list.append(krec.mean(dim=-1).cpu().numpy())
        fps_iou_list.append(fiou.cpu().numpy())
        fps_recall_list.append(frec.cpu().numpy())

    if not knn_iou_list:
        return {
            "knn_iou_mean": float("nan"),
            "knn_recall_mean": float("nan"),
            "fps_idx_recall_mean": float("nan"),
            "fps_idx_recall_std": float("nan"),
            "fps_set_iou_mean": float("nan"),
            "coord_quant_mse_mean": float("nan"),
            "coord_quant_mse_std": float("nan"),
            "coord_quant_rmse_mean": float("nan"),
            "n_batches": 0,
        }

    knn_iou_all = np.concatenate([x.reshape(-1) for x in knn_iou_list])
    knn_rec_all = np.concatenate([x.reshape(-1) for x in knn_recall_list])
    fps_iou_all = np.concatenate([x.reshape(-1) for x in fps_iou_list])
    fps_rec_all = np.concatenate([x.reshape(-1) for x in fps_recall_list])
    mse_arr = np.array(coord_mse_batch, dtype=np.float64)
    mse_mean = float(np.mean(mse_arr))
    mse_std = float(np.std(mse_arr))

    return {
        "knn_iou_mean": float(np.mean(knn_iou_all)),
        "knn_iou_std": float(np.std(knn_iou_all)),
        "knn_recall_mean": float(np.mean(knn_rec_all)),
        "knn_recall_std": float(np.std(knn_rec_all)),
        "fps_idx_recall_mean": float(np.mean(fps_rec_all)),
        "fps_idx_recall_std": float(np.std(fps_rec_all)),
        "fps_set_iou_mean": float(np.mean(fps_iou_all)),
        "fps_set_iou_std": float(np.std(fps_iou_all)),
        "coord_quant_mse_mean": mse_mean,
        "coord_quant_mse_std": mse_std,
        "coord_quant_rmse_mean": float(np.sqrt(max(mse_mean, 0.0))),
        "n_batches": int(n_batches),
    }


def _metrics_row_complete(row: Any) -> bool:
    """判断某 bit 行是否已写完（可安全跳过）。"""
    if not isinstance(row, dict):
        return False
    for k in ("knn_iou_mean", "knn_recall_mean", "fps_idx_recall_mean", "n_batches"):
        if k not in row:
            return False
    try:
        nb = int(row["n_batches"])
    except (TypeError, ValueError):
        return False
    return nb >= 0


def atomic_write_json(path: str, data: Any) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def try_load_resume_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _dataset_signature_block(
    K: int, k_neighbors: int, max_batches: int, bs: int, quant_cfg: Dict[str, Any]
) -> Tuple[int, int, int, int, str]:
    """用于判断与已存 JSON 是否同一评测设定（粗略：quant 用 JSON 串）。"""
    qc = quant_cfg_for_json(dict(quant_cfg))
    qkey = json.dumps(qc, sort_keys=True, ensure_ascii=False)
    return (K, k_neighbors, max_batches, bs, qkey)


def parse_bits_list(bits_str: str) -> List[int]:
    """
    支持:
      - \"1-16\" 闭区间连续整数
      - \"4,6,8\" 逗号列表
      - 混写: \"1-3,8,10-12\"（段之间逗号分隔）
    """
    s = (bits_str or "").strip()
    if not s:
        return list(range(1, 17))
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    # 去重保序
    seen = set()
    uniq: List[int] = []
    for b in out:
        if b < 1:
            raise ValueError(f"bits 须 >= 1，收到 {b}")
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def parse_args():
    p = argparse.ArgumentParser(description="FPS+KNN 量化 xyz 相对 vanilla 的 Recall/IoU 曲线数据")
    p.add_argument("--config", type=str, default=os.path.join(BENCH_DIR, "datasets.yaml"))
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="逗号分隔数据集键（对应 yaml 中 datasets 下 key），空表示全部",
    )
    p.add_argument(
        "--bits",
        type=str,
        default="2,4,6,8,10,12,14,16",
        help='每轴量化 bit：如 "1-16" 连续扫描、"8,10,12" 或 "1-4,8,12-14"',
    )
    p.add_argument("--fps_steps", type=int, default=0, help="覆盖 yaml 默认，0 表示用配置")
    p.add_argument("--k_neighbors", type=int, default=0, help="覆盖 yaml 默认，0 表示用配置")
    p.add_argument("--batch_size", type=int, default=0, help="覆盖单数据集 batch_size，0 表示用 yaml 各数据集")
    p.add_argument("--max_samples", type=int, default=-1, help="每数据集最多 batch 数，-1 用 yaml defaults.max_samples")
    p.add_argument(
        "--fast",
        action="store_true",
        help="快速模式：不全量扫样本；每数据集最多跑 --fast_max_batches（或 yaml defaults.fast_max_batches），除非同时显式传入 --max_samples",
    )
    p.add_argument(
        "--fast_max_batches",
        type=int,
        default=None,
        metavar="N",
        help="仅与 --fast 配合；省略时读配置 defaults.fast_max_batches，再否则 20",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=-1, help="-1 使用 yaml defaults.seed")
    p.add_argument("--out_json", type=str, default="", help="默认写入本目录 fps_knn_quant_recall_results.json")
    p.add_argument(
        "--resume",
        action="store_true",
        help="若 out_json 已存在则读入已有 datasets，跳过 metrics_vs_vanilla 中已完整的 bit；"
        "若该集 fps_steps/k_neighbors/max_batches/batch_size/quant 与当前不一致则整集重测",
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    g = cfg.get("defaults") or {}
    bits_list = parse_bits_list(args.bits)
    ds_cfg_all: Dict[str, Any] = cfg.get("datasets") or {}

    keys = [k.strip() for k in args.datasets.split(",") if k.strip()] if args.datasets.strip() else list(ds_cfg_all.keys())
    print(f"[info] 命令行 --bits 默认（共 {len(bits_list)} 档）: {bits_list}；各集可被 quant.bits_list 覆盖")
    print(f"[info] 将评测 yaml 中的数据集: {', '.join(keys)}（缺数据路径的会在加载时 skip）")

    seed = int(args.seed) if args.seed >= 0 else int(g.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    max_batches_default = int(g.get("max_samples", 0))
    fast_cap_cfg = int(g.get("fast_max_batches", 20))

    if args.fast:
        if args.max_samples >= 0:
            max_batches = args.max_samples
            sampling_mode = "fast_explicit_max_samples"
        else:
            max_batches = int(args.fast_max_batches) if args.fast_max_batches is not None else fast_cap_cfg
            sampling_mode = "fast_capped"
        print(
            f"[info] 快速模式: 每数据集最多 {max_batches} 个 batch"
            f"（全量请去掉 --fast，并令 --max_samples -1 且 yaml defaults.max_samples: 0）"
        )
    elif args.max_samples >= 0:
        max_batches = args.max_samples
        sampling_mode = "explicit_max_samples"
    else:
        max_batches = max_batches_default
        sampling_mode = "yaml_or_full"

    device = torch.device(args.device)
    out_path = args.out_json.strip() or os.path.join(BENCH_DIR, "fps_knn_quant_recall_results.json")

    prior: Optional[Dict[str, Any]] = None
    if args.resume:
        prior = try_load_resume_json(out_path)
        if prior:
            print(f"[info] --resume: 已加载 {out_path}，将跳过已完成的 (数据集, bit)")
        else:
            print(f"[info] --resume: 未找到或未读入已有 JSON，将从头评测 → {out_path}")

    out: Dict[str, Any] = {
        "config_path": os.path.abspath(args.config),
        "bits_per_axis_cli_default": bits_list,
        "datasets": copy.deepcopy(prior.get("datasets") or {}) if prior else {},
        "experiment": {
            "fps_seed": "first",
            "fps_start_point_index": 0,
            "note_zh": "全精度与量化路径均使用 seed=first：FPS 第一个采样中心为点云张量第 0 个点（索引一致、顺序一致），再比较后续 FPS 下标集合与 KNN 邻居集合。",
            "coord_mse_note_zh": "coord_quant_mse：逐样本 mean((p-p_q)^2) 再聚合；p_q 由 quant.scheme 决定（per_cloud_minmax / symmetric_neg1_1 / global_per_axis），与当前 bits 档位一致。",
            "quant_schemes_note_zh": "symmetric_neg1_1：[-1,1] 对称 qmax=2^(b-1)-1, zp=0。global_per_axis：固定 x/y/z_range（可 bounds_from_stats 从 dataset_xyz_bounds.json 注入）；centered=true 时轴对称 zp=0。quant.bits_list 覆盖命令行 --bits。",
            "fps_idx_recall_note_zh": "fps_idx_recall：以全精度 vanilla 的 FPS 下标集合 S_v 为参考，|S_v∩S_q|/K（K=fps_steps）；S_q 为量化坐标下 FPS 下标集合（集合语义，去重）。fps_set_iou 为 |S_v∩S_q|/|S_v∪S_q|。",
            "fast_mode": bool(args.fast),
            "sampling_mode": sampling_mode,
            "max_batches_cap_per_dataset": max_batches,
            "fast_max_batches_from_config": fast_cap_cfg,
            "resume": bool(args.resume),
            "incremental_atomic_save_zh": "每完成一个 bit 后原子写入 out_json（.tmp 再 replace），降低被 kill 时整文件损坏概率。",
        },
        "dataset_keys_requested": keys,
    }

    K_def = int(g.get("fps_steps", 512))
    k_def = int(g.get("k_neighbors", 24))

    for key in keys:
        if key not in ds_cfg_all:
            print(f"[skip] 未知数据集键: {key}")
            continue
        entry = dict(ds_cfg_all[key])
        quant_cfg = resolve_quant_cfg(g, entry)
        try:
            quant_cfg = finalize_quant_cfg(quant_cfg, g, key, BENCH_DIR)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"[skip] {key}: 量化边界来自统计 JSON 失败: {e}")
            continue
        K = int(args.fps_steps) if args.fps_steps > 0 else int(entry.get("fps_steps", K_def))
        k_neighbors = int(args.k_neighbors) if args.k_neighbors > 0 else int(entry.get("k_neighbors", k_def))
        bs = int(args.batch_size) if args.batch_size > 0 else int(entry.get("batch_size", g.get("batch_size", 4)))

        try:
            ds, dname = build_torch_dataset(key, entry, g, seed, max_batches if max_batches > 0 else -1)
        except FileNotFoundError as e:
            print(f"[skip] {key}: {e}")
            continue
        except Exception as e:
            print(f"[skip] {key}: {e}")
            continue

        loader = DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_points,
        )

        ds_bits = dataset_bits_list(bits_list, quant_cfg)
        eff_bits = effective_bits_list(ds_bits, quant_cfg)
        skipped_bits = [b for b in ds_bits if b not in eff_bits]
        if skipped_bits:
            print(f"[info] {key} quant.scheme={quant_cfg.get('scheme')} 跳过非法 bits: {skipped_bits}")
        if not eff_bits:
            print(f"[skip] {key}: 当前 bits 列表与 quant.scheme 组合下无可用 bit 档")
            continue

        print(f"[info] {key} 本集 bits 档位: {eff_bits}（scheme={quant_cfg.get('scheme')}）")

        sig_new = _dataset_signature_block(K, k_neighbors, max_batches, bs, quant_cfg)
        prior_block = out["datasets"].get(key) if args.resume else None
        per_bits: Dict[str, Any] = {}
        if prior_block and args.resume:
            per_bits = dict(prior_block.get("metrics_vs_vanilla") or {})
            old_sig = prior_block.get("resume_signature")
            if old_sig is None:
                print(
                    f"[warn] {key} 已有 JSON 缺少 resume_signature，无法校验 K/k/batch/quant，该集将整集重测"
                )
                per_bits = {}
            else:
                try:
                    old_t = (
                        int(old_sig["fps_steps"]),
                        int(old_sig["k_neighbors"]),
                        int(old_sig["max_batches"]),
                        int(old_sig["batch_size"]),
                        str(old_sig.get("quant_json_key", "")),
                    )
                    new_t = (sig_new[0], sig_new[1], sig_new[2], sig_new[3], sig_new[4])
                    if old_t != new_t:
                        print(
                            f"[warn] {key} 与已有 JSON 的 K/k/max_batches/batch/quant 不一致，该集 metrics 清空后重测"
                        )
                        per_bits = {}
                except (KeyError, TypeError, ValueError):
                    print(
                        f"[warn] {key} resume_signature 字段异常，该集将整集重测"
                    )
                    per_bits = {}

        for b in eff_bits:
            sb = str(b)
            if args.resume and sb in per_bits and _metrics_row_complete(per_bits[sb]):
                print(f"[resume] skip {key} bits={b}（已有完整结果）")
                continue
            per_bits[sb] = eval_bits_on_loader(
                loader, device, K, k_neighbors, b, max_batches, quant_cfg
            )
            out["datasets"][key] = {
                "display_name": dname,
                "type": entry.get("type"),
                "data_root": entry.get("data_root"),
                "num_points": entry.get("num_points", g.get("num_points")),
                "fps_steps": K,
                "k_neighbors": k_neighbors,
                "batch_size": bs,
                "max_batches": max_batches,
                "quant": quant_cfg_for_json(quant_cfg),
                "bits_evaluated": eff_bits,
                "metrics_vs_vanilla": per_bits,
                "resume_signature": {
                    "fps_steps": K,
                    "k_neighbors": k_neighbors,
                    "max_batches": max_batches,
                    "batch_size": bs,
                    "quant_json_key": sig_new[4],
                },
            }
            atomic_write_json(out_path, out)

        print(
            f"[ok] {key} ({dname})  K={K} k={k_neighbors} batches_cap={max_batches}  quant={quant_cfg.get('scheme')}"
        )

    print(f"已写入: {out_path}")


if __name__ == "__main__":
    main()
