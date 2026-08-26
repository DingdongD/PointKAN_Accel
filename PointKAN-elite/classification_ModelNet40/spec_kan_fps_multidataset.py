#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集 FPS radii 曲线提取 + KAN/MLP 训练（与 spec_kan_fps.py 同一套 FPS 与 loss）。

支持:
  - modelnet40   : 与 spec_kan_fps 相同（ModelNet40Wrapper）
  - scanobjectnn : HDF5，需 --scanobjectnn_train_h5 / --scanobjectnn_test_h5 或放在 --data_root 下按默认文件名
  - shapenet_part: ShapeNet Part，需 --shapenet_root
  - kitti / nuscenes / s3dis / waymo: 目录下扫文件，按排序后比例划分 train/test（与 FPS_KNN_Ablation 中 *_fps_exp / run_knn_coarse_fine_eval_* 一致的数据布局）
  - --preset voxelnet: 使用与本仓库 run_knn_coarse_fine_eval_*.sh、ScanObjectNN inference.sh 相同的默认绝对路径（见 VOXELNET_PRESETS）

radii 缓存文件名含数据集标签，可与多组 prefix_ratio 共用同一 --radii_dir:
  radii_train_K{K}_{dataset}.npy / radii_test_K{K}_{dataset}.npy
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ELITE_DIR = os.path.dirname(SCRIPT_DIR)
PT_ROOT = os.path.dirname(ELITE_DIR)
FPS_ABLATION = os.path.join(PT_ROOT, "FPS_KNN_Ablation")

for d in (SCRIPT_DIR, ELITE_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from spec_kan_fps import (  # noqa: E402
    Config,
    RadiiIOTensorDataset,
    build_model,
    collate_points,
    eval_mse,
    extract_radii_from_dataloader,
    set_seed,
    setup_logger,
    train_one_epoch,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _fps_ablation_imports():
    if not os.path.isdir(FPS_ABLATION):
        raise RuntimeError(
            f"未找到 FPS_KNN_Ablation 目录: {FPS_ABLATION}，请确认仓库布局。"
        )
    if FPS_ABLATION not in sys.path:
        sys.path.insert(0, FPS_ABLATION)
    try:
        from kitti_fps_exp import find_bin_files, load_kitti_bin, roi_filter
    except Exception as e:
        raise RuntimeError(f"无法从 kitti_fps_exp 导入: {e}") from e
    try:
        from s3dis_fps_exp import load_s3dis_npy
    except Exception as e:
        raise RuntimeError(f"无法从 s3dis_fps_exp 导入: {e}") from e
    try:
        from waymo_fps_exp import load_waymo_bin
    except Exception as e:
        raise RuntimeError(f"无法从 waymo_fps_exp 导入: {e}") from e

    def find_nuscenes_files(data_dir: str) -> List[str]:
        return sorted(glob.glob(os.path.join(data_dir, "*.pcd.bin")))

    return load_kitti_bin, roi_filter, find_bin_files, find_nuscenes_files, load_s3dis_npy, load_waymo_bin


# 与 FPS_KNN_Ablation/run_knn_coarse_fine_eval_*.sh、classification_ScanObjectNN/inference.sh 中 DATA_DIR/DATA_ROOT 一致
VOXELNET_PRESETS: dict = {
    "modelnet40": "/workspace/VoxelNet/pc_data",
    "scanobjectnn": "/workspace/VoxelNet/scannet_kan/main_split",
    "shapenet_part": "/workspace/VoxelNet/pc_data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
    "kitti": "/workspace/VoxelNet/KITTI_Mini/training/velodyne",
    "nuscenes": "/workspace/VoxelNet/Nuscenes_Mini/sweeps/LIDAR_TOP",
    "s3dis": "/workspace/VoxelNet/pc_data/s3dis_trainval",
    "waymo": "/workspace/VoxelNet/Waymo_Mini/Waymo_mini/kitti_format/training/velodyne",
}


def finalize_preset_paths(args: argparse.Namespace) -> None:
    """
    在 parse_args 之后调用：根据 --preset 填充空的 data_root / shapenet_root。
    preset=none 且 data_root 为空时，物体类数据集默认 ./data；室外类必须显式路径或 voxelnet。
    """
    ds = args.dataset.strip().lower()
    dr = (args.data_root or "").strip()

    if getattr(args, "preset", "none") == "voxelnet":
        if ds == "shapenet_part":
            if not (args.shapenet_root or "").strip():
                args.shapenet_root = VOXELNET_PRESETS["shapenet_part"]
            return
        if ds == "modelnet40":
            if not dr:
                args.data_root = VOXELNET_PRESETS["modelnet40"]
            return
        if ds in ("scanobjectnn", "kitti", "nuscenes", "s3dis", "waymo"):
            if not dr:
                args.data_root = VOXELNET_PRESETS[ds]
            return
        return

    if not dr:
        if ds in ("kitti", "nuscenes", "s3dis", "waymo"):
            raise ValueError(
                f"数据集 {ds} 需要设置 --data_root（数据帧目录），或使用 --preset voxelnet 自动使用仓库默认路径。"
            )
        args.data_root = "./data"


def _radii_paths_tagged(radii_dir: str, fps_steps: int, dataset_tag: str) -> Tuple[str, str]:
    tag = dataset_tag.replace(os.sep, "_").replace("/", "_")
    return (
        os.path.join(radii_dir, f"radii_train_K{fps_steps}_{tag}.npy"),
        os.path.join(radii_dir, f"radii_test_K{fps_steps}_{tag}.npy"),
    )


def _split_paths(paths: List[str], train_ratio: float) -> Tuple[List[str], List[str]]:
    if not paths:
        raise ValueError("文件列表为空，请检查 --data_root 与通配规则。")
    n = len(paths)
    n_train = int(round(train_ratio * n))
    n_train = max(1, min(n_train, n - 1)) if n > 1 else 1
    if n == 1:
        return paths, paths
    return paths[:n_train], paths[n_train:]


def _maybe_limit(xs: List[str], limit: int) -> List[str]:
    if limit and limit > 0:
        return xs[:limit]
    return xs


def pc_normalize_np(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc / (m + 1e-8)


def translate_pointcloud_np(pointcloud: np.ndarray) -> np.ndarray:
    xyz1 = np.random.uniform(low=2.0 / 3.0, high=3.0 / 2.0, size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    return np.add(np.multiply(pointcloud, xyz1), xyz2).astype(np.float32)


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------


class ScanObjectNNForRadii(Dataset):
    """ScanObjectNN HDF5；训练集与官方一致做平移增强 + shuffle。"""

    def __init__(self, h5_path: str, num_points: int, train: bool):
        import h5py

        self.num_points = num_points
        self.train = train
        with h5py.File(h5_path, "r") as f:
            self.data = f["data"][:].astype(np.float32)
            self.label = f["label"][:].astype(np.int64)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, item: int):
        pointcloud = self.data[item][: self.num_points].copy()
        label = int(self.label[item])
        if self.train:
            pointcloud = translate_pointcloud_np(pointcloud)
            np.random.shuffle(pointcloud)
        pts = torch.from_numpy(pointcloud[:, :3]).float()
        return pts, label


class ShapeNetPartXYZDataset(Dataset):
    """ShapeNet Part：仅 xyz，可配置 root；train 随机采样点，test 固定种子。"""

    def __init__(self, root: str, npoints: int, split: str, normalize: bool = False):
        self.npoints = npoints
        self.root = root
        self.normalize = normalize
        self.split = split
        catfile = os.path.join(self.root, "synsetoffset2category.txt")
        self.cat = {}
        with open(catfile, "r", encoding="utf-8") as f:
            for line in f:
                ls = line.strip().split()
                if len(ls) >= 2:
                    self.cat[ls[0]] = ls[1]

        tdir = os.path.join(self.root, "train_test_split")
        with open(os.path.join(tdir, "shuffled_train_file_list.json"), "r", encoding="utf-8") as f:
            train_ids = {str(d.split("/")[2]) for d in json.load(f)}
        with open(os.path.join(tdir, "shuffled_val_file_list.json"), "r", encoding="utf-8") as f:
            val_ids = {str(d.split("/")[2]) for d in json.load(f)}
        with open(os.path.join(tdir, "shuffled_test_file_list.json"), "r", encoding="utf-8") as f:
            test_ids = {str(d.split("/")[2]) for d in json.load(f)}

        self.meta: dict = {item: [] for item in self.cat}
        for item in self.cat:
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split == "trainval":
                fns = [fn for fn in fns if (fn[:-4] in train_ids or fn[:-4] in val_ids)]
            elif split == "train":
                fns = [fn for fn in fns if fn[:-4] in train_ids]
            elif split == "val":
                fns = [fn for fn in fns if fn[:-4] in val_ids]
            elif split == "test":
                fns = [fn for fn in fns if fn[:-4] in test_ids]
            else:
                raise ValueError(f"Unknown split: {split}")

            for fn in fns:
                token = os.path.splitext(os.path.basename(fn))[0]
                self.meta[item].append(os.path.join(dir_point, token + ".txt"))

        self.datapath: List[Tuple[str, str]] = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = dict(zip(self.cat, range(len(self.cat))))

    def __len__(self) -> int:
        return len(self.datapath)

    def __getitem__(self, index: int):
        cat, path = self.datapath[index]
        cls = int(self.classes[cat])
        data = np.loadtxt(path).astype(np.float32)
        point_set = data[:, 0:3]
        if self.normalize:
            point_set = pc_normalize_np(point_set)
        if self.split == "train":
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(42 + index)
        choice = rng.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice]
        return torch.from_numpy(point_set.astype(np.float32)), cls


class OutdoorFileDataset(Dataset):
    """KITTI / NuScenes / S3DIS：按路径懒加载，下采样到固定 N 点。"""

    def __init__(
        self,
        file_paths: Sequence[str],
        num_points: int,
        load_xyz: Callable[[str], np.ndarray],
        train: bool,
        rng_base_seed: int = 0,
    ):
        self.paths = list(file_paths)
        self.num_points = num_points
        self.load_xyz = load_xyz
        self.train = train
        self.rng_base_seed = rng_base_seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        xyz = self.load_xyz(self.paths[i])
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"期望 xyz [N,3+], 得到 {xyz.shape} @ {self.paths[i]}")
        xyz = np.ascontiguousarray(xyz[:, :3], dtype=np.float32)
        if self.train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng((self.rng_base_seed + i * 100003) % (2**32))
        n = xyz.shape[0]
        if n >= self.num_points:
            idx = rng.choice(n, size=self.num_points, replace=False)
        else:
            idx = rng.choice(n, size=self.num_points, replace=True)
        pts = torch.from_numpy(xyz[idx].copy())
        return pts, 0


def build_dataset_pair(
    args: argparse.Namespace,
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[Dataset, Dataset, str]:
    """
    返回 (train_ds, test_ds, dataset_tag)。
    """
    ds_name = args.dataset.strip().lower()
    if ds_name == "modelnet40":
        from spec_kan_fps import ModelNet40Wrapper

        train_ds = ModelNet40Wrapper(partition="train", num_points=cfg.num_points, root=args.data_root)
        test_ds = ModelNet40Wrapper(partition="test", num_points=cfg.num_points, root=args.data_root)
        return train_ds, test_ds, "modelnet40"

    if ds_name == "scanobjectnn":
        train_h5 = (args.scanobjectnn_train_h5 or "").strip()
        test_h5 = (args.scanobjectnn_test_h5 or "").strip()
        if not train_h5:
            cand = os.path.join(args.data_root, "training_objectdataset_augmentedrot_scale75.h5")
            if os.path.isfile(cand):
                train_h5 = cand
        if not test_h5:
            cand = os.path.join(args.data_root, "test_objectdataset_augmentedrot_scale75.h5")
            if os.path.isfile(cand):
                test_h5 = cand
        if not train_h5 or not os.path.isfile(train_h5):
            raise FileNotFoundError(
                "ScanObjectNN 需要训练集 h5。请设置 --scanobjectnn_train_h5 或将文件放在 --data_root 下 "
                "(默认查找 training_objectdataset_augmentedrot_scale75.h5)。"
            )
        if not test_h5 or not os.path.isfile(test_h5):
            raise FileNotFoundError(
                "ScanObjectNN 需要测试集 h5。请设置 --scanobjectnn_test_h5 或将文件放在 --data_root 下 "
                "(默认查找 test_objectdataset_augmentedrot_scale75.h5)。"
            )
        train_ds = ScanObjectNNForRadii(train_h5, cfg.num_points, train=True)
        test_ds = ScanObjectNNForRadii(test_h5, cfg.num_points, train=False)
        logger.info("ScanObjectNN train_h5=%s test_h5=%s", train_h5, test_h5)
        return train_ds, test_ds, "scanobjectnn"

    if ds_name == "shapenet_part":
        root = (args.shapenet_root or "").strip()
        if not root or not os.path.isdir(root):
            raise FileNotFoundError(
                "ShapeNet Part 需要存在的数据根目录（含 synsetoffset2category.txt）：--shapenet_root"
            )
        train_ds = ShapeNetPartXYZDataset(
            root, cfg.num_points, split="train", normalize=args.shapenet_normalize
        )
        test_ds = ShapeNetPartXYZDataset(
            root, cfg.num_points, split="test", normalize=args.shapenet_normalize
        )
        return train_ds, test_ds, "shapenet_part"

    load_kitti_bin, roi_filter, find_bin_files, find_nuscenes_files, load_s3dis_npy, load_waymo_bin = (
        _fps_ablation_imports()
    )
    data_root = (args.data_root or "").strip()
    if not data_root or not os.path.isdir(data_root):
        raise FileNotFoundError(f"--data_root 必须是已存在的目录: {data_root!r}")

    if ds_name in ("kitti", "waymo"):
        # Waymo kitti_format .bin 为 6×float32/点；KITTI/NuScenes 为 4×（见 waymo_fps_exp / kitti_fps_exp）
        load_pc = load_waymo_bin if ds_name == "waymo" else load_kitti_bin

        def load_xyz(p: str) -> np.ndarray:
            pc = load_pc(p)
            if args.kitti_use_roi:
                pc = roi_filter(
                    pc,
                    x_range=(args.roi_x0, args.roi_x1),
                    y_range=(args.roi_y0, args.roi_y1),
                    z_range=(args.roi_z0, args.roi_z1),
                )
            return pc[:, :3]

        all_paths = find_bin_files(data_root)
        all_paths = _maybe_limit(all_paths, args.max_list_files)
        tr, te = _split_paths(all_paths, args.outdoor_train_ratio)
        tr = _maybe_limit(tr, args.max_train_files)
        te = _maybe_limit(te, args.max_test_files)
        logger.info(
            "%s files total=%s train=%s test=%s (roi=%s)",
            ds_name.upper(),
            len(all_paths),
            len(tr),
            len(te),
            args.kitti_use_roi,
        )
        return (
            OutdoorFileDataset(tr, cfg.num_points, load_xyz, train=True),
            OutdoorFileDataset(te, cfg.num_points, load_xyz, train=False, rng_base_seed=args.outdoor_eval_seed),
            ds_name,
        )

    if ds_name == "nuscenes":

        def load_xyz(p: str) -> np.ndarray:
            pc = load_kitti_bin(p)
            if args.kitti_use_roi:
                pc = roi_filter(
                    pc,
                    x_range=(args.roi_x0, args.roi_x1),
                    y_range=(args.roi_y0, args.roi_y1),
                    z_range=(args.roi_z0, args.roi_z1),
                )
            return pc[:, :3]

        all_paths = find_nuscenes_files(data_root)
        all_paths = _maybe_limit(all_paths, args.max_list_files)
        tr, te = _split_paths(all_paths, args.outdoor_train_ratio)
        tr = _maybe_limit(tr, args.max_train_files)
        te = _maybe_limit(te, args.max_test_files)
        logger.info("NuScenes files total=%s train=%s test=%s", len(all_paths), len(tr), len(te))
        return (
            OutdoorFileDataset(tr, cfg.num_points, load_xyz, train=True),
            OutdoorFileDataset(te, cfg.num_points, load_xyz, train=False, rng_base_seed=args.outdoor_eval_seed),
            "nuscenes",
        )

    if ds_name == "s3dis":

        def load_xyz(p: str) -> np.ndarray:
            return load_s3dis_npy(p)

        all_paths = sorted(glob.glob(os.path.join(data_root, "*.npy")))
        all_paths = _maybe_limit(all_paths, args.max_list_files)
        tr, te = _split_paths(all_paths, args.outdoor_train_ratio)
        tr = _maybe_limit(tr, args.max_train_files)
        te = _maybe_limit(te, args.max_test_files)
        logger.info("S3DIS files total=%s train=%s test=%s", len(all_paths), len(tr), len(te))
        return (
            OutdoorFileDataset(tr, cfg.num_points, load_xyz, train=True),
            OutdoorFileDataset(te, cfg.num_points, load_xyz, train=False, rng_base_seed=args.outdoor_eval_seed),
            "s3dis",
        )

    raise ValueError(f"未知 --dataset: {args.dataset}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多数据集 FPS radii + KAN/MLP 训练")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=(
            "modelnet40",
            "scanobjectnn",
            "shapenet_part",
            "kitti",
            "nuscenes",
            "s3dis",
            "waymo",
        ),
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="",
        help="数据根目录；留空且 --preset voxelnet 时用仓库默认路径；物体数据集留空且 preset=none 时等价于 ./data",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=("none", "voxelnet"),
        help="voxelnet: 使用与 run_knn_coarse_fine_eval_*.sh / ScanObjectNN inference.sh 一致的默认绝对路径",
    )
    p.add_argument("--shapenet_root", type=str, default="", help="ShapeNet Part 根目录（synsetoffset2category.txt 所在目录）")
    p.add_argument(
        "--shapenet_normalize",
        action="store_true",
        help="ShapeNet 样本是否先做 pc_normalize（与部分分割实验一致可选）",
    )
    p.add_argument("--scanobjectnn_train_h5", type=str, default="", help="ScanObjectNN 训练 h5 绝对或相对路径")
    p.add_argument("--scanobjectnn_test_h5", type=str, default="", help="ScanObjectNN 测试 h5")
    p.add_argument(
        "--outdoor_train_ratio",
        type=float,
        default=0.8,
        help="kitti/nuscenes/s3dis/waymo：按文件名排序后前若干为训练，其余为测试",
    )
    p.add_argument("--outdoor_eval_seed", type=int, default=0, help="室外测试集子采样固定随机种子基值")
    p.add_argument("--kitti_use_roi", action="store_true", help="KITTI/NuScenes 是否使用 kitti_fps_exp.roi_filter")
    p.add_argument("--roi_x0", type=float, default=0.0)
    p.add_argument("--roi_x1", type=float, default=70.4)
    p.add_argument("--roi_y0", type=float, default=-40.0)
    p.add_argument("--roi_y1", type=float, default=40.0)
    p.add_argument("--roi_z0", type=float, default=-3.0)
    p.add_argument("--roi_z1", type=float, default=1.0)
    p.add_argument("--max_list_files", type=int, default=0, help="先截断总文件数再划分（0=不限制）")
    p.add_argument("--max_train_files", type=int, default=0, help="训练集最多帧数（0=不限制）")
    p.add_argument("--max_test_files", type=int, default=0, help="测试集最多帧数（0=不限制）")

    p.add_argument("--save_dir", type=str, default="./runs/fps_radii_mlp", help="checkpoint / log / results.json")
    p.add_argument("--resume", type=str, default="", help="last.pt 或含 last.pt 的目录")
    p.add_argument("--fps_steps", type=int, default=512)
    p.add_argument("--prefix_ratio", type=float, default=0.10)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size_train", type=int, default=256)
    p.add_argument("--batch_size_fps", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--model_type", type=str, default="mlp", choices=("mlp", "kan"))
    p.add_argument(
        "--radii_dir",
        type=str,
        default="",
        help="radii npy 缓存目录；默认同 save_dir。多 prefix_ratio 建议共用同一 radii_dir",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    finalize_preset_paths(args)
    cfg = Config()
    cfg.root = args.data_root or "./data"
    cfg.save_dir = args.save_dir
    cfg.fps_steps = args.fps_steps
    cfg.prefix_ratio = args.prefix_ratio
    cfg.num_points = args.num_points
    cfg.epochs = args.epochs
    cfg.batch_size_train = args.batch_size_train
    cfg.batch_size_fps = args.batch_size_fps
    cfg.lr = args.lr
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.model_type = args.model_type
    radii_dir = (args.radii_dir or "").strip() or cfg.save_dir
    cfg.radii_dir = radii_dir if radii_dir != cfg.save_dir else ""

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(radii_dir, exist_ok=True)
    logger = setup_logger(cfg.save_dir)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    train_ds, test_ds, dataset_tag = build_dataset_pair(args, cfg, logger)

    train_radii_path, test_radii_path = _radii_paths_tagged(radii_dir, cfg.fps_steps, dataset_tag)

    logger.info(
        "dataset=%s save_dir=%s radii_dir=%s fps_steps=%s prefix_ratio=%s model_type=%s",
        dataset_tag,
        cfg.save_dir,
        radii_dir,
        cfg.fps_steps,
        cfg.prefix_ratio,
        cfg.model_type,
    )

    train_loader_fps = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_fps,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )
    test_loader_fps = DataLoader(
        test_ds,
        batch_size=cfg.batch_size_fps,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )

    if os.path.exists(train_radii_path):
        train_radii = np.load(train_radii_path)
        logger.info("[Load] %s shape=%s", train_radii_path, train_radii.shape)
    else:
        train_radii = extract_radii_from_dataloader(
            cfg, train_loader_fps, desc=f"Extract radii [train:{dataset_tag}]"
        )
        np.save(train_radii_path, train_radii)
        logger.info("[Save] %s shape=%s", train_radii_path, train_radii.shape)

    if os.path.exists(test_radii_path):
        test_radii = np.load(test_radii_path)
        logger.info("[Load] %s shape=%s", test_radii_path, test_radii.shape)
    else:
        test_radii = extract_radii_from_dataloader(
            cfg, test_loader_fps, desc=f"Extract radii [test:{dataset_tag}]"
        )
        np.save(test_radii_path, test_radii)
        logger.info("[Save] %s shape=%s", test_radii_path, test_radii.shape)

    ds_train = RadiiIOTensorDataset(train_radii, cfg.prefix_ratio, cfg.in_dim, cfg.out_dim)
    ds_test = RadiiIOTensorDataset(test_radii, cfg.prefix_ratio, cfg.in_dim, cfg.out_dim)

    train_loader = DataLoader(
        ds_train, batch_size=cfg.batch_size_train, shuffle=True, drop_last=False, num_workers=0
    )
    test_loader = DataLoader(
        ds_test, batch_size=cfg.batch_size_train, shuffle=False, drop_last=False, num_workers=0
    )

    model = build_model(cfg, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    logger.info("========== Model structure ==========")
    for line in str(model).split("\n"):
        logger.info("%s", line)
    logger.info("=====================================")

    best_test_mse = float("inf")
    best_path = os.path.join(cfg.save_dir, "best.pt")
    last_path = os.path.join(cfg.save_dir, "last.pt")
    log = {"config": cfg.__dict__, "train_mse": [], "test_mse": [], "dataset": dataset_tag}
    start_epoch = 1

    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "last.pt")
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"断点续训需要 last.pt，未找到: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_test_mse = ckpt.get("best_test_mse", float("inf"))
        if "log" in ckpt:
            log = ckpt["log"]
        logger.info("[Resume] from %s -> epoch %s, best_test_mse=%.6e", resume_path, start_epoch, best_test_mse)

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device)
        te = eval_mse(model, test_loader, device)
        log["train_mse"].append(tr)
        log["test_mse"].append(te)

        if te < best_test_mse:
            best_test_mse = te
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_test_mse": best_test_mse,
                    "epoch": epoch,
                    "dataset": dataset_tag,
                },
                best_path,
            )
            logger.info("  -> new best, saved to %s", best_path)

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "epoch": epoch,
                "best_test_mse": best_test_mse,
                "cfg": cfg.__dict__,
                "log": log,
            },
            last_path,
        )
        logger.info("Epoch %03d | train_mse=%.6e | test_mse=%.6e | best=%.6e", epoch, tr, te, best_test_mse)

    with open(os.path.join(cfg.save_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    results = {
        "dataset": dataset_tag,
        "prefix_ratio": cfg.prefix_ratio,
        "fps_steps": cfg.fps_steps,
        "best_test_mse": float(best_test_mse),
        "epochs": cfg.epochs,
    }
    results_path = os.path.join(cfg.save_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("[Done] best checkpoint: %s (best_test_mse=%.6e)", best_path, best_test_mse)
    logger.info("[Done] results: %s", results_path)


if __name__ == "__main__":
    main()
