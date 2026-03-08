# -*- coding: utf-8 -*-
"""
KITTI 点云最远点采样 (FPS) 实验脚本。
功能：加载 KITTI .bin 点云、ROI 过滤、FPS 采样、覆盖/均匀性指标、
     BEV 投影可视化、半径曲线、数据集点云数目分布统计与直方图。
"""
import os
import glob
import time
import math
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# 日志配置（注释保持中文，绘图变量与 log 消息可英文）
def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """配置根 logger：控制台 + 可选文件。"""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)


logger = logging.getLogger(__name__)


# ============================================================
# 工具函数：KITTI 点云加载
# ============================================================

def load_kitti_bin(bin_path: str) -> np.ndarray:
    """
    加载 KITTI 激光雷达 .bin 文件。
    每个点 4×float32：x, y, z, reflectance（NuScenes 同为 4 维；Waymo 为 6 维需用 load_waymo_bin）。

    Returns:
        points: [N, 4]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def roi_filter(points: np.ndarray,
               x_range: Tuple[float, float] = (0.0, 70.4),
               y_range: Tuple[float, float] = (-40.0, 40.0),
               z_range: Tuple[float, float] = (-3.0, 1.0)) -> np.ndarray:
    """
    在 ROI 内过滤点，与常见 KITTI 检测范围一致。
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    mask = (
        (x >= x_range[0]) & (x <= x_range[1]) &
        (y >= y_range[0]) & (y <= y_range[1]) &
        (z >= z_range[0]) & (z <= z_range[1])
    )
    return points[mask]


# ============================================================
# 最远点采样 (FPS) 实现
# ============================================================

def farthest_point_sampling(xyz: np.ndarray, k: int, start_idx: int = 0):
    """
    基于 NumPy 的 CPU 朴素 FPS。复杂度 O(N * K)。

    Args:
        xyz: [N, 3] 点云坐标
        k: 采样点数
        start_idx: 起始点索引

    Returns:
        sampled_idx: [k] 被选中的点索引
        sampled_xyz: [k, 3] 采样点坐标
        radii: [k] 每步选点后的覆盖半径 radii[t] = max_i min_{s in S_t} ||p_i - s||
        min_d2: [N] 每个原始点到已选点集的最小距离平方
    """
    assert xyz.ndim == 2 and xyz.shape[1] == 3
    n = xyz.shape[0]
    if n == 0:
        raise ValueError("Empty point cloud after filtering.")
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, n)

    sampled_idx = np.zeros((k,), dtype=np.int64)
    radii = np.zeros((k,), dtype=np.float64)

    # 到已选点集的最小距离平方，初始为无穷
    min_d2 = np.full((n,), np.inf, dtype=np.float64)

    current = int(np.clip(start_idx, 0, n - 1))

    for t in range(k):
        sampled_idx[t] = current
        current_point = xyz[current]  # [3]

        diff = xyz - current_point[None, :]
        d2 = np.sum(diff * diff, axis=1)
        min_d2 = np.minimum(min_d2, d2)

        # 选完当前点后的覆盖半径
        radii[t] = math.sqrt(float(np.max(min_d2)))

        # 下一个点：距已选点集最远的点
        current = int(np.argmax(min_d2))

    sampled_xyz = xyz[sampled_idx]
    return sampled_idx, sampled_xyz, radii, min_d2


# ============================================================
# 指标计算
# ============================================================

def pairwise_nearest_neighbor_dist(points: np.ndarray) -> np.ndarray:
    """
    计算给定采样点两两之间的最近邻距离。O(M^2)，仅当 M 不大时使用。

    Args:
        points: [M, 3]

    Returns:
        nn_dist: [M]，每个点到其他采样点的最近邻距离
    """
    m = points.shape[0]
    if m <= 1:
        return np.zeros((m,), dtype=np.float64)

    diff = points[:, None, :] - points[None, :, :]   # [M, M, 3]
    d2 = np.sum(diff * diff, axis=2)                # [M, M]

    # 忽略自身距离
    np.fill_diagonal(d2, np.inf)
    nn_dist = np.sqrt(np.min(d2, axis=1))
    return nn_dist


def coverage_metrics(min_d2: np.ndarray) -> Dict[str, float]:
    """
    从原始点到采样集的覆盖指标。

    Args:
        min_d2: [N]，每个原始点到最近采样点的距离平方

    Returns:
        覆盖相关指标的字典
    """
    d = np.sqrt(min_d2)
    metrics = {
        "cover_mean": float(np.mean(d)),
        "cover_max": float(np.max(d)),
        "cover_p95": float(np.percentile(d, 95)),
        "cover_p99": float(np.percentile(d, 99)),
    }
    return metrics


def uniformity_metrics(sampled_xyz: np.ndarray) -> Dict[str, float]:
    """
    基于最近邻距离的采样点均匀性指标。
    """
    nn_dist = pairwise_nearest_neighbor_dist(sampled_xyz)
    metrics = {
        "sample_nn_mean": float(np.mean(nn_dist)),
        "sample_nn_min": float(np.min(nn_dist)),
        "sample_nn_std": float(np.std(nn_dist)),
    }
    return metrics


def radial_bin_retention(original_xyz: np.ndarray,
                         sampled_xyz: np.ndarray,
                         bins=(0, 10, 20, 30, 40, 50, 60, 80)) -> Dict[str, Dict[str, float]]:
    """
    按到原点的径向距离分段统计点数量分布（原始 vs 采样）。
    """
    ro = np.linalg.norm(original_xyz, axis=1)
    rs = np.linalg.norm(sampled_xyz, axis=1)

    result = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        cnt_o = int(((ro >= lo) & (ro < hi)).sum())
        cnt_s = int(((rs >= lo) & (rs < hi)).sum())
        ratio = (cnt_s / cnt_o) if cnt_o > 0 else 0.0
        result[f"{lo}-{hi}m"] = {
            "orig_count": cnt_o,
            "sampled_count": cnt_s,
            "ratio": ratio
        }
    return result


# ============================================================
# 可视化
# ============================================================

def plot_radius_curve(radii: np.ndarray, save_path: str) -> None:
    """绘制 FPS 每步覆盖半径衰减曲线。绘图变量使用英文。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(1, len(radii) + 1), radii)
    ax.set_xlabel("FPS step")
    ax.set_ylabel("Radius r_t")
    ax.set_title("FPS Radius Decay Curve")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.debug("Saved radius curve: %s", save_path)


def visualize_point_cloud_2d_bev(original_xyz: np.ndarray,
                                 sampled_xyz: np.ndarray,
                                 save_path: str,
                                 title: str = "Point Cloud (BEV)",
                                 margin_ratio: float = 0.02) -> None:
    """BEV 投影可视化：x-y 俯视图，按数据范围设置比例与边距。绘图变量使用英文。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(original_xyz[:, 0], original_xyz[:, 1], s=0.5, alpha=0.35, c="gray", label="original")
    ax.scatter(sampled_xyz[:, 0], sampled_xyz[:, 1], s=8.0, alpha=0.9, c="red", label="FPS sampled")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # 按数据范围设置 xlim/ylim，留边距
    all_xy = np.vstack([original_xyz[:, :2], sampled_xyz[:, :2]])
    xlim = [float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0]))]
    ylim = [float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1]))]
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    if x_range <= 0:
        x_range = 1.0
    if y_range <= 0:
        y_range = 1.0
    mx = margin_ratio * x_range
    my = margin_ratio * y_range
    ax.set_xlim(xlim[0] - mx, xlim[1] + mx)
    ax.set_ylim(ylim[0] - my, ylim[1] + my)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.debug("Saved BEV: %s", save_path)


# ============================================================
# 数据集点云数目分布统计与直方图
# ============================================================

def xyz_range_stats(xyz: np.ndarray) -> Dict[str, float]:
    """
    单帧点云 x/y/z 方向的最小、最大值及范围。
    xyz: [N, 3]，返回 x_min, x_max, y_min, y_max, z_min, z_max, x_range, y_range, z_range。
    """
    if xyz.shape[0] == 0:
        return {
            "x_min": np.nan, "x_max": np.nan, "y_min": np.nan, "y_max": np.nan,
            "z_min": np.nan, "z_max": np.nan,
            "x_range": np.nan, "y_range": np.nan, "z_range": np.nan,
        }
    x_min, x_max = float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))
    y_min, y_max = float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1]))
    z_min, z_max = float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2]))
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "x_range": x_max - x_min, "y_range": y_max - y_min, "z_range": z_max - z_min,
    }


def collect_point_counts(data_dir: str,
                         do_roi: bool = True,
                         max_frames: int = -1,
                         load_fn=None,
                         find_fn=None) -> Tuple[np.ndarray, List[str]]:
    """
    扫描数据目录下点云文件，统计每帧点云数量（ROI 过滤后）。
    load_fn(path)->[N,4], find_fn(data_dir)->list of paths；默认 KITTI .bin。
    """
    load_fn = load_fn or load_kitti_bin
    find_fn = find_fn or find_bin_files
    files = find_fn(data_dir)
    if max_frames > 0:
        files = files[:max_frames]
    counts_list = []
    paths_used = []
    for path in files:
        points = load_fn(path)
        if do_roi:
            points = roi_filter(points)
        counts_list.append(points.shape[0])
        paths_used.append(path)
    return np.array(counts_list, dtype=np.int64), paths_used


def collect_point_counts_and_xyz_ranges(data_dir: str,
                                        do_roi: bool = True,
                                        max_frames: int = -1,
                                        load_fn=None,
                                        find_fn=None
                                        ) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    一次扫描：每帧点云数量 + 每帧 x/y/z 的 min/max。
    load_fn/find_fn 同 collect_point_counts；默认 KITTI。
    """
    load_fn = load_fn or load_kitti_bin
    find_fn = find_fn or find_bin_files
    files = find_fn(data_dir)
    if max_frames > 0:
        files = files[:max_frames]
    counts_list = []
    paths_used = []
    x_mins, x_maxs = [], []
    y_mins, y_maxs = [], []
    z_mins, z_maxs = [], []
    for path in files:
        points = load_fn(path)
        if do_roi:
            points = roi_filter(points)
        n = points.shape[0]
        counts_list.append(n)
        paths_used.append(path)
        if n > 0:
            xyz = points[:, :3]
            x_mins.append(np.min(xyz[:, 0]))
            x_maxs.append(np.max(xyz[:, 0]))
            y_mins.append(np.min(xyz[:, 1]))
            y_maxs.append(np.max(xyz[:, 1]))
            z_mins.append(np.min(xyz[:, 2]))
            z_maxs.append(np.max(xyz[:, 2]))
        else:
            x_mins.append(np.nan)
            x_maxs.append(np.nan)
            y_mins.append(np.nan)
            y_maxs.append(np.nan)
            z_mins.append(np.nan)
            z_maxs.append(np.nan)
    xyz_ranges = {
        "x_min": np.array(x_mins, dtype=np.float64),
        "x_max": np.array(x_maxs, dtype=np.float64),
        "y_min": np.array(y_mins, dtype=np.float64),
        "y_max": np.array(y_maxs, dtype=np.float64),
        "z_min": np.array(z_mins, dtype=np.float64),
        "z_max": np.array(z_maxs, dtype=np.float64),
    }
    return np.array(counts_list, dtype=np.int64), paths_used, xyz_ranges


def plot_point_count_histogram(counts: np.ndarray,
                               save_path: str,
                               bins: int = 30,
                               title: str = "Point count per frame (dataset)") -> None:
    """
    绘制点云数目分布直方图，并叠加基本统计信息。绘图变量使用英文。
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counts, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(counts.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean = {counts.mean():.0f}")
    ax.axvline(np.median(counts), color="orange", linestyle="-.", linewidth=2, label=f"Median = {np.median(counts):.0f}")
    ax.set_xlabel("Point count per frame")
    ax.set_ylabel("Number of frames")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    textstr = f"Total frames: {len(counts)}\nMin: {counts.min()}\nMax: {counts.max()}\nStd: {counts.std():.1f}"
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.debug("Saved point count histogram: %s", save_path)


def plot_xyz_range_histograms(xyz_ranges: Dict[str, np.ndarray],
                              save_path: str,
                              bins: int = 30) -> None:
    """
    绘制每帧 x/y/z 方向 min、max 的分布直方图（2x3：行 x/y/z，列 min/max）。英文标签。
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    keys = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    titles = ["x min (per frame)", "x max (per frame)", "y min (per frame)",
              "y max (per frame)", "z min (per frame)", "z max (per frame)"]
    for i, (key, title) in enumerate(zip(keys, titles)):
        arr = xyz_ranges[key]
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            axes[i].set_title(title)
            continue
        axes[i].hist(arr, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        axes[i].axvline(np.mean(arr), color="red", linestyle="--", linewidth=1.5, label=f"Mean={np.mean(arr):.2f}")
        axes[i].set_xlabel(key)
        axes[i].set_ylabel("Number of frames")
        axes[i].set_title(title)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    plt.suptitle("XYZ range distribution per frame (after ROI)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.debug("Saved xyz range histograms: %s", save_path)


# ============================================================
# 单帧实验
# ============================================================

def run_single_file(bin_path: str,
                    out_dir: str,
                    k: int = 1024,
                    do_roi: bool = True,
                    start_idx_mode: str = "first",
                    load_fn=None) -> Dict:
    """
    对单帧点云执行 FPS 实验：加载、ROI、FPS、指标、半径曲线与 BEV、保存数组。
    load_fn(path)->[N,4]，默认 load_kitti_bin。
    """
    load_fn = load_fn or load_kitti_bin
    logger.debug("run_single_file: %s k=%d", bin_path, k)
    os.makedirs(out_dir, exist_ok=True)

    points = load_fn(bin_path)
    if do_roi:
        points = roi_filter(points)

    xyz = points[:, :3]
    n = xyz.shape[0]

    if n == 0:
        raise RuntimeError(f"No valid points after ROI filter: {bin_path}")

    if start_idx_mode == "first":
        start_idx = 0
    elif start_idx_mode == "farthest_from_origin":
        start_idx = int(np.argmax(np.sum(xyz * xyz, axis=1)))
    else:
        raise ValueError(f"Unsupported start_idx_mode: {start_idx_mode}")

    t0 = time.time()
    sampled_idx, sampled_xyz, radii, min_d2 = farthest_point_sampling(
        xyz, k=k, start_idx=start_idx
    )
    elapsed = time.time() - t0

    cov = coverage_metrics(min_d2)
    uni = uniformity_metrics(sampled_xyz)
    radial = radial_bin_retention(xyz, sampled_xyz)
    xyz_range = xyz_range_stats(xyz)

    stem = os.path.splitext(os.path.basename(bin_path))[0]
    if stem.endswith(".pcd"):
        stem = os.path.splitext(stem)[0]

    # 半径曲线 + BEV 投影可视化
    plot_radius_curve(radii, os.path.join(out_dir, f"{stem}_radius_curve.png"))
    visualize_point_cloud_2d_bev(
        xyz, sampled_xyz,
        os.path.join(out_dir, f"{stem}_bev.png"),
        title=f"{stem} | N={n}, K={len(sampled_xyz)}"
    )

    # 保存数组
    np.save(os.path.join(out_dir, f"{stem}_sampled_idx.npy"), sampled_idx)
    np.save(os.path.join(out_dir, f"{stem}_sampled_xyz.npy"), sampled_xyz)
    np.save(os.path.join(out_dir, f"{stem}_radii.npy"), radii)
    np.save(os.path.join(out_dir, f"{stem}_min_d2.npy"), min_d2)

    result = {
        "file": bin_path,
        "frame_id": stem,
        "num_points": int(n),
        "k": int(len(sampled_xyz)),
        "time_sec": float(elapsed),
        **cov,
        **uni,
        "radial_retention": radial,
        **xyz_range,
    }
    return result


# ============================================================
# 批处理与汇总
# ============================================================

def summarize_results(results: List[Dict]) -> Dict[str, float]:
    """对各帧的标量指标求平均与标准差（含 xyz 分布区间）。"""
    keys = [
        "num_points", "k", "time_sec",
        "cover_mean", "cover_max", "cover_p95", "cover_p99",
        "sample_nn_mean", "sample_nn_min", "sample_nn_std",
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
        "x_range", "y_range", "z_range",
    ]
    summary = {}
    for key in keys:
        vals = [r[key] for r in results if key in r]
        if not vals:
            continue
        summary[f"{key}_avg"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        summary[f"{key}_min"] = float(np.min(vals))
        summary[f"{key}_max"] = float(np.max(vals))
    return summary


def print_result(result: Dict):
    """打印单帧实验结果。"""
    print("=" * 80)
    print(f"帧: {result['frame_id']}")
    print(f"文件: {result['file']}")
    print(f"N={result['num_points']}, K={result['k']}, 耗时={result['time_sec']:.4f}s")
    print(
        f"覆盖: mean={result['cover_mean']:.4f}, "
        f"max={result['cover_max']:.4f}, "
        f"p95={result['cover_p95']:.4f}, "
        f"p99={result['cover_p99']:.4f}"
    )
    print(
        f"均匀性(采样最近邻): mean={result['sample_nn_mean']:.4f}, "
        f"min={result['sample_nn_min']:.4f}, "
        f"std={result['sample_nn_std']:.4f}"
    )
    print("XYZ 分布区间:")
    print(
        f"  x: [{result['x_min']:.2f}, {result['x_max']:.2f}], "
        f"y: [{result['y_min']:.2f}, {result['y_max']:.2f}], "
        f"z: [{result['z_min']:.2f}, {result['z_max']:.2f}]"
    )
    print("径向保留:")
    for k, v in result["radial_retention"].items():
        print(
            f"  {k}: 原始={v['orig_count']}, 采样={v['sampled_count']}, 比例={v['ratio']:.4f}"
        )


def find_bin_files(data_dir: str) -> List[str]:
    """返回数据目录下所有 .bin 文件路径（已排序）。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    return files


def _file_stem(path: str) -> str:
    """返回用于输出子目录/文件名的 stem（.pcd.bin -> 去掉 .pcd）。"""
    s = os.path.splitext(os.path.basename(path))[0]
    if s.endswith(".pcd"):
        s = os.path.splitext(s)[0]
    return s


# ============================================================
# 主程序
# ============================================================

def get_parser(description: str = "KITTI 点云 FPS 实验：统计、可视化与批处理"):
    """共用的参数解析器，供 KITTI/NuScenes/Waymo 复用。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/VoxelNet/KITTI_Mini/training/velodyne",
        help="点云目录"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./kitti_fps_outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1024,
        help="FPS 采样点数"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="处理的帧数；-1 表示全部"
    )
    parser.add_argument(
        "--start_idx_mode",
        type=str,
        default="first",
        choices=["first", "farthest_from_origin"],
        help="FPS 起始点选取方式"
    )
    parser.add_argument(
        "--no_roi",
        action="store_true",
        help="关闭 ROI 过滤"
    )
    parser.add_argument(
        "--no_stats",
        action="store_true",
        help="跳过数据集点云数目分布统计与直方图"
    )
    parser.add_argument(
        "--stats_frames",
        type=int,
        default=-1,
        help="统计直方图时最多扫描的帧数，-1 表示全部"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志输出文件路径（可选）"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用 DEBUG 级别日志"
    )
    return parser


def run_main(args):
    """执行 FPS 实验主流程；args 可带 _load_fn / _find_files 覆盖默认 KITTI 加载与查找。"""
    level = logging.DEBUG if args.verbose else logging.INFO
    log_path = args.log_file if args.log_file else os.path.join(args.out_dir, "fps_exp.log")
    setup_logging(level=level, log_file=log_path)

    os.makedirs(args.out_dir, exist_ok=True)

    find_fn = getattr(args, "_find_files", None) or find_bin_files
    load_fn = getattr(args, "_load_fn", None) or load_kitti_bin
    bin_files = find_fn(args.data_dir)
    if len(bin_files) == 0:
        raise FileNotFoundError(f"在 {args.data_dir} 中未找到点云文件")

    # 数据集点云数目分布 + 每帧 xyz 分布区间统计与直方图
    if not args.no_stats:
        n_stats = len(bin_files) if args.stats_frames <= 0 else min(len(bin_files), args.stats_frames)
        logger.info("Collecting point counts and XYZ ranges (scanning %d files)...", n_stats)
        counts, _, xyz_ranges = collect_point_counts_and_xyz_ranges(
            args.data_dir,
            do_roi=(not args.no_roi),
            max_frames=args.stats_frames if args.stats_frames > 0 else -1,
            load_fn=load_fn,
            find_fn=find_fn,
        )
        hist_path = os.path.join(args.out_dir, "point_count_histogram.png")
        plot_point_count_histogram(
            counts,
            save_path=hist_path,
            bins=min(40, max(10, len(np.unique(counts)) // 2)),
            title="Point count per frame (after ROI)"
        )
        logger.info("Stats over %d frames: count range=[%d, %d], mean=%.0f", len(counts), int(counts.min()), int(counts.max()), counts.mean())
        logger.info("Histogram saved: %s", hist_path)

        xyz_hist_path = os.path.join(args.out_dir, "xyz_range_histograms.png")
        plot_xyz_range_histograms(xyz_ranges, save_path=xyz_hist_path)
        # 汇总 xyz 区间（全数据集）
        for ax_name in ["x", "y", "z"]:
            mn = xyz_ranges[f"{ax_name}_min"]
            mx = xyz_ranges[f"{ax_name}_max"]
            mn, mx = mn[~np.isnan(mn)], mx[~np.isnan(mx)]
            if len(mn) > 0:
                logger.info("  %s: min across frames [%.2f, %.2f], max across frames [%.2f, %.2f]",
                            ax_name, float(np.min(mn)), float(np.max(mn)), float(np.min(mx)), float(np.max(mx)))
        logger.info("XYZ range histograms saved: %s", xyz_hist_path)

    if args.num_frames > 0:
        bin_files = bin_files[:args.num_frames]

    results = []

    for i, bin_path in enumerate(bin_files):
        logger.info("[%d/%d] Processing: %s", i + 1, len(bin_files), bin_path)
        frame_out_dir = os.path.join(args.out_dir, _file_stem(bin_path))

        result = run_single_file(
            bin_path=bin_path,
            out_dir=frame_out_dir,
            k=args.k,
            do_roi=(not args.no_roi),
            start_idx_mode=args.start_idx_mode,
            load_fn=load_fn,
        )
        print_result(result)
        results.append(result)

    summary = summarize_results(results)

    logger.info("Summary over frames:")
    for k, v in summary.items():
        logger.info("  %s: %.6f", k, v)

    # 保存汇总文本
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 80 + "\n")
            f.write(f"帧: {r['frame_id']}\n")
            f.write(f"文件: {r['file']}\n")
            f.write(f"N={r['num_points']}, K={r['k']}, 耗时={r['time_sec']:.6f}s\n")
            f.write(
                f"覆盖: mean={r['cover_mean']:.6f}, max={r['cover_max']:.6f}, "
                f"p95={r['cover_p95']:.6f}, p99={r['cover_p99']:.6f}\n"
            )
            f.write(
                f"均匀性: mean={r['sample_nn_mean']:.6f}, "
                f"min={r['sample_nn_min']:.6f}, std={r['sample_nn_std']:.6f}\n"
            )
            f.write(
                f"XYZ 分布: x=[{r['x_min']:.2f}, {r['x_max']:.2f}], "
                f"y=[{r['y_min']:.2f}, {r['y_max']:.2f}], z=[{r['z_min']:.2f}, {r['z_max']:.2f}]\n"
            )
            f.write("径向保留:\n")
            for kk, vv in r["radial_retention"].items():
                f.write(
                    f"  {kk}: 原始={vv['orig_count']}, 采样={vv['sampled_count']}, 比例={vv['ratio']:.6f}\n"
                )
            f.write("\n")

        f.write("#" * 80 + "\n")
        f.write("各帧汇总\n")
        for k, v in summary.items():
            f.write(f"{k}: {v:.6f}\n")

    logger.info("Summary saved: %s", summary_path)
    logger.info("Visualizations saved to: %s", args.out_dir)


def main():
    parser = get_parser(description="KITTI 点云 FPS 实验：统计、可视化与批处理")
    args = parser.parse_args()
    run_main(args)


if __name__ == "__main__":
    main()