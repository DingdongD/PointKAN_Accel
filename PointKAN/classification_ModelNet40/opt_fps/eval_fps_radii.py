"""
FPS 校准集与 radii 曲线分析（基于 ModelNet40）。
- 构建层次采样（按类别）的校准集合
- 对每个样本运行 FPS，记录每步的最小距离 radii
- 统计校准集上 radii 分布，绘图：横轴=采样迭代次数，纵轴=radii（mean±std）
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# 使用与 main 一致的数据路径
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ModelNet40
from opt_fps_knn import fps_with_knn_adaptive


def build_calibration_set(dataset, samples_per_class=50, partition="train", seed=42):
    """
    按类别层次采样，构建校准集索引。
    dataset: ModelNet40(partition=partition, num_points=...)
    samples_per_class: 每类最多采样数
    """
    labels = dataset.label.squeeze()  # 获取数据集标签并挤压维度
    classes = np.unique(labels)  # 获取唯一的类别
    np.random.seed(seed)  # 设置随机种子
    indices = []  # 存储校准集索引
    for c in classes:  # 遍历每个类别
        idx_c = np.where(labels == c)[0]  # 获取当前类别的所有索引
        n = min(samples_per_class, len(idx_c))  # 确定当前类别的采样数量
        indices.extend(np.random.choice(idx_c, size=n, replace=False).tolist())  # 随机选择样本
    return np.array(indices)  # 返回校准集索引数组


def compute_fps_radii_batch(X, K, device, seed="first"):
    """
    X: [B, N, 3], 返回 radii [B, K]
    """
    idx, radii, *_ = fps_with_knn_adaptive(
        X, K, k_neighbors=16, init_threshold=0.1,
        max_iter=10, tol=0.05, seed=seed
    )  # 运行自适应 KNN 的 FPS 算法
    return radii  # 返回半径数组


def run_calibration_radii(
    partition="train",
    num_points=1024,
    samples_per_class=50,
    fps_steps=1024,
    batch_size=32,
    seed=42,
    device="cuda",
):
    """
    在校准集上运行 FPS，收集所有 radii，返回 (iterations, mean_radii, std_radii)。
    """
    dataset = ModelNet40(partition=partition, num_points=num_points)  # 创建 ModelNet40 数据集实例
    calib_indices = build_calibration_set(dataset, samples_per_class=samples_per_class, partition=partition, seed=seed)  # 构建校准集
    K = min(fps_steps, num_points)  # 确定 FPS 迭代次数
    print(f"  Calibration set size: {len(calib_indices)}, FPS steps: {K}")  # 打印校准集大小和 FPS 步数

    all_radii = []  # 存储所有半径
    loader = DataLoader(
        Subset(dataset, calib_indices),  # 创建校准集的子集
        batch_size=batch_size,  # 批量大小
        shuffle=False,  # 不打乱顺序
        num_workers=0,  # 数据加载 worker 数量
        drop_last=False,  # 不丢弃最后一批
    )
    n_batches = len(loader)  # 批次数
    for data, _ in tqdm(loader, total=n_batches, desc="FPS radii", unit="batch", ncols=100):  # 遍历数据
        # data: [B, N, 3]
        data = data.to(device, dtype=torch.float32)  # 将数据移至设备并转换为 float32
        radii = compute_fps_radii_batch(data, K, device, seed="first")  # 计算 FPS 半径
        all_radii.append(radii.cpu().numpy())  # 将半径移至 CPU 并转换为 NumPy 数组
    # [N_calib, K]
    all_radii = np.concatenate(all_radii, axis=0)  # 合并所有半径
    mean_radii = np.mean(all_radii, axis=0)  # 计算每个步骤的平均半径
    std_radii = np.std(all_radii, axis=0)  # 计算每个步骤的半径标准差
    return np.arange(K), mean_radii, std_radii, all_radii  # 返回迭代次数、平均半径、标准差和所有半径


def plot_radii(iterations, mean_radii, std_radii, out_dir, prefix="fps_radii"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    if any("Arial" in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = ["Arial"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(5, 3.5))  # 创建图形和坐标轴
    ax.plot(iterations, mean_radii, color="C0", linewidth=1.5, label="Mean")  # 绘制平均半径曲线
    ax.fill_between(iterations, mean_radii - std_radii, mean_radii + std_radii, color="C0", alpha=0.3)  # 绘制标准差区域
    ax.set_xlabel("FPS iteration", fontsize=11)  # 设置 x 轴标签
    ax.set_ylabel("Min distance (radii)", fontsize=11)  # 设置 y 轴标签
    ax.tick_params(axis="both", direction="in")  # 设置刻度方向向内
    ax.grid(linestyle="--", alpha=0.4)  # 添加网格线
    for spine in ax.spines.values():  # 设置坐标轴边框
        spine.set_visible(True)  # 显示边框
        spine.set_linewidth(1.5)  # 设置边框线宽
    ax.legend(loc="best", frameon=True, fontsize=9)  # 添加图例
    ax.set_title("FPS radii on calibration set (ModelNet40)")  # 设置标题
    plt.tight_layout(pad=0.5)  # 调整布局
    os.makedirs(out_dir, exist_ok=True)  # 创建输出目录
    for ext in ["png", "pdf"]:  # 保存为 png 和 pdf 格式
        path = os.path.join(out_dir, f"{prefix}.{ext}")  # 构建文件路径
        plt.savefig(
            path,
            dpi=600,  # 设置 DPI 为 600
            bbox_inches="tight",  # 紧凑边界
            pad_inches=0.02,  # 边距
            transparent=True,  # 透明背景
        )
        print(f"Saved {path}")  # 打印保存路径
    plt.close()  # 关闭图形


def main():
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--partition", type=str, default="train", choices=["train", "test"])  # 数据集分区
    parser.add_argument("--num_points", type=int, default=1024)  # 点云点数
    parser.add_argument("--samples_per_class", type=int, default=50)  # 每类采样数
    parser.add_argument("--fps_steps", type=int, default=1024, help="FPS 迭代次数（如 1024）")  # FPS 迭代次数
    parser.add_argument("--batch_size", type=int, default=32)  # 批量大小
    parser.add_argument("--seed", type=int, default=42)  # 随机种子
    parser.add_argument("--out_dir", type=str, default="opt_fps/figures")  # 输出目录
    parser.add_argument("--no_plot", action="store_true", help="只统计不绘图")  # 是否不绘图
    args = parser.parse_args()  # 解析参数

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备
    if args.seed is not None:  # 设置随机种子
        torch.manual_seed(args.seed)  # 设置 PyTorch 随机种子
        np.random.seed(args.seed)  # 设置 NumPy 随机种子

    print("Building calibration set and computing FPS radii...")  # 打印开始信息
    print(f"  partition={args.partition}, samples_per_class={args.samples_per_class}, fps_steps={args.fps_steps}, batch_size={args.batch_size}")  # 打印参数
    iterations, mean_radii, std_radii, all_radii = run_calibration_radii(
        partition=args.partition,
        num_points=args.num_points,
        samples_per_class=args.samples_per_class,
        fps_steps=args.fps_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )  # 运行校准半径计算
    print(f"Calibration samples: {all_radii.shape[0]}, FPS steps: {all_radii.shape[1]}")  # 打印校准样本数和 FPS 步数
    print(f"Radii mean range: [{mean_radii.min():.6f}, {mean_radii.max():.6f}]")  # 打印半径均值范围

    if not args.no_plot:  # 如果需要绘图
        out_dir = args.out_dir  # 输出目录
        if not os.path.isabs(out_dir):  # 如果不是绝对路径
            out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), out_dir)  # 转换为绝对路径
        plot_radii(iterations, mean_radii, std_radii, out_dir, prefix="fps_radii")  # 绘制半径曲线



if __name__ == "__main__":
    main()
