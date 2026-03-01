"""
FPS-KNN 对比接口：vanilla FPS-KNN vs fps_with_knn_adaptive。
使用朴素 FPS 实现，与优化版采用相同种子策略，便于对比。
支持对动态阈值 KNN 参数 tol、max_iter 做网格搜索，结果存 JSON 并出图。
"""
import json
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def build_calibration_set(dataset, samples_per_class=50, seed=42):
    """按类别层次采样构建校准集索引（与 eval_fps_radii 一致）。"""
    labels = dataset.label.squeeze()
    classes = np.unique(labels)
    np.random.seed(seed)
    indices = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        n = min(samples_per_class, len(idx_c))
        indices.extend(np.random.choice(idx_c, size=n, replace=False).tolist())
    return np.array(indices)


def parse_float_list(s):
    """解析逗号或空格分隔的浮点数列表，如 '0.01,0.05,0.1' 或 '0.01 0.05 0.1'"""
    s = s.replace(",", " ").strip()
    return [float(x) for x in s.split() if x]


def parse_int_list(s):
    """解析逗号或空格分隔的整数列表"""
    s = s.replace(",", " ").strip()
    return [int(x) for x in s.split() if x]

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ModelNet40
from models.pointkan import index_points, knn_point
from opt_fps_knn import fps_with_knn_adaptive


@torch.no_grad()
def naive_fps(xyz, K, seed="first"):
    """
    朴素 FPS，与 opt_fps_knn 的种子策略一致，便于与 adaptive 版本对比。
    xyz: [B, N, 3], K: 采样数
    seed: "first" -> 首点索引 0, "random" -> 随机, "max_norm" -> 范数最大点
    Returns: fps_idx [B, K]
    """
    B, N, _ = xyz.shape
    device, dtype = xyz.device, xyz.dtype
    K = min(K, N)
    x2 = (xyz * xyz).sum(dim=-1)  # (B, N)
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    if seed == "first":
        first = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        first = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    idx[:, 0] = first
    centroid = xyz[batch_idx, first]
    dist = torch.sum((xyz - centroid.unsqueeze(1)) ** 2, dim=-1)
    min_dist = torch.min(min_dist, dist)
    min_dist[batch_idx, first] = 0.0

    for t in range(1, K):
        farthest = torch.argmax(min_dist, dim=1)
        idx[:, t] = farthest
        centroid = xyz[batch_idx, farthest]
        dist = torch.sum((xyz - centroid.unsqueeze(1)) ** 2, dim=-1)
        min_dist = torch.min(min_dist, dist)
        min_dist[batch_idx, farthest] = 0.0

    return idx


def vanilla_fps_knn(xyz, K, k_neighbors, device, fps_seed="first"):
    """
    Vanilla FPS（朴素实现，与 adaptive 同种子）+ KNN。
    xyz: [B, N, 3]
    Returns:
        fps_idx: [B, K]
        knn_idx: [B, K, k_neighbors]
    """
    fps_idx = naive_fps(xyz, K, seed=fps_seed)
    new_xyz = index_points(xyz, fps_idx)
    knn_idx = knn_point(k_neighbors, xyz, new_xyz)
    return fps_idx, knn_idx


def iou_sets(a, b):
    """a, b 为两个索引集合（1D array），返回 IoU = |intersection| / |union|。"""
    sa, sb = set(a.flatten().tolist()), set(b.flatten().tolist())  # 将索引数组转换为集合
    inter = len(sa & sb)  # 计算交集大小
    union = len(sa | sb)  # 计算并集大小
    return inter / union if union else 1.0  # 计算 IoU，处理除零情况


def compare_fps_iou(fps_vanilla, fps_adaptive):
    """
    fps_vanilla, fps_adaptive: [B, K]
    返回每样本 FPS 索引集 IoU，再取平均。
    """
    B = fps_vanilla.shape[0]  # 获取批量大小
    ious = []  # 存储每个样本的 IoU
    for b in range(B):  # 遍历每个样本
        ious.append(iou_sets(fps_vanilla[b].cpu().numpy(), fps_adaptive[b].cpu().numpy()))  # 计算 IoU
    return np.mean(ious)  # 返回平均 IoU


def compare_knn_iou(knn_vanilla, knn_adaptive):
    """
    knn_vanilla, knn_adaptive: [B, K, k]
    对每个 (b, t) 计算该中心下 KNN 索引集的 IoU，再对所有中心、样本取平均。
    """
    B, K, k = knn_vanilla.shape  # 获取批量大小、FPS 点数和 KNN 邻居数
    ious = []  # 存储每个中心的 IoU
    for b in range(B):  # 遍历每个样本
        for t in range(K):  # 遍历每个 FPS 点
            iou = iou_sets(knn_vanilla[b, t].cpu().numpy(), knn_adaptive[b, t].cpu().numpy())  # 计算 IoU
            ious.append(iou)  # 添加到列表
    return np.mean(ious)  # 返回平均 IoU


def compare_fps_iou_per_sample(fps_vanilla, fps_adaptive):
    """返回每样本 FPS IoU 列表，长度 B。"""
    B = fps_vanilla.shape[0]
    return [float(iou_sets(fps_vanilla[b].cpu().numpy(), fps_adaptive[b].cpu().numpy())) for b in range(B)]


def compare_knn_iou_per_sample(knn_vanilla, knn_adaptive):
    """返回每样本 KNN IoU 列表（该样本内所有 FPS 中心的 KNN IoU 均值），长度 B。"""
    B, K, _ = knn_vanilla.shape
    out = []
    for b in range(B):
        ious = [iou_sets(knn_vanilla[b, t].cpu().numpy(), knn_adaptive[b, t].cpu().numpy()) for t in range(K)]
        out.append(float(np.mean(ious)))
    return out


def run_comparison(
    partition="test",
    num_points=1024,
    num_batches=50,
    batch_size=1,
    K=512,
    k_neighbors=24,
    seed=42,
    device="cuda",
    adaptive_kw=None,
    fps_seed="first",
    save_indices_n=0,
    use_calibration=False,
    samples_per_class=50,
    num_workers=0,
):
    """
    在 ModelNet40 上逐样本计算 vanilla vs adaptive 的 FPS IoU 与 KNN IoU。
    use_calibration=True 时在按类别分层采样的校准集上评估（与 eval_fps_radii 一致）；
    否则在指定 partition 上按 shuffle 取前 num_batches 个样本。
    batch_size>1 时 GPU 批处理加速，仍会得到每样本 IoU；num_workers>0 时 DataLoader 多进程预取。
    """
    if adaptive_kw is None:
        adaptive_kw = {}
    kwargs_adaptive = {**adaptive_kw, "seed": fps_seed}
    dataset = ModelNet40(partition=partition, num_points=num_points)
    if use_calibration:
        calib_indices = build_calibration_set(dataset, samples_per_class=samples_per_class, seed=seed)
        n_use = min(num_batches * batch_size, len(calib_indices))
        dataset = Subset(dataset, calib_indices[:n_use])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    per_sample_fps_iou = []
    per_sample_knn_iou = []
    saved_indices = []  # list of dict for first save_indices_n samples
    all_iters_used = []   # 收集 iters_used [B,K]，用于统计迭代次数
    all_thresholds = []   # 收集 thresholds [B,K]，用于统计最终阈值
    batch_count = 0
    total_samples = 0
    pbar = tqdm(total=num_batches, desc="FPS-KNN compare", unit="batch", ncols=100)
    for data, _ in loader:
        if batch_count >= num_batches:
            break
        data = data.to(device, dtype=torch.float32)
        B, N = data.size(0), data.size(1)
        K_actual = min(K, N)
        k_actual = min(k_neighbors, N)

        fps_v, knn_v = vanilla_fps_knn(data, K_actual, k_actual, device, fps_seed=fps_seed)
        idx_a, _, knn_a, _dist, _knn_radii, thresholds, iters_used = fps_with_knn_adaptive(
            data, K_actual, k_neighbors=k_actual, **kwargs_adaptive
        )
        all_iters_used.append(iters_used.cpu().numpy())
        all_thresholds.append(thresholds.cpu().numpy())

        fps_iou_list = compare_fps_iou_per_sample(fps_v, idx_a)
        knn_iou_list = compare_knn_iou_per_sample(knn_v, knn_a)
        per_sample_fps_iou.extend(fps_iou_list)
        per_sample_knn_iou.extend(knn_iou_list)

        for b in range(B):
            if total_samples < save_indices_n:
                saved_indices.append({
                    "sample_index": total_samples,
                    "fps_vanilla": fps_v[b : b + 1].cpu().numpy().tolist(),
                    "fps_adaptive": idx_a[b : b + 1].cpu().numpy().tolist(),
                    "knn_vanilla": knn_v[b : b + 1].cpu().numpy().tolist(),
                    "knn_adaptive": knn_a[b : b + 1].cpu().numpy().tolist(),
                    "fps_iou": fps_iou_list[b],
                    "knn_iou": knn_iou_list[b],
                })
            total_samples += 1

        batch_count += 1
        pbar.set_postfix(fps_iou=f"{np.mean(per_sample_fps_iou):.3f}", knn_iou=f"{np.mean(per_sample_knn_iou):.3f}", n=len(per_sample_fps_iou))
        pbar.update(1)
    pbar.close()

    fps_ious = per_sample_fps_iou
    knn_ious = per_sample_knn_iou
    iters_flat = np.concatenate([a.flatten() for a in all_iters_used]) if all_iters_used else np.array([0.0])
    thresh_flat = np.concatenate([a.flatten() for a in all_thresholds]) if all_thresholds else np.array([0.0])
    return {
        "fps_iou_mean": float(np.mean(fps_ious)),
        "fps_iou_std": float(np.std(fps_ious)),
        "knn_iou_mean": float(np.mean(knn_ious)),
        "knn_iou_std": float(np.std(knn_ious)),
        "iters_used_mean": float(np.mean(iters_flat)),
        "iters_used_std": float(np.std(iters_flat)),
        "thresholds_mean": float(np.mean(thresh_flat)),
        "thresholds_std": float(np.std(thresh_flat)),
        "num_samples": len(per_sample_fps_iou),
        "per_sample_fps_iou": per_sample_fps_iou,
        "per_sample_knn_iou": per_sample_knn_iou,
        "saved_indices": saved_indices,
    }


def run_sweep_tol_max_iter(
    tol_list,
    max_iter_list,
    init_threshold_list=None,
    partition="test",
    num_points=1024,
    num_batches=50,
    batch_size=1,
    K=512,
    k_neighbors=24,
    seed=42,
    device="cuda",
    fps_seed="first",
    use_calibration=False,
    samples_per_class=50,
    num_workers=0,
):
    """
    对 (tol, max_iter) 或 (tol, max_iter, init_threshold) 做网格搜索。
    返回每组的 fps/knn IoU、iters_used 与 thresholds 的均值/标准差。
    """
    if init_threshold_list is None:
        init_threshold_list = [None]  # 占位，使用 opt_fps_knn 默认
    grid = []
    total_runs = len(tol_list) * len(max_iter_list) * len(init_threshold_list)
    pbar = tqdm(total=total_runs, desc="sweep tol & max_iter & init_threshold", unit="run", ncols=100)
    for tol in tol_list:
        for max_iter in max_iter_list:
            for init_threshold in init_threshold_list:
                adaptive_kw = {"tol": tol, "max_iter": max_iter}
                if init_threshold is not None:
                    adaptive_kw["init_threshold"] = init_threshold
                res = run_comparison(
                    partition=partition,
                    num_points=num_points,
                    num_batches=num_batches,
                    batch_size=batch_size,
                    K=K,
                    k_neighbors=k_neighbors,
                    seed=seed,
                    device=device,
                    adaptive_kw=adaptive_kw,
                    fps_seed=fps_seed,
                    save_indices_n=0,
                    use_calibration=use_calibration,
                    samples_per_class=samples_per_class,
                    num_workers=num_workers,
                )
                item = {
                    "tol": tol,
                    "max_iter": max_iter,
                    "fps_iou_mean": res["fps_iou_mean"],
                    "fps_iou_std": res["fps_iou_std"],
                    "knn_iou_mean": res["knn_iou_mean"],
                    "knn_iou_std": res["knn_iou_std"],
                    "iters_used_mean": res["iters_used_mean"],
                    "iters_used_std": res["iters_used_std"],
                    "thresholds_mean": res["thresholds_mean"],
                    "thresholds_std": res["thresholds_std"],
                    "num_samples": res["num_samples"],
                }
                if init_threshold is not None:
                    item["init_threshold"] = init_threshold
                grid.append(item)
                pbar.set_postfix(tol=tol, max_iter=max_iter, knn_iou=f"{res['knn_iou_mean']:.3f}", iters=f"{res['iters_used_mean']:.1f}")
                pbar.update(1)
    pbar.close()
    return grid


def _build_2d_matrices(grid, tol_list, max_iter_list, init_threshold_val=None):
    """从 grid 中提取 tol × max_iter 的 2D 矩阵；若 init_threshold_val 非 None 则只取该 init_threshold 的项。"""
    knn_iou = np.full((len(tol_list), len(max_iter_list)), np.nan)
    fps_iou = np.full((len(tol_list), len(max_iter_list)), np.nan)
    iters_mean = np.full((len(tol_list), len(max_iter_list)), np.nan)
    thresh_mean = np.full((len(tol_list), len(max_iter_list)), np.nan)
    for r in grid:
        if init_threshold_val is not None and r.get("init_threshold") != init_threshold_val:
            continue
        i = tol_list.index(r["tol"])
        j = max_iter_list.index(r["max_iter"])
        knn_iou[i, j] = r["knn_iou_mean"]
        fps_iou[i, j] = r["fps_iou_mean"]
        iters_mean[i, j] = r["iters_used_mean"]
        thresh_mean[i, j] = r["thresholds_mean"]
    return knn_iou, fps_iou, iters_mean, thresh_mean


def _build_tol_vs_init_threshold(grid, tol_list, max_iter_list, init_threshold_list, max_iter_fix):
    """固定 max_iter=max_iter_fix，构建 tol × init_threshold 矩阵。"""
    knn = np.full((len(tol_list), len(init_threshold_list)), np.nan)
    iters = np.full((len(tol_list), len(init_threshold_list)), np.nan)
    thresh = np.full((len(tol_list), len(init_threshold_list)), np.nan)
    for r in grid:
        if r.get("max_iter") != max_iter_fix:
            continue
        i = tol_list.index(r["tol"])
        if "init_threshold" not in r:
            continue
        k = init_threshold_list.index(r["init_threshold"])
        knn[i, k] = r["knn_iou_mean"]
        iters[i, k] = r["iters_used_mean"]
        thresh[i, k] = r["thresholds_mean"]
    return knn, iters, thresh


def _build_max_iter_vs_init_threshold(grid, tol_list, max_iter_list, init_threshold_list, tol_fix):
    """固定 tol=tol_fix，构建 max_iter × init_threshold 矩阵。"""
    knn = np.full((len(max_iter_list), len(init_threshold_list)), np.nan)
    iters = np.full((len(max_iter_list), len(init_threshold_list)), np.nan)
    thresh = np.full((len(max_iter_list), len(init_threshold_list)), np.nan)
    for r in grid:
        if r.get("tol") != tol_fix:
            continue
        j = max_iter_list.index(r["max_iter"])
        if "init_threshold" not in r:
            continue
        k = init_threshold_list.index(r["init_threshold"])
        knn[j, k] = r["knn_iou_mean"]
        iters[j, k] = r["iters_used_mean"]
        thresh[j, k] = r["thresholds_mean"]
    return knn, iters, thresh


def plot_sweep_results(grid, tol_list, max_iter_list, init_threshold_list=None, out_dir=".", prefix="sweep_tol_max_iter"):
    """绘制 tol × max_iter 的 KNN IoU、iters_used、thresholds 热力图及曲线图；支持多 init_threshold 分片出图。"""
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

    has_init = init_threshold_list is not None and len(init_threshold_list) > 0 and init_threshold_list[0] is not None
    if has_init:
        slices = [(v, f"_init_threshold_{v}") for v in init_threshold_list]
    else:
        slices = [(None, "")]

    os.makedirs(out_dir, exist_ok=True)
    xtick_labels = [str(m) for m in max_iter_list]
    ytick_labels = [str(t) for t in tol_list]

    for init_val, suffix in slices:
        knn_iou_matrix, fps_iou_matrix, iters_matrix, thresh_matrix = _build_2d_matrices(
            grid, tol_list, max_iter_list, init_threshold_val=init_val
        )
        if np.all(np.isnan(knn_iou_matrix)):
            continue
        p = prefix + suffix

        # 图1: KNN IoU 热力图
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        vmin1, vmax1 = np.nanmin(knn_iou_matrix), np.nanmax(knn_iou_matrix)
        im1 = ax1.imshow(knn_iou_matrix, aspect="auto", cmap="viridis", vmin=vmin1, vmax=vmax1)
        ax1.set_xticks(np.arange(len(max_iter_list)))
        ax1.set_yticks(np.arange(len(tol_list)))
        ax1.set_xticklabels(xtick_labels)
        ax1.set_yticklabels(ytick_labels)
        ax1.set_xlabel("max_iter", fontsize=11)
        ax1.set_ylabel("tol", fontsize=11)
        ax1.set_title("KNN index IoU (vanilla vs adaptive)" + (f" (init_threshold={init_val})" if init_val is not None else ""))
        for i in range(len(tol_list)):
            for j in range(len(max_iter_list)):
                ax1.text(j, i, f"{knn_iou_matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(im1, ax=ax1, label="KNN IoU")
        ax1.tick_params(axis="both", direction="in")
        for spine in ax1.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        plt.tight_layout(pad=0.5)
        path = os.path.join(out_dir, f"{p}_knn_iou.png")
        plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        print(f"Saved {path}")
        plt.close()

        # 图2: FPS IoU 热力图
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        vmin2, vmax2 = np.nanmin(fps_iou_matrix), np.nanmax(fps_iou_matrix)
        im2 = ax2.imshow(fps_iou_matrix, aspect="auto", cmap="viridis", vmin=vmin2, vmax=vmax2)
        ax2.set_xticks(np.arange(len(max_iter_list)))
        ax2.set_yticks(np.arange(len(tol_list)))
        ax2.set_xticklabels(xtick_labels)
        ax2.set_yticklabels(ytick_labels)
        ax2.set_xlabel("max_iter", fontsize=11)
        ax2.set_ylabel("tol", fontsize=11)
        ax2.set_title("FPS index IoU (vanilla vs adaptive)")
        for i in range(len(tol_list)):
            for j in range(len(max_iter_list)):
                ax2.text(j, i, f"{fps_iou_matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(im2, ax=ax2, label="FPS IoU")
        ax2.tick_params(axis="both", direction="in")
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        plt.tight_layout(pad=0.5)
        path = os.path.join(out_dir, f"{p}_fps_iou.png")
        plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        print(f"Saved {path}")
        plt.close()

        # 图3: iters_used 均值热力图（动态阈值迭代次数）
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        vmin3, vmax3 = np.nanmin(iters_matrix), np.nanmax(iters_matrix)
        im3 = ax3.imshow(iters_matrix, aspect="auto", cmap="plasma", vmin=vmin3, vmax=vmax3)
        ax3.set_xticks(np.arange(len(max_iter_list)))
        ax3.set_yticks(np.arange(len(tol_list)))
        ax3.set_xticklabels(xtick_labels)
        ax3.set_yticklabels(ytick_labels)
        ax3.set_xlabel("max_iter", fontsize=11)
        ax3.set_ylabel("tol", fontsize=11)
        ax3.set_title("Mean iters_used (threshold convergence)" + (f" (init_threshold={init_val})" if init_val is not None else ""))
        for i in range(len(tol_list)):
            for j in range(len(max_iter_list)):
                ax3.text(j, i, f"{iters_matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
        plt.colorbar(im3, ax=ax3, label="iters_used (mean)")
        ax3.tick_params(axis="both", direction="in")
        for spine in ax3.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        plt.tight_layout(pad=0.5)
        path = os.path.join(out_dir, f"{p}_iters_used.png")
        plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        print(f"Saved {path}")
        plt.close()

        # 图4: 最终阈值 thresholds 均值热力图
        fig4, ax4 = plt.subplots(figsize=(5, 4))
        vmin4, vmax4 = np.nanmin(thresh_matrix), np.nanmax(thresh_matrix)
        im4 = ax4.imshow(thresh_matrix, aspect="auto", cmap="cividis", vmin=vmin4, vmax=vmax4)
        ax4.set_xticks(np.arange(len(max_iter_list)))
        ax4.set_yticks(np.arange(len(tol_list)))
        ax4.set_xticklabels(xtick_labels)
        ax4.set_yticklabels(ytick_labels)
        ax4.set_xlabel("max_iter", fontsize=11)
        ax4.set_ylabel("tol", fontsize=11)
        ax4.set_title("Mean final threshold" + (f" (init_threshold={init_val})" if init_val is not None else ""))
        for i in range(len(tol_list)):
            for j in range(len(max_iter_list)):
                ax4.text(j, i, f"{thresh_matrix[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)
        plt.colorbar(im4, ax=ax4, label="threshold (mean)")
        ax4.tick_params(axis="both", direction="in")
        for spine in ax4.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        plt.tight_layout(pad=0.5)
        path = os.path.join(out_dir, f"{p}_thresholds.png")
        plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
        print(f"Saved {path}")
        plt.close()

        # 图5: 曲线 — KNN IoU vs tol（按 max_iter）
        if len(max_iter_list) >= 1 and len(tol_list) > 1:
            fig5, ax5 = plt.subplots(figsize=(5, 3.5))
            for j, max_iter in enumerate(max_iter_list):
                ax5.plot(tol_list, knn_iou_matrix[:, j], "o-", label=f"max_iter={max_iter}", linewidth=1.5)
            ax5.set_xlabel("tol", fontsize=11)
            ax5.set_ylabel("KNN index IoU", fontsize=11)
            ax5.set_title("KNN IoU vs tol (per max_iter)")
            ax5.legend(loc="best", fontsize=9)
            ax5.tick_params(axis="both", direction="in")
            ax5.grid(linestyle="--", alpha=0.4)
            for spine in ax5.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            plt.tight_layout(pad=0.5)
            path = os.path.join(out_dir, f"{p}_knn_vs_tol.png")
            plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
            print(f"Saved {path}")
            plt.close()

        # 图6: 曲线 — KNN IoU vs max_iter（按 tol）
        if len(tol_list) >= 1 and len(max_iter_list) > 1:
            fig6, ax6 = plt.subplots(figsize=(5, 3.5))
            for i, tol in enumerate(tol_list):
                ax6.plot(max_iter_list, knn_iou_matrix[i, :], "o-", label=f"tol={tol}", linewidth=1.5)
            ax6.set_xlabel("max_iter", fontsize=11)
            ax6.set_ylabel("KNN index IoU", fontsize=11)
            ax6.set_title("KNN IoU vs max_iter (per tol)")
            ax6.legend(loc="best", fontsize=9)
            ax6.tick_params(axis="both", direction="in")
            ax6.grid(linestyle="--", alpha=0.4)
            for spine in ax6.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            plt.tight_layout(pad=0.5)
            path = os.path.join(out_dir, f"{p}_knn_vs_max_iter.png")
            plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
            print(f"Saved {path}")
            plt.close()

    # 当扫了 init_threshold 时：增加「不同 init_threshold」的对比图（tol vs init_threshold、max_iter vs init_threshold）
    if has_init and len(init_threshold_list) > 1:
        init_xticks = [str(t) for t in init_threshold_list]
        # 固定 max_iter = 第一个，看 tol × init_threshold
        max_iter_fix = max_iter_list[0]
        knn_tol_init, iters_tol_init, thresh_tol_init = _build_tol_vs_init_threshold(
            grid, tol_list, max_iter_list, init_threshold_list, max_iter_fix
        )
        if not np.all(np.isnan(knn_tol_init)):
            for name, mat, title, cmap in [
                ("knn_iou", knn_tol_init, f"KNN IoU (max_iter={max_iter_fix})", "viridis"),
                ("iters_used", iters_tol_init, f"iters_used (max_iter={max_iter_fix})", "plasma"),
                ("thresholds", thresh_tol_init, f"thresholds (max_iter={max_iter_fix})", "cividis"),
            ]:
                fig, ax = plt.subplots(figsize=(5, 4))
                vmin, vmax = np.nanmin(mat), np.nanmax(mat)
                im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks(np.arange(len(init_threshold_list)))
                ax.set_yticks(np.arange(len(tol_list)))
                ax.set_xticklabels(init_xticks)
                ax.set_yticklabels(ytick_labels)
                ax.set_xlabel("init_threshold", fontsize=11)
                ax.set_ylabel("tol", fontsize=11)
                ax.set_title(title)
                for i in range(len(tol_list)):
                    for k in range(len(init_threshold_list)):
                        val = mat[i, k]
                        ax.text(k, i, f"{val:.3f}" if name == "knn_iou" else (f"{val:.1f}" if name == "iters_used" else f"{val:.4f}"), ha="center", va="center", color="white", fontsize=8)
                plt.colorbar(im, ax=ax)
                ax.tick_params(axis="both", direction="in")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                plt.tight_layout(pad=0.5)
                path = os.path.join(out_dir, f"{prefix}_vs_init_threshold_tol_{name}.png")
                plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
                print(f"Saved {path}")
                plt.close()

        # 固定 tol = 第一个，看 max_iter × init_threshold
        tol_fix = tol_list[0]
        knn_mi_init, iters_mi_init, thresh_mi_init = _build_max_iter_vs_init_threshold(
            grid, tol_list, max_iter_list, init_threshold_list, tol_fix
        )
        if not np.all(np.isnan(knn_mi_init)):
            for name, mat, title, cmap in [
                ("knn_iou", knn_mi_init, f"KNN IoU (tol={tol_fix})", "viridis"),
                ("iters_used", iters_mi_init, f"iters_used (tol={tol_fix})", "plasma"),
                ("thresholds", thresh_mi_init, f"thresholds (tol={tol_fix})", "cividis"),
            ]:
                fig, ax = plt.subplots(figsize=(5, 4))
                vmin, vmax = np.nanmin(mat), np.nanmax(mat)
                im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks(np.arange(len(init_threshold_list)))
                ax.set_yticks(np.arange(len(max_iter_list)))
                ax.set_xticklabels(init_xticks)
                ax.set_yticklabels(xtick_labels)
                ax.set_xlabel("init_threshold", fontsize=11)
                ax.set_ylabel("max_iter", fontsize=11)
                ax.set_title(title)
                for j in range(len(max_iter_list)):
                    for k in range(len(init_threshold_list)):
                        val = mat[j, k]
                        ax.text(k, j, f"{val:.3f}" if name == "knn_iou" else (f"{val:.1f}" if name == "iters_used" else f"{val:.4f}"), ha="center", va="center", color="white", fontsize=8)
                plt.colorbar(im, ax=ax)
                ax.tick_params(axis="both", direction="in")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.5)
                plt.tight_layout(pad=0.5)
                path = os.path.join(out_dir, f"{prefix}_vs_init_threshold_max_iter_{name}.png")
                plt.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.02, transparent=True)
                print(f"Saved {path}")
                plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", type=str, default="test")
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1, help="批大小；>1 时 GPU 批处理加速，仍会得到每样本 IoU")
    parser.add_argument("--K", type=int, default=512, help="FPS 采样点数（与 PointKAN 首阶段一致）")
    parser.add_argument("--k_neighbors", type=int, default=24)
    parser.add_argument("--fps_seed", type=str, default="first", choices=["first", "random", "max_norm"],
                        help="FPS 初始点与 adaptive 一致")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_txt", type=str, default=None, help="结果摘要写入该文件")
    parser.add_argument("--out_json", type=str, default=None, help="完整结果写入 JSON：每样本 IoU、可选索引")
    parser.add_argument("--save_indices_n", type=int, default=0, help="将前 N 个样本的 FPS/KNN 索引写入 JSON")
    # 动态阈值 KNN 参数网格搜索
    parser.add_argument("--sweep_tol", type=str, default=None, help="tol 列表，逗号或空格分隔，如 '0.01,0.05,0.1'；与 --sweep_max_iter 同时指定时启用网格搜索")
    parser.add_argument("--sweep_max_iter", type=str, default=None, help="max_iter 列表，逗号或空格分隔，如 '5,10,20'")
    parser.add_argument("--sweep_init_threshold", type=str, default=None, help="init_threshold 列表，如 '0.05,0.1,0.2'；与 sweep 同时指定时加入网格")
    parser.add_argument("--sweep_out_dir", type=str, default=None, help="网格搜索时图保存目录，默认与 out_json 同目录")
    parser.add_argument("--use_calibration", action="store_true", help="sweep/对比在校准集上进行（按类别分层采样，与 eval_fps_radii 一致）；默认在 partition 上 shuffle 取前 num_batches 样本")
    parser.add_argument("--samples_per_class", type=int, default=50, help="--use_calibration 时每类采样数")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 预取进程数，>0 可加速数据加载")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 网格搜索模式：探索 tol、max_iter、init_threshold 对 KNN IoU 及 iters_used/thresholds 的影响
    if args.sweep_tol is not None and args.sweep_max_iter is not None:
        tol_list = parse_float_list(args.sweep_tol)
        max_iter_list = parse_int_list(args.sweep_max_iter)
        if not tol_list or not max_iter_list:
            raise ValueError("--sweep_tol 与 --sweep_max_iter 至少各包含一个值")
        init_threshold_list = parse_float_list(args.sweep_init_threshold) if args.sweep_init_threshold else None
        if init_threshold_list is not None and len(init_threshold_list) == 0:
            init_threshold_list = None
        print(f"[Sweep] partition={args.partition}, use_calibration={args.use_calibration}, tol={tol_list}, max_iter={max_iter_list}, init_threshold={init_threshold_list}, num_batches={args.num_batches}, device={device}")
        grid = run_sweep_tol_max_iter(
            tol_list=tol_list,
            max_iter_list=max_iter_list,
            init_threshold_list=init_threshold_list,
            partition=args.partition,
            num_points=args.num_points,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            K=args.K,
            k_neighbors=args.k_neighbors,
            seed=args.seed,
            device=device,
            fps_seed=args.fps_seed,
            use_calibration=args.use_calibration,
            samples_per_class=args.samples_per_class,
            num_workers=args.num_workers,
        )
        out_json = args.out_json or "opt_fps/figures/sweep_tol_max_iter.json"
        out_dir = args.sweep_out_dir or os.path.dirname(out_json) or "."
        os.makedirs(out_dir, exist_ok=True)
        sweep_result = {
            "config": {
                "partition": args.partition,
                "use_calibration": args.use_calibration,
                "samples_per_class": args.samples_per_class,
                "num_points": args.num_points,
                "num_batches": args.num_batches,
                "K": args.K,
                "k_neighbors": args.k_neighbors,
                "fps_seed": args.fps_seed,
                "seed": args.seed,
                "tol_list": tol_list,
                "max_iter_list": max_iter_list,
                "init_threshold_list": init_threshold_list,
            },
            "grid": grid,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(sweep_result, f, indent=2, ensure_ascii=False)
        print(f"Written to {out_json}")
        plot_sweep_results(grid, tol_list, max_iter_list, init_threshold_list=init_threshold_list, out_dir=out_dir, prefix="sweep_tol_max_iter")
        return

    print(f"[FPS-KNN compare] partition={args.partition}, use_calibration={args.use_calibration}, num_batches={args.num_batches}, K={args.K}, k_neighbors={args.k_neighbors}, fps_seed={args.fps_seed}, device={device}")
    res = run_comparison(
        partition=args.partition,
        num_points=args.num_points,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        K=args.K,
        k_neighbors=args.k_neighbors,
        seed=args.seed,
        device=device,
        fps_seed=args.fps_seed,
        save_indices_n=args.save_indices_n,
        use_calibration=args.use_calibration,
        samples_per_class=args.samples_per_class,
        num_workers=args.num_workers,
    )
    msg = (
        f"FPS-KNN comparison (vanilla vs adaptive) on ModelNet40 {args.partition}\n"
        f"  FPS index IoU: {res['fps_iou_mean']:.4f} ± {res['fps_iou_std']:.4f}\n"
        f"  KNN index IoU: {res['knn_iou_mean']:.4f} ± {res['knn_iou_std']:.4f}\n"
        f"  Samples: {res['num_samples']}\n"
    )
    print(msg)
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
        with open(args.out_txt, "w") as f:
            f.write(msg)
        print(f"Written to {args.out_txt}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        out = {
            "config": {
                "partition": args.partition,
                "use_calibration": args.use_calibration,
                "samples_per_class": args.samples_per_class,
                "num_points": args.num_points,
                "num_samples": res["num_samples"],
                "K": args.K,
                "k_neighbors": args.k_neighbors,
                "fps_seed": args.fps_seed,
                "seed": args.seed,
            },
            "summary": {
                "fps_iou_mean": res["fps_iou_mean"],
                "fps_iou_std": res["fps_iou_std"],
                "knn_iou_mean": res["knn_iou_mean"],
                "knn_iou_std": res["knn_iou_std"],
                "iters_used_mean": res["iters_used_mean"],
                "iters_used_std": res["iters_used_std"],
                "thresholds_mean": res["thresholds_mean"],
                "thresholds_std": res["thresholds_std"],
            },
            "per_sample_fps_iou": res["per_sample_fps_iou"],
            "per_sample_knn_iou": res["per_sample_knn_iou"],
        }
        if res["saved_indices"]:
            out["saved_indices"] = res["saved_indices"]
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Written to {args.out_json}")


if __name__ == "__main__":
    main()
