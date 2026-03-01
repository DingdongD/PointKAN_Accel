#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理验证与 Group-Rational KAN 分组因果曲线 f(x) 可视化

功能概述：
1. 加载 best_checkpoint.pth，在 ModelNet40 测试集上做推理并汇报准确率、损失等指标。
2. 使用校准集（train/test 子集）前向传播，收集每一层 KAT_Group 的输入激活范围 [min, max]，
   作为 f(x) 绘图的 x 轴动态范围。
3. 可视化每一层 KAT 的分组因果曲线 f(x) = P(x)/Q(x)，即有理函数在真实输入范围内的形状。

使用示例：
    python inference_and_kan_analysis.py --checkpoint checkpoints/pointKANElite-pointkan_elite_exp-42
    python inference_and_kan_analysis.py --checkpoint ... --no_vis
    python inference_and_kan_analysis.py --checkpoint ... --calibration_partition test --calibration_samples 300
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn.metrics as metrics

# 将当前任务目录与项目根目录加入 path，便于导入 models / data / utils 以及 kat_rational
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（PointKAN-elite），用于 kat_rational 等包
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from data import ModelNet40
import models as models
from utils import cal_loss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PointKAN-Elite 推理与 KAN 因果多项式分析")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pointKANElite-pointkan_elite_exp-42",
        help="检查点目录（含 best_checkpoint.pth）或直接指定 best_checkpoint.pth 路径",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=1024,
        help="点云采样点数，需与训练时一致",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="推理时 batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader 的 num_workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="若指定，则只做推理不生成可视化",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="可视化结果保存目录，默认与 checkpoint 同目录下的 kan_analysis",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备",
    )
    # 校准集：用于收集每一层 KAT 的输入激活范围，从而确定 f(x) 的绘制区间
    parser.add_argument(
        "--calibration_partition",
        type=str,
        default="train",
        choices=("train", "test"),
        help="校准集来自 train 或 test",
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=500,
        help="校准集样本数，用于统计每层 KAT 输入的最小/最大值",
    )
    parser.add_argument(
        "--calibration_margin",
        type=float,
        default=0.05,
        help="绘制 f(x) 时在校准范围两侧扩展的比例，如 0.05 表示左右各扩展 5%%",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="将激活范围与组间相似性/压缩分析结果写入该 JSON 文件，默认与 out_dir 下的 kan_analysis_results.json",
    )
    parser.add_argument(
        "--export_grf_json",
        type=str,
        default=None,
        help="将每层每组 GRF 的 numerator/denominator 系数导出到该 JSON 文件，便于从 JSON 直接还原 GRF-KAN",
    )
    return parser.parse_args()


def get_checkpoint_path(checkpoint_arg):
    """
    根据传入的 checkpoint 参数，解析出 best_checkpoint.pth 的绝对路径。
    若传入的是目录，则取目录下的 best_checkpoint.pth；
    若传入的是 .pth 文件路径，则直接使用。
    """
    path = os.path.abspath(checkpoint_arg)
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
    if os.path.isdir(path):
        candidate = os.path.join(path, "best_checkpoint.pth")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"目录下未找到 best_checkpoint.pth: {path}")
    raise FileNotFoundError(f"检查点路径不存在: {path}")


def load_state_dict_for_model(checkpoint_path, device):
    """
    从 checkpoint 中加载 net 的 state_dict。
    若保存时使用了 DataParallel，state_dict 的 key 会带 "module." 前缀；
    本函数返回去掉 "module." 后的 state_dict，便于加载到单卡模型上。
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("net", ckpt)
    # 若为 DataParallel 保存的，则去掉 "module." 前缀
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def build_model_and_load_checkpoint(checkpoint_path, device):
    """
    构建与训练时一致的 pointKANElite 模型，并加载 best 权重。
    返回：加载好权重的模型（单卡，已 eval）。
    """
    net = models.__dict__["pointKANElite"]()
    state = load_state_dict_for_model(checkpoint_path, device)
    net.load_state_dict(state, strict=True)
    net = net.to(device)
    net.eval()
    return net


def run_evaluation(net, test_loader, criterion, device):
    """
    在测试集上做一次完整推理，计算 loss、accuracy、balanced accuracy。
    与 main.py 中 validate 逻辑一致。
    """
    net.eval()
    test_loss = 0.0
    test_true = []
    test_pred = []
    total = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device).squeeze()
            data = data.permute(0, 2, 1)  # [B, 3, N]
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.cpu().numpy())
            total += label.size(0)
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    avg_loss = test_loss / max(len(test_loader), 1)
    acc = float(metrics.accuracy_score(test_true, test_pred) * 100.0)
    acc_avg = float(metrics.balanced_accuracy_score(test_true, test_pred) * 100.0)
    return {
        "loss": avg_loss,
        "acc": acc,
        "acc_avg": acc_avg,
        "num_samples": total,
    }


def collect_kan_polynomial_weights(state_dict):
    """
    从 state_dict 中收集所有 KAT_Group 的因果多项式系数。

    Group-Rational KAN 的“因果多项式”含义：
    将输入按通道分成 num_groups 组，每组独立地通过有理函数 P(x)/Q(x) 映射，
    系数即刻画各组（因果）的理性近似形状。

    数学形式（与 kat_1dgroup_torch / rational_triton 一致）：
    - 分子 P(x) = a_0 + a_1*x + ... + a_5*x^5  （6 个系数，order 5）
    - 分母 Q(x) = 1 + |b_1|*x + |b_2|*x^2 + ... （4 个系数，保证分母正性）

    state_dict 中对应：
    - weight_numerator: shape (1, 6) 或 (num_groups, 6)，表示分子系数
    - weight_denominator: shape (num_groups, 4)，表示分母系数（每组一组系数）

    返回：列表，每项为 dict：
        {
            "name": "LFP_blocks_list.0.operation1.0.act2",
            "weight_numerator": np.ndarray shape (6,) or (num_groups, 6),
            "weight_denominator": np.ndarray shape (num_groups, 4),
        }
    """
    result = []
    for key in sorted(state_dict.keys()):
        if "weight_numerator" in key and "act" in key:
            # 成对出现：同一子模块下有 weight_numerator 与 weight_denominator
            base_key = key.replace(".weight_numerator", "")
            denom_key = base_key + ".weight_denominator"
            if denom_key not in state_dict:
                continue
            num_t = state_dict[key].detach().cpu().numpy()
            den_t = state_dict[denom_key].detach().cpu().numpy()
            result.append({
                "name": base_key,
                "weight_numerator": num_t,
                "weight_denominator": den_t,
            })
    return result


def collect_activation_ranges(net, calibration_loader, device, max_samples=500):
    """
    在校准集上前向传播，通过 forward_pre_hook 收集每一层 KAT_Group 的输入激活范围。
    返回 dict: layer_name -> {"min": float, "max": float}，用于后续在真实动态范围内绘制 f(x)。
    """
    activation_ranges = {}
    hooks = []

    def make_hook(name):
        def hook(module, input):
            t = input[0].detach()
            if name not in activation_ranges:
                activation_ranges[name] = {"min": float("inf"), "max": float("-inf")}
            low, high = t.min().item(), t.max().item()
            activation_ranges[name]["min"] = min(activation_ranges[name]["min"], low)
            activation_ranges[name]["max"] = max(activation_ranges[name]["max"], high)
        return hook

    for name, mod in net.named_modules():
        if hasattr(mod, "weight_numerator") and hasattr(mod, "weight_denominator"):
            h = mod.register_forward_pre_hook(make_hook(name))
            hooks.append(h)

    net.eval()
    with torch.no_grad():
        n_done = 0
        for data, _ in calibration_loader:
            if n_done >= max_samples:
                break
            data = data.to(device).permute(0, 2, 1)
            _ = net(data)
            n_done += data.size(0)

    for h in hooks:
        h.remove()

    return activation_ranges


def eval_rational_fx(x, weight_numerator, weight_denominator):
    """
    在标量数组 x 上计算 f(x) = P(x)/Q(x)。
    - weight_numerator: (1, 6) 或 (G, 6)，对应 P(x) = a0 + a1*x + ... + a5*x^5
    - weight_denominator: (G, 4)，对应 Q(x) = 1 + |b1|*x + |b2|*x^2 + |b3|*x^3 + |b4|*x^4
    返回: (n_points,) 若分子为 (1,6)；或 (n_groups, n_points) 若分子为 (G,6)，这里统一返回 (n_groups, n_points)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    num_w = np.asarray(weight_numerator, dtype=np.float64)
    den_w = np.asarray(weight_denominator, dtype=np.float64)
    if num_w.ndim == 1:
        num_w = num_w[np.newaxis, :]
    n_groups = den_w.shape[0]
    # 分子若为 (1, 6) 则所有组共用同一组系数；若为 (G, 6) 则按组取
    num_rows = num_w.shape[0]
    # P: 1, x, x^2, ..., x^5  -> (6, n_pts)
    xp_num = np.stack([x**k for k in range(6)], axis=0)
    # Q: 1, x, x^2, x^3, x^4  -> (5, n_pts)，首项为 1
    xp_den = np.stack([x**k for k in range(5)], axis=0)
    out = []
    for g in range(n_groups):
        num_row = num_w[min(g, num_rows - 1)]
        P = num_row @ xp_num
        q_coeff = np.concatenate([[1.0], np.abs(den_w[g])])
        Q = q_coeff @ xp_den
        Q = np.maximum(Q, 1e-8)
        out.append(P / Q)
    return np.stack(out, axis=0)


def compute_layer_similarity_and_compression(item, activation_ranges, args, n_pts=300):
    """
    对单层 KAT：在校准 x 范围上计算各组 f(x)，得到组间相似性与可压缩性统计。
    返回可 JSON 序列化的 dict（含 correlation 矩阵、PCA 解释方差、相对均值的偏差等）。
    """
    name = item["name"]
    num_w = item["weight_numerator"]
    den_w = item["weight_denominator"]
    if num_w.ndim == 1:
        num_w = num_w[np.newaxis, :]
    margin = getattr(args, "calibration_margin", 0.05)
    x_min, x_max = -3.0, 3.0
    if activation_ranges and name in activation_ranges:
        r = activation_ranges[name]
        lo, hi = r["min"], r["max"]
        span = max(hi - lo, 1e-6)
        x_min = lo - margin * span
        x_max = hi + margin * span
    x = np.linspace(x_min, x_max, n_pts)
    fg = eval_rational_fx(x, num_w, den_w)  # (n_groups, n_pts)
    G, N = fg.shape

    # 组间相似性：Pearson（保留）、余弦相似性、MSE（更直观：形状相似度 + 数值差异）
    corr_pearson = np.corrcoef(fg)
    if not np.isfinite(corr_pearson).all():
        corr_pearson = np.nan_to_num(corr_pearson, nan=0.0, posinf=0.0, neginf=0.0)
    # 余弦相似性：把每条曲线当向量，cos_sim(i,j)=dot(fi,fj)/(||fi||*||fj||)，[-1,1]，越接近1越相似
    norms = np.linalg.norm(fg, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    cos_sim = (fg @ fg.T) / (norms * norms.T)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    if not np.isfinite(cos_sim).all():
        cos_sim = np.nan_to_num(cos_sim, nan=0.0, posinf=1.0, neginf=-1.0)
    # MSE：逐点均方误差，越小越相似，直接反映数值差异（对 base+残差 存法有意义）
    mse_mat = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            mse_mat[i, j] = float(np.mean((fg[i] - fg[j]) ** 2))
    triu = np.triu_indices(G, k=1)
    pairwise_pearson = corr_pearson[triu]
    pairwise_cos = cos_sim[triu]
    pairwise_mse = mse_mat[triu]
    mean_corr = float(np.mean(pairwise_pearson))
    min_corr = float(np.min(pairwise_pearson))
    mean_cos = float(np.mean(pairwise_cos))
    min_cos = float(np.min(pairwise_cos))
    mean_mse = float(np.mean(pairwise_mse))
    max_mse = float(np.max(pairwise_mse))

    # 均值曲线与各组相对均值的 L2 偏差（用于评估“用一条曲线+小残差”的压缩潜力）
    # 注：L2 为 300 个 x 点上差值的向量范数，>1 表示在 300 点上累积；除以 sqrt(N) 得“每点平均偏差”
    mean_curve = fg.mean(axis=0)
    l2_deviations = [float(np.linalg.norm(fg[g] - mean_curve)) for g in range(G)]
    mean_l2_dev = float(np.mean(l2_deviations))
    max_l2_dev = float(np.max(l2_deviations))
    mean_l2_dev_per_point = mean_l2_dev / (N ** 0.5) if N > 0 else 0.0

    # PCA：在 (n_groups, n_points) 上做，看多少主成分可解释 95% 方差 → 压缩表示所需维度（最多 G 个）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(G, N))
    pca.fit(fg)
    evr = [float(v) for v in pca.explained_variance_ratio_]
    cumsum = np.cumsum(evr)
    idx_95 = int(np.searchsorted(cumsum, 0.95)) if len(cumsum) > 0 else 0
    n_95 = min(idx_95 + 1, len(evr))

    # ----- 输出动态范围：vanilla 存每组 f(x) vs base+残差 存法 -----
    # vanilla：直接存每层每组 f(x) 时，输出值落在的区间
    f_min, f_max = float(fg.min()), float(fg.max())
    vanilla_span = f_max - f_min

    # base + 残差：base 取 mean / max / min 曲线时，base 与残差各自的动态范围
    def _range(arr):
        a, b = float(np.min(arr)), float(np.max(arr))
        return {"min": round(a, 6), "max": round(b, 6), "span": round(b - a, 6)}

    max_curve = fg.max(axis=0)   # 逐点取各组最大
    min_curve = fg.min(axis=0)   # 逐点取各组最小
    res_mean = fg - mean_curve
    res_max_base = fg - max_curve
    res_min_base = fg - min_curve

    output_dynamic_range = {
        "vanilla": {
            "min": round(f_min, 6), "max": round(f_max, 6), "span": round(vanilla_span, 6),
            "description": "per-layer per-group store f(x) as-is",
        },
        "base_plus_residual": {
            "mean_base": {
                "base": _range(mean_curve),
                "residual": _range(res_mean),
                "description": "base=mean_curve, residual=group_curve-base",
            },
            "max_base": {
                "base": _range(max_curve),
                "residual": _range(res_max_base),
                "description": "base=pointwise max over groups, residual<=0",
            },
            "min_base": {
                "base": _range(min_curve),
                "residual": _range(res_min_base),
                "description": "base=pointwise min over groups, residual>=0",
            },
        },
    }

    return {
        "pairwise_correlation_mean": round(mean_corr, 6),
        "pairwise_correlation_min": round(min_corr, 6),
        "correlation_matrix": [[round(corr_pearson[i, j], 4) for j in range(G)] for i in range(G)],
        "cosine_similarity_matrix": [[round(cos_sim[i, j], 4) for j in range(G)] for i in range(G)],
        "pairwise_cosine_mean": round(mean_cos, 6),
        "pairwise_cosine_min": round(min_cos, 6),
        "mse_matrix": [[round(mse_mat[i, j], 6) for j in range(G)] for i in range(G)],
        "pairwise_mse_mean": round(mean_mse, 6),
        "pairwise_mse_max": round(max_mse, 6),
        "mean_curve_l2_deviations": [round(d, 6) for d in l2_deviations],
        "mean_l2_deviation": round(mean_l2_dev, 6),
        "mean_l2_deviation_per_point": round(mean_l2_dev_per_point, 6),
        "max_l2_deviation": round(max_l2_dev, 6),
        "pca_explained_variance_ratio": [round(v, 6) for v in evr],
        "n_components_for_95pct": n_95,
        "output_dynamic_range": output_dynamic_range,
        "compression_suggestion": f"mean_curve + {n_95} PC(s) can explain 95% variance; high correlation suggests storing 1 curve + small per-group residuals.",
    }


def plot_grf_kan_similarity(similarity_per_layer, out_dir):
    """
    组间 GRF-KAN 相似性绘图：每层组间相关热力图 + 跨层总览（平均相关、95%% 方差所需主成分数）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过相似性绘图。")
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    names = [n for n, v in similarity_per_layer.items() if "error" not in v]
    if not names:
        return

    # 1) 每层：组间余弦相似性热力图 + MSE 热力图（余弦看形状相似度，MSE 看数值差异）
    for i, name in enumerate(names):
        stats = similarity_per_layer[name]
        has_cos = "cosine_similarity_matrix" in stats
        has_mse = "mse_matrix" in stats
        if not has_cos and "correlation_matrix" not in stats:
            continue
        G = len(stats.get("cosine_similarity_matrix", stats.get("correlation_matrix", [[]])))
        if G == 0:
            continue
        fig, axes = plt.subplots(1, 2 if (has_cos and has_mse) else 1, figsize=(10, 4) if (has_cos and has_mse) else (6, 5))
        ax0 = axes[0] if (has_cos and has_mse) else axes
        ax1 = axes[1] if (has_cos and has_mse) else None
        # 余弦相似性
        if has_cos:
            cos_mat = np.array(stats["cosine_similarity_matrix"])
            im0 = ax0.imshow(cos_mat, aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
            ax0.set_xticks(range(G))
            ax0.set_yticks(range(G))
            ax0.set_xticklabels([f"g{j}" for j in range(G)])
            ax0.set_yticklabels([f"g{j}" for j in range(G)])
            for i0 in range(G):
                for j0 in range(G):
                    ax0.text(j0, i0, f"{cos_mat[i0, j0]:.2f}", ha="center", va="center", fontsize=7)
            ax0.set_xlabel("Group")
            ax0.set_ylabel("Group")
            ax0.set_title(f"Cosine similarity: {name}\n(mean={stats.get('pairwise_cosine_mean', 0):.3f})")
            plt.colorbar(im0, ax=ax0, label="cosine sim")
        # MSE
        if has_mse:
            mse_mat = np.array(stats["mse_matrix"])
            ax = ax1 if (has_cos and has_mse) else ax0
            im1 = ax.imshow(mse_mat, aspect="auto", cmap="viridis_r")
            ax.set_xticks(range(G))
            ax.set_yticks(range(G))
            ax.set_xticklabels([f"g{j}" for j in range(G)])
            ax.set_yticklabels([f"g{j}" for j in range(G)])
            for i0 in range(G):
                for j0 in range(G):
                    ax.text(j0, i0, f"{mse_mat[i0, j0]:.3f}", ha="center", va="center", fontsize=6)
            ax.set_xlabel("Group")
            ax.set_ylabel("Group")
            ax.set_title(f"MSE (mean={stats.get('pairwise_mse_mean', 0):.4f})")
            plt.colorbar(im1, ax=ax, label="MSE")
        plt.tight_layout()
        safe_name = name.replace(".", "_")
        fig.savefig(os.path.join(out_dir, f"kan_similarity_corr_{i:02d}_{safe_name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 2) 总览：各层平均组间余弦相似性 + 平均 MSE + 达到 95% 方差所需主成分数
    mean_cos_s = [similarity_per_layer[n].get("pairwise_cosine_mean", similarity_per_layer[n].get("pairwise_correlation_mean", np.nan)) for n in names]
    mean_mse_s = [similarity_per_layer[n].get("pairwise_mse_mean", np.nan) for n in names]
    n_95s = [similarity_per_layer[n].get("n_components_for_95pct", np.nan) for n in names]
    x = np.arange(len(names))
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1.bar(x - 0.2, mean_cos_s, width=0.4, label="mean pairwise cosine similarity", color="steelblue", alpha=0.8)
    ax1.axhline(0, color="k", linewidth=0.5)
    ax1.set_ylabel("Mean cosine similarity")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.bar(x, mean_mse_s, width=0.4, label="mean pairwise MSE", color="coral", alpha=0.8)
    ax2.set_ylabel("Mean MSE")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax3.bar(x + 0.2, n_95s, width=0.4, label="n_components for 95% variance", color="seagreen", alpha=0.8)
    ax3.set_ylabel("Number of components")
    ax3.set_xlabel("Layer")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    fig2.suptitle("GRF-KAN inter-group similarity overview (cosine + MSE)", fontsize=12)
    plt.tight_layout()
    fig2.savefig(os.path.join(out_dir, "kan_similarity_overview.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # 3) 各组相对均值曲线的 L2 偏差（每层一图，条形图）
    for i, name in enumerate(names):
        stats = similarity_per_layer[name]
        if "mean_curve_l2_deviations" not in stats:
            continue
        devs = stats["mean_curve_l2_deviations"]
        fig3, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(devs)), devs, color="teal", alpha=0.7)
        ax.set_xlabel("Group")
        ax.set_ylabel("L2 deviation from mean curve")
        ax.set_title(f"{name}")  # Per-group deviation from mean f(x)
        ax.set_xticks(range(len(devs)))
        ax.set_xticklabels([f"g{j}" for j in range(len(devs))])
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        safe_name = name.replace(".", "_")
        fig3.savefig(os.path.join(out_dir, f"kan_similarity_l2dev_{i:02d}_{safe_name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig3)

    # 4) 输出动态范围对比：vanilla vs base+残差（mean/max/min base）各层 span 对比
    if any("output_dynamic_range" in similarity_per_layer.get(n, {}) for n in names):
        vanilla_spans = []
        res_mean_spans = []
        res_max_spans = []
        res_min_spans = []
        for n in names:
            odr = similarity_per_layer[n].get("output_dynamic_range", {})
            if not odr:
                vanilla_spans.append(np.nan)
                res_mean_spans.append(np.nan)
                res_max_spans.append(np.nan)
                res_min_spans.append(np.nan)
                continue
            vanilla_spans.append(odr["vanilla"]["span"])
            bpr = odr.get("base_plus_residual", {})
            res_mean_spans.append(bpr.get("mean_base", {}).get("residual", {}).get("span", np.nan))
            res_max_spans.append(bpr.get("max_base", {}).get("residual", {}).get("span", np.nan))
            res_min_spans.append(bpr.get("min_base", {}).get("residual", {}).get("span", np.nan))
        x = np.arange(len(names))
        w = 0.2
        fig4, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - 1.5 * w, vanilla_spans, width=w, label="vanilla (store all f(x))", color="gray", alpha=0.8)
        ax.bar(x - 0.5 * w, res_mean_spans, width=w, label="residual (base=mean)", color="steelblue", alpha=0.8)
        ax.bar(x + 0.5 * w, res_max_spans, width=w, label="residual (base=max)", color="coral", alpha=0.8)
        ax.bar(x + 1.5 * w, res_min_spans, width=w, label="residual (base=min)", color="seagreen", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Output dynamic range (span)")
        ax.set_xlabel("Layer")
        ax.set_title("Per-layer output range: vanilla vs base+residual storage")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig4.savefig(os.path.join(out_dir, "kan_output_dynamic_range_comparison.png"), dpi=120, bbox_inches="tight")
        plt.close(fig4)

    print(f"组间 GRF-KAN 相似性图已保存到: {out_dir}")


def build_and_save_kan_analysis_json(kan_weights_list, activation_ranges, args, out_dir, similarity_per_layer=None):
    """
    汇总每层输入激活范围与组间 GRF-KAN 相似性/压缩分析，写入 JSON。
    similarity_per_layer 若已由外部计算可传入，否则内部计算。
    """
    # 激活范围：layer_name -> { min, max }
    activation_ranges_serializable = {}
    for name, r in (activation_ranges or {}).items():
        activation_ranges_serializable[name] = {
            "min": round(float(r["min"]), 6),
            "max": round(float(r["max"]), 6),
        }

    if similarity_per_layer is None:
        similarity_per_layer = {}
        for item in kan_weights_list:
            name = item["name"]
            try:
                similarity_per_layer[name] = compute_layer_similarity_and_compression(
                    item, activation_ranges, args
                )
            except Exception as e:
                similarity_per_layer[name] = {"error": str(e)}

    payload = {
        "activation_ranges": activation_ranges_serializable,
        "similarity_and_compression": similarity_per_layer,
    }

    path = args.json_output
    if not path:
        path = os.path.join(out_dir, "kan_analysis_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"激活范围与相似性/压缩分析已写入: {path}")
    return path


def plot_kan_fx_curves(kan_weights_list, activation_ranges, out_dir, args):
    """
    可视化每一层 KAT 的分组因果曲线 f(x) = P(x)/Q(x)。
    x 轴使用校准集统计得到的该层输入激活范围，使曲线落在真实动态范围内。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过可视化。")
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    margin = getattr(args, "calibration_margin", 0.05)
    n_pts = 300

    for i, item in enumerate(kan_weights_list):
        name = item["name"]
        num_w = item["weight_numerator"]
        den_w = item["weight_denominator"]
        if num_w.ndim == 1:
            num_w = num_w[np.newaxis, :]
        n_groups = den_w.shape[0]
        x_min, x_max = None, None
        if activation_ranges and name in activation_ranges:
            r = activation_ranges[name]
            lo, hi = r["min"], r["max"]
            span = max(hi - lo, 1e-6)
            x_min = lo - margin * span
            x_max = hi + margin * span
        if x_min is None or x_max is None:
            x_min, x_max = -3.0, 3.0
        x = np.linspace(x_min, x_max, n_pts)
        fg = eval_rational_fx(x, num_w, den_w)
        mean_curve = fg.mean(axis=0)

        fig, ax = plt.subplots(figsize=(7, 4))
        for g in range(n_groups):
            ax.plot(x, fg[g], "-", label=f"group {g}", alpha=0.85)
        ax.plot(x, mean_curve, "k--", linewidth=2.5, label="mean", zorder=10, alpha=0.95)
        ax.set_xlabel("x (input activation, calibrated range)")
        ax.set_ylabel("f(x) = P(x)/Q(x)")
        ax.set_title(f"KAT_Group: {name} — per-group curves + mean")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="k", linewidth=0.5)
        safe_name = name.replace(".", "_")
        fig.savefig(os.path.join(out_dir, f"kan_fx_{i:02d}_{safe_name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"f(x) 曲线图已保存到: {out_dir}")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = get_checkpoint_path(args.checkpoint)
    device = torch.device(args.device)
    print(f"加载检查点: {checkpoint_path}")

    # 1) 构建模型并加载权重
    net = build_model_and_load_checkpoint(checkpoint_path, device)

    # 2) 准备测试集并推理
    test_dataset = ModelNet40(partition="test", num_points=args.num_points)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    eval_result = run_evaluation(net, test_loader, cal_loss, device)
    print("\n" + "=" * 50)
    print("推理验证结果 (ModelNet40 Test)")
    print("=" * 50)
    print(f"  测试样本数: {eval_result['num_samples']}")
    print(f"  Test Loss: {eval_result['loss']:.4f}")
    print(f"  Test Acc (accuracy): {eval_result['acc']:.2f}%")
    print(f"  Test Acc (balanced): {eval_result['acc_avg']:.2f}%")
    print("=" * 50 + "\n")

    # 3) 提取 KAT_Group 因果多项式系数（直接从 state_dict，避免再次加载）
    state_dict = load_state_dict_for_model(checkpoint_path, device)
    kan_weights_list = collect_kan_polynomial_weights(state_dict)
    print(f"共提取到 {len(kan_weights_list)} 个 KAT_Group 的因果多项式系数。")
    for w in kan_weights_list:
        print(f"  - {w['name']}: numerator shape {w['weight_numerator'].shape}, denominator shape {w['weight_denominator'].shape}")

    # 4) 校准集收集每层 KAT 输入激活范围（用于 f(x) 绘图与相似性分析的 x 轴；需校准时执行）
    activation_ranges = {}
    need_calibration = (not args.no_vis or args.json_output) and len(kan_weights_list) > 0
    if need_calibration:
        cal_dataset = ModelNet40(partition=args.calibration_partition, num_points=args.num_points)
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=min(args.batch_size, 32),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        if net is None:
            net = build_model_and_load_checkpoint(checkpoint_path, device)
        activation_ranges = collect_activation_ranges(
            net, cal_loader, device, max_samples=args.calibration_samples
        )
        print(f"校准集 ({args.calibration_partition}, n={args.calibration_samples}) 统计得到各层输入范围:")
        for name, r in sorted(activation_ranges.items()):
            print(f"  - {name}: x in [{r['min']:.4f}, {r['max']:.4f}]")

    # 5) 组间相似性与压缩分析（用于 JSON 与相似性绘图，只算一次）
    similarity_per_layer = None
    if len(kan_weights_list) > 0 and (args.json_output or not args.no_vis):
        if not activation_ranges and args.json_output:
            print("未运行校准，相似性分析将使用默认 x 范围 [-3, 3]。")
        similarity_per_layer = {}
        for item in kan_weights_list:
            name = item["name"]
            try:
                similarity_per_layer[name] = compute_layer_similarity_and_compression(
                    item, activation_ranges, args
                )
            except Exception as e:
                similarity_per_layer[name] = {"error": str(e)}

    # 6) 可视化：f(x) 曲线 + 组间相似性图
    out_dir = args.out_dir or os.path.join(os.path.dirname(checkpoint_path), "kan_analysis")
    if not args.no_vis and len(kan_weights_list) > 0:
        plot_kan_fx_curves(kan_weights_list, activation_ranges, out_dir, args)
        if similarity_per_layer:
            plot_grf_kan_similarity(similarity_per_layer, out_dir)
    elif args.no_vis:
        print("已指定 --no_vis，跳过可视化。")

    # 7) JSON 输出：激活范围 + 每层组间相似性与压缩表示分析
    if len(kan_weights_list) > 0 and (args.json_output or not args.no_vis) and similarity_per_layer is not None:
        build_and_save_kan_analysis_json(
            kan_weights_list, activation_ranges, args, out_dir,
            similarity_per_layer=similarity_per_layer,
        )
    # 可选：导出每层每组 GRF 系数到独立 JSON，便于从 JSON 直接还原 GRF-KAN
    if getattr(args, "export_grf_json", None) and len(kan_weights_list) > 0:
        try:
            from grf_kan_json_and_residual import export_grf_coefficients_to_json
            export_grf_coefficients_to_json(kan_weights_list, activation_ranges or {}, args.export_grf_json)
            print(f"GRF 系数已导出到: {args.export_grf_json}")
        except Exception as e:
            print(f"导出 GRF JSON 失败: {e}")


# ---------------------------------------------------------------------------
# 可视化结果解读（供分析时参考）
# ---------------------------------------------------------------------------
# 输出图为 kan_fx_XX_<层名>.png：每一层 KAT_Group 一张图，横轴 x 为该校准得到的输入激活范围，
# 纵轴为 f(x)=P(x)/Q(x)。每条曲线对应一个分组（group），即该层的“分组因果曲线”。
# 若某层各组曲线差异大，说明该层不同通道组学到的激活形状分化明显。
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
