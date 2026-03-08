#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part Segmentation 推理验证与 Group-Rational KAN 分组因果曲线 f(x) 可视化

功能概述：
1. 加载 best_insiou_model.pth（或 best_checkpoint.pth），在 PartNormal 测试集上做推理，
   汇报 loss、accuracy、instance mIoU。
2. 使用校准集前向传播，收集每一层 KAT_Group 的输入激活范围，用于 f(x) 绘图。
3. 可视化每一层 KAT 的分组因果曲线 f(x) = P(x)/Q(x)。

使用示例：
    python inference_and_kan_analysis.py --checkpoint checkpoints/pointKANElite_part_exp
    python inference_and_kan_analysis.py --checkpoint ... --no_vis
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from util.data_util import PartNormalDataset
from util.util import to_categorical, compute_overall_iou
import model as models

NUM_PART = 50
NUM_CLASSES = 16


def parse_args():
    parser = argparse.ArgumentParser(description="PointKAN-Elite Part Segmentation 推理与 KAN 因果多项式分析")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/pointKANElite_part_exp",
        help="检查点目录（含 best_insiou_model.pth 或 best_checkpoint.pth）或直接指定 .pth 路径",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="点云采样点数，需与训练时一致",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="推理时 batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="测试集 batch size，默认与 batch_size 相同",
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
        help="随机种子",
    )
    parser.add_argument("--no_vis", action="store_true", help="只做推理不生成可视化")
    parser.add_argument("--out_dir", type=str, default=None, help="可视化结果保存目录")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备",
    )
    parser.add_argument(
        "--calibration_partition",
        type=str,
        default="train",
        choices=("train", "test"),
        help="校准集来自 train(trainval) 或 test",
    )
    parser.add_argument("--calibration_samples", type=int, default=500, help="校准集样本数")
    parser.add_argument("--calibration_margin", type=float, default=0.05, help="f(x) 绘制范围扩展比例")
    parser.add_argument("--json_output", type=str, default=None, help="激活范围与相似性结果 JSON 路径")
    parser.add_argument(
        "--eval_fps_opt",
        action="store_true",
        help="（Part Segmentation 暂未实现 FPS 优化版，此选项仅打印提示）",
    )
    return parser.parse_args()


def get_checkpoint_path(checkpoint_arg):
    path = os.path.abspath(checkpoint_arg)
    if os.path.isfile(path) and path.endswith(".pth"):
        return path
    if os.path.isdir(path):
        for name in ("best_insiou_model.pth", "best_checkpoint.pth"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"目录下未找到 best_insiou_model.pth 或 best_checkpoint.pth: {path}")
    raise FileNotFoundError(f"检查点路径不存在: {path}")


def load_state_dict_for_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def build_model_and_load_checkpoint(checkpoint_path, device):
    net = models.__dict__["pointKANElite"](num_classes=NUM_PART)
    state = load_state_dict_for_model(checkpoint_path, device)
    net.load_state_dict(state, strict=True)
    net = net.to(device)
    net.eval()
    return net


def run_evaluation(net, test_loader, device):
    """Part Segmentation 测试集评估：loss、accuracy、instance mIoU。"""
    net.eval()
    test_loss = 0.0
    count = 0.0
    accuracy_list = []
    shape_ious_sum = 0.0
    with torch.no_grad():
        for batch_id, (points, label, target, norm_plt) in enumerate(test_loader):
            batch_size, num_point, _ = points.size()
            points = points.float().to(device).transpose(2, 1)
            norm_plt = norm_plt.float().to(device).transpose(2, 1)
            label = label.squeeze(1).long().to(device)
            target = target.long().to(device)
            onehot = to_categorical(label, NUM_CLASSES)
            seg_pred = net(points, norm_plt, onehot)
            batch_shapeious = compute_overall_iou(seg_pred, target, NUM_PART)
            shape_ious_sum += sum(batch_shapeious)
            seg_pred_flat = seg_pred.contiguous().view(-1, NUM_PART)
            target_flat = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(seg_pred_flat, target_flat)
            pred_choice = seg_pred_flat.max(1)[1]
            correct = pred_choice.eq(target_flat).sum().item()
            test_loss += loss.item() * batch_size
            count += batch_size
            accuracy_list.append(correct / (batch_size * num_point))
    n = int(count)
    avg_loss = test_loss / max(count, 1)
    acc = float(np.mean(accuracy_list) * 100.0)
    ins_iou = float(shape_ious_sum / max(count, 1))
    return {"loss": avg_loss, "acc": acc, "shape_avg_iou": ins_iou, "num_samples": n}


def collect_kan_polynomial_weights(state_dict):
    result = []
    for key in sorted(state_dict.keys()):
        if "weight_numerator" in key and "act" in key:
            base_key = key.replace(".weight_numerator", "")
            denom_key = base_key + ".weight_denominator"
            if denom_key not in state_dict:
                continue
            num_t = state_dict[key].detach().cpu().numpy()
            den_t = state_dict[denom_key].detach().cpu().numpy()
            result.append({"name": base_key, "weight_numerator": num_t, "weight_denominator": den_t})
    return result


def collect_activation_ranges_part_seg(net, calibration_loader, device, max_samples=500, num_classes=16):
    """Part Segmentation：校准集前向传播，收集每层 KAT_Group 输入激活范围。"""
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
    n_done = 0
    with torch.no_grad():
        for points, label, target, norm_plt in calibration_loader:
            if n_done >= max_samples:
                break
            points = points.float().to(device).transpose(2, 1)
            norm_plt = norm_plt.float().to(device).transpose(2, 1)
            label = label.squeeze(1).long().to(device)
            onehot = to_categorical(label, num_classes)
            _ = net(points, norm_plt, onehot)
            n_done += points.size(0)
    for h in hooks:
        h.remove()
    return activation_ranges


def eval_rational_fx(x, weight_numerator, weight_denominator):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0:
        x = x.reshape(1)
    num_w = np.asarray(weight_numerator, dtype=np.float64)
    den_w = np.asarray(weight_denominator, dtype=np.float64)
    if num_w.ndim == 1:
        num_w = num_w[np.newaxis, :]
    n_groups = den_w.shape[0]
    num_rows = num_w.shape[0]
    xp_num = np.stack([x ** k for k in range(6)], axis=0)
    xp_den = np.stack([x ** k for k in range(5)], axis=0)
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
    from sklearn.decomposition import PCA
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
    fg = eval_rational_fx(x, num_w, den_w)
    G, N = fg.shape
    corr_pearson = np.corrcoef(fg)
    corr_pearson = np.nan_to_num(corr_pearson, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(fg, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    cos_sim = (fg @ fg.T) / (norms * norms.T)
    cos_sim = np.clip(np.nan_to_num(cos_sim, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    mse_mat = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            mse_mat[i, j] = float(np.mean((fg[i] - fg[j]) ** 2))
    triu = np.triu_indices(G, k=1)
    mean_corr = float(np.mean(corr_pearson[triu]))
    mean_cos = float(np.mean(cos_sim[triu]))
    mean_mse = float(np.mean(mse_mat[triu]))
    pca = PCA(n_components=min(G, N))
    pca.fit(fg)
    evr = [float(v) for v in pca.explained_variance_ratio_]
    cumsum = np.cumsum(evr)
    idx_95 = int(np.searchsorted(cumsum, 0.95)) if len(cumsum) > 0 else 0
    n_95 = min(idx_95 + 1, len(evr))
    return {
        "pairwise_correlation_mean": round(mean_corr, 6),
        "correlation_matrix": [[round(corr_pearson[i, j], 4) for j in range(G)] for i in range(G)],
        "cosine_similarity_matrix": [[round(cos_sim[i, j], 4) for j in range(G)] for i in range(G)],
        "pairwise_cosine_mean": round(mean_cos, 6),
        "mse_matrix": [[round(mse_mat[i, j], 6) for j in range(G)] for i in range(G)],
        "pairwise_mse_mean": round(mean_mse, 6),
        "pca_explained_variance_ratio": [round(v, 6) for v in evr],
        "n_components_for_95pct": n_95,
    }


def plot_grf_kan_similarity(similarity_per_layer, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(out_dir, exist_ok=True)
    names = [n for n, v in similarity_per_layer.items() if "error" not in v]
    if not names:
        return
    for i, name in enumerate(names):
        stats = similarity_per_layer[name]
        if "cosine_similarity_matrix" not in stats:
            continue
        G = len(stats["cosine_similarity_matrix"])
        if G == 0:
            continue
        cos_mat = np.array(stats["cosine_similarity_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cos_mat, aspect="auto", cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(G))
        ax.set_yticks(range(G))
        ax.set_title(f"Cosine similarity: {name}")
        plt.colorbar(im, ax=ax)
        safe_name = name.replace(".", "_")
        fig.savefig(os.path.join(out_dir, f"kan_similarity_{i:02d}_{safe_name}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"组间 GRF-KAN 相似性图已保存到: {out_dir}")


def build_and_save_kan_analysis_json(kan_weights_list, activation_ranges, args, out_dir, similarity_per_layer=None):
    activation_ranges_serializable = {}
    for name, r in (activation_ranges or {}).items():
        activation_ranges_serializable[name] = {"min": round(float(r["min"]), 6), "max": round(float(r["max"]), 6)}
    if similarity_per_layer is None:
        similarity_per_layer = {}
        for item in kan_weights_list:
            try:
                similarity_per_layer[item["name"]] = compute_layer_similarity_and_compression(
                    item, activation_ranges, args
                )
            except Exception as e:
                similarity_per_layer[item["name"]] = {"error": str(e)}
    path = args.json_output or os.path.join(out_dir, "kan_analysis_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"activation_ranges": activation_ranges_serializable, "similarity_and_compression": similarity_per_layer}, f, indent=2, ensure_ascii=False)
    print(f"激活范围与相似性已写入: {path}")
    return path


def plot_kan_fx_curves(kan_weights_list, activation_ranges, out_dir, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(out_dir, exist_ok=True)
    margin = getattr(args, "calibration_margin", 0.05)
    n_pts = 300
    for i, item in enumerate(kan_weights_list):
        name = item["name"]
        num_w = item["weight_numerator"]
        den_w = item["weight_denominator"]
        if num_w.ndim == 1:
            num_w = num_w[np.newaxis, :]
        n_groups = den_w.shape[0]
        x_min, x_max = -3.0, 3.0
        if activation_ranges and name in activation_ranges:
            r = activation_ranges[name]
            lo, hi = r["min"], r["max"]
            span = max(hi - lo, 1e-6)
            x_min, x_max = lo - margin * span, hi + margin * span
        x = np.linspace(x_min, x_max, n_pts)
        fg = eval_rational_fx(x, num_w, den_w)
        mean_curve = fg.mean(axis=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        for g in range(n_groups):
            ax.plot(x, fg[g], "-", label=f"group {g}", alpha=0.85)
        ax.plot(x, mean_curve, "k--", linewidth=2.5, label="mean")
        ax.set_xlabel("x (input activation)")
        ax.set_ylabel("f(x) = P(x)/Q(x)")
        ax.set_title(f"KAT_Group: {name}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
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

    if getattr(args, "eval_fps_opt", False):
        print("注意: Part Segmentation 当前未实现 FPS 优化版模型，--eval_fps_opt 已忽略。")

    checkpoint_path = get_checkpoint_path(args.checkpoint)
    device = torch.device(args.device)
    print(f"加载检查点: {checkpoint_path}")

    net = build_model_and_load_checkpoint(checkpoint_path, device)

    test_batch = args.test_batch_size or args.batch_size
    test_dataset = PartNormalDataset(npoints=args.num_points, split="test", normalize=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )
    eval_result = run_evaluation(net, test_loader, device)

    print("\n" + "=" * 50)
    print("推理验证结果 (Part Segmentation Test)")
    print("=" * 50)
    print(f"  测试样本数: {eval_result['num_samples']}")
    print(f"  Test Loss: {eval_result['loss']:.4f}")
    print(f"  Test Acc: {eval_result['acc']:.2f}%")
    print(f"  Test Instance mIoU: {eval_result['shape_avg_iou']:.4f}")
    print("=" * 50 + "\n")

    state_dict = load_state_dict_for_model(checkpoint_path, device)
    kan_weights_list = collect_kan_polynomial_weights(state_dict)
    print(f"共提取到 {len(kan_weights_list)} 个 KAT_Group 的因果多项式系数。")
    for w in kan_weights_list:
        print(f"  - {w['name']}: numerator shape {w['weight_numerator'].shape}, denominator shape {w['weight_denominator'].shape}")

    activation_ranges = {}
    need_calibration = (not args.no_vis or args.json_output) and len(kan_weights_list) > 0
    if need_calibration:
        cal_split = "trainval" if args.calibration_partition == "train" else "test"
        cal_dataset = PartNormalDataset(npoints=args.num_points, split=cal_split, normalize=False)
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=min(args.batch_size, 16),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        activation_ranges = collect_activation_ranges_part_seg(
            net, cal_loader, device, max_samples=args.calibration_samples, num_classes=NUM_CLASSES
        )
        print(f"校准集 ({cal_split}, n={args.calibration_samples}) 统计得到各层输入范围:")
        for name, r in sorted(activation_ranges.items()):
            print(f"  - {name}: x in [{r['min']:.4f}, {r['max']:.4f}]")

    similarity_per_layer = None
    if len(kan_weights_list) > 0 and (args.json_output or not args.no_vis):
        similarity_per_layer = {}
        for item in kan_weights_list:
            try:
                similarity_per_layer[item["name"]] = compute_layer_similarity_and_compression(
                    item, activation_ranges, args
                )
            except Exception as e:
                similarity_per_layer[item["name"]] = {"error": str(e)}

    out_dir = args.out_dir or os.path.join(os.path.dirname(checkpoint_path), "kan_analysis")
    if not args.no_vis and len(kan_weights_list) > 0:
        plot_kan_fx_curves(kan_weights_list, activation_ranges, out_dir, args)
        if similarity_per_layer:
            plot_grf_kan_similarity(similarity_per_layer, out_dir)
    elif args.no_vis:
        print("已指定 --no_vis，跳过可视化。")

    if len(kan_weights_list) > 0 and (args.json_output or not args.no_vis) and similarity_per_layer is not None:
        build_and_save_kan_analysis_json(
            kan_weights_list, activation_ranges, args, out_dir, similarity_per_layer=similarity_per_layer
        )


if __name__ == "__main__":
    main()
