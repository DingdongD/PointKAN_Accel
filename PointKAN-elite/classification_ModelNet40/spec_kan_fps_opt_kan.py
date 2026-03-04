"""
spec_kan_fps_opt_kan.py：增强版 seg_filter FPS IOU 评估，支持「KAN 预测 radii 曲线」作为 segment 阈值

与 spec_kan_fps_opt.py 的对比：
  - 原版：radii_ref 来自同批点云的 exact FPS 结果（真实半径曲线），用于 seg_filter 的 tau。
  - 本增强版：在相同流程基础上，额外用「已训练的 KAN」对 exact 得到的 radii 做 prefix(10%)->KAN->suffix(90%)
    重建完整曲线，用该曲线作为 radii_ref 再跑 seg_filter，得到 idx_seg_kan；
    同时保留「用 exact radii 作 radii_ref」的 seg_filter 得到 idx_seg_exact_ref。
  - 评估：对比 IOU(vanilla, seg_exact_ref) 与 IOU(vanilla, seg_kan_ref)，看 KAN 预测的 radii 是否造成 FPS IOU 损失。

依赖：run_fps_radii_prefix_sweep.sh / run_eval_prefix_ratio.sh 训练得到的 KAN 权重（如 runs/fps_radii_kan_K512_pr0.10/best.pt）。
"""
import os
import sys
import json
import argparse
import numpy as np
from typing import Tuple, Optional, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 原版 FPS、数据集、IOU
from spec_kan_fps_opt import (
    fps_adaptive_argmax,
    compute_iou_batch,
    ModelNet40Wrapper,
    collate_points,
)

# KAN/MLP 模型与 radii 前后处理（与 spec_kan_fps / eval_fps_radii 一致）
from spec_kan_fps import (
    Config,
    build_model,
    build_prefix_suffix_io,
    lin_resample_1d,
)


# =========================================================
# 1. KAN 加载与「由 exact 平方半径 → 完整曲线（平方）」预测
# =========================================================

def load_kan_radii_model(kan_save_dir: str, device: torch.device) -> Tuple[torch.nn.Module, Config]:
    """
    从 run_fps_radii_prefix_sweep / run_eval_prefix_ratio 产出的目录加载 KAN（或 MLP）权重。
    目录内需有 best.pt（含 model, cfg）。
    返回 (model, cfg)，model 已 eval 且加载到 device。
    """
    ckpt_path = os.path.join(kan_save_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到 KAN 权重: {ckpt_path}，请先运行 run_fps_radii_prefix_sweep.sh 等完成训练。")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config()
    for k, v in ckpt.get("cfg", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict_full_radii_squared_from_exact(
    radii_sq: torch.Tensor,
    model: torch.nn.Module,
    cfg: Config,
    device: torch.device,
) -> torch.Tensor:
    """
    由 exact FPS 得到的平方半径曲线 radii_sq [B, K]，用 KAN 预测完整曲线（仍为平方半径），
    供 seg_filter 的 radii_ref 使用。

    流程（与 spec_kan_fps / eval_fps_radii 一致）：
      - 训练时 radii 为欧氏半径；本脚本中 exact 得到的是平方半径，故先 sqrt 得到欧氏 prefix。
      - prefix_eucl = sqrt(radii_sq[:, :prefix_len])，resample 到 in_dim -> x，model(x) -> 64 维，
      - 64 维上采样到 suffix_len 得到 suffix_pred_eucl，full_eucl = [prefix_eucl, suffix_pred_eucl]，
      - 返回 full_eucl^2 作为 radii_ref（seg_filter 内部用平方距离）。
    """
    B, K = radii_sq.shape
    prefix_ratio = getattr(cfg, "prefix_ratio", 0.10)
    in_dim = getattr(cfg, "in_dim", 32)
    out_dim = getattr(cfg, "out_dim", 64)

    prefix_len = max(2, min(int(round(K * prefix_ratio)), K - 2))
    suffix_len = K - prefix_len

    radii_sq = radii_sq.to(device, dtype=torch.float32)
    prefix_sq = radii_sq[:, :prefix_len].clamp(min=0.0)
    prefix_eucl = torch.sqrt(prefix_sq)

    x = lin_resample_1d(prefix_eucl, in_dim)
    yhat = model(x)
    suffix_pred_eucl = lin_resample_1d(yhat, suffix_len)
    full_eucl = torch.cat([prefix_eucl, suffix_pred_eucl], dim=1)
    full_sq = full_eucl ** 2
    return full_sq


# =========================================================
# 2. 过滤/候选统计汇总（与 spec_kan_fps_opt 一致）
# =========================================================

def _aggregate_filter_stats(
    all_candidate_ratio: List[torch.Tensor],
    all_used_fallback: List[torch.Tensor],
    num_points: int,
    K: int,
    segment_len: int,
) -> Dict:
    """
    由各 batch 的 candidate_ratio (B,K)、used_fallback (B,K) 汇总为与 spec_kan_fps_opt 相同的
    每步过滤、扫描节省、per_segment 等统计。
    """
    candidate_ratio_cat = torch.cat([t.flatten() for t in all_candidate_ratio])
    used_fallback_cat = torch.cat([t.flatten() for t in all_used_fallback])
    cr_stack = torch.cat(all_candidate_ratio, dim=0)
    per_step_candidate_ratio_mean = cr_stack.mean(dim=0).cpu().numpy()
    per_step_candidate_ratio_std = cr_stack.std(dim=0).cpu().numpy()

    overall_candidate_ratio_mean = candidate_ratio_cat.mean().item()
    overall_candidate_ratio_std = candidate_ratio_cat.std().item()
    search_space_reduction_ratio = 1.0 - overall_candidate_ratio_mean
    total_scan_vanilla = K * num_points - K * (K + 1) // 2
    remaining_per_step = np.maximum(0, num_points - 1 - np.arange(K))
    total_scan_seg_filter = float(np.sum(per_step_candidate_ratio_mean * remaining_per_step))
    scan_saved_ratio = 1.0 - (total_scan_seg_filter / total_scan_vanilla) if total_scan_vanilla > 0 else 0.0

    num_segments = (K + segment_len - 1) // segment_len
    per_segment_candidate_ratio_mean = []
    per_segment_filtered_out_mean = []
    for seg_j in range(num_segments):
        start_t = seg_j * segment_len
        end_t = min((seg_j + 1) * segment_len, K)
        seg_slice = per_step_candidate_ratio_mean[start_t:end_t]
        seg_mean = float(np.mean(seg_slice))
        per_segment_candidate_ratio_mean.append(seg_mean)
        per_segment_filtered_out_mean.append(float(num_points * (1.0 - seg_mean)))

    return {
        "candidate_ratio_mean": overall_candidate_ratio_mean,
        "candidate_ratio_std": overall_candidate_ratio_std,
        "used_fallback_any_ratio": used_fallback_cat.float().mean().item(),
        "num_points": num_points,
        "per_step_candidate_ratio_mean": per_step_candidate_ratio_mean.tolist(),
        "per_step_candidate_ratio_std": per_step_candidate_ratio_std.tolist(),
        "per_step_candidates_mean": (num_points * per_step_candidate_ratio_mean).tolist(),
        "per_step_filtered_out_mean": (num_points * (1.0 - per_step_candidate_ratio_mean)).tolist(),
        "search_space_reduction_ratio": search_space_reduction_ratio,
        "scan_saved_ratio": scan_saved_ratio,
        "total_scan_vanilla": total_scan_vanilla,
        "total_scan_seg_filter": total_scan_seg_filter,
        "per_segment_candidate_ratio_mean": per_segment_candidate_ratio_mean,
        "per_segment_filtered_out_mean": per_segment_filtered_out_mean,
    }


# =========================================================
# 3. 增强评估：exact_ref 与 kan_ref 双路 seg_filter，对比 IOU + 过滤统计
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="seg_filter FPS IOU 评估（增强版：对比 exact radii_ref 与 KAN 预测 radii_ref）"
    )
    p.add_argument("--root", type=str, default="./data", help="ModelNet40 数据根目录")
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--segment_len", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=str, default="first")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./runs/fps_seg_filter_iou_kan",
                   help="结果保存目录")
    p.add_argument("--kan_save_dir", type=str, default="",
                   help="KAN 权重目录，如 ./runs/fps_radii_kan_K512_pr0.10（必填才跑 KAN 分支）")
    return p.parse_args()


@torch.no_grad()
def run_evaluation_enhanced(
    loader: DataLoader,
    K: int,
    segment_len: int,
    seed: str,
    device: torch.device,
    num_points: int,
    kan_save_dir: Optional[str] = None,
) -> Dict:
    """
    对每个 batch：
      1) exact FPS -> idx_exact, radii_sq
      2) seg_filter(radii_ref=radii_sq) -> idx_seg_exact_ref，计算 IOU 并收集 candidate_ratio/used_fallback
      3) 若 kan_save_dir 非空：KAN 预测 full_radii_sq，seg_filter(radii_ref=full_radii_sq) -> idx_seg_kan，计算 IOU 并收集统计

    返回：IOU 汇总 + filter_stats_exact_ref（与 spec_kan_fps_opt 同结构的过滤/候选/节省统计）+ 若有 KAN 则 filter_stats_kan_ref。
    """
    all_iou_exact_ref: List[float] = []
    all_iou_kan_ref: List[float] = []
    all_candidate_ratio_exact: List[torch.Tensor] = []
    all_used_fallback_exact: List[torch.Tensor] = []
    all_candidate_ratio_kan: List[torch.Tensor] = []
    all_used_fallback_kan: List[torch.Tensor] = []

    model_kan = None
    cfg_kan = None
    if kan_save_dir and os.path.isdir(kan_save_dir):
        model_kan, cfg_kan = load_kan_radii_model(kan_save_dir, device)
        assert cfg_kan.fps_steps == K, f"KAN 训练时 fps_steps={cfg_kan.fps_steps}，当前 K={K}，需一致"

    for pts, _ in tqdm(loader, desc="IOU 评估 (exact_ref + kan_ref)", ncols=100):
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)

        idx_exact, radii_sq, _, _ = fps_adaptive_argmax(pts, K, seed=seed, mode="exact")

        # 原版：用 exact 的 radii 作 radii_ref，并收集过滤统计
        idx_seg_exact_ref, _, _, stats_exact = fps_adaptive_argmax(
            pts, K, seed=seed, mode="seg_filter",
            radii_ref=radii_sq, segment_len=segment_len,
        )
        iou_per, _, _ = compute_iou_batch(idx_exact, idx_seg_exact_ref)
        all_iou_exact_ref.extend(iou_per.cpu().numpy().tolist())
        all_candidate_ratio_exact.append(stats_exact["candidate_ratio"])
        all_used_fallback_exact.append(stats_exact["used_fallback"])

        if model_kan is not None and cfg_kan is not None:
            radii_ref_kan = predict_full_radii_squared_from_exact(radii_sq, model_kan, cfg_kan, device)
            idx_seg_kan, _, _, stats_kan = fps_adaptive_argmax(
                pts, K, seed=seed, mode="seg_filter",
                radii_ref=radii_ref_kan, segment_len=segment_len,
            )
            iou_kan, _, _ = compute_iou_batch(idx_exact, idx_seg_kan)
            all_iou_kan_ref.extend(iou_kan.cpu().numpy().tolist())
            all_candidate_ratio_kan.append(stats_kan["candidate_ratio"])
            all_used_fallback_kan.append(stats_kan["used_fallback"])

    mean_exact = float(np.mean(all_iou_exact_ref))
    std_exact = float(np.std(all_iou_exact_ref))
    n = len(all_iou_exact_ref)

    filter_exact = _aggregate_filter_stats(
        all_candidate_ratio_exact, all_used_fallback_exact,
        num_points, K, segment_len,
    )
    filter_kan = None
    if all_candidate_ratio_kan:
        filter_kan = _aggregate_filter_stats(
            all_candidate_ratio_kan, all_used_fallback_kan,
            num_points, K, segment_len,
        )

    out = {
        "num_samples": n,
        "mean_iou_exact_ref": mean_exact,
        "std_iou_exact_ref": std_exact,
        "filter_stats_exact_ref": filter_exact,
    }

    if all_iou_kan_ref:
        mean_kan = float(np.mean(all_iou_kan_ref))
        std_kan = float(np.std(all_iou_kan_ref))
        out["mean_iou_kan_ref"] = mean_kan
        out["std_iou_kan_ref"] = std_kan
        out["iou_drop_kan_vs_exact_ref"] = mean_exact - mean_kan
        out["kan_save_dir"] = kan_save_dir
        out["filter_stats_kan_ref"] = filter_kan
    else:
        out["mean_iou_kan_ref"] = None
        out["std_iou_kan_ref"] = None
        out["iou_drop_kan_vs_exact_ref"] = None
        out["kan_save_dir"] = None
        out["filter_stats_kan_ref"] = None

    return out


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed or "first"

    ds = ModelNet40Wrapper(partition=args.partition, num_points=args.num_points, root=args.root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )

    summary = run_evaluation_enhanced(
        loader,
        K=args.K,
        segment_len=args.segment_len,
        seed=seed,
        device=device,
        num_points=args.num_points,
        kan_save_dir=args.kan_save_dir.strip() or None,
    )

    results = {
        "partition": args.partition,
        "num_points": args.num_points,
        "K": args.K,
        "segment_len": args.segment_len,
        "seed": seed,
        "num_samples": summary["num_samples"],
        "mean_iou_exact_ref": summary["mean_iou_exact_ref"],
        "std_iou_exact_ref": summary["std_iou_exact_ref"],
        "filter_stats_exact_ref": summary["filter_stats_exact_ref"],
        "mean_iou_kan_ref": summary.get("mean_iou_kan_ref"),
        "std_iou_kan_ref": summary.get("std_iou_kan_ref"),
        "iou_drop_kan_vs_exact_ref": summary.get("iou_drop_kan_vs_exact_ref"),
        "kan_save_dir": summary.get("kan_save_dir"),
        "filter_stats_kan_ref": summary.get("filter_stats_kan_ref"),
    }
    results_path = os.path.join(args.save_dir, "iou_results_kan_compare.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    def _print_filter_block(label: str, fs: Dict, num_pts: int, seg_len: int, K_val: int) -> None:
        if label:
            print(f"  {label}")
        print(f"    每步平均候选比例 candidate_ratio: {fs['candidate_ratio_mean']:.4f} ± {fs['candidate_ratio_std']:.4f}")
        print(f"    每步平均保留候选点数: {num_pts * fs['candidate_ratio_mean']:.1f} / {num_pts}")
        print(f"    每步平均过滤掉点数: {num_pts * (1 - fs['candidate_ratio_mean']):.1f} / {num_pts}")
        print(f"    搜索空间节省比例: {fs['search_space_reduction_ratio']:.4f} ({fs['search_space_reduction_ratio']*100:.2f}%)")
        print(f"    总扫描量 vanilla（已选点不重复扫描）: {fs['total_scan_vanilla']}  seg_filter 等效: {fs['total_scan_seg_filter']:.0f}  节省: {fs['scan_saved_ratio']*100:.2f}%")
        print(f"    used_fallback 比例: {fs['used_fallback_any_ratio']:.4f}")
        print("    按 segment 汇总 (每段内步取平均):")
        for j in range(len(fs["per_segment_candidate_ratio_mean"])):
            cr = fs["per_segment_candidate_ratio_mean"][j]
            fo = fs["per_segment_filtered_out_mean"][j]
            start_t = j * seg_len
            end_t = min((j + 1) * seg_len, K_val)
            print(f"      segment {j:2d} (步 {start_t:3d}~{end_t-1:3d}): 候选比例={cr:.4f}  平均过滤点数={fo:.1f}")

    print("=" * 60)
    print("seg_filter FPS IOU 评估（增强版：exact_ref vs KAN_predicted_ref）")
    print("=" * 60)
    print(f"  分区: {args.partition}  N={args.num_points}  K={args.K}  segment_len={args.segment_len}  seed={seed}")
    print(f"  样本数: {summary['num_samples']}")
    print("-" * 60)
    print("  [1] 参考标准 radii_ref = exact FPS 半径曲线（原版 seg_filter）:")
    print(f"      mean_iou = {summary['mean_iou_exact_ref']:.6f}  std_iou = {summary['std_iou_exact_ref']:.6f}")
    print("  segment_filter 每次迭代过滤与相对 vanilla 的节省 (exact_ref):")
    _print_filter_block("", summary["filter_stats_exact_ref"], args.num_points, args.segment_len, args.K)

    if summary.get("mean_iou_kan_ref") is not None:
        print("-" * 60)
        print("  [2] KAN 预测 radii_ref（prefix 10% -> KAN -> suffix 90% 作为 seg_filter 阈值）:")
        print(f"      mean_iou = {summary['mean_iou_kan_ref']:.6f}  std_iou = {summary['std_iou_kan_ref']:.6f}")
        drop = summary["iou_drop_kan_vs_exact_ref"]
        print(f"      IOU 变化（KAN_ref 相对 exact_ref）: {drop:+.6f}  {'(有损失)' if drop > 0 else '(无损失/更优)'}")
        print(f"      KAN 权重目录: {summary.get('kan_save_dir')}")
        print("  segment_filter 每次迭代过滤与相对 vanilla 的节省 (kan_ref):")
        _print_filter_block("", summary["filter_stats_kan_ref"], args.num_points, args.segment_len, args.K)
    else:
        print("-" * 60)
        print("  未传入 --kan_save_dir，未运行 KAN 分支，无 KAN 过滤统计。")
    print("=" * 60)
    print(f"  结果已保存: {results_path}")


if __name__ == "__main__":
    main()
