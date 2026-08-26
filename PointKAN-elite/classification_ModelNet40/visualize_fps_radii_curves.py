#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 FPS radii KAN/MLP：在指定 prefix_ratio 下对比「真实 radii 曲线」与「模型预测的完整曲线」。

支持：
  - 单数据集：与 eval_fps_radii.py 类似，额外支持 multidataset 的 radii 缓存命名
    radii_test_K{K}_{dataset}.npy
  - 多数据集对比：同一图中按行/列排列多个数据集、多个样本，便于横向比较

一键对比「全部已训练」数据集（推荐，与 run_fps_radii_one_click_all 目录布局一致）:
  python visualize_fps_radii_curves.py --compare \\
    --auto_from ./runs/fps_radii_oneclick_all --auto_k 512 --auto_model_type kan --auto_pr 0.10 \\
    --prefix_ratio 0.10 --out_dir ./runs/fps_radii_curve_vis_all

手动指定若干数据集:
  python visualize_fps_radii_curves.py --compare \\
    --entry "modelnet40|.../fps_radii_kan_K512_modelnet40_pr0.10|.../radii_cache_md_modelnet40_K512" \\
    --entry "s3dis|...|..."

单数据集:
  python visualize_fps_radii_curves.py \\
    --save_dir ./runs/fps_radii_kan_K512_s3dis_pr0.10 \\
    --radii_dir ./runs/fps_radii_oneclick_all/radii_cache_md_s3dis_K512 \\
    --dataset s3dis \\
    --out_dir ./vis_s3dis
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from eval_fps_radii import plot_sample, predict_full_curve  # noqa: E402
from spec_kan_fps import Config, _radii_paths, build_model  # noqa: E402


def resolve_test_radii_path(
    radii_dir: str,
    fps_steps: int,
    dataset_tag: Optional[str] = None,
) -> str:
    """multidataset: radii_test_K{K}_{tag}.npy；否则 legacy radii_test_K{K}.npy。"""
    radii_dir = (radii_dir or "").strip()
    if not radii_dir:
        raise ValueError("需要 radii_dir")
    if dataset_tag:
        tagged = os.path.join(radii_dir, f"radii_test_K{fps_steps}_{dataset_tag}.npy")
        if os.path.isfile(tagged):
            return tagged
    _, legacy = _radii_paths(radii_dir, fps_steps)
    if os.path.isfile(legacy):
        return legacy
    if dataset_tag:
        raise FileNotFoundError(
            f"未找到测试 radii: {tagged}（或 legacy {legacy}）"
        )
    raise FileNotFoundError(f"未找到测试 radii: {legacy}")


def load_checkpoint_and_cfg(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, Config, dict]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config()
    for k, v in ckpt.get("cfg", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def prefix_len_for_ratio(K: int, prefix_ratio: float) -> int:
    pl = max(2, int(round(K * prefix_ratio)))
    return min(pl, K - 2)


def _safe_pr_tag(x: float) -> str:
    """用于路径/文件名中的 pr 片段，避免把 .png 里的点替换坏。"""
    return f"{x:g}".replace(".", "p")


def discover_auto_entries(base_dir: str, model_type: str, k: int, target_pr: float) -> List[str]:
    """
    扫描 base_dir 下 fps_radii_{model}_K{k}_{dataset}_pr* ，目录名末尾 pr 与 target_pr 一致（浮点比较）则加入。
    与 run_fps_radii_prefix_sweep_multidataset / one_click 目录命名一致。
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"--auto_from 不是目录: {base_dir}")
    prefix = f"fps_radii_{model_type}_K{k}_"
    pattern = os.path.join(base_dir, f"{prefix}*_pr*")
    entries: List[str] = []
    seen = set()
    for sd in sorted(glob.glob(pattern)):
        if not os.path.isdir(sd):
            continue
        name = os.path.basename(sd)
        if not name.startswith(prefix):
            continue
        mid = name[len(prefix) :]
        idx = mid.rfind("_pr")
        if idx < 0:
            continue
        dataset = mid[:idx]
        pr_str = mid[idx + len("_pr") :]
        try:
            pr_v = float(pr_str)
        except ValueError:
            continue
        if abs(pr_v - float(target_pr)) > 1e-5:
            continue
        rd = os.path.join(base_dir, f"radii_cache_md_{dataset}_K{k}")
        rp = os.path.join(rd, f"radii_test_K{k}_{dataset}.npy")
        bp = os.path.join(sd, "best.pt")
        if not os.path.isfile(bp):
            print(f"[auto skip] {dataset}: 无 best.pt -> {sd}")
            continue
        if not os.path.isfile(rp):
            print(f"[auto skip] {dataset}: 无 {rp}")
            continue
        line = f"{dataset}|{sd}|{rd}"
        if line in seen:
            continue
        seen.add(line)
        entries.append(line)
        print(f"[auto ok] {dataset}")
    entries.sort(key=lambda s: s.split("|", 1)[0].lower())
    return entries


@torch.no_grad()
def visualize_single(
    save_dir: str,
    radii_dir: str,
    dataset_tag: Optional[str],
    checkpoint: str,
    prefix_ratio: Optional[float],
    num_vis: int,
    out_dir: str,
    device: torch.device,
    seed: int,
) -> None:
    ckpt_path = checkpoint or os.path.join(save_dir, "best.pt")
    model, cfg, _ckpt = load_checkpoint_and_cfg(ckpt_path, device)
    fps_steps = cfg.fps_steps
    pr = float(prefix_ratio if prefix_ratio is not None else cfg.prefix_ratio)
    if prefix_ratio is not None and abs(prefix_ratio - cfg.prefix_ratio) > 1e-6:
        print(
            f"[注意] 使用 prefix_ratio={pr} 可视化，与 checkpoint 中训练比例 {cfg.prefix_ratio} 不同，"
            "曲线仅作参考（分布与训练时不一致）。"
        )

    test_path = resolve_test_radii_path(radii_dir, fps_steps, dataset_tag)
    test_radii = np.load(test_path)
    if test_radii.ndim != 2 or test_radii.shape[1] != fps_steps:
        raise ValueError(
            f"radii 形状应为 [N,K]，K={fps_steps}，实际 {test_radii.shape} @ {test_path}"
        )

    rng = np.random.default_rng(seed)
    n = len(test_radii)
    num_vis = min(num_vis, n)
    indices = (
        rng.choice(n, size=num_vis, replace=False)
        if num_vis < n
        else np.arange(n, dtype=int)
    )

    K = test_radii.shape[1]
    plen = prefix_len_for_ratio(K, pr)
    tag = dataset_tag or "dataset"
    vis_dir = os.path.join(out_dir, f"vis_{tag}_pr{_safe_pr_tag(pr)}")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"  checkpoint: {ckpt_path}")
    print(f"  test_radii: {test_path}  shape={test_radii.shape}")
    print(f"  prefix_ratio={pr}  prefix_len={plen}  -> 输出目录 {vis_dir}")

    for i, idx in enumerate(indices):
        r = torch.from_numpy(test_radii[idx]).float()
        gt_curve = test_radii[idx]
        pred_curve = predict_full_curve(model, r, pr, cfg.in_dim, cfg.out_dim, device)
        out_path = os.path.join(vis_dir, f"sample_{i:02d}_idx{int(idx)}.png")
        plot_sample(
            gt_curve,
            pred_curve,
            out_path,
            title=f"{tag} sample={i} idx={int(idx)} prefix_len={plen}",
            prefix_len=plen,
        )
        print(f"    保存 {out_path}")


def parse_entry(s: str) -> Tuple[str, str, str]:
    parts = s.split("|")
    if len(parts) != 3:
        raise ValueError(
            f"--entry 格式应为 dataset|save_dir|radii_dir，得到: {s!r}"
        )
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def apply_radii_plot_style(ax, grid_minor: bool = True) -> None:
    """GT vs Pred 子图：简洁 spine + 主次网格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="-", linewidth=0.45, alpha=0.35)
    if grid_minor:
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.2)


@torch.no_grad()
def visualize_compare(
    entries: List[str],
    prefix_ratio: Optional[float],
    num_vis: int,
    out_dir: str,
    device: torch.device,
    seed: int,
    pred_label: str = "",
    skip_detail: bool = False,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("对比图需要 matplotlib") from e

    parsed: List[Tuple[str, str, str]] = [parse_entry(e) for e in entries]
    os.makedirs(out_dir, exist_ok=True)

    # 加载每个数据集的模型与测试 radii
    bundles: List[Dict] = []
    for dataset_tag, save_dir, radii_dir in parsed:
        ckpt_path = os.path.join(save_dir, "best.pt")
        model, cfg, _ = load_checkpoint_and_cfg(ckpt_path, device)
        pr = float(prefix_ratio if prefix_ratio is not None else cfg.prefix_ratio)
        if prefix_ratio is not None and abs(prefix_ratio - cfg.prefix_ratio) > 1e-6:
            print(
                f"[注意] {dataset_tag}: 可视化 prefix_ratio={pr}，训练时为 {cfg.prefix_ratio}"
            )
        test_path = resolve_test_radii_path(radii_dir, cfg.fps_steps, dataset_tag)
        test_radii = np.load(test_path)
        if test_radii.shape[1] != cfg.fps_steps:
            raise ValueError(f"{dataset_tag}: K 维 {test_radii.shape[1]} != cfg.fps_steps {cfg.fps_steps}")
        bundles.append(
            {
                "tag": dataset_tag,
                "model": model,
                "cfg": cfg,
                "pr": pr,
                "test_radii": test_radii,
            }
        )

    plab = (pred_label or "").strip()
    if not plab:
        mt0 = getattr(bundles[0]["cfg"], "model_type", "mlp")
        plab = "KAN" if mt0 == "kan" else "MLP" if mt0 == "mlp" else "Pred"

    K = bundles[0]["cfg"].fps_steps
    for b in bundles:
        if b["cfg"].fps_steps != K:
            raise ValueError("对比模式下所有条目的 fps_steps(K) 必须相同")

    rng = np.random.default_rng(seed)
    n_min = min(len(b["test_radii"]) for b in bundles)
    num_vis = min(num_vis, n_min)
    indices = (
        rng.choice(n_min, size=num_vis, replace=False).tolist()
        if num_vis < n_min
        else list(range(num_vis))
    )

    n_ds = len(bundles)
    fig, axes = plt.subplots(
        n_ds,
        num_vis,
        figsize=(3.2 * num_vis, 2.8 * n_ds),
        squeeze=False,
    )

    prs = [b["pr"] for b in bundles]
    pr_note = f"prefix_ratio={prs[0]:g}" if len(set(prs)) == 1 else f"prefix_ratio per-row {prs}"
    plen0 = prefix_len_for_ratio(K, prs[0])

    for row, b in enumerate(bundles):
        model = b["model"]
        cfg = b["cfg"]
        pr = b["pr"]
        test_radii = b["test_radii"]
        tag = b["tag"]
        plen_b = prefix_len_for_ratio(K, pr)
        for col, idx in enumerate(indices):
            ax = axes[row][col]
            gt_curve = test_radii[idx]
            r = torch.from_numpy(gt_curve).float()
            pred_curve = predict_full_curve(model, r, pr, cfg.in_dim, cfg.out_dim, device)
            x_axis = np.arange(K)
            ax.plot(x_axis, gt_curve, "b-", label="GT radii", linewidth=1.2, alpha=0.9)
            ax.plot(
                x_axis,
                pred_curve,
                "r--",
                label=plab,
                linewidth=1.2,
                alpha=0.9,
            )
            if plen_b > 0 and plen_b < K:
                ax.axvline(x=plen_b - 0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(f"{tag}\nidx={idx}", fontsize=8)
            apply_radii_plot_style(ax, grid_minor=True)
            if row == n_ds - 1:
                ax.set_xlabel("step")
            if col == 0:
                ax.set_ylabel("radius")
            if row == 0 and col == num_vis - 1:
                ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(
        f"GT vs {plab} radii curve ({pr_note}, prefix_len@row0={plen0}, K={K})",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = os.path.join(
        out_dir,
        f"compare_grid_pr{_safe_pr_tag(prs[0])}_K{K}_n{num_vis}.png",
    )
    fig.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"对比总图已保存: {grid_path}")

    if skip_detail:
        return

    # 各数据集详细双栏图（与 eval_fps_radii 一致）
    for bi, b in enumerate(bundles):
        save_d = next(s for d, s, _r in parsed if d == b["tag"])
        radii_d = next(r for d, _s, r in parsed if d == b["tag"])
        sub_out = os.path.join(out_dir, f"detail_{b['tag']}_pr{_safe_pr_tag(b['pr'])}")
        visualize_single(
            save_dir=save_d,
            radii_dir=radii_d,
            dataset_tag=b["tag"],
            checkpoint="",
            prefix_ratio=b["pr"],
            num_vis=min(num_vis, 6),
            out_dir=sub_out,
            device=device,
            seed=seed + bi * 10007,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可视化 FPS radii KAN/MLP：GT vs 预测曲线")
    p.add_argument("--compare", action="store_true", help="多数据集对比")
    p.add_argument(
        "--entry",
        action="append",
        default=[],
        help='对比项：dataset|save_dir|radii_dir（可重复；可与 --auto_from 叠加）',
    )
    p.add_argument(
        "--auto_from",
        type=str,
        default="",
        help="训练输出根目录：自动扫描其下全部 fps_radii_*_K{k}_*_pr* 且含 best.pt + radii_test 的条目",
    )
    p.add_argument("--auto_k", type=int, default=512, help="与训练 fps_steps / 目录中 K 一致")
    p.add_argument("--auto_model_type", type=str, default="kan", help="目录名中的模型类型：kan 或 mlp")
    p.add_argument(
        "--auto_pr",
        type=float,
        default=None,
        help="只纳入目录名中该 prefix_ratio 的 run（与 pr0.10 等一致）；使用 --auto_from 时必填",
    )
    p.add_argument("--save_dir", type=str, default="", help="单数据集：训练输出目录（含 best.pt）")
    p.add_argument("--radii_dir", type=str, default="", help="单数据集：radii 缓存目录")
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        help="单数据集：标签，用于解析 radii_test_K*_dataset.npy；留空则用 legacy 文件名",
    )
    p.add_argument("--checkpoint", type=str, default="", help="单数据集：权重路径，默认 save_dir/best.pt")
    p.add_argument(
        "--prefix_ratio",
        type=float,
        default=None,
        help="可视化用的前缀比例；默认与 checkpoint 内 cfg 一致",
    )
    p.add_argument("--num_vis", type=int, default=4, help="可视化样本数（对比模式为每行样本数）")
    p.add_argument("--out_dir", type=str, default="./runs/fps_radii_curve_vis", help="输出根目录")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_detail",
        action="store_true",
        help="对比模式只生成总览网格图，不写各数据集 detail 子目录",
    )
    p.add_argument(
        "--pred_label",
        type=str,
        default="",
        help="预测曲线图例文字，默认 KAN（model_type=kan）否则 Pred",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.compare:
        all_entries: List[str] = list(args.entry)
        auto_from = (args.auto_from or "").strip()
        if auto_from:
            if args.auto_pr is None:
                raise SystemExit("使用 --auto_from 时必须指定 --auto_pr（与训练目录 pr0.xx 一致）")
            all_entries.extend(
                discover_auto_entries(auto_from, args.auto_model_type, args.auto_k, args.auto_pr)
            )
        if not all_entries:
            raise SystemExit(
                "对比模式需要至少一个 --entry，或使用 --auto_from + --auto_pr 自动发现全部数据集"
            )
        _seen = set()
        _dedup: List[str] = []
        for e in all_entries:
            if e in _seen:
                continue
            _seen.add(e)
            _dedup.append(e)
        all_entries = _dedup
        visualize_compare(
            all_entries,
            args.prefix_ratio,
            args.num_vis,
            args.out_dir,
            device,
            args.seed,
            pred_label=args.pred_label,
            skip_detail=args.no_detail,
        )
        return

    if not args.save_dir:
        raise SystemExit("单数据集模式需要 --save_dir（或改用 --compare）")
    ds_tag = (args.dataset or "").strip() or None
    visualize_single(
        save_dir=args.save_dir,
        radii_dir=args.radii_dir or args.save_dir,
        dataset_tag=ds_tag,
        checkpoint=args.checkpoint,
        prefix_ratio=args.prefix_ratio,
        num_vis=args.num_vis,
        out_dir=args.out_dir,
        device=device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
