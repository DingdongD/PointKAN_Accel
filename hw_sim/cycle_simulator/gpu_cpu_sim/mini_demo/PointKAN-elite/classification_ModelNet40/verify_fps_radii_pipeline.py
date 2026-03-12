#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FPS radii 算法流程验证：prefix 下采样 32 -> 模型 -> 64 上采样回 suffix 长度。

用于硬件对照实现分析，使用 ModelNet40 生成 radii 参考曲线，逐步输出各阶段 shape 与参考数据。

算法流程（单条曲线，长度 K=512）:
  1. 参考曲线: radii [K]  (来自 ModelNet40 + FPS)
  2. 划分: prefix_len = round(K * prefix_ratio), suffix_len = K - prefix_len
  3. prefix [prefix_len] --线性下采样--> input_32 [32]   (模型输入)
  4. 模型输出: output_64 [64]
  5. output_64 [64] --线性上采样--> suffix_recon [suffix_len]
  6. 完整重建: full_recon = concat(prefix, suffix_recon) [K]

运行:
  python verify_fps_radii_pipeline.py --radii_dir ./runs/radii_cache_K512 --out_dir ./runs/fps_radii_pipeline_ref
  python verify_fps_radii_pipeline.py --radii_dir ./runs/radii_cache_K512 --out_dir ./runs/fps_radii_pipeline_ref --checkpoint ./runs/fps_radii_kan_K512_pr0.10/best.pt
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from spec_kan_fps import (
    lin_resample_1d,
    build_prefix_suffix_io,
    _radii_paths,
    build_model,
    Config,
)


def algorithm_steps(
    radii: np.ndarray,
    K: int,
    prefix_ratio: float,
    in_dim: int = 32,
    out_dim: int = 64,
    device: torch.device = torch.device("cpu"),
    model=None,
):
    """
    单条 radii 的完整数据流，返回各阶段数组与说明（用于硬件对照）。

    radii: [K] 参考曲线
    返回: dict 含各阶段 key -> (array, shape_str, description)
    """
    assert radii.ndim == 1 and len(radii) == K
    prefix_len = max(2, min(int(round(K * prefix_ratio)), K - 2))
    suffix_len = K - prefix_len

    r = torch.from_numpy(radii.astype(np.float32)).unsqueeze(0).to(device)  # [1, K]

    # Step 1: 划分 prefix / suffix
    prefix = r[:, :prefix_len]   # [1, prefix_len]
    suffix_gt = r[:, prefix_len:] # [1, suffix_len]

    # Step 2: prefix 下采样 -> 32 (模型输入)
    input_32 = lin_resample_1d(prefix, in_dim)  # [1, 32]

    # Step 3: 模型输出 64（若无模型则用 suffix_gt 下采样到 64 作为参考）
    if model is not None:
        model.eval()
        with torch.no_grad():
            output_64 = model(input_32)  # [1, 64]
    else:
        output_64 = lin_resample_1d(suffix_gt, out_dim)  # [1, 64] 用 GT 下采样模拟

    # Step 4: 64 上采样回 suffix_len
    suffix_recon = lin_resample_1d(output_64, suffix_len)  # [1, suffix_len]
    print(suffix_recon)

    # Step 5: 完整重建 [prefix; suffix_recon]
    full_recon = torch.cat([prefix, suffix_recon], dim=1)  # [1, K]

    def to_np(t):
        return t.squeeze(0).cpu().numpy()

    return {
        "radii_ref": (to_np(r), f"[{K}]", "Reference radii curve (ModelNet40 + FPS)"),
        "prefix": (to_np(prefix), f"[{prefix_len}]", "Prefix segment (input to downsampler)"),
        "suffix_gt": (to_np(suffix_gt), f"[{suffix_len}]", "Ground truth suffix segment"),
        "input_32": (to_np(input_32), f"[{in_dim}]", "Model input: prefix downsampled to 32"),
        "output_64": (to_np(output_64), f"[{out_dim}]", "Model output: 64 points"),
        "suffix_recon": (to_np(suffix_recon), f"[{suffix_len}]", "Suffix reconstructed: 64 upsampled to suffix_len"),
        "full_recon": (to_np(full_recon), f"[{K}]", "Full curve: concat(prefix, suffix_recon)"),
    }, prefix_len, suffix_len


def main():
    parser = argparse.ArgumentParser(description="Verify FPS radii pipeline for hardware reference")
    parser.add_argument("--radii_dir", type=str, default="./runs/radii_cache_K512",
                        help="Directory containing radii_test_K*.npy (or radii_train_*)")
    parser.add_argument("--out_dir", type=str, default="./runs/fps_radii_pipeline_ref",
                        help="Output directory for reference arrays and algorithm description")
    parser.add_argument("--fps_steps", type=int, default=512, help="Radii length K")
    parser.add_argument("--prefix_ratio", type=float, default=0.10)
    parser.add_argument("--in_dim", type=int, default=32)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--sample_idx", type=int, default=0, help="Which test sample to use as reference")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Optional: best.pt to use real model output for output_64")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load radii reference from cache
    _, test_path = _radii_paths(args.radii_dir, args.fps_steps)
    if not os.path.isfile(test_path):
        train_path, _ = _radii_paths(args.radii_dir, args.fps_steps)
        if os.path.isfile(train_path):
            test_path = train_path
    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"No radii file found in {args.radii_dir}. "
            "Run spec_kan_fps.py or run_fps_radii_prefix_sweep.sh first to generate radii_cache_K*."
        )

    radii_all = np.load(test_path)  # [N, K]
    K = radii_all.shape[1]
    if args.fps_steps != K:
        print(f"[Warn] fps_steps={args.fps_steps} but radii have K={K}, using K={K}")
    idx = min(args.sample_idx, radii_all.shape[0] - 1)
    radii = radii_all[idx]

    model = None
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = Config()
        for k, v in ckpt.get("cfg", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        model = build_model(cfg, device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"[Model] Loaded from {args.checkpoint} for output_64")
    else:
        print("[Ref] No checkpoint: output_64 will be GT suffix downsampled to 64 (reference only)")

    steps, prefix_len, suffix_len = algorithm_steps(
        radii, K, args.prefix_ratio,
        in_dim=args.in_dim, out_dim=args.out_dim,
        device=device, model=model,
    )

    # Algorithm description (for hardware doc)
    desc = {
        "K": K,
        "prefix_ratio": args.prefix_ratio,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "in_dim": args.in_dim,
        "out_dim": args.out_dim,
        "steps": [
            "1. radii_ref [K]  = reference curve from ModelNet40 + FPS",
            f"2. prefix [prefix_len={prefix_len}] = radii_ref[0:prefix_len]",
            f"3. input_32 [{args.in_dim}] = linear_downsample(prefix, {args.in_dim})  # model input",
            f"4. output_64 [{args.out_dim}] = model(input_32)",
            f"5. suffix_recon [suffix_len={suffix_len}] = linear_upsample(output_64, suffix_len)",
            f"6. full_recon [K] = concat(prefix, suffix_recon)",
        ],
        "linear_resample": "src_indices in [0, L-1], out_len = target length; y = lerp(x[left], x[right], w)",
    }
    with open(os.path.join(args.out_dir, "algorithm_flow.json"), "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=2)

    # Save reference arrays
    for name, (arr, shape_str, detail) in steps.items():
        path = os.path.join(args.out_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  {name} {shape_str}  -> {path}  # {detail}")

    # Text summary for hardware
    summary_path = os.path.join(args.out_dir, "pipeline_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FPS Radii Pipeline (hardware reference)\n")
        f.write("========================================\n")
        f.write(f"K={K}  prefix_ratio={args.prefix_ratio}  prefix_len={prefix_len}  suffix_len={suffix_len}\n")
        f.write(f"in_dim={args.in_dim}  out_dim={args.out_dim}\n\n")
        for step in desc["steps"]:
            f.write(step + "\n")
        f.write("\nSaved arrays: radii_ref, prefix, suffix_gt, input_32, output_64, suffix_recon, full_recon\n")
    print(f"\nSummary: {summary_path}")
    print(f"Reference arrays and algorithm_flow.json saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
