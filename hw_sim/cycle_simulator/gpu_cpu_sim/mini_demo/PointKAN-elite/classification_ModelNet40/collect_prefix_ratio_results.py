#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总多组 prefix_ratio 实验的 results.json，输出汇总 JSON 与 Test MSE vs prefix_ratio 的 PNG 图。
用法:
  python collect_prefix_ratio_results.py --base_dir ./runs --pattern "fps_radii_mlp_K512_pr*"
  python collect_prefix_ratio_results.py --base_dir ./runs --pattern "fps_radii_mlp_K512_pr*" --out_json ./runs/summary.json --out_png ./runs/summary.png
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser(description="汇总 prefix_ratio 实验的 test loss，输出 JSON 与 PNG")
    p.add_argument("--base_dir", type=str, default="./runs", help="实验根目录")
    p.add_argument("--pattern", type=str, default="fps_radii_mlp_K*_pr*", help="匹配子目录的 glob 模式（相对 base_dir）")
    p.add_argument("--out_json", type=str, default="", help="汇总 JSON 路径，默认 base_dir/prefix_ratio_summary.json")
    p.add_argument("--out_png", type=str, default="", help="汇总 PNG 路径，默认 base_dir/prefix_ratio_summary.png")
    args = p.parse_args()

    search = os.path.join(args.base_dir, args.pattern)
    dirs = sorted(glob.glob(search))
    dirs = [d for d in dirs if os.path.isdir(d)]

    rows = []
    for d in dirs:
        results_path = os.path.join(d, "results.json")
        if not os.path.isfile(results_path):
            continue
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append({
            "save_dir": d,
            "prefix_ratio": data["prefix_ratio"],
            "fps_steps": data.get("fps_steps"),
            "best_test_mse": data["best_test_mse"],
        })

    if not rows:
        print(f"未在 {search} 下找到任何 results.json")
        return

    rows.sort(key=lambda x: x["prefix_ratio"])
    summary = {
        "base_dir": args.base_dir,
        "pattern": args.pattern,
        "runs": rows,
    }

    out_json = args.out_json or os.path.join(args.base_dir, "prefix_ratio_summary.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总 JSON: {out_json}")

    out_png = args.out_png or os.path.join(args.base_dir, "prefix_ratio_summary.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("未安装 matplotlib，跳过 PNG 输出。")
        return

    ratios = [r["prefix_ratio"] for r in rows]
    mses = [r["best_test_mse"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ratios, mses, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("prefix_ratio")
    ax.set_ylabel("Test MSE")
    ax.set_title("Test MSE vs prefix_ratio")
    ax.grid(True, alpha=0.3)
    for i, (pr, mse) in enumerate(zip(ratios, mses)):
        ax.annotate(f"{mse:.2e}", (pr, mse), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"汇总 PNG: {out_png}")


if __name__ == "__main__":
    main()
