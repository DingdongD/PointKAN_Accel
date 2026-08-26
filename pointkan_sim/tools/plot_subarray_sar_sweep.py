#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 verify_neurosim_adc_power.py 做 --subarray 扫参，绘制 SAR 分项的 latency / 功耗 / 能量。

说明：verify 脚本会按 Chip FloorPlan 自动抬高 in_c/out_c，subarray 越大通常网络也越大，
     曲线反映的是「该扫参流程下」的综合结果，不是「同一网络只改子阵列行数」的单变量实验。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_RE_E = re.compile(r"E_sar_per_tile = ([\d.]+) pJ")
_RE_T = re.compile(r"t_ADC_layer = ([\d.]+) ns")
_RE_P = re.compile(r"P_sar_per_tile = ([\d.]+|nan) mW", re.I)


def _run_verify(repo: Path, subarray: int, num_col_muxed: int) -> str:
    cmd = [
        sys.executable,
        str(repo / "pointkan_sim/tools/verify_neurosim_adc_power.py"),
        "--num_col_muxed",
        str(num_col_muxed),
        "--subarray",
        str(subarray),
    ]
    p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"subarray={subarray} failed: {p.stderr[:800]}")
    return p.stdout


def _parse(stdout: str) -> tuple[float, float, float]:
    m_e = _RE_E.search(stdout)
    m_t = _RE_T.search(stdout)
    m_p = _RE_P.search(stdout)
    if not (m_e and m_t and m_p):
        raise ValueError("parse failed:\n" + stdout[-2000:])
    e = float(m_e.group(1))
    t = float(m_t.group(1))
    ps = m_p.group(1).lower()
    pw = float("nan") if ps == "nan" else float(ps)
    return e, t, pw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512],
        help="numRowSubArray 扫参列表",
    )
    ap.add_argument("--num_col_muxed", type=int, default=8)
    ap.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="输出 PNG 路径；默认写到本脚本同目录 subarray_sar_sweep.png",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out_path = Path(args.output) if args.output else Path(__file__).resolve().parent / "subarray_sar_sweep.png"

    xs: list[int] = []
    Es: list[float] = []
    Ts: list[float] = []
    Ps: list[float] = []

    for s in args.sizes:
        stdout = _run_verify(repo, s, args.num_col_muxed)
        e, t, p = _parse(stdout)
        xs.append(s)
        Es.append(e)
        Ts.append(t)
        Ps.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    fig.suptitle(
        "SAR ADC vs numRowSubArray (verify script auto-scales in_c/out_c for FloorPlan)",
        fontsize=11,
    )

    axes[0].plot(xs, Ts, "o-", color="#1f77b4")
    axes[0].set_xlabel("numRowSubArray (--subarray)")
    axes[0].set_ylabel("t_ADC_layer (ns)")
    axes[0].set_title("ADC latency (layer)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, Es, "o-", color="#ff7f0e")
    axes[1].set_xlabel("numRowSubArray (--subarray)")
    axes[1].set_ylabel("E_sar_per_tile (pJ)")
    axes[1].set_title("E_sar per tile (equal split)")
    axes[1].grid(True, alpha=0.3)

    valid = [(x, p) for x, p in zip(xs, Ps) if p == p]
    if valid:
        vx, vp = zip(*valid)
        axes[2].plot(vx, vp, "o-", color="#2ca02c")
    axes[2].set_xlabel("numRowSubArray (--subarray)")
    axes[2].set_ylabel("P_sar_per_tile (mW)")
    axes[2].set_title("P_sar ~ E/t (pJ/ns)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
