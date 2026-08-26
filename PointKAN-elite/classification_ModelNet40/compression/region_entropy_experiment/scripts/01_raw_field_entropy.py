#!/usr/bin/env python3
"""阶段 1：absolute xyz 的 sign / exp / mantissa 字段分离熵与有效位宽。"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.utils.entropy_stats import marginal_coord_entropy_12bit  # noqa: E402
from scripts.utils.ieee32_stats import ieee32_stream_field_stats  # noqa: E402
from scripts.utils.io import load_test_npz  # noqa: E402


def _axis_field_stats(vals: np.ndarray) -> dict:
    """vals: (N,) float32 单轴坐标。"""
    st = ieee32_stream_field_stats(vals)
    st = dict(st)
    st["H_raw_coord_12bit"] = marginal_coord_entropy_12bit(vals)
    return st


def per_sample_row(sample_id_str: str, xyz: np.ndarray) -> dict:
    row: dict = {"sample_id": sample_id_str}
    for d, ax in enumerate(["x", "y", "z"]):
        st = _axis_field_stats(xyz[:, d])
        row[f"H{ax}_raw"] = st["H_raw_coord_12bit"]
        row[f"Hexp_{ax}_raw"] = st["Hexp"]
        row[f"Hsign_{ax}_raw"] = st["Hsign"]
        row[f"Hmant8_{ax}_raw"] = st["Hmant8"]
        row[f"Hsignexp_{ax}_raw"] = st["Hsignexp"]
        row[f"Hexpmant8_{ax}_raw"] = st["Hexpmant8"]
        row[f"top1_exp_cov_{ax}"] = st["top1_exp_cov"]
        row[f"top2_exp_cov_{ax}"] = st["top2_exp_cov"]
        row[f"top3_exp_cov_{ax}"] = st["top3_exp_cov"]
        row[f"avg_effective_bits_{ax}"] = st["avg_effective_bits"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=os.path.join(_ROOT, "data", "modelnet40_test.npz"),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_ROOT, "outputs", "phase_a_raw"),
    )
    ap.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    args = ap.parse_args()

    points, _labels, _ids, id_strings = load_test_npz(args.input)
    n = points.shape[0]
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for i in range(n):
        sid = id_strings[i] if i < len(id_strings) else str(int(_ids[i]))
        rows.append(per_sample_row(sid, points[i]))

    df = pd.DataFrame(rows)
    per_path = os.path.join(args.output_dir, "raw_field_entropy_per_sample.csv")
    df.to_csv(per_path, index=False)

    num_cols = [c for c in df.columns if c != "sample_id"]
    summ = df[num_cols].agg(["mean", "std"])
    summ_path = os.path.join(args.output_dir, "raw_field_entropy_summary.csv")
    summ.T.to_csv(summ_path)

    print("per_sample:", per_path)
    print("summary:", summ_path)
    print("mean Hexp_x_raw:", float(summ.loc["mean", "Hexp_x_raw"]))


if __name__ == "__main__":
    main()
