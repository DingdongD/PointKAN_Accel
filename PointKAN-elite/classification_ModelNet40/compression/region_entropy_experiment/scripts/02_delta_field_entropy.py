#!/usr/bin/env python3
"""阶段 2：不同点序下相邻差分的 IEEE 字段熵与有效位（对齐清单）。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_COMPRESSION = os.path.normpath(os.path.join(_ROOT, ".."))
if _COMPRESSION not in sys.path:
    sys.path.insert(0, _COMPRESSION)

from region_entropy.spatial_order import apply_order, sort_indices_hilbert, sort_indices_morton  # noqa: E402

from scripts.utils.entropy_stats import marginal_delta_entropy_12bit  # noqa: E402
from scripts.utils.ieee32_stats import ieee32_stream_field_stats  # noqa: E402
from scripts.utils.io import load_test_npz  # noqa: E402

SCRATCH_ORDERS = frozenset({"shuffled_input", "morton_after_shuffle", "hilbert_after_shuffle"})

KNOWN_ORDERS = frozenset(
    {
        "unordered",
        "random_shuffle",
        "morton_sorted",
        "hilbert_sorted",
        "shuffled_input",
        "morton_after_shuffle",
        "hilbert_after_shuffle",
    }
)


def _hilbert_available() -> bool:
    return sort_indices_hilbert(np.zeros((8, 3), np.float32), p=3) is not None


def _filter_order_names(names: List[str]) -> List[str]:
    ok = _hilbert_available()
    out: List[str] = []
    for o in names:
        if o in ("hilbert_sorted", "hilbert_after_shuffle") and not ok:
            print(f"警告: 跳过 {o}（未安装 hilbertcurve）")
            continue
        out.append(o)
    return out


def _sequence_for_order(
    xyz: np.ndarray,
    order_type: str,
    rng: np.random.Generator,
    scratch_cache: Dict[str, Optional[np.ndarray]],
) -> Optional[np.ndarray]:
    """同一 scratch_cache 保证 shuffled_input / morton_after_shuffle / hilbert_after_shuffle 共用一次置换。"""
    n = xyz.shape[0]

    if order_type == "unordered":
        return xyz.copy()

    if order_type == "random_shuffle":
        return apply_order(xyz, rng.permutation(n))

    if order_type == "morton_sorted":
        return apply_order(xyz, sort_indices_morton(xyz, p=10))

    if order_type == "hilbert_sorted":
        h = sort_indices_hilbert(xyz, p=10)
        return apply_order(xyz, h) if h is not None else None

    if order_type in SCRATCH_ORDERS:
        if scratch_cache["xs"] is None:
            scratch_cache["xs"] = xyz[rng.permutation(n)].copy()
        xs = scratch_cache["xs"]
        if order_type == "shuffled_input":
            return xs.copy()
        if order_type == "morton_after_shuffle":
            return apply_order(xs, sort_indices_morton(xs, p=10))
        if order_type == "hilbert_after_shuffle":
            hh = sort_indices_hilbert(xs, p=10)
            return apply_order(xs, hh) if hh is not None else None

    return None


def _delta_row(sample_id: str, order_type: str, seq_xyz: np.ndarray) -> Optional[dict]:
    d = np.diff(seq_xyz.astype(np.float64), axis=0).astype(np.float32)
    if d.shape[0] == 0:
        return None
    norms = np.linalg.norm(d.astype(np.float64), axis=1)
    row: dict = {"sample_id": sample_id, "order_type": order_type}
    for k, an in enumerate(["dx", "dy", "dz"]):
        col = d[:, k]
        dr = float(np.percentile(np.abs(col), 99) + 1e-6)
        dr = max(dr, 1e-9)
        row[f"H{an}"] = float(marginal_delta_entropy_12bit(col, dr))
        st = ieee32_stream_field_stats(col)
        row[f"Hexp_{an}"] = float(st["Hexp"])
        row[f"Hsign_{an}"] = float(st["Hsign"])
        row[f"Hmant8_{an}"] = float(st["Hmant8"])
        row[f"Hsignexp_{an}"] = float(st["Hsignexp"])
        row[f"Hexpmant8_{an}"] = float(st["Hexpmant8"])
        row[f"avg_effective_bits_{an}"] = float(st["avg_effective_bits"])
    row["mean_neighbor_dist_in_sequence"] = float(np.mean(norms))
    row["median_neighbor_dist_in_sequence"] = float(np.median(norms))
    for j, t in enumerate([1e-2, 5e-2, 1e-1]):
        row[f"p_norm_lt_tau_r_{j}"] = float(np.mean(norms < t))
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
        default=os.path.join(_ROOT, "outputs", "phase_b_delta"),
    )
    ap.add_argument(
        "--orders",
        nargs="*",
        default=None,
        help="默认含 unordered/random_shuffle/morton_sorted/hilbert_sorted "
        "及 shuffled_input/morton_after_shuffle/hilbert_after_shuffle",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=0, help="0 表示全部")
    args = ap.parse_args()

    default_orders = [
        "unordered",
        "random_shuffle",
        "morton_sorted",
        "hilbert_sorted",
        "shuffled_input",
        "morton_after_shuffle",
        "hilbert_after_shuffle",
    ]
    raw_names = args.orders if args.orders is not None else default_orders
    unknown = [o for o in raw_names if o not in KNOWN_ORDERS]
    if unknown:
        raise SystemExit(f"未知 order_type: {unknown}；已知: {sorted(KNOWN_ORDERS)}")
    order_names = _filter_order_names(raw_names)

    rng = np.random.default_rng(args.seed)
    points, _labels, _ids, id_strings = load_test_npz(args.input)
    n = points.shape[0]
    if args.max_samples > 0:
        n = min(n, args.max_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    rows: List[dict] = []
    for i in range(n):
        sid = id_strings[i] if i < len(id_strings) else str(int(_ids[i]))
        xyz = points[i]
        scratch_cache: Dict[str, Optional[np.ndarray]] = {"xs": None}
        for ot in order_names:
            seq = _sequence_for_order(xyz, ot, rng, scratch_cache)
            if seq is None:
                continue
            r = _delta_row(sid, ot, seq)
            if r is not None:
                rows.append(r)

    df = pd.DataFrame(rows)
    per_path = os.path.join(args.output_dir, "delta_field_entropy_per_sample.csv")
    df.to_csv(per_path, index=False)

    num_cols = [c for c in df.columns if c not in ("sample_id", "order_type")]
    summ_rows = []
    for ot, g in df.groupby("order_type"):
        m = g[num_cols].mean(numeric_only=True)
        s = g[num_cols].std(numeric_only=True)
        one = {"order_type": ot}
        for c in num_cols:
            one[f"{c}_mean"] = float(m[c])
            one[f"{c}_std"] = float(s[c])
        summ_rows.append(one)
    summ_df = pd.DataFrame(summ_rows)
    summ_path = os.path.join(args.output_dir, "delta_field_entropy_summary.csv")
    summ_df.to_csv(summ_path, index=False)

    print("per_sample:", per_path)
    print("summary:", summ_path)
    print("order_types:", ", ".join(order_names))


if __name__ == "__main__":
    main()
