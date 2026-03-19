#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Conv1d/Linear 脉动阵列能耗：按层汇总 MAC 数，用公式 MACs × 7.144 × clock_ns / 1000 计算期望能耗，
与 scalesim_conv_linear_layers.csv 中的 mac_energy_nj 汇总对比；可选与 report.json 中 systolic 事件汇总对比。

用法（在 PointKAN-pytorch 根目录）:
  python scripts/verify_conv_linear_mac_energy.py sim_results_systolic_sweep/systolic_sram_1048576B
  python scripts/verify_conv_linear_mac_energy.py --csv path/to/scalesim_conv_linear_layers.csv
  python scripts/verify_conv_linear_mac_energy.py --report path/to/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 与 hw_config.SystolicArrayConfig 默认一致
DEFAULT_MAC_POWER_MW = 7.144
DEFAULT_CLOCK_NS = 1.0


def load_scalesim_csv(csv_path: Path) -> list[dict]:
    """读取 scalesim_conv_linear_layers.csv，返回行列表（每行 dict）。"""
    import csv
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def parse_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_int(v, default: int = 0) -> int:
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def verify_from_csv(
    csv_path: Path,
    mac_power_mw: float = DEFAULT_MAC_POWER_MW,
    clock_ns: float = DEFAULT_CLOCK_NS,
) -> dict:
    """
    从 scalesim_conv_linear_layers.csv 按 Conv1d/Linear 汇总 macs 与 mac_energy_nj，
    用公式 sum(macs)*mac_power_mw*clock_ns/1000 得到期望 MAC 能耗，与 sum(mac_energy_nj) 对比。
    返回 { "Conv1d": {...}, "Linear": {...}, "summary": [...] }。
    """
    rows = load_scalesim_csv(csv_path)
    by_op = {"Conv1d": {"macs": 0, "mac_energy_nj": 0.0, "layers": 0}, "Linear": {"macs": 0, "mac_energy_nj": 0.0, "layers": 0}}

    for row in rows:
        op = (row.get("operator") or "").strip()
        if op not in by_op:
            continue
        macs = parse_int(row.get("macs"), 0)
        mac_nj = parse_float(row.get("mac_energy_nj"), 0.0)
        by_op[op]["macs"] += macs
        by_op[op]["mac_energy_nj"] += mac_nj
        by_op[op]["layers"] += 1

    out = {"Conv1d": {}, "Linear": {}, "summary": []}
    for op in ("Conv1d", "Linear"):
        m = by_op[op]
        total_macs = m["macs"]
        reported_mac_nj = m["mac_energy_nj"]
        expected_mac_nj = total_macs * mac_power_mw * clock_ns / 1000.0
        diff = reported_mac_nj - expected_mac_nj
        diff_pct = (diff / expected_mac_nj * 100.0) if expected_mac_nj else 0.0
        out[op] = {
            "total_macs": total_macs,
            "reported_mac_energy_nj": reported_mac_nj,
            "expected_mac_energy_nj": expected_mac_nj,
            "diff_nj": diff,
            "diff_pct": diff_pct,
            "layers": m["layers"],
        }
        out["summary"].append(
            f"  {op}: MACs={total_macs:,}, reported_mac_energy_nj={reported_mac_nj:.4f}, "
            f"expected (MACs×{mac_power_mw}×{clock_ns}/1000)={expected_mac_nj:.4f}, diff={diff:.4f} nJ ({diff_pct:.4f}%)"
        )
    return out


def systolic_energy_from_report(report_path: Path) -> dict[str, float]:
    """从 report.json 的 operator_sequence 中汇总 circuit_module==systolic_array 的 energy_nj，按 operator 分。"""
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    seq = report.get("operator_sequence") or []
    by_op = {"Conv1d": 0.0, "Linear": 0.0}
    for item in seq:
        if (item.get("circuit_module") or "").strip().lower() != "systolic_array":
            continue
        op = (item.get("operator") or "").strip()
        if op not in by_op:
            continue
        by_op[op] += float(item.get("energy_nj") or 0.0)
    return by_op


def main():
    parser = argparse.ArgumentParser(
        description="Verify Conv1d/Linear MAC energy: sum(MACs)*7.144/1000 vs reported mac_energy_nj"
    )
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help="Sim results dir containing scalesim_conv_linear_layers.csv and optionally report.json",
    )
    parser.add_argument("--csv", type=str, default=None, help="Path to scalesim_conv_linear_layers.csv")
    parser.add_argument("--report", type=str, default=None, help="Path to report.json (optional, for cross-check)")
    parser.add_argument("--mac_power_mw", type=float, default=DEFAULT_MAC_POWER_MW, help="MAC power (mW), default 7.144")
    parser.add_argument("--clock_ns", type=float, default=DEFAULT_CLOCK_NS, help="Clock period (ns), default 1.0")
    args = parser.parse_args()

    csv_path = None
    report_path = None
    if args.csv:
        csv_path = Path(args.csv)
    elif args.result_dir:
        d = Path(args.result_dir)
        csv_path = d / "scalesim_conv_linear_layers.csv"
        report_path = d / "report.json"
    if args.report:
        report_path = Path(args.report)

    if not csv_path or not csv_path.is_file():
        print("Error: need scalesim_conv_linear_layers.csv (via result_dir or --csv).")
        return 1

    result = verify_from_csv(csv_path, mac_power_mw=args.mac_power_mw, clock_ns=args.clock_ns)
    print("=== Conv1d/Linear MAC energy verification ===")
    print("Formula: expected_mac_energy_nj = sum(MACs) * mac_power_mw * clock_ns / 1000")
    print(f"  mac_power_mw = {args.mac_power_mw}, clock_ns = {args.clock_ns}")
    print()
    for line in result["summary"]:
        print(line)
    print()

    # 简单一致性判断
    tol_pct = 0.01  # 0.01% 相对误差内认为一致
    all_ok = True
    for op in ("Conv1d", "Linear"):
        r = result[op]
        if abs(r["diff_pct"]) > tol_pct:
            all_ok = False
    if all_ok:
        print("Result: reported mac_energy_nj matches formula (within 0.01% relative error).")
    else:
        print("Result: reported vs expected differ beyond 0.01%; check formula or CSV source.")

    if report_path and report_path.is_file():
        by_op = systolic_energy_from_report(report_path)
        print()
        print("=== Cross-check with report.json (systolic_array events only) ===")
        for op in ("Conv1d", "Linear"):
            from_report = by_op.get(op, 0.0)
            from_csv = result[op]["reported_mac_energy_nj"]
            print(f"  {op}: report.json systolic sum = {from_report:.4f} nJ, CSV sum(mac_energy_nj) = {from_csv:.4f} nJ")
    return 0


if __name__ == "__main__":
    exit(main())
