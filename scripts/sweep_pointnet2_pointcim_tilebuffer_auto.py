#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path("/workspace/PointKAN-pytorch")


def _parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        v = int(chunk)
        if v <= 0:
            raise ValueError(f"invalid size: {v}")
        vals.append(v)
    if not vals:
        raise ValueError("empty size list")
    return vals


def _read_breakdown_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: Dict[str, float] = {}
            for k, v in row.items():
                if k is None:
                    continue
                if v is None or v == "":
                    out[k] = 0.0
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = 0.0
            rows.append(out)
    return rows


def _sum_col(rows: Iterable[Dict[str, float]], key: str) -> float:
    return sum(float(r.get(key, 0.0) or 0.0) for r in rows)


def _max_col(rows: Iterable[Dict[str, float]], key: str) -> float:
    vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
    return max(vals) if vals else 0.0


def _build_env(args: argparse.Namespace, tbuf_row: int, tbuf_col: int, results_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ACCELERATOR": "pointcim",
            "NETWORK": args.network,
            "NUM_POINTS": str(args.num_points),
            "NUM_CLASSES": str(args.num_classes),
            "USE_REAL_DATA": "1",
            "DEVICE": args.device,
            "BATCH_SIZE": "1",
            "MAX_BATCHES": str(args.max_batches),
            "NEUROSIM_FULL": "1",
            "NEUROSIM_PYBIND": "0",
            "NEUROSIM_ROOT": "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
            "NEUROSIM_CLK_MODE": args.neurosim_clk_mode,
            "NEUROSIM_CLK_FREQ": str(int(args.neurosim_clk_freq)),
            "NEUROSIM_ADC_CLK_FREQ": str(int(args.neurosim_adc_clk_freq)),
            "NEUROSIM_TECHNODE": str(args.neurosim_technode),
            "NEUROSIM_WL_WEIGHT": str(args.wl_weight),
            "NEUROSIM_WL_ACTIVATION": str(args.wl_activation),
            "NEUROSIM_NUM_COL_MUXED": str(args.num_col_muxed or args.wl_weight),
            "NEUROSIM_SUBARRAY_ROWS": str(args.subarray_row),
            "NEUROSIM_SUBARRAY_COLS": str(args.subarray_col),
            "NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW": str(tbuf_row),
            "NEUROSIM_TILE_BUFFER_CORE_SIZE_COL": str(tbuf_col),
            "LOGIC_CLK_FREQ": str(int(args.logic_clk_freq)),
            "PROGRESS_INTERVAL_S": str(args.progress_interval_s),
            "KNN_CONV_PIPELINE_MODE": "strict",
            "BN_EN": "1",
            "QD_EN": "1",
            "EXPORT_MEM_TXN": "1",
            "SIM_RESULTS_DIR": str(results_dir),
            "SIM_LOG_DIR": str(ROOT / "sim_log"),
            "OUTPUT_JSON": "report.json",
        }
    )
    return env


def _run_one(args: argparse.Namespace, tbuf_row: int, tbuf_col: int, results_dir: Path, log_path: Path) -> None:
    env = _build_env(args, tbuf_row, tbuf_col, results_dir)
    cmd = ["bash", str(ROOT / "run_sim_with_real_eval.sh"), "--pointcim_sa_conv_pipeline"]
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"run failed for tileBuffer={tbuf_row}x{tbuf_col}, see log: {log_path}")


def _parse_clkperiod_ns(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"clkPeriod:\\s*([0-9.eE+-]+)", text)
    if not matches:
        return 0.0
    return float(matches[-1]) * 1e9


def _collect_one(args: argparse.Namespace, tbuf_row: int, tbuf_col: int, results_dir: Path, log_path: Path) -> Dict[str, float]:
    report_path = results_dir / "report.json"
    breakdown_path = results_dir / "neurosim_layer_breakdown.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"missing report: {report_path}")
    if not breakdown_path.exists():
        raise FileNotFoundError(f"missing breakdown csv: {breakdown_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    breakdown_rows = _read_breakdown_csv(breakdown_path)
    e2e = report.get("end_to_end") or {}
    ns_sum = report.get("neurosim_summary") or {}
    tile_stats = report.get("rram_tile_stats") or {}
    tile_size = tile_stats.get("tile_size") or {}
    pe_size = tile_stats.get("pe_size") or {}
    xbar_size = tile_stats.get("xbar_size") or {}

    sum_tile_buf_lat = _sum_col(breakdown_rows, "other_sub_tile_buffer_htree_latency_ns")
    max_tile_buf_lat = _max_col(breakdown_rows, "other_sub_tile_buffer_htree_latency_ns")
    sum_other_lat = _sum_col(breakdown_rows, "core_latency_other_ns")
    sum_buf_lat = _sum_col(breakdown_rows, "buffer_latency_ns")
    sum_ic_lat = _sum_col(breakdown_rows, "ic_latency_ns")
    sum_layer_lat = _sum_col(breakdown_rows, "latency_ns")
    sum_adc_lat = _sum_col(breakdown_rows, "core_latency_adc_ns")
    sum_accum_lat = _sum_col(breakdown_rows, "core_latency_accum_ns")
    clk_ns = _parse_clkperiod_ns(log_path)

    return {
        "subarray_row": float(args.subarray_row),
        "subarray_col": float(args.subarray_col),
        "tile_buffer_core_size_row": float(tbuf_row),
        "tile_buffer_core_size_col": float(tbuf_col),
        "e2e_latency_ns": float(e2e.get("latency_ns", 0.0) or 0.0),
        "e2e_latency_ms": float(e2e.get("latency_ms", 0.0) or 0.0),
        "e2e_energy_nj": float(e2e.get("total_energy_nj", 0.0) or 0.0),
        "e2e_energy_uj": float(e2e.get("total_energy_nj", 0.0) or 0.0) / 1000.0,
        "layer_sum_latency_ns": float(ns_sum.get("layer_sum_latency_ns", 0.0) or 0.0),
        "sum_adc_latency_ns": sum_adc_lat,
        "sum_accum_latency_ns": sum_accum_lat,
        "sum_other_latency_ns": sum_other_lat,
        "sum_buffer_latency_ns": sum_buf_lat,
        "sum_ic_latency_ns": sum_ic_lat,
        "sum_tile_buffer_htree_latency_ns": sum_tile_buf_lat,
        "max_tile_buffer_htree_latency_ns": max_tile_buf_lat,
        "tile_buffer_share_of_layer_sum_pct": (sum_tile_buf_lat / sum_layer_lat * 100.0) if sum_layer_lat > 0 else 0.0,
        "tile_buffer_share_of_other_pct": (sum_tile_buf_lat / sum_other_lat * 100.0) if sum_other_lat > 0 else 0.0,
        "total_tile_count": float(tile_stats.get("total_tile_count", 0.0) or 0.0),
        "total_pe_count": float(tile_stats.get("total_PE_count", 0.0) or 0.0),
        "xbar_row": float(xbar_size.get("row", 0.0) or 0.0),
        "xbar_col": float(xbar_size.get("column", 0.0) or 0.0),
        "tile_row": float(tile_size.get("row", 0.0) or 0.0),
        "tile_col": float(tile_size.get("column", 0.0) or 0.0),
        "pe_row": float(pe_size.get("row", 0.0) or 0.0),
        "pe_col": float(pe_size.get("column", 0.0) or 0.0),
        "pe_per_tile": float(tile_stats.get("pe_per_tile", 0.0) or 0.0),
        "clk_period_ns": clk_ns,
        "clk_freq_mhz": (1000.0 / clk_ns) if clk_ns > 0 else 0.0,
        "results_dir": str(results_dir),
        "report_path": str(report_path),
        "breakdown_path": str(breakdown_path),
        "log_path": str(log_path),
    }


def _write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict[str, float]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    by_lat = sorted(rows, key=lambda r: float(r["e2e_latency_ns"]))
    by_tilebuf = sorted(rows, key=lambda r: float(r["sum_tile_buffer_htree_latency_ns"]))
    best_lat = by_lat[0]
    best_tilebuf = by_tilebuf[0]
    lines = [
        "# PointNet++ PointCIM TileBuffer Sweep",
        "",
        f"- network: `{args.network}`",
        f"- points: `{args.num_points}`",
        f"- fixed subarray: `{args.subarray_row}x{args.subarray_col}`",
        f"- tileBuffer row sweep: `{','.join(str(v) for v in args.tilebuffer_rows)}`",
        f"- tileBuffer col sweep: `{','.join(str(v) for v in args.tilebuffer_cols)}`",
        "",
        "## Key Findings",
        "",
        f"- Lowest end-to-end latency: `tileBuffer={int(best_lat['tile_buffer_core_size_row'])}x{int(best_lat['tile_buffer_core_size_col'])}` with `{best_lat['e2e_latency_ms']:.6f} ms`",
        f"- Lowest tileBufferHTree latency: `tileBuffer={int(best_tilebuf['tile_buffer_core_size_row'])}x{int(best_tilebuf['tile_buffer_core_size_col'])}` with `{best_tilebuf['sum_tile_buffer_htree_latency_ns']/1e6:.6f} ms`",
        "",
        "## Summary Table",
        "",
        "| tileBuffer core | e2e_ms | other_ms | tileBuffer_ms | tileBuffer/other | buffer_ms | ic_ms | clkPeriod_ns | tile | pe | total_tiles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_lat:
        lines.append(
            "| {buf} | {e2e:.6f} | {other:.6f} | {tb:.6f} | {share:.2f}% | {buflat:.6f} | {ic:.6f} | {clk:.6f} | {tile} | {pe} | {tiles:.0f} |".format(
                buf=f"{int(row['tile_buffer_core_size_row'])}x{int(row['tile_buffer_core_size_col'])}",
                e2e=row["e2e_latency_ms"],
                other=float(row["sum_other_latency_ns"]) / 1e6,
                tb=float(row["sum_tile_buffer_htree_latency_ns"]) / 1e6,
                share=row["tile_buffer_share_of_other_pct"],
                buflat=float(row["sum_buffer_latency_ns"]) / 1e6,
                ic=float(row["sum_ic_latency_ns"]) / 1e6,
                clk=row["clk_period_ns"],
                tile=f"{int(row['tile_row'])}x{int(row['tile_col'])}",
                pe=f"{int(row['pe_row'])}x{int(row['pe_col'])}",
                tiles=row["total_tile_count"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep NeuroSim tileBufferCoreSizeRow/Col for pointnet++ pointcim.")
    p.add_argument("--network", default="pointnet2-partseg-ssg")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--num-classes", type=int, default=50)
    p.add_argument("--subarray-row", type=int, default=128)
    p.add_argument("--subarray-col", type=int, default=128)
    p.add_argument("--tilebuffer-rows", type=_parse_int_list, default=[32, 64, 128], help="tileBufferCoreSizeRow sweep list")
    p.add_argument("--tilebuffer-cols", type=_parse_int_list, default=[32, 64, 128], help="tileBufferCoreSizeCol sweep list")
    p.add_argument("--wl-weight", type=int, default=8)
    p.add_argument("--wl-activation", type=int, default=8)
    p.add_argument("--num-col-muxed", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--neurosim-clk-mode", default="capped", choices=["capped", "critical"])
    p.add_argument("--neurosim-clk-freq", type=float, default=100_000_000)
    p.add_argument("--neurosim-adc-clk-freq", type=float, default=1_000_000_000)
    p.add_argument("--neurosim-technode", type=int, default=32)
    p.add_argument("--logic-clk-freq", type=float, default=1_000_000_000)
    p.add_argument("--progress-interval-s", type=float, default=30.0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    p.add_argument("--out-dir", default=str(ROOT / "analysis_tilebuffer_pointnet2_pointcim_auto"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    logs_dir = out_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    combos: List[Tuple[int, int]] = [(r, c) for r in args.tilebuffer_rows for c in args.tilebuffer_cols]
    rows: List[Dict[str, float]] = []

    for tbuf_row, tbuf_col in combos:
        tag = f"pointnet2-partseg-ssg_pointcim_p{args.num_points}_w{args.wl_weight}a{args.wl_activation}_suba{args.subarray_row}x{args.subarray_col}_tbuf{tbuf_row}x{tbuf_col}_auto"
        results_dir = runs_dir / tag
        log_path = logs_dir / f"{tag}.log"
        if not args.summarize_only:
            if results_dir.exists() and not args.force:
                print(f"[skip] tileBuffer={tbuf_row}x{tbuf_col} already exists")
            else:
                results_dir.mkdir(parents=True, exist_ok=True)
                print(f"[run] tileBuffer={tbuf_row}x{tbuf_col} -> {results_dir}")
                _run_one(args, tbuf_row, tbuf_col, results_dir, log_path)
        if results_dir.exists():
            rows.append(_collect_one(args, tbuf_row, tbuf_col, results_dir, log_path))

    csv_path = out_dir / "tilebuffer_sweep_summary.csv"
    md_path = out_dir / "tilebuffer_sweep_summary.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, args)
    print(f"Wrote summary CSV: {csv_path}")
    print(f"Wrote summary MD : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
