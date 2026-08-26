#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


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


def _build_env(args: argparse.Namespace, subarray_row: int, subarray_col: int, results_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    square_subarray = str(subarray_row) if subarray_row == subarray_col else ""
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
            "NEUROSIM_SUBARRAY": square_subarray,
            "NEUROSIM_SUBARRAY_ROWS": str(subarray_row),
            "NEUROSIM_SUBARRAY_COLS": str(subarray_col),
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
    if args.network == "pointnet2-sem-ssg":
        env["INPUT_CHANNELS"] = str(args.input_channels)
    return env


def _run_one(args: argparse.Namespace, subarray_row: int, subarray_col: int, results_dir: Path, log_path: Path) -> None:
    env = _build_env(args, subarray_row, subarray_col, results_dir)
    cmd = [
        "bash",
        str(ROOT / "run_sim_with_real_eval.sh"),
        "--pointcim_sa_conv_pipeline",
    ]
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
        raise RuntimeError(f"run failed for subarray={subarray_row}x{subarray_col}, see log: {log_path}")


def _collect_one(subarray_row: int, subarray_col: int, results_dir: Path) -> Dict[str, float]:
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

    return {
        "subarray_row": float(subarray_row),
        "subarray_col": float(subarray_col),
        "e2e_latency_ns": float(e2e.get("latency_ns", 0.0) or 0.0),
        "e2e_latency_ms": float(e2e.get("latency_ms", 0.0) or 0.0),
        "e2e_energy_nj": float(e2e.get("total_energy_nj", 0.0) or 0.0),
        "e2e_energy_uj": float(e2e.get("total_energy_nj", 0.0) or 0.0) / 1000.0,
        "layer_sum_latency_ns": float(ns_sum.get("layer_sum_latency_ns", 0.0) or 0.0),
        "chip_pipeline_latency_ns": float(ns_sum.get("chip_pipeline_latency_ns", 0.0) or 0.0),
        "non_cim_latency_ns": float(ns_sum.get("non_cim_latency_ns", 0.0) or 0.0),
        "chip_pipeline_plus_non_cim_ns": float(ns_sum.get("chip_pipeline_plus_non_cim_ns", 0.0) or 0.0),
        "total_tile_count": float(tile_stats.get("total_tile_count", 0.0) or 0.0),
        "total_pe_count": float(tile_stats.get("total_PE_count", 0.0) or 0.0),
        "xbar_row": float(xbar_size.get("row", 0.0) or 0.0),
        "xbar_col": float(xbar_size.get("column", 0.0) or 0.0),
        "tile_row": float(tile_size.get("row", 0.0) or 0.0),
        "tile_col": float(tile_size.get("column", 0.0) or 0.0),
        "pe_row": float(pe_size.get("row", 0.0) or 0.0),
        "pe_col": float(pe_size.get("column", 0.0) or 0.0),
        "pe_per_tile": float(tile_stats.get("pe_per_tile", 0.0) or 0.0),
        "sum_layer_latency_ns_from_csv": sum_layer_lat,
        "sum_adc_latency_ns": sum_adc_lat,
        "sum_accum_latency_ns": sum_accum_lat,
        "sum_other_latency_ns": sum_other_lat,
        "sum_buffer_latency_ns": sum_buf_lat,
        "sum_ic_latency_ns": sum_ic_lat,
        "sum_tile_buffer_htree_latency_ns": sum_tile_buf_lat,
        "max_tile_buffer_htree_latency_ns": max_tile_buf_lat,
        "tile_buffer_share_of_layer_sum_pct": (sum_tile_buf_lat / sum_layer_lat * 100.0) if sum_layer_lat > 0 else 0.0,
        "tile_buffer_share_of_other_pct": (sum_tile_buf_lat / sum_other_lat * 100.0) if sum_other_lat > 0 else 0.0,
        "results_dir": str(results_dir),
        "report_path": str(report_path),
        "breakdown_path": str(breakdown_path),
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
    rows_by_lat = sorted(rows, key=lambda r: float(r["e2e_latency_ns"]))
    rows_by_tilebuf = sorted(rows, key=lambda r: float(r["sum_tile_buffer_htree_latency_ns"]), reverse=True)
    best = rows_by_lat[0]
    worst_buf = rows_by_tilebuf[0]
    lines = [
        f"# PointNet++ PointCIM Subarray Auto Sweep",
        "",
        f"- network: `{args.network}`",
        f"- points: `{args.num_points}`",
        f"- accelerator: `pointcim`",
        f"- wl_weight / wl_activation: `{args.wl_weight}` / `{args.wl_activation}`",
        f"- subarrays: `{', '.join([f'{r}x{c}' for r, c in args.subarray_pairs])}`",
        "",
        "## Key Findings",
        "",
        f"- Lowest end-to-end latency: `subarray={int(best['subarray_row'])}x{int(best['subarray_col'])}` with `{best['e2e_latency_ms']:.6f} ms`",
        f"- Highest tileBufferHTree latency sum: `subarray={int(worst_buf['subarray_row'])}x{int(worst_buf['subarray_col'])}` with `{worst_buf['sum_tile_buffer_htree_latency_ns']:.3f} ns`",
        f"- Best config auto-selected tile=`{int(best['tile_row'])}x{int(best['tile_col'])}`, pe=`{int(best['pe_row'])}x{int(best['pe_col'])}`, total_tiles=`{best['total_tile_count']:.0f}`",
        "",
        "## Summary Table",
        "",
        "| subarray | tile | pe | total_tiles | e2e_ms | layer_sum_ms | tileBuffer_sum_ms | tileBuffer/other | tileBuffer/layer_sum |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_by_lat:
        lines.append(
            "| {sub} | {tile} | {pe} | {tiles:.0f} | {e2e:.6f} | {ls:.6f} | {tb:.6f} | {tb_other:.2f}% | {tb_layer:.2f}% |".format(
                sub=f"{int(row['subarray_row'])}x{int(row['subarray_col'])}",
                tile=f"{int(row['tile_row'])}x{int(row['tile_col'])}",
                pe=f"{int(row['pe_row'])}x{int(row['pe_col'])}",
                tiles=row["total_tile_count"],
                e2e=row["e2e_latency_ms"],
                ls=float(row["layer_sum_latency_ns"]) / 1e6,
                tb=float(row["sum_tile_buffer_htree_latency_ns"]) / 1e6,
                tb_other=row["tile_buffer_share_of_other_pct"],
                tb_layer=row["tile_buffer_share_of_layer_sum_pct"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep NeuroSim subarray for pointnet++ pointcim with auto PE/tile.")
    p.add_argument("--network", default="pointnet2-partseg-ssg")
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--num-classes", type=int, default=50)
    p.add_argument("--subarrays", type=_parse_int_list, default=None, help="兼容旧接口：正方形 subarray 列表，如 64,128,256,512")
    p.add_argument("--subarray-rows", type=_parse_int_list, default=None, help="numRowSubArray 扫描列表，如 64,128,256")
    p.add_argument("--subarray-cols", type=_parse_int_list, default=None, help="numColSubArray 扫描列表，如 64,128,256")
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
    p.add_argument("--input-channels", type=int, default=12)
    p.add_argument("--out-dir", type=Path, default=ROOT / "analysis_subarray_pointnet2_pointcim_auto")
    p.add_argument("--force", action="store_true", help="rerun even if report.json already exists")
    p.add_argument("--summarize-only", action="store_true", help="do not run, only summarize existing result directories")
    args = p.parse_args()
    if args.subarray_rows is not None or args.subarray_cols is not None:
        if args.subarray_rows is None or args.subarray_cols is None:
            p.error("--subarray-rows and --subarray-cols must be provided together")
        if len(args.subarray_rows) != len(args.subarray_cols):
            p.error("--subarray-rows and --subarray-cols must have the same length")
        args.subarray_pairs = list(zip(args.subarray_rows, args.subarray_cols))
    else:
        square_list = args.subarrays if args.subarrays is not None else [64, 128, 256, 512]
        args.subarray_pairs = [(v, v) for v in square_list]
    return args


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    runs_dir = out_dir / "runs"
    logs_dir = out_dir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for subarray_row, subarray_col in args.subarray_pairs:
        tag = f"{args.network}_pointcim_p{args.num_points}_w{args.wl_weight}a{args.wl_activation}_suba{subarray_row}x{subarray_col}_auto"
        results_dir = runs_dir / tag
        log_path = logs_dir / f"{tag}.log"
        report_path = results_dir / "report.json"
        if not args.summarize_only and (args.force or not report_path.exists()):
            print(f"[run] subarray={subarray_row}x{subarray_col} -> {results_dir}")
            _run_one(args, subarray_row, subarray_col, results_dir, log_path)
        elif report_path.exists():
            print(f"[skip] subarray={subarray_row}x{subarray_col} already exists")
        else:
            print(f"[missing] subarray={subarray_row}x{subarray_col} has no report and summarize-only enabled", file=sys.stderr)
            continue
        rows.append(_collect_one(subarray_row, subarray_col, results_dir))

    summary_csv = out_dir / "subarray_sweep_summary.csv"
    summary_md = out_dir / "subarray_sweep_summary.md"
    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, rows, args)
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary MD : {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
