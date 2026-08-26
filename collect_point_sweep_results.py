#!/usr/bin/env python3
import csv
import json
import os
import sys
from pathlib import Path


def load_report(report_path: Path) -> dict:
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    out_root_name = sys.argv[1] if len(sys.argv) > 1 else "analysis_real_eval_point_sweep"
    out_root = repo_root / out_root_name
    all_runs = []
    operator_rows = []

    report_paths = []
    for root, _, files in os.walk(out_root, followlinks=False):
        if "report.json" in files:
            report_paths.append(Path(root) / "report.json")

    for report_path in sorted(report_paths):
        parts = report_path.relative_to(out_root).parts
        if len(parts) < 3:
            continue

        network = parts[0]
        run_tag = parts[1]
        try:
            accelerator, ptag = run_tag.rsplit("_p", 1)
            input_points = int(ptag)
        except Exception:
            accelerator = run_tag
            input_points = -1

        report = load_report(report_path)
        end = report.get("end_to_end", {}) or {}
        neurosim_summary = report.get("neurosim_summary", {}) or {}
        total_cycles = end.get("total_cycles", 0) or 0
        total_energy_nj = end.get("total_energy_nj", 0.0) or 0.0
        latency_ns = end.get("latency_ns", 0.0) or 0.0
        cycle_ns = (latency_ns / total_cycles) if total_cycles else 0.0
        ops = {}

        breakdown_path = report_path.with_name("breakdown_detail.json")
        if breakdown_path.exists():
            breakdown = json.loads(breakdown_path.read_text())
            ops = breakdown.get("by_operator", {}) or {}
            total_area_um2 = float(end.get("total_area_um2", 0.0) or 0.0)
            for op_name, item in sorted(ops.items()):
                op_latency_ns = float(item.get("latency_ns", 0.0) or 0.0)
                if op_latency_ns <= 0.0:
                    op_cycles = float(item.get("cycles", 0.0) or 0.0)
                    op_latency_ns = op_cycles * cycle_ns
                op_energy_nj = float(item.get("energy_nj", 0.0) or 0.0)
                op_area_um2 = float(item.get("area_um2", 0.0) or 0.0)
                operator_rows.append({
                    "network": network,
                    "accelerator": accelerator,
                    "input_points": input_points,
                    "operator": op_name,
                    "latency_ns": op_latency_ns,
                    "latency_ms": op_latency_ns / 1e6,
                    "latency_pct": (op_latency_ns / latency_ns * 100.0) if latency_ns else 0.0,
                    "energy_nj": op_energy_nj,
                    "energy_mj": op_energy_nj / 1e6,
                    "energy_pct": (op_energy_nj / total_energy_nj * 100.0) if total_energy_nj else 0.0,
                    "area_um2": op_area_um2,
                    "area_pct": (op_area_um2 / total_area_um2 * 100.0) if total_area_um2 else 0.0,
                    "event_count": int(item.get("event_count", 0) or 0),
                    "report_path": str(report_path),
                    "e2e_latency_ms": float(end.get("latency_ms", 0.0) or 0.0),
                    "e2e_energy_mj": float(total_energy_nj) / 1e6,
                    "total_area_um2": total_area_um2,
                    "neurosim_layer_sum_latency_ms": float(neurosim_summary.get("layer_sum_latency_ms", 0.0) or 0.0),
                    "neurosim_chip_pipeline_latency_ms": float(neurosim_summary.get("chip_pipeline_latency_ms", 0.0) or 0.0),
                    "neurosim_non_cim_latency_ms": float(neurosim_summary.get("non_cim_latency_ms", 0.0) or 0.0),
                    "neurosim_chip_pipeline_plus_non_cim_ms": float(neurosim_summary.get("chip_pipeline_plus_non_cim_ms", 0.0) or 0.0),
                })

        record = {
            "network": network,
            "accelerator": accelerator,
            "input_points": input_points,
            "report_path": str(report_path),
            "total_cycles": total_cycles,
            "total_energy_nj": total_energy_nj,
            "latency_ns": latency_ns,
            "latency_ms": end.get("latency_ms", 0.0),
            "total_area_um2": end.get("total_area_um2", 0.0),
            "neurosim_layer_sum_latency_ns": neurosim_summary.get("layer_sum_latency_ns", 0.0),
            "neurosim_chip_pipeline_latency_ns": neurosim_summary.get("chip_pipeline_latency_ns", 0.0),
            "neurosim_non_cim_latency_ns": neurosim_summary.get("non_cim_latency_ns", 0.0),
            "neurosim_chip_pipeline_plus_non_cim_ns": neurosim_summary.get("chip_pipeline_plus_non_cim_ns", 0.0),
        }
        for key in ("DRAM", "FPS", "KNN", "Conv1d", "KAN", "SPool", "AffineNorm", "SRAM", "InterpN3"):
            item = ops.get(key, {}) or {}
            record[f"{key}__energy_nj"] = item.get("energy_nj", 0.0)
            op_cycles = item.get("cycles", 0.0) or 0.0
            record[f"{key}__latency_ns"] = op_cycles * cycle_ns
            record[f"{key}__energy_pct"] = (
                (item.get("energy_nj", 0.0) or 0.0) / total_energy_nj * 100.0 if total_energy_nj else 0.0
            )
            record[f"{key}__latency_pct"] = (op_cycles / total_cycles * 100.0) if total_cycles else 0.0
        all_runs.append(record)

    all_runs.sort(key=lambda r: (r["network"], r["accelerator"], r["input_points"]))

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "all_runs.json").write_text(json.dumps(all_runs, indent=2, ensure_ascii=False))

    if all_runs:
        fieldnames = list(all_runs[0].keys())
        with (out_root / "all_runs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_runs)
    else:
        (out_root / "all_runs.csv").write_text("")

    operator_rows.sort(key=lambda r: (r["network"], r["accelerator"], r["input_points"], r["operator"]))
    by_network_acc = {}
    for row in operator_rows:
        key = (row["network"], row["accelerator"])
        by_network_acc.setdefault(key, []).append(row)
    for (network, accelerator), rows in by_network_acc.items():
        network_dir = out_root / network
        network_dir.mkdir(parents=True, exist_ok=True)
        out_csv = network_dir / f"operator_breakdown_all_points_all_{accelerator}.csv"
        fieldnames = [
            "network",
            "accelerator",
            "input_points",
            "operator",
            "latency_ns",
            "latency_ms",
            "latency_pct",
            "energy_nj",
            "energy_mj",
            "energy_pct",
            "area_um2",
            "area_pct",
            "event_count",
            "report_path",
            "e2e_latency_ms",
            "e2e_energy_mj",
            "total_area_um2",
            "neurosim_layer_sum_latency_ms",
            "neurosim_chip_pipeline_latency_ms",
            "neurosim_non_cim_latency_ms",
            "neurosim_chip_pipeline_plus_non_cim_ms",
        ]
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_csv}")

    print(f"wrote {out_root / 'all_runs.json'}")
    print(f"wrote {out_root / 'all_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
