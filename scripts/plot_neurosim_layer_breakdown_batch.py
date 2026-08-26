#!/usr/bin/env python3
"""
Batch plot NeuroSim per-layer breakdowns for multiple sim_results directories.

This script discovers existing neurosim_layer_breakdown.csv/json files and
reuses plot_neurosim_layer_breakdown.py to generate per-run tables and PNGs.
It also writes a manifest CSV summarizing processed runs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Dict, List


_SINGLE_RUN_SCRIPT = Path(__file__).with_name("plot_neurosim_layer_breakdown.py")
_SPEC = importlib.util.spec_from_file_location("plot_neurosim_layer_breakdown", _SINGLE_RUN_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to import single-run plotting helper from {_SINGLE_RUN_SCRIPT}")
_PLOT_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLOT_MODULE)


def discover_breakdown_inputs(root: Path) -> List[Path]:
    root = Path(root)
    candidates: List[Path] = []
    for pattern in ("**/neurosim_layer_breakdown.csv", "**/neurosim_layer_breakdown.json"):
        candidates.extend(root.glob(pattern))

    grouped: Dict[Path, Path] = {}
    for candidate in sorted(candidates):
        parent = candidate.parent
        existing = grouped.get(parent)
        if existing is None or (existing.suffix.lower() == ".json" and candidate.suffix.lower() == ".csv"):
            grouped[parent] = candidate
    return sorted(grouped.values())


def _sanitize_run_name(name: str) -> str:
    return str(name).replace(" ", "_")


def _write_manifest(rows: List[Dict[str, str]], out_dir: Path) -> Path:
    manifest_path = out_dir / "neurosim_layer_breakdown_batch_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_name", "input_path", "output_dir", "table_count", "plot_count"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def run_batch(root: Path, out_dir: Path) -> List[Dict[str, str]]:
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = discover_breakdown_inputs(root)
    if not inputs:
        raise FileNotFoundError(f"No neurosim_layer_breakdown.csv/json found under {root}")

    manifest_rows: List[Dict[str, str]] = []
    for input_path in inputs:
        run_name = _sanitize_run_name(input_path.parent.name)
        run_out_dir = out_dir / run_name
        outputs = _PLOT_MODULE.run(input_path, run_out_dir)
        manifest_rows.append(
            {
                "run_name": run_name,
                "input_path": str(input_path),
                "output_dir": str(run_out_dir),
                "table_count": str(len(outputs.get("tables", []))),
                "plot_count": str(len(outputs.get("plots", []))),
            }
        )

    _write_manifest(manifest_rows, out_dir)
    return manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch plot NeuroSim layer breakdowns for multiple result directories.")
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan recursively for neurosim_layer_breakdown.csv/json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="neurosim_batch_plots",
        help="Directory to store per-run plots and the batch manifest.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    results = run_batch(root, out_dir)
    print(f"Discovered {len(results)} run(s) under: {root}")
    print(f"Wrote batch manifest: {out_dir / 'neurosim_layer_breakdown_batch_manifest.csv'}")
    for row in results:
        print(f"[{row['run_name']}] input={row['input_path']} output={row['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
