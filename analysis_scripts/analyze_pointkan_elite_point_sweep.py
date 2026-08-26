#!/usr/bin/env python3
"""
Analyze PointKAN-Elite point-count scaling with simulator breakdowns.

Outputs:
1. Incremental JSON/CSV records for each run
2. Baseline stacked-bar breakdown plots across point counts
3. Baseline vs base_v2 FPS/KNN latency/energy comparison plots

Default network set (the three PointKAN-Elite variants):
- pointkan-elite
- pointkan-elite-scanobj
- pointkan-elite-partseg

Notes:
- This script defaults to Scheduler.run_simulation_fast() because full event
  timeline simulation becomes very slow at large point counts.
- The fast path still keeps event generation and exclusive-resource constraints,
  so it is suitable for breakdown trend analysis and sorter comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

import sys
import time

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pointkan_sim.main import run_network_simulation  # noqa: E402
from pointkan_sim.run_sim_with_real_eval import apply_network_defaults  # noqa: E402
from pointkan_sim.simulator.scheduler import Scheduler  # noqa: E402


DEFAULT_NETWORKS = [
    "pointkan-elite",
    "pointkan-elite-scanobj",
    "pointkan-elite-partseg",
]
DEFAULT_ACCELERATORS = ["baseline", "base_v2"]
DEFAULT_POINTS = [2048, 4096, 8192, 16384, 32768]
STACK_TOP_N = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep PointKAN-Elite variants over point counts and plot breakdown trends."
    )
    p.add_argument(
        "--networks",
        nargs="+",
        default=DEFAULT_NETWORKS,
        help="Networks to analyze. Default: the three PointKAN-Elite variants.",
    )
    p.add_argument(
        "--accelerators",
        nargs="+",
        default=DEFAULT_ACCELERATORS,
        choices=["baseline", "base_v2", "base_v3", "acc"],
        help="Accelerators to sweep. Default: baseline base_v2",
    )
    p.add_argument(
        "--points",
        nargs="+",
        type=int,
        default=DEFAULT_POINTS,
        help="Input point counts to sweep.",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "analysis_pointkan_elite_point_sweep"),
        help="Directory for JSON/CSV/plots.",
    )
    p.add_argument(
        "--use-fast-scheduler",
        action="store_true",
        default=True,
        help="Use Scheduler.run_simulation_fast() for tractable large-point sweeps. Default: on.",
    )
    p.add_argument(
        "--full-scheduler",
        action="store_true",
        help="Use full Scheduler.run_simulation() instead of fast mode.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing JSON records and rerun everything.",
    )
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only run sweeps and save JSON/CSV; do not generate plots.",
    )
    return p.parse_args()


def make_sim_args(network: str, accelerator: str, points: int) -> SimpleNamespace:
    args = SimpleNamespace(
        network=network,
        accelerator=accelerator,
        input_points=points,
        num_points=points,
        num_classes=40,
        input_channels=3,
        show_layers=False,
        neurosim_full=False,
        no_layer_buffer=False,
        show_shapes=False,
        show_events=False,
        enable_w8a8_quant_dequant=False,
        bn_en=1,
        cim_backend=None,
        neurosim_root=None,
        use_scalesim_systolic=True,
        scalesim_root=None,
        scalesim_config=None,
        knn_conv_pipeline_mode="disabled",
        fc_num_slices=1,
        fc_cores_per_slice=16,
        fc_muls_per_core=1024,
        fc_coalescing_factor=0.0,
        output_timeline_csv=None,
        output_timeline_png=None,
        timeline_stride=500,
        timeline_max_cycle=2000,
        logic_clk_freq=1_000_000_000,
        ns_simulation_mode="event",
        logic_par_layernorm=None,
        logic_par_relu=None,
        logic_par_spool=None,
        logic_par_quant_dequant=None,
        real_conv_linear_list=None,
    )
    args = apply_network_defaults(args)
    args.input_points = points
    args.num_points = points
    return args


def sanitize(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def save_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False))


def build_tasks(networks: Sequence[str], accelerators: Sequence[str], points_list: Sequence[int]) -> List[tuple[str, str, int]]:
    return [
        (str(network), str(accelerator), int(points))
        for network in networks
        for accelerator in accelerators
        for points in points_list
    ]


def print_progress(prefix: str, index: int, total: int, network: str, accelerator: str, points: int, extra: str = "") -> None:
    pct = (100.0 * index / total) if total > 0 else 100.0
    suffix = f" | {extra}" if extra else ""
    print(
        f"[{prefix}] {index}/{total} ({pct:5.1f}%) | network={network} accelerator={accelerator} points={points}{suffix}",
        flush=True,
    )


def make_record(
    network: str,
    accelerator: str,
    points: int,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    breakdown = result.get("operator_breakdown", {}) or {}
    row: Dict[str, Any] = {
        "network": network,
        "accelerator": accelerator,
        "input_points": points,
        "total_cycles": float(sum(v.get("cycles", 0.0) for v in breakdown.values())),
        "total_latency_ns": float(sum(v.get("latency_ns", 0.0) for v in breakdown.values())),
        "total_energy_nj": float(sum(v.get("energy_nj", 0.0) for v in breakdown.values())),
    }
    row["total_latency_ms"] = row["total_latency_ns"] / 1e6
    for op, vals in breakdown.items():
        row[f"{op}__cycles"] = float(vals.get("cycles", 0.0))
        row[f"{op}__latency_ns"] = float(vals.get("latency_ns", 0.0))
        row[f"{op}__energy_nj"] = float(vals.get("energy_nj", 0.0))
    return row


def records_to_dataframe(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df.sort_values(["network", "accelerator", "input_points"]).reset_index(drop=True)


def extract_metric_table(
    df: pd.DataFrame,
    network: str,
    accelerator: str,
    metric_suffix: str,
) -> pd.DataFrame:
    sub = df[(df["network"] == network) & (df["accelerator"] == accelerator)].copy()
    if sub.empty:
        return pd.DataFrame()
    metric_cols = [c for c in sub.columns if c.endswith(metric_suffix)]
    out = pd.DataFrame({"input_points": sub["input_points"]})
    for c in metric_cols:
        op = c[: -len(metric_suffix)]
        out[op] = sub[c].fillna(0.0)
    return out.set_index("input_points").sort_index()


def collapse_for_stack(table: pd.DataFrame, top_n: int = STACK_TOP_N) -> pd.DataFrame:
    if table.empty:
        return table
    totals = table.sum(axis=0).sort_values(ascending=False)
    keep = list(totals.head(top_n).index)
    other = [c for c in table.columns if c not in keep]
    out = table[keep].copy()
    if other:
        out["Other"] = table[other].sum(axis=1)
    return out


def plot_stacked(table: pd.DataFrame, title: str, ylabel: str, path: Path) -> None:
    if table.empty:
        return
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    ax.set_title(title)
    ax.set_xlabel("Input points")
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_fps_knn_compare(
    df: pd.DataFrame,
    network: str,
    metric_suffix: str,
    ylabel: str,
    path: Path,
) -> None:
    sub = df[df["network"] == network].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["accelerator", "input_points"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, op in zip(axes, ["FPS", "KNN"]):
        for acc in sorted(sub["accelerator"].unique()):
            ss = sub[sub["accelerator"] == acc]
            col = f"{op}__{metric_suffix}"
            y = ss[col].fillna(0.0) if col in ss.columns else 0.0
            if metric_suffix == "latency_ns":
                y = y / 1e6
            ax.plot(ss["input_points"], y, marker="o", linewidth=2, label=acc)
        ax.set_title(f"{network}: {op}")
        ax.set_xlabel("Input points")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    df.to_csv(out_dir / "all_runs.csv", index=False)
    for network in sorted(df["network"].unique()):
        for accelerator in sorted(df["accelerator"].unique()):
            sub = df[(df["network"] == network) & (df["accelerator"] == accelerator)].copy()
            if not sub.empty:
                sub.to_csv(out_dir / f"{sanitize(network)}__{sanitize(accelerator)}__raw.csv", index=False)


def generate_plots(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for network in sorted(df["network"].unique()):
        baseline_lat = extract_metric_table(df, network, "baseline", "__latency_ns")
        baseline_eng = extract_metric_table(df, network, "baseline", "__energy_nj")

        if not baseline_lat.empty:
            lat_ms = baseline_lat / 1e6
            lat_pct = lat_ms.div(lat_ms.sum(axis=1), axis=0) * 100.0
            plot_stacked(
                collapse_for_stack(lat_ms),
                f"{network} baseline breakdown vs input points (latency, ms)",
                "Latency (ms)",
                plots_dir / f"{sanitize(network)}__baseline_latency_ms_stack.png",
            )
            plot_stacked(
                collapse_for_stack(lat_pct),
                f"{network} baseline breakdown vs input points (latency share, %)",
                "Latency share (%)",
                plots_dir / f"{sanitize(network)}__baseline_latency_pct_stack.png",
            )

        if not baseline_eng.empty:
            eng_pct = baseline_eng.div(baseline_eng.sum(axis=1), axis=0) * 100.0
            plot_stacked(
                collapse_for_stack(baseline_eng),
                f"{network} baseline breakdown vs input points (energy, nJ)",
                "Energy (nJ)",
                plots_dir / f"{sanitize(network)}__baseline_energy_nj_stack.png",
            )
            plot_stacked(
                collapse_for_stack(eng_pct),
                f"{network} baseline breakdown vs input points (energy share, %)",
                "Energy share (%)",
                plots_dir / f"{sanitize(network)}__baseline_energy_pct_stack.png",
            )

        sub = df[(df["network"] == network) & (df["accelerator"].isin(["baseline", "base_v2"]))].copy()
        if not sub.empty:
            plot_fps_knn_compare(
                sub,
                network,
                "latency_ns",
                "Latency (ms)",
                plots_dir / f"{sanitize(network)}__baseline_vs_base_v2__fps_knn_latency.png",
            )
            plot_fps_knn_compare(
                sub,
                network,
                "energy_nj",
                "Energy (nJ)",
                plots_dir / f"{sanitize(network)}__baseline_vs_base_v2__fps_knn_energy.png",
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "all_runs.json"

    if args.use_fast_scheduler and not args.full_scheduler:
        Scheduler.run_simulation = Scheduler.run_simulation_fast

    records = [] if args.overwrite else load_records(json_path)
    done = {
        (str(r.get("network")), str(r.get("accelerator")), int(r.get("input_points")))
        for r in records
        if "network" in r and "accelerator" in r and "input_points" in r
    }

    tasks = build_tasks(args.networks, args.accelerators, args.points)
    total_tasks = len(tasks)
    initial_done = sum(1 for task in tasks if task in done)
    print(
        f"[plan] total_tasks={total_tasks} completed_existing={initial_done} remaining={total_tasks - initial_done}",
        flush=True,
    )

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_tasks, desc="point-sweep", unit="case", dynamic_ncols=True)
        if initial_done:
            pbar.update(initial_done)

    completed = initial_done
    for task_idx, (network, accelerator, points) in enumerate(tasks, start=1):
        key = (network, accelerator, int(points))
        if key in done:
            print_progress("skip", task_idx, total_tasks, network, accelerator, points, "existing result")
            continue

        completed += 1
        remaining_before = total_tasks - completed
        print_progress("run", task_idx, total_tasks, network, accelerator, points, f"remaining_after_start={remaining_before}")
        if pbar is not None:
            pbar.set_postfix_str(f"{network} | {accelerator} | N={points}")

        start_t = time.time()
        sim_args = make_sim_args(network, accelerator, int(points))
        result = run_network_simulation(sim_args)
        elapsed = time.time() - start_t

        record = make_record(network, accelerator, int(points), result)
        records = [
            r for r in records
            if (str(r.get("network")), str(r.get("accelerator")), int(r.get("input_points", -1))) != key
        ]
        records.append(record)
        records = sorted(
            records,
            key=lambda r: (str(r["network"]), str(r["accelerator"]), int(r["input_points"])),
        )
        save_records(json_path, records)
        print_progress(
            "saved",
            task_idx,
            total_tasks,
            network,
            accelerator,
            points,
            f"elapsed={elapsed:.1f}s total_latency_ms={record['total_latency_ms']:.3f}",
        )
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    df = records_to_dataframe(records)
    save_csvs(df, out_dir)
    if not args.skip_plots:
        generate_plots(df, out_dir)
    print(f"[done] output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
