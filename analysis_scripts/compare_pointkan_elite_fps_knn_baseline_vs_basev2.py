#!/usr/bin/env python3
"""
Compare baseline vs base_v2 FPS/KNN cost across PointKAN-Elite variants.

Outputs:
- all_runs.json / all_runs.csv
- per-network comparison CSVs
- plots for FPS/KNN cycles, energy, and area

Default networks:
- pointkan-elite
- pointkan-elite-scanobj
- pointkan-elite-partseg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

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
DEFAULT_POINTS = [1024, 2048, 4096, 8192, 16384, 32768]
OPS = ["FPS", "KNN"]
METRICS = [
    ("cycles", "Cycles"),
    ("energy_nj", "Energy (nJ)"),
    ("area_um2", "Area (um^2)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare baseline vs base_v2 FPS/KNN cost.")
    p.add_argument("--networks", nargs="+", default=DEFAULT_NETWORKS)
    p.add_argument("--accelerators", nargs="+", default=DEFAULT_ACCELERATORS, choices=["baseline", "base_v2"])
    p.add_argument("--points", nargs="+", type=int, default=DEFAULT_POINTS)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "analysis_pointkan_elite_fps_knn_compare"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--full-scheduler", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
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


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def save_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False))


def sanitize(name: str) -> str:
    return name.replace('/', '_').replace(' ', '_')


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
    print(f"[{prefix}] {index}/{total} ({pct:5.1f}%) | network={network} accelerator={accelerator} points={points}{suffix}", flush=True)


def make_record(network: str, accelerator: str, points: int, result: Dict[str, Any]) -> Dict[str, Any]:
    breakdown = result.get("operator_breakdown", {}) or {}
    row: Dict[str, Any] = {
        "network": network,
        "accelerator": accelerator,
        "input_points": points,
        "total_cycles": float(sum(v.get("cycles", 0.0) for v in breakdown.values())),
        "total_latency_ns": float(sum(v.get("latency_ns", 0.0) for v in breakdown.values())),
        "total_energy_nj": float(sum(v.get("energy_nj", 0.0) for v in breakdown.values())),
        "total_area_um2": float(sum(v.get("area_um2", 0.0) for v in breakdown.values())),
    }
    for op in OPS:
        vals = breakdown.get(op, {}) or {}
        row[f"{op}__cycles"] = float(vals.get("cycles", 0.0))
        row[f"{op}__latency_ns"] = float(vals.get("latency_ns", 0.0))
        row[f"{op}__energy_nj"] = float(vals.get("energy_nj", 0.0))
        row[f"{op}__area_um2"] = float(vals.get("area_um2", 0.0))
    return row


def records_to_df(records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(["network", "accelerator", "input_points"]).reset_index(drop=True)


def save_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    df.to_csv(out_dir / 'all_runs.csv', index=False)
    for network in sorted(df['network'].unique()):
        sub = df[df['network'] == network].copy()
        sub.to_csv(out_dir / f'{sanitize(network)}__fps_knn_compare.csv', index=False)


def plot_metric(df: pd.DataFrame, network: str, op: str, metric: str, ylabel: str, out_path: Path) -> None:
    sub = df[df['network'] == network].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for accelerator in sorted(sub['accelerator'].unique()):
        ss = sub[sub['accelerator'] == accelerator].sort_values('input_points')
        y = ss[f'{op}__{metric}'].fillna(0.0)
        ax.plot(ss['input_points'], y, marker='o', linewidth=2, label=accelerator)
    ax.set_title(f'{network}: {op} {metric}')
    ax.set_xlabel('Input points')
    ax.set_ylabel(ylabel)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_grouped_bars(df: pd.DataFrame, network: str, metric: str, ylabel: str, out_path: Path) -> None:
    sub = df[df['network'] == network].copy()
    if sub.empty:
        return
    points = sorted(sub['input_points'].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, op in zip(axes, OPS):
        pivot = sub.pivot_table(index='input_points', columns='accelerator', values=f'{op}__{metric}', aggfunc='first').reindex(points)
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{network}: {op}')
        ax.set_xlabel('Input points')
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(title='accelerator')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def generate_plots(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    for network in sorted(df['network'].unique()):
        for op in OPS:
            for metric, ylabel in METRICS:
                plot_metric(
                    df, network, op, metric, ylabel,
                    plots_dir / f'{sanitize(network)}__{op.lower()}__{metric}_line.png'
                )
        for metric, ylabel in METRICS:
            plot_grouped_bars(
                df, network, metric, ylabel,
                plots_dir / f'{sanitize(network)}__fps_knn__{metric}_bars.png'
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'all_runs.json'

    if not args.full_scheduler:
        Scheduler.run_simulation = Scheduler.run_simulation_fast

    records = [] if args.overwrite else load_records(json_path)
    done = {
        (str(r.get('network')), str(r.get('accelerator')), int(r.get('input_points')))
        for r in records
        if 'network' in r and 'accelerator' in r and 'input_points' in r
    }

    tasks = build_tasks(args.networks, args.accelerators, args.points)
    total_tasks = len(tasks)
    initial_done = sum(1 for task in tasks if task in done)
    print(f'[plan] total_tasks={total_tasks} completed_existing={initial_done} remaining={total_tasks - initial_done}', flush=True)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_tasks, desc='fps-knn-compare', unit='case', dynamic_ncols=True)
        if initial_done:
            pbar.update(initial_done)

    for task_idx, (network, accelerator, points) in enumerate(tasks, start=1):
        key = (network, accelerator, int(points))
        if key in done:
            print_progress('skip', task_idx, total_tasks, network, accelerator, points, 'existing result')
            continue

        if pbar is not None:
            pbar.set_postfix_str(f'{network} | {accelerator} | N={points}')
        print_progress('run', task_idx, total_tasks, network, accelerator, points)
        start_t = time.time()
        sim_args = make_sim_args(network, accelerator, int(points))
        result = run_network_simulation(sim_args)
        elapsed = time.time() - start_t
        record = make_record(network, accelerator, int(points), result)
        records = [
            r for r in records
            if (str(r.get('network')), str(r.get('accelerator')), int(r.get('input_points', -1))) != key
        ]
        records.append(record)
        records = sorted(records, key=lambda r: (str(r['network']), str(r['accelerator']), int(r['input_points'])))
        save_records(json_path, records)
        print_progress('saved', task_idx, total_tasks, network, accelerator, points, f"elapsed={elapsed:.1f}s FPScy={record['FPS__cycles']:.0f} KNNcy={record['KNN__cycles']:.0f}")
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    df = records_to_df(records)
    save_csvs(df, out_dir)
    if not args.skip_plots:
        generate_plots(df, out_dir)
    print(f'[done] output_dir={out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
