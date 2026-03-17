#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用真实 checkpoint 与 ModelNet40 数据评估仿真器
================================================

与 PointKAN-elite/classification_ModelNet40/inference_and_kan_analysis.py 及
run_inference_kan.sh 中模型权重、ModelNet40 数据一致，先（可选）跑真实推理得到准确率，
再跑 PPA 仿真，所有日志与报告写入 sim_log 目录。

使用示例（在 PointKAN-pytorch 根目录）:
  python -m pointkan_sim.run_sim_with_real_eval
  python -m pointkan_sim.run_sim_with_real_eval --checkpoint PointKAN-elite/classification_ModelNet40/checkpoints/pointKANElite-pointkan_elite_exp-42
  python -m pointkan_sim.run_sim_with_real_eval --run_real_inference --mnsim_full --output report.json

完整 shape 分析（Conv/Linear + BN/LayerNorm/ReLU/池化/S-Pool 等）:
  必须使用真实数据模式并开启 --show_shapes，会做一次真实前向并用 hook 收集各层维度后打印。
  方式一（推荐）:
    USE_REAL_DATA=1 SHOW_SHAPES=1 ./run_sim_with_real_eval.sh
  方式二（仅 shape，不跑 MNSIM/NeuroSim）:
    python -m pointkan_sim.run_sim_with_real_eval --use_real_data --show_shapes --batch_size 2
  日志中会先后出现「Conv/Linear 层 shape」与「非线性算子激活」两段，含 in/out shape、numel、字节数。

时钟（双时钟，可脚本配置）:
  NeuroSim（Conv/Linear）：由 NEUROSIM_CLK_FREQ 传入频率(Hz)，可变（如 100M/200M），不设则用 NeuroSim 关键路径。
  数字逻辑（SPool/LayerNorm/ReLU 等）：由 --logic_clk_freq / LOGIC_CLK_FREQ 传入频率(Hz)，默认 1 GHz。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 工程根目录（PointKAN-pytorch）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = PROJECT_ROOT / "PointKAN-elite" / "classification_ModelNet40"

# 与 run_inference_kan.sh 默认一致
DEFAULT_CKPT = "checkpoints/pointKANElite-pointkan_elite_exp-42"
DEFAULT_NUM_POINTS = 1024
DEFAULT_NUM_CLASSES = 40

DEFAULT_SIM_RESULTS_DIR = "sim_results"


def _collect_runtime_memory_rows(sim_once: dict):
    sim_result = sim_once.get("sim_result")
    events = getattr(sim_result, "events", []) if sim_result is not None else []
    rows = []
    tracked_sram_buffers = {
        "feature_sram_buffer",
        "systolic_feature_sram_buffer",
        "weight_sram_buffer",
        "dist_sram_buffer",
        "fps_min_distance_table_sram",
        "sort_cache_sram",
    }

    def _infer_fps_knn_buffer(e, meta):
        op = (getattr(e, "operator", "") or "").strip()
        stage = (getattr(e, "stage", "") or "").strip()
        operation = (getattr(e, "operation", "") or "").strip()
        if op == "FPS":
            if stage == "min_distance_update":
                return ("fps_min_distance_table_sram", meta.get("buffer_phase", "update"), meta.get("resource_id", "fps_min_table_shared"))
            if stage == "merge_sort_reg_file":
                return ("sort_cache_sram", meta.get("buffer_phase", "read" if "read" in operation else "write"), meta.get("resource_id", "sort_cache_shared"))
            if stage in ("bitonic_sort", "merge_sort") and meta.get("buffer_name"):
                return (str(meta.get("buffer_name")), meta.get("buffer_phase", "sort_read"), meta.get("resource_id", ""))
        if op == "KNN":
            if stage == "sram_write":
                return ("dist_sram_buffer", meta.get("buffer_phase", "write"), meta.get("resource_id", "dist_sram_shared"))
            if stage == "merge_sort_reg_file":
                return ("sort_cache_sram", meta.get("buffer_phase", "read" if "read" in operation else "write"), meta.get("resource_id", "sort_cache_shared"))
            if stage in ("bitonic_sort", "merge_sort") and meta.get("buffer_name"):
                return (str(meta.get("buffer_name")), meta.get("buffer_phase", "sort_read"), meta.get("resource_id", ""))
        return ("", "", "")

    for e in events:
        op = (getattr(e, "operator", "") or "").strip()
        meta = getattr(e, "metadata", None) or {}
        buffer_name = str(meta.get("buffer_name", "") or "")
        inferred_buffer_name, inferred_phase, inferred_resource = _infer_fps_knn_buffer(e, meta)
        if not buffer_name and inferred_buffer_name:
            buffer_name = inferred_buffer_name
        is_tracked_sram = buffer_name in tracked_sram_buffers
        if op not in ("DRAM", "LayerBuffer") and not is_tracked_sram:
            continue
        event_bytes = meta.get("bytes", getattr(e, "data_size", 0) or 0)
        if (not event_bytes) and meta.get("source") == "scalesim_memory":
            if op == "DRAM":
                event_bytes = (
                    int(meta.get("feature_input_dram_bytes", 0) or 0)
                    + int(meta.get("feature_output_dram_bytes", 0) or 0)
                    + int(meta.get("weight_dram_bytes", 0) or 0)
                )
            elif op == "LayerBuffer":
                bytes_per_access = int(meta.get("bytes_per_access", 0) or 0)
                if bytes_per_access <= 0:
                    data_bytes = int(meta.get("data_bytes", 0) or 0)
                    bytes_per_access = data_bytes if data_bytes > 0 else 2
                event_bytes = (
                    int(meta.get("sram_ifmap_reads", 0) or 0)
                    + int(meta.get("sram_filter_reads", 0) or 0)
                    + int(meta.get("sram_ofmap_writes", 0) or 0)
                ) * bytes_per_access
        if not event_bytes:
            if buffer_name == "sort_cache_sram":
                event_bytes = int(meta.get("num_reads", 0) or meta.get("num_writes", 0) or 0) * 4
            elif buffer_name == "dist_sram_buffer":
                event_bytes = int(meta.get("bytes", 0) or 0)
            elif buffer_name == "fps_min_distance_table_sram":
                event_bytes = int(meta.get("bytes", 0) or 0)
        rows.append({
            "event_id": getattr(e, "event_id", 0),
            "owner_operator": str(meta.get("owner_operator", op) or op),
            "operator": "SRAM" if (op == "LayerBuffer" or is_tracked_sram) else op,
            "memory_tier": "SRAM" if (op == "LayerBuffer" or is_tracked_sram) else op,
            "cycle": int(getattr(e, "cycle", 0) or 0),
            "end_cycle": int(getattr(e, "end_cycle", 0) or 0),
            "duration": int(getattr(e, "duration", 0) or 0),
            "clock_period_ns": float(getattr(e, "clock_period_ns", 1.0) or 1.0),
            "latency_ns": float(int(getattr(e, "duration", 0) or 0) * float(getattr(e, "clock_period_ns", 1.0) or 1.0)),
            "energy_nj": float(getattr(e, "energy", 0.0) or 0.0),
            "power_mw": float(getattr(e, "power", 0.0) or 0.0),
            "stage": getattr(e, "stage", "") or "",
            "sub_stage": getattr(e, "sub_stage", "") or "",
            "circuit_module": getattr(e, "circuit_module", "") or "",
            "operation": getattr(e, "operation", "") or "",
            "layer_name": meta.get("layer_name", ""),
            "layer_type": meta.get("layer_type", ""),
            "buffer_name": buffer_name or ("inter_layer_buffer" if (op == "LayerBuffer" or is_tracked_sram) else "offchip_dram"),
            "buffer_slot": meta.get("buffer_slot", ""),
            "buffer_phase": meta.get("buffer_phase", inferred_phase),
            "resource_id": meta.get("resource_id", inferred_resource),
            "source": meta.get("source", ""),
            "target": meta.get("target", ""),
            "bytes": int(event_bytes or 0),
            "overlapped": bool(getattr(e, "overlapped", False)),
            "runtime_mode": "actual_runtime_event",
        })
    return rows


def _summarize_runtime_memory_rows(rows):
    from collections import defaultdict
    by_key = defaultdict(lambda: {"events": 0, "bytes": 0, "energy_nj": 0.0, "latency_ns": 0.0})
    for row in rows:
        key = (row["operator"], row.get("owner_operator", ""), row["buffer_name"], row["layer_name"], row["buffer_phase"], row["resource_id"])
        agg = by_key[key]
        agg["events"] += 1
        agg["bytes"] += int(row.get("bytes", 0) or 0)
        agg["energy_nj"] += float(row.get("energy_nj", 0.0) or 0.0)
        if not row.get("overlapped", False):
            agg["latency_ns"] += float(row.get("latency_ns", 0.0) or 0.0)
    out = []
    for (operator, owner_operator, buffer_name, layer_name, buffer_phase, resource_id), agg in sorted(by_key.items(), key=lambda kv: (-kv[1]["energy_nj"], kv[0])):
        out.append({
            "operator": operator,
            "owner_operator": owner_operator,
            "buffer_name": buffer_name,
            "layer_name": layer_name,
            "buffer_phase": buffer_phase,
            "resource_id": resource_id,
            "events": agg["events"],
            "bytes": agg["bytes"],
            "energy_nj": agg["energy_nj"],
            "latency_ns": agg["latency_ns"],
        })
    return out


def _scheduler_feature_mode(mem_contract) -> tuple[str, bool, int]:
    plan = getattr(mem_contract, "feature_residency_plan", []) or []
    if plan:
        row = plan[0]
        return (
            str(row.get("pipeline_mode", "without_pipeline") or "without_pipeline"),
            bool(row.get("double_buffering", False)),
            int(row.get("buffer_slots", 1) or 1),
        )
    return ("without_pipeline", False, 1)


def _build_scheduler_buffer_constraints(mem_contract, accelerator: str):
    mode = str(accelerator or "baseline")
    pipeline_mode, double_buffering, slots = _scheduler_feature_mode(mem_contract)
    rows = []
    for rel in getattr(mem_contract, "buffer_relationships", []) or []:
        overlap = "conditional"
        if rel.buffer_name == "feature_sram_buffer":
            overlap = "disallowed" if pipeline_mode != "pipeline" else ("allowed_via_double_buffering" if double_buffering else "limited_same_slot")
        elif rel.buffer_name in ("dist_sram_buffer", "fps_min_distance_table_sram", "sort_cache_sram", "weight_sram_buffer", "systolic_feature_sram_buffer"):
            overlap = "serialized_on_same_resource"
        rows.append({
            "accelerator_mode": mode,
            "buffer_name": rel.buffer_name,
            "shared_group": rel.shared_group,
            "scheduler_resource_id": rel.scheduler_resource_id,
            "physical_sharing": rel.physical_sharing,
            "mutual_exclusion": rel.mutual_exclusion,
            "reset_scope": rel.reset_scope,
            "pipeline_mode": pipeline_mode,
            "double_buffering": bool(double_buffering),
            "buffer_slots": int(slots),
            "overlap_policy": overlap,
            "current_model_support": "implemented_runtime" if rel.buffer_name in ("feature_sram_buffer", "weight_sram_buffer", "dist_sram_buffer", "fps_min_distance_table_sram", "sort_cache_sram") else "implemented_contract",
            "note": rel.note,
        })
    return rows


def _build_scheduler_timing_constraints(mem_contract, accelerator: str):
    mode = str(accelerator or "baseline")
    pipeline_mode, double_buffering, slots = _scheduler_feature_mode(mem_contract)
    rows = []
    def add(constraint_id, stage_scope, producer, consumer, buffer_name, dep_type, constraint, support, note=""):
        rows.append({
            "constraint_id": constraint_id,
            "accelerator_mode": mode,
            "pipeline_mode": pipeline_mode,
            "double_buffering": bool(double_buffering),
            "buffer_slots": int(slots),
            "stage_scope": stage_scope,
            "producer": producer,
            "consumer": consumer,
            "buffer_name": buffer_name,
            "dependency_type": dep_type,
            "constraint": constraint,
            "current_model_support": support,
            "note": note,
        })

    add("feature_layer_order", "cross_layer", "layer_n.output_write", "layer_n+1.input_read", "feature_sram_buffer", "data_dependency", "Consumer layer cannot read activation before producer layer commits it to feature SRAM or DRAM spill path.", "implemented_runtime", "Current runtime uses feature residency plan and slot tagging.")
    add("feature_slot_mutex", "cross_layer", "feature_slot_i.write", "feature_slot_i.read/write", "feature_sram_buffer", "resource_mutex", "Same feature SRAM slot is exclusive; without pipeline a single slot is reused serially.", "implemented_runtime", "resource_id=feature_sram_slot_<n>.")
    if pipeline_mode == "pipeline" and double_buffering:
        add("feature_double_buffer", "cross_layer", "feature_slot_i.write", "feature_slot_j.read", "feature_sram_buffer", "overlap_rule", "Adjacent layers may overlap only when using different slots under double buffering.", "implemented_runtime", "Current runtime has slot tags; fine-grain overlap still conservative.")
    else:
        add("feature_no_overlap", "cross_layer", "feature_slot_0.write", "feature_slot_0.read", "feature_sram_buffer", "overlap_rule", "Without pipeline or without double buffering, next-layer feature read must wait for previous-layer write completion on the same slot.", "implemented_runtime", "Matches current default mode.")

    if mode == "baseline":
        add("fps_base_v1", "GAM", "phase2_distance_compute(full vector)", "fps_bitonic_sort", "fps_min_distance_table_sram", "data_dependency", "base_v1 FPS waits until all distances/min-table updates for one iteration complete before sort starts.", "implemented_runtime", "Full-vector sorting path.")
        add("knn_base_v1", "GAM", "phase2_distance_compute(full vector)", "knn_bitonic_sort", "dist_sram_buffer", "data_dependency", "base_v1 KNN waits until all query distances are written to dist SRAM before top-k sort starts.", "implemented_runtime", "Full-vector sorting path.")
    elif mode == "base_v2":
        add("fps_base_v2_stream", "GAM", "distance_batch64 + min_update", "fps_merge_sort_batch", "sort_cache_sram", "data_dependency", "Each 64-distance FPS batch updates min-table, then merge-sort consumes prior cached winner and current batch before cache writeback.", "implemented_runtime", "sort_cache_sram read/write visible in runtime table.")
        add("knn_base_v2_stream", "GAM", "dist_sram_write(batch64)", "knn_merge_sort_batch", "sort_cache_sram", "data_dependency", "Each 64-distance KNN batch writes dist SRAM, reloads cached top-k, performs merge-sort, then writes updated cache.", "implemented_runtime", "sort_cache_sram read/sort_read/write visible in runtime table.")
        add("sort_cache_mutex", "GAM", "fps_or_knn_sort_cache_access", "fps_or_knn_sort_cache_access", "sort_cache_sram", "resource_mutex", "sort_cache_sram is shared over time between FPS and KNN; accesses serialize on sort_cache_shared/merge_sort_shared resources.", "implemented_runtime", "base_v2 streaming sorter cache.")
    else:
        add("acc_placeholder", "global", "feature_sram_buffer", "rram_cim_compute", "feature_sram_buffer", "design_note", "ACC constraints deferred: weights fixed in RRAM-CIM, focus on feature movement and lighter FPS/KNN interaction.", "planned", "To refine with user-provided acc details.")

    add("weight_tile_refill", "conv_linear", "DRAM weight_read(tile)", "weight_sram_buffer -> Conv/Linear", "weight_sram_buffer", "data_dependency", "Weight tile must be refilled into weight_sram_buffer before corresponding Conv/Linear compute tile consumes it.", "implemented_runtime", "32KB weight SRAM causes repeated tiled refills in baseline/base_v2.")
    add("weight_sram_mutex", "conv_linear", "conv_or_linear.weight_stage", "conv_or_linear.weight_stage", "weight_sram_buffer", "resource_mutex", "weight_sram_buffer is shared across Conv/Linear ops and is exclusive within each active staging window.", "implemented_runtime", "resource_id=weight_sram_shared.")
    add("local_feature_mutex", "conv_linear", "conv_or_linear.compute_local", "conv_or_linear.compute_local", "systolic_feature_sram_buffer", "resource_mutex", "Operator-local systolic feature SRAM is serialized on the shared local buffer resource.", "implemented_runtime", "resource_id=systolic_feature_sram_shared.")
    add("dist_buffer_mutex", "GAM", "knn_or_fps_distance_write", "sorter_read_same_bank", "dist_sram_buffer", "resource_mutex", "Distance SRAM write and sorter read must serialize when they target the same shared dist SRAM resource/bank.", "implemented_runtime", "Current runtime models shared resource IDs; bank-level refinement can be added later.")
    add("fps_min_table_mutex", "GAM", "fps_min_update", "fps_sort_read", "fps_min_distance_table_sram", "resource_mutex", "FPS min-distance table update and sorter read are mutually exclusive on the same logical table region.", "implemented_runtime", "resource_id=fps_min_table_shared.")
    return rows


def get_sim_mode_results_dir(args: argparse.Namespace) -> str:
    """
    根据仿真模式生成 sim_results 输出目录名，便于区分 base_v1/base_v2/acc、是否 pipeline、acc 的 mnsim/neurosim 模式。
    格式: sim_results_<accelerator>[_scalesim][_mnsim_full|_neurosim_full|_mnsim|_neurosim|_fallback][_pipeline]
    """
    acc = (getattr(args, "accelerator", None) or "baseline").strip().lower()
    parts = [acc]
    if acc in ("baseline", "base_v2"):
        if getattr(args, "use_scalesim_systolic", False):
            parts.append("scalesim")
    elif acc == "acc":
        if getattr(args, "neurosim_full", False):
            parts.append("neurosim_full")
        elif getattr(args, "mnsim_full", False):
            parts.append("mnsim_full")
        else:
            backend = getattr(args, "cim_backend", None) or "mnsim"
            parts.append(backend)
    if getattr(args, "qd_en", 0):
        parts.append("pipeline")
    return "sim_results_" + "_".join(parts)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数（网络、加速器、checkpoint、真实数据/仿真模式、BN/量化开关、导出选项等）。"""
    parser = argparse.ArgumentParser(
        description="使用真实 checkpoint 与 ModelNet40 配置运行 PPA 仿真，可选先跑真实推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"检查点目录或 .pth 路径；默认相对 classification_ModelNet40 的 {DEFAULT_CKPT}",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="ModelNet40 数据根目录（预留；当前 inference 脚本使用 data.py 内路径）",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=f"点云点数，需与训练/推理一致，默认 {DEFAULT_NUM_POINTS}",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=DEFAULT_NUM_CLASSES,
        help=f"分类数，默认 {DEFAULT_NUM_CLASSES}",
    )
    parser.add_argument(
        "--run_real_inference",
        action="store_true",
        help="先运行 inference_and_kan_analysis 在 ModelNet40 上推理，将日志写入 sim_log，再跑仿真",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="baseline",
        help="加速器配置名（baseline=base_v1, base_v2, acc）",
    )
    parser.add_argument(
        "--use_scalesim_systolic",
        action="store_true",
        help="base_v1/base_v2 下用 SCALE-Sim 统计 Conv/Linear 周期与 MAC/访问；能耗仍用本工程内存建模。需 SCALE_SIM_ROOT 或 --scalesim_root",
    )
    parser.add_argument(
        "--scalesim_root",
        type=str,
        default=None,
        help="SCALE-Sim 根目录，默认用环境变量 SCALE_SIM_ROOT 或 /workspace/SCALE-Sim",
    )
    parser.add_argument(
        "--scalesim_config",
        type=str,
        default=None,
        help="SCALE-Sim 配置文件路径，默认 scalesim_root/configs/demo_64x64_32kb_ws.cfg",
    )
    parser.add_argument(
        "--systolic_rows",
        type=int,
        default=None,
        metavar="N",
        help="脉动阵列行数（ArrayHeight），仅 baseline+SCALE-Sim 时生效；默认 64；例：--systolic_rows 16 得 16×cols",
    )
    parser.add_argument(
        "--systolic_cols",
        type=int,
        default=None,
        metavar="N",
        help="脉动阵列列数（ArrayWidth），仅 baseline+SCALE-Sim 时生效；默认 64；例：--systolic_cols 16 得 16×16",
    )
    parser.add_argument(
        "--systolic_sram_size_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="SCALE-Sim 局部 SRAM 总容量（bytes），用于 feature_local/weight SRAM 的 CACTI runtime；默认 32768",
    )
    parser.add_argument(
        "--systolic_ifmap_sram_size_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="SCALE-Sim IFMAP 本地 SRAM 容量（bytes）；不设则回退到 --systolic_sram_size_bytes",
    )
    parser.add_argument(
        "--systolic_filter_sram_size_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="SCALE-Sim FILTER/weight 本地 SRAM 容量（bytes）；不设则回退到 --systolic_sram_size_bytes",
    )
    parser.add_argument(
        "--systolic_ofmap_sram_size_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="SCALE-Sim OFMAP 本地 SRAM 容量（bytes）；不设则回退到 --systolic_sram_size_bytes",
    )
    parser.add_argument(
        "--systolic_sram_block_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="SCALE-Sim 局部 SRAM block size（bytes），用于 CACTI；默认 64",
    )
    parser.add_argument(
        "--systolic_sram_num_banks",
        type=int,
        default=None,
        metavar="N",
        help="SCALE-Sim 局部 SRAM bank 数，用于 CACTI；默认 1",
    )
    parser.add_argument(
        "--systolic_sram_bus_width_bits",
        type=int,
        default=None,
        metavar="BITS",
        help="SCALE-Sim 局部 SRAM 总线位宽（bits），用于 CACTI；默认 128",
    )
    parser.add_argument(
        "--systolic_sram_freq_hz",
        type=float,
        default=None,
        metavar="HZ",
        help="SCALE-Sim 局部 SRAM 时钟频率（Hz）；不设则跟随数字逻辑时钟",
    )
    parser.add_argument(
        "--mnsim_full",
        action="store_true",
        help="对 conv1d/linear 使用 MNSIM 全 NetStruct",
    )
    parser.add_argument(
        "--neurosim_full",
        action="store_true",
        help="对 conv1d/linear 使用 NeuroSim 全网络仿真（DNN_NeuroSim main）；与 --mnsim_full 二选一时优先 NeuroSim",
    )
    parser.add_argument(
        "--cim_backend",
        type=str,
        default=None,
        choices=["mnsim", "neurosim", "fallback"],
        help="Conv/Linear 单层估算后端：mnsim（默认）、neurosim（DNN_NeuroSim）、fallback（解析回退）；无全网络时生效",
    )
    parser.add_argument(
        "--neurosim_root",
        type=str,
        default=None,
        help="NeuroSim 根目录（Inference_pytorch），默认 /workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    )
    parser.add_argument(
        "--neurosim_clk_freq",
        type=float,
        default=None,
        help="NeuroSim 全网络仿真时工作频率上限(Hz)，如 100000000=100MHz；不设则使用 NeuroSim 关键路径频率",
    )
    parser.add_argument(
        "--neurosim_technode",
        type=int,
        default=None,
        help="NeuroSim 工艺节点(nm)，如 32；仅支持 130/90/65/45/32/22/14/10/7，28 会自动改为 32",
    )
    parser.add_argument(
        "--logic_clk_freq",
        type=float,
        default=None,
        metavar="Hz",
        help="数字逻辑时钟频率(Hz)，用于 SPool/LayerNorm/ReLU 等；默认 1e9=1GHz；不设则用 hw_config 默认",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告输出文件名（写入 sim_log 目录），如 report.json",
    )
    parser.add_argument(
        "--sim_log_dir",
        type=str,
        default="sim_log",
        help="仿真与推理日志统一输出目录",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="真实推理时禁用可视化（与 inference_and_kan_analysis 一致）",
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        help="用 ModelNet40 真实样本与 checkpoint 权重：加载模型与数据集，从真实前向收集 conv/linear 维度驱动 MNSIM，按批调度仿真并汇总准确率与 PPA",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="--use_real_data 时测试集 batch size",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="--use_real_data 时最多跑多少 batch（默认全量测试集）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="--use_real_data 时运行设备（真实前向含 FPS 等需 GPU，无 GPU 时可改用 --use_real_data 不加或先跑仿真）",
    )
    parser.add_argument(
        "--show_shapes",
        action="store_true",
        help="--use_real_data 时打印 LayerDef 与真实前向的每层 conv/linear 的输入激活与权重 shape（及 indata/rdata 字节）",
    )
    parser.add_argument(
        "--sim_results_dir",
        type=str,
        default=DEFAULT_SIM_RESULTS_DIR,
        help="--use_real_data 时 PPA 报告与 breakdown 输出目录；为默认值时按仿真模式自动命名（如 sim_results_baseline_scalesim）",
    )
    parser.add_argument(
        "--bn_en",
        type=int,
        default=1,
        choices=[0, 1],
        help="BN 融合开关：1=启用融合（BN 不单独计入），0=关闭融合（保留 BN 事件）",
    )
    parser.add_argument(
        "--qd_en",
        type=int,
        default=0,
        choices=[0, 1],
        help="量化/反量化开关：1=插入 FP32↔INT8 量化事件，0=不插入",
    )
    parser.add_argument(
        "--export_memory_transactions",
        action="store_true",
        help="导出内存交互事务序列（memory_transactions.txt）",
    )
    parser.add_argument(
        "--inter_layer_sram_bw",
        type=int,
        default=None,
        metavar="B/CYCLE",
        help="覆盖层间 SRAM 带宽（bytes/cycle），用于消融/sweep；如 128、256、512",
    )
    parser.add_argument(
        "--inter_layer_sram_size_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="覆盖层间 feature SRAM 总容量（bytes），用于消融/sweep；如 65536、131072、262144",
    )
    parser.add_argument(
        "--dram_bw_bytes_per_cycle",
        type=int,
        default=None,
        metavar="B/CYCLE",
        help="覆盖抽象 DRAM 带宽（bytes/cycle），用于消融/sweep；如 16、26、32、64",
    )
    parser.add_argument(
        "--dram_channels",
        type=int,
        default=None,
        metavar="N",
        help="覆盖 DRAMSim3 ini 中的 channel 数，用于 sweep；如 1、2、4",
    )
    parser.add_argument(
        "--feature_buffer_pipeline_mode",
        type=str,
        choices=["without_pipeline", "pipeline"],
        default=None,
        help="层间 feature SRAM 调度模式：without_pipeline 或 pipeline",
    )
    parser.add_argument(
        "--feature_double_buffering",
        action="store_true",
        help="层间 feature SRAM 启用双缓冲；通常与 --feature_buffer_pipeline_mode pipeline 配合",
    )
    parser.add_argument(
        "--feature_buffer_slots",
        type=int,
        default=None,
        metavar="N",
        help="层间 feature SRAM 并发槽数；不设则由 pipeline_mode/double_buffering 自动推导",
    )
    parser.add_argument(
        "--export_memory_contract",
        action="store_true",
        help="导出逐层/逐算子内存契约分析（memory_contract.txt + memory_contract.csv）",
    )
    parser.add_argument(
        "--logic_par_layernorm",
        type=int,
        default=None,
        metavar="N",
        help="逻辑电路 LayerNorm 并行度（≥1），不设则用 hw_config 默认",
    )
    parser.add_argument(
        "--logic_par_relu",
        type=int,
        default=None,
        metavar="N",
        help="逻辑电路 ReLU 并行度（≥1）",
    )
    parser.add_argument(
        "--logic_par_spool",
        type=int,
        default=None,
        metavar="N",
        help="逻辑电路 SPool 并行度（≥1）",
    )
    parser.add_argument(
        "--logic_par_quant_dequant",
        type=int,
        default=None,
        metavar="N",
        help="逻辑电路量化/反量化并行度（≥1）",
    )
    return parser.parse_args()


def _ensure_sim_log(sim_log_dir: str) -> Path:
    p = Path(sim_log_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_real_inference(
    checkpoint_path: str,
    classification_dir: Path,
    sim_log_dir: Path,
    num_points: int,
    no_vis: bool,
) -> int:
    """
    在 classification_ModelNet40 目录下执行 inference_and_kan_analysis.py，
    将 stdout/stderr 写入 sim_log/real_inference_log.txt，返回进程退出码。
    """
    script = classification_dir / "inference_and_kan_analysis.py"
    if not script.is_file():
        print(f"[run_sim_with_real_eval] 未找到 {script}，跳过真实推理。")
        return -1

    cmd = [
        sys.executable,
        str(script),
        "--checkpoint",
        checkpoint_path,
        "--num_points",
        str(num_points),
        "--no_vis",
    ]

    log_file = sim_log_dir / "real_inference_log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"CWD: {classification_dir}\n\n")
    with open(log_file, "a", encoding="utf-8") as f:
        ret = subprocess.run(
            cmd,
            cwd=str(classification_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(f"[run_sim_with_real_eval] 真实推理日志已写入: {log_file}")
    return ret.returncode


def resolve_checkpoint_path(checkpoint_arg: Optional[str]) -> str:
    """解析 checkpoint 路径：若为相对路径则优先相对 classification_ModelNet40。"""
    if not checkpoint_arg:
        path = CLASSIFICATION_DIR / DEFAULT_CKPT
    else:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            # 先尝试相对当前目录，再尝试相对 classification 目录
            if not path.exists():
                alt = CLASSIFICATION_DIR / checkpoint_arg
                if alt.exists() or (alt / "best_checkpoint.pth").exists():
                    path = alt
            path = path.resolve()
    return str(path)


def main() -> int:
    args = _parse_args()

    sim_log_path = _ensure_sim_log(args.sim_log_dir)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from datetime import datetime
    from pointkan_sim.utils.logger import setup_logger, LogLevel
    log_path = sim_log_path / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(level=LogLevel.INFO, log_file=str(log_path))

    from pointkan_sim.main import (
        load_network,
        get_accelerator_config,
        compute_activation_shapes,
        _write_report_to_file,
        log_info,
        log_section,
        log_separator,
    )
    from pointkan_sim.memory.transactions import extract_memory_transactions, summarize_memory_transactions
    from pointkan_sim.memory.memory_contract import (
        analyze_memory_contract,
        format_memory_contract_report,
        iter_buffer_relationship_rows,
        iter_buffer_requirement_rows,
        iter_memory_contract_rows,
    )

    # -------------------------------------------------------------------------
    # 模式：用 ModelNet40 真实样本 + 真实权重 调度仿真，MNSIM 由真实前向得到的层维度驱动
    # -------------------------------------------------------------------------
    if args.use_real_data:
        from pointkan_sim.configs.hw_config import set_digital_clock_freq_hz, set_logic_parallelism
        from pointkan_sim.constants import (
            KEY_CYCLE,
            KEY_DURATION,
            KEY_END_CYCLE,
            KEY_ENERGY_NJ,
            KEY_IDX,
            KEY_LAYER_NAME,
            KEY_OPERATION,
            KEY_OPERATOR,
            KEY_STAGE,
            OPERATOR_SEQUENCE_HEADER,
        )
        from pointkan_sim.integration.real_data_eval import (
            load_model_and_dataset,
            collect_conv_linear_from_forward,
            collect_nonlinear_activations_from_forward,
            real_data_overrides_from_conv_linear_list,
            log_conv_linear_shapes,
            log_nonlinear_activation_shapes,
            run_real_inference_and_sim_over_dataset,
            RealEvalResult,
        )
        from pointkan_sim.integration.mnsim_netstruct import run_mnsim_full
        import math

        log_section("真实数据评估：ModelNet40 样本 + checkpoint 权重 + MNSIM 对应层 PPA")
        log_info(f"Checkpoint: {checkpoint_path}")
        log_info(f"num_points={args.num_points}, num_classes={args.num_classes}, batch_size={args.batch_size}")

        try:
            net, test_loader, criterion = load_model_and_dataset(
                checkpoint_path,
                num_points=args.num_points,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                workers=0,
                device=args.device,
            )
        except Exception as e:
            log_info(f"加载模型/数据集失败: {e}")
            return 1

        # 层定义（与后续 MNSIM、仿真共用）
        hw_config = get_accelerator_config(args.accelerator)
        # 用户可控开关：BN 融合 / 量化反量化 / CIM 后端 / SCALE-Sim（baseline 脉动 Conv/Linear）
        hw_config.bn_fuse_with_conv_linear = bool(args.bn_en)
        hw_config.enable_w8a8_quant_dequant = bool(args.qd_en)
        if getattr(args, "cim_backend", None):
            hw_config.cim_backend = args.cim_backend
        if getattr(args, "neurosim_root", None):
            hw_config.neurosim_root = args.neurosim_root
        if getattr(args, "use_scalesim_systolic", False) and getattr(hw_config, "is_baseline", False):
            hw_config.use_scalesim_for_systolic = True
            if getattr(args, "scalesim_root", None):
                hw_config.scalesim_root = args.scalesim_root
            if getattr(args, "scalesim_config", None):
                hw_config.scalesim_config_path = args.scalesim_config
            # 脉动阵列尺寸（仅 baseline+SCALE-Sim：传入 SCALE-Sim ArrayHeight/ArrayWidth）
            if getattr(hw_config, "systolic_array", None) is not None:
                sr = getattr(args, "systolic_rows", None)
                sc = getattr(args, "systolic_cols", None)
                if sr is not None and sr >= 1:
                    hw_config.systolic_array.rows = int(sr)
                if sc is not None and sc >= 1:
                    hw_config.systolic_array.cols = int(sc)
                if getattr(args, "systolic_sram_size_bytes", None) is not None:
                    hw_config.systolic_array.sram_size_bytes = max(1, int(args.systolic_sram_size_bytes))
                if getattr(args, "systolic_ifmap_sram_size_bytes", None) is not None:
                    hw_config.systolic_array.ifmap_sram_size_bytes = max(1, int(args.systolic_ifmap_sram_size_bytes))
                if getattr(args, "systolic_filter_sram_size_bytes", None) is not None:
                    hw_config.systolic_array.filter_sram_size_bytes = max(1, int(args.systolic_filter_sram_size_bytes))
                if getattr(args, "systolic_ofmap_sram_size_bytes", None) is not None:
                    hw_config.systolic_array.ofmap_sram_size_bytes = max(1, int(args.systolic_ofmap_sram_size_bytes))
                if getattr(args, "systolic_sram_block_bytes", None) is not None:
                    hw_config.systolic_array.sram_block_bytes = max(1, int(args.systolic_sram_block_bytes))
                if getattr(args, "systolic_sram_num_banks", None) is not None:
                    hw_config.systolic_array.sram_num_banks = max(1, int(args.systolic_sram_num_banks))
                if getattr(args, "systolic_sram_bus_width_bits", None) is not None:
                    hw_config.systolic_array.sram_bus_width_bits = max(1, int(args.systolic_sram_bus_width_bits))
                if getattr(args, "systolic_sram_freq_hz", None) is not None and float(args.systolic_sram_freq_hz) > 0:
                    hw_config.systolic_array.sram_freq_hz = float(args.systolic_sram_freq_hz)
                if any(v is not None for v in (sr, sc, getattr(args, "systolic_sram_size_bytes", None), getattr(args, "systolic_ifmap_sram_size_bytes", None), getattr(args, "systolic_filter_sram_size_bytes", None), getattr(args, "systolic_ofmap_sram_size_bytes", None), getattr(args, "systolic_sram_block_bytes", None), getattr(args, "systolic_sram_num_banks", None), getattr(args, "systolic_sram_bus_width_bits", None), getattr(args, "systolic_sram_freq_hz", None))):
                    log_info(
                        "脉动阵列/局部SRAM已按参数设置: "
                        f"rows={hw_config.systolic_array.rows} cols={hw_config.systolic_array.cols} "
                        f"sram_size_bytes={hw_config.systolic_array.sram_size_bytes} "
                        f"sram_block_bytes={hw_config.systolic_array.sram_block_bytes} "
                        f"sram_num_banks={hw_config.systolic_array.sram_num_banks} "
                        f"sram_bus_width_bits={hw_config.systolic_array.sram_bus_width_bits} "
                        f"sram_freq_hz={getattr(hw_config.systolic_array, 'sram_freq_hz', None)}"
                    )
        logic_freq = getattr(args, "logic_clk_freq", None)
        set_digital_clock_freq_hz(hw_config, logic_freq)  # 不传或≤0 时保持默认 1 GHz
        if getattr(args, "inter_layer_sram_size_bytes", None) is not None and getattr(hw_config, "inter_layer_buffer", None) is not None:
            hw_config.inter_layer_buffer.size_bytes = max(1, int(args.inter_layer_sram_size_bytes))
        if getattr(args, "inter_layer_sram_bw", None) is not None and getattr(hw_config, "inter_layer_buffer", None) is not None:
            hw_config.inter_layer_buffer.bandwidth_bytes_per_cycle = max(1, int(args.inter_layer_sram_bw))
        if getattr(args, "dram_bw_bytes_per_cycle", None) is not None and getattr(hw_config, "dram", None) is not None:
            hw_config.dram.bandwidth_bytes_per_cycle = max(1, int(args.dram_bw_bytes_per_cycle))
            if getattr(hw_config, "inter_layer_buffer", None) is not None:
                hw_config.inter_layer_buffer.dram_bandwidth_bytes_per_cycle = max(1, int(args.dram_bw_bytes_per_cycle))
        if getattr(args, "dram_channels", None) is not None and getattr(hw_config, "dram", None) is not None:
            hw_config.dram.dramsim3_channels_override = max(1, int(args.dram_channels))
        if getattr(hw_config, "inter_layer_buffer", None) is not None:
            if getattr(args, "feature_buffer_pipeline_mode", None):
                hw_config.inter_layer_buffer.pipeline_mode = str(args.feature_buffer_pipeline_mode)
            if getattr(args, "feature_double_buffering", False):
                hw_config.inter_layer_buffer.double_buffering = True
            if getattr(args, "feature_buffer_slots", None) is not None:
                hw_config.inter_layer_buffer.buffer_slots = max(1, int(args.feature_buffer_slots))

        set_logic_parallelism(
            hw_config,
            layernorm=getattr(args, "logic_par_layernorm", None),
            relu=getattr(args, "logic_par_relu", None),
            spool=getattr(args, "logic_par_spool", None),
            quant_dequant=getattr(args, "logic_par_quant_dequant", None),
        )
        layers = load_network("pointkan-elite", args.num_points, args.num_classes)

        # 打印加速器与层间 buffer/量化配置（与 main.run_network_simulation 一致，便于对照）
        at = getattr(hw_config, "accelerator_type", None)
        at_name = getattr(at, "name", str(at)) if at is not None else "N/A"
        log_info(f"加速器: {hw_config.accelerator_name} ({at_name})")
        log_info(f"CIM 并行度: {getattr(hw_config, 'cim_parallelism', 'N/A')}")
        log_info(f"Bit-serial 周期: {getattr(hw_config, 'bit_serial_cycles', 'N/A')}")
        buf = getattr(hw_config, "inter_layer_buffer", None)
        if buf is not None:
            log_info(
                f"层间 buffer: size={buf.size_bytes} B, bandwidth={buf.bandwidth_bytes_per_cycle} B/cycle, "
                f"spill_to_dram={getattr(buf, 'spill_to_dram', False)}, "
                f"pipeline_mode={getattr(buf, 'pipeline_mode', 'without_pipeline')}, "
                f"double_buffering={getattr(buf, 'double_buffering', False)}, "
                f"buffer_slots={getattr(buf, 'buffer_slots', 0)}"
            )
            if getattr(hw_config, "dram", None) is not None:
                log_info(f"DRAM: bandwidth={hw_config.dram.bandwidth_bytes_per_cycle} B/cycle, channels_override={getattr(hw_config.dram, 'dramsim3_channels_override', None)}")
        else:
            log_info("层间 buffer: 未配置（inter_layer_buffer 为空）")
        qd = getattr(hw_config, "enable_w8a8_quant_dequant", False)
        bn_fuse = getattr(hw_config, "bn_fuse_with_conv_linear", True)
        log_info(f"层间量化/反量化: {'启用 (FP32↔INT8 计入 cycle/功耗)' if qd else '未启用 (默认，假定 W8A8 已就绪)'}")
        log_info(f"BN 融合: {'启用 (BN 不单独计入)' if bn_fuse else '关闭 (保留 BN 事件)'}")
        if getattr(hw_config, "use_scalesim_for_systolic", False):
            log_info("SCALE-Sim 脉动 Conv/Linear: 启用（周期与 MAC/访问由 SCALE-Sim 统计，能耗仍用本工程内存建模）")
            if getattr(hw_config, "systolic_array", None) is not None:
                log_info(f"脉动阵列尺寸: {hw_config.systolic_array.rows}×{hw_config.systolic_array.cols} (ArrayHeight×ArrayWidth)")
        log_info(f"数字逻辑时钟: {hw_config.clock_period:.4f} ns ({1000.0 / hw_config.clock_period:.2f} MHz){' (由 --logic_clk_freq 设置)' if logic_freq is not None and logic_freq > 0 else ' (默认 1 GHz)'}")
        if any(
            getattr(args, k, None) is not None
            for k in ("logic_par_layernorm", "logic_par_relu", "logic_par_spool", "logic_par_quant_dequant")
        ):
            lc = getattr(hw_config, "logic_circuit", None)
            if lc:
                par_ln = getattr(lc.layernorm, "parallelism", 1) if lc.layernorm else 1
                par_relu = getattr(lc.relu, "parallelism", 1) if lc.relu else 1
                par_sp = getattr(lc.spool, "parallelism", 1) if lc.spool else 1
                par_q8 = getattr(lc.quant_pipeline_8bit, "parallelism", 1) if getattr(lc, "quant_pipeline_8bit", None) else 1
                par_dq = getattr(lc.dequant_pipeline, "parallelism", 1) if getattr(lc, "dequant_pipeline", None) else 1
                log_info(f"逻辑电路并行度: layernorm={par_ln} relu={par_relu} spool={par_sp} quant_8bit={par_q8} dequant={par_dq}")
        log_info("")

        # 先做一次真实前向，收集 conv/linear 顺序与真实 indata/rdata 字节数，用于 MNSIM 按层矩阵乘适配
        conv_linear_list = collect_conv_linear_from_forward(
            net, args.device, args.num_points, batch_size=min(2, args.batch_size)
        )
        real_data_overrides = real_data_overrides_from_conv_linear_list(conv_linear_list)

        if args.show_shapes:
            log_conv_linear_shapes(
                layers,
                conv_linear_list,
                real_data_overrides=real_data_overrides,
                log_info_fn=log_info,
            )
            nonlinear_list = collect_nonlinear_activations_from_forward(
                net, args.device, args.num_points, batch_size=min(2, args.batch_size)
            )
            log_nonlinear_activation_shapes(nonlinear_list, log_info_fn=log_info)

        # MNSIM / NeuroSim 全网络：仅 acc 模式下运行并输出；baseline/base_v2 不输出 MNSIM/NeuroSim 信息
        activation_shapes = compute_activation_shapes(layers, args.num_points, input_channels=3)
        acc_name = (args.accelerator or "").strip().lower()

        mnsim_context = None
        n_mnsim = 0
        netstruct = None
        conv_linear_order = None
        mnsim_result = None
        used_neurosim_for_ppa = False
        neurosim_result = None
        neurosim_root = getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch")

        if acc_name == "acc" and args.neurosim_full:
            from pointkan_sim.integration.neurosim_full import run_neurosim_full
            neurosim_result = run_neurosim_full(
                conv_linear_list,
                neurosim_root=neurosim_root,
                clk_freq_hz=getattr(args, "neurosim_clk_freq", None),
                technode=getattr(args, "neurosim_technode", None),
            )
            if neurosim_result.success:
                used_neurosim_for_ppa = True
                if getattr(args, "neurosim_clk_freq", None) is not None or getattr(args, "neurosim_technode", None) is not None:
                    log_info(f"NeuroSim 频率/工艺: clk_freq_hz={getattr(args, 'neurosim_clk_freq', None)}, technode={getattr(args, 'neurosim_technode', None)}")
                neurosim_clock_ns = getattr(neurosim_result, "clock_period_ns", None) or 0.0
                mnsim_context = {
                    "mnsim_latency_ns": neurosim_result.layer_latency_ns,
                    "mnsim_energy_nj": neurosim_result.layer_energy_nj,
                    "mnsim_index": 0,
                }
                if neurosim_clock_ns > 0:
                    mnsim_context["mnsim_clock_period_ns"] = float(neurosim_clock_ns)
                if getattr(neurosim_result, "layer_area_um2", None) is not None:
                    mnsim_context["mnsim_area_um2"] = neurosim_result.layer_area_um2
                if conv_linear_list:
                    mnsim_context["mnsim_layer_names"] = [c.name for c in conv_linear_list[: len(neurosim_result.layer_latency_ns)]]
                n_mnsim = len(neurosim_result.layer_latency_ns)
                log_info("已用 NeuroSim 全网络仿真（DNN_NeuroSim main）得到逐层 latency/energy")
                # 数字逻辑时钟由 hw_config.clock_period（可 --logic_clk_freq 设置，默认 1 GHz）；Conv/Linear 单独用 NeuroSim 时钟
                if neurosim_clock_ns > 0:
                    freq_mhz = 1000.0 / neurosim_clock_ns
                    log_info(f"NeuroSim Conv/Linear 时钟: {neurosim_clock_ns:.4f} ns ({freq_mhz:.2f} MHz)，由 NEUROSIM_CLK_FREQ 传入（可变，不限于 100M）；数字逻辑时钟见上方")
                else:
                    log_info("未解析到 NeuroSim 时钟周期，Conv/Linear 亦使用数字逻辑时钟 (hw_config.clock_period)")
                log_info("以下层使用 NeuroSim 接口计算 PPA：")
                clock_ns_log = mnsim_context.get("mnsim_clock_period_ns") or hw_config.clock_period
                from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
                for idx, (lat_ns, en_nj) in enumerate(zip(neurosim_result.layer_latency_ns, neurosim_result.layer_energy_nj)):
                    name = conv_linear_list[idx].name if idx < len(conv_linear_list) else f"layer_{idx}"
                    cyc = max(1, int(math.ceil(lat_ns / clock_ns_log)))
                    act_s, w_s = format_activation_weight_shapes(conv_linear_list[idx]) if idx < len(conv_linear_list) else ("?", "?")
                    log_info(f"  NeuroSim [{name}] activation={act_s}, weight={w_s}  latency={lat_ns:.1f} ns, cycles={cyc}, energy={en_nj:.2f} nJ")
                log_info(f"  （共 {n_mnsim} 层）")
                # NeuroSim 模式：输出消耗的 tile/PE 与对应 size（占位或解析结果）
                if getattr(neurosim_result, "rram_tile_stats", None):
                    st = neurosim_result.rram_tile_stats
                    log_info("NeuroSim Tile/PE 统计:")
                    log_info(f"  总 Tile 数: {st.get('total_tile_count', 0)},  总 PE 数: {st.get('total_PE_count', 0)}")
                    xbar = st.get("xbar_size") or {}
                    log_info(f"  Crossbar size: {xbar.get('row', '?')} x {xbar.get('column', '?')}  (pe_per_tile={st.get('pe_per_tile', '?')})")
                    if st.get("per_layer"):
                        log_info(f"  逐层: 共 {len(st['per_layer'])} 层")
                        for pl in st["per_layer"][:8]:
                            log_info(f"    layer_{pl.get('layer_index', '?')} ({pl.get('layer_type', '')}): tilenum={pl.get('tilenum', 0)}, PE_count={pl.get('PE_count', 0)}, max_row={pl.get('max_row', 0)}, max_column={pl.get('max_column', 0)}")
                        if len(st["per_layer"]) > 8:
                            log_info(f"    ... 其余 {len(st['per_layer']) - 8} 层")
            else:
                log_info(f"NeuroSim 全网络未启用: {neurosim_result.error}")
                if neurosim_result.error and len(neurosim_result.error) > 200:
                    for line in neurosim_result.error.split("\n")[:15]:
                        if line.strip():
                            log_info(f"  NeuroSim stderr: {line.strip()}")
                log_info("  提示: 若为主进程崩溃(returncode≠0)，可检查 DNN_NeuroSim 是否支持当前层数/维度；层数较多时可尝试减少 batch 或先单独运行 NeuroSIM/main 排查。")

        if acc_name == "acc" and mnsim_context is None and args.mnsim_full:
            from pointkan_sim.integration.mnsim_netstruct import build_netstruct_from_layers, run_mnsim_full
            netstruct, conv_linear_order = build_netstruct_from_layers(
                layers, real_data_overrides=real_data_overrides
            )
            if netstruct:
                mnsim_result = run_mnsim_full(netstruct, conv_linear_order=conv_linear_order)
                if mnsim_result.success:
                    mnsim_context = {
                        "mnsim_latency_ns": mnsim_result.layer_latency_ns,
                        "mnsim_energy_nj": mnsim_result.layer_energy_nj,
                        "mnsim_index": 0,
                    }
                    if getattr(mnsim_result, "layer_area_um2", None):
                        mnsim_context["mnsim_area_um2"] = mnsim_result.layer_area_um2
                    if conv_linear_order:
                        mnsim_context["mnsim_layer_names"] = [conv_linear_order[i][0] for i in range(len(conv_linear_order))]
                    n_mnsim = len(mnsim_result.layer_latency_ns)
                    log_info("已用与 checkpoint 一致的层结构 + 真实按层 indata/rdata 驱动 MNSIM 全 NetStruct（Tile/NoC）")
                    clock_ns = hw_config.clock_period
                    from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
                    for idx, (lat_ns, en_nj) in enumerate(zip(mnsim_result.layer_latency_ns, mnsim_result.layer_energy_nj)):
                        name = conv_linear_order[idx][0] if idx < len(conv_linear_order) else f"layer_{idx}"
                        cyc = max(1, int(math.ceil(lat_ns / clock_ns)))
                        act_s, w_s = format_activation_weight_shapes(conv_linear_list[idx]) if idx < len(conv_linear_list) else ("?", "?")
                        log_info(f"  MNSIM [{name}] activation={act_s}, weight={w_s}  latency={lat_ns:.1f} ns, cycles={cyc}, energy={en_nj:.2f} nJ")
                    # MNSIM 模式：输出 Tile/PE 与面积
                    if getattr(mnsim_result, "rram_tile_stats", None):
                        st = mnsim_result.rram_tile_stats
                        log_info("MNSIM Tile/PE 统计:")
                        log_info(f"  总 Tile 数: {st.get('total_tile_count', 0)},  总 PE 数: {st.get('total_PE_count', 0)}")
                        xbar = st.get("xbar_size") or {}
                        log_info(f"  Crossbar size: {xbar.get('row', '?')} x {xbar.get('column', '?')}  (pe_per_tile={st.get('pe_per_tile', '?')})")
                    if getattr(mnsim_result, "total_area_um2", None) and mnsim_result.total_area_um2 > 0:
                        log_info(f"  MNSIM Conv/Linear 总面积: {mnsim_result.total_area_um2:.2f} um^2")
                else:
                    log_info(f"MNSIM 未启用: {mnsim_result.error}")

        if acc_name == "acc" and mnsim_context is not None:
            n_mnsim = len(mnsim_context["mnsim_latency_ns"])
            log_info(f"真实前向共 {len(conv_linear_list)} 个 conv/linear；全网络 PPA 使用 {n_mnsim} 层")
            if len(conv_linear_list) != n_mnsim:
                log_info(f"  注意：真实前向层数({len(conv_linear_list)})与 LayerDef 层数({n_mnsim})不一致；overrides 按索引取前 {n_mnsim} 条对应到 MNSIM。建议使用 --show_shapes 核对。")
        elif acc_name != "acc":
            log_info(f"真实前向共 {len(conv_linear_list)} 个 conv/linear（baseline 脉动由 SCALE-Sim 逐层统计）")

        # 按 ModelNet40 批次调度：每批一次真实前向（准确率）+ 一次 PPA 仿真（累加）
        eval_result = run_real_inference_and_sim_over_dataset(
            net,
            test_loader,
            criterion,
            args.device,
            layers,
            hw_config,
            mnsim_context,
            enable_layer_buffer=True,
            activation_shapes=activation_shapes,
            max_batches=args.max_batches,
            repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

        if not eval_result.success:
            log_info(f"真实数据评估失败: {eval_result.error}")
            return 1

        log_separator()
        if used_neurosim_for_ppa:
            log_info("真实数据评估结果 (ModelNet40 测试集 + 对应层 NeuroSim PPA)")
        elif mnsim_context and args.mnsim_full:
            log_info("真实数据评估结果 (ModelNet40 测试集 + 对应层 MNSIM PPA)")
        else:
            log_info("真实数据评估结果 (ModelNet40 测试集)")
        log_separator()
        log_info(f"  样本数:        {eval_result.num_samples}")
        log_info(f"  准确率:        {eval_result.accuracy:.2f}%")
        log_info(f"  平衡准确率:    {eval_result.balanced_accuracy:.2f}%")
        log_info(f"  Test Loss:     {eval_result.test_loss:.4f}")
        # 端到端延迟：各事件按自身 clock_period_ns 换算后累加（Conv/Linear 用 NeuroSim 时钟，数字逻辑用 1 ns）
        latency_per_sample_ns = getattr(eval_result, "latency_per_sample_ns", None)
        if latency_per_sample_ns is None or latency_per_sample_ns <= 0:
            # 回退：若结果未带 latency_ns 则用单时钟近似（不准确）
            clock_fallback = (mnsim_context.get("mnsim_clock_period_ns") if mnsim_context else None) or hw_config.clock_period
            latency_per_sample_ns = eval_result.cycles_per_sample * clock_fallback
        log_info(f"  总周期:        {eval_result.total_cycles:,} cycles")
        log_info(f"  总能耗:        {eval_result.total_energy_nj:.2f} nJ")
        log_info(f"  端到端延迟:    {latency_per_sample_ns/1e6:.4f} ms (每样本, 各模块按自身时钟换算 ns 后累加)")
        log_info(f"  每样本周期:    {eval_result.cycles_per_sample:.1f} cycles")
        log_info(f"  每样本能耗:    {eval_result.energy_per_sample_nj:.2f} nJ")
        log_separator()

        # 单次仿真获取全算子 breakdown（FPS/KNN/BN/Conv/Linear 等），用于饼图与报告
        from pointkan_sim.main import simulate_network
        mnsim_ctx = dict(mnsim_context) if mnsim_context else None
        if mnsim_ctx is not None:
            mnsim_ctx["mnsim_index"] = 0
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sim_once = simulate_network(
            layers,
            hw_config,
            show_layers=False,
            mnsim_context=mnsim_ctx,
            activation_shapes=activation_shapes,
            enable_layer_buffer=True,
            show_shapes=False,
            repo_root=_repo_root,
        )
        full_operator_breakdown = sim_once.get("operator_breakdown") or {}
        operator_cycle_detail = sim_once.get("operator_cycle_detail") or []
        operator_sequence = sim_once.get("operator_sequence") or []
        total_area_um2 = float(sim_once.get("total_area_um2", 0) or 0)
        clock_ns = getattr(hw_config, "clock_period", 1.0)  # 数字逻辑 1 GHz；仅用于 fallback 与报告

        # 控制台打印：端到端延迟、能耗与面积构成（按算子类型）；Quant/Dequant/BN 单独列出
        if full_operator_breakdown:
            total_cy = sum(v.get("cycles", 0) for v in full_operator_breakdown.values())
            total_en = sum(v.get("energy_nj", 0) for v in full_operator_breakdown.values())
            total_area = sum(v.get("area_um2", 0) for v in full_operator_breakdown.values())
            if total_area <= 0:
                total_area = total_area_um2
            latency_total_ns = sum(v.get("latency_ns", 0) for v in full_operator_breakdown.values())
            if latency_total_ns <= 0:
                latency_total_ns = total_cy * (hw_config.clock_period or 1.0)
            log_separator()
            log_info("端到端延迟、能耗与面积构成 (每样本，按算子类型；面积来自 NeuroSim/MNSIM 或逻辑模块)")
            log_info("  说明: Conv/Linear 用 NeuroSim 时钟（NEUROSIM_CLK_FREQ 传入，可变）；数字逻辑用 --logic_clk_freq/LOGIC_CLK_FREQ 接口，默认 1 GHz。")
            log_info("  AffineNorm=GAM 仿射归一化（不含 BN）；BN 在 BN_EN=1 时融合进 Conv/Linear 不单独出现，BN_EN=0 时单独列出为 BN；QD_EN=1 时 Quant/Dequant 已单独列出。")
            log_info("  查看入口: AffineNorm=pointkan_sim.operators.common_ops.gen_affine_norm_events；BN=gen_bn_events；BN 融合=dispatcher；Quant/Dequant=common_ops.gen_quant_events/gen_dequant_events。")
            log_info("  内存: SRAM=层间 SRAM 读写（原 LayerBuffer）；DRAM=首层点云加载与 spill。")
            log_info("  能耗: 除 SRAM CIM Array 用用户数据外，LayerBuffer 仅来自 CACTI 仿真（fill_inter_layer_buffer_from_cacti）；首层与 spill 的 DRAM 仅来自 trace→DRAMSim3（repo_root + hw_config.dram.dramsim3_bin 时自动跑）。无回退。详见 ARCHITECTURE.md §1.4。")
            log_separator()
            log_info(f"  {'算子':<20} {'cycles':>12} {'cycles%':>8} {'latency(ns)':>14} {'latency%':>8} {'energy(nJ)':>12} {'energy%':>8} {'area(um2)':>12} {'area%':>8}")
            log_info("  " + "-" * 110)
            for op_name, v in sorted(full_operator_breakdown.items(), key=lambda x: -x[1].get("latency_ns", x[1].get("cycles", 0))):
                display_name = "SRAM" if op_name == "LayerBuffer" else op_name
                cy = v.get("cycles", 0)
                en = v.get("energy_nj", 0)
                area_u = v.get("area_um2", 0)
                lat_ns = v.get("latency_ns", cy * clock_ns if clock_ns > 0 else 0)
                cy_pct = (cy / total_cy * 100) if total_cy > 0 else 0
                lat_pct = (lat_ns / latency_total_ns * 100) if latency_total_ns > 0 else 0
                en_pct = (en / total_en * 100) if total_en > 0 else 0
                area_pct = (area_u / total_area * 100) if total_area > 0 else 0
                log_info(f"  {display_name:<20} {cy:>12,.0f} {cy_pct:>7.1f}% {lat_ns:>14,.1f} {lat_pct:>7.1f}% {en:>12,.2f} {en_pct:>7.1f}% {area_u:>12,.2f} {area_pct:>7.1f}%")
            log_info("  " + "-" * 110)
            log_info(f"  {'合计':<20} {total_cy:>12,.0f} {100.0:>7.1f}% {latency_total_ns:>14,.1f} {100.0:>7.1f}% {total_en:>12,.2f} {100.0:>7.1f}% {total_area:>12,.2f} {100.0:>7.1f}%")
            log_info(f"  端到端延迟合计: {latency_total_ns/1e6:.4f} ms  |  能耗合计: {total_en:.2f} nJ  |  系统总面积: {total_area:.2f} um^2")
            log_separator()

        # 每个算子 cycle 的详细构成：按 (算子类型, 层名) 聚合并打印
        if operator_cycle_detail:
            from collections import defaultdict
            by_op_layer: dict = defaultdict(lambda: {"cycles": 0, "latency_ns": 0.0, "energy_nj": 0.0, "area_um2": 0.0})
            for d in operator_cycle_detail:
                op = d.get("operator", "other")
                layer = d.get("layer_name", "")
                key = (op, layer)
                by_op_layer[key]["cycles"] += d.get("duration", 0)
                by_op_layer[key]["latency_ns"] += d.get("latency_ns", 0.0)
                by_op_layer[key]["energy_nj"] += d.get("energy_nj", 0.0)
                by_op_layer[key]["area_um2"] += d.get("area_um2", 0.0)
            log_separator()
            log_info("每个算子 cycle 的详细构成（按算子类型 + 层名）")
            log_info("  operator_cycle_detail 按 (算子, 层名) 聚合；以下按算子类型分组，组内按 cycles 降序。")
            log_separator()
            # 按算子类型分组
            by_op: dict = defaultdict(list)
            for (op, layer), v in by_op_layer.items():
                by_op[op].append((layer, v["cycles"], v["latency_ns"], v["energy_nj"], v["area_um2"]))
            for op_name in sorted(by_op.keys(), key=lambda o: (-sum(x[1] for x in by_op[o]), o)):
                rows = by_op[op_name]
                rows.sort(key=lambda x: -x[1])
                total_op_cy = sum(r[1] for r in rows)
                total_op_lat = sum(r[2] for r in rows)
                total_op_en = sum(r[3] for r in rows)
                total_op_area = sum(r[4] for r in rows)
                log_info(f"  [{op_name}] 合计: cycles={total_op_cy:,.0f}, latency_ns={total_op_lat:,.1f}, energy_nJ={total_op_en:.2f}, area_um2={total_op_area:,.2f}")
                for layer_name, cy, lat_ns, en, area_u in rows:
                    log_info(f"    layer={layer_name!r}  cycles={cy:,.0f}  latency_ns={lat_ns:,.1f}  energy_nJ={en:.2f}  area_um2={area_u:,.2f}")
            log_separator()

            # Conv1d 与 Linear 面积明细（逐层 area_um2，便于分析面积过大的来源）
            for op_name in ("Conv1d", "Linear"):
                if op_name not in by_op:
                    continue
                rows = by_op[op_name]
                total_op_area = sum(r[4] for r in rows)
                if total_op_area <= 0:
                    continue
                rows_with_area = [(r[0], r[4]) for r in rows if r[4] > 0]
                if not rows_with_area:
                    continue
                rows_with_area.sort(key=lambda x: -x[1])
                log_info(f"Conv1d 与 Linear 面积明细 — [{op_name}] 合计 area={total_op_area:,.2f} um^2（占系统总面积比例见上表）")
                log_info(f"  {'层名':<50} {'area_um2':>16} {'占该类型%':>10}")
                log_info("  " + "-" * 78)
                for layer_name, area_u in rows_with_area:
                    pct = (area_u / total_op_area * 100) if total_op_area > 0 else 0
                    name_show = (layer_name[:47] + "...") if len(layer_name) > 50 else layer_name or "(未命名)"
                    log_info(f"  {name_show:<50} {area_u:>16,.2f} {pct:>9.1f}%")
                log_info("  " + "-" * 78)
                log_separator()

        # 端到端 PPA 报告与算子级 breakdown 可视化：report.json + latency/energy 饼图（由 write_ppa_report_and_plots 统一写入 sim_results）
        from pointkan_sim.integration.ppa_report import build_ppa_report, write_ppa_report_and_plots
        mnsim_lat = mnsim_context["mnsim_latency_ns"] if mnsim_context else None
        mnsim_en = mnsim_context["mnsim_energy_nj"] if mnsim_context else None
        # Tile/PE 统计：NeuroSim 与 MNSIM 均可能提供
        rram_tile_stats = None
        if used_neurosim_for_ppa and neurosim_result:
            rram_tile_stats = getattr(neurosim_result, "rram_tile_stats", None)
        if rram_tile_stats is None and mnsim_result:
            rram_tile_stats = getattr(mnsim_result, "rram_tile_stats", None)
        ppa_report = build_ppa_report(
            eval_result,
            clock_ns,
            netstruct=netstruct,
            conv_linear_order=conv_linear_order,
            mnsim_latency_ns=mnsim_lat,
            mnsim_energy_nj=mnsim_en,
            full_operator_breakdown=full_operator_breakdown,
            checkpoint=checkpoint_path,
            mnsim_full=args.mnsim_full,
            rram_tile_stats=rram_tile_stats,
            operator_sequence=operator_sequence,
            latency_per_sample_ns=getattr(eval_result, "latency_per_sample_ns", None),
            total_area_um2=total_area_um2,
        )
        use_auto_dir = args.sim_results_dir == DEFAULT_SIM_RESULTS_DIR
        effective_sim_results_dir = get_sim_mode_results_dir(args) if use_auto_dir else args.sim_results_dir
        sim_results_dir = Path(effective_sim_results_dir)
        sim_results_dir.mkdir(parents=True, exist_ok=True)
        if use_auto_dir:
            log_info(f"PPA 与 breakdown 输出目录: {sim_results_dir}（按仿真模式自动命名）")
        # NeuroSim 模式：将参与计算的层名及 latency/energy/激活与权重 shape 写入 CSV
        if used_neurosim_for_ppa and mnsim_context and conv_linear_list:
            import csv
            from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
            lat_ns_list = mnsim_context["mnsim_latency_ns"]
            en_nj_list = mnsim_context["mnsim_energy_nj"]
            neurosim_csv_path = sim_results_dir / "neurosim_layer_ppa.csv"
            with open(neurosim_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["layer_name", "layer_type", "latency_ns", "energy_nj", "input_activation_shape", "weight_shape"])
                for idx in range(min(len(lat_ns_list), len(en_nj_list), len(conv_linear_list))):
                    info = conv_linear_list[idx]
                    act_s, w_s = format_activation_weight_shapes(info)
                    w.writerow([
                        info.name,
                        info.layer_type,
                        f"{lat_ns_list[idx]:.4f}",
                        f"{en_nj_list[idx]:.4f}",
                        act_s,
                        w_s,
                    ])
            log_info(f"NeuroSim 逐层 PPA 与 shape 已写入: {neurosim_csv_path}（共 {min(len(lat_ns_list), len(conv_linear_list))} 层）")
        report_filename = os.path.basename(args.output) if args.output else "report.json"
        write_ppa_report_and_plots(
            ppa_report,
            sim_results_dir,
            report_filename=report_filename,
            log_info_fn=log_info,
        )
        # 按执行顺序输出完整硬件算子序列到 operator_sequence.txt（列与 constants 键名一致）
        if operator_sequence:
            import csv as _csv
            seq_path = sim_results_dir / "operator_sequence.txt"
            with open(seq_path, "w", encoding="utf-8") as f:
                f.write("# 硬件算子执行顺序（按 cycle 排序）\n")
                f.write(OPERATOR_SEQUENCE_HEADER)
                for item in operator_sequence:
                    f.write(
                        "{idx}\t{cycle}\t{end_cycle}\t{duration}\t{operator}\t{stage}\t{operation}\t{layer_name}\t{energy_nj}\n".format(
                            idx=item.get(KEY_IDX, ""),
                            cycle=item.get(KEY_CYCLE, ""),
                            end_cycle=item.get(KEY_END_CYCLE, ""),
                            duration=item.get(KEY_DURATION, ""),
                            operator=item.get(KEY_OPERATOR, ""),
                            stage=item.get(KEY_STAGE, ""),
                            operation=item.get(KEY_OPERATION, ""),
                            layer_name=item.get(KEY_LAYER_NAME, ""),
                            energy_nj=item.get(KEY_ENERGY_NJ, ""),
                        )
                    )
            seq_csv_path = sim_results_dir / "operator_sequence.csv"
            with open(seq_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["idx", "cycle", "end_cycle", "duration", "operator", "stage", "operation", "layer_name", "energy_nj"])
                for item in operator_sequence:
                    w.writerow([
                        item.get(KEY_IDX, ""),
                        item.get(KEY_CYCLE, ""),
                        item.get(KEY_END_CYCLE, ""),
                        item.get(KEY_DURATION, ""),
                        item.get(KEY_OPERATOR, ""),
                        item.get(KEY_STAGE, ""),
                        item.get(KEY_OPERATION, ""),
                        item.get(KEY_LAYER_NAME, ""),
                        item.get(KEY_ENERGY_NJ, ""),
                    ])
            log_info(f"完整硬件算子序列已写入: {seq_path}（共 {len(operator_sequence)} 条）")
            log_info(f"完整硬件算子序列CSV已写入: {seq_csv_path}")

        # 写入每个算子 cycle 的详细构成（按算子+层名聚合）
        if operator_cycle_detail:
            import csv as _csv
            from collections import defaultdict
            by_op_layer_file = defaultdict(lambda: {"cycles": 0, "latency_ns": 0.0, "energy_nj": 0.0, "activation_input_shape": "", "activation_output_shape": ""})
            for d in operator_cycle_detail:
                op = d.get("operator", "other")
                layer = d.get("layer_name", "")
                key = (op, layer)
                by_op_layer_file[key]["cycles"] += d.get("duration", 0)
                by_op_layer_file[key]["latency_ns"] += d.get("latency_ns", 0.0)
                by_op_layer_file[key]["energy_nj"] += d.get("energy_nj", 0.0)
                if not by_op_layer_file[key]["activation_input_shape"] and d.get("activation_input_shape"):
                    by_op_layer_file[key]["activation_input_shape"] = str(d.get("activation_input_shape", ""))
                if not by_op_layer_file[key]["activation_output_shape"] and d.get("activation_output_shape"):
                    by_op_layer_file[key]["activation_output_shape"] = str(d.get("activation_output_shape", ""))
            by_op_file = defaultdict(list)
            for (op, layer), v in by_op_layer_file.items():
                by_op_file[op].append((layer, v["cycles"], v["latency_ns"], v["energy_nj"], v["activation_input_shape"], v["activation_output_shape"]))
            detail_path = sim_results_dir / "operator_cycle_detail.txt"
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write("# 每个算子 cycle 的详细构成（按算子类型 + 层名聚合）\n")
                f.write("# operator\tlayer_name\tcycles\tlatency_ns\tenergy_nJ\tactivation_input_shape\tactivation_output_shape\n")
                for op_name in sorted(by_op_file.keys(), key=lambda o: (-sum(x[1] for x in by_op_file[o]), o)):
                    for layer_name, cy, lat_ns, en, act_in, act_out in sorted(by_op_file[op_name], key=lambda x: -x[1]):
                        f.write(f"{op_name}\t{layer_name}\t{cy:.0f}\t{lat_ns:.2f}\t{en:.2f}\t{act_in}\t{act_out}\n")
            detail_csv_path = sim_results_dir / "operator_cycle_detail.csv"
            with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["operator", "layer_name", "cycles", "latency_ns", "energy_nj", "activation_input_shape", "activation_output_shape"])
                for op_name in sorted(by_op_file.keys(), key=lambda o: (-sum(x[1] for x in by_op_file[o]), o)):
                    for layer_name, cy, lat_ns, en, act_in, act_out in sorted(by_op_file[op_name], key=lambda x: -x[1]):
                        w.writerow([op_name, layer_name, f"{cy:.0f}", f"{lat_ns:.2f}", f"{en:.2f}", act_in, act_out])
            log_info(f"算子 cycle 构成明细已写入: {detail_path}")
            log_info(f"算子 cycle 构成明细CSV已写入: {detail_csv_path}")

            # KAN/KAT 逐条激活 shape 与 cycle/energy（每行一个 KAT 块，便于核对）
            # 说明: 仅统计有效 KAT(act2)；Efficient_KAN 内 act1(Identity) 不占 cycle/energy，未计入
            kan_rows = [d for d in operator_cycle_detail if (d.get("operator") or "").strip() == "KAN"]
            if kan_rows:
                kan_csv_path = sim_results_dir / "kan_kat_layers.csv"
                import csv as _csv
                with open(kan_csv_path, "w", newline="", encoding="utf-8") as f:
                    f.write("# KAN = 有效 KAT(act2)；Efficient_KAN 内 act1(Identity) 不占 cycle/energy，未计入。\n")
                    w = _csv.writer(f)
                    w.writerow([
                        "layer_name", "num_points", "in_features", "out_features",
                        "activation_input_shape", "activation_output_shape",
                        "duration_cycles", "latency_ns", "energy_nj",
                    ])
                    for d in kan_rows:
                        w.writerow([
                            d.get("layer_name", ""),
                            d.get("num_points", ""),
                            d.get("in_features", ""),
                            d.get("out_features", ""),
                            d.get("activation_input_shape", ""),
                            d.get("activation_output_shape", ""),
                            d.get("duration", ""),
                            f"{d.get('latency_ns', 0):.2f}",
                            f"{d.get('energy_nj', 0):.4f}",
                        ])
                log_info(f"KAN/KAT 逐条激活 shape 与 cycle/energy 已写入: {kan_csv_path}（共 {len(kan_rows)} 条，仅有效 KAT/act2；Identity act1 未计入）")

        # SCALE-Sim 逐层 Conv/Linear：周期、利用率、MACs、能耗（仅 baseline+scalesim 时有）
        feature_residency_plan = sim_once.get("feature_residency_plan") or []
        if feature_residency_plan:
            import csv as _csv
            residency_txt_path = sim_results_dir / "feature_residency_plan.txt"
            with open(residency_txt_path, "w", encoding="utf-8") as f:
                f.write("# Layer-by-layer persistent feature SRAM residency plan\n")
                f.write("# layer_name\tlayer_type\tinput_bytes\tinput_sram_bytes\tinput_dram_bytes\toutput_bytes\toutput_sram_bytes\toutput_dram_bytes\tusable_buffer_bytes\ttotal_buffer_bytes\tbuffer_slots\tpipeline_mode\tdouble_buffering\tnote\n")
                for row in feature_residency_plan:
                    f.write(
                        "{layer_name}\t{layer_type}\t{input_bytes}\t{input_sram_bytes}\t{input_dram_bytes}\t{output_bytes}\t{output_sram_bytes}\t{output_dram_bytes}\t{usable_buffer_bytes}\t{total_buffer_bytes}\t{buffer_slots}\t{pipeline_mode}\t{double_buffering}\t{note}\n".format(
                            layer_name=row.get("layer_name", ""),
                            layer_type=row.get("layer_type", ""),
                            input_bytes=row.get("input_bytes", 0),
                            input_sram_bytes=row.get("input_sram_bytes", 0),
                            input_dram_bytes=row.get("input_dram_bytes", 0),
                            output_bytes=row.get("output_bytes", 0),
                            output_sram_bytes=row.get("output_sram_bytes", 0),
                            output_dram_bytes=row.get("output_dram_bytes", 0),
                            usable_buffer_bytes=row.get("usable_buffer_bytes", 0),
                            total_buffer_bytes=row.get("total_buffer_bytes", 0),
                            buffer_slots=row.get("buffer_slots", 0),
                            pipeline_mode=row.get("pipeline_mode", ""),
                            double_buffering=row.get("double_buffering", False),
                            note=row.get("note", ""),
                        )
                    )
            residency_csv_path = sim_results_dir / "feature_residency_plan.csv"
            with open(residency_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["layer_name", "layer_type", "input_bytes", "input_sram_bytes", "input_dram_bytes", "output_bytes", "output_sram_bytes", "output_dram_bytes", "usable_buffer_bytes", "total_buffer_bytes", "buffer_slots", "pipeline_mode", "double_buffering", "note"])
                for row in feature_residency_plan:
                    w.writerow([row.get("layer_name", ""), row.get("layer_type", ""), row.get("input_bytes", 0), row.get("input_sram_bytes", 0), row.get("input_dram_bytes", 0), row.get("output_bytes", 0), row.get("output_sram_bytes", 0), row.get("output_dram_bytes", 0), row.get("usable_buffer_bytes", 0), row.get("total_buffer_bytes", 0), row.get("buffer_slots", 0), row.get("pipeline_mode", ""), row.get("double_buffering", False), row.get("note", "")])
            log_info(f"Feature SRAM residency plan 已写入: {residency_txt_path}")
            log_info(f"Feature SRAM residency plan CSV已写入: {residency_csv_path}")

        scalesim_layer_details = sim_once.get("scalesim_layer_details") or []
        if scalesim_layer_details:
            import csv as _csv
            scalesim_csv_path = sim_results_dir / "scalesim_conv_linear_layers.csv"
            with open(scalesim_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow([
                    "layer_name", "operator", "total_cycles", "stall_cycles",
                    "overall_util_pct", "mapping_eff_pct", "compute_util_pct",
                    "macs", "energy_nj", "mac_energy_nj", "sram_energy_nj", "dram_energy_nj",
                    "sram_cycles", "dram_cycles", "sram_clock_period_ns", "dram_clock_period_ns",
                    "feature_input_dram_bytes", "feature_output_dram_bytes", "weight_dram_bytes", "weight_tensor_bytes", "weight_dram_trace_bytes", "feature_local_sram_trace_bytes", "weight_sram_trace_bytes", "dram_trace_dir",
                    "feature_input_dram_trace_dir", "weight_dram_trace_dir", "feature_output_dram_trace_dir",
                    "weight_refill_tiles", "weight_refill_tile_bytes",
                    "weight_sram_energy_nj", "weight_sram_cycles", "feature_local_sram_energy_nj", "feature_local_sram_cycles",
                    "systolic_array_rows", "systolic_array_cols",
                    "activation_input_shape", "activation_output_shape",
                    "num_points", "in_channels", "out_channels", "kernel_size",
                    "tokens", "in_features", "out_features",
                ])
                for row in scalesim_layer_details:
                    w.writerow([
                        row.get("layer_name", ""),
                        row.get("operator", ""),
                        row.get("total_cycles", ""),
                        row.get("stall_cycles", ""),
                        row.get("overall_util_pct") if row.get("overall_util_pct") is not None else "",
                        row.get("mapping_eff_pct") if row.get("mapping_eff_pct") is not None else "",
                        row.get("compute_util_pct") if row.get("compute_util_pct") is not None else "",
                        row.get("macs", ""),
                        f"{row.get('energy_nj', 0):.4f}",
                        f"{row.get('mac_energy_nj', 0):.4f}",
                        f"{row.get('sram_energy_nj', 0):.4f}",
                        f"{row.get('dram_energy_nj', 0):.4f}",
                        row.get("sram_cycles", ""),
                        row.get("dram_cycles", ""),
                        row.get("sram_clock_period_ns", ""),
                        row.get("dram_clock_period_ns", ""),
                        row.get("feature_input_dram_bytes", ""),
                        row.get("feature_output_dram_bytes", ""),
                        row.get("weight_dram_bytes", ""),
                        row.get("weight_tensor_bytes", ""),
                        row.get("weight_dram_trace_bytes", ""),
                        row.get("feature_local_sram_trace_bytes", ""),
                        row.get("weight_sram_trace_bytes", ""),
                        row.get("dram_trace_dir", ""),
                        row.get("feature_input_dram_trace_dir", ""),
                        row.get("weight_dram_trace_dir", ""),
                        row.get("feature_output_dram_trace_dir", ""),
                        row.get("weight_refill_tiles", ""),
                        row.get("weight_refill_tile_bytes", ""),
                        row.get("weight_sram_energy_nj", ""),
                        row.get("weight_sram_cycles", ""),
                        row.get("feature_local_sram_energy_nj", ""),
                        row.get("feature_local_sram_cycles", ""),
                        row.get("systolic_array_rows", ""),
                        row.get("systolic_array_cols", ""),
                        row.get("activation_input_shape", ""),
                        row.get("activation_output_shape", ""),
                        row.get("num_points", ""),
                        row.get("in_channels", ""),
                        row.get("out_channels", ""),
                        row.get("kernel_size", ""),
                        row.get("tokens", ""),
                        row.get("in_features", ""),
                        row.get("out_features", ""),
                    ])
            log_info(f"SCALE-Sim 逐层 Conv/Linear（周期、利用率、能耗、每层激活 shape）已写入: {scalesim_csv_path}（共 {len(scalesim_layer_details)} 层）")

        # 可选：导出内存交互事务（统一接口 src/dst/kind），并写聚合统计
        if args.export_memory_transactions and operator_sequence:
            import csv as _csv
            mem_txn = extract_memory_transactions(operator_sequence)
            mem_path = sim_results_dir / "memory_transactions.txt"
            with open(mem_path, "w", encoding="utf-8") as f:
                f.write("# 内存交互事务序列（统一接口，按 cycle 排序）\n")
                f.write("# idx\tcycle\tend_cycle\tduration\tkind\tsrc\tdst\toperator\toperation\tcircuit_module\tlayer_name\tenergy_nj\n")
                for tx in mem_txn:
                    row = tx.to_dict()
                    f.write(
                        "{idx}\t{cycle}\t{end_cycle}\t{duration}\t{kind}\t{src}\t{dst}\t{operator}\t{operation}\t{circuit_module}\t{layer_name}\t{energy_nj}\n".format(
                            **row
                        )
                    )
            mem_csv_path = sim_results_dir / "memory_transactions.csv"
            with open(mem_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["idx", "cycle", "end_cycle", "duration", "kind", "src", "dst", "operator", "operation", "circuit_module", "layer_name", "energy_nj"])
                for tx in mem_txn:
                    row = tx.to_dict()
                    w.writerow([row.get("idx", ""), row.get("cycle", ""), row.get("end_cycle", ""), row.get("duration", ""), row.get("kind", ""), row.get("src", ""), row.get("dst", ""), row.get("operator", ""), row.get("operation", ""), row.get("circuit_module", ""), row.get("layer_name", ""), row.get("energy_nj", "")])
            route_counter = summarize_memory_transactions(mem_txn)
            summary_path = sim_results_dir / "memory_transactions_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("# 内存事务聚合统计（src->dst, kind）\n")
                f.write("# src\tdst\tkind\tcount\n")
                for (src, dst, kind), cnt in sorted(route_counter.items(), key=lambda x: (-x[1], x[0])):
                    f.write(f"{src}\t{dst}\t{kind}\t{cnt}\n")
            summary_csv_path = sim_results_dir / "memory_transactions_summary.csv"
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["src", "dst", "kind", "count"])
                for (src, dst, kind), cnt in sorted(route_counter.items(), key=lambda x: (-x[1], x[0])):
                    w.writerow([src, dst, kind, cnt])
            log_info(f"内存交互事务已写入: {mem_path}（共 {len(mem_txn)} 条）")
            log_info(f"内存交互聚合已写入: {summary_path}")
            log_info(f"内存交互事务CSV已写入: {mem_csv_path}")
            log_info(f"内存交互聚合CSV已写入: {summary_csv_path}")

        runtime_memory_rows = _collect_runtime_memory_rows(sim_once)
        if runtime_memory_rows:
            import csv as _csv
            runtime_mem_csv = sim_results_dir / "runtime_memory_events.csv"
            with open(runtime_mem_csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["event_id", "owner_operator", "operator", "memory_tier", "cycle", "end_cycle", "duration", "clock_period_ns", "latency_ns", "energy_nj", "power_mw", "stage", "sub_stage", "circuit_module", "operation", "layer_name", "layer_type", "buffer_name", "buffer_slot", "buffer_phase", "resource_id", "source", "target", "bytes", "overlapped", "runtime_mode"])
                for row in runtime_memory_rows:
                    w.writerow([row.get(k, "") for k in ["event_id", "owner_operator", "operator", "memory_tier", "cycle", "end_cycle", "duration", "clock_period_ns", "latency_ns", "energy_nj", "power_mw", "stage", "sub_stage", "circuit_module", "operation", "layer_name", "layer_type", "buffer_name", "buffer_slot", "buffer_phase", "resource_id", "source", "target", "bytes", "overlapped", "runtime_mode"]])
            runtime_mem_summary = _summarize_runtime_memory_rows(runtime_memory_rows)
            runtime_mem_summary_csv = sim_results_dir / "runtime_memory_summary.csv"
            with open(runtime_mem_summary_csv, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["operator", "owner_operator", "buffer_name", "layer_name", "buffer_phase", "resource_id", "events", "bytes", "energy_nj", "latency_ns"])
                for row in runtime_mem_summary:
                    w.writerow([row.get(k, "") for k in ["operator", "owner_operator", "buffer_name", "layer_name", "buffer_phase", "resource_id", "events", "bytes", "energy_nj", "latency_ns"]])
            log_info(f"Runtime内存事件CSV已写入: {runtime_mem_csv}")
            log_info(f"Runtime内存汇总CSV已写入: {runtime_mem_summary_csv}")

        if args.export_memory_contract:
            mem_contract = analyze_memory_contract(layers, activation_shapes, hw_config)
            contract_txt_path = sim_results_dir / "memory_contract.txt"
            with open(contract_txt_path, "w", encoding="utf-8") as f:
                f.write(format_memory_contract_report(mem_contract))

            contract_csv_path = sim_results_dir / "memory_contract.csv"
            import csv
            with open(contract_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["layer_name", "op_name", "tensor_name", "src_tier", "src_name", "dst_tier", "dst_name", "src", "dst", "bytes", "mib", "estimated_energy_nj", "estimated_energy_model", "estimated_energy_scope", "note"])
                for row in iter_memory_contract_rows(mem_contract):
                    w.writerow([
                        row.get("layer_name", ""),
                        row.get("op_name", ""),
                        row.get("tensor_name", ""),
                        row.get("src_tier", ""),
                        row.get("src_name", ""),
                        row.get("dst_tier", ""),
                        row.get("dst_name", ""),
                        row.get("src", ""),
                        row.get("dst", ""),
                        row.get("bytes", 0),
                        f"{float(row.get('mib', 0.0)):.6f}",
                        f"{float(row.get('estimated_energy_nj', 0.0)):.6f}",
                        row.get("estimated_energy_model", ""),
                        row.get("estimated_energy_scope", ""),
                        row.get("note", ""),
                    ])

            relationship_csv_path = sim_results_dir / "memory_buffer_relationships.csv"
            with open(relationship_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["buffer_name", "shared_group", "role", "physical_sharing", "mutual_exclusion", "reset_scope", "scheduler_resource_id", "pipeline_behavior", "note"])
                for row in iter_buffer_relationship_rows(mem_contract):
                    w.writerow([
                        row.get("buffer_name", ""),
                        row.get("shared_group", ""),
                        row.get("role", ""),
                        row.get("physical_sharing", ""),
                        row.get("mutual_exclusion", ""),
                        row.get("reset_scope", ""),
                        row.get("scheduler_resource_id", ""),
                        row.get("pipeline_behavior", ""),
                        row.get("note", ""),
                    ])

            buffer_csv_path = sim_results_dir / "memory_buffer_requirements.csv"
            with open(buffer_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["layer_name", "op_name", "buffer_tier", "buffer_name", "peak_bytes", "peak_mib", "purpose", "note"])
                for row in iter_buffer_requirement_rows(mem_contract):
                    w.writerow([
                        row.get("layer_name", ""),
                        row.get("op_name", ""),
                        row.get("buffer_tier", ""),
                        row.get("buffer_name", ""),
                        row.get("peak_bytes", 0),
                        f"{float(row.get('peak_mib', 0.0)):.6f}",
                        row.get("purpose", ""),
                        row.get("note", ""),
                    ])

            scheduler_buffer_csv = sim_results_dir / "memory_scheduler_buffer_constraints.csv"
            with open(scheduler_buffer_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["accelerator_mode", "buffer_name", "shared_group", "scheduler_resource_id", "physical_sharing", "mutual_exclusion", "reset_scope", "pipeline_mode", "double_buffering", "buffer_slots", "overlap_policy", "current_model_support", "note"])
                for row in _build_scheduler_buffer_constraints(mem_contract, args.accelerator):
                    w.writerow([row.get("accelerator_mode", ""), row.get("buffer_name", ""), row.get("shared_group", ""), row.get("scheduler_resource_id", ""), row.get("physical_sharing", ""), row.get("mutual_exclusion", ""), row.get("reset_scope", ""), row.get("pipeline_mode", ""), row.get("double_buffering", False), row.get("buffer_slots", 0), row.get("overlap_policy", ""), row.get("current_model_support", ""), row.get("note", "")])

            scheduler_timing_csv = sim_results_dir / "memory_scheduler_timing_constraints.csv"
            with open(scheduler_timing_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["constraint_id", "accelerator_mode", "pipeline_mode", "double_buffering", "buffer_slots", "stage_scope", "producer", "consumer", "buffer_name", "dependency_type", "constraint", "current_model_support", "note"])
                for row in _build_scheduler_timing_constraints(mem_contract, args.accelerator):
                    w.writerow([row.get("constraint_id", ""), row.get("accelerator_mode", ""), row.get("pipeline_mode", ""), row.get("double_buffering", False), row.get("buffer_slots", 0), row.get("stage_scope", ""), row.get("producer", ""), row.get("consumer", ""), row.get("buffer_name", ""), row.get("dependency_type", ""), row.get("constraint", ""), row.get("current_model_support", ""), row.get("note", "")])

            log_info(f"内存契约分析已写入: {contract_txt_path}")
            log_info(f"内存契约明细CSV已写入: {contract_csv_path}")
            log_info(f"Buffer关系表CSV已写入: {relationship_csv_path}")
            log_info(f"算子Buffer需求CSV已写入: {buffer_csv_path}")
            log_info(f"调度Buffer约束CSV已写入: {scheduler_buffer_csv}")
            log_info(f"调度时序约束CSV已写入: {scheduler_timing_csv}")

        return 0

    # -------------------------------------------------------------------------
    # 模式：仅跑仿真（可选先跑 inference 脚本）
    # -------------------------------------------------------------------------
    if args.run_real_inference:
        print("[run_sim_with_real_eval] 运行真实推理 (ModelNet40)...")
        run_real_inference(
            checkpoint_path=checkpoint_path,
            classification_dir=CLASSIFICATION_DIR,
            sim_log_dir=sim_log_path,
            num_points=args.num_points,
            no_vis=True,
        )
        print()

    from pointkan_sim.main import run_network_simulation

    class SimArgs:
        network = "pointkan-elite"
        accelerator = args.accelerator
        input_points = args.num_points
        num_classes = args.num_classes
        show_layers = True
        mnsim_full = args.mnsim_full
        no_mnsim_full = False
        no_layer_buffer = False
        show_shapes = False
        show_events = False
        enable_w8a8_quant_dequant = bool(args.qd_en)
        bn_en = int(args.bn_en)
        cim_backend = getattr(args, "cim_backend", None)
        neurosim_root = getattr(args, "neurosim_root", None)
        use_scalesim_systolic = getattr(args, "use_scalesim_systolic", False)
        scalesim_root = getattr(args, "scalesim_root", None)
        scalesim_config = getattr(args, "scalesim_config", None)
        output_timeline_csv = None
        timeline_stride = 500

    sim_args = SimArgs()
    log_section("PPA 仿真（与 checkpoint/ModelNet40 配置一致）")
    log_info(f"Checkpoint: {checkpoint_path}")
    log_info(f"num_points={args.num_points}, num_classes={args.num_classes}")

    results = run_network_simulation(sim_args)
    hw_config = get_accelerator_config(sim_args.accelerator)
    hw_config.bn_fuse_with_conv_linear = bool(args.bn_en)
    hw_config.enable_w8a8_quant_dequant = bool(args.qd_en)
    if getattr(sim_args, "cim_backend", None):
        hw_config.cim_backend = sim_args.cim_backend
    if getattr(sim_args, "neurosim_root", None):
        hw_config.neurosim_root = sim_args.neurosim_root

    if args.output:
        out_path = sim_log_path / os.path.basename(args.output)
        _write_report_to_file(
            results,
            str(out_path),
            network_name=sim_args.network,
            accelerator_name=hw_config.accelerator_name,
        )
        log_info(f"报告已写入: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
