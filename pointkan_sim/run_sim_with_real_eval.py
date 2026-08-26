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
  python -m pointkan_sim.run_sim_with_real_eval --run_real_inference --neurosim_full --output report.json

完整 shape 分析（Conv/Linear + BN/LayerNorm/ReLU/池化/S-Pool 等）:
  必须使用真实数据模式并开启 --show_shapes，会做一次真实前向并用 hook 收集各层维度后打印。
  方式一（推荐）:
    USE_REAL_DATA=1 SHOW_SHAPES=1 ./run_sim_with_real_eval.sh
  方式二（仅 shape，不跑 NeuroSim 全网络）:
    python -m pointkan_sim.run_sim_with_real_eval --use_real_data --show_shapes --batch_size 2
  日志中会先后出现「Conv/Linear 层 shape」与「非线性算子激活」两段，含 in/out shape、numel、字节数。

时钟（双时钟，可脚本配置）:
  NeuroSim（Conv/Linear）：由 NEUROSIM_CLK_FREQ 传入频率(Hz)，可变（如 100M/200M），不设则用 NeuroSim 关键路径。
  数字逻辑（SPool/LayerNorm/ReLU 等）：由 --logic_clk_freq / LOGIC_CLK_FREQ 传入频率(Hz)，默认 1 GHz。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from pointkan_sim.utils.shared_helpers import (
    sanitize_log_tag as _sanitize_log_tag,
    resolve_neurosim_sar_column_power_w,
)
from pointkan_sim.integration.cim_full_context import (
    CIM_CLOCK_PERIOD_NS,
    CIM_FULL_NETWORK_ACTIVE,
    CIM_LAYER_ENERGY_NJ,
    CIM_LAYER_INDEX,
    CIM_LAYER_LATENCY_NS,
    build_cim_full_context_from_neurosim_lists,
)

# 工程根目录（PointKAN-pytorch）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSIFICATION_DIR = PROJECT_ROOT / "PointKAN-elite" / "classification_ModelNet40"
SCANOBJ_DIR = PROJECT_ROOT / "PointKAN-elite" / "classification_ScanObjectNN"
PARTSEG_DIR = PROJECT_ROOT / "PointKAN-elite" / "part_segmentation"
ISEDA26_DIR = Path("/workspace/ISEDA26")

# 与 run_inference_kan.sh 默认一致
DEFAULT_CKPT = "checkpoints/pointKANElite-pointkan_elite_exp-42"
DEFAULT_SCANOBJ_CKPT = "checkpoints/pointKANElite-scan_exp-42"
DEFAULT_PARTSEG_CKPT = "checkpoints/pointKANElite_part_exp"
DEFAULT_POINTNET2_CLS_CKPT = str(ISEDA26_DIR / "ckpts" / "modelnet_pointnet2ssg_p1024_k20" / "checkpoints" / "best_model.pth")
DEFAULT_POINTNET2_PARTSEG_CKPT = str(ISEDA26_DIR / "ckpts" / "shapenet_pointnet2ssg_p1024_k20" / "checkpoints" / "best_model.pth")
DEFAULT_POINTNET2_SEMSEG_CKPT = str(ISEDA26_DIR / "ckpts" / "s3dis_pointnet2ssg_p4096_k40" / "checkpoints" / "best_model.pth")
DEFAULT_NUM_POINTS = 1024
DEFAULT_NUM_CLASSES = 40

DEFAULT_SIM_RESULTS_DIR = "sim_results"


def _build_export_options(args: argparse.Namespace) -> Dict[str, bool]:
    """Derive which result artifacts should be exported for the current run."""
    lightweight = bool(getattr(args, "lightweight_export", False))
    return {
        "lightweight_export": lightweight,
        "embed_operator_sequence_in_report": not lightweight,
        "export_timelines": not lightweight,
        "export_operator_sequence_files": not lightweight,
        "export_runtime_memory_events": not lightweight,
        "export_memory_transactions": bool(getattr(args, "export_memory_transactions", False)) and not lightweight,
    }


def _cleanup_optional_gam_stage0_timeline_artifacts(sim_results_dir: Path, log_fn) -> None:
    """Remove stale optional GAM-stage0 timeline artifacts before a new export pass."""
    stale_names = [
        "block_intervals_gam_stage0_warmup.json",
        "cycle_timeline_gam_stage0_warmup.txt",
        "block_intervals_gam_stage0_postpredict.json",
        "cycle_timeline_gam_stage0_postpredict.txt",
        "block_intervals_gam_stage0_speculative.json",
        "cycle_timeline_gam_stage0_speculative.txt",
        "block_intervals_gam_stage0_switch.json",
        "cycle_timeline_gam_stage0_switch.txt",
        "block_intervals_gam_stage0_predictor_switch_local.json",
        "cycle_timeline_gam_stage0_predictor_switch_local.txt",
    ]
    removed = []
    for name in stale_names:
        path = sim_results_dir / name
        if path.exists():
            path.unlink()
            removed.append(path.name)
    if removed:
        log_fn(f"GAM-stage0 可选 timeline 旧文件已清理: {', '.join(sorted(removed))}")


def _log_gam_stage0_timeline_status(log_fn, label: str, generated: bool) -> None:
    if generated:
        return
    log_fn(f"GAM-stage0 {label} timeline 本次未生成（无对应事件）")


def _network_dataset_label(network: str) -> str:
    net = (network or "pointkan-elite").lower()
    if net in {"pointkan-elite-partseg", "pointnet2-partseg-ssg"}:
        return "ShapeNetPart"
    if net == "pointkan-elite-scanobj":
        return "ScanObjectNN"
    if net == "pointnet2-sem-ssg":
        return "S3DIS"
    return "ModelNet40"


def _real_eval_title(network: str, backend: Optional[str] = None) -> str:
    dataset = _network_dataset_label(network)
    if backend:
        return f"真实数据评估结果 ({dataset} 测试集 + 对应层 {backend} PPA)"
    return f"真实数据评估结果 ({dataset} 测试集)"


def _collect_runtime_memory_rows(sim_once: dict):
    sim_result = sim_once.get("sim_result")
    events = getattr(sim_result, "events", []) if sim_result is not None else []
    rows = []
    tracked_sram_buffers = {
        "feature_sram_buffer",
        "systolic_feature_sram_buffer",
        "weight_sram_buffer",
        "xyz_sram_buffer",
        "query_xyz_sram_buffer",
        "key_xyz_sram_buffer",
        "fps_index_sram_buffer",
        "residual_sram_buffer",
        "state_sram_buffer",
        "nonlinear_workspace_sram",
        "nonlinear_param_sram",
        "self_distance_sram_buffer",
        "dist_sram_buffer",
        "fps_min_distance_table_sram",
        "temp_max_distance_sram",
        "topk_sram_buffer",
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
        is_dram_event = op == "DRAM"
        is_sram_event = (op == "LayerBuffer") or (is_tracked_sram and not is_dram_event)
        rows.append({
            "event_id": getattr(e, "event_id", 0),
            "owner_operator": str(meta.get("owner_operator", op) or op),
            "operator": "DRAM" if is_dram_event else ("SRAM" if is_sram_event else op),
            "memory_tier": "DRAM" if is_dram_event else ("SRAM" if is_sram_event else op),
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


def _build_conv_linear_summary(
    neurosim_result: Any,
    system_e2e_latency_ns: Optional[float],
) -> Optional[Dict[str, float]]:
    if neurosim_result is None:
        return None
    layer_sum_latency_ns = float(sum(getattr(neurosim_result, "layer_latency_ns", []) or []))
    chip_pipeline_latency_ns_raw = getattr(neurosim_result, "chip_pipeline_latency_ns", None)
    scale_factors = list(getattr(neurosim_result, "workload_scale_factors", []) or [])
    chip_pipeline_exact_for_workload = not any(abs(float(sf) - 1.0) > 1e-9 for sf in scale_factors)
    chip_pipeline_latency_ns = (
        chip_pipeline_latency_ns_raw if chip_pipeline_exact_for_workload else None
    )
    chip_pipeline_plus_non_cim_ns = None
    if system_e2e_latency_ns is not None and chip_pipeline_latency_ns is not None:
        non_cim_latency_ns = max(0.0, float(system_e2e_latency_ns) - layer_sum_latency_ns)
        chip_pipeline_plus_non_cim_ns = float(chip_pipeline_latency_ns) + non_cim_latency_ns
    return {
        "latency_layer_sum_ns": layer_sum_latency_ns,
        "latency_layer_sum_ms": layer_sum_latency_ns / 1e6,
        "latency_chip_pipeline_ns": float(chip_pipeline_latency_ns) if chip_pipeline_latency_ns is not None else None,
        "latency_chip_pipeline_ms": (float(chip_pipeline_latency_ns) / 1e6) if chip_pipeline_latency_ns is not None else None,
        "latency_chip_pipeline_raw_ns": float(chip_pipeline_latency_ns_raw) if chip_pipeline_latency_ns_raw is not None else None,
        "latency_chip_pipeline_raw_ms": (float(chip_pipeline_latency_ns_raw) / 1e6) if chip_pipeline_latency_ns_raw is not None else None,
        "chip_pipeline_exact_for_workload": bool(chip_pipeline_exact_for_workload),
        "latency_chip_pipeline_plus_non_cim_ns": chip_pipeline_plus_non_cim_ns,
        "latency_chip_pipeline_plus_non_cim_ms": (chip_pipeline_plus_non_cim_ns / 1e6) if chip_pipeline_plus_non_cim_ns is not None else None,
        "energy_layer_sum_nj": float(sum(getattr(neurosim_result, "layer_energy_nj", []) or [])),
        "clock_period_ns": float(getattr(neurosim_result, "clock_period_ns", 0.0) or 0.0),
        "modeling_note": (
            "latency_layer_sum_* is the sum of per-layer Conv/Linear latencies and is a conservative serial upper bound. "
            "latency_chip_pipeline_* is only populated when NeuroSim ran the exact workload width for every layer. "
            "If max_in_vectors capped any layer, only latency_chip_pipeline_raw_* is reported because the raw NeuroSim chip throughput cannot be scaled back exactly."
        ),
    }


def _build_fallback_conv_linear_summary(
    full_operator_breakdown: Optional[Dict[str, Dict[str, float]]],
) -> Optional[Dict[str, float]]:
    if not full_operator_breakdown:
        return None
    conv_linear_keys = ("Conv1d", "Conv2d", "Linear")
    latency_layer_sum_ns = 0.0
    energy_layer_sum_nj = 0.0
    matched = False
    for key in conv_linear_keys:
        row = full_operator_breakdown.get(key)
        if not row:
            continue
        matched = True
        latency_layer_sum_ns += float(row.get("latency_ns", 0.0) or 0.0)
        energy_layer_sum_nj += float(row.get("energy_nj", 0.0) or 0.0)
    if not matched:
        return None
    return {
        "latency_layer_sum_ns": latency_layer_sum_ns,
        "latency_layer_sum_ms": latency_layer_sum_ns / 1e6,
        "latency_chip_pipeline_ns": None,
        "latency_chip_pipeline_ms": None,
        "latency_chip_pipeline_plus_non_cim_ns": None,
        "latency_chip_pipeline_plus_non_cim_ms": None,
        "energy_layer_sum_nj": energy_layer_sum_nj,
        "clock_period_ns": None,
        "modeling_note": (
            "This run used per-layer fallback Conv/Linear estimation instead of NeuroSim full-network. "
            "latency_layer_sum_* is the summed Conv/Linear event latency present in the simulator; "
            "latency_chip_pipeline_* is unavailable in fallback mode."
        ),
    }


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
        elif rel.buffer_name == "query_xyz_sram_buffer":
            overlap = "query_resident_reuse"
        elif rel.buffer_name == "key_xyz_sram_buffer":
            overlap = "ping_pong_tile_refill" if (pipeline_mode == "pipeline" and double_buffering) else "serialized_tile_refill"
        elif rel.buffer_name == "nonlinear_workspace_sram":
            overlap = "chunkable_with_feature_pipeline" if pipeline_mode == "pipeline" else "serialized_per_operator_window"
        elif rel.buffer_name == "nonlinear_param_sram":
            overlap = "resident_param_reuse"
        elif rel.buffer_name in ("dist_sram_buffer", "fps_min_distance_table_sram", "sort_cache_sram", "systolic_feature_sram_buffer"):
            overlap = "serialized_on_same_resource"
        elif rel.buffer_name == "weight_sram_buffer":
            overlap = "not_applicable_in_acc" if mode == "acc" else "serialized_on_same_resource"
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
            "current_model_support": "implemented_runtime" if rel.buffer_name in ("feature_sram_buffer", "query_xyz_sram_buffer", "key_xyz_sram_buffer", "dist_sram_buffer", "fps_min_distance_table_sram", "sort_cache_sram", "residual_sram_buffer", "fps_index_sram_buffer") or (rel.buffer_name == "weight_sram_buffer" and mode != "acc") else "implemented_contract",
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
    add("query_xyz_residency", "GAM", "query_xyz_refill", "distance_query_consume", "query_xyz_sram_buffer", "data_dependency", "Current FPS/KNN query xyz block must be loaded into query SRAM before repeated reads by dist2engine; once loaded it can be reused across key-tile scans.", "implemented_contract", "Runtime export support added; full event generation pending.")
    add("key_xyz_tile_refill", "GAM", "DRAM->key_xyz_sram(tile_k)", "distance_compute(tile_k)", "key_xyz_sram_buffer", "data_dependency", "Each key xyz tile must be refilled from DRAM into key SRAM before the corresponding dist2engine scan can start.", "implemented_contract", "Covers src-read plus tgt-write semantics for each key tile.")
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
        add("acc_feature_backbone", "cross_layer", "feature_sram_buffer", "rram_cim_compute", "feature_sram_buffer", "data_dependency", "ACC keeps the same feature-SRAM-first policy as baseline, but removes repeated weight refill because weights stay resident in RRAM-CIM.", "implemented_contract", "Feature movement, not weight staging, is the dominant cross-layer memory path.")
        add("acc_fps_warmup_full_sort", "GAM", "distance_batch128 + min_update", "fps_full_sort_warmup", "fps_min_distance_table_sram", "data_dependency", "For the first 10% FPS points, ACC still follows full-sort FPS: each 128-distance block updates the min-distance table and updates the running max candidate before the predictor is ready.", "implemented_contract", "No coarse/fine speculative thresholding is used during the warmup prefix.")
        add("acc_speculative_overlap", "GAM", "exact_radii_prefix>=2", "resampler_downsample", "temp_max_distance_sram", "overlap_rule", "After at least two exact FPS radii points exist, downsampling/resampler work may overlap the main full-sort FPS loop while the warmup prefix continues growing.", "implemented_contract", "The same temp_max_distance_sram is time-multiplexed for max candidates and speculative radii staging.")
        add("acc_predictor_gate", "GAM", "exact_radii_prefix==10%", "spec_efficient_kan + resampler_upsample", "temp_max_distance_sram", "data_dependency", "Coarse/fine FPS can start only after the first 10% exact FPS points are complete, the predictor emits the 64-point radii curve, and upsampling reconstructs the full radii schedule.", "implemented_contract", "Until then, FPS remains in full-sort mode.")
        add("acc_post_predict_coarse_fine", "GAM", "predicted_whole_radii_ready", "coarse_filter/fine_filter", "temp_max_distance_sram", "data_dependency", "Once the whole predicted radii curve is ready, ACC switches FPS from full-sort to coarse/fine filtering using the predicted threshold schedule.", "implemented_contract", "KNN is unaffected and stays in dynamic-threshold mode throughout.")
        add("acc_knn_dynamic_threshold", "GAM", "distance_batch128", "coarse_filter/topk_refine", "topk_sram_buffer", "data_dependency", "ACC KNN keeps dynamic-threshold filtering for all blocks; candidate writes to topk SRAM and threshold-refinement rereads continue until fine-filter bandwidth is satisfied.", "implemented_contract", "Independent from the FPS speculative-radii path.")

    if mode != "acc":
        add("weight_tile_refill", "conv_linear", "DRAM weight_read(tile)", "weight_sram_buffer -> Conv/Linear", "weight_sram_buffer", "data_dependency", "Weight tile must be refilled into weight_sram_buffer before corresponding Conv/Linear compute tile consumes it.", "implemented_runtime", "32KB weight SRAM causes repeated tiled refills in baseline/base_v2.")
    add("nonlinear_input_ready", "nonlinear", "feature_sram_buffer chunk ready", "nonlinear_engine consume", "nonlinear_workspace_sram", "data_dependency", "AffineNorm/SPool/KAT/ReLU/Add/LayerNorm may start on a chunk only after the corresponding feature chunk is visible in feature SRAM.", "implemented_contract", "Matches chunked feature pipeline semantics; runtime still coarse for most nonlinear ops.")
    add("nonlinear_workspace_mutex", "nonlinear", "nonlinear_op.workspace_use", "nonlinear_op.workspace_use", "nonlinear_workspace_sram", "resource_mutex", "Nonlinear workspace SRAM is shared across AffineNorm/SPool/KAT/Pool-like operators and is exclusive within each active operator window.", "implemented_contract", "Future runtime path should map individual ops to this shared workspace resource.")
    add("nonlinear_param_fetch", "nonlinear", "nonlinear_param_sram", "nonlinear_engine", "nonlinear_param_sram", "data_dependency", "AffineNorm/LayerNorm/KAT parameter or LUT SRAM must be resident before the corresponding nonlinear engine evaluates the chunk.", "implemented_contract", "Small-parameter SRAM is expected to stay resident across many chunks/operators.")
    add("add_residual_read", "nonlinear", "main/residual tensors ready", "add_unit", "feature_sram_buffer", "data_dependency", "ADD waits for both main tensor and residual tensor to be available; residual may come from feature SRAM or DRAM refill in conservative mode.", "implemented_contract", "Residual path can be forced to DRAM in the current conservative baseline assumption.")
    if mode != "acc":
        add("weight_sram_mutex", "conv_linear", "conv_or_linear.weight_stage", "conv_or_linear.weight_stage", "weight_sram_buffer", "resource_mutex", "weight_sram_buffer is shared across Conv/Linear ops and is exclusive within each active staging window.", "implemented_runtime", "resource_id=weight_sram_shared.")
    add("local_feature_mutex", "conv_linear", "conv_or_linear.compute_local", "conv_or_linear.compute_local", "systolic_feature_sram_buffer", "resource_mutex", "Operator-local systolic feature SRAM is serialized on the shared local buffer resource.", "implemented_runtime", "resource_id=systolic_feature_sram_shared.")
    if pipeline_mode == "pipeline" and double_buffering:
        add("key_xyz_ping_pong", "GAM", "key_xyz_slot_a.refill", "key_xyz_slot_b.consume", "key_xyz_sram_buffer", "overlap_rule", "With double buffering, key xyz tile refill for tile k+1 may overlap distance compute consuming tile k from the companion slot.", "implemented_contract", "Ping-pong key SRAM is the intended baseline/base_v2 distance frontend model.")
    else:
        add("key_xyz_no_ping_pong", "GAM", "key_xyz_tile_refill", "key_xyz_tile_consume", "key_xyz_sram_buffer", "overlap_rule", "Without pipeline or without double buffering, key xyz tile refill and key consumption serialize on the same SRAM buffer.", "implemented_contract", "Matches conservative non-pipelined distance frontend.")
    add("dist_buffer_mutex", "GAM", "knn_or_fps_distance_write", "sorter_read_same_bank", "dist_sram_buffer", "resource_mutex", "Distance SRAM write and sorter read must serialize when they target the same shared dist SRAM resource/bank.", "implemented_runtime", "Current runtime models shared resource IDs; bank-level refinement can be added later.")
    add("fps_min_table_mutex", "GAM", "fps_min_update", "fps_sort_read", "fps_min_distance_table_sram", "resource_mutex", "FPS min-distance table update and sorter read are mutually exclusive on the same logical table region.", "implemented_runtime", "resource_id=fps_min_table_shared.")
    return rows


def get_sim_mode_results_dir(args: argparse.Namespace) -> str:
    """
    根据仿真模式生成 sim_results 输出目录名，便于区分网络/加速器/后端/流水线/点数。
    格式:
      sim_results_<network>_<accelerator>
        [_scalesim|_neurosim_full|_neurosim_pybind|_<backend>]
        [_pipeline]
        [_knnconv_<mode>]
        [_saconv]
        _p<num_points>
    """
    network = (getattr(args, "network", None) or "pointkan-elite").strip().lower()
    acc = (getattr(args, "accelerator", None) or "baseline").strip().lower()
    parts = [_sanitize_log_tag(network), acc]
    if acc in ("baseline", "base_v2"):
        if getattr(args, "use_scalesim_systolic", False):
            parts.append("scalesim")
    elif acc in ("acc", "pointcim"):
        if getattr(args, "neurosim_full", False):
            parts.append("neurosim_full")
        elif getattr(args, "neurosim_pybind", False):
            parts.append("neurosim_pybind")
        else:
            backend = getattr(args, "cim_backend", None) or "neurosim"
            parts.append(backend)
    if getattr(args, "qd_en", 0):
        parts.append("pipeline")
    knn_conv_mode = str(getattr(args, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower()
    if knn_conv_mode != "disabled":
        parts.append(f"knnconv_{knn_conv_mode}")
    if bool(getattr(args, "pointcim_sa_conv_pipeline", False)):
        parts.append("saconv")
    num_points = int(getattr(args, "num_points", 0) or 0)
    if num_points > 0:
        parts.append(f"p{num_points}")
    return "sim_results_" + "_".join(parts)


def _normalize_and_validate_acc_backend(args: argparse.Namespace) -> argparse.Namespace:
    acc = str(getattr(args, "accelerator", "") or "").strip().lower()
    if acc in {"baseline", "base_v2"} and not bool(getattr(args, "use_scalesim_systolic", False)):
        args.use_scalesim_systolic = True

    if acc != "acc":
        return args

    if not bool(getattr(args, "use_real_data", False)):
        args.use_real_data = True

    cim_backend = getattr(args, "cim_backend", None)
    if cim_backend:
        raise ValueError("ACC mode no longer supports per-layer cim_backend fallback. Please use --neurosim_full or --neurosim_pybind; default ACC path is NeuroSim full-network (subprocess).")

    if bool(getattr(args, "use_scalesim_systolic", False)):
        raise ValueError("ACC mode does not support SCALE-Sim systolic execution. Use baseline/base_v2 for SCALE-Sim, or ACC with NeuroSim full-network backend.")

    if not bool(getattr(args, "neurosim_full", False)) and not bool(getattr(args, "neurosim_pybind", False)):
        args.neurosim_full = True

    return args


def _parse_args() -> argparse.Namespace:
    """解析命令行参数（网络、加速器、checkpoint、真实数据/仿真模式、BN/量化开关、导出选项等）。"""
    env_accelerator = os.getenv("ACCELERATOR", "baseline")
    env_num_points = int(os.getenv("NUM_POINTS", str(DEFAULT_NUM_POINTS)))
    env_max_batches = os.getenv("MAX_BATCHES")
    env_use_real_data = str(os.getenv("USE_REAL_DATA", "") or "").strip().lower() in {"1", "true", "yes", "on"}
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
        default=env_num_points,
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
        default=env_accelerator,
        help="加速器配置名（baseline=base_v1, base_v2, base_v3, acc）",
    )
    parser.add_argument(
        "--ns_simulation_mode",
        type=str,
        default="event",
        choices=["event", "cycle_accurate"],
        help=(
            "NS 引擎仿真模式: event=事件驱动估算（默认）, "
            "cycle_accurate=FusionArch 风格逐周期仿真（DC+SA 流水线）"
        ),
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
        "--systolic_area_um2",
        type=float,
        default=None,
        metavar="UM2",
        help="baseline 脉动阵列物理面积（um^2）；用于面积 breakdown 与总面积核算，不影响周期/能耗",
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
        "--neurosim_full",
        action="store_true",
        help="对 conv1d/linear 使用 NeuroSim 全网络仿真（DNN_NeuroSim main subprocess）；acc/pointcim 默认启用",
    )
    parser.add_argument(
        "--neurosim_pybind",
        action="store_true",
        help="对 conv1d/linear 使用 neurosim_core pybind 模式（chip_setup/chip_run_all，无 subprocess、无文件 IO）；"
             "与 --neurosim_full 互斥，优先级低于 --neurosim_full",
    )
    parser.add_argument(
        "--cim_backend",
        type=str,
        default=None,
        choices=["neurosim", "fallback"],
        help="Conv/Linear 单层估算后端：neurosim（DNN_NeuroSim）、fallback；无全网络注入时生效",
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
        "--neurosim_clk_mode",
        type=str,
        default="capped",
        choices=["capped", "critical"],
        help=(
            "NeuroSim 时钟模式："
            "capped=使用 max(关键路径 clkPeriod, 用户 --neurosim_clk_freq 对应周期)（默认，保守，可降频）；"
            "critical=忽略用户频率上限，仅按关键路径 clkPeriod 估算（用于“按最坏关键路径频率”对比）。"
        ),
    )
    parser.add_argument(
        "--neurosim_num_col_muxed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "NeuroSim numColMuxed（列复用比）：多少列共享 1 个 ADC/SenseAmp。"
            "不设则使用 NeuroSim 默认（通常 8）。常见对齐策略：W4A4 时设为 4。"
        ),
    )
    parser.add_argument(
        "--neurosim_subarray",
        type=int,
        default=None,
        metavar="N",
        help=(
            "NeuroSim SubArray 行/列尺寸（正方形，兼容旧接口），例如 128 表示 128x128。"
            "若设置，则作为 numRowSubArray / numColSubArray / numRowParallel 的默认值；"
            "若同时设置 --neurosim_subarray_rows / --neurosim_subarray_cols，则以后者为准。"
        ),
    )
    parser.add_argument(
        "--neurosim_subarray_rows",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim numRowSubArray；若设置，则优先覆盖行维度，并默认同步覆盖 numRowParallel。",
    )
    parser.add_argument(
        "--neurosim_subarray_cols",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim numColSubArray；若设置，则优先覆盖列维度。",
    )
    parser.add_argument(
        "--neurosim_tile_buffer_core_size_row",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim tileBufferCoreSizeRow；tile buffer core 行尺寸，默认沿用 Param.cpp。",
    )
    parser.add_argument(
        "--neurosim_tile_buffer_core_size_col",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim tileBufferCoreSizeCol；tile buffer core 列尺寸，同时影响 tile buffer interface width。",
    )
    parser.add_argument(
        "--neurosim_force_tile_size_cm",
        type=int,
        default=None,
        metavar="N",
        help="强制 NeuroSim Conventional Mapping 的 tile size（如 1024）；用于 iso-hardware 对比。",
    )
    parser.add_argument(
        "--neurosim_force_pe_size_cm",
        type=int,
        default=None,
        metavar="N",
        help="强制 NeuroSim Conventional Mapping 的 PE size（如 512）；用于 iso-hardware 对比。",
    )
    parser.add_argument(
        "--neurosim_force_num_tiles_cm",
        type=int,
        default=None,
        metavar="N",
        help="强制 NeuroSim Conventional Mapping 的 tile 总数预算（如 60）；用于 iso-hardware 对比。",
    )
    parser.add_argument(
        "--neurosim_relu_mode",
        type=int,
        choices=[0, 1],
        default=None,
        metavar="0|1",
        help="NeuroSim subprocess main：1=ReLU，0=Sigmoid；不设则沿用配置/NeuroSim 默认。",
    )
    parser.add_argument(
        "--neurosim_chip_activation_mode",
        type=int,
        choices=[0, 1],
        default=None,
        metavar="0|1",
        help="NeuroSim：1=chip-level activation，0=tile-level activation；不设则默认。",
    )
    parser.add_argument(
        "--neurosim_enable_activation",
        type=int,
        choices=[0, 1],
        default=None,
        metavar="0|1",
        help="NeuroSim：是否计入片上 activation 相关开销；0/1 覆盖默认。",
    )
    parser.add_argument(
        "--neurosim_adc_clk_freq",
        type=float,
        default=None,
        metavar="HZ",
        help="NeuroSim SAR ADC 专用时钟频率(Hz)；与 --neurosim_clk_freq 独立",
    )
    parser.add_argument(
        "--neurosim_sar_column_power_w",
        type=float,
        default=None,
        metavar="W",
        help="NeuroSim SAR ADC Column_Power（W）；仅 NeuroSim 模式生效",
    )
    parser.add_argument(
        "--neurosim_sar_column_power_uw",
        type=float,
        default=None,
        metavar="UW",
        help="NeuroSim SAR ADC Column_Power（uW）；与 --neurosim_sar_column_power_w 二选一",
    )
    parser.add_argument(
        "--neurosim_technode",
        type=int,
        default=32,
        help="NeuroSim 工艺节点(nm)，如 32；仅支持 130/90/65/45/32/22/14/10/7，28 会自动改为 32",
    )
    parser.add_argument(
        "--neurosim_timeout_s",
        type=int,
        default=None,
        metavar="SEC",
        help="NeuroSim 全网络仿真 main 超时秒数；默认不设(无限等待)。需要防卡死时可手动设为 300/900 等",
    )
    parser.add_argument(
        "--neurosim_max_in_vectors",
        type=int,
        default=None,
        metavar="N",
        help="NeuroSim 全网络每层输入向量上限(W_in/tokens)；不设=完整 workload，不做 latency/energy 线性缩放（与 pybind 对齐）",
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
        default=env_use_real_data,
        help="用 ModelNet40 真实样本与 checkpoint 权重：加载模型与数据集，从真实前向收集 conv/linear 维度驱动 NeuroSim 全网络（或单层估算），按批调度仿真并汇总准确率与 PPA",
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
        default=(int(env_max_batches) if env_max_batches is not None and str(env_max_batches).strip() != "" else None),
        help="--use_real_data 时最多跑多少 batch（默认全量测试集）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help="--use_real_data 时最多处理多少个样本（默认不限制）；设为 1 可强制只跑 1 个样本（与 batch_size 无关）",
    )
    parser.add_argument(
        "--progress_interval_s",
        type=float,
        default=30.0,
        metavar="SEC",
        help="长耗时阶段（NeuroSim full / DRAMSim3）心跳日志间隔秒数；0 或负数表示关闭",
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
        help="--use_real_data 时 PPA 报告与 breakdown 输出目录；为默认值时按网络/模式自动命名（如 sim_results_pointkan-elite_baseline_scalesim_p1024）",
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
        "--lightweight_export",
        action="store_true",
        help="保留最终 PPA/breakdown 结果，但关闭超大体积明细导出（如 operator_sequence、timeline、runtime memory events）",
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
        "--fps_min_distance_table_sram_size_bytes",
        type=int,
        default=None,
        metavar="BYTES",
        help="覆盖 FPS 最小距离表 SRAM 容量（bytes），用于消融/sweep；如 4096、8192、16384、32768",
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
        "--knn_conv_pipeline_mode",
        type=str,
        choices=["disabled", "strict"],
        default="disabled",
        help="KNN→Conv producer-consumer 调度模式：disabled 保留原始层流水；strict 仅在 Conv1d 输入长度严格满足 query_points×k 时启用流水",
    )
    parser.add_argument(
        "--pointcim_sa_conv_pipeline",
        action="store_true",
        default=False,
        help="PointCIM 模式专用：每个 query 的 BallQuery 完成后立即驱动后续 Conv2d(kernel_size=1)，与下一 query 的 FPS+BQ 并行（SA sample_and_group ↔ MLP 流水线）",
    )
    parser.add_argument("--fc_num_slices", type=int, default=1, help="Feature Computing parallel slices (1=single-block)")
    parser.add_argument("--fc_cores_per_slice", type=int, default=16, help="Cores per FC slice")
    parser.add_argument("--fc_muls_per_core", type=int, default=1024, help="MACs per FC core")
    parser.add_argument("--fc_coalescing_factor", type=float, default=0.0, help="DRAM read savings from LHT coalescing (0.0-1.0)")
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
        "--knn_dynamic_threshold_rounds_mean",
        type=int,
        default=None,
        metavar="N",
        help="acc KNN 动态阈值平均额外收敛轮数；0=首轮 block 粗筛后直接 fine，1=再做 1 轮 threshold refine，依此类推",
    )
    parser.add_argument(
        "--knn_candidates_per_block",
        type=int,
        default=None,
        metavar="N",
        help="acc KNN 每个 block 粗筛后平均写入 topk SRAM 的候选个数；该参数直接决定 topk_candidate_write 的候选点数，后续 refine 轮数由 --knn_dynamic_threshold_rounds_mean 控制",
    )
    parser.add_argument(
        "--fps_block_topk",
        type=int,
        default=None,
        metavar="N",
        help="acc FPS 后90%% coarse 阶段每个 block 平均保留的候选个数；这些候选写入 temp_max_distance_sram，随后全局读出再做 fps_fine",
    )
    parser.add_argument(
        "--acc_segment_count",
        type=int,
        default=None,
        metavar="N",
        help="acc speculative FPS 的 segment 数；例如 32 表示把 512 个 FPS 点切成 32 段，每段 16 点",
    )
    parser.add_argument(
        "--resampler_parallelism",
        type=int,
        default=None,
        metavar="N",
        help="acc resampler 并行度 P；上/下采样核心周期按 ceil(N/P)+2 建模",
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
    parser.add_argument(
        "--network",
        type=str,
        default="pointkan-elite",
        choices=[
            "pointkan",
            "pointkan-elite",
            "pointkan-elite-scanobj",
            "pointkan-elite-partseg",
            "pointnet2-cls-ssg",
            "pointnet2-partseg-ssg",
            "pointnet2-sem-ssg",
        ],
        help=(
            "仿真目标网络: pointkan | pointkan-elite | "
            "pointkan-elite-scanobj | pointkan-elite-partseg | "
            "pointnet2-cls-ssg | pointnet2-partseg-ssg | pointnet2-sem-ssg; "
            "默认 pointkan-elite。"
            "pointkan-elite-scanobj: ScanObjectNN 分类(1024pts, 15类); "
            "pointkan-elite-partseg: ShapeNetPart 零件分割(2048pts, 50类, use_xyz=True)"
        ),
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=None,
        metavar="C",
        help=(
            "每点输入特征维度；仅 pointnet2-sem-ssg 需要指定（默认 12=xyz+rgb+norm_xyz）; "
            "其他网络忽略此参数"
        ),
    )
    parser.add_argument(
        "--neurosim_wl_weight",
        type=int,
        default=8,
        metavar="BITS",
        help="NeuroSim 权重量化位宽（wl_weight），默认 8；4 表示 4-bit 权重",
    )
    parser.add_argument(
        "--neurosim_wl_activation",
        type=int,
        default=8,
        metavar="BITS",
        help="NeuroSim 输入激活量化位宽（wl_activation），默认 8；4 表示 4-bit 激活",
    )
    return parser.parse_args()


def _apply_neurosim_cli_overrides(hw_config, args: argparse.Namespace) -> None:
    if getattr(args, "neurosim_root", None):
        hw_config.neurosim_root = args.neurosim_root
    if getattr(args, "neurosim_pybind", False):
        hw_config.neurosim_pybind = True
    clk_freq = getattr(args, "neurosim_clk_freq", None)
    if clk_freq is not None and float(clk_freq) > 0:
        hw_config.neurosim_clk_freq_hz = float(clk_freq)
    if getattr(args, "neurosim_clk_mode", None) is not None:
        hw_config.neurosim_clk_mode = str(getattr(args, "neurosim_clk_mode", "capped") or "capped")
    if getattr(args, "neurosim_num_col_muxed", None) is not None:
        v = int(getattr(args, "neurosim_num_col_muxed", 0) or 0)
        hw_config.neurosim_num_col_muxed = v if v > 0 else None
    if getattr(args, "neurosim_subarray", None) is not None:
        v = int(getattr(args, "neurosim_subarray", 0) or 0)
        hw_config.neurosim_subarray = v if v > 0 else None
    if getattr(args, "neurosim_subarray_rows", None) is not None:
        v = int(getattr(args, "neurosim_subarray_rows", 0) or 0)
        hw_config.neurosim_subarray_rows = v if v > 0 else None
    if getattr(args, "neurosim_subarray_cols", None) is not None:
        v = int(getattr(args, "neurosim_subarray_cols", 0) or 0)
        hw_config.neurosim_subarray_cols = v if v > 0 else None
    if getattr(args, "neurosim_tile_buffer_core_size_row", None) is not None:
        v = int(getattr(args, "neurosim_tile_buffer_core_size_row", 0) or 0)
        hw_config.neurosim_tile_buffer_core_size_row = v if v > 0 else None
    if getattr(args, "neurosim_tile_buffer_core_size_col", None) is not None:
        v = int(getattr(args, "neurosim_tile_buffer_core_size_col", 0) or 0)
        hw_config.neurosim_tile_buffer_core_size_col = v if v > 0 else None
    if getattr(args, "neurosim_force_tile_size_cm", None) is not None:
        v = int(getattr(args, "neurosim_force_tile_size_cm", 0) or 0)
        hw_config.neurosim_force_tile_size_cm = v if v > 0 else None
    if getattr(args, "neurosim_force_pe_size_cm", None) is not None:
        v = int(getattr(args, "neurosim_force_pe_size_cm", 0) or 0)
        hw_config.neurosim_force_pe_size_cm = v if v > 0 else None
    if getattr(args, "neurosim_force_num_tiles_cm", None) is not None:
        v = int(getattr(args, "neurosim_force_num_tiles_cm", 0) or 0)
        hw_config.neurosim_force_num_tiles_cm = v if v > 0 else None
    if getattr(args, "neurosim_relu_mode", None) is not None:
        hw_config.neurosim_relu_mode = int(args.neurosim_relu_mode)
    if getattr(args, "neurosim_chip_activation_mode", None) is not None:
        hw_config.neurosim_chip_activation_mode = int(args.neurosim_chip_activation_mode)
    if getattr(args, "neurosim_enable_activation", None) is not None:
        hw_config.neurosim_enable_activation = int(args.neurosim_enable_activation)
    if getattr(args, "neurosim_technode", None) is not None:
        hw_config.neurosim_technode = int(args.neurosim_technode)
    adc_clk = getattr(args, "neurosim_adc_clk_freq", None)
    if adc_clk is not None and float(adc_clk) > 0:
        hw_config.neurosim_adc_clk_freq_hz = float(adc_clk)
    sar_power_w = resolve_neurosim_sar_column_power_w(
        getattr(args, "neurosim_sar_column_power_w", None),
        getattr(args, "neurosim_sar_column_power_uw", None),
    )
    if sar_power_w is not None:
        hw_config.neurosim_sar_column_power_w = float(sar_power_w)


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


def resolve_checkpoint_path(checkpoint_arg: Optional[str], network: Optional[str] = None) -> str:
    """解析 checkpoint 路径：按任务优先相对对应工程目录解析。"""
    net = str(network or "").strip().lower()
    if net == "pointkan-elite-partseg":
        default_base = PARTSEG_DIR
        default_ckpt = DEFAULT_PARTSEG_CKPT
    elif net == "pointkan-elite-scanobj":
        default_base = SCANOBJ_DIR
        default_ckpt = DEFAULT_SCANOBJ_CKPT
    elif net == "pointnet2-cls-ssg":
        default_base = ISEDA26_DIR
        default_ckpt = DEFAULT_POINTNET2_CLS_CKPT
    elif net == "pointnet2-partseg-ssg":
        default_base = ISEDA26_DIR
        default_ckpt = DEFAULT_POINTNET2_PARTSEG_CKPT
    elif net == "pointnet2-sem-ssg":
        default_base = ISEDA26_DIR
        default_ckpt = DEFAULT_POINTNET2_SEMSEG_CKPT
    else:
        default_base = CLASSIFICATION_DIR
        default_ckpt = DEFAULT_CKPT
    if not checkpoint_arg:
        path = Path(default_ckpt)
        if not path.is_absolute():
            path = default_base / default_ckpt
    else:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            if not path.exists():
                alt = default_base / checkpoint_arg
                if alt.exists() or (alt / "best_checkpoint.pth").exists() or (alt / "best_insiou_model.pth").exists() or (alt / "best_model.pth").exists():
                    path = alt
            path = path.resolve()
    return str(path)


def apply_network_defaults(args: argparse.Namespace) -> argparse.Namespace:
    network = str(getattr(args, "network", "") or "").strip().lower()
    if network.startswith("pointnet2-"):
        if str(getattr(args, "knn_conv_pipeline_mode", "disabled") or "disabled").strip().lower() == "disabled":
            args.knn_conv_pipeline_mode = "strict"
    if network == "pointkan-elite-partseg":
        if getattr(args, "num_points", None) == DEFAULT_NUM_POINTS:
            args.num_points = 2048
        if getattr(args, "num_classes", None) == DEFAULT_NUM_CLASSES:
            args.num_classes = 50
    elif network == "pointkan-elite-scanobj":
        if getattr(args, "num_points", None) == DEFAULT_NUM_POINTS:
            args.num_points = 1024
        if getattr(args, "num_classes", None) == DEFAULT_NUM_CLASSES:
            args.num_classes = 15
    elif network == "pointnet2-cls-ssg":
        if getattr(args, "num_points", None) == DEFAULT_NUM_POINTS:
            args.num_points = 1024
        if getattr(args, "num_classes", None) == DEFAULT_NUM_CLASSES:
            args.num_classes = 40
    elif network == "pointnet2-partseg-ssg":
        if getattr(args, "num_points", None) == DEFAULT_NUM_POINTS:
            args.num_points = 1024
        if getattr(args, "num_classes", None) == DEFAULT_NUM_CLASSES:
            args.num_classes = 50
    elif network == "pointnet2-sem-ssg":
        if getattr(args, "num_points", None) == DEFAULT_NUM_POINTS:
            args.num_points = 4096
        if getattr(args, "num_classes", None) == DEFAULT_NUM_CLASSES:
            args.num_classes = 13
        if getattr(args, "input_channels", None) is None:
            args.input_channels = 12
    return args


def main() -> int:
    args = _parse_args()
    args = apply_network_defaults(args)
    args = _normalize_and_validate_acc_backend(args)

    sim_log_path = _ensure_sim_log(args.sim_log_dir)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, args.network)

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    from datetime import datetime
    from pointkan_sim.utils.logger import setup_logger, LogLevel
    network_hint = args.network if getattr(args, "network", None) else Path(str(args.checkpoint)).stem
    log_tag = _sanitize_log_tag(network_hint or "sim")
    log_path = sim_log_path / f"simulation_{log_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    from pointkan_sim.memory.feature_residency import get_effective_feature_buffer_bytes
    from pointkan_sim.memory.memory_contract import (
        analyze_memory_contract,
        format_memory_contract_report,
        iter_buffer_relationship_rows,
        iter_buffer_requirement_rows,
        iter_memory_contract_rows,
    )

    def _export_standard_sim_outputs(results: Dict[str, Any], sim_args: Any, hw_cfg: Any) -> Path:
        import csv as _csv
        from types import SimpleNamespace
        from pointkan_sim.analysis.timeline import (
            block_intervals_to_json_rows,
            build_block_intervals,
            render_ascii_timeline,
        )
        from pointkan_sim.integration.ppa_report import build_ppa_report, write_ppa_report_and_plots
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

        use_auto_dir = args.sim_results_dir == DEFAULT_SIM_RESULTS_DIR
        effective_sim_results_dir = get_sim_mode_results_dir(args) if use_auto_dir else args.sim_results_dir
        sim_results_dir = Path(effective_sim_results_dir)
        sim_results_dir.mkdir(parents=True, exist_ok=True)
        if use_auto_dir:
            log_info(f"PPA 与 breakdown 输出目录: {sim_results_dir}（按仿真模式自动命名）")

        report_filename = os.path.basename(args.output) if args.output else "report.json"
        eval_like = SimpleNamespace(
            total_cycles=int(results.get("total_cycles", 0) or 0),
            total_energy_nj=float(results.get("total_energy_nJ", 0.0) or 0.0),
            num_samples=1,
            accuracy=0.0,
            balanced_accuracy=0.0,
            test_loss=0.0,
        )
        export_options = _build_export_options(sim_args)

        ppa_report = build_ppa_report(
            eval_like,
            float(getattr(hw_cfg, "clock_period", 1.0) or 1.0),
            full_operator_breakdown=results.get("operator_breakdown") or {},
            checkpoint="",
            neurosim_full_network=bool(getattr(sim_args, "neurosim_full", False)),
            operator_sequence=((results.get("operator_sequence") or []) if export_options["embed_operator_sequence_in_report"] else None),
            latency_per_sample_ns=float(results.get("latency_sum_ns", results.get("latency_ns", 0.0)) or 0.0),
            critical_path_latency_per_sample_ns=float(results.get("latency_critical_path_ns", results.get("latency_ns", 0.0)) or 0.0),
            total_area_um2=float(results.get("total_area_um2", 0.0) or 0.0),
            event_breakdown=results.get("breakdown") or {},
            activation_shapes=results.get("activation_shapes") or [],
            pointcim_distance_summary=results.get("pointcim_distance_summary"),
            dist_chain_analysis=results.get("dist_chain_analysis"),
        )
        ppa_report.setdefault("metadata", {})
        ppa_report["metadata"].update({
            "network": sim_args.network,
            "accelerator": hw_cfg.accelerator_name,
            "generated_from": "only_sim_export",
        })
        report_path, figure_paths = write_ppa_report_and_plots(
            ppa_report,
            sim_results_dir,
            report_filename=report_filename,
            log_info_fn=log_info,
        )
        if figure_paths:
            log_info(f"breakdown 饼图已写入: {sim_results_dir}")

        breakdown_path = sim_results_dir / "breakdown_detail.json"
        with open(breakdown_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(results.get("breakdown") or {}, f, indent=2, ensure_ascii=False)
        log_info(f"breakdown 明细已写入: {breakdown_path}")

        sim_result = results.get("sim_result")
        sim_events = list(getattr(sim_result, "events", []) or []) if sim_result is not None else []
        block_intervals = getattr(sim_result, "block_intervals", None) or (build_block_intervals(sim_events) if sim_events else [])
        operator_sequence = results.get("operator_sequence") or []
        operator_cycle_detail = results.get("operator_cycle_detail") or []

        if export_options["export_timelines"] and block_intervals:
            timeline_json_path = sim_results_dir / "block_intervals.json"
            with open(timeline_json_path, "w", encoding="utf-8") as f:
                f.write(block_intervals_to_json_rows(block_intervals))
            global_txt_path = sim_results_dir / "cycle_timeline.txt"
            with open(global_txt_path, "w", encoding="utf-8") as f:
                f.write(render_ascii_timeline(block_intervals, max_cycle=500))
            log_info(f"文本版 Cycle timeline 已写入: {global_txt_path}")

        def _write_filtered_timeline(name: str, events_subset, *, start_from_first: bool = True, max_cycle: int = 500, label_width: int = 48):
            if not events_subset:
                return
            start_cycle = None
            if start_from_first:
                start_cycle = min(int(getattr(e, "cycle", 0) or 0) for e in events_subset)
            intervals = build_block_intervals(
                events_subset,
                detailed=True,
                start_cycle=start_cycle,
                end_cycle=(None if start_cycle is None else start_cycle + max_cycle),
            )
            if not intervals:
                return
            json_path = sim_results_dir / f"block_intervals_{name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(block_intervals_to_json_rows(intervals))
            txt_path = sim_results_dir / f"cycle_timeline_{name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(render_ascii_timeline(intervals, max_cycle=max_cycle, label_width=label_width))
            log_info(f"{name} 文本版 timeline 已写入: {txt_path}")

        if export_options["export_timelines"] and sim_events:
            gam_stage0_events = [
                e for e in sim_events
                if str(getattr(e, "layer_name", "") or ((getattr(e, "metadata", None) or {}).get("layer_name", "")) or "") == "GAM-stage0"
            ]
            if gam_stage0_events:
                _write_filtered_timeline("gam_stage0", gam_stage0_events, max_cycle=500)
                _write_filtered_timeline("gam_stage0_long", gam_stage0_events, max_cycle=3000)
                compute_ops = {
                    'self_distance_read',
                    'cross_term_distance', 'distance_merge_accumulate',
                    'fps_min_distance_table_read', 'min_distance_compare', 'fps_min_distance_table_write',
                    'fps_warmup_coarse_top32', 'fps_warmup_fine_max', 'knn_coarse_filter',
                    'topk_candidate_write', 'knn_threshold_update', 'knn_threshold_refine',
                    'topk_candidate_refine_write', 'knn_fine_filter_topk', 'fps_coarse_filter_max',
                    'temp_max_candidate_append', 'fps_tempmax_reduce_read', 'fps_tempmax_reduce_write',
                    'fps_tempmax_global_read', 'fps_fine_filter_max',
                    'topk_candidate_aggregate_read', 'topk_candidate_fine_read',
                }
                compute_events = [e for e in gam_stage0_events if str(getattr(e, "operation", "") or "") in compute_ops]
                if compute_events:
                    _write_filtered_timeline("gam_stage0_compute", compute_events, max_cycle=500)
                    _write_filtered_timeline("gam_stage0_compute_long", compute_events, max_cycle=3000)
                fps_compute_events = [e for e in compute_events if str(getattr(e, "operator", "") or "") == "FPS"]
                if fps_compute_events:
                    _write_filtered_timeline("gam_stage0_fps_compute", fps_compute_events, max_cycle=500)
                    _write_filtered_timeline("gam_stage0_fps_compute_long", fps_compute_events, max_cycle=3000)
                knn_compute_events = [e for e in compute_events if str(getattr(e, "operator", "") or "") == "KNN"]
                if knn_compute_events:
                    _write_filtered_timeline("gam_stage0_knn_compute", knn_compute_events, max_cycle=500)
                    _write_filtered_timeline("gam_stage0_knn_compute_long", knn_compute_events, max_cycle=3000)
                feature_events = [
                    e for e in sim_events
                    if str(getattr(e, "operator", "") or "") in {"LayerBuffer", "SRAM", "Quant", "Dequant", "ADD", "ReLU", "AffineNorm", "LayerNorm", "KAN"}
                ]
                _write_filtered_timeline("feature_pipeline", feature_events, max_cycle=500)
                _write_filtered_timeline("feature_pipeline_long", feature_events, max_cycle=5000)
            conv_linear_events = [e for e in sim_events if str(getattr(e, "operator", "") or "") in {"Conv1d", "Linear", "Quant", "Dequant"}]
            _write_filtered_timeline("conv_linear", conv_linear_events, max_cycle=500)
            _write_filtered_timeline("conv_linear_long", conv_linear_events, max_cycle=5000)
            nonlinear_events = [e for e in sim_events if str(getattr(e, "operator", "") or "") in {"ReLU", "AffineNorm", "LayerNorm", "KAN", "ADD"}]
            _write_filtered_timeline("nonlinear", nonlinear_events, max_cycle=500)
            _write_filtered_timeline("nonlinear_long", nonlinear_events, max_cycle=3000)

        if export_options["export_operator_sequence_files"] and operator_sequence:
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
                base_fields = ["idx", "cycle", "end_cycle", "duration", "operator", "stage", "operation", "layer_name", "energy_nj"]
                fieldnames = list(base_fields)
                seen = set(fieldnames)
                for item in operator_sequence:
                    for key in item.keys():
                        if key not in seen:
                            seen.add(key)
                            fieldnames.append(key)
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for item in operator_sequence:
                    w.writerow(item)
        if operator_cycle_detail:
            detail_csv_path = sim_results_dir / "operator_cycle_detail.csv"
            with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
                if operator_cycle_detail:
                    fieldnames = []
                    seen = set()
                    for row in operator_cycle_detail:
                        for key in row.keys():
                            if key not in seen:
                                seen.add(key)
                                fieldnames.append(key)
                    w = _csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    for row in operator_cycle_detail:
                        w.writerow(row)
        return sim_results_dir

    # -------------------------------------------------------------------------
    # 模式：用 ModelNet40 真实样本 + 真实权重 调度仿真，NeuroSim 由真实前向得到的层维度驱动
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
        import math

        log_section("真实数据评估：真实样本 + checkpoint 权重 + 对应层 PPA")
        if args.accelerator in {"acc", "pointcim"}:
            if args.neurosim_full:
                log_info("ACC backend selection: using NeuroSim full-network path (subprocess unless --neurosim_pybind).")
            elif args.neurosim_pybind:
                log_info("ACC backend selection: using NeuroSim pybind path.")

        log_info(f"Checkpoint: {checkpoint_path}")
        log_info(f"num_points={args.num_points}, num_classes={args.num_classes}, batch_size={args.batch_size}")

        try:
            net, test_loader, criterion = load_model_and_dataset(
                checkpoint_path,
                network=args.network,
                num_points=args.num_points,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                workers=0,
                device=args.device,
            )
        except Exception as e:
            log_info(f"加载模型/数据集失败: {e}")
            return 1

        # 层定义（与后续 NeuroSim、仿真共用）
        hw_config = get_accelerator_config(args.accelerator)
        # NS 引擎仿真模式（event / cycle_accurate）
        hw_config.ns_simulation_mode = getattr(args, "ns_simulation_mode", "event") or "event"
        # 用户可控开关：BN 融合 / 量化反量化 / CIM 后端 / SCALE-Sim（baseline 脉动 Conv/Linear）
        hw_config.bn_fuse_with_conv_linear = bool(args.bn_en)
        hw_config.enable_w8a8_quant_dequant = (
            True if args.accelerator in {"acc", "pointcim"} and args.qd_en is None else bool(args.qd_en)
        )
        if getattr(args, "cim_backend", None):
            hw_config.cim_backend = args.cim_backend
        _apply_neurosim_cli_overrides(hw_config, args)
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
                if getattr(args, "systolic_area_um2", None) is not None:
                    hw_config.systolic_array.area = max(0.0, float(args.systolic_area_um2))
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
        if getattr(args, "fps_min_distance_table_sram_size_bytes", None) is not None and getattr(hw_config, "fps_min_distance_table_sram", None) is not None:
            hw_config.fps_min_distance_table_sram.size_bytes = max(1, int(args.fps_min_distance_table_sram_size_bytes))
        if getattr(args, "inter_layer_sram_bw", None) is not None and getattr(hw_config, "inter_layer_buffer", None) is not None:
            hw_config.inter_layer_buffer.bandwidth_bytes_per_cycle = max(1, int(args.inter_layer_sram_bw))
        if getattr(args, "dram_bw_bytes_per_cycle", None) is not None and getattr(hw_config, "dram", None) is not None:
            hw_config.dram.bandwidth_bytes_per_cycle = max(1, int(args.dram_bw_bytes_per_cycle))
            if getattr(hw_config, "inter_layer_buffer", None) is not None:
                hw_config.inter_layer_buffer.dram_bandwidth_bytes_per_cycle = max(1, int(args.dram_bw_bytes_per_cycle))
        if getattr(args, "dram_channels", None) is not None and getattr(hw_config, "dram", None) is not None:
            hw_config.dram.dramsim3_channels_override = max(1, int(args.dram_channels))
        if getattr(args, "knn_dynamic_threshold_rounds_mean", None) is not None:
            hw_config.knn_dynamic_threshold_rounds_mean = max(0, int(args.knn_dynamic_threshold_rounds_mean))
        if getattr(args, "knn_candidates_per_block", None) is not None:
            hw_config.knn_candidates_per_block = max(1, int(args.knn_candidates_per_block))
        if getattr(args, "fps_block_topk", None) is not None:
            hw_config.fps_block_topk = max(1, int(args.fps_block_topk))
        if getattr(args, "acc_segment_count", None) is not None:
            hw_config.acc_segment_count = max(1, int(args.acc_segment_count))
        if getattr(args, "resampler_parallelism", None) is not None:
            if getattr(getattr(hw_config, "logic_circuit", None), "resampler", None) is not None:
                hw_config.logic_circuit.resampler.parallelism = max(1, int(args.resampler_parallelism))
        if getattr(hw_config, "inter_layer_buffer", None) is not None:
            if getattr(args, "feature_buffer_pipeline_mode", None):
                hw_config.inter_layer_buffer.pipeline_mode = str(args.feature_buffer_pipeline_mode)
            if getattr(args, "feature_double_buffering", False):
                hw_config.inter_layer_buffer.double_buffering = True
            if getattr(args, "feature_buffer_slots", None) is not None:
                hw_config.inter_layer_buffer.buffer_slots = max(1, int(args.feature_buffer_slots))
        hw_config.knn_conv_pipeline_mode = str(getattr(args, "knn_conv_pipeline_mode", "disabled") or "disabled")
        hw_config.pointcim_sa_conv_pipeline = bool(getattr(args, "pointcim_sa_conv_pipeline", False))
        _fc_slices = getattr(args, "fc_num_slices", 1) or 1
        if _fc_slices > 1 or getattr(args, "fc_coalescing_factor", 0.0):
            from pointkan_sim.configs.hw_config import FeatureComputingConfig
            hw_config.feature_computing = FeatureComputingConfig(
                num_slices=_fc_slices,
                cores_per_slice=getattr(args, "fc_cores_per_slice", 16) or 16,
                muls_per_core=getattr(args, "fc_muls_per_core", 1024) or 1024,
                coalescing_factor=getattr(args, "fc_coalescing_factor", 0.0) or 0.0,
            )

        qdq_parallelism = getattr(args, "logic_par_quant_dequant", None)
        if getattr(hw_config, "is_acc", False) and getattr(hw_config, "enable_w8a8_quant_dequant", False) and qdq_parallelism is None:
            qdq_parallelism = 64
        set_logic_parallelism(
            hw_config,
            layernorm=getattr(args, "logic_par_layernorm", None),
            relu=getattr(args, "logic_par_relu", None),
            spool=getattr(args, "logic_par_spool", None),
            quant_dequant=qdq_parallelism,
        )
        _net_name = getattr(args, "network", "pointkan-elite")
        # PointNet++ does not use PointKAN-elite GAM speculative FPS predictor.
        # In acc/pointcim configs fps_engine_mode defaults to "fuse", which introduces
        # Resampler/SpecPredictor events. For pointnet2-* we force baseline FPS engine.
        if str(_net_name).startswith("pointnet2-") and getattr(hw_config, "is_acc", False):
            hw_config.fps_engine_mode = "baseline"
        _net_kwargs = {}
        if _net_name == "pointnet2-sem-ssg" and getattr(args, "input_channels", None):
            _net_kwargs["input_channels"] = int(args.input_channels)
        _enable_dr = getattr(hw_config, "distance_matrix_reuse", False)
        layers = load_network(
            _net_name, args.num_points, args.num_classes,
            enable_distance_reuse=_enable_dr, **_net_kwargs,
        )
        if getattr(args, "fps_block_topk", None) is not None:
            fps_topk = max(1, int(args.fps_block_topk))
            for layer in layers:
                for op in getattr(layer, "operators", []) or []:
                    if getattr(op, "op_type", "") == "fps":
                        op.params["fps_block_topk"] = fps_topk
        if getattr(args, "knn_candidates_per_block", None) is not None:
            knn_candidates = max(1, int(args.knn_candidates_per_block))
            for layer in layers:
                for op in getattr(layer, "operators", []) or []:
                    if getattr(op, "op_type", "") == "knn":
                        op.params["knn_candidates_per_block"] = knn_candidates

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
                log_info(
                    f"DRAM: strict_latency=DRAMSim3, legacy_bw_hint={hw_config.dram.bandwidth_bytes_per_cycle} B/cycle, "
                    f"channels_override={getattr(hw_config.dram, 'dramsim3_channels_override', None)}"
                )
            if getattr(hw_config, 'is_acc', False):
                log_info(f"ACC KNN dynamic threshold rounds(mean extra refine): {getattr(hw_config, 'knn_dynamic_threshold_rounds_mean', 0)}")
                log_info(f"ACC KNN candidates per block: {getattr(hw_config, 'knn_candidates_per_block', 'default')}")
                log_info(f"ACC FPS block top-k: {getattr(hw_config, 'fps_block_topk', 'default')}")
                log_info(f"ACC segment count: {getattr(hw_config, 'acc_segment_count', 'default')}")
                log_info(f"ACC resampler parallelism: {getattr(getattr(getattr(hw_config, 'logic_circuit', None), 'resampler', None), 'parallelism', 'default')}")
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

        try:
            contract_policy = MemoryContractPolicy()
        except Exception:
            contract_policy = None
        def _buf_size(name: str, default: Any = "N/A") -> Any:
            cfg = getattr(hw_config, name, None)
            if cfg is None:
                return default
            return getattr(cfg, "size_bytes", default)
        log_info("Memory Config Summary:")
        dram_cfg = getattr(hw_config, "dram", None)
        if dram_cfg is not None:
            log_info(
                f"  DRAM: ini={getattr(dram_cfg, 'dramsim3_config_ini', None)}, "
                f"strict_latency=DRAMSim3, "
                f"legacy_bw_hint={getattr(dram_cfg, 'bandwidth_bytes_per_cycle', 'N/A')} B/cycle, "
                f"block={getattr(dram_cfg, 'block_size_bytes', 'N/A')} B, "
                f"channels_override={getattr(dram_cfg, 'dramsim3_channels_override', None)}"
            )
        if buf is not None:
            log_info(
                f"  feature_sram_buffer: size={getattr(buf, 'size_bytes', 'N/A')} B, "                f"effective_per_slot={get_effective_feature_buffer_bytes(hw_config)} B, "                f"block={getattr(buf, 'block_size_bytes', 'N/A')} B, "                f"bw={getattr(buf, 'bandwidth_bytes_per_cycle', 'N/A')} B/cycle"
            )
        sa_cfg = getattr(hw_config, "systolic_array", None)
        if sa_cfg is not None:
            if args.accelerator in {"acc", "pointcim"}:
                log_info(
                    f"  acc_local_compute: rram_cim(weight_resident)=True, local_state_sram={getattr(sa_cfg, 'sram_size_bytes', 'N/A')} B, "
                    f"block={getattr(sa_cfg, 'sram_block_bytes', 'N/A')} B, banks={getattr(sa_cfg, 'sram_num_banks', 'N/A')}, "
                    f"bus={getattr(sa_cfg, 'sram_bus_width_bits', 'N/A')} bit"
                )
            else:
                log_info(
                    f"  systolic_local_sram: unified={getattr(sa_cfg, 'sram_size_bytes', 'N/A')} B, "                f"ifmap={getattr(sa_cfg, 'ifmap_sram_size_bytes', None) or getattr(sa_cfg, 'sram_size_bytes', 'N/A')} B, "                f"filter={getattr(sa_cfg, 'filter_sram_size_bytes', None) or getattr(sa_cfg, 'sram_size_bytes', 'N/A')} B, "                f"ofmap={getattr(sa_cfg, 'ofmap_sram_size_bytes', None) or getattr(sa_cfg, 'sram_size_bytes', 'N/A')} B, "                f"block={getattr(sa_cfg, 'sram_block_bytes', 'N/A')} B, banks={getattr(sa_cfg, 'sram_num_banks', 'N/A')}, "                f"bus={getattr(sa_cfg, 'sram_bus_width_bits', 'N/A')} bit"
                )
        if contract_policy is not None:
            if args.accelerator in {"acc", "pointcim"}:
                macro = getattr(hw_config, "sram_cim_macro", None)
                array = getattr(macro, "array", None) if macro is not None else None
                cim_parallel = max(1, int(getattr(hw_config, "cim_parallelism", 128) or 128))
                array_rows = max(cim_parallel, int(getattr(array, "rows", cim_parallel) or cim_parallel))
                bank_capacity_points = max(cim_parallel, array_rows)
                acc_input_points = max(1, int(getattr(args, "num_points", 1) or 1))
                acc_num_banks = max(1, int(math.ceil(acc_input_points / float(bank_capacity_points))))
                acc_total_cim_rows = acc_num_banks * array_rows
                acc_total_self_distance_bytes = acc_input_points * 4
                log_info(
                    f"  acc_distance_buffers: query_xyz={_buf_size('query_xyz_sram_buffer')} B, key_xyz_tile={_buf_size('xyz_sram_buffer')} B, "
                    f"self_distance={_buf_size('self_distance_sram_buffer')} B, dist={_buf_size('dist_sram_buffer')} B, "
                    f"fps_min_table={_buf_size('fps_min_distance_table_sram')} B"
                )
                log_info(
                    f"  acc_adaptive_capacity: per_bank_cim_rows={array_rows}, per_bank_keys={bank_capacity_points}, "
                    f"active_banks={acc_num_banks}, total_cim_rows={acc_total_cim_rows}, "
                    f"effective_self_distance_total={acc_total_self_distance_bytes} B"
                )
                log_info(
                    f"  acc_streaming_buffers: temp_max/spec_radii={_buf_size('temp_max_distance_sram')} B, "
                    f"topk={_buf_size('topk_sram_buffer')} B, sort_cache={_buf_size('sort_cache_sram')} B"
                )
                log_info(
                    f"  acc_nonlinear_buffers: kan_workspace={contract_policy.kan_workspace_sram_bytes} B, residual={contract_policy.residual_sram_bytes} B, "
                    f"xyz_offchip_bytes={contract_policy.point_xyz_offchip_bytes}, xyz_onchip_bytes={contract_policy.point_xyz_onchip_bytes}, distance_bytes={contract_policy.distance_bytes}, index_bytes={contract_policy.index_bytes}"
                )
                log_info(
                    "  acc_speculative_path: temp_max_distance_sram is time-multiplexed across warmup running-max, predictor staging, and post-predict coarse/fine FPS"
                )
            else:
                log_info(
                    f"  distance/nonlinear runtime buffers: query_xyz={_buf_size('query_xyz_sram_buffer')} B, key_xyz={_buf_size('xyz_sram_buffer')} B, "
                    f"dist={_buf_size('dist_sram_buffer')} B, fps_min_table={_buf_size('fps_min_distance_table_sram')} B, "
                    f"sort_cache={_buf_size('sort_cache_sram')} B"
                )
                log_info(
                    "  query_xyz_sram_buffer: runtime now uses hw_config.size_bytes for capacity/spill checks"
                )
                log_info(
                    "  key xyz / distance SRAM buffers: runtime now uses the corresponding hw_config buffer sizes"
                )
                log_info(
                    f"  dist_sram_buffer: {_buf_size('dist_sram_buffer')} B, "
                    f"fps_min_distance_table_sram: {_buf_size('fps_min_distance_table_sram')} B, "
                    f"sort_cache_sram: {_buf_size('sort_cache_sram')} B"
                )
                log_info(
                    f"  nonlinear_workspace_sram: kan={contract_policy.kan_workspace_sram_bytes} B, residual={contract_policy.residual_sram_bytes} B"
                )
                log_info(
                    "  nonlinear_param_sram: small resident SRAM/LUT for AffineNorm/LayerNorm/KAT-GRF params "
                    "(modeled as resident; negligible capacity pressure in current baseline contract)"
                )
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

        # 先做一次真实前向，收集 conv/linear 顺序与真实 indata/rdata 字节数，用于 NeuroSim 按层矩阵乘适配
        first_batch = None
        try:
            first_batch = next(iter(test_loader))
        except StopIteration:
            first_batch = None

        conv_linear_list = collect_conv_linear_from_forward(
            net,
            args.device,
            args.num_points,
            batch_size=min(2, args.batch_size),
            network=args.network,
            batch=first_batch,
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
                net,
                args.device,
                args.num_points,
                batch_size=min(2, args.batch_size),
                network=args.network,
                batch=first_batch,
            )
            log_nonlinear_activation_shapes(nonlinear_list, log_info_fn=log_info)

        # NeuroSim 全网络：仅 acc 模式下运行并输出；baseline/base_v2 不输出 NeuroSim 全网络信息
        activation_shapes = compute_activation_shapes(layers, args.num_points, input_channels=3)
        acc_name = (args.accelerator or "").strip().lower()
        is_accelerator = bool(getattr(hw_config, "is_acc", False))

        cim_full_context = None
        n_cim_layers = 0
        netstruct = None
        conv_linear_order = None
        used_neurosim_for_ppa = False
        neurosim_result = None
        neurosim_root = getattr(hw_config, "neurosim_root", "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch")

        use_pybind = (
            is_accelerator
            and bool(getattr(args, "neurosim_pybind", False))
            and not bool(getattr(args, "neurosim_full", False))
        )
        if use_pybind:
            from pointkan_sim.integration.neurosim_full import run_neurosim_full_pybind
            _pybind_so_dir = os.path.join(
                getattr(args, "neurosim_root",
                        getattr(hw_config, "neurosim_root",
                                "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch")),
                "NeuroSIM",
            )
            log_info("使用 neurosim_core pybind 模式（chip_setup/chip_run_all，无 subprocess）")
            _ns_cap = getattr(args, "neurosim_max_in_vectors", None)
            _ns_cap = int(_ns_cap) if _ns_cap is not None and int(_ns_cap) > 0 else None
            neurosim_result = run_neurosim_full_pybind(
                conv_linear_list,
                neurosim_so_dir=_pybind_so_dir,
                wl_weight=int(getattr(args, "neurosim_wl_weight", 8) or 8),
                wl_activation=int(getattr(args, "neurosim_wl_activation", 8) or 8),
                subarray=int(getattr(hw_config, "neurosim_subarray_rows", getattr(hw_config, "neurosim_subarray", 128)) or getattr(hw_config, "neurosim_subarray", 128) or 128),
                subarray_cols=int(getattr(hw_config, "neurosim_subarray_cols", getattr(hw_config, "neurosim_subarray", 128)) or getattr(hw_config, "neurosim_subarray", 128) or 128),
                parallel_read=int(getattr(hw_config, "neurosim_subarray_rows", getattr(hw_config, "neurosim_subarray", 128)) or getattr(hw_config, "neurosim_subarray", 128) or 128),
                clk_freq_hz=getattr(args, "neurosim_clk_freq", None),
                technode=getattr(args, "neurosim_technode", None),
                adc_clk_freq_hz=getattr(args, "neurosim_adc_clk_freq", None),
                sar_column_power_w=resolve_neurosim_sar_column_power_w(
                    getattr(args, "neurosim_sar_column_power_w", None),
                    getattr(args, "neurosim_sar_column_power_uw", None),
                ),
                num_col_muxed=int(getattr(hw_config, "neurosim_num_col_muxed", 0) or 0) or 8,
                max_in_vectors=_ns_cap,
            )
            if not neurosim_result.success:
                log_info(f"neurosim_core pybind 失败: {neurosim_result.error}，回退到 subprocess 模式")
                use_pybind = False
                args.neurosim_full = True   # trigger subprocess fallback below
            else:
                # pybind 成功：写入 cim_full_context，供 Conv1d/Linear dispatch 使用
                used_neurosim_for_ppa = True
                pybind_clock_ns = getattr(neurosim_result, "clock_period_ns", None) or 0.0
                cim_full_context = build_cim_full_context_from_neurosim_lists(
                    conv_linear_list,
                    layer_latency_ns=list(neurosim_result.layer_latency_ns),
                    layer_energy_nj=list(neurosim_result.layer_energy_nj),
                    clock_period_ns=float(pybind_clock_ns) if pybind_clock_ns > 0 else None,
                    layer_area_um2=list(getattr(neurosim_result, "layer_area_um2", []) or []),
                )
                n_cim_layers = len(neurosim_result.layer_latency_ns)
                log_info("已用 NeuroSim 全网络仿真（pybind: chip_setup/chip_run_all）得到逐层 latency/energy")
                if getattr(neurosim_result, "total_area_um2", None):
                    log_info(f"NeuroSim Conv/Linear 总面积: {float(neurosim_result.total_area_um2):.2f} um^2")
                if pybind_clock_ns > 0:
                    freq_mhz = 1000.0 / pybind_clock_ns
                    log_info(f"NeuroSim Conv/Linear 时钟: {pybind_clock_ns:.4f} ns ({freq_mhz:.2f} MHz)")
                else:
                    log_info("未解析到 NeuroSim pybind 时钟周期，Conv/Linear 亦使用数字逻辑时钟")
                log_info("以下层使用 NeuroSim pybind 接口计算 PPA：")
                clock_ns_log = (cim_full_context.get(CIM_CLOCK_PERIOD_NS) if cim_full_context else None) or hw_config.clock_period
                from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
                for idx, (lat_ns, en_nj) in enumerate(
                    zip(neurosim_result.layer_latency_ns, neurosim_result.layer_energy_nj)
                ):
                    name = conv_linear_list[idx].name if idx < len(conv_linear_list) else f"layer_{idx}"
                    cyc = max(1, int(math.ceil(lat_ns / clock_ns_log)))
                    act_s, w_s = (
                        format_activation_weight_shapes(conv_linear_list[idx])
                        if idx < len(conv_linear_list)
                        else ("?", "?")
                    )
                    log_info(f"  NeuroSim pybind [{name}] activation={act_s}, weight={w_s}  latency={lat_ns:.1f} ns, cycles={cyc}, energy={en_nj:.2f} nJ")
                log_info(f"  （共 {n_cim_layers} 层）")

        if is_accelerator and args.neurosim_full:
            from pointkan_sim.integration.neurosim_full import run_neurosim_full
            _ns_cap = getattr(args, "neurosim_max_in_vectors", None)
            _ns_cap = int(_ns_cap) if _ns_cap is not None and int(_ns_cap) > 0 else None
            _timeout = getattr(args, "neurosim_timeout_s", None)
            _timeout = int(_timeout) if _timeout is not None and int(_timeout) > 0 else None
            _prog = getattr(args, "progress_interval_s", 30.0)
            try:
                _prog = float(_prog) if _prog is not None else 0.0
            except Exception:
                _prog = 0.0
            _prog = _prog if _prog > 0 else None

            log_info(
                "NeuroSim full(subprocess): start "
                f"({len(conv_linear_list)} layers), timeout_s={_timeout}, heartbeat={_prog}s"
            )
            _t0 = time.perf_counter()
            neurosim_result = run_neurosim_full(
                conv_linear_list,
                neurosim_root=neurosim_root,
                wl_weight=int(getattr(args, "neurosim_wl_weight", 8) or 8),
                wl_activation=int(getattr(args, "neurosim_wl_activation", 8) or 8),
                subarray=int(getattr(hw_config, "neurosim_subarray_rows", getattr(hw_config, "neurosim_subarray", 128)) or getattr(hw_config, "neurosim_subarray", 128) or 128),
                subarray_cols=int(getattr(hw_config, "neurosim_subarray_cols", getattr(hw_config, "neurosim_subarray", 128)) or getattr(hw_config, "neurosim_subarray", 128) or 128),
                parallel_read=int(getattr(hw_config, "neurosim_subarray_rows", getattr(hw_config, "neurosim_subarray", 128)) or getattr(hw_config, "neurosim_subarray", 128) or 128),
                timeout_s=_timeout,
                clk_freq_hz=getattr(args, "neurosim_clk_freq", None),
                technode=getattr(args, "neurosim_technode", None),
                adc_clk_freq_hz=getattr(args, "neurosim_adc_clk_freq", None),
                sar_column_power_w=resolve_neurosim_sar_column_power_w(
                    getattr(args, "neurosim_sar_column_power_w", None),
                    getattr(args, "neurosim_sar_column_power_uw", None),
                ),
                num_col_muxed=getattr(hw_config, "neurosim_num_col_muxed", None),
                tile_buffer_core_size_row=getattr(hw_config, "neurosim_tile_buffer_core_size_row", None),
                tile_buffer_core_size_col=getattr(hw_config, "neurosim_tile_buffer_core_size_col", None),
                force_tile_size_cm=getattr(hw_config, "neurosim_force_tile_size_cm", None),
                force_pe_size_cm=getattr(hw_config, "neurosim_force_pe_size_cm", None),
                force_num_tiles_cm=getattr(hw_config, "neurosim_force_num_tiles_cm", None),
                neurosim_relu_mode=getattr(hw_config, "neurosim_relu_mode", None),
                neurosim_chip_activation_mode=getattr(hw_config, "neurosim_chip_activation_mode", None),
                neurosim_enable_activation=getattr(hw_config, "neurosim_enable_activation", None),
                clk_mode=str(getattr(args, "neurosim_clk_mode", "capped") or "capped"),
                max_in_vectors=_ns_cap,
                progress_interval_s=_prog,
                progress_log_info=log_info,
            )
            log_info(
                f"NeuroSim full(subprocess): done in {(time.perf_counter() - _t0):.1f}s "
                f"(success={bool(getattr(neurosim_result, 'success', False))})"
            )
            if neurosim_result.success:
                used_neurosim_for_ppa = True
                if (
                    getattr(args, "neurosim_clk_freq", None) is not None
                    or getattr(args, "neurosim_adc_clk_freq", None) is not None
                    or getattr(args, "neurosim_sar_column_power_w", None) is not None
                    or getattr(args, "neurosim_sar_column_power_uw", None) is not None
                    or getattr(args, "neurosim_technode", None) is not None
                    or getattr(args, "neurosim_timeout_s", None) is not None
                ):
                    log_info(
                        "NeuroSim 频率/ADC/工艺/超时: "
                        f"clk_freq_hz={getattr(args, 'neurosim_clk_freq', None)}, "
                        f"adc_clk_freq_hz={getattr(args, 'neurosim_adc_clk_freq', None)}, "
                        f"sar_column_power_w={resolve_neurosim_sar_column_power_w(getattr(args, 'neurosim_sar_column_power_w', None), getattr(args, 'neurosim_sar_column_power_uw', None))}, "
                        f"technode={getattr(args, 'neurosim_technode', None)}, "
                        f"timeout_s={getattr(args, 'neurosim_timeout_s', None)}"
                    )
                neurosim_clock_ns = getattr(neurosim_result, "clock_period_ns", None) or 0.0
                cim_full_context = build_cim_full_context_from_neurosim_lists(
                    conv_linear_list,
                    layer_latency_ns=list(neurosim_result.layer_latency_ns),
                    layer_energy_nj=list(neurosim_result.layer_energy_nj),
                    clock_period_ns=float(neurosim_clock_ns) if neurosim_clock_ns > 0 else None,
                    layer_area_um2=list(getattr(neurosim_result, "layer_area_um2", []) or []),
                )
                n_cim_layers = len(neurosim_result.layer_latency_ns)
                _backend_tag = "pybind (chip_setup/chip_run_all)" if use_pybind else "subprocess (DNN_NeuroSim main)"
                log_info(f"已用 NeuroSim 全网络仿真（{_backend_tag}）得到逐层 latency/energy")
                if getattr(neurosim_result, "total_area_um2", None):
                    log_info(f"NeuroSim Conv/Linear 总面积: {float(neurosim_result.total_area_um2):.2f} um^2")
                area_src = str(getattr(neurosim_result, "layer_area_source", "") or "")
                if area_src:
                    log_info(f"NeuroSim 逐层面积来源: {area_src}")
                # 数字逻辑时钟由 hw_config.clock_period（可 --logic_clk_freq 设置，默认 1 GHz）；Conv/Linear 单独用 NeuroSim 时钟
                if neurosim_clock_ns > 0:
                    freq_mhz = 1000.0 / neurosim_clock_ns
                    log_info(f"NeuroSim Conv/Linear 时钟: {neurosim_clock_ns:.4f} ns ({freq_mhz:.2f} MHz)，由 NEUROSIM_CLK_FREQ 传入（可变，不限于 100M）；数字逻辑时钟见上方")
                else:
                    log_info("未解析到 NeuroSim 时钟周期，Conv/Linear 亦使用数字逻辑时钟 (hw_config.clock_period)")
                log_info("以下层使用 NeuroSim 接口计算 PPA：")
                clock_ns_log = (cim_full_context.get(CIM_CLOCK_PERIOD_NS) if cim_full_context else None) or hw_config.clock_period
                from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
                for idx, (lat_ns, en_nj) in enumerate(zip(neurosim_result.layer_latency_ns, neurosim_result.layer_energy_nj)):
                    name = conv_linear_list[idx].name if idx < len(conv_linear_list) else f"layer_{idx}"
                    cyc = max(1, int(math.ceil(lat_ns / clock_ns_log)))
                    act_s, w_s = format_activation_weight_shapes(conv_linear_list[idx]) if idx < len(conv_linear_list) else ("?", "?")
                    log_info(f"  NeuroSim [{name}] activation={act_s}, weight={w_s}  latency={lat_ns:.1f} ns, cycles={cyc}, energy={en_nj:.2f} nJ")
                log_info(f"  （共 {n_cim_layers} 层）")
                # NeuroSim 模式：输出消耗的 tile/PE 与对应 size（占位或解析结果）
                if getattr(neurosim_result, "rram_tile_stats", None):
                    st = neurosim_result.rram_tile_stats
                    src = str(st.get("source", "") or "")
                    log_info("NeuroSim Tile/PE 统计:")
                    if src == "neurosim_stdout":
                        log_info("  统计来源: NeuroSim stdout 严格解析")
                    elif src == "neurosim_placeholder":
                        log_info("  注意: 当前 Tile/PE 统计是占位估计，不是 NeuroSim stdout 严格解析值。")
                    total_tiles = st.get('total_tile_count', '?')
                    total_pes = st.get('total_PE_count', '?')
                    log_info(f"  总 Tile 数: {total_tiles},  总 PE 数: {total_pes if total_pes is not None else '?'}")
                    tile_sz = st.get("tile_size") or {}
                    pe_sz = st.get("pe_size") or {}
                    xbar = st.get("xbar_size") or {}
                    tile_row = tile_sz.get('row', '?') if tile_sz.get('row', None) is not None else '?'
                    tile_col = tile_sz.get('column', '?') if tile_sz.get('column', None) is not None else '?'
                    pe_row = pe_sz.get('row', '?') if pe_sz.get('row', None) is not None else '?'
                    pe_col = pe_sz.get('column', '?') if pe_sz.get('column', None) is not None else '?'
                    row = xbar.get('row', '?') if xbar.get('row', None) is not None else '?'
                    col = xbar.get('column', '?') if xbar.get('column', None) is not None else '?'
                    pe_per_tile = st.get('pe_per_tile', '?') if st.get('pe_per_tile', None) is not None else '?'
                    log_info(f"  Tile size: {tile_row} x {tile_col}  |  PE size: {pe_row} x {pe_col}  |  Crossbar size: {row} x {col}  (pe_per_tile={pe_per_tile})")
                    if st.get("per_layer"):
                        log_info(f"  逐层: 共 {len(st['per_layer'])} 层")
                        for pl in st["per_layer"][:8]:
                            tilenum = pl.get('tilenum', '?')
                            pe_count = pl.get('PE_count', '?') if pl.get('PE_count', None) is not None else '?'
                            max_row = pl.get('max_row', '?') if pl.get('max_row', None) is not None else '?'
                            max_col = pl.get('max_column', '?') if pl.get('max_column', None) is not None else '?'
                            log_info(f"    layer_{pl.get('layer_index', '?')} ({pl.get('layer_type', '')}): tilenum={tilenum}, PE_count={pe_count}, max_row={max_row}, max_column={max_col}")
                        if len(st["per_layer"]) > 8:
                            log_info(f"    ... 其余 {len(st['per_layer']) - 8} 层")
            else:
                log_info(f"NeuroSim 全网络未启用: {neurosim_result.error}")
                if neurosim_result.error and len(neurosim_result.error) > 200:
                    for line in neurosim_result.error.split("\n")[:15]:
                        if line.strip():
                            log_info(f"  NeuroSim stderr: {line.strip()}")
                log_info("  提示: 若为主进程崩溃(returncode≠0)，可检查 DNN_NeuroSim 是否支持当前层数/维度；层数较多时可尝试减少 batch 或先单独运行 NeuroSIM/main 排查。")

        need_area_fallback = False
        if is_accelerator and used_neurosim_for_ppa and neurosim_result is not None:
            layer_area_list = getattr(neurosim_result, "layer_area_um2", None)
            total_area_val = float(getattr(neurosim_result, "total_area_um2", 0.0) or 0.0)
            need_area_fallback = not layer_area_list or not any(float(x or 0.0) > 0.0 for x in layer_area_list) or total_area_val <= 0.0
            if need_area_fallback:
                log_info("NeuroSim full 未提供可靠 per-layer/total area；继续使用 NeuroSim latency/energy，面积 breakdown 可能不完整。")

        if is_accelerator and cim_full_context is not None:
            n_inj = len(cim_full_context[CIM_LAYER_LATENCY_NS])
            log_info(f"真实前向共 {len(conv_linear_list)} 个 conv/linear；全网络 PPA 使用 {n_inj} 层")
            if len(conv_linear_list) != n_inj:
                log_info(f"  注意：真实前向层数({len(conv_linear_list)})与 LayerDef 层数({n_inj})不一致；overrides 按索引取前 {n_inj} 条对应到全网络 CIM 注入。建议使用 --show_shapes 核对。")
        elif is_accelerator:
            log_info("真实前向已收集 Conv/Linear shape，但全网络 NeuroSim 当前不可用；Conv1d/Linear 将回退到单层 NeuroSim/RRAM-CIM 估算。")
        else:
            log_info(f"真实前向共 {len(conv_linear_list)} 个 conv/linear（baseline 脉动由 SCALE-Sim 逐层统计）")

        # 按对应数据集批次调度：每批一次真实前向（任务指标）+ 一次 PPA 仿真（累加）
        eval_result = run_real_inference_and_sim_over_dataset(
            net,
            test_loader,
            criterion,
            args.device,
            layers,
            hw_config,
            cim_full_context,
            enable_layer_buffer=True,
            activation_shapes=activation_shapes,
            max_batches=args.max_batches,
            max_samples=args.max_samples,
            repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            real_conv_linear_list=conv_linear_list,
            network=args.network,
        )

        if not eval_result.success:
            log_info(f"真实数据评估失败: {eval_result.error}")
            return 1

        log_separator()
        if used_neurosim_for_ppa or (cim_full_context and cim_full_context.get(CIM_FULL_NETWORK_ACTIVE)):
            log_info(_real_eval_title(args.network, "NeuroSim"))
        else:
            log_info(_real_eval_title(args.network))
        log_separator()
        log_info(f"  样本数:        {eval_result.num_samples}")
        log_info(f"  准确率:        {eval_result.accuracy:.2f}%")
        if getattr(eval_result, "task_type", "classification") == "partseg":
            log_info(f"  Instance mIoU: {getattr(eval_result, 'instance_miou', 0.0):.2f}%")
        elif getattr(eval_result, "task_type", "classification") == "semseg":
            log_info(f"  mAcc/平衡准确率:{eval_result.balanced_accuracy:.2f}%")
        else:
            log_info(f"  平衡准确率:    {eval_result.balanced_accuracy:.2f}%")
        log_info(f"  Test Loss:     {eval_result.test_loss:.4f}")
        # 端到端延迟：同时打印求和口径与关键路径口径（真实事件时间轴 makespan）
        latency_per_sample_ns = getattr(eval_result, "latency_per_sample_ns", None)
        if latency_per_sample_ns is None or latency_per_sample_ns <= 0:
            # 回退：若结果未带 latency_ns 则用单时钟近似（不准确）
            clock_fallback = (cim_full_context.get(CIM_CLOCK_PERIOD_NS) if cim_full_context else None) or hw_config.clock_period
            latency_per_sample_ns = eval_result.cycles_per_sample * clock_fallback
        critical_path_latency_ns = (
            float(getattr(eval_result, "critical_path_latency_per_sample_ns", 0.0) or 0.0)
            if hasattr(eval_result, "critical_path_latency_per_sample_ns")
            else 0.0
        )
        if critical_path_latency_ns <= 0:
            critical_path_latency_ns = latency_per_sample_ns
        log_info(f"  总周期(等效基础时钟, 关键路径): {eval_result.total_cycles:,} cycles")
        log_info(f"  总能耗:        {eval_result.total_energy_nj:.2f} nJ")
        log_info(f"  事件延迟求和(非墙钟): {latency_per_sample_ns/1e6:.4f} ms")
        log_info(f"  端到端延迟(关键路径 makespan): {critical_path_latency_ns/1e6:.4f} ms")
        log_info(f"  每样本周期(等效基础时钟, 关键路径): {eval_result.cycles_per_sample:.1f} cycles")
        log_info(f"  每样本能耗:    {eval_result.energy_per_sample_nj:.2f} nJ")
        log_separator()

        # 单次仿真获取全算子 breakdown（FPS/KNN/BN/Conv/Linear 等），用于饼图与报告
        from pointkan_sim.main import simulate_network, _attach_breakdown_and_ppa
        sim_once = getattr(eval_result, "sim_once", None)
        if not isinstance(sim_once, dict) or not sim_once:
            cim_ctx_once = dict(cim_full_context) if cim_full_context else None
            if cim_ctx_once is not None:
                cim_ctx_once[CIM_LAYER_INDEX] = 0
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sim_once = simulate_network(
                layers,
                hw_config,
                show_layers=False,
                cim_full_context=cim_ctx_once,
                activation_shapes=activation_shapes,
                enable_layer_buffer=True,
                show_shapes=False,
                repo_root=_repo_root,
            )
        if isinstance(sim_once, dict) and "breakdown" not in sim_once:
            _attach_breakdown_and_ppa(sim_once, hw_config)
        backend_total_area_um2 = 0.0
        if used_neurosim_for_ppa and neurosim_result is not None:
            backend_total_area_um2 = float(getattr(neurosim_result, "total_area_um2", 0.0) or 0.0)
        if backend_total_area_um2 > 0.0:
            sim_once["total_area_um2"] = backend_total_area_um2
        full_operator_breakdown = sim_once.get("operator_breakdown") or {}
        operator_cycle_detail = sim_once.get("operator_cycle_detail") or []
        operator_sequence = sim_once.get("operator_sequence") or []
        total_area_um2 = float(sim_once.get("total_area_um2", 0) or 0)
        clock_ns = getattr(hw_config, "clock_period", 1.0)  # 数字逻辑 1 GHz；仅用于 fallback 与报告

        # 控制台打印：端到端延迟、能耗与面积构成（按算子类型）；Quant/Dequant/BN 单独列出
        if full_operator_breakdown:
            total_cy = sum(v.get("cycles", 0) for v in full_operator_breakdown.values())
            total_en = sum(v.get("energy_nj", 0) for v in full_operator_breakdown.values())
            total_area = total_area_um2 if total_area_um2 > 0 else sum(v.get("area_um2", 0) for v in full_operator_breakdown.values())
            latency_total_ns = sum(v.get("latency_ns", 0) for v in full_operator_breakdown.values())
            if latency_total_ns <= 0:
                latency_total_ns = total_cy * (hw_config.clock_period or 1.0)
            critical_path_breakdown_ns = float(getattr(eval_result, "critical_path_latency_per_sample_ns", 0.0) or 0.0)
            if critical_path_breakdown_ns <= 0:
                critical_path_breakdown_ns = float(getattr(eval_result, "latency_sum_per_sample_ns", latency_total_ns) or latency_total_ns)
            gap_stall_ns = max(0.0, critical_path_breakdown_ns - latency_total_ns)
            log_separator()
            log_info("按算子 Busy-Time、能耗与面积构成 (每样本；面积来自 NeuroSim 或逻辑模块)")
            log_info("  说明: Conv/Linear 用 NeuroSim 时钟（NEUROSIM_CLK_FREQ 传入，可变）；数字逻辑用 --logic_clk_freq/LOGIC_CLK_FREQ 接口，默认 1 GHz。")
            log_info("  AffineNorm=GAM 仿射归一化（不含 BN）；BN 在 BN_EN=1 时融合进 Conv/Linear 不单独出现，BN_EN=0 时单独列出为 BN；QD_EN=1 时 Quant/Dequant 已单独列出。")
            log_info("  查看入口: AffineNorm=pointkan_sim.operators.common_ops.gen_affine_norm_events；BN=gen_bn_events；BN 融合=dispatcher；Quant/Dequant=common_ops.gen_quant_events/gen_dequant_events。")
            log_info("  内存: SRAM=层间 SRAM 读写（原 LayerBuffer）；DRAM=首层点云加载与 spill。")
            log_info("  能耗: 除 SRAM CIM Array 用用户数据外，LayerBuffer 仅来自 CACTI 仿真（fill_inter_layer_buffer_from_cacti）；首层与 spill 的 DRAM 仅来自 trace→DRAMSim3（repo_root + hw_config.dram.dramsim3_bin 时自动跑）。无回退。详见 ARCHITECTURE.md §1.4。")
            log_info("  口径: 表内 latency(ns) 为按 (算子, 层名) 合并区间后的 busy-time，不等于端到端墙钟延迟；未归因的 gap/stall 单独在表后给出。")
            log_separator()
            log_info(f"  {'算子':<20} {'cycles':>12} {'cycles%':>8} {'busy(ns)':>14} {'busy%':>8} {'energy(nJ)':>12} {'energy%':>8} {'area(um2)':>12} {'area%':>8}")
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
            log_info(f"  算子 busy-time 合计: {latency_total_ns/1e6:.4f} ms  |  能耗合计: {total_en:.2f} nJ  |  系统总面积: {total_area:.2f} um^2")
            log_info(f"  端到端关键路径: {critical_path_breakdown_ns/1e6:.4f} ms  |  未归因 gap/stall: {gap_stall_ns/1e6:.4f} ms")
            log_separator()

        # 距离链 (FPS+KNN) 关键路径 / busy-time 分析
        dist_chain_analysis = sim_once.get("dist_chain_analysis") or {}
        if dist_chain_analysis:
            clk_ns_dc = getattr(hw_config, "clock_period", 1.0)
            total_busy_cyc = dist_chain_analysis.get("total_busy_cycles", 0)
            total_busy_us = dist_chain_analysis.get("total_busy_us", 0.0)
            total_mksp_cyc = dist_chain_analysis.get("total_makespan_cycles", 0)
            total_mksp_us = dist_chain_analysis.get("total_makespan_us", 0.0)
            util_pct = dist_chain_analysis.get("utilization_pct", 0.0)
            log_separator()
            log_info("距离链 (FPS+KNN) 关键路径分析（merge-intervals busy time）")
            log_info("  说明: busy_cycles = 合并后非重叠事件区间之和（硬件实际忙碌时间）；makespan = 首事件→末事件时间窗口（含流水间隙）。")
            log_info(f"  总 busy:    {total_busy_cyc:>12,} cycles  ({total_busy_us:.4f} us)")
            log_info(f"  总 makespan:{total_mksp_cyc:>12,} cycles  ({total_mksp_us:.4f} us)")
            log_info(f"  利用率:     {util_pct:.1f}%  (busy / makespan)")
            by_op = dist_chain_analysis.get("by_op") or {}
            if by_op:
                log_info("  ── 按算子类型 + 层名 ──")
                for dc_op in sorted(by_op.keys()):
                    layers_dc = by_op[dc_op]
                    op_total_busy = sum(v.get("busy_cycles", 0) for v in layers_dc.values())
                    op_total_mksp = sum(v.get("makespan_cycles", 0) for v in layers_dc.values())
                    log_info(f"  [{dc_op}] busy={op_total_busy:,} cyc, makespan={op_total_mksp:,} cyc  (跨层累加)")
                    for lname in sorted(layers_dc.keys()):
                        lv = layers_dc[lname]
                        log_info(f"    layer={lname!r:30s}  busy={lv['busy_cycles']:>9,}  makespan={lv['makespan_cycles']:>9,}  busy_ns={lv['busy_ns']:>10.1f}")
            log_separator()

        # 每个算子 cycle 的详细构成：按 (算子类型, 层名) 聚合并打印
        if operator_cycle_detail:
            from collections import defaultdict
            by_op_layer: dict = defaultdict(lambda: {"cycles": 0, "latency_ns": 0.0, "energy_nj": 0.0, "area_um2": 0.0})
            for d in operator_cycle_detail:
                op = d.get("operator", "other")
                layer = d.get("layer_name", "")
                key = (op, layer)
                if not d.get("overlapped", False):
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
        cim_lat = cim_full_context[CIM_LAYER_LATENCY_NS] if cim_full_context else None
        cim_en = cim_full_context[CIM_LAYER_ENERGY_NJ] if cim_full_context else None
        rram_tile_stats = None
        if used_neurosim_for_ppa and neurosim_result:
            rram_tile_stats = getattr(neurosim_result, "rram_tile_stats", None)
        neurosim_summary = None
        conv_linear_summary = None
        if used_neurosim_for_ppa and neurosim_result:
            layer_sum_latency_ns = float(sum(getattr(neurosim_result, "layer_latency_ns", []) or []))
            chip_pipeline_latency_ns_raw = getattr(neurosim_result, "chip_pipeline_latency_ns", None)
            chip_pipeline_buffer_latency_ns = getattr(neurosim_result, "chip_pipeline_buffer_latency_ns", None)
            chip_pipeline_ic_latency_ns = getattr(neurosim_result, "chip_pipeline_ic_latency_ns", None)
            scale_factors = list(getattr(neurosim_result, "workload_scale_factors", []) or [])
            chip_pipeline_exact_for_workload = not any(abs(float(sf) - 1.0) > 1e-9 for sf in scale_factors)
            chip_pipeline_latency_ns = (
                chip_pipeline_latency_ns_raw if chip_pipeline_exact_for_workload else None
            )
            system_e2e_latency_ns = getattr(eval_result, "latency_per_sample_ns", None)
            non_cim_latency_ns = None
            chip_pipeline_plus_non_cim_ns = None
            if system_e2e_latency_ns is not None and system_e2e_latency_ns > 0:
                non_cim_latency_ns = max(0.0, float(system_e2e_latency_ns) - layer_sum_latency_ns)
                if chip_pipeline_latency_ns is not None:
                    chip_pipeline_plus_non_cim_ns = float(chip_pipeline_latency_ns) + non_cim_latency_ns
            neurosim_summary = {
                "layer_sum_latency_ns": layer_sum_latency_ns,
                "layer_sum_latency_ms": layer_sum_latency_ns / 1e6,
                "chip_pipeline_latency_ns": float(chip_pipeline_latency_ns) if chip_pipeline_latency_ns is not None else None,
                "chip_pipeline_latency_ms": (float(chip_pipeline_latency_ns) / 1e6) if chip_pipeline_latency_ns is not None else None,
                "chip_pipeline_latency_raw_ns": float(chip_pipeline_latency_ns_raw) if chip_pipeline_latency_ns_raw is not None else None,
                "chip_pipeline_latency_raw_ms": (float(chip_pipeline_latency_ns_raw) / 1e6) if chip_pipeline_latency_ns_raw is not None else None,
                "chip_pipeline_exact_for_workload": bool(chip_pipeline_exact_for_workload),
                "chip_pipeline_buffer_latency_ns": float(chip_pipeline_buffer_latency_ns) if chip_pipeline_buffer_latency_ns is not None else None,
                "chip_pipeline_ic_latency_ns": float(chip_pipeline_ic_latency_ns) if chip_pipeline_ic_latency_ns is not None else None,
                "non_cim_latency_ns": non_cim_latency_ns,
                "non_cim_latency_ms": (non_cim_latency_ns / 1e6) if non_cim_latency_ns is not None else None,
                "chip_pipeline_plus_non_cim_ns": chip_pipeline_plus_non_cim_ns,
                "chip_pipeline_plus_non_cim_ms": (chip_pipeline_plus_non_cim_ns / 1e6) if chip_pipeline_plus_non_cim_ns is not None else None,
                "layer_sum_over_chip_pipeline": (
                    (layer_sum_latency_ns / float(chip_pipeline_latency_ns))
                    if chip_pipeline_latency_ns is not None and float(chip_pipeline_latency_ns) > 0
                    else None
                ),
                "system_e2e_over_chip_pipeline": (
                    (float(system_e2e_latency_ns) / float(chip_pipeline_latency_ns))
                    if chip_pipeline_latency_ns is not None and system_e2e_latency_ns is not None and float(chip_pipeline_latency_ns) > 0
                    else None
                ),
                "modeling_note": (
                    "chip_pipeline_latency_ns is NeuroSim's pipelined Conv/Linear chip throughput only when the exact workload was simulated. "
                    "If any layer used max_in_vectors capping, chip_pipeline_latency_ns is withheld and chip_pipeline_latency_raw_ns is diagnostic-only; "
                    "system end-to-end latency still sums Conv/Linear events plus non-CIM modules."
                ),
            }
            conv_linear_summary = _build_conv_linear_summary(
                neurosim_result=neurosim_result,
                system_e2e_latency_ns=system_e2e_latency_ns,
            )
        elif full_operator_breakdown:
            conv_linear_summary = _build_fallback_conv_linear_summary(full_operator_breakdown)
        export_options = _build_export_options(args)
        neurosim_layer_breakdown = list(getattr(neurosim_result, "layer_breakdown", []) or []) if (used_neurosim_for_ppa and neurosim_result) else None
        ppa_report = build_ppa_report(
            eval_result,
            clock_ns,
            netstruct=netstruct,
            conv_linear_order=conv_linear_order,
            cim_layer_latency_ns=cim_lat,
            cim_layer_energy_nj=cim_en,
            full_operator_breakdown=full_operator_breakdown,
            checkpoint=checkpoint_path,
            neurosim_full_network=bool(
                used_neurosim_for_ppa
                or (cim_full_context is not None and bool(cim_full_context.get(CIM_FULL_NETWORK_ACTIVE)))
            ),
            rram_tile_stats=rram_tile_stats,
            operator_sequence=(operator_sequence if export_options["embed_operator_sequence_in_report"] else None),
            latency_per_sample_ns=getattr(eval_result, "latency_sum_per_sample_ns", getattr(eval_result, "latency_per_sample_ns", None)),
            critical_path_latency_per_sample_ns=critical_path_latency_ns,
            total_area_um2=total_area_um2,
            event_breakdown=sim_once.get("breakdown") or {},
            activation_shapes=activation_shapes or [],
            neurosim_summary=neurosim_summary,
            neurosim_layer_breakdown=neurosim_layer_breakdown,
            pointcim_distance_summary=sim_once.get("pointcim_distance_summary"),
            dist_chain_analysis=sim_once.get("dist_chain_analysis"),
        )
        ppa_report.setdefault("end_to_end", {})
        ppa_report["end_to_end"]["latency_sum_ns"] = float(getattr(eval_result, "latency_sum_per_sample_ns", latency_per_sample_ns))
        ppa_report["end_to_end"]["latency_sum_ms"] = float(getattr(eval_result, "latency_sum_per_sample_ns", latency_per_sample_ns)) / 1e6
        ppa_report["end_to_end"]["latency_critical_path_ns"] = float(critical_path_latency_ns)
        ppa_report["end_to_end"]["latency_critical_path_ms"] = float(critical_path_latency_ns) / 1e6
        _busy_time_sum_ns = float(sum(v.get("latency_ns", 0) for v in full_operator_breakdown.values()) or 0.0)
        _gap_stall_ns = max(0.0, float(critical_path_latency_ns) - _busy_time_sum_ns)
        ppa_report["end_to_end"]["busy_time_sum_ns"] = _busy_time_sum_ns
        ppa_report["end_to_end"]["busy_time_sum_ms"] = _busy_time_sum_ns / 1e6
        ppa_report["end_to_end"]["unattributed_gap_stall_ns"] = _gap_stall_ns
        ppa_report["end_to_end"]["unattributed_gap_stall_ms"] = _gap_stall_ns / 1e6
        ppa_report["end_to_end"]["modeling_note"] = (
            "latency_sum_* is the sum of event durations on native clocks and is a conservative serial upper bound. "
            "latency_critical_path_* is event-timeline makespan on the unified ns axis "
            "(min(start_ns) -> max(end_ns)) and should be used as end-to-end wall-clock latency. "
            "busy_time_sum_* is the sum of merged per-operator busy intervals; "
            "unattributed_gap_stall_* = latency_critical_path_* - busy_time_sum_*."
        )
        if conv_linear_summary:
            ppa_report["conv_linear_summary"] = conv_linear_summary
        use_auto_dir = args.sim_results_dir == DEFAULT_SIM_RESULTS_DIR
        effective_sim_results_dir = get_sim_mode_results_dir(args) if use_auto_dir else args.sim_results_dir
        sim_results_dir = Path(effective_sim_results_dir)
        sim_results_dir.mkdir(parents=True, exist_ok=True)
        if use_auto_dir:
            log_info(f"PPA 与 breakdown 输出目录: {sim_results_dir}（按仿真模式自动命名）")
        # NeuroSim 模式：将参与计算的层名及 latency/energy/激活与权重 shape 写入 CSV
        if used_neurosim_for_ppa and cim_full_context and conv_linear_list:
            import csv
            import json as _json
            from pointkan_sim.integration.real_data_eval import format_activation_weight_shapes
            lat_ns_list = cim_full_context[CIM_LAYER_LATENCY_NS]
            en_nj_list = cim_full_context[CIM_LAYER_ENERGY_NJ]
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
            layer_breakdown_rows = list(getattr(neurosim_result, "layer_breakdown", []) or []) if neurosim_result else []
            if layer_breakdown_rows:
                neurosim_breakdown_csv_path = sim_results_dir / "neurosim_layer_breakdown.csv"
                neurosim_breakdown_json_path = sim_results_dir / "neurosim_layer_breakdown.json"
                ordered_keys = [
                    "layer_name",
                    "layer_type",
                    "layer_index",
                    "latency_ns",
                    "energy_nj",
                    "buffer_latency_ns",
                    "buffer_energy_nj",
                    "ic_latency_ns",
                    "ic_energy_nj",
                    "core_latency_adc_ns",
                    "core_latency_accum_ns",
                    "core_latency_other_ns",
                    "core_latency_other_visible_ns",
                    "core_latency_other_hidden_ns",
                    "core_latency_other_visible_share_pct",
                    "core_latency_other_hidden_share_pct",
                    "core_energy_adc_nj",
                    "core_energy_accum_nj",
                    "core_energy_other_nj",
                    "core_energy_other_visible_nj",
                    "core_energy_other_hidden_nj",
                    "array_latency_ns",
                    "array_energy_nj",
                    "adc_latency_precharger_ns",
                    "adc_latency_col_delay_ns",
                    "adc_latency_senseamp_ns",
                    "adc_latency_mlsa_ns",
                    "adc_latency_mlencoder_ns",
                    "adc_latency_sar_ns",
                    "adc_energy_precharger_nj",
                    "adc_energy_senseamp_nj",
                    "adc_energy_mlsa_nj",
                    "adc_energy_mlencoder_nj",
                    "adc_energy_sar_nj",
                    "adc_subarray_energy_nj",
                    "adc_precharger_energy_nj",
                    "adc_senseamp_energy_nj",
                    "adc_mlsa_energy_nj",
                    "adc_mlencoder_energy_nj",
                    "adc_sar_energy_nj",
                    "accum_latency_adder_ns",
                    "accum_latency_dff_ns",
                    "accum_latency_shiftadd_weight_ns",
                    "accum_latency_shiftadd_input_ns",
                    "accum_latency_adder_tree_ns",
                    "accum_latency_tile_ns",
                    "accum_energy_adder_nj",
                    "accum_energy_dff_nj",
                    "accum_energy_shiftadd_weight_nj",
                    "accum_energy_shiftadd_input_nj",
                    "accum_energy_adder_tree_nj",
                    "accum_energy_tile_nj",
                    "other_sub_tile_latency_ns",
                    "other_sub_tile_pe_other_latency_ns",
                    "other_sub_tile_buffer_htree_latency_ns",
                    "other_sub_activation_latency_ns",
                    "other_sub_tile_activation_latency_ns",
                    "other_sub_chip_activation_latency_ns",
                    "other_sub_tile_pe_wl_decoder_latency_ns",
                    "other_sub_tile_pe_wl_driver_latency_ns",
                    "other_sub_tile_pe_wl_switch_latency_ns",
                    "other_sub_tile_pe_sl_switch_latency_ns",
                    "other_sub_tile_pe_mux_latency_ns",
                    "other_sub_tile_pe_mux_decoder_latency_ns",
                    "other_sub_tile_pe_level_shifter_latency_ns",
                    "other_sub_tile_pe_wl_path_latency_ns",
                    "other_sub_tile_pe_mux_path_latency_ns",
                    "other_sub_tile_pe_selected_path_latency_ns",
                    "other_sub_tile_pe_other_multiplier",
                    "other_sub_tile_pe_num_col_muxed",
                    "other_sub_tile_pe_num_add",
                    "other_sub_tile_pe_num_row_activity",
                    "other_sub_pooling_latency_ns",
                    "other_sub_global_buffer_latency_ns",
                    "other_sub_global_ic_latency_ns",
                    "other_sub_global_htree_latency_ns",
                    "other_sub_global_xybus_latency_ns",
                    "other_sub_tile_energy_nj",
                    "other_sub_tile_pe_other_energy_nj",
                    "other_sub_tile_buffer_htree_energy_nj",
                    "other_sub_activation_energy_nj",
                    "other_sub_tile_activation_energy_nj",
                    "other_sub_chip_activation_energy_nj",
                    "other_sub_pooling_energy_nj",
                    "other_sub_global_buffer_energy_nj",
                    "other_sub_global_ic_energy_nj",
                    "other_sub_global_htree_energy_nj",
                    "other_sub_global_xybus_energy_nj",
                ]
                enriched_rows = []
                for idx, row in enumerate(layer_breakdown_rows[: len(conv_linear_list)]):
                    info = conv_linear_list[idx]
                    item = {
                        "layer_name": getattr(info, "name", "") or f"layer_{idx}",
                        "layer_type": getattr(info, "layer_type", ""),
                        "layer_index": idx,
                        "latency_ns": float((row or {}).get("read_latency_ns", lat_ns_list[idx] if idx < len(lat_ns_list) else 0.0) or 0.0),
                        "energy_nj": float((row or {}).get("read_energy_nj", en_nj_list[idx] if idx < len(en_nj_list) else 0.0) or 0.0),
                        "buffer_latency_ns": float((row or {}).get("buffer_latency_ns", 0.0) or 0.0),
                        "buffer_energy_nj": float((row or {}).get("buffer_energy_nj", 0.0) or 0.0),
                        "ic_latency_ns": float((row or {}).get("ic_latency_ns", 0.0) or 0.0),
                        "ic_energy_nj": float((row or {}).get("ic_energy_nj", 0.0) or 0.0),
                        "core_latency_adc_ns": float((row or {}).get("core_latency_adc_ns", 0.0) or 0.0),
                        "core_latency_accum_ns": float((row or {}).get("core_latency_accum_ns", 0.0) or 0.0),
                        "core_latency_other_ns": float((row or {}).get("core_latency_other_ns", 0.0) or 0.0),
                        "core_energy_adc_nj": float((row or {}).get("core_energy_adc_nj", 0.0) or 0.0),
                        "core_energy_accum_nj": float((row or {}).get("core_energy_accum_nj", 0.0) or 0.0),
                        "core_energy_other_nj": float((row or {}).get("core_energy_other_nj", 0.0) or 0.0),
                        "array_latency_ns": float((row or {}).get("array_latency_ns", 0.0) or 0.0),
                        "array_energy_nj": float((row or {}).get("array_energy_nj", 0.0) or 0.0),
                        "adc_latency_precharger_ns": float((row or {}).get("adc_latency_precharger_ns", 0.0) or 0.0),
                        "adc_latency_col_delay_ns": float((row or {}).get("adc_latency_col_delay_ns", 0.0) or 0.0),
                        "adc_latency_senseamp_ns": float((row or {}).get("adc_latency_senseamp_ns", 0.0) or 0.0),
                        "adc_latency_mlsa_ns": float((row or {}).get("adc_latency_mlsa_ns", 0.0) or 0.0),
                        "adc_latency_mlencoder_ns": float((row or {}).get("adc_latency_mlencoder_ns", 0.0) or 0.0),
                        "adc_latency_sar_ns": float((row or {}).get("adc_latency_sar_ns", 0.0) or 0.0),
                        "adc_energy_precharger_nj": float((row or {}).get("adc_energy_precharger_nj", (row or {}).get("adc_precharger_energy_nj", 0.0)) or 0.0),
                        "adc_energy_senseamp_nj": float((row or {}).get("adc_energy_senseamp_nj", (row or {}).get("adc_senseamp_energy_nj", 0.0)) or 0.0),
                        "adc_energy_mlsa_nj": float((row or {}).get("adc_energy_mlsa_nj", (row or {}).get("adc_mlsa_energy_nj", 0.0)) or 0.0),
                        "adc_energy_mlencoder_nj": float((row or {}).get("adc_energy_mlencoder_nj", (row or {}).get("adc_mlencoder_energy_nj", 0.0)) or 0.0),
                        "adc_energy_sar_nj": float((row or {}).get("adc_energy_sar_nj", (row or {}).get("adc_sar_energy_nj", 0.0)) or 0.0),
                        "adc_subarray_energy_nj": float((row or {}).get("adc_subarray_energy_nj", 0.0) or 0.0),
                        "adc_precharger_energy_nj": float((row or {}).get("adc_precharger_energy_nj", 0.0) or 0.0),
                        "adc_senseamp_energy_nj": float((row or {}).get("adc_senseamp_energy_nj", 0.0) or 0.0),
                        "adc_mlsa_energy_nj": float((row or {}).get("adc_mlsa_energy_nj", 0.0) or 0.0),
                        "adc_mlencoder_energy_nj": float((row or {}).get("adc_mlencoder_energy_nj", 0.0) or 0.0),
                        "adc_sar_energy_nj": float((row or {}).get("adc_sar_energy_nj", 0.0) or 0.0),
                        "accum_latency_adder_ns": float((row or {}).get("accum_latency_adder_ns", 0.0) or 0.0),
                        "accum_latency_dff_ns": float((row or {}).get("accum_latency_dff_ns", 0.0) or 0.0),
                        "accum_latency_shiftadd_weight_ns": float((row or {}).get("accum_latency_shiftadd_weight_ns", 0.0) or 0.0),
                        "accum_latency_shiftadd_input_ns": float((row or {}).get("accum_latency_shiftadd_input_ns", 0.0) or 0.0),
                        "accum_latency_adder_tree_ns": float((row or {}).get("accum_latency_adder_tree_ns", 0.0) or 0.0),
                        "accum_latency_tile_ns": float((row or {}).get("accum_latency_tile_ns", 0.0) or 0.0),
                        "accum_energy_adder_nj": float((row or {}).get("accum_energy_adder_nj", 0.0) or 0.0),
                        "accum_energy_dff_nj": float((row or {}).get("accum_energy_dff_nj", 0.0) or 0.0),
                        "accum_energy_shiftadd_weight_nj": float((row or {}).get("accum_energy_shiftadd_weight_nj", 0.0) or 0.0),
                        "accum_energy_shiftadd_input_nj": float((row or {}).get("accum_energy_shiftadd_input_nj", 0.0) or 0.0),
                        "accum_energy_adder_tree_nj": float((row or {}).get("accum_energy_adder_tree_nj", 0.0) or 0.0),
                        "accum_energy_tile_nj": float((row or {}).get("accum_energy_tile_nj", 0.0) or 0.0),
                        "other_sub_tile_latency_ns": float((row or {}).get("other_sub_tile_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_other_latency_ns": float((row or {}).get("other_sub_tile_pe_other_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_buffer_htree_latency_ns": float((row or {}).get("other_sub_tile_buffer_htree_latency_ns", 0.0) or 0.0),
                        "other_sub_activation_latency_ns": float((row or {}).get("other_sub_activation_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_activation_latency_ns": float((row or {}).get("other_sub_tile_activation_latency_ns", 0.0) or 0.0),
                        "other_sub_chip_activation_latency_ns": float((row or {}).get("other_sub_chip_activation_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_wl_decoder_latency_ns": float((row or {}).get("other_sub_tile_pe_wl_decoder_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_wl_driver_latency_ns": float((row or {}).get("other_sub_tile_pe_wl_driver_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_wl_switch_latency_ns": float((row or {}).get("other_sub_tile_pe_wl_switch_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_sl_switch_latency_ns": float((row or {}).get("other_sub_tile_pe_sl_switch_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_mux_latency_ns": float((row or {}).get("other_sub_tile_pe_mux_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_mux_decoder_latency_ns": float((row or {}).get("other_sub_tile_pe_mux_decoder_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_level_shifter_latency_ns": float((row or {}).get("other_sub_tile_pe_level_shifter_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_wl_path_latency_ns": float((row or {}).get("other_sub_tile_pe_wl_path_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_mux_path_latency_ns": float((row or {}).get("other_sub_tile_pe_mux_path_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_selected_path_latency_ns": float((row or {}).get("other_sub_tile_pe_selected_path_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_pe_other_multiplier": float((row or {}).get("other_sub_tile_pe_other_multiplier", 0.0) or 0.0),
                        "other_sub_tile_pe_num_col_muxed": float((row or {}).get("other_sub_tile_pe_num_col_muxed", 0.0) or 0.0),
                        "other_sub_tile_pe_num_add": float((row or {}).get("other_sub_tile_pe_num_add", 0.0) or 0.0),
                        "other_sub_tile_pe_num_row_activity": float((row or {}).get("other_sub_tile_pe_num_row_activity", 0.0) or 0.0),
                        "other_sub_pooling_latency_ns": float((row or {}).get("other_sub_pooling_latency_ns", 0.0) or 0.0),
                        "other_sub_global_buffer_latency_ns": float((row or {}).get("other_sub_global_buffer_latency_ns", 0.0) or 0.0),
                        "other_sub_global_ic_latency_ns": float((row or {}).get("other_sub_global_ic_latency_ns", 0.0) or 0.0),
                        "other_sub_global_htree_latency_ns": float((row or {}).get("other_sub_global_htree_latency_ns", 0.0) or 0.0),
                        "other_sub_global_xybus_latency_ns": float((row or {}).get("other_sub_global_xybus_latency_ns", 0.0) or 0.0),
                        "other_sub_tile_energy_nj": float((row or {}).get("other_sub_tile_energy_nj", 0.0) or 0.0),
                        "other_sub_tile_pe_other_energy_nj": float((row or {}).get("other_sub_tile_pe_other_energy_nj", 0.0) or 0.0),
                        "other_sub_tile_buffer_htree_energy_nj": float((row or {}).get("other_sub_tile_buffer_htree_energy_nj", 0.0) or 0.0),
                        "other_sub_activation_energy_nj": float((row or {}).get("other_sub_activation_energy_nj", 0.0) or 0.0),
                        "other_sub_tile_activation_energy_nj": float((row or {}).get("other_sub_tile_activation_energy_nj", 0.0) or 0.0),
                        "other_sub_chip_activation_energy_nj": float((row or {}).get("other_sub_chip_activation_energy_nj", 0.0) or 0.0),
                        "other_sub_pooling_energy_nj": float((row or {}).get("other_sub_pooling_energy_nj", 0.0) or 0.0),
                        "other_sub_global_buffer_energy_nj": float((row or {}).get("other_sub_global_buffer_energy_nj", 0.0) or 0.0),
                        "other_sub_global_ic_energy_nj": float((row or {}).get("other_sub_global_ic_energy_nj", 0.0) or 0.0),
                        "other_sub_global_htree_energy_nj": float((row or {}).get("other_sub_global_htree_energy_nj", 0.0) or 0.0),
                        "other_sub_global_xybus_energy_nj": float((row or {}).get("other_sub_global_xybus_energy_nj", 0.0) or 0.0),
                    }
                    item["core_latency_other_visible_ns"] = item["buffer_latency_ns"] + item["ic_latency_ns"]
                    item["core_latency_other_hidden_ns"] = max(
                        item["core_latency_other_ns"] - item["core_latency_other_visible_ns"],
                        0.0,
                    )
                    if item["core_latency_other_ns"] > 0:
                        item["core_latency_other_visible_share_pct"] = (
                            item["core_latency_other_visible_ns"] / item["core_latency_other_ns"] * 100.0
                        )
                        item["core_latency_other_hidden_share_pct"] = (
                            item["core_latency_other_hidden_ns"] / item["core_latency_other_ns"] * 100.0
                        )
                    else:
                        item["core_latency_other_visible_share_pct"] = 0.0
                        item["core_latency_other_hidden_share_pct"] = 0.0
                    item["core_energy_other_visible_nj"] = item["buffer_energy_nj"] + item["ic_energy_nj"]
                    item["core_energy_other_hidden_nj"] = max(
                        item["core_energy_other_nj"] - item["core_energy_other_visible_nj"],
                        0.0,
                    )
                    enriched_rows.append(item)
                with open(neurosim_breakdown_csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=ordered_keys)
                    w.writeheader()
                    for item in enriched_rows:
                        w.writerow(item)
                with open(neurosim_breakdown_json_path, "w", encoding="utf-8") as f:
                    _json.dump(enriched_rows, f, indent=2, ensure_ascii=False)
                log_info(f"NeuroSim 逐层细粒度 breakdown 已写入: {neurosim_breakdown_csv_path}")
                log_info(f"NeuroSim 逐层细粒度 breakdown JSON已写入: {neurosim_breakdown_json_path}")
            if neurosim_summary:
                chip_pipeline_ms = neurosim_summary.get("chip_pipeline_latency_ms")
                layer_sum_ms = neurosim_summary.get("layer_sum_latency_ms")
                pipeline_plus_non_cim_ms = neurosim_summary.get("chip_pipeline_plus_non_cim_ms")
                if layer_sum_ms is not None:
                    log_info(f"NeuroSim Conv/Linear 逐层累加延迟: {float(layer_sum_ms):.6f} ms")
                if chip_pipeline_ms is not None:
                    log_info(f"NeuroSim Chip pipeline 吞吐延迟: {float(chip_pipeline_ms):.6f} ms")
                if pipeline_plus_non_cim_ms is not None:
                    log_info(f"NeuroSim Chip pipeline + 非CIM模块 估计端到端: {float(pipeline_plus_non_cim_ms):.6f} ms")
        report_filename = os.path.basename(args.output) if args.output else "report.json"
        write_ppa_report_and_plots(
            ppa_report,
            sim_results_dir,
            report_filename=report_filename,
            log_info_fn=log_info,
        )
        breakdown_detail_path = sim_results_dir / "breakdown_detail.json"
        with open(breakdown_detail_path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(sim_once.get("breakdown") or {}, f, indent=2, ensure_ascii=False)
        log_info(f"breakdown 明细已写入: {breakdown_detail_path}")
        sim_result = sim_once.get("sim_result") if isinstance(sim_once, dict) else None
        block_intervals = getattr(sim_result, "block_intervals", None) if sim_result is not None else None
        sim_events = list(getattr(sim_result, "events", []) or []) if sim_result is not None else []
        if export_options["export_timelines"] and block_intervals:
            from pointkan_sim.analysis.timeline import block_intervals_to_json_rows, build_block_intervals, render_ascii_timeline

            def _write_filtered_timeline(name: str, events_subset, *, start_from_first: bool = True, max_cycle: int = 500):
                if not events_subset:
                    return
                start_cycle = None
                if start_from_first:
                    start_cycle = min(int(getattr(e, "cycle", 0) or 0) for e in events_subset)
                intervals = build_block_intervals(
                    events_subset,
                    detailed=True,
                    start_cycle=start_cycle,
                    end_cycle=(None if start_cycle is None else start_cycle + max_cycle),
                )
                if not intervals:
                    return
                json_path = sim_results_dir / f"block_intervals_{name}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(block_intervals_to_json_rows(intervals))
                txt_path = sim_results_dir / f"cycle_timeline_{name}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(render_ascii_timeline(intervals, max_cycle=max_cycle))
                log_info(f"{name} 文本版 timeline 已写入: {txt_path}")

            timeline_json_path = sim_results_dir / "block_intervals.json"
            with open(timeline_json_path, "w", encoding="utf-8") as f:
                f.write(block_intervals_to_json_rows(block_intervals))
            log_info(f"Block intervals JSON已写入: {timeline_json_path}")
            global_ascii = render_ascii_timeline(block_intervals, max_cycle=500)
            global_txt_path = sim_results_dir / "cycle_timeline.txt"
            with open(global_txt_path, "w", encoding="utf-8") as f:
                f.write(global_ascii)
            log_info(f"文本版 Cycle timeline 已写入: {global_txt_path}")
            if sim_events:
                gam_stage0_events = [e for e in sim_events if str(getattr(e, 'layer_name', '') or ((getattr(e, 'metadata', None) or {}).get('layer_name', '')) or '') == 'GAM-stage0']
                if gam_stage0_events:
                    _cleanup_optional_gam_stage0_timeline_artifacts(sim_results_dir, log_info)
                    gam_start = min(int(getattr(e, 'cycle', 0) or 0) for e in gam_stage0_events)
                    gam_intervals = build_block_intervals(
                        gam_stage0_events,
                        detailed=True,
                        layer_filter='GAM-stage0',
                        start_cycle=gam_start,
                        end_cycle=gam_start + 500,
                    )
                    if gam_intervals:
                        gam_json_path = sim_results_dir / "block_intervals_gam_stage0.json"
                        with open(gam_json_path, "w", encoding="utf-8") as f:
                            f.write(block_intervals_to_json_rows(gam_intervals))
                        log_info(f"GAM-stage0 block intervals JSON已写入: {gam_json_path}")
                        gam_ascii = render_ascii_timeline(gam_intervals, max_cycle=500)
                        gam_txt_path = sim_results_dir / "cycle_timeline_gam_stage0.txt"
                        with open(gam_txt_path, "w", encoding="utf-8") as f:
                            f.write(gam_ascii)
                        log_info(f"GAM-stage0 文本版 timeline 已写入: {gam_txt_path}")
                        core_events = [
                            e for e in gam_stage0_events
                            if str(getattr(e, 'operator', '') or '') in {'FPS', 'KNN', 'Resampler', 'SpecPredictor'}
                        ]
                        if core_events:
                            core_start = min(int(getattr(e, 'cycle', 0) or 0) for e in core_events)
                            core_intervals = build_block_intervals(
                                gam_stage0_events,
                                detailed=True,
                                layer_filter='GAM-stage0',
                                start_cycle=core_start,
                                end_cycle=core_start + 500,
                            )
                            if core_intervals:
                                core_json_path = sim_results_dir / 'block_intervals_gam_stage0_core.json'
                                with open(core_json_path, 'w', encoding='utf-8') as f:
                                    f.write(block_intervals_to_json_rows(core_intervals))
                                log_info(f"GAM-stage0 core block intervals JSON已写入: {core_json_path}")
                                core_ascii = render_ascii_timeline(core_intervals, max_cycle=500)
                                core_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_core.txt'
                                with open(core_txt_path, 'w', encoding='utf-8') as f:
                                    f.write(core_ascii)
                                log_info(f"GAM-stage0 core 文本版 timeline 已写入: {core_txt_path}")
                                warmup_generated = False
                                post_generated = False
                                switch_generated = False
                                compute_ops = {
                                    'self_distance_read',
                                    'cross_term_distance',
                                    'distance_merge_accumulate',
                                    'fps_min_distance_table_read',
                                    'min_distance_compare',
                                    'fps_min_distance_table_write',
                                    'fps_warmup_coarse_top32',
                                    'fps_warmup_fine_max',
                                    'key_xyz_buffer_write',
                                    'key_xyz_buffer_read',
                                    'rram_program_keyset',
                                    'rram_self_k2_init',
                                    'self_distance_write',
                                    'rram_cross_term_distance',
                                    'query_self_q',
                                    'knn_coarse_filter',
                                    'topk_candidate_write',
                                    'knn_threshold_update',
                                    'knn_threshold_refine',
                                    'topk_candidate_refine_write',
                                    'knn_fine_filter_topk',
                                    'fps_coarse_filter_max',
                                    'temp_max_candidate_append',
                                    'fps_tempmax_reduce_read',
                                    'fps_tempmax_reduce_write',
                                    'fps_tempmax_global_read',
                                    'fps_fine_filter_max',
                                    'topk_candidate_aggregate_read',
                                    'topk_candidate_fine_read',
                                }
                                compute_events = [
                                    e for e in gam_stage0_events
                                    if str(getattr(e, 'operation', '') or '') in compute_ops
                                ]
                                if compute_events:
                                    compute_start = min(int(getattr(e, 'cycle', 0) or 0) for e in compute_events)
                                    compute_intervals = build_block_intervals(
                                        compute_events,
                                        detailed=True,
                                        layer_filter='GAM-stage0',
                                        start_cycle=compute_start,
                                        end_cycle=compute_start + 500,
                                    )
                                    if compute_intervals:
                                        compute_json_path = sim_results_dir / 'block_intervals_gam_stage0_compute.json'
                                        with open(compute_json_path, 'w', encoding='utf-8') as f:
                                            f.write(block_intervals_to_json_rows(compute_intervals))
                                        log_info(f"GAM-stage0 compute block intervals JSON已写入: {compute_json_path}")
                                        compute_ascii = render_ascii_timeline(compute_intervals, max_cycle=500)
                                        compute_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_compute.txt'
                                        with open(compute_txt_path, 'w', encoding='utf-8') as f:
                                            f.write(compute_ascii)
                                        log_info(f"GAM-stage0 compute 文本版 timeline 已写入: {compute_txt_path}")

                                    warmup_events = [
                                        e for e in compute_events
                                        if str((getattr(e, 'metadata', None) or {}).get('phase', '') or '') == 'warmup_full_sort'
                                        or str(getattr(e, 'operation', '') or '') in {
                                            'key_xyz_buffer_write',
                                            'key_xyz_buffer_read',
                                            'rram_program_keyset',
                                            'rram_self_k2_init',
                                            'self_distance_write',
                                            'query_xyz_buffer_write',
                                            'query_xyz_buffer_read',
                                            'self_distance_read',
                                            'cross_term_distance',
                                            'rram_cross_term_distance',
                                            'query_self_q',
                                            'distance_merge_accumulate',
                                            'distance_write',
                                        }
                                    ]
                                    if warmup_events:
                                        warmup_start = min(int(getattr(e, 'cycle', 0) or 0) for e in warmup_events)
                                        warmup_intervals = build_block_intervals(
                                            warmup_events,
                                            detailed=True,
                                            layer_filter='GAM-stage0',
                                            start_cycle=warmup_start,
                                            end_cycle=warmup_start + 500,
                                        )
                                        if warmup_intervals:
                                            warmup_json_path = sim_results_dir / 'block_intervals_gam_stage0_warmup.json'
                                            with open(warmup_json_path, 'w', encoding='utf-8') as f:
                                                f.write(block_intervals_to_json_rows(warmup_intervals))
                                            warmup_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_warmup.txt'
                                            with open(warmup_txt_path, 'w', encoding='utf-8') as f:
                                                f.write(render_ascii_timeline(warmup_intervals, max_cycle=500))
                                            log_info(f"GAM-stage0 warmup 文本版 timeline 已写入: {warmup_txt_path}")
                                            warmup_generated = True

                                    post_events = [
                                        e for e in compute_events
                                        if str((getattr(e, 'metadata', None) or {}).get('phase', '') or '') == 'post_predict_coarse_fine'
                                    ]
                                    if post_events:
                                        post_start = min(int(getattr(e, 'cycle', 0) or 0) for e in post_events)
                                        post_intervals = build_block_intervals(
                                            post_events,
                                            detailed=True,
                                            layer_filter='GAM-stage0',
                                            start_cycle=post_start,
                                            end_cycle=post_start + 500,
                                        )
                                        if post_intervals:
                                            post_json_path = sim_results_dir / 'block_intervals_gam_stage0_postpredict.json'
                                            with open(post_json_path, 'w', encoding='utf-8') as f:
                                                f.write(block_intervals_to_json_rows(post_intervals))
                                            post_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_postpredict.txt'
                                            with open(post_txt_path, 'w', encoding='utf-8') as f:
                                                f.write(render_ascii_timeline(post_intervals, max_cycle=500))
                                            log_info(f"GAM-stage0 post-predict 文本版 timeline 已写入: {post_txt_path}")
                                            post_generated = True

                                    speculative_events = [
                                        e for e in gam_stage0_events
                                        if str((getattr(e, 'metadata', None) or {}).get('phase', '') or '') in {'speculative_overlap', 'predictor_gate'}
                                        or str(getattr(e, 'operator', '') or '') in {'Resampler', 'SpecPredictor'}
                                    ]
                                    if speculative_events:
                                        spec_start = compute_start
                                        spec_intervals = build_block_intervals(
                                            speculative_events,
                                            detailed=True,
                                            layer_filter='GAM-stage0',
                                            start_cycle=spec_start,
                                            end_cycle=spec_start + 5000,
                                        )
                                        if spec_intervals:
                                            spec_json_path = sim_results_dir / 'block_intervals_gam_stage0_speculative.json'
                                            with open(spec_json_path, 'w', encoding='utf-8') as f:
                                                f.write(block_intervals_to_json_rows(spec_intervals))
                                            spec_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_speculative.txt'
                                            with open(spec_txt_path, 'w', encoding='utf-8') as f:
                                                f.write(render_ascii_timeline(spec_intervals, max_cycle=5000, label_width=72))
                                            log_info(f"GAM-stage0 speculative 文本版 timeline 已写入: {spec_txt_path}")
                                            down_rows = ["out_idx,src_pos,left,right,w,ready_query"]
                                            seen_outputs = set()
                                            for e in speculative_events:
                                                meta = getattr(e, 'metadata', None) or {}
                                                if str(getattr(e, 'operation', '') or '') != 'resampler_downsample_step':
                                                    continue
                                                out_idx = meta.get('output_idx')
                                                if out_idx is None or out_idx in seen_outputs:
                                                    continue
                                                seen_outputs.add(out_idx)
                                                src_pos = meta.get('src_pos', '')
                                                left = meta.get('pair_left', '')
                                                right = meta.get('pair_right', '')
                                                w = meta.get('interp_weight', '')
                                                ready_q = meta.get('ready_query', meta.get('query_index', ''))
                                                down_rows.append(f"{out_idx},{src_pos},{left},{right},{w},{ready_q}")
                                            if len(down_rows) > 1:
                                                down_csv_path = sim_results_dir / 'gam_stage0_downsample_schedule.csv'
                                                with open(down_csv_path, 'w', encoding='utf-8') as f:
                                                    f.write("\n".join(down_rows) + "\n")
                                                log_info(f"GAM-stage0 downsample schedule 已写入: {down_csv_path}")
                                            up_rows = ["out_idx,src_pos,left,right,w,target_query,emit_query"]
                                            seen_up = set()
                                            for e in speculative_events:
                                                meta = getattr(e, 'metadata', None) or {}
                                                if str(getattr(e, 'operation', '') or '') != 'resampler_upsample_step':
                                                    continue
                                                out_idx = meta.get('output_idx')
                                                if out_idx is None or out_idx in seen_up:
                                                    continue
                                                seen_up.add(out_idx)
                                                up_rows.append(
                                                    f"{out_idx},{meta.get('src_pos','')},{meta.get('pair_left','')},{meta.get('pair_right','')},"
                                                    f"{meta.get('interp_weight','')},{meta.get('target_query','')},{meta.get('emit_query_index','')}"
                                                )
                                            if len(up_rows) > 1:
                                                up_csv_path = sim_results_dir / 'gam_stage0_upsample_schedule.csv'
                                                with open(up_csv_path, 'w', encoding='utf-8') as f:
                                                    f.write("\n".join(up_rows) + "\n")
                                                log_info(f"GAM-stage0 upsample schedule 已写入: {up_csv_path}")

                                    if speculative_events and post_events:
                                        switch_anchor = compute_start
                                        switch_events = [
                                            e for e in gam_stage0_events
                                            if str((getattr(e, 'metadata', None) or {}).get('phase', '') or '') in {
                                                'speculative_overlap',
                                                'predictor_gate',
                                                'post_predict_coarse_fine',
                                            }
                                            or str(getattr(e, 'operator', '') or '') in {'Resampler', 'SpecPredictor', 'FPS'}
                                            or str(getattr(e, 'operation', '') or '') in {
                                                'fps_coarse_filter_max',
                                                'temp_max_candidate_append',
                                                'fps_fine_filter_max',
                                            }
                                        ]
                                        switch_intervals = build_block_intervals(
                                            switch_events,
                                            detailed=True,
                                            layer_filter='GAM-stage0',
                                            start_cycle=switch_anchor,
                                            end_cycle=switch_anchor + 5000,
                                        )
                                        if switch_intervals:
                                            switch_json_path = sim_results_dir / 'block_intervals_gam_stage0_switch.json'
                                            with open(switch_json_path, 'w', encoding='utf-8') as f:
                                                f.write(block_intervals_to_json_rows(switch_intervals))
                                            switch_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_switch.txt'
                                            with open(switch_txt_path, 'w', encoding='utf-8') as f:
                                                f.write(render_ascii_timeline(switch_intervals, max_cycle=5000, label_width=72))
                                            log_info(f"GAM-stage0 switch 文本版 timeline 已写入: {switch_txt_path}")
                                            switch_generated = True

                                    predictor_switch_events = [
                                        e for e in gam_stage0_events
                                        if str((getattr(e, 'metadata', None) or {}).get('phase', '') or '') in {
                                            'warmup_full_sort',
                                            'speculative_overlap',
                                            'predictor_gate',
                                            'post_predict_coarse_fine',
                                        }
                                        or str(getattr(e, 'operator', '') or '') in {'Resampler', 'SpecPredictor', 'FPS'}
                                        or str(getattr(e, 'operation', '') or '') in {
                                            'fps_prefix_sample_write',
                                            'resampler_downsample_lut',
                                            'resampler_downsample_step',
                                            'predictor_fc1_linear',
                                            'predictor_kat',
                                            'predictor_fc2_linear',
                                            'resampler_upsample_lut',
                                            'resampler_upsample_step',
                                            'fps_coarse_filter_max',
                                            'temp_max_candidate_append',
                                            'fps_tempmax_global_read',
                                            'fps_fine_filter_max',
                                        }
                                    ]
                                    if predictor_switch_events:
                                        predictor_anchor_candidates = [
                                            int(getattr(e, 'cycle', 0) or 0)
                                            for e in predictor_switch_events
                                            if str(getattr(e, 'operation', '') or '') in {
                                                'predictor_fc1_linear',
                                                'predictor_kat',
                                                'predictor_fc2_linear',
                                            }
                                        ]
                                        first_post_candidates = [
                                            int(getattr(e, 'cycle', 0) or 0)
                                            for e in predictor_switch_events
                                            if str(getattr(e, 'operation', '') or '') in {
                                                'fps_coarse_filter_max',
                                                'temp_max_candidate_append',
                                                'fps_fine_filter_max',
                                            }
                                        ]
                                        if predictor_anchor_candidates:
                                            local_anchor = max(
                                                compute_start,
                                                min(predictor_anchor_candidates) - 300,
                                            )
                                        elif first_post_candidates:
                                            local_anchor = max(
                                                compute_start,
                                                min(first_post_candidates) - 500,
                                            )
                                        else:
                                            local_anchor = compute_start
                                        local_end = local_anchor + 1800
                                        local_intervals = build_block_intervals(
                                            predictor_switch_events,
                                            detailed=True,
                                            layer_filter='GAM-stage0',
                                            start_cycle=local_anchor,
                                            end_cycle=local_end,
                                        )
                                        if local_intervals:
                                            local_json_path = sim_results_dir / 'block_intervals_gam_stage0_predictor_switch_local.json'
                                            with open(local_json_path, 'w', encoding='utf-8') as f:
                                                f.write(block_intervals_to_json_rows(local_intervals))
                                            local_txt_path = sim_results_dir / 'cycle_timeline_gam_stage0_predictor_switch_local.txt'
                                            with open(local_txt_path, 'w', encoding='utf-8') as f:
                                                f.write(render_ascii_timeline(local_intervals, max_cycle=local_end - local_anchor, label_width=72))
                                            log_info(f"GAM-stage0 predictor/switch 局部 timeline 已写入: {local_txt_path}")
                                _log_gam_stage0_timeline_status(log_info, "warmup", warmup_generated)
                                _log_gam_stage0_timeline_status(log_info, "post-predict", post_generated)
                                _log_gam_stage0_timeline_status(log_info, "switch", switch_generated)

            if sim_events:
                feature_events = [
                    e for e in sim_events
                    if str(getattr(e, "operator", "") or "") in {"LayerBuffer", "SRAM", "Quant", "Dequant", "ADD", "ReLU", "AffineNorm", "LayerNorm", "KAN"}
                    or str(getattr(e, "circuit_module", "") or "") in {"feature_sram_buffer", "residual_sram_buffer", "nonlinear_workspace_sram", "nonlinear_param_sram", "rram_lut", "rram_lut_speculative"}
                    or str((getattr(e, "metadata", None) or {}).get("buffer_name", "") or "") in {"feature_sram_buffer", "residual_sram_buffer", "nonlinear_workspace_sram", "nonlinear_param_sram"}
                ]
                _write_filtered_timeline("feature_pipeline", feature_events, max_cycle=500)

                conv_linear_events = [
                    e for e in sim_events
                    if str(getattr(e, "operator", "") or "") in {"Conv1d", "Linear", "Quant", "Dequant"}
                ]
                _write_filtered_timeline("conv_linear", conv_linear_events, max_cycle=500)

                nonlinear_events = [
                    e for e in sim_events
                    if str(getattr(e, "operator", "") or "") in {"ReLU", "AffineNorm", "LayerNorm", "KAN", "ADD"}
                    or str(getattr(e, "circuit_module", "") or "") in {"rram_lut", "rram_lut_speculative", "nonlinear_workspace_sram", "nonlinear_param_sram"}
                ]
                _write_filtered_timeline("nonlinear", nonlinear_events, max_cycle=500)
        # 按执行顺序输出完整硬件算子序列到 operator_sequence.txt（列与 constants 键名一致）
        if export_options["export_operator_sequence_files"] and operator_sequence:
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
                base_fields = ["idx", "cycle", "end_cycle", "duration", "operator", "stage", "operation", "layer_name", "energy_nj"]
                fieldnames = list(base_fields)
                seen = set(fieldnames)
                for item in operator_sequence:
                    for key in item.keys():
                        if key not in seen:
                            seen.add(key)
                            fieldnames.append(key)
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for item in operator_sequence:
                    w.writerow(item)
            seq_detailed_csv_path = sim_results_dir / "operator_sequence_detailed.csv"
            with open(seq_detailed_csv_path, "w", newline="", encoding="utf-8") as f:
                base_fields = [
                    "idx", "cycle", "end_cycle", "duration", "operator", "stage", "sub_stage",
                    "operation", "circuit_module", "layer_name", "energy_nj", "latency_ns",
                    "power_mw", "clock_period_ns", "overlapped",
                ]
                fieldnames = list(base_fields)
                seen = set(fieldnames)
                for item in operator_sequence:
                    for key in item.keys():
                        if key not in seen:
                            seen.add(key)
                            fieldnames.append(key)
                w = _csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for item in operator_sequence:
                    w.writerow(item)
            log_info(f"完整硬件算子序列已写入: {seq_path}（共 {len(operator_sequence)} 条）")
            log_info(f"完整硬件算子序列CSV已写入: {seq_csv_path}")
            log_info(f"完整硬件算子详细CSV已写入: {seq_detailed_csv_path}")

        # 写入每个算子 cycle 的详细构成（按算子+层名聚合）
        if operator_cycle_detail:
            import csv as _csv
            from collections import defaultdict
            by_op_layer_file = defaultdict(lambda: {"cycles": 0, "latency_ns": 0.0, "energy_nj": 0.0, "area_um2": 0.0, "activation_input_shape": "", "activation_output_shape": ""})
            for d in operator_cycle_detail:
                op = d.get("operator", "other")
                layer = d.get("layer_name", "")
                key = (op, layer)
                if not d.get("overlapped", False):
                    by_op_layer_file[key]["cycles"] += d.get("duration", 0)
                    by_op_layer_file[key]["latency_ns"] += d.get("latency_ns", 0.0)
                by_op_layer_file[key]["energy_nj"] += d.get("energy_nj", 0.0)
                by_op_layer_file[key]["area_um2"] += d.get("area_um2", 0.0)
                if not by_op_layer_file[key]["activation_input_shape"] and d.get("activation_input_shape"):
                    by_op_layer_file[key]["activation_input_shape"] = str(d.get("activation_input_shape", ""))
                if not by_op_layer_file[key]["activation_output_shape"] and d.get("activation_output_shape"):
                    by_op_layer_file[key]["activation_output_shape"] = str(d.get("activation_output_shape", ""))
            by_op_file = defaultdict(list)
            for (op, layer), v in by_op_layer_file.items():
                by_op_file[op].append((layer, v["cycles"], v["latency_ns"], v["energy_nj"], v["area_um2"], v["activation_input_shape"], v["activation_output_shape"]))
            detail_path = sim_results_dir / "operator_cycle_detail.txt"
            with open(detail_path, "w", encoding="utf-8") as f:
                f.write("# 每个算子 cycle 的详细构成（按算子类型 + 层名聚合）\n")
                f.write("# operator\tlayer_name\tcycles\tlatency_ns\tenergy_nJ\tarea_um2\tactivation_input_shape\tactivation_output_shape\n")
                for op_name in sorted(by_op_file.keys(), key=lambda o: (-sum(x[1] for x in by_op_file[o]), o)):
                    for layer_name, cy, lat_ns, en, area_u, act_in, act_out in sorted(by_op_file[op_name], key=lambda x: -x[1]):
                        f.write(f"{op_name}\t{layer_name}\t{cy:.0f}\t{lat_ns:.2f}\t{en:.2f}\t{area_u:.2f}\t{act_in}\t{act_out}\n")
            detail_csv_path = sim_results_dir / "operator_cycle_detail.csv"
            with open(detail_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["operator", "layer_name", "cycles", "latency_ns", "energy_nj", "area_um2", "activation_input_shape", "activation_output_shape"])
                for op_name in sorted(by_op_file.keys(), key=lambda o: (-sum(x[1] for x in by_op_file[o]), o)):
                    for layer_name, cy, lat_ns, en, area_u, act_in, act_out in sorted(by_op_file[op_name], key=lambda x: -x[1]):
                        w.writerow([op_name, layer_name, f"{cy:.0f}", f"{lat_ns:.2f}", f"{en:.2f}", f"{area_u:.2f}", act_in, act_out])
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
                            streaming_feature=row.get("streaming_feature", False),
                            retained_residual=row.get("retained_residual", False),
                            note=row.get("note", ""),
                        )
                    )
            residency_csv_path = sim_results_dir / "feature_residency_plan.csv"
            with open(residency_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(["layer_name", "layer_type", "input_bytes", "input_sram_bytes", "input_dram_bytes", "output_bytes", "output_sram_bytes", "output_dram_bytes", "usable_buffer_bytes", "total_buffer_bytes", "buffer_slots", "pipeline_mode", "double_buffering", "streaming_feature", "retained_residual", "retained_residual_bytes", "streaming_usable_bytes", "note"])
                for row in feature_residency_plan:
                    w.writerow([row.get("layer_name", ""), row.get("layer_type", ""), row.get("input_bytes", 0), row.get("input_sram_bytes", 0), row.get("input_dram_bytes", 0), row.get("output_bytes", 0), row.get("output_sram_bytes", 0), row.get("output_dram_bytes", 0), row.get("usable_buffer_bytes", 0), row.get("total_buffer_bytes", 0), row.get("buffer_slots", 0), row.get("pipeline_mode", ""), row.get("double_buffering", False), row.get("streaming_feature", False), row.get("retained_residual", False), row.get("retained_residual_bytes", 0), row.get("streaming_usable_bytes", 0), row.get("note", "")])
            log_info(f"Feature SRAM residency plan 已写入: {residency_txt_path}")
            log_info(f"Feature SRAM residency plan CSV已写入: {residency_csv_path}")

        scalesim_layer_details = sim_once.get("scalesim_layer_details") or []
        if scalesim_layer_details:
            import csv as _csv

            trace_output_root = sim_results_dir / "scalesim_traces"
            trace_output_root.mkdir(parents=True, exist_ok=True)

            def _localize_trace_dirs(raw_value: Any) -> str:
                if not raw_value:
                    return ""
                localized_paths = []
                seen: set[str] = set()
                for piece in str(raw_value).split(";"):
                    src = piece.strip()
                    if not src:
                        continue
                    src_path = Path(src)
                    if not src_path.exists() or not src_path.is_dir():
                        localized_paths.append(src)
                        continue
                    key = str(src_path.resolve())
                    dst_path = trace_output_root / src_path.name
                    if key not in seen:
                        if dst_path.exists():
                            shutil.rmtree(dst_path)
                        shutil.copytree(src_path, dst_path)
                        seen.add(key)
                    localized_paths.append(str(dst_path))
                return ";".join(localized_paths)

            localized_scalesim_rows = []
            for row in scalesim_layer_details:
                localized_row = dict(row)
                localized_row["dram_trace_dir"] = _localize_trace_dirs(row.get("dram_trace_dir", ""))
                localized_row["feature_input_dram_trace_dir"] = _localize_trace_dirs(row.get("feature_input_dram_trace_dir", ""))
                localized_row["weight_dram_trace_dir"] = _localize_trace_dirs(row.get("weight_dram_trace_dir", ""))
                localized_row["feature_output_dram_trace_dir"] = _localize_trace_dirs(row.get("feature_output_dram_trace_dir", ""))
                localized_scalesim_rows.append(localized_row)

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
                for row in localized_scalesim_rows:
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

            shapes_csv_path = sim_results_dir / "scalesim_layer_shapes.csv"
            with open(shapes_csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow([
                    "layer_name", "operator", "activation_input_shape", "activation_output_shape",
                    "num_points", "tokens", "in_channels", "out_channels",
                    "in_features", "out_features", "kernel_size",
                    "systolic_array_rows", "systolic_array_cols",
                ])
                for row in localized_scalesim_rows:
                    w.writerow([
                        row.get("layer_name", ""),
                        row.get("operator", ""),
                        row.get("activation_input_shape", ""),
                        row.get("activation_output_shape", ""),
                        row.get("num_points", ""),
                        row.get("tokens", ""),
                        row.get("in_channels", ""),
                        row.get("out_channels", ""),
                        row.get("in_features", ""),
                        row.get("out_features", ""),
                        row.get("kernel_size", ""),
                        row.get("systolic_array_rows", ""),
                        row.get("systolic_array_cols", ""),
                    ])
            log_info(f"SCALE-Sim 每层输入输出 shape 汇总已写入: {shapes_csv_path}")

        # 可选：导出内存交互事务（统一接口 src/dst/kind），并写聚合统计
        if export_options["export_memory_transactions"] and operator_sequence:
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

        runtime_memory_rows = _collect_runtime_memory_rows(sim_once) if export_options["export_runtime_memory_events"] else []
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
            contract_rows = list(iter_memory_contract_rows(mem_contract))
            relationship_rows = list(iter_buffer_relationship_rows(mem_contract))
            buffer_requirement_rows = list(iter_buffer_requirement_rows(mem_contract))
            contract_txt_path = sim_results_dir / "memory_contract.txt"
            with open(contract_txt_path, "w", encoding="utf-8") as f:
                f.write(format_memory_contract_report(mem_contract))

            contract_csv_path = sim_results_dir / "memory_contract.csv"
            import csv
            with open(contract_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["layer_name", "op_name", "tensor_name", "src_tier", "src_name", "dst_tier", "dst_name", "src", "dst", "bytes", "mib", "estimated_energy_nj", "estimated_energy_model", "estimated_energy_scope", "note"])
                for row in contract_rows:
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
                for row in relationship_rows:
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
                for row in buffer_requirement_rows:
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

            tracked_buffers = [
                "feature_sram_buffer",
                "residual_sram_buffer",
                "xyz_sram_buffer",
                "query_xyz_sram_buffer",
                "fps_index_sram_buffer",
                "dist_sram_buffer",
                "fps_min_distance_table_sram",
                "temp_max_distance_sram",
                "topk_sram_buffer",
                "sort_cache_sram",
            ]
            runtime_by_buffer = {}
            for row in runtime_memory_rows or []:
                buf = str(row.get("buffer_name", "") or "")
                if not buf:
                    continue
                stat = runtime_by_buffer.setdefault(buf, {"events": 0, "bytes": 0, "energy_nj": 0.0})
                stat["events"] += 1
                stat["bytes"] += int(row.get("bytes", 0) or 0)
                stat["energy_nj"] += float(row.get("energy_nj", 0.0) or 0.0)
            contract_refs = {}
            for row in contract_rows:
                for field in ("src_name", "dst_name"):
                    buf = str(row.get(field, "") or "")
                    if not buf:
                        continue
                    stat = contract_refs.setdefault(buf, {"rows": 0, "bytes": 0})
                    stat["rows"] += 1
                    stat["bytes"] += int(row.get("bytes", 0) or 0)
            peak_by_buffer = {}
            for row in buffer_requirement_rows:
                buf = str(row.get("buffer_name", "") or "")
                if not buf:
                    continue
                peak_by_buffer[buf] = max(int(row.get("peak_bytes", 0) or 0), peak_by_buffer.get(buf, 0))

            usage_csv_path = sim_results_dir / "memory_buffer_usage_summary.csv"
            with open(usage_csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["buffer_name", "contract_rows", "contract_total_bytes", "peak_required_bytes", "runtime_events", "runtime_bytes", "runtime_energy_nj", "active_in_run"])
                for buf in tracked_buffers:
                    c = contract_refs.get(buf, {})
                    r = runtime_by_buffer.get(buf, {})
                    peak = peak_by_buffer.get(buf, 0)
                    active = bool(c.get("rows", 0) or r.get("events", 0) or peak)
                    w.writerow([
                        buf,
                        c.get("rows", 0),
                        c.get("bytes", 0),
                        peak,
                        r.get("events", 0),
                        r.get("bytes", 0),
                        f"{float(r.get('energy_nj', 0.0) or 0.0):.6f}",
                        int(active),
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
        print(f"[run_sim_with_real_eval] 运行真实推理 ({_network_dataset_label(args.network)})...")
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
        network = getattr(args, "network", "pointkan-elite")
        accelerator = args.accelerator
        input_points = args.num_points
        num_classes = args.num_classes
        input_channels = getattr(args, "input_channels", None)
        show_layers = True
        neurosim_full = args.neurosim_full
        neurosim_pybind = bool(getattr(args, "neurosim_pybind", False))
        no_layer_buffer = False
        show_shapes = False
        show_events = False
        enable_w8a8_quant_dequant = (
            True if args.accelerator in {"acc", "pointcim"} and args.qd_en is None else bool(args.qd_en)
        )
        bn_en = int(args.bn_en)
        cim_backend = getattr(args, "cim_backend", None)
        neurosim_root = getattr(args, "neurosim_root", None)
        neurosim_technode = getattr(args, "neurosim_technode", None)
        neurosim_adc_clk_freq = getattr(args, "neurosim_adc_clk_freq", None)
        neurosim_sar_column_power_w = getattr(args, "neurosim_sar_column_power_w", None)
        neurosim_sar_column_power_uw = getattr(args, "neurosim_sar_column_power_uw", None)
        use_scalesim_systolic = getattr(args, "use_scalesim_systolic", False)
        scalesim_root = getattr(args, "scalesim_root", None)
        scalesim_config = getattr(args, "scalesim_config", None)
        knn_conv_pipeline_mode = getattr(args, "knn_conv_pipeline_mode", "disabled")
        fc_num_slices = getattr(args, "fc_num_slices", 1)
        fc_cores_per_slice = getattr(args, "fc_cores_per_slice", 16)
        fc_muls_per_core = getattr(args, "fc_muls_per_core", 1024)
        fc_coalescing_factor = getattr(args, "fc_coalescing_factor", 0.0)
        output_timeline_csv = None
        output_timeline_png = None
        timeline_stride = 500
        timeline_max_cycle = 2000

    sim_args = SimArgs()
    sim_args.real_conv_linear_list = locals().get("conv_linear_list", None)
    log_section(f"PPA 仿真（与 checkpoint/{_network_dataset_label(args.network)} 配置一致）")
    log_info(f"Checkpoint: {checkpoint_path}")
    log_info(f"num_points={args.num_points}, num_classes={args.num_classes}")

    results = run_network_simulation(sim_args)
    hw_config = get_accelerator_config(sim_args.accelerator)
    hw_config.bn_fuse_with_conv_linear = bool(args.bn_en)
    hw_config.enable_w8a8_quant_dequant = (
        True if args.accelerator in {"acc", "pointcim"} and args.qd_en is None else bool(args.qd_en)
    )
    if getattr(sim_args, "cim_backend", None):
        hw_config.cim_backend = sim_args.cim_backend
    _apply_neurosim_cli_overrides(hw_config, sim_args)

    sim_results_dir = _export_standard_sim_outputs(results, sim_args, hw_config)
    if args.output:
        out_path = sim_log_path / os.path.basename(args.output)
        _write_report_to_file(
            results,
            str(out_path),
            network_name=sim_args.network,
            accelerator_name=hw_config.accelerator_name,
        )
        log_info(f"报告已写入: {out_path}")
    log_info(f"结构仿真结果目录: {sim_results_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
