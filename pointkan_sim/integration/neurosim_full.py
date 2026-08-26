"""
NeuroSim 全网络仿真（单次 main 调用，逐层 config + weight/input CSV）
=====================================================================

根据 conv_linear_list（真实前向收集的 Conv1d/Linear 层信息）生成 NeuroSim config CSV 与每层
weight/input CSV，调用 NeuroSIM/main 一次，解析 stdout 得到每层 latency(ns) 与 energy(nJ)，
供 run_sim_with_real_eval 注入 `cim_full_context`，驱动全网络 Conv1d/Linear PPA。
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..operators.neurosim_adapter import (
    _parse_layer_results,
    _run_neurosim_main,
    parse_clock_period_ns,
    parse_critical_clk_period_ns,
    parse_chip_clk_period_ns,
    _write_matrix_activation_conv,
    _write_matrix_weight,
)
from .real_data_eval import ConvLinearLayerInfo
from ..utils.shared_helpers import stretch_input_conv1d as _stretch_input_conv1d


_PATTERN_TILE_SECTION = re.compile(r"^-+\s*# of tile used for each layer\s*-+$", re.IGNORECASE)
_PATTERN_TILE_LINE = re.compile(r"^layer(\d+)\s*:\s*([\d.eE+-]+)")
_PATTERN_TILE_SIZE = re.compile(r"^Desired\s+Conventional\s+Mapped\s+Tile\s+Storage\s+Size:\s*([\d.eE+-]+)x([\d.eE+-]+)", re.IGNORECASE)
_PATTERN_PE_SIZE = re.compile(r"^Desired\s+Conventional\s+PE\s+Storage\s+Size:\s*([\d.eE+-]+)x([\d.eE+-]+)", re.IGNORECASE)
_PATTERN_SUBARRAY_SIZE = re.compile(r"^User-defined\s+SubArray\s+Size:\s*([\d.eE+-]+)x([\d.eE+-]+)", re.IGNORECASE)
_PATTERN_CHIP_AREA = re.compile(r"^ChipArea\s*:\s*([\d.eE+-]+)\s*um\^2", re.IGNORECASE)
_PATTERN_CHIP_PIPELINE_LATENCY = re.compile(r"^Chip\s+pipeline-system-clock-cycle\s+\(per image\)\s+is:\s*([\d.eE+-]+)\s*ns", re.IGNORECASE)
_PATTERN_CHIP_PIPELINE_BUFFER_LATENCY = re.compile(r"^Chip\s+pipeline-system\s+buffer\s+readLatency\s+\(per image\)\s+is:\s*([\d.eE+-]+)\s*ns", re.IGNORECASE)
_PATTERN_CHIP_PIPELINE_IC_LATENCY = re.compile(r"^Chip\s+pipeline-system\s+ic\s+readLatency\s+\(per image\)\s+is:\s*([\d.eE+-]+)\s*ns", re.IGNORECASE)
_PATTERN_LAYER_HEADER = re.compile(r"^--------------------\s*Estimation of Layer\s+(\d+)\s*----------------------$")
_PATTERN_LAYER_READ_LATENCY = re.compile(r"^layer(\d+)'s readLatency is:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_LAYER_READ_ENERGY = re.compile(r"^layer(\d+)'s readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_LAYER_BUFFER_LATENCY = re.compile(r"^layer(\d+)'s buffer latency is:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_LAYER_BUFFER_ENERGY = re.compile(r"^layer(\d+)'s buffer readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_LAYER_IC_LATENCY = re.compile(r"^layer(\d+)'s ic latency is:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_LAYER_IC_ENERGY = re.compile(r"^layer(\d+)'s ic readDynamicEnergy is:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_CORE_ADC_LATENCY = re.compile(
    r"^-----------\s*ADC\s+\(or S/As and precharger for SRAM\)\s+readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_CORE_ACCUM_LATENCY = re.compile(
    r"^-----------\s*Accumulation Circuits\s+\(subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units\)\s+readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
# 仅匹配「总 Other」行；勿用 `Other Peripheries .* readLatency`，否则会匹配 visible-system / hidden-internal 并覆盖正确值
_PATTERN_CORE_OTHER_LATENCY = re.compile(
    r"^-----------\s*Other Peripheries \(e\.g\. decoders, mux, switchmatrix, buffers, IC, pooling and activation units\) readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_CORE_ADC_ENERGY = re.compile(
    r"^-----------\s*ADC\s+\(or S/As and precharger for SRAM\)\s+readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_CORE_ACCUM_ENERGY = re.compile(
    r"^-----------\s*Accumulation Circuits\s+\(subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units\)\s+readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_CORE_OTHER_ENERGY = re.compile(
    r"^-----------\s*Other Peripheries \(e\.g\. decoders, mux, switchmatrix, buffers, IC, pooling and activation units\) readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_ARRAY_LATENCY = re.compile(r"^-----------\s*Array readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ARRAY_ENERGY = re.compile(r"^-----------\s*Array readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_LAT_PRECHARGER = re.compile(r"^-----------\s*ADC fine-grained precharger readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ADC_LAT_COL_DELAY = re.compile(r"^-----------\s*ADC fine-grained colDelay readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ADC_LAT_SENSEAMP = re.compile(r"^-----------\s*ADC fine-grained senseAmp readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ADC_LAT_MLSA = re.compile(r"^-----------\s*ADC fine-grained multilevelSenseAmp readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ADC_LAT_MLENC = re.compile(r"^-----------\s*ADC fine-grained multilevelSAEncoder readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ADC_LAT_SAR = re.compile(r"^-----------\s*ADC fine-grained sarADC readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ADC_EN_PRECHARGER = re.compile(r"^-----------\s*ADC fine-grained precharger readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_EN_SENSEAMP = re.compile(r"^-----------\s*ADC fine-grained senseAmp readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_EN_MLSA = re.compile(r"^-----------\s*ADC fine-grained multilevelSenseAmp readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_EN_MLENC = re.compile(r"^-----------\s*ADC fine-grained multilevelSAEncoder readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_EN_SAR = re.compile(r"^-----------\s*ADC fine-grained sarADC readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ACCUM_LAT_ADDER = re.compile(r"^-----------\s*Accum fine-grained adder readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ACCUM_LAT_DFF = re.compile(r"^-----------\s*Accum fine-grained dff readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ACCUM_LAT_SHIFTADD_W = re.compile(r"^-----------\s*Accum fine-grained shiftAddWeight readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ACCUM_LAT_SHIFTADD_I = re.compile(r"^-----------\s*Accum fine-grained shiftAddInput readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ACCUM_LAT_ADDER_TREE = re.compile(r"^-----------\s*Accum fine-grained adderTree readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ACCUM_LAT_TILE = re.compile(r"^-----------\s*Accum fine-grained tileAccum readLatency is\s*:\s*([\d.eE+-]+)\s*ns$", re.IGNORECASE)
_PATTERN_ACCUM_EN_ADDER = re.compile(r"^-----------\s*Accum fine-grained adder readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ACCUM_EN_DFF = re.compile(r"^-----------\s*Accum fine-grained dff readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ACCUM_EN_SHIFTADD_W = re.compile(r"^-----------\s*Accum fine-grained shiftAddWeight readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ACCUM_EN_SHIFTADD_I = re.compile(r"^-----------\s*Accum fine-grained shiftAddInput readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ACCUM_EN_ADDER_TREE = re.compile(r"^-----------\s*Accum fine-grained adderTree readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ACCUM_EN_TILE = re.compile(r"^-----------\s*Accum fine-grained tileAccum readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_SUB_ARRAY = re.compile(r"^-----------\s*ADC sub-breakdown readDynamicEnergyArray is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_SUB_PRECHARGER = re.compile(r"^-----------\s*ADC sub-breakdown precharger is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_SUB_SENSEAMP = re.compile(r"^-----------\s*ADC sub-breakdown senseAmp .* is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_SUB_MLSA = re.compile(r"^-----------\s*ADC sub-breakdown multilevelSenseAmp is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_SUB_MLENC = re.compile(r"^-----------\s*ADC sub-breakdown multilevelSAEncoder is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_ADC_SUB_SAR = re.compile(r"^-----------\s*ADC sub-breakdown sarADC is\s*:\s*([\d.eE+-]+)\s*pJ$", re.IGNORECASE)
_PATTERN_OTHER_SUB_TILE_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tile-local readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_OTHER_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEOther readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_BUF_HT_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tileBufferHTree readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_WL_DECODER_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEWLDecoder readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_WL_DRIVER_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEWLDriver readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_WL_SWITCH_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEWLSwitch readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_SL_SWITCH_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePESLSwitch readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_MUX_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEMux readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_MUX_DECODER_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEMuxDecoder readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_LEVEL_SHIFTER_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePELevelShifter readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_WL_PATH_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEWLPath readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_MUX_PATH_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEMuxPath readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_SELECTED_PATH_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePESelectedPath readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_OTHER_MULTIPLIER = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEOtherMultiplier is\s*:\s*([\d.eE+-]+)$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_NUM_COL_MUXED = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePENumColMuxed is\s*:\s*([\d.eE+-]+)$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_NUM_ADD = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePENumAdd is\s*:\s*([\d.eE+-]+)$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_NUM_ROW_ACTIVITY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePENumRowActivity is\s*:\s*([\d.eE+-]+)$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_ACT_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown activation readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_ACT_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tileActivation readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_CHIP_ACT_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown chipActivation readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_POOL_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown pooling readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GB_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalBuffer readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GIC_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalIC readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GHTREE_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalHTree readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GXYBUS_LATENCY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalXYBus readLatency is\s*:\s*([\d.eE+-]+)\s*ns$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tile-local readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_PE_OTHER_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tilePEOther readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_BUF_HT_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tileBufferHTree readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_ACT_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown activation readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_TILE_ACT_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown tileActivation readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_CHIP_ACT_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown chipActivation readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_POOL_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown pooling readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GB_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalBuffer readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GIC_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalIC readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GHTREE_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalHTree readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)
_PATTERN_OTHER_SUB_GXYBUS_ENERGY = re.compile(
    r"^-----------\s*Other hidden sub-breakdown globalXYBus readDynamicEnergy is\s*:\s*([\d.eE+-]+)\s*pJ$",
    re.IGNORECASE,
)


def _parse_neurosim_layer_breakdown(stdout: str, num_layers: int) -> List[Dict[str, float]]:
    """Parse per-layer NeuroSim core/buffer/IC breakdown from subprocess stdout."""
    if num_layers <= 0:
        return []

    # Summary 段会再次打印与逐层同格式的 ADC/Accum/Other 行（芯片累加），若不截断会把最后一层的 core_* 覆盖成芯片级数值
    _sum_mark = "------------------------------ Summary"
    if _sum_mark in stdout:
        stdout = stdout.split(_sum_mark, 1)[0]

    def _new_entry(layer_idx: int) -> Dict[str, float]:
        return {
            "layer_index": float(layer_idx),
            "read_latency_ns": 0.0,
            "read_energy_nj": 0.0,
            "buffer_latency_ns": 0.0,
            "buffer_energy_nj": 0.0,
            "ic_latency_ns": 0.0,
            "ic_energy_nj": 0.0,
            "core_latency_adc_ns": 0.0,
            "core_latency_accum_ns": 0.0,
            "core_latency_other_ns": 0.0,
            "core_energy_adc_nj": 0.0,
            "core_energy_accum_nj": 0.0,
            "core_energy_other_nj": 0.0,
            "array_latency_ns": 0.0,
            "array_energy_nj": 0.0,
            "adc_latency_precharger_ns": 0.0,
            "adc_latency_col_delay_ns": 0.0,
            "adc_latency_senseamp_ns": 0.0,
            "adc_latency_mlsa_ns": 0.0,
            "adc_latency_mlencoder_ns": 0.0,
            "adc_latency_sar_ns": 0.0,
            "adc_subarray_energy_nj": 0.0,
            "adc_energy_precharger_nj": 0.0,
            "adc_energy_senseamp_nj": 0.0,
            "adc_energy_mlsa_nj": 0.0,
            "adc_energy_mlencoder_nj": 0.0,
            "adc_energy_sar_nj": 0.0,
            "adc_precharger_energy_nj": 0.0,
            "adc_senseamp_energy_nj": 0.0,
            "adc_mlsa_energy_nj": 0.0,
            "adc_mlencoder_energy_nj": 0.0,
            "adc_sar_energy_nj": 0.0,
            "accum_latency_adder_ns": 0.0,
            "accum_latency_dff_ns": 0.0,
            "accum_latency_shiftadd_weight_ns": 0.0,
            "accum_latency_shiftadd_input_ns": 0.0,
            "accum_latency_adder_tree_ns": 0.0,
            "accum_latency_tile_ns": 0.0,
            "accum_energy_adder_nj": 0.0,
            "accum_energy_dff_nj": 0.0,
            "accum_energy_shiftadd_weight_nj": 0.0,
            "accum_energy_shiftadd_input_nj": 0.0,
            "accum_energy_adder_tree_nj": 0.0,
            "accum_energy_tile_nj": 0.0,
            "other_sub_tile_latency_ns": 0.0,
            "other_sub_activation_latency_ns": 0.0,
            "other_sub_pooling_latency_ns": 0.0,
            "other_sub_global_buffer_latency_ns": 0.0,
            "other_sub_global_ic_latency_ns": 0.0,
            "other_sub_global_htree_latency_ns": 0.0,
            "other_sub_global_xybus_latency_ns": 0.0,
            "other_sub_tile_pe_other_latency_ns": 0.0,
            "other_sub_tile_buffer_htree_latency_ns": 0.0,
            "other_sub_tile_activation_latency_ns": 0.0,
            "other_sub_chip_activation_latency_ns": 0.0,
            "other_sub_tile_pe_wl_decoder_latency_ns": 0.0,
            "other_sub_tile_pe_wl_driver_latency_ns": 0.0,
            "other_sub_tile_pe_wl_switch_latency_ns": 0.0,
            "other_sub_tile_pe_sl_switch_latency_ns": 0.0,
            "other_sub_tile_pe_mux_latency_ns": 0.0,
            "other_sub_tile_pe_mux_decoder_latency_ns": 0.0,
            "other_sub_tile_pe_level_shifter_latency_ns": 0.0,
            "other_sub_tile_pe_wl_path_latency_ns": 0.0,
            "other_sub_tile_pe_mux_path_latency_ns": 0.0,
            "other_sub_tile_pe_selected_path_latency_ns": 0.0,
            "other_sub_tile_pe_other_multiplier": 0.0,
            "other_sub_tile_pe_num_col_muxed": 0.0,
            "other_sub_tile_pe_num_add": 0.0,
            "other_sub_tile_pe_num_row_activity": 0.0,
            "other_sub_tile_energy_nj": 0.0,
            "other_sub_activation_energy_nj": 0.0,
            "other_sub_pooling_energy_nj": 0.0,
            "other_sub_global_buffer_energy_nj": 0.0,
            "other_sub_global_ic_energy_nj": 0.0,
            "other_sub_global_htree_energy_nj": 0.0,
            "other_sub_global_xybus_energy_nj": 0.0,
            "other_sub_tile_pe_other_energy_nj": 0.0,
            "other_sub_tile_buffer_htree_energy_nj": 0.0,
            "other_sub_tile_activation_energy_nj": 0.0,
            "other_sub_chip_activation_energy_nj": 0.0,
        }

    per_layer: List[Dict[str, float]] = [_new_entry(i) for i in range(num_layers)]
    current_idx: Optional[int] = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _PATTERN_LAYER_HEADER.match(line)
        if m:
            idx = max(0, int(m.group(1)) - 1)
            current_idx = idx if idx < len(per_layer) else None
            continue

        matched_top_level = False
        for pattern, key, scale in (
            (_PATTERN_LAYER_READ_LATENCY, "read_latency_ns", 1.0),
            (_PATTERN_LAYER_READ_ENERGY, "read_energy_nj", 1.0 / 1000.0),
            (_PATTERN_LAYER_BUFFER_LATENCY, "buffer_latency_ns", 1.0),
            (_PATTERN_LAYER_BUFFER_ENERGY, "buffer_energy_nj", 1.0 / 1000.0),
            (_PATTERN_LAYER_IC_LATENCY, "ic_latency_ns", 1.0),
            (_PATTERN_LAYER_IC_ENERGY, "ic_energy_nj", 1.0 / 1000.0),
        ):
            m = pattern.match(line)
            if m:
                idx = max(0, int(m.group(1)) - 1)
                if idx < len(per_layer):
                    per_layer[idx][key] = float(m.group(2)) * scale
                    current_idx = idx
                matched_top_level = True
                break
        if matched_top_level or current_idx is None:
            continue

        entry = per_layer[current_idx]
        for pattern, key, scale in (
            (_PATTERN_CORE_ADC_LATENCY, "core_latency_adc_ns", 1.0),
            (_PATTERN_CORE_ACCUM_LATENCY, "core_latency_accum_ns", 1.0),
            (_PATTERN_CORE_OTHER_LATENCY, "core_latency_other_ns", 1.0),
            (_PATTERN_CORE_ADC_ENERGY, "core_energy_adc_nj", 1.0 / 1000.0),
            (_PATTERN_CORE_ACCUM_ENERGY, "core_energy_accum_nj", 1.0 / 1000.0),
            (_PATTERN_CORE_OTHER_ENERGY, "core_energy_other_nj", 1.0 / 1000.0),
            (_PATTERN_ARRAY_LATENCY, "array_latency_ns", 1.0),
            (_PATTERN_ARRAY_ENERGY, "array_energy_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_LAT_PRECHARGER, "adc_latency_precharger_ns", 1.0),
            (_PATTERN_ADC_LAT_COL_DELAY, "adc_latency_col_delay_ns", 1.0),
            (_PATTERN_ADC_LAT_SENSEAMP, "adc_latency_senseamp_ns", 1.0),
            (_PATTERN_ADC_LAT_MLSA, "adc_latency_mlsa_ns", 1.0),
            (_PATTERN_ADC_LAT_MLENC, "adc_latency_mlencoder_ns", 1.0),
            (_PATTERN_ADC_LAT_SAR, "adc_latency_sar_ns", 1.0),
            (_PATTERN_ADC_EN_PRECHARGER, "adc_energy_precharger_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_EN_SENSEAMP, "adc_energy_senseamp_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_EN_MLSA, "adc_energy_mlsa_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_EN_MLENC, "adc_energy_mlencoder_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_EN_SAR, "adc_energy_sar_nj", 1.0 / 1000.0),
            (_PATTERN_ACCUM_LAT_ADDER, "accum_latency_adder_ns", 1.0),
            (_PATTERN_ACCUM_LAT_DFF, "accum_latency_dff_ns", 1.0),
            (_PATTERN_ACCUM_LAT_SHIFTADD_W, "accum_latency_shiftadd_weight_ns", 1.0),
            (_PATTERN_ACCUM_LAT_SHIFTADD_I, "accum_latency_shiftadd_input_ns", 1.0),
            (_PATTERN_ACCUM_LAT_ADDER_TREE, "accum_latency_adder_tree_ns", 1.0),
            (_PATTERN_ACCUM_LAT_TILE, "accum_latency_tile_ns", 1.0),
            (_PATTERN_ACCUM_EN_ADDER, "accum_energy_adder_nj", 1.0 / 1000.0),
            (_PATTERN_ACCUM_EN_DFF, "accum_energy_dff_nj", 1.0 / 1000.0),
            (_PATTERN_ACCUM_EN_SHIFTADD_W, "accum_energy_shiftadd_weight_nj", 1.0 / 1000.0),
            (_PATTERN_ACCUM_EN_SHIFTADD_I, "accum_energy_shiftadd_input_nj", 1.0 / 1000.0),
            (_PATTERN_ACCUM_EN_ADDER_TREE, "accum_energy_adder_tree_nj", 1.0 / 1000.0),
            (_PATTERN_ACCUM_EN_TILE, "accum_energy_tile_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_SUB_ARRAY, "adc_subarray_energy_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_SUB_PRECHARGER, "adc_precharger_energy_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_SUB_SENSEAMP, "adc_senseamp_energy_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_SUB_MLSA, "adc_mlsa_energy_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_SUB_MLENC, "adc_mlencoder_energy_nj", 1.0 / 1000.0),
            (_PATTERN_ADC_SUB_SAR, "adc_sar_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_TILE_LATENCY, "other_sub_tile_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_OTHER_LATENCY, "other_sub_tile_pe_other_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_BUF_HT_LATENCY, "other_sub_tile_buffer_htree_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_WL_DECODER_LATENCY, "other_sub_tile_pe_wl_decoder_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_WL_DRIVER_LATENCY, "other_sub_tile_pe_wl_driver_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_WL_SWITCH_LATENCY, "other_sub_tile_pe_wl_switch_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_SL_SWITCH_LATENCY, "other_sub_tile_pe_sl_switch_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_MUX_LATENCY, "other_sub_tile_pe_mux_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_MUX_DECODER_LATENCY, "other_sub_tile_pe_mux_decoder_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_LEVEL_SHIFTER_LATENCY, "other_sub_tile_pe_level_shifter_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_WL_PATH_LATENCY, "other_sub_tile_pe_wl_path_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_MUX_PATH_LATENCY, "other_sub_tile_pe_mux_path_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_SELECTED_PATH_LATENCY, "other_sub_tile_pe_selected_path_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_OTHER_MULTIPLIER, "other_sub_tile_pe_other_multiplier", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_NUM_COL_MUXED, "other_sub_tile_pe_num_col_muxed", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_NUM_ADD, "other_sub_tile_pe_num_add", 1.0),
            (_PATTERN_OTHER_SUB_TILE_PE_NUM_ROW_ACTIVITY, "other_sub_tile_pe_num_row_activity", 1.0),
            (_PATTERN_OTHER_SUB_ACT_LATENCY, "other_sub_activation_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_ACT_LATENCY, "other_sub_tile_activation_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_CHIP_ACT_LATENCY, "other_sub_chip_activation_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_POOL_LATENCY, "other_sub_pooling_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_GB_LATENCY, "other_sub_global_buffer_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_GIC_LATENCY, "other_sub_global_ic_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_GHTREE_LATENCY, "other_sub_global_htree_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_GXYBUS_LATENCY, "other_sub_global_xybus_latency_ns", 1.0),
            (_PATTERN_OTHER_SUB_TILE_ENERGY, "other_sub_tile_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_TILE_PE_OTHER_ENERGY, "other_sub_tile_pe_other_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_TILE_BUF_HT_ENERGY, "other_sub_tile_buffer_htree_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_ACT_ENERGY, "other_sub_activation_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_TILE_ACT_ENERGY, "other_sub_tile_activation_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_CHIP_ACT_ENERGY, "other_sub_chip_activation_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_POOL_ENERGY, "other_sub_pooling_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_GB_ENERGY, "other_sub_global_buffer_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_GIC_ENERGY, "other_sub_global_ic_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_GHTREE_ENERGY, "other_sub_global_htree_energy_nj", 1.0 / 1000.0),
            (_PATTERN_OTHER_SUB_GXYBUS_ENERGY, "other_sub_global_xybus_energy_nj", 1.0 / 1000.0),
        ):
            m = pattern.match(line)
            if m:
                entry[key] = float(m.group(1)) * scale
                break

    return per_layer


def _scale_layer_breakdown_items(
    items: Optional[List[Dict[str, Any]]],
    scale_factors: Optional[List[float]],
) -> Optional[List[Dict[str, Any]]]:
    if not items:
        return None
    scaled: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        sf = float(scale_factors[idx]) if scale_factors and idx < len(scale_factors) else 1.0
        row: Dict[str, Any] = {}
        for key, value in (item or {}).items():
            if key == "layer_index":
                row[key] = int(float(value))
            elif isinstance(value, (int, float)) and (
                str(key).endswith("_latency_ns")
                or str(key).endswith("_energy_nj")
                or key in {"read_latency_ns", "read_energy_nj"}
            ):
                row[key] = float(value) * sf
            else:
                row[key] = value
        scaled.append(row)
    return scaled


def _parse_rram_tile_stats(stdout: str, conv_linear_list: List[ConvLinearLayerInfo]) -> Optional[dict]:
    lines = [ln.strip() for ln in stdout.splitlines()]
    start = None
    for i, ln in enumerate(lines):
        if _PATTERN_TILE_SECTION.search(ln):
            start = i + 1
            break
    if start is None:
        return None

    per_layer = []
    total_tiles = 0.0
    for ln in lines[start:]:
        m = _PATTERN_TILE_LINE.match(ln)
        if not m:
            if per_layer:
                break
            continue
        idx = int(m.group(1)) - 1
        tilenum = float(m.group(2))
        total_tiles += tilenum
        info = conv_linear_list[idx] if 0 <= idx < len(conv_linear_list) else None
        per_layer.append({
            "layer_index": idx,
            "layer_type": getattr(info, "layer_type", ""),
            "tilenum": tilenum,
            "PE_count": None,
            "max_row": None,
            "max_column": None,
        })

    if not per_layer:
        return None

    tile_row = tile_col = None
    pe_row = pe_col = None
    xbar_row = xbar_col = None
    for ln in lines:
        m = _PATTERN_TILE_SIZE.match(ln)
        if m:
            tile_row = int(float(m.group(1)))
            tile_col = int(float(m.group(2)))
            continue
        m = _PATTERN_PE_SIZE.match(ln)
        if m:
            pe_row = int(float(m.group(1)))
            pe_col = int(float(m.group(2)))
            continue
        m = _PATTERN_SUBARRAY_SIZE.match(ln)
        if m:
            xbar_row = int(float(m.group(1)))
            xbar_col = int(float(m.group(2)))
            continue

    pe_per_tile = None
    total_pes = None
    if tile_row and tile_col and pe_row and pe_col and pe_row > 0 and pe_col > 0:
        pe_per_tile = int(math.ceil(tile_row / float(pe_row)) * math.ceil(tile_col / float(pe_col)))
        total_pes = float(total_tiles) * pe_per_tile
        for item in per_layer:
            item["PE_count"] = float(item.get("tilenum", 0.0)) * pe_per_tile
            item["max_row"] = pe_row
            item["max_column"] = pe_col

    return {
        "total_tile_count": total_tiles,
        "total_PE_count": total_pes,
        "xbar_size": {"row": xbar_row, "column": xbar_col},
        "tile_size": {"row": tile_row, "column": tile_col},
        "pe_size": {"row": pe_row, "column": pe_col},
        "pe_per_tile": pe_per_tile,
        "per_layer": per_layer,
        "source": "neurosim_stdout",
    }


def _per_layer_tile_counts_from_neurosim_matrix(num_tile_each: object) -> List[float]:
    """
    chip_setup['num_tile_each_layer'] mirrors C++ numTileEachLayer: typically two rows
    (tile rows × tile cols per layer), i.e. tilenum(layer i) = row0[i] * row1[i].
    """
    if not num_tile_each or not isinstance(num_tile_each, (list, tuple)):
        return []
    rows = list(num_tile_each)
    if (
        len(rows) >= 2
        and isinstance(rows[0], (list, tuple))
        and isinstance(rows[1], (list, tuple))
    ):
        r0, r1 = rows[0], rows[1]
        n = min(len(r0), len(r1))
        return [float(r0[i]) * float(r1[i]) for i in range(n)]
    if len(rows) == 1 and isinstance(rows[0], (list, tuple)):
        return [float(x) for x in rows[0]]
    return []


def build_rram_tile_stats_from_pybind_setup(
    setup: dict,
    conv_linear_list: List[ConvLinearLayerInfo],
    *,
    subarray: int,
    parallel_read: int,
) -> Optional[dict]:
    """
    Build rram_tile_stats dict isomorphic to _parse_rram_tile_stats(stdout, ...),
    from chip_setup() return value (num_tile_each_layer).  PE/tile *pixel* sizes are
    not exported by pybind today; subarray / parallel_read mirror configure_param.
    """
    tile_counts = _per_layer_tile_counts_from_neurosim_matrix(setup.get("num_tile_each_layer"))
    if not tile_counts:
        return None
    per_layer: List[dict] = []
    total_tiles = 0.0
    n = min(len(conv_linear_list), len(tile_counts))
    for idx in range(n):
        tilenum = float(tile_counts[idx])
        total_tiles += tilenum
        info = conv_linear_list[idx]
        per_layer.append(
            {
                "layer_index": idx,
                "layer_type": getattr(info, "layer_type", ""),
                "tilenum": tilenum,
                "PE_count": None,
                "max_row": int(subarray),
                "max_column": int(subarray),
            }
        )
    if not per_layer or total_tiles <= 0:
        return None
    return {
        "total_tile_count": total_tiles,
        "total_PE_count": None,
        "xbar_size": {"row": int(subarray), "column": int(subarray)},
        "tile_size": {"row": None, "column": None},
        "pe_size": {"row": int(subarray), "column": int(parallel_read)},
        "pe_per_tile": None,
        "per_layer": per_layer,
        "source": "neurosim_pybind_chip_setup",
    }


def _parse_total_area_um2(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        m = _PATTERN_CHIP_AREA.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _parse_chip_pipeline_latency_ns(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        m = _PATTERN_CHIP_PIPELINE_LATENCY.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _parse_chip_pipeline_buffer_latency_ns(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        m = _PATTERN_CHIP_PIPELINE_BUFFER_LATENCY.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _parse_chip_pipeline_ic_latency_ns(stdout: str) -> Optional[float]:
    for line in stdout.splitlines():
        m = _PATTERN_CHIP_PIPELINE_IC_LATENCY.match(line.strip())
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _estimate_layer_area_um2_from_tiles(
    total_area_um2: Optional[float],
    rram_tile_stats: Optional[dict],
    conv_linear_list: List[ConvLinearLayerInfo],
) -> Optional[List[float]]:
    """
    NeuroSim stdout does not print per-layer physical area directly.
    Use the parsed per-layer tile counts to distribute ChipArea proportionally.

    This keeps the area export path usable in acc+neurosim mode while making the
    derivation explicit in code comments and result metadata.
    """
    total_area = float(total_area_um2 or 0.0)
    if total_area <= 0.0 or not rram_tile_stats:
        return None

    per_layer_stats = rram_tile_stats.get("per_layer") or []
    total_tiles = float(rram_tile_stats.get("total_tile_count", 0.0) or 0.0)
    if total_tiles <= 0.0 or not per_layer_stats:
        return None

    layer_area_um2 = [0.0] * len(conv_linear_list)
    assigned_area = 0.0
    last_valid_idx: Optional[int] = None

    for item in per_layer_stats:
        idx = int(item.get("layer_index", -1))
        if idx < 0 or idx >= len(layer_area_um2):
            continue
        tile_count = float(item.get("tilenum", 0.0) or 0.0)
        area = total_area * tile_count / total_tiles if tile_count > 0.0 else 0.0
        layer_area_um2[idx] = area
        assigned_area += area
        last_valid_idx = idx

    if last_valid_idx is None:
        return None

    # Keep the sum numerically aligned with total_area_um2.
    residual = total_area - assigned_area
    if abs(residual) > 1e-6:
        layer_area_um2[last_valid_idx] += residual

    return layer_area_um2


def _pad_k_dimension_to_alignment(
    weight_data: np.ndarray,
    input_stretched: np.ndarray,
    *,
    alignment: int,
    kernel_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Pad the K dimension (input channels/features) to an integer multiple of alignment.
    Dummy channels are zero-filled in both weight and input.

    Returns: (padded_weight, padded_input, original_k, padded_k)
    """
    original_k = int(weight_data.reshape(weight_data.shape[0], -1).shape[1])
    if alignment <= 0:
        return weight_data, input_stretched, original_k, original_k
    padded_k = int(math.ceil(original_k / float(alignment)) * alignment)
    if padded_k == original_k:
        return weight_data, input_stretched, original_k, padded_k

    if kernel_size > 1 and padded_k % kernel_size != 0:
        return weight_data, input_stretched, original_k, original_k

    padded_weight = np.zeros((weight_data.shape[0], padded_k), dtype=np.float32)
    flat_weight = weight_data.reshape(weight_data.shape[0], -1)
    padded_weight[:, :original_k] = flat_weight

    padded_input = np.zeros((input_stretched.shape[0], input_stretched.shape[1], padded_k), dtype=np.float32)
    padded_input[:, :, :original_k] = input_stretched
    return padded_weight, padded_input, original_k, padded_k


@dataclass
class NeuroSimFullResult:
    """全网络 NeuroSim 逐层结果。"""
    layer_latency_ns: List[float]
    layer_energy_nj: List[float]
    success: bool
    error: Optional[str] = None
    clock_period_ns: Optional[float] = None  # NeuroSim 实际工作时钟周期(ns)，用于 latency_ns -> cycles 换算
    # 面积（um^2）：total_area 直接解析自 stdout；per-layer 由 tile 占比推导
    layer_area_um2: Optional[List[float]] = None
    total_area_um2: Optional[float] = None
    layer_area_source: Optional[str] = None
    # Tile/PE 统计：与报告字段 rram_tile_stats 同构；若 NeuroSim 输出则解析，否则可依 subarray/层数构造占位
    rram_tile_stats: Optional[dict] = None
    # Chip-level pipelined throughput summary reported by NeuroSim stdout.
    chip_pipeline_latency_ns: Optional[float] = None
    chip_pipeline_buffer_latency_ns: Optional[float] = None
    chip_pipeline_ic_latency_ns: Optional[float] = None
    # Per-layer actual/capped input width ratio when max_in_vectors is set (else all 1.0).
    workload_scale_factors: Optional[List[float]] = None
    # Per-layer detailed breakdown for plotting/reporting.
    layer_breakdown: Optional[List[Dict[str, Any]]] = None


def _conv_linear_list_to_net_structure(
    conv_linear_list: List[ConvLinearLayerInfo],
    max_in_vectors: Optional[int] = None,
) -> Tuple[List[List[float]], List[float]]:
    """
    Convert ConvLinearLayerInfo list → NeuroSim netStructure format.
    Each row: [H_in, W_in, C_in, K_h, K_w, C_out, quantized_flag, stride]
    - Conv1d: H_in=1, W_in=length, C_in=in_channels, K_h=1, K_w=kernel_size, C_out=out_channels
    - Linear:  H_in=1, W_in=tokens,  C_in=in_features, K_h=1, K_w=1,           C_out=out_features
    Matches the CSV format written by run_neurosim_full for the subprocess path.

    max_in_vectors: optional cap on W_in / tokens per layer.  None (default) runs the
        full workload with no latency/energy scaling — required for pybind/subprocess
        parity and PointCIM precise modeling.  Set a positive int for faster approximations.

    Returns: (net_structure_rows, scale_factors)
        scale_factors[i] = actual_in_w / capped_in_w  (1.0 when no capping applied)
    """
    rows: List[List[float]] = []
    scale_factors: List[float] = []
    for info in conv_linear_list:
        if info.layer_type == "conv":
            raw_in_w = max(1, info.length)
            capped_in_w = min(raw_in_w, max_in_vectors) if max_in_vectors and max_in_vectors > 0 else raw_in_w
            scale_factors.append(float(raw_in_w) / float(capped_in_w))
            rows.append([
                1.0,
                float(capped_in_w),
                float(max(1, info.in_channels)),
                1.0,
                float(max(1, info.kernel_size)),
                float(max(1, info.out_channels)),
                0.0,   # quantized_flag (followedByMaxPool) – off for PointKAN
                1.0,   # stride
            ])
        else:
            # fc / linear
            raw_tokens = max(1, info.tokens)
            capped_tokens = min(raw_tokens, max_in_vectors) if max_in_vectors and max_in_vectors > 0 else raw_tokens
            scale_factors.append(float(raw_tokens) / float(capped_tokens))
            rows.append([
                1.0,
                float(capped_tokens),
                float(max(1, info.in_features)),
                1.0,
                1.0,
                float(max(1, info.out_features)),
                0.0,
                1.0,
            ])
    return rows, scale_factors


def run_neurosim_full_pybind(
    conv_linear_list: List[ConvLinearLayerInfo],
    neurosim_so_dir: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM",
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    subarray_cols: Optional[int] = None,
    parallel_read: int = 128,
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
    adc_clk_freq_hz: Optional[float] = None,
    sar_column_power_w: Optional[float] = None,
    activity_row_read: float = 0.5,
    cell_bit: int = 1,
    num_col_muxed: int = 8,
    level_output: int = 32,
    resistance_on: float = 100e3,
    resistance_off: float = 100e3 * 17,
    max_in_vectors: Optional[int] = None,
) -> NeuroSimFullResult:
    """
    Pybind variant of run_neurosim_full: calls neurosim_core.chip_setup / chip_run_all
    directly in-process — no subprocess spawn, no CSV file I/O.

    Parameters match run_neurosim_full where applicable.  Hardware defaults (cell_bit,
    num_col_muxed, level_output, resistance_on/off) are set to match Param.cpp defaults
    for the SRAM path (memcelltype=1, operationmode=2/conventionalParallel).

    clk_freq_hz:     NeuroSim system/chip clock (Hz). Default 100 MHz (1e8). NeuroSim may
                     cap this at the circuit critical-path frequency if it is slower.
    adc_clk_freq_hz: SAR ADC clock (Hz). Default 1 GHz (1e9), matching subprocess default.
    activity_row_read: fraction of active rows used for synthetic input generation
                       in ChipCalculatePerformanceFromArrays (default 0.5, consistent
                       with the random activation data written by the subprocess path).

    Synchronous chip timing is enabled (set_synchronous(True)) so global_clk_period_s
    and layer latencies follow the same model as DNN_NeuroSim main when param.synchronous
    is true.
    """
    if not conv_linear_list:
        return NeuroSimFullResult(layer_latency_ns=[], layer_energy_nj=[], success=True)

    # NeuroSim Technology.cpp only supports 130/90/65/45/32/22/14/10/7 nm
    if technode is not None and technode == 28:
        technode = 32

    import sys as _sys
    _sys.path.insert(0, neurosim_so_dir)
    try:
        import neurosim_core as nc
    except ImportError as e:
        return NeuroSimFullResult(
            layer_latency_ns=[], layer_energy_nj=[], success=False,
            error=f"Failed to import neurosim_core from {neurosim_so_dir}: {e}",
        )

    net_structure, scale_factors = _conv_linear_list_to_net_structure(conv_linear_list, max_in_vectors=max_in_vectors)
    if not net_structure:
        return NeuroSimFullResult(layer_latency_ns=[], layer_energy_nj=[], success=False,
                                  error="net_structure is empty after conversion")

    num_layers = len(net_structure)
    # num_col_per_synapse: ceil(synapseBit / cellBit), matches main.cpp switch logic
    num_col_per_synapse = max(1, int(math.ceil(wl_weight / float(cell_bit))))

    try:
        # Configure hardware parameters (matches Param.cpp defaults + user overrides)
        nc.configure_param(
            wl_weight,                                     # synapse_bit
            wl_activation,                                 # num_bit_input
            subarray,                                      # num_row_subarray
            int(subarray_cols) if subarray_cols else subarray,  # num_col_subarray
            parallel_read,                                 # num_row_parallel
            num_col_per_synapse,                           # num_col_per_synapse
            num_col_muxed,                                 # num_col_muxed
            level_output,                                  # level_output
            cell_bit,                                      # cell_bit
            float(clk_freq_hz) if clk_freq_hz else 100e6,  # clk_freq_hz: default 100 MHz
            float(adc_clk_freq_hz) if adc_clk_freq_hz else 1e9,  # adc_clk_freq_hz: default 1 GHz
            int(technode) if technode else 32,             # technode
            resistance_on,                                 # resistance_on
            resistance_off,                                # resistance_off
            False,                                         # saradc
            True,                                          # current_mode
        )
        if sar_column_power_w and sar_column_power_w > 0:
            nc.set_sar_column_power_w(float(sar_column_power_w))

        # Enable synchronous mode to match subprocess (Param.cpp default: synchronous=true).
        # configure_param() hardcodes synchronous=false, so we override it here.
        # In synchronous mode chip_run_all:
        #   1st pass: finds global critical-path clock period across all layers.
        #   2nd pass: returns readLatency in CYCLES (not seconds) — must multiply by
        #             actual clock period to get time.
        # This matches main.cpp behaviour and aligns latency with the subprocess path.
        nc.set_synchronous(True)

        # FloorPlan: one-time setup for all layers
        setup = nc.chip_setup(net_structure)
        total_area_um2 = float(setup.get("chip_area_m2", 0.0) or 0.0) * 1e12  # m² → µm²

        # Run all layers with synthetic data
        activity_list = [activity_row_read] * num_layers
        maxpool_list = [bool(row[6]) for row in net_structure]
        result_dict = nc.chip_run_all(activity_list, maxpool_list)

        layers_out = result_dict.get("layers", [])
        if len(layers_out) != num_layers:
            return NeuroSimFullResult(
                layer_latency_ns=[], layer_energy_nj=[], success=False,
                error=f"chip_run_all returned {len(layers_out)} layers, expected {num_layers}",
            )

        # Clock period derivation (mirrors main.cpp logic for synchronous mode):
        #
        #   global_clk_period_s = circuit critical-path period found by pybind 1st pass.
        #     In synchronous mode chip_run_all populates this field (unlike asynchronous
        #     mode where it was always 0.0).
        #
        #   Main.cpp clock capping rules:
        #     (a) if user_clk_freq > 1/critical_path_period → cap to critical path freq
        #     (b) if user_clk_period > critical_path_period → use user period for latency
        #
        #   Result: actual_period = max(critical_path_period, 1/user_clk_freq)
        #           i.e. latency = cycles × max(circuit_period, 1/user_freq)
        #
        #   For 100 MHz user, 32nm RRAM critical path ~2.54 ns (393 MHz):
        #     actual_period = max(2.54 ns, 10 ns) = 10 ns  ← matches subprocess
        _eff_clk_hz = float(clk_freq_hz) if (clk_freq_hz and clk_freq_hz > 0) else 100e6
        _user_period_s = 1.0 / _eff_clk_hz

        gcp_s = float(result_dict.get("global_clk_period_s", 0.0) or 0.0)
        if gcp_s > 0:
            # Use max(critical_path_period, user_period): matches main.cpp rule (b)
            actual_period_s = max(gcp_s, _user_period_s)
        else:
            # Fallback: use user period directly (accurate when user_freq ≤ critical_path_freq)
            actual_period_s = _user_period_s
        clock_period_ns = actual_period_s * 1e9

        # Per-layer latency: chip_run_all returns readLatency in CYCLES when synchronous=True.
        # Multiply by actual_period_s to convert to seconds, then to ns.
        # Energy (read_energy_nj) is in joules×1e9 regardless of synchronous mode.
        layer_latency_ns: List[float] = []
        layer_energy_nj: List[float] = []
        layer_breakdown: List[Dict[str, Any]] = []
        for i, ld in enumerate(layers_out):
            sf = scale_factors[i] if i < len(scale_factors) else 1.0
            cycles = float(ld.get("read_latency_s", 0.0) or 0.0)  # CYCLES when synchronous
            latency_ns = cycles * actual_period_s * 1e9
            layer_latency_ns.append(latency_ns * sf)
            layer_energy_nj.append(float(ld.get("read_energy_nj", 0.0) or 0.0) * sf)
            layer_breakdown.append({
                "layer_index": i,
                "read_latency_ns": latency_ns * sf,
                "read_energy_nj": float(ld.get("read_energy_nj", 0.0) or 0.0) * sf,
                "buffer_latency_ns": float(ld.get("buffer_latency_ns", 0.0) or 0.0) * sf,
                "buffer_energy_nj": float(ld.get("buffer_energy_nj", 0.0) or 0.0) * sf,
                "ic_latency_ns": float(ld.get("ic_latency_ns", 0.0) or 0.0) * sf,
                "ic_energy_nj": float(ld.get("ic_energy_nj", 0.0) or 0.0) * sf,
                "core_latency_adc_ns": float(ld.get("core_latency_adc_ns", 0.0) or 0.0) * sf,
                "core_latency_accum_ns": float(ld.get("core_latency_accum_ns", 0.0) or 0.0) * sf,
                "core_latency_other_ns": float(ld.get("core_latency_other_ns", 0.0) or 0.0) * sf,
                "core_energy_adc_nj": float(ld.get("core_energy_adc_nj", 0.0) or 0.0) * sf,
                "core_energy_accum_nj": float(ld.get("core_energy_accum_nj", 0.0) or 0.0) * sf,
                "core_energy_other_nj": float(ld.get("core_energy_other_nj", 0.0) or 0.0) * sf,
                "array_latency_ns": float(ld.get("array_latency_ns", 0.0) or 0.0) * sf,
                "array_energy_nj": float(ld.get("array_energy_nj", 0.0) or 0.0) * sf,
                "adc_latency_precharger_ns": float(ld.get("adc_latency_precharger_ns", 0.0) or 0.0) * sf,
                "adc_latency_col_delay_ns": float(ld.get("adc_latency_col_delay_ns", 0.0) or 0.0) * sf,
                "adc_latency_senseamp_ns": float(ld.get("adc_latency_senseamp_ns", 0.0) or 0.0) * sf,
                "adc_latency_mlsa_ns": float(ld.get("adc_latency_mlsa_ns", 0.0) or 0.0) * sf,
                "adc_latency_mlencoder_ns": float(ld.get("adc_latency_mlencoder_ns", 0.0) or 0.0) * sf,
                "adc_latency_sar_ns": float(ld.get("adc_latency_sar_ns", 0.0) or 0.0) * sf,
                "adc_energy_precharger_nj": float(ld.get("adc_energy_precharger_nj", ld.get("adc_precharger_energy_nj", 0.0)) or 0.0) * sf,
                "adc_energy_senseamp_nj": float(ld.get("adc_energy_senseamp_nj", ld.get("adc_senseamp_energy_nj", 0.0)) or 0.0) * sf,
                "adc_energy_mlsa_nj": float(ld.get("adc_energy_mlsa_nj", ld.get("adc_mlsa_energy_nj", 0.0)) or 0.0) * sf,
                "adc_energy_mlencoder_nj": float(ld.get("adc_energy_mlencoder_nj", ld.get("adc_mlencoder_energy_nj", 0.0)) or 0.0) * sf,
                "adc_energy_sar_nj": float(ld.get("adc_energy_sar_nj", ld.get("adc_sar_energy_nj", 0.0)) or 0.0) * sf,
                "accum_latency_adder_ns": float(ld.get("accum_latency_adder_ns", 0.0) or 0.0) * sf,
                "accum_latency_dff_ns": float(ld.get("accum_latency_dff_ns", 0.0) or 0.0) * sf,
                "accum_latency_shiftadd_weight_ns": float(ld.get("accum_latency_shiftadd_weight_ns", 0.0) or 0.0) * sf,
                "accum_latency_shiftadd_input_ns": float(ld.get("accum_latency_shiftadd_input_ns", 0.0) or 0.0) * sf,
                "accum_latency_adder_tree_ns": float(ld.get("accum_latency_adder_tree_ns", 0.0) or 0.0) * sf,
                "accum_latency_tile_ns": float(ld.get("accum_latency_tile_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_latency_ns": float(ld.get("other_sub_tile_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_activation_latency_ns": float(ld.get("other_sub_activation_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_pooling_latency_ns": float(ld.get("other_sub_pooling_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_global_buffer_latency_ns": float(ld.get("other_sub_global_buffer_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_global_ic_latency_ns": float(ld.get("other_sub_global_ic_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_global_htree_latency_ns": float(ld.get("other_sub_global_htree_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_global_xybus_latency_ns": float(ld.get("other_sub_global_xybus_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_other_latency_ns": float(ld.get("other_sub_tile_pe_other_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_buffer_htree_latency_ns": float(ld.get("other_sub_tile_buffer_htree_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_activation_latency_ns": float(ld.get("other_sub_tile_activation_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_chip_activation_latency_ns": float(ld.get("other_sub_chip_activation_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_wl_decoder_latency_ns": float(ld.get("other_sub_tile_pe_wl_decoder_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_wl_driver_latency_ns": float(ld.get("other_sub_tile_pe_wl_driver_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_wl_switch_latency_ns": float(ld.get("other_sub_tile_pe_wl_switch_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_sl_switch_latency_ns": float(ld.get("other_sub_tile_pe_sl_switch_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_mux_latency_ns": float(ld.get("other_sub_tile_pe_mux_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_mux_decoder_latency_ns": float(ld.get("other_sub_tile_pe_mux_decoder_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_level_shifter_latency_ns": float(ld.get("other_sub_tile_pe_level_shifter_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_wl_path_latency_ns": float(ld.get("other_sub_tile_pe_wl_path_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_mux_path_latency_ns": float(ld.get("other_sub_tile_pe_mux_path_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_selected_path_latency_ns": float(ld.get("other_sub_tile_pe_selected_path_latency_ns", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_other_multiplier": float(ld.get("other_sub_tile_pe_other_multiplier", 0.0) or 0.0),
                "other_sub_tile_pe_num_col_muxed": float(ld.get("other_sub_tile_pe_num_col_muxed", 0.0) or 0.0),
                "other_sub_tile_pe_num_add": float(ld.get("other_sub_tile_pe_num_add", 0.0) or 0.0),
                "other_sub_tile_pe_num_row_activity": float(ld.get("other_sub_tile_pe_num_row_activity", 0.0) or 0.0),
                "other_sub_tile_energy_nj": float(ld.get("other_sub_tile_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_activation_energy_nj": float(ld.get("other_sub_activation_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_pooling_energy_nj": float(ld.get("other_sub_pooling_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_global_buffer_energy_nj": float(ld.get("other_sub_global_buffer_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_global_ic_energy_nj": float(ld.get("other_sub_global_ic_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_global_htree_energy_nj": float(ld.get("other_sub_global_htree_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_global_xybus_energy_nj": float(ld.get("other_sub_global_xybus_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_tile_pe_other_energy_nj": float(ld.get("other_sub_tile_pe_other_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_tile_buffer_htree_energy_nj": float(ld.get("other_sub_tile_buffer_htree_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_tile_activation_energy_nj": float(ld.get("other_sub_tile_activation_energy_nj", 0.0) or 0.0) * sf,
                "other_sub_chip_activation_energy_nj": float(ld.get("other_sub_chip_activation_energy_nj", 0.0) or 0.0) * sf,
                "accum_energy_adder_nj": float(ld.get("accum_energy_adder_nj", 0.0) or 0.0) * sf,
                "accum_energy_dff_nj": float(ld.get("accum_energy_dff_nj", 0.0) or 0.0) * sf,
                "accum_energy_shiftadd_weight_nj": float(ld.get("accum_energy_shiftadd_weight_nj", 0.0) or 0.0) * sf,
                "accum_energy_shiftadd_input_nj": float(ld.get("accum_energy_shiftadd_input_nj", 0.0) or 0.0) * sf,
                "accum_energy_adder_tree_nj": float(ld.get("accum_energy_adder_tree_nj", 0.0) or 0.0) * sf,
                "accum_energy_tile_nj": float(ld.get("accum_energy_tile_nj", 0.0) or 0.0) * sf,
            })

        # Distribute chip area proportionally by tile count if available
        tile_counts_area = _per_layer_tile_counts_from_neurosim_matrix(
            setup.get("num_tile_each_layer")
        )
        layer_area_um2: Optional[List[float]] = None
        if tile_counts_area and total_area_um2 > 0:
            n_area = min(len(tile_counts_area), len(conv_linear_list))
            subset_tc = tile_counts_area[:n_area]
            total_tiles = sum(subset_tc)
            if total_tiles > 0:
                layer_area_um2 = [
                    total_area_um2 * tc / total_tiles for tc in subset_tc
                ]

        rram_tile_stats = build_rram_tile_stats_from_pybind_setup(
            setup,
            conv_linear_list,
            subarray=subarray,
            parallel_read=parallel_read,
        )

        return NeuroSimFullResult(
            layer_latency_ns=layer_latency_ns,
            layer_energy_nj=layer_energy_nj,
            success=True,
            clock_period_ns=clock_period_ns,
            total_area_um2=total_area_um2 if total_area_um2 > 0 else None,
            layer_area_um2=layer_area_um2,
            layer_area_source="neurosim_pybind_chiparea_weighted_by_tile_count" if layer_area_um2 else None,
            rram_tile_stats=rram_tile_stats,
            workload_scale_factors=list(scale_factors),
            layer_breakdown=layer_breakdown,
        )

    except Exception as e:
        import traceback
        return NeuroSimFullResult(
            layer_latency_ns=[], layer_energy_nj=[], success=False,
            error=f"neurosim_core pybind error: {e}\n{traceback.format_exc()}",
        )


def run_neurosim_full(
    conv_linear_list: List[ConvLinearLayerInfo],
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch",
    wl_weight: int = 8,
    wl_activation: int = 8,
    subarray: int = 128,
    subarray_cols: Optional[int] = None,
    parallel_read: int = 128,
    timeout_s: Optional[int] = None,
    clk_freq_hz: Optional[float] = None,
    technode: Optional[int] = None,
    adc_clk_freq_hz: Optional[float] = None,
    sar_column_power_w: Optional[float] = None,
    num_col_muxed: Optional[int] = None,
    tile_buffer_core_size_row: Optional[int] = None,
    tile_buffer_core_size_col: Optional[int] = None,
    force_tile_size_cm: Optional[int] = None,
    force_pe_size_cm: Optional[int] = None,
    force_num_tiles_cm: Optional[int] = None,
    neurosim_relu_mode: Optional[int] = None,
    neurosim_chip_activation_mode: Optional[int] = None,
    neurosim_enable_activation: Optional[int] = None,
    clk_mode: str = "capped",
    debug_export_dir: Optional[str] = None,
    pad_k_alignment: Optional[int] = None,
    max_in_vectors: Optional[int] = None,
    progress_interval_s: Optional[float] = None,
    progress_log_info: Optional[Any] = None,
) -> NeuroSimFullResult:
    """
    对 conv_linear_list 中每一层生成 NeuroSim config 行与 weight/input CSV，
    调用 NeuroSIM/main 一次，解析每层 readLatency + buffer latency、readDynamicEnergy + buffer energy，
    转为 latency_ns 与 energy_nj 列表，供注入 `cim_full_context`。
    clk_freq_hz: 可选，工作频率上限(Hz)，如 100e6=100MHz；传入 main 实现频率控制。
    technode: 可选，工艺节点(nm)；NeuroSim Technology 仅支持 130/90/65/45/32/22/14/10/7，28 会自动改为 32。
    adc_clk_freq_hz: 可选，SAR ADC 专用时钟(Hz)，传入 main。
    sar_column_power_w: 可选，SAR ADC Column_Power (W)，传入 main。
    num_col_muxed: 可选，numColMuxed（多少列共享 1 个 ADC/SenseAmp）；不设则使用 NeuroSim 默认。
    tile_buffer_core_size_row / tile_buffer_core_size_col:
      可选，覆盖 NeuroSim Param.cpp 中 tileBufferCoreSizeRow/Col，用于扫描 tile buffer core 容量与 interface width。
    force_tile_size_cm / force_pe_size_cm / force_num_tiles_cm:
      可选，强制 Conventional Mapping 的 tile/PE 尺寸与 tile 总数预算，用于 iso-hardware 对比。
    neurosim_relu_mode / neurosim_chip_activation_mode / neurosim_enable_activation:
      可选，传入 NeuroSIM/main 末尾 tri-state 开关（0/1 覆盖，None 表示不覆盖）；仅 subprocess 路径生效。
    clk_mode:
      - capped（默认）：保持当前行为（用户可用 clk_freq_hz 降频，上限通过 main/配置生效）。
      - critical：忽略用户频率上限，尝试按 stdout 中的关键路径 clkPeriod 对 per-layer latency 做比例缩放，
        用于“按最坏关键路径频率”生成一份对比用 latency（能耗不变）。
    """
    if not conv_linear_list:
        return NeuroSimFullResult(
            layer_latency_ns=[],
            layer_energy_nj=[],
            success=True,
        )

    # NeuroSim Technology.cpp 仅支持 130,90,65,45,32,22,14,10,7 nm；28nm 会触发 exit(-1)
    if technode is not None and technode == 28:
        technode = 32

    main_dir = os.path.join(neurosim_root, "NeuroSIM")
    if not os.path.isdir(main_dir):
        return NeuroSimFullResult(
            layer_latency_ns=[],
            layer_energy_nj=[],
            success=False,
            error=f"NeuroSIM 目录不存在: {main_dir}",
        )

    config_rows: List[str] = []
    weight_input_pairs: List[Tuple[str, str]] = []
    subprocess_scale_factors: List[float] = []   # per-layer scale: actual/capped in_w
    k_alignment = int(pad_k_alignment or 1)
    # 在 NeuroSIM 目录下创建临时目录，便于 main 用相对路径读取（cwd=main_dir）
    tmpdir = tempfile.mkdtemp(dir=main_dir, prefix="pointkan_neurosim_")
    rel_prefix = os.path.basename(tmpdir)

    try:
        for idx, info in enumerate(conv_linear_list):
            if info.layer_type == "conv":
                raw_in_w = max(1, info.length)
                # Cap W_in for NeuroSim efficiency: PointNet++ SA Conv2d can have
                # length = N_points × K_neighbors = 16384+, causing 16k input-vector
                # evaluations in NeuroSim which is extremely slow.  We simulate with a
                # capped value and scale the resulting latency/energy back to the true
                # workload size.
                if max_in_vectors and max_in_vectors > 0:
                    in_w = min(raw_in_w, max_in_vectors)
                else:
                    in_w = raw_in_w
                subprocess_scale_factors.append(float(raw_in_w) / float(in_w))
                in_h = 1
                in_c = max(1, info.in_channels)
                out_c = max(1, info.out_channels)
                k_h, k_w = 1, max(1, info.kernel_size)
                stride = 1
                pool = 0
                num_patches = (in_w - k_w) // stride + 1
                K = in_c * k_h * k_w
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                input_stretched = _stretch_input_conv1d(in_c, in_w, k_w, stride)
                padded_weight, padded_input, original_k, padded_k = _pad_k_dimension_to_alignment(
                    weight_data,
                    input_stretched,
                    alignment=k_alignment,
                    kernel_size=k_w,
                )
                effective_in_c = in_c if padded_k == original_k else int(padded_k // (k_h * k_w))
                config_rows.append(
                    f"{in_h},{in_w},{effective_in_c},{k_h},{k_w},{out_c},{pool},{stride}"
                )
                _write_matrix_weight(padded_weight.reshape(out_c, -1), weight_path)
                _write_matrix_activation_conv(padded_input, wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))
            else:
                # fc/linear: in_h=1, in_w=tokens, in_c=in_f, k_h=1, k_w=1, out_c=out_f
                in_f = max(1, info.in_features)
                out_f = max(1, info.out_features)
                raw_tokens = max(1, info.tokens)
                if max_in_vectors and max_in_vectors > 0:
                    M = min(raw_tokens, max_in_vectors)
                else:
                    M = raw_tokens
                subprocess_scale_factors.append(float(raw_tokens) / float(M))
                in_h, in_w = 1, M
                in_c, out_c = in_f, out_f
                k_h, k_w = 1, 1
                pool, stride = 0, 1
                weight_path = os.path.join(tmpdir, f"weight{idx}.csv")
                input_path = os.path.join(tmpdir, f"input{idx}.csv")
                num_patches = (in_w - k_w) // stride + 1
                K = in_c * k_h * k_w
                weight_data = np.random.randn(out_c, in_c, k_h, k_w).astype(np.float32)
                input_stretched = np.random.randn(1, num_patches, K).astype(np.float32)
                padded_weight, padded_input, original_k, padded_k = _pad_k_dimension_to_alignment(
                    weight_data,
                    input_stretched,
                    alignment=k_alignment,
                    kernel_size=k_w,
                )
                effective_in_c = in_c if padded_k == original_k else padded_k
                config_rows.append(
                    f"{in_h},{in_w},{effective_in_c},{k_h},{k_w},{out_c},{pool},{stride}"
                )
                _write_matrix_weight(padded_weight.reshape(out_c, -1), weight_path)
                _write_matrix_activation_conv(padded_input, wl_activation, input_path)
                weight_input_pairs.append((os.path.join(rel_prefix, f"weight{idx}.csv"), os.path.join(rel_prefix, f"input{idx}.csv")))

        config_path = os.path.join(tmpdir, "config.csv")
        with open(config_path, "w") as f:
            f.write("\n".join(config_rows) + "\n")

        if debug_export_dir:
            export_dir = os.path.abspath(debug_export_dir)
            os.makedirs(export_dir, exist_ok=True)
            shutil.copy2(config_path, os.path.join(export_dir, "config.csv"))
            with open(os.path.join(export_dir, "weight_input_pairs.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["layer_index", "name", "layer_type", "weight_csv", "input_csv"])
                for idx, info in enumerate(conv_linear_list):
                    writer.writerow([
                        idx,
                        getattr(info, "name", ""),
                        getattr(info, "layer_type", ""),
                        f"weight{idx}.csv",
                        f"input{idx}.csv",
                    ])
            with open(os.path.join(export_dir, "layer_shapes.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "layer_index",
                    "name",
                    "layer_type",
                    "input_shape",
                    "output_shape",
                    "in_channels",
                    "out_channels",
                    "kernel_size",
                    "length",
                    "in_features",
                    "out_features",
                    "tokens",
                    "indata_bytes",
                    "rdata_bytes",
                ])
                for idx, info in enumerate(conv_linear_list):
                    writer.writerow([
                        idx,
                        getattr(info, "name", ""),
                        getattr(info, "layer_type", ""),
                        list(getattr(info, "input_shape", ()) or ()),
                        list(getattr(info, "output_shape", ()) or ()),
                        getattr(info, "in_channels", 0),
                        getattr(info, "out_channels", 0),
                        getattr(info, "kernel_size", 0),
                        getattr(info, "length", 0),
                        getattr(info, "in_features", 0),
                        getattr(info, "out_features", 0),
                        getattr(info, "tokens", 0),
                        getattr(info, "indata_bytes", 0.0),
                        getattr(info, "rdata_bytes", 0.0),
                    ])
            for idx in range(len(conv_linear_list)):
                shutil.copy2(os.path.join(tmpdir, f"weight{idx}.csv"), os.path.join(export_dir, f"weight{idx}.csv"))
                shutil.copy2(os.path.join(tmpdir, f"input{idx}.csv"), os.path.join(export_dir, f"input{idx}.csv"))

        rel_config = os.path.join(rel_prefix, "config.csv")
        ok, out, err = _run_neurosim_main(
            rel_config,
            weight_input_pairs,
            neurosim_root,
            wl_weight=wl_weight,
            wl_activation=wl_activation,
            subarray=subarray,
            subarray_cols=subarray_cols,
            parallel_read=parallel_read,
            timeout_s=timeout_s,
            clk_freq_hz=clk_freq_hz,
            technode=technode,
            adc_clk_freq_hz=adc_clk_freq_hz,
            sar_column_power_w=sar_column_power_w,
            num_col_muxed=num_col_muxed,
            tile_buffer_core_size_row=tile_buffer_core_size_row,
            tile_buffer_core_size_col=tile_buffer_core_size_col,
            force_tile_size_cm=force_tile_size_cm,
            force_pe_size_cm=force_pe_size_cm,
            force_num_tiles_cm=force_num_tiles_cm,
            neurosim_relu_mode=neurosim_relu_mode,
            neurosim_chip_activation_mode=neurosim_chip_activation_mode,
            neurosim_enable_activation=neurosim_enable_activation,
            progress_interval_s=progress_interval_s,
            progress_log_info=progress_log_info,
        )

        if not ok or not out:
            err_msg = err or "NeuroSim main failed or no stdout"
            # NeuroSim often reports fatal config/technode issues in stdout while
            # stderr stays empty. Surface a concise tail to make debugging visible.
            if out:
                tail = "\n".join([ln for ln in out.splitlines()[-20:] if ln.strip()])
                if tail:
                    err_msg += "; stdout_tail: " + tail[:2000]
            return NeuroSimFullResult(
                layer_latency_ns=[],
                layer_energy_nj=[],
                success=False,
                error=err_msg,
            )

        parsed = _parse_layer_results(out)
        if len(parsed) != len(conv_linear_list):
            return NeuroSimFullResult(
                layer_latency_ns=[],
                layer_energy_nj=[],
                success=False,
                error=f"NeuroSim parsed {len(parsed)} layers, expected {len(conv_linear_list)}",
            )

        layer_latency_ns = []
        layer_energy_nj = []
        for i, (read_ns, read_pj, buf_ns, buf_pj) in enumerate(parsed):
            # DNN_NeuroSim main's per-layer readLatency/readDynamicEnergy already
            # include buffer + IC contributions in the synchronous full-network path.
            # Keep the buffer values only as diagnostics and avoid adding them again.
            # Apply scale factor to restore the actual workload size (see max_in_vectors cap).
            sf = subprocess_scale_factors[i] if i < len(subprocess_scale_factors) else 1.0
            layer_latency_ns.append(read_ns * sf)
            layer_energy_nj.append(read_pj / 1000.0 * sf)
        layer_breakdown = _scale_layer_breakdown_items(
            _parse_neurosim_layer_breakdown(out, len(conv_linear_list)),
            subprocess_scale_factors,
        )

        # clock_period_ns: 默认用于上层记录/展示；critical 模式下会优先使用关键路径周期
        clock_period_ns = parse_clock_period_ns(out)
        critical_period_ns = parse_critical_clk_period_ns(out)
        chip_period_ns = parse_chip_clk_period_ns(out)
        chip_pipeline_latency_ns = _parse_chip_pipeline_latency_ns(out)
        chip_pipeline_buffer_latency_ns = _parse_chip_pipeline_buffer_latency_ns(out)
        chip_pipeline_ic_latency_ns = _parse_chip_pipeline_ic_latency_ns(out)

        normalized_clk_mode = str(clk_mode or "capped").strip().lower()
        if normalized_clk_mode == "critical" and critical_period_ns and critical_period_ns > 0:
            # 当前 per-layer readLatency 通常与用户设定 chip clock period（或其上限）线性相关；
            # 若用户传入的周期更慢（降频），则按 critical_period / used_period 比例缩短 latency。
            used_period_ns = None
            if clk_freq_hz is not None and float(clk_freq_hz) > 0:
                used_period_ns = 1e9 / float(clk_freq_hz)
            elif chip_period_ns is not None and float(chip_period_ns) > 0:
                used_period_ns = float(chip_period_ns)

            if used_period_ns and used_period_ns > 0 and used_period_ns > float(critical_period_ns):
                sf_clk = float(critical_period_ns) / float(used_period_ns)
                layer_latency_ns = [float(x) * sf_clk for x in layer_latency_ns]

                def _scale_latency_fields(items):
                    if not items:
                        return items
                    scaled = []
                    for row in items:
                        rr = dict(row)
                        for k, v in row.items():
                            if isinstance(v, (int, float)) and ("latency_ns" in str(k)):
                                rr[k] = float(v) * sf_clk
                        scaled.append(rr)
                    return scaled

                layer_breakdown = _scale_latency_fields(layer_breakdown)
                if chip_pipeline_latency_ns is not None:
                    chip_pipeline_latency_ns = float(chip_pipeline_latency_ns) * sf_clk
                if chip_pipeline_buffer_latency_ns is not None:
                    chip_pipeline_buffer_latency_ns = float(chip_pipeline_buffer_latency_ns) * sf_clk
                if chip_pipeline_ic_latency_ns is not None:
                    chip_pipeline_ic_latency_ns = float(chip_pipeline_ic_latency_ns) * sf_clk

            # 对外汇报：critical 模式下记录关键路径周期（若可得）
            clock_period_ns = float(critical_period_ns)

        rram_tile_stats = _parse_rram_tile_stats(out, conv_linear_list)
        total_area_um2 = _parse_total_area_um2(out)
        layer_area_um2 = _estimate_layer_area_um2_from_tiles(total_area_um2, rram_tile_stats, conv_linear_list)
        layer_area_source = None
        if layer_area_um2 is not None:
            layer_area_source = "neurosim_chiparea_weighted_by_tile_count"

        return NeuroSimFullResult(
            layer_latency_ns=layer_latency_ns,
            layer_energy_nj=layer_energy_nj,
            success=True,
            clock_period_ns=clock_period_ns,
            layer_area_um2=layer_area_um2,
            total_area_um2=total_area_um2,
            layer_area_source=layer_area_source,
            rram_tile_stats=rram_tile_stats,
            chip_pipeline_latency_ns=chip_pipeline_latency_ns,
            chip_pipeline_buffer_latency_ns=chip_pipeline_buffer_latency_ns,
            chip_pipeline_ic_latency_ns=chip_pipeline_ic_latency_ns,
            workload_scale_factors=list(subprocess_scale_factors),
            layer_breakdown=layer_breakdown,
        )
    finally:
        if os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
