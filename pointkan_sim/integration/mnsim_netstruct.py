"""
MNSIM-2.0 full NetStruct integration (Tile/NoC-level).

Builds a synthetic NetStruct from PointKAN conv1d/linear operators only,
runs MNSIM Model_latency and Model_energy to get per-layer latency (ns) and
energy (nJ), then these can be used when dispatching conv1d/linear to emit
one Event per layer with the MNSIM-derived values.

Conv/Linear 与 MNSIM 的映射（W8A8、矩阵乘形式）：
- Conv1d：映射为 MNSIM conv，空间 (1, L)；等价于 (out_ch, in_ch*kernel) × (in_ch*kernel, L)
  的矩阵乘，通过 TCG 做 Tile/NoC 级仿真。Inputbit/Weightbit/outputbit = 8，即 W8A8；MNSIM 侧为 INT 计算。
- Linear：映射为 MNSIM fc，Infeature×Outfeature 矩阵乘，同样 8bit 输入/权/输出，Tile/NoC 仿真。
若需在仿真中显式计入量化和反量化（FP32↔INT8）的 cycle/功耗，请启用 hw_config.enable_w8a8_quant_dequant，
并在 logic_circuit.quant_dequant 中配置 power/cycles。
"""

from __future__ import annotations

import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..network.pointkan_net import LayerDef, OperatorRef


@dataclass
class MNSIMFullResult:
    """Per-layer results from MNSIM Model_latency + Model_energy."""
    layer_latency_ns: List[float] = field(default_factory=list)
    layer_energy_nj: List[float] = field(default_factory=list)
    conv_linear_order: List[Tuple[str, int]] = field(default_factory=list)  # (layer_name, op_index)
    success: bool = False
    error: Optional[str] = None


def _mnsim_path() -> str:
    return os.environ.get("MNSIM_ROOT", "/workspace/MNSIM-2.0")


def _ensure_mnsim() -> None:
    root = _mnsim_path()
    if root not in sys.path:
        sys.path.insert(0, root)


def _layer_dict_conv(
    in_ch: int,
    out_ch: int,
    length: int,
    kernel_size: int = 1,
    layer_index: int = 0,
    input_index: int = -1,
    output_index: int = 1,
    input_bit: int = 8,
    weight_bit: int = 8,
    output_bit: int = 8,
) -> OrderedDict:
    """Build MNSIM conv layer_dict (Conv1d mapped to conv with spatial [1, L])."""
    # MNSIM conv: Inputsize/Outputsize are 2D (H, W). Conv1d: (1, L).
    inp_h, inp_w = 1, max(1, length)
    out_h, out_w = 1, max(1, length)
    return OrderedDict([
        ("type", "conv"),
        ("Inputchannel", in_ch),
        ("Inputsize", [inp_h, inp_w]),
        ("Kernelsize", kernel_size),
        ("Stride", 1),
        ("Padding", 0),
        ("Depthwise", "normal"),
        ("Outputchannel", out_ch),
        ("Outputsize", [out_h, out_w]),
        ("Inputbit", input_bit),
        ("Weightbit", weight_bit),
        ("outputbit", output_bit),
        ("row_split_num", 1),
        ("weight_bit_split_part", 8),
        ("Inputindex", [input_index]),
        ("Outputindex", [output_index]),
        ("Layerindex", layer_index),
    ])


def _layer_dict_fc(
    in_f: int,
    out_f: int,
    layer_index: int = 0,
    input_index: int = -1,
    output_index: int = 1,
    input_bit: int = 8,
    weight_bit: int = 8,
    output_bit: int = 8,
) -> OrderedDict:
    """Build MNSIM fc layer_dict."""
    return OrderedDict([
        ("type", "fc"),
        ("Infeature", in_f),
        ("Outfeature", out_f),
        ("Inputbit", input_bit),
        ("Weightbit", weight_bit),
        ("outputbit", output_bit),
        ("row_split_num", 1),
        ("weight_bit_split_part", 8),
        ("Inputindex", [input_index]),
        ("Outputindex", [output_index]),
        ("Layerindex", layer_index),
    ])


def build_netstruct_from_layers(
    layers: List["LayerDef"],
) -> Tuple[List[Any], List[Tuple[str, int]]]:
    """
    Build MNSIM NetStruct and ordered list of (layer_name, op_index) for conv1d/linear only.

    Returns:
        NetStruct: list of [(layer_dict, None)] for each conv1d and linear in order.
        conv_linear_order: list of (layer_name, operator_index_in_layer) for each entry.
    """
    netstruct: List[Any] = []
    conv_linear_order: List[Tuple[str, int]] = []
    layer_index = 0

    for layer in layers:
        if not layer.operators:
            continue
        for op_idx, op_ref in enumerate(layer.operators):
            if op_ref.op_type not in ("conv1d", "linear"):
                continue
            params = op_ref.params
            if op_ref.op_type == "conv1d":
                in_ch = int(params.get("in_channels", 1))
                out_ch = int(params.get("out_channels", 1))
                kernel = int(params.get("kernel_size", 1))
                num_points = int(params.get("num_points", 1))
                ld = _layer_dict_conv(
                    in_ch, out_ch, num_points, kernel,
                    layer_index=layer_index,
                    input_index=-1,  # 首层表示外部输入，非首层表示前一层（layer_id-1）
                    output_index=1,
                )
            else:
                in_f = int(params.get("in_features", 1))
                out_f = int(params.get("out_features", 1))
                ld = _layer_dict_fc(
                    in_f, out_f,
                    layer_index=layer_index,
                    input_index=-1,  # 同上
                    output_index=1,
                )
            netstruct.append([(ld, None)])
            conv_linear_order.append((layer.name, op_idx))
            layer_index += 1

    return netstruct, conv_linear_order


def run_mnsim_full(
    netstruct: List[Any],
    simconfig_path: Optional[str] = None,
    use_pipeline: bool = False,
    conv_linear_order: Optional[List[Tuple[str, int]]] = None,
) -> MNSIMFullResult:
    """
    Run MNSIM Model_latency and Model_energy on the given NetStruct.

    Returns per-layer latency (ns) and energy (nJ) in the same order as NetStruct.
    conv_linear_order: 可选，与 netstruct 同序的 (layer_name, op_idx) 列表，用于日志与报告。
    """
    if not netstruct:
        return MNSIMFullResult(success=True)

    if simconfig_path is None:
        simconfig_path = os.path.join(_mnsim_path(), "SimConfig.ini")
    if not os.path.isfile(simconfig_path):
        return MNSIMFullResult(
            success=False,
            error=f"SimConfig not found: {simconfig_path}",
        )

    try:
        _ensure_mnsim()
        from MNSIM.Mapping_Model.Tile_connection_graph import TCG
        from MNSIM.Latency_Model.Model_latency import Model_latency
        from MNSIM.Power_Model.Model_inference_power import Model_inference_power
        from MNSIM.Energy_Model.Model_energy import Model_energy
    except Exception as e:
        return MNSIMFullResult(success=False, error=str(e))

    try:
        tcg = TCG(netstruct, simconfig_path)
        model_latency = Model_latency(
            netstruct,
            simconfig_path,
            TCG_mapping=tcg,
        )
        if use_pipeline:
            model_latency.calculate_model_latency(mode=2)
        else:
            model_latency.calculate_model_latency_nopipe()

        model_power = Model_inference_power(
            netstruct,
            simconfig_path,
            TCG_mapping=tcg,
        )
        model_energy = Model_energy(
            netstruct,
            simconfig_path,
            TCG_mapping=tcg,
            model_latency=model_latency,
            model_power=model_power,
        )

        n = len(netstruct)
        # 校验 MNSIM 返回的列表长度与可访问性，避免 empty sequence 等错误
        def _len(x: Any) -> int:
            return len(x) if x is not None else 0
        if _len(model_latency.total_buffer_latency) < n or _len(model_latency.total_computing_latency) < n:
            return MNSIMFullResult(
                success=False,
                error="MNSIM total_buffer_latency/total_computing_latency 长度不足",
            )
        if _len(model_energy.arch_energy) < n:
            return MNSIMFullResult(
                success=False,
                error="MNSIM arch_energy 长度不足",
            )

        latency_ns: List[float] = []
        energy_nj: List[float] = []

        for i in range(n):
            # Total layer latency: sum of component latencies (MNSIM uses ns)
            total_lat = (
                model_latency.total_buffer_latency[i]
                + model_latency.total_computing_latency[i]
                + model_latency.total_digital_latency[i]
                + model_latency.total_intra_tile_latency[i]
                + model_latency.total_inter_tile_latency[i]
            )
            latency_ns.append(float(total_lat))
            energy_nj.append(float(model_energy.arch_energy[i]))

        return MNSIMFullResult(
            layer_latency_ns=latency_ns,
            layer_energy_nj=energy_nj,
            conv_linear_order=conv_linear_order or [],
            success=True,
        )
    except Exception as e:
        return MNSIMFullResult(success=False, error=str(e))
