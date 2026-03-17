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
并在 logic_circuit.quant_pipeline_8bit / dequant_pipeline 中配置 power、pipeline_latency、parallelism。

real_indata_bytes / real_rdata_bytes（可选，按层传入）：
- indata：该层输入激活数据量（字节）。Conv1d 对应输入 [B,C_in,L] 的 numel×inputbit/8；
  Linear 对应输入 [B,in_f] 或 [B,T,in_f] 的 numel×inputbit/8。符合「读入 iReg 的输入数据量」。
- rdata：该层权重量（字节）。Conv1d 对应 [C_out,C_in,K]；Linear 对应 [out_f,in_f]。符合矩阵乘的权重复用/读取量。
- 在 MNSIM 中仅用于 Model_latency 的 buffer 读/写延迟（update_tile_latency）；Crossbar 的计算延迟与功耗
  仍由结构（read_row/read_column）决定，与数据分布无关。遵循 MNSIM-2.0 main.py 的推理仿真原则，不做任何后处理缩放。
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
    """Per-layer results from MNSIM Model_latency + Model_energy + Model_area."""
    layer_latency_ns: List[float] = field(default_factory=list)
    layer_energy_nj: List[float] = field(default_factory=list)
    conv_linear_order: List[Tuple[str, int]] = field(default_factory=list)  # (layer_name, op_index)
    success: bool = False
    error: Optional[str] = None
    # RRAM 阵列 / PE / Tile 统计（来自 TCG），仅当 success 且运行过 MNSIM 时非空
    rram_tile_stats: Optional[Dict[str, Any]] = None
    # 面积（um^2）：来自 Model_area，与 netstruct 层序一致
    layer_area_um2: List[float] = field(default_factory=list)
    total_area_um2: float = 0.0


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
    real_data_overrides: Optional[List[Dict[str, float]]] = None,
) -> Tuple[List[Any], List[Tuple[str, int]]]:
    """
    Build MNSIM NetStruct and ordered list of (layer_name, op_index) for conv1d/linear only.

    real_data_overrides: 可选，与 conv/linear 顺序一致的列表，每项为
        {"real_indata_bytes": float, "real_rdata_bytes": float}，用于按层真实权重/激活数据量驱动 MNSIM。

    Returns:
        NetStruct: list of [(layer_dict, None)] for each conv1d and linear in order.
        conv_linear_order: list of (layer_name, operator_index_in_layer) for each entry.
    """
    netstruct: List[Any] = []
    conv_linear_order: List[Tuple[str, int]] = []
    layer_index = 0
    overrides = real_data_overrides or []

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
            if layer_index < len(overrides) and isinstance(overrides[layer_index], dict):
                o = overrides[layer_index]
                if "real_indata_bytes" in o and "real_rdata_bytes" in o:
                    ld["real_indata_bytes"] = float(o["real_indata_bytes"])
                    ld["real_rdata_bytes"] = float(o["real_rdata_bytes"])
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
        from MNSIM.Area_Model.Model_Area import Model_area
    except Exception as e:
        return MNSIMFullResult(success=False, error=str(e))

    try:
        tcg = TCG(netstruct, simconfig_path)
        # 统计 RRAM 阵列 / PE / Tile 数量与尺寸（仅 conv/fc 使用 crossbar）
        rram_tile_stats: Optional[Dict[str, Any]] = None
        try:
            total_tiles = 0
            total_PE = 0
            pe_per_tile = int(getattr(tcg.tile, "tile_PE_total_num", 0)) or 1
            xbar_row = int(getattr(tcg.tile, "xbar_row", 0))
            xbar_column = int(getattr(tcg.tile, "xbar_column", 0))
            if hasattr(tcg.tile, "xbar_size") and not (xbar_row and xbar_column):
                xs = tcg.tile.xbar_size
                if isinstance(xs, (list, tuple)) and len(xs) >= 2:
                    xbar_row, xbar_column = int(xs[0]), int(xs[1])
            per_layer: List[Dict[str, Any]] = []
            for i in range(len(tcg.layer_tileinfo)):
                info = tcg.layer_tileinfo[i]
                layer_type = info.get("type", "")
                if layer_type not in ("conv", "fc"):
                    per_layer.append({"layer_index": i, "layer_type": layer_type, "tilenum": 0, "PE_count": 0, "max_row": 0, "max_column": 0})
                    continue
                tilenum = int(info.get("tilenum", 0))
                PEnum = int(info.get("PEnum", 0))
                max_row = int(info.get("max_row", 0))
                max_column = int(info.get("max_column", 0))
                total_tiles += tilenum
                total_PE += PEnum
                per_layer.append({
                    "layer_index": i,
                    "layer_type": layer_type,
                    "tilenum": tilenum,
                    "PE_count": PEnum,
                    "max_row": max_row,
                    "max_column": max_column,
                })
            rram_tile_stats = {
                "total_tile_count": total_tiles,
                "total_PE_count": total_PE,
                "xbar_size": {"row": xbar_row, "column": xbar_column},
                "pe_per_tile": pe_per_tile,
                "per_layer": per_layer,
            }
        except Exception:  # 不因统计失败影响主流程
            rram_tile_stats = None

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

        # 面积：Model_area 与 TCG 同序，得到每层面积与总面积
        layer_area_um2: List[float] = []
        total_area_um2 = 0.0
        try:
            model_area = Model_area(
                NetStruct=netstruct,
                SimConfig_path=simconfig_path,
                TCG_mapping=tcg,
            )
            for i in range(n):
                layer_area_um2.append(float(getattr(model_area, "arch_area", [0] * n)[i]))
            total_area_um2 = float(getattr(model_area, "arch_total_area", 0) or 0)
        except Exception:
            layer_area_um2 = [0.0] * n
            total_area_um2 = 0.0

        return MNSIMFullResult(
            layer_latency_ns=latency_ns,
            layer_energy_nj=energy_nj,
            conv_linear_order=conv_linear_order or [],
            success=True,
            rram_tile_stats=rram_tile_stats,
            layer_area_um2=layer_area_um2,
            total_area_um2=total_area_um2,
        )
    except Exception as e:
        return MNSIMFullResult(success=False, error=str(e))
