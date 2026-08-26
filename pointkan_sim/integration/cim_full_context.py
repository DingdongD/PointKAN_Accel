"""
全网络 CIM（当前实现为 NeuroSim）注入上下文。

调度器 `dispatch_layer` 与各入口（`simulate_network`、真实数据评估）共用同一套 dict 键名，
避免散落魔法字符串。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# --- 注入 dict 键名（acc / pointcim 全网络 Conv1d、Linear）---
CIM_LAYER_LATENCY_NS = "cim_layer_latency_ns"
CIM_LAYER_ENERGY_NJ = "cim_layer_energy_nj"
CIM_LAYER_INDEX = "cim_layer_index"
CIM_CLOCK_PERIOD_NS = "cim_clock_period_ns"
CIM_LAYER_AREA_UM2 = "cim_layer_area_um2"
CIM_LAYER_NAMES = "cim_layer_names"
CIM_LAT_ENERGY_BY_NAME = "cim_lat_energy_by_name"
CIM_FULL_NETWORK_ACTIVE = "cim_full_network_active"
CIM_FULL_BACKEND = "cim_full_backend"
ACC_CIM_KEY_PRELOADED = "acc_cim_key_preloaded"


def build_cim_full_context(
    *,
    layer_latency_ns: List[float],
    layer_energy_nj: List[float],
    clock_period_ns: Optional[float] = None,
    layer_area_um2: Optional[List[float]] = None,
    layer_names: Optional[List[str]] = None,
    lat_energy_by_name: Optional[Dict[str, Tuple[float, float]]] = None,
    full_network_active: bool = True,
    backend: str = "neurosim",
) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        CIM_LAYER_LATENCY_NS: layer_latency_ns,
        CIM_LAYER_ENERGY_NJ: layer_energy_nj,
        CIM_LAYER_INDEX: 0,
        CIM_FULL_NETWORK_ACTIVE: full_network_active,
        CIM_FULL_BACKEND: backend,
    }
    if clock_period_ns is not None and float(clock_period_ns) > 0:
        ctx[CIM_CLOCK_PERIOD_NS] = float(clock_period_ns)
    if layer_area_um2 is not None:
        ctx[CIM_LAYER_AREA_UM2] = layer_area_um2
    if layer_names is not None:
        ctx[CIM_LAYER_NAMES] = layer_names
    if lat_energy_by_name is not None:
        ctx[CIM_LAT_ENERGY_BY_NAME] = lat_energy_by_name
    return ctx


def build_cim_full_context_from_neurosim_lists(
    conv_linear_list: List[Any],
    *,
    layer_latency_ns: List[float],
    layer_energy_nj: List[float],
    clock_period_ns: Optional[float] = None,
    layer_area_um2: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """由真实前向层列表 + NeuroSim 逐层结果构造注入上下文（含按层名的 latency/energy 映射）。"""
    n = min(len(layer_latency_ns), len(layer_energy_nj))
    lat_ns = layer_latency_ns[:n]
    en_nj = layer_energy_nj[:n]
    slice_list = conv_linear_list[:n]
    names = [getattr(c, "name", "") or f"layer_{i}" for i, c in enumerate(slice_list)]
    lat_energy_by_name = {
        getattr(c, "name", "") or f"layer_{i}": (float(lat), float(en))
        for i, (c, lat, en) in enumerate(zip(slice_list, lat_ns, en_nj))
    }
    area_slice: Optional[List[float]] = None
    if layer_area_um2 is not None:
        area_slice = [float(x) for x in layer_area_um2[:n]]
    return build_cim_full_context(
        layer_latency_ns=lat_ns,
        layer_energy_nj=en_nj,
        clock_period_ns=clock_period_ns,
        layer_area_um2=area_slice,
        layer_names=names,
        lat_energy_by_name=lat_energy_by_name,
    )
