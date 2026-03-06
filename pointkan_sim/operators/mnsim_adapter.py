"""
MNSIM 适配器（PE 级延迟与功耗估算）
====================================

通过 MNSIM 的 PE 级接口估算 Conv1d/Linear 的延迟与功耗，供无全 NetStruct 时
common_ops.gen_conv1d_rram_cim_events / gen_linear_rram_cim_events 使用。
若 MNSIM 调用失败则回退到保守解析模型。
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class MNSIMEstimate:
    """单次 MNSIM 或回退估算结果：周期、延迟(ns)、能耗(nJ)、平均功耗(mW)、来源标识。"""
    cycles: int
    latency_ns: float
    energy_nj: float
    avg_power_mw: float
    source: str


def _ensure_mnsim_importable(mnsim_root: str) -> None:
    """将 MNSIM 根目录加入 sys.path 以便 import MNSIM。"""
    if mnsim_root not in sys.path:
        sys.path.insert(0, mnsim_root)


def estimate_rram_cim_op(
    *,
    in_features: int,
    out_features: int,
    tokens: int,
    clock_period_ns: float,
    cim_parallelism: int,
    mnsim_root: str = "/workspace/MNSIM-2.0",
    simconfig_path: Optional[str] = None,
    fallback_power_mw: float = 25.0,
    fallback_cycles_per_token: int = 8,
) -> MNSIMEstimate:
    """
    用 MNSIM PE 级接口估算 Conv1d/Linear 的延迟与能耗；失败时用回退解析模型。

    Uses MNSIM PE-level latency and power models:
      - MNSIM.Latency_Model.PE_latency.PE_latency_analysis
      - MNSIM.Hardware_Model.PE.ProcessElement

    If MNSIM invocation fails, falls back to a conservative analytical model.
    """
    if simconfig_path is None:
        simconfig_path = os.path.join(mnsim_root, "SimConfig.ini")

    tokens = max(1, int(tokens))
    in_features = max(1, int(in_features))
    out_features = max(1, int(out_features))
    cim_parallelism = max(1, int(cim_parallelism))

    try:
        _ensure_mnsim_importable(mnsim_root)
        from MNSIM.Latency_Model.PE_latency import PE_latency_analysis
        from MNSIM.Hardware_Model.PE import ProcessElement

        pe_lat = PE_latency_analysis(
            simconfig_path,
            read_row=in_features,
            read_column=out_features,
            indata=max(1, math.ceil(in_features / 8)),
            rdata=max(1, math.ceil(in_features / 8)),
            inprecision=8,
            default_buf_size=16,
        )
        per_token_latency_ns = float(pe_lat.PE_latency)

        pe = ProcessElement(simconfig_path)
        pe.calculate_PE_read_power_fast(
            max_column=out_features,
            max_row=in_features,
            max_group=max(1, min(pe.group_num, 1)),
            SimConfig_path=simconfig_path,
            default_inbuf_size=16,
        )
        pe_power_mw = float(pe.PE_read_power) * 1e3

        batches = math.ceil(tokens / cim_parallelism)
        latency_ns = batches * per_token_latency_ns
        cycles = max(1, math.ceil(latency_ns / clock_period_ns))

        active_parallel = min(tokens, cim_parallelism)
        avg_power_mw = pe_power_mw * active_parallel
        energy_nj = avg_power_mw * latency_ns / 1e3

        return MNSIMEstimate(
            cycles=cycles,
            latency_ns=latency_ns,
            energy_nj=energy_nj,
            avg_power_mw=avg_power_mw,
            source="mnsim-pe",
        )

    except Exception:
        cycles = max(1, math.ceil(tokens / cim_parallelism) * fallback_cycles_per_token)
        latency_ns = cycles * clock_period_ns
        avg_power_mw = fallback_power_mw
        energy_nj = avg_power_mw * latency_ns / 1e3
        return MNSIMEstimate(
            cycles=cycles,
            latency_ns=latency_ns,
            energy_nj=energy_nj,
            avg_power_mw=avg_power_mw,
            source="fallback",
        )
