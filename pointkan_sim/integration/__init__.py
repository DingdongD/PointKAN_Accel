"""
Integration with external simulators (MNSIM-2.0, memory_model CACTI/DRAMSim3).

- mnsim_netstruct: Build NetStruct from PointKAN conv/linear ops and run
  Model_latency / Model_energy for Tile/NoC-level results.
- memory_model_adapter: 用 hw_sim/cycle_simulator/memory_model 的 CACTI/DRAMSim3 回填 Buffer、DRAM 功耗。
"""

from .mnsim_netstruct import (
    build_netstruct_from_layers,
    run_mnsim_full,
    MNSIMFullResult,
)

__all__ = [
    "build_netstruct_from_layers",
    "run_mnsim_full",
    "MNSIMFullResult",
]

# 统一 SRAM/DRAM 查询与 CACTI/DRAMSim3 回填（memory_model_adapter）
from .memory_model_adapter import query_sram_cycles_energy, query_dram_cycles_energy
__all__.extend(["query_sram_cycles_energy", "query_dram_cycles_energy"])
try:
    from .memory_model_adapter import (
        fill_buffers_from_cacti,
        fill_inter_layer_buffer_from_cacti,
        run_dram_trace_for_layer_buffer_dram,
    )
    __all__.extend([
        "fill_buffers_from_cacti",
        "fill_inter_layer_buffer_from_cacti",
        "run_dram_trace_for_layer_buffer_dram",
    ])
except Exception:
    pass

try:
    from .memory_runtime import create_memory_runtime_context, MemoryRuntimeContext
    __all__.extend(["create_memory_runtime_context", "MemoryRuntimeContext"])
except Exception:
    pass
