from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Iterable, List, Tuple


class MemoryLevel(str, Enum):
    """统一内存分级命名。"""

    DRAM = "DRAM"
    LAYER_BUFFER = "LAYER_BUFFER"
    SRAM = "SRAM"
    SRAM_II = "SRAM_II"
    STREAM = "STREAM"
    UNKNOWN = "UNKNOWN"


class MemoryTxnKind(str, Enum):
    """统一事务类型。"""

    READ = "READ"
    WRITE = "WRITE"
    SPILL = "SPILL"
    COMPUTE_SCAN = "COMPUTE_SCAN"
    UNKNOWN = "UNKNOWN"


@dataclass
class MemoryTransaction:
    """内存事务统一记录。"""

    idx: int
    cycle: int
    end_cycle: int
    duration: int
    layer_name: str
    operator: str
    stage: str
    operation: str
    circuit_module: str
    energy_nj: float
    src: MemoryLevel = MemoryLevel.UNKNOWN
    dst: MemoryLevel = MemoryLevel.UNKNOWN
    kind: MemoryTxnKind = MemoryTxnKind.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "cycle": self.cycle,
            "end_cycle": self.end_cycle,
            "duration": self.duration,
            "layer_name": self.layer_name,
            "operator": self.operator,
            "stage": self.stage,
            "operation": self.operation,
            "circuit_module": self.circuit_module,
            "energy_nj": self.energy_nj,
            "src": self.src.value,
            "dst": self.dst.value,
            "kind": self.kind.value,
        }


def _module_to_level(module: str) -> MemoryLevel:
    m = (module or "").lower()
    if m == "dram":
        return MemoryLevel.DRAM
    if m == "inter_layer_buffer":
        return MemoryLevel.LAYER_BUFFER
    if m == "sram_ii":
        return MemoryLevel.SRAM_II
    if m == "buffer":
        return MemoryLevel.SRAM
    return MemoryLevel.UNKNOWN


def infer_memory_transaction(item: Dict[str, Any]) -> Optional[MemoryTransaction]:
    """
    从 operator_sequence 的一条记录推断内存事务。

    这层将 DRAM/SRAM/LAYER_BUFFER/SRAM-II 用统一字段 src/dst/kind 表达，
    便于后续统计与可视化，不再在业务逻辑里散落 if/else。
    """
    op = str(item.get("operator", "")).lower()
    stage = str(item.get("stage", "")).lower()
    operation = str(item.get("operation", "")).lower()
    circuit_module = str(item.get("circuit_module", ""))
    cm_level = _module_to_level(circuit_module)

    is_mem = False
    if op == "layerbuffer":
        is_mem = True
    if stage in ("buffer_io", "distance") and any(k in operation for k in ("read", "write", "spill", "load", "phase1", "phase2_query", "sram")):
        is_mem = True
    if cm_level in (MemoryLevel.DRAM, MemoryLevel.LAYER_BUFFER, MemoryLevel.SRAM_II, MemoryLevel.SRAM):
        if any(k in operation for k in ("read", "write", "spill", "load", "phase1", "phase2_query", "sram")):
            is_mem = True

    if not is_mem:
        return None

    src = MemoryLevel.UNKNOWN
    dst = MemoryLevel.UNKNOWN
    kind = MemoryTxnKind.UNKNOWN

    if "phase1_point_cloud_to_sram_ii" in operation:
        src, dst, kind = MemoryLevel.DRAM, MemoryLevel.SRAM_II, MemoryTxnKind.READ
    elif "phase2_query_point_read" in operation:
        src, dst, kind = MemoryLevel.DRAM, MemoryLevel.SRAM, MemoryTxnKind.READ
    elif "spill_read" in operation:
        src, dst, kind = MemoryLevel.DRAM, MemoryLevel.LAYER_BUFFER, MemoryTxnKind.SPILL
    elif "spill_write" in operation:
        src, dst, kind = MemoryLevel.LAYER_BUFFER, MemoryLevel.DRAM, MemoryTxnKind.SPILL
    elif "buffer_read" in operation:
        src, dst, kind = MemoryLevel.LAYER_BUFFER, MemoryLevel.STREAM, MemoryTxnKind.READ
    elif "buffer_write" in operation:
        src, dst, kind = MemoryLevel.STREAM, MemoryLevel.LAYER_BUFFER, MemoryTxnKind.WRITE
    elif "distance_compute" in operation:
        src, dst, kind = MemoryLevel.SRAM_II, MemoryLevel.STREAM, MemoryTxnKind.COMPUTE_SCAN
    elif "dram_read" in operation or operation.endswith("_read"):
        src, dst, kind = MemoryLevel.DRAM, cm_level if cm_level != MemoryLevel.UNKNOWN else MemoryLevel.STREAM, MemoryTxnKind.READ
    elif "dram_write" in operation or operation.endswith("_write"):
        src, dst, kind = cm_level if cm_level != MemoryLevel.UNKNOWN else MemoryLevel.STREAM, MemoryLevel.DRAM, MemoryTxnKind.WRITE

    return MemoryTransaction(
        idx=int(item.get("idx", 0) or 0),
        cycle=int(item.get("cycle", 0) or 0),
        end_cycle=int(item.get("end_cycle", 0) or 0),
        duration=int(item.get("duration", 0) or 0),
        layer_name=str(item.get("layer_name", "") or ""),
        operator=str(item.get("operator", "") or ""),
        stage=str(item.get("stage", "") or ""),
        operation=str(item.get("operation", "") or ""),
        circuit_module=str(item.get("circuit_module", "") or ""),
        energy_nj=float(item.get("energy_nj", 0.0) or 0.0),
        src=src,
        dst=dst,
        kind=kind,
    )


def extract_memory_transactions(operator_sequence: Iterable[Dict[str, Any]]) -> List[MemoryTransaction]:
    """从 operator_sequence 批量提取内存事务。"""
    return [tx for tx in (infer_memory_transaction(it) for it in operator_sequence) if tx is not None]


def summarize_memory_transactions(memory_txns: Iterable[MemoryTransaction]) -> Dict[Tuple[str, str, str], int]:
    """
    统计内存事务通路负载。

    Returns:
        {(src, dst, kind): count}
    """
    summary: Dict[Tuple[str, str, str], int] = {}
    for tx in memory_txns:
        key = (tx.src.value, tx.dst.value, tx.kind.value)
        summary[key] = summary.get(key, 0) + 1
    return summary
