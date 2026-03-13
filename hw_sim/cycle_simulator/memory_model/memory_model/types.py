from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class CactiInput:
    size_bytes: int
    block_size_bytes: int
    technology_u: float = 0.028
    output_bus_width_bits: Optional[int] = None
    cache_type: str = "ram"   # "ram" / "cache"
    associativity: int = 1
    read_write_port: int = 1
    exclusive_read_port: int = 0
    exclusive_write_port: int = 0
    single_ended_read_ports: int = 0
    # CACTI UCA bank 数。>1 时 CACTI 会按多 bank（bank 间 bus）建模面积/功耗/单次访问延迟；
    # 当前 SRAMBuffer 运行时仍按“单口 SRAM”使用 CACTI 的 access_time/energy，不区分多 bank 并行访问。
    uca_bank_count: int = 2


@dataclass
class CactiResult:
    access_time_ns: Optional[float] = None
    cycle_time_ns: Optional[float] = None
    read_energy_nj: Optional[float] = None
    write_energy_nj: Optional[float] = None
    leak_power_mw: Optional[float] = None
    gate_leak_power_mw: Optional[float] = None
    height_mm: Optional[float] = None
    width_mm: Optional[float] = None
    area_mm2: Optional[float] = None
    raw_text: str = ""
    cfg_path: Optional[str] = None


@dataclass
class DRAMSim3Input:
    config_path: str
    trace_path: str
    num_cycles: Optional[int] = None
    output_dir: Optional[str] = None


@dataclass
class DRAMSim3Result:
    stdout: str = ""
    stderr: str = ""
    json_stats: Dict[str, Any] = field(default_factory=dict)
    epoch_stats: List[Dict[str, Any]] = field(default_factory=list)
    output_dir: Optional[str] = None


@dataclass
class MemoryAccessResult:
    bytes_requested: int
    accesses: int
    service_cycles: int
    stall_cycles: int
    start_cycle: int
    end_cycle: int
    dynamic_energy_nj: float


@dataclass
class SRAMRuntimeStats:
    total_read_bytes: int = 0
    total_write_bytes: int = 0
    total_read_accesses: int = 0
    total_write_accesses: int = 0
    total_dynamic_read_energy_nj: float = 0.0
    total_dynamic_write_energy_nj: float = 0.0
    total_busy_cycles: int = 0
    total_stall_cycles: int = 0


@dataclass
class BufferResidency:
    """
    Describe what logical region is currently resident in an on-chip SRAM buffer.

    region_name: logical tensor / region name, e.g. "xyz", "feature", "min_dist"
    start_elem, num_elem: element range [start_elem, start_elem + num_elem)
    ready_cycle: cycle when refill/write-into-buffer is completed and data becomes available
    """
    region_name: str
    start_elem: int
    num_elem: int
    ready_cycle: int = 0

    @property
    def end_elem(self) -> int:
        return self.start_elem + self.num_elem

    def contains(self, region_name: str, start_elem: int, num_elem: int) -> bool:
        if self.region_name != region_name:
            return False
        return start_elem >= self.start_elem and (start_elem + num_elem) <= self.end_elem


@dataclass
class BufferResidency:
    """
    Describe what logical region is currently resident in an on-chip SRAM buffer.

    region_name:
        logical tensor / region name, e.g. "xyz", "feature", "min_dist"

    start_elem, num_elem:
        element range [start_elem, start_elem + num_elem)

    ready_cycle:
        cycle when refill/write-into-buffer is completed and data becomes available
    """
    
    region_name: str
    start_elem: int
    num_elem: int
    ready_cycle: int = 0

    @property
    def end_elem(self) -> int:
        return self.start_elem + self.num_elem

    def contains(self, region_name: str, start_elem: int, num_elem: int) -> bool:
        if self.region_name != region_name:
            return False
        return start_elem >= self.start_elem and (start_elem + num_elem) <= self.end_elem