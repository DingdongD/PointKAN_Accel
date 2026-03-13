from dataclasses import dataclass
from typing import List, Dict, Any
import os

from .dram_trace import DRAMTrace


@dataclass
class TensorRegion:
    name: str
    base_addr: int
    num_elements: int
    bytes_per_element: int

    @property
    def size_bytes(self) -> int:
        return self.num_elements * self.bytes_per_element

    def addr_of(self, index: int) -> int:
        return self.base_addr + index * self.bytes_per_element


class PointCloudTraceGenerator:
    def __init__(self, burst_bytes: int = 64, default_issue_gap: int = 1):
        self.trace = DRAMTrace()
        self.burst_bytes = burst_bytes
        self.default_issue_gap = default_issue_gap

    def read_tensor_range(self, region: "TensorRegion", start_elem: int, num_elem: int, cycle: int):
        start_addr = region.addr_of(start_elem)
        num_bytes = num_elem * region.bytes_per_element
        self.trace.extend_burst(
            base_addr=start_addr,
            num_bytes=num_bytes,
            burst_bytes=self.burst_bytes,
            is_write=False,
            start_cycle=cycle,
            issue_gap=self.default_issue_gap,
        )

    def write_tensor_range(self, region: "TensorRegion", start_elem: int, num_elem: int, cycle: int):
        start_addr = region.addr_of(start_elem)
        num_bytes = num_elem * region.bytes_per_element
        self.trace.extend_burst(
            base_addr=start_addr,
            num_bytes=num_bytes,
            burst_bytes=self.burst_bytes,
            is_write=True,
            start_cycle=cycle,
            issue_gap=self.default_issue_gap,
        )

    def gen_fps_pass(
        self,
        xyz_region: "TensorRegion",
        min_dist_region: "TensorRegion",
        n_points: int,
        point_dim: int = 3,
        tile_points: int = 32,
        start_cycle: int = 0,
        issue_gap: int = 1,
        compute_gap: int = 0,
    ):
        cycle = start_cycle
        elems_per_point = point_dim
        for tile_start in range(0, n_points, tile_points):
            tile_n = min(tile_points, n_points - tile_start)
            self.read_tensor_range(xyz_region, tile_start * elems_per_point, tile_n * elems_per_point, cycle)
            cycle += issue_gap
            self.read_tensor_range(min_dist_region, tile_start, tile_n, cycle)
            cycle += issue_gap
            self.write_tensor_range(min_dist_region, tile_start, tile_n, cycle)
            cycle += issue_gap
            cycle += compute_gap
        return cycle

    def dump(self, path: str):
        self.trace.dump(path)

    def get_trace_stats(self):
        return self.trace.get_stats()


@dataclass
class DRAMTraceEvent:
    """
    DRAMsim3 transaction trace event.

    Trace format:
        <address> <READ/WRITE> <cycle>

    Example:
        0x2000D5C0 READ 30
        0x1FF96FC0 WRITE 160
    """
    address: int
    is_write: bool
    cycle: int = 0

    def to_line(self) -> str:
        op = "WRITE" if self.is_write else "READ"
        return f"0x{self.address:X} {op} {self.cycle}"


class DRAMTrace:
    def __init__(self):
        self.events: List[DRAMTraceEvent] = []

    def add_event(self, address: int, is_write: bool, cycle: int = 0):
        self.events.append(DRAMTraceEvent(address=address, is_write=is_write, cycle=cycle))

    def add_read(self, address: int, cycle: int = 0):
        self.add_event(address=address, is_write=False, cycle=cycle)

    def add_write(self, address: int, cycle: int = 0):
        self.add_event(address=address, is_write=True, cycle=cycle)

    def extend_burst(
        self,
        base_addr: int,
        num_bytes: int,
        burst_bytes: int,
        is_write: bool,
        start_cycle: int = 0,
        issue_gap: int = 1,
    ):
        """
        Split a large memory request into multiple DRAM transactions.

        Example:
            num_bytes = 192, burst_bytes = 64
            -> generate 3 transactions:
               base+0, base+64, base+128

        Args:
            base_addr: start byte address
            num_bytes: total bytes to transfer
            burst_bytes: bytes per DRAM transaction
            is_write: read or write
            start_cycle: cycle of first transaction arrival
            issue_gap: gap cycles between adjacent transactions
        """
        assert burst_bytes > 0, "burst_bytes must be positive"
        assert num_bytes >= 0, "num_bytes must be non-negative"
        assert issue_gap >= 0, "issue_gap must be non-negative"

        if num_bytes == 0:
            return

        num_txn = (num_bytes + burst_bytes - 1) // burst_bytes
        for i in range(num_txn):
            addr = base_addr + i * burst_bytes
            cycle = start_cycle + i * issue_gap
            self.add_event(address=addr, is_write=is_write, cycle=cycle)

    def sort_by_cycle(self):
        self.events.sort(key=lambda x: (x.cycle, x.address, x.is_write))

    def dump(self, path: str, sort_first: bool = True):
        if sort_first:
            self.sort_by_cycle()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(ev.to_line() + "\n")

    def get_stats(self) -> Dict[str, Any]:
        num_read = sum(1 for e in self.events if not e.is_write)
        num_write = sum(1 for e in self.events if e.is_write)

        if self.events:
            min_cycle = min(e.cycle for e in self.events)
            max_cycle = max(e.cycle for e in self.events)
            min_addr = min(e.address for e in self.events)
            max_addr = max(e.address for e in self.events)
        else:
            min_cycle = max_cycle = 0
            min_addr = max_addr = 0

        return {
            "num_events": len(self.events),
            "num_read": num_read,
            "num_write": num_write,
            "min_cycle": min_cycle,
            "max_cycle": max_cycle,
            "min_addr": hex(min_addr),
            "max_addr": hex(max_addr),
        }

    def clear(self):
        self.events.clear()