# import math
# from dataclasses import asdict
# from typing import Optional

# from .types import (
#     CactiInput,
#     MemoryAccessResult,
#     SRAMRuntimeStats,
#     BufferResidency,
# )
# from .cacti_runner import CactiRunner
# from .utils import ceil_div, ns_to_cycles


# class SRAMBuffer:
#     def __init__(
#         self,
#         name: str,
#         cacti_runner: CactiRunner,
#         cacti_cfg: CactiInput,
#         freq_hz: int,
#         capacity_bytes: Optional[int] = None,
#         num_banks: int = 1,
#     ):
#         self.name = name
#         self.cacti_cfg = cacti_cfg
#         self.freq_hz = freq_hz
#         self.result = cacti_runner.run(cacti_cfg)

#         if self.result.access_time_ns is None:
#             raise ValueError(f"CACTI result missing access_time_ns for buffer {name}")
#         if self.result.read_energy_nj is None or self.result.write_energy_nj is None:
#             raise ValueError(f"CACTI result missing read/write energy for buffer {name}")

#         self.access_cycles = ns_to_cycles(self.result.access_time_ns, freq_hz)
#         self.block_size = cacti_cfg.block_size_bytes
#         self.capacity_bytes = capacity_bytes if capacity_bytes is not None else cacti_cfg.size_bytes

#         self.num_banks = max(1, int(num_banks))
#         self.bank_busy_until = [0] * self.num_banks
#         self.busy_until = 0  # 单 bank 或兼容用
#         self.stats = SRAMRuntimeStats()

#         # New: current resident region metadata
#         self.residency: Optional[BufferResidency] = None

#     def _map_to_bank(self, start_elem: int, bytes_per_elem: int) -> int:
#         """Block index 由 (start_elem * bytes_per_elem) 对齐 block_size 得到，bank = block_index % num_banks。"""
#         block_index = (start_elem * bytes_per_elem) // self.block_size
#         return block_index % self.num_banks

#     def _used_banks_for_region(self, start_elem: int, num_elem: int, bytes_per_elem: int) -> set:
#         """一段 region 访问涉及的 block 对应的 bank 集合。"""
#         start_byte = start_elem * bytes_per_elem
#         num_bytes = num_elem * bytes_per_elem
#         accesses = ceil_div(num_bytes, self.block_size)
#         start_block = start_byte // self.block_size
#         return {(start_block + i) % self.num_banks for i in range(accesses)}

#     # ---------------------------------------------------------
#     # Basic SRAM runtime model (single-bank or multi-bank)
#     # ---------------------------------------------------------
#     def _read_or_write(
#         self,
#         num_bytes: int,
#         cur_cycle: int,
#         used_banks: set,
#         is_read: bool,
#     ) -> MemoryAccessResult:
#         accesses = ceil_div(num_bytes, self.block_size)
#         if self.num_banks <= 1:
#             service_cycles = accesses * self.access_cycles
#             start_cycle = max(cur_cycle, self.busy_until)
#         else:
#             # 多 bank：服务时间 ≈ ceil(accesses / num_banks) * access_cycles
#             service_cycles = ceil_div(accesses, self.num_banks) * self.access_cycles
#             start_cycle = cur_cycle
#             if used_banks:
#                 start_cycle = max(start_cycle, max(self.bank_busy_until[b] for b in used_banks))

#         end_cycle = start_cycle + service_cycles
#         stall_cycles = start_cycle - cur_cycle

#         if self.num_banks <= 1:
#             self.busy_until = end_cycle
#         else:
#             for b in used_banks:
#                 self.bank_busy_until[b] = end_cycle

#         energy_nj = (self.result.read_energy_nj if is_read else self.result.write_energy_nj) * accesses
#         if is_read:
#             self.stats.total_read_bytes += num_bytes
#             self.stats.total_read_accesses += accesses
#             self.stats.total_dynamic_read_energy_nj += energy_nj
#         else:
#             self.stats.total_write_bytes += num_bytes
#             self.stats.total_write_accesses += accesses
#             self.stats.total_dynamic_write_energy_nj += energy_nj
#         self.stats.total_busy_cycles += service_cycles
#         self.stats.total_stall_cycles += stall_cycles

#         return MemoryAccessResult(
#             bytes_requested=num_bytes,
#             accesses=accesses,
#             service_cycles=service_cycles,
#             stall_cycles=stall_cycles,
#             start_cycle=start_cycle,
#             end_cycle=end_cycle,
#             dynamic_energy_nj=float(energy_nj),
#         )

#     def read(self, num_bytes: int, cur_cycle: int) -> MemoryAccessResult:
#         accesses = ceil_div(num_bytes, self.block_size)
#         used_banks = {i % self.num_banks for i in range(accesses)} if self.num_banks > 1 else set()
#         return self._read_or_write(num_bytes, cur_cycle, used_banks, is_read=True)

#     def write(self, num_bytes: int, cur_cycle: int) -> MemoryAccessResult:
#         accesses = ceil_div(num_bytes, self.block_size)
#         used_banks = {i % self.num_banks for i in range(accesses)} if self.num_banks > 1 else set()
#         return self._read_or_write(num_bytes, cur_cycle, used_banks, is_read=False)

#     # ---------------------------------------------------------
#     # Residency / refill helpers
#     # ---------------------------------------------------------
#     def can_hold(self, num_bytes: int) -> bool:
#         return num_bytes <= self.capacity_bytes

#     def contains(self, region_name: str, start_elem: int, num_elem: int, cur_cycle: Optional[int] = None) -> bool:
#         if self.residency is None:
#             return False
#         if not self.residency.contains(region_name, start_elem, num_elem):
#             return False
#         if cur_cycle is not None and cur_cycle < self.residency.ready_cycle:
#             return False
#         return True

#     def evict(self):
#         self.residency = None

#     def fill(
#         self,
#         region_name: str,
#         start_elem: int,
#         num_elem: int,
#         ready_cycle: int,
#     ):
#         """
#         Mark that a logical region slice is now resident in this buffer
#         and becomes available at ready_cycle.
#         """
#         self.residency = BufferResidency(
#             region_name=region_name,
#             start_elem=start_elem,
#             num_elem=num_elem,
#             ready_cycle=ready_cycle,
#         )

#     def read_region(
#         self,
#         region_name: str,
#         start_elem: int,
#         num_elem: int,
#         bytes_per_elem: int,
#         cur_cycle: int,
#     ) -> MemoryAccessResult:
#         if not self.contains(region_name, start_elem, num_elem, cur_cycle):
#             raise RuntimeError(
#                 f"{self.name}: region miss when reading "
#                 f"{region_name}[{start_elem}:{start_elem+num_elem}] at cycle {cur_cycle}; "
#                 f"residency={self.residency}"
#             )
#         num_bytes = num_elem * bytes_per_elem
#         if self.num_banks <= 1:
#             return self.read(num_bytes, cur_cycle)
#         used_banks = self._used_banks_for_region(start_elem, num_elem, bytes_per_elem)
#         return self._read_or_write(num_bytes, cur_cycle, used_banks, is_read=True)

#     def write_region(
#         self,
#         region_name: str,
#         start_elem: int,
#         num_elem: int,
#         bytes_per_elem: int,
#         cur_cycle: int,
#     ) -> MemoryAccessResult:
#         num_bytes = num_elem * bytes_per_elem
#         if self.num_banks <= 1:
#             return self.write(num_bytes, cur_cycle)
#         used_banks = self._used_banks_for_region(start_elem, num_elem, bytes_per_elem)
#         return self._read_or_write(num_bytes, cur_cycle, used_banks, is_read=False)

#     def get_summary(self):
#         runtime_s = (self.busy_until / self.freq_hz)
#         leak_energy_nj = 0.0
#         if self.result.leak_power_mw is not None:
#             leak_energy_mj = float(self.result.leak_power_mw) * runtime_s
#             leak_energy_nj = leak_energy_mj * 1e6

#         return {
#             "name": self.name,
#             "capacity_bytes": self.capacity_bytes,
#             "block_size_bytes": self.block_size,
#             "num_banks": self.num_banks,
#             "access_cycles": self.access_cycles,
#             "area_mm2": self.result.area_mm2,
#             "leak_power_mw": self.result.leak_power_mw,
#             "residency": None if self.residency is None else {
#                 "region_name": self.residency.region_name,
#                 "start_elem": self.residency.start_elem,
#                 "num_elem": self.residency.num_elem,
#                 "ready_cycle": self.residency.ready_cycle,
#             },
#             "runtime_stats": asdict(self.stats),
#             "leak_energy_nj": leak_energy_nj,
#             "total_energy_nj": (
#                 self.stats.total_dynamic_read_energy_nj
#                 + self.stats.total_dynamic_write_energy_nj
#                 + leak_energy_nj
#             ),
#         }



import math
from dataclasses import asdict
from typing import Optional, Dict, List

from .types import (
    CactiInput,
    MemoryAccessResult,
    SRAMRuntimeStats,
    BufferResidency,
)
from .cacti_runner import CactiRunner
from .utils import ceil_div, ns_to_cycles


class SRAMBuffer:
    def __init__(
        self,
        name: str,
        cacti_runner: CactiRunner,
        cacti_cfg: CactiInput,
        freq_hz: int,
        num_banks: int = 1,
    ):
        self.name = name
        self.cacti_cfg = cacti_cfg
        self.freq_hz = freq_hz
        self.num_banks = max(1, int(num_banks))

        # 让 CACTI 的 bank count 与运行时 bank 数一致
        self.cacti_cfg.uca_bank_count = self.num_banks

        self.result = cacti_runner.run(cacti_cfg)

        if self.result.access_time_ns is None:
            raise ValueError(f"CACTI result missing access_time_ns for buffer {name}")
        if self.result.read_energy_nj is None or self.result.write_energy_nj is None:
            raise ValueError(f"CACTI result missing read/write energy for buffer {name}")

        self.access_cycles = ns_to_cycles(self.result.access_time_ns, freq_hz)
        self.block_size = cacti_cfg.block_size_bytes
        self.capacity_bytes = cacti_cfg.size_bytes

        # 单 bank 旧接口兼容
        self.busy_until = 0

        # 新增：每个 bank 独立 busy 时间
        self.bank_busy_until: List[int] = [0 for _ in range(self.num_banks)]

        self.stats = SRAMRuntimeStats()
        self.residency: Optional[BufferResidency] = None

    # ---------------------------------------------------------
    # Residency / refill helpers
    # ---------------------------------------------------------
    def can_hold(self, num_bytes: int) -> bool:
        return num_bytes <= self.capacity_bytes

    def contains(self, region_name: str, start_elem: int, num_elem: int, cur_cycle: Optional[int] = None) -> bool:
        if self.residency is None:
            return False
        if not self.residency.contains(region_name, start_elem, num_elem):
            return False
        if cur_cycle is not None and cur_cycle < self.residency.ready_cycle:
            return False
        return True

    def evict(self):
        self.residency = None

    def fill(
        self,
        region_name: str,
        start_elem: int,
        num_elem: int,
        ready_cycle: int,
    ):
        self.residency = BufferResidency(
            region_name=region_name,
            start_elem=start_elem,
            num_elem=num_elem,
            ready_cycle=ready_cycle,
        )

    # ---------------------------------------------------------
    # Multi-bank helpers
    # ---------------------------------------------------------
    def _bank_access_histogram_from_byte_range(self, start_byte: int, num_bytes: int) -> Dict[int, int]:
        """
        将一个连续 byte range 映射到各 bank 的 block access 次数。
        采用 block interleaving：
            block_idx = floor(byte_addr / block_size)
            bank_id = block_idx % num_banks
        """
        if num_bytes <= 0:
            return {}

        first_block = start_byte // self.block_size
        last_block = (start_byte + num_bytes - 1) // self.block_size

        hist: Dict[int, int] = {}
        for blk in range(first_block, last_block + 1):
            bank_id = blk % self.num_banks
            hist[bank_id] = hist.get(bank_id, 0) + 1
        return hist

    def _schedule_bank_histogram(
        self,
        bank_hist: Dict[int, int],
        cur_cycle: int,
        is_write: bool,
    ) -> MemoryAccessResult:
        """
        根据 bank_hist 调度一次多 bank 访问。
        每个 bank 上的 service_cycles = accesses_on_bank * access_cycles
        请求整体完成时间 = max(bank_end_cycle)
        """
        if not bank_hist:
            return MemoryAccessResult(
                bytes_requested=0,
                accesses=0,
                service_cycles=0,
                stall_cycles=0,
                start_cycle=cur_cycle,
                end_cycle=cur_cycle,
                dynamic_energy_nj=0.0,
            )

        total_accesses = sum(bank_hist.values())
        per_access_energy = float(self.result.write_energy_nj if is_write else self.result.read_energy_nj)

        max_end_cycle = cur_cycle
        min_start_cycle = None

        for bank_id, bank_accesses in bank_hist.items():
            bank_start = max(cur_cycle, self.bank_busy_until[bank_id])
            bank_service = bank_accesses * self.access_cycles
            bank_end = bank_start + bank_service

            self.bank_busy_until[bank_id] = bank_end

            if min_start_cycle is None or bank_start < min_start_cycle:
                min_start_cycle = bank_start
            if bank_end > max_end_cycle:
                max_end_cycle = bank_end

        self.busy_until = max(self.bank_busy_until)

        overall_start = cur_cycle if min_start_cycle is None else min_start_cycle
        overall_end = max_end_cycle
        overall_latency = overall_end - cur_cycle
        stall_cycles = overall_start - cur_cycle if overall_start > cur_cycle else 0
        dynamic_energy_nj = total_accesses * per_access_energy

        return MemoryAccessResult(
            bytes_requested=0,  # 这里上层会覆盖
            accesses=total_accesses,
            service_cycles=overall_latency,
            stall_cycles=stall_cycles,
            start_cycle=overall_start,
            end_cycle=overall_end,
            dynamic_energy_nj=dynamic_energy_nj,
        )

    # ---------------------------------------------------------
    # Basic SRAM runtime model
    # ---------------------------------------------------------
    def read(self, num_bytes: int, cur_cycle: int) -> MemoryAccessResult:
        bank_hist = self._bank_access_histogram_from_byte_range(start_byte=0, num_bytes=num_bytes)
        stat = self._schedule_bank_histogram(bank_hist, cur_cycle, is_write=False)
        stat.bytes_requested = num_bytes

        self.stats.total_read_bytes += num_bytes
        self.stats.total_read_accesses += stat.accesses
        self.stats.total_dynamic_read_energy_nj += stat.dynamic_energy_nj
        self.stats.total_busy_cycles += stat.service_cycles
        self.stats.total_stall_cycles += stat.stall_cycles
        return stat

    def write(self, num_bytes: int, cur_cycle: int) -> MemoryAccessResult:
        bank_hist = self._bank_access_histogram_from_byte_range(start_byte=0, num_bytes=num_bytes)
        stat = self._schedule_bank_histogram(bank_hist, cur_cycle, is_write=True)
        stat.bytes_requested = num_bytes

        self.stats.total_write_bytes += num_bytes
        self.stats.total_write_accesses += stat.accesses
        self.stats.total_dynamic_write_energy_nj += stat.dynamic_energy_nj
        self.stats.total_busy_cycles += stat.service_cycles
        self.stats.total_stall_cycles += stat.stall_cycles
        return stat

    # ---------------------------------------------------------
    # Region-aware access
    # ---------------------------------------------------------
    def read_region(
        self,
        region_name: str,
        start_elem: int,
        num_elem: int,
        bytes_per_elem: int,
        cur_cycle: int,
    ) -> MemoryAccessResult:
        if not self.contains(region_name, start_elem, num_elem, cur_cycle):
            raise RuntimeError(
                f"{self.name}: region miss when reading "
                f"{region_name}[{start_elem}:{start_elem+num_elem}] at cycle {cur_cycle}; "
                f"residency={self.residency}"
            )

        start_byte = start_elem * bytes_per_elem
        num_bytes = num_elem * bytes_per_elem
        bank_hist = self._bank_access_histogram_from_byte_range(start_byte, num_bytes)
        stat = self._schedule_bank_histogram(bank_hist, cur_cycle, is_write=False)
        stat.bytes_requested = num_bytes

        self.stats.total_read_bytes += num_bytes
        self.stats.total_read_accesses += stat.accesses
        self.stats.total_dynamic_read_energy_nj += stat.dynamic_energy_nj
        self.stats.total_busy_cycles += stat.service_cycles
        self.stats.total_stall_cycles += stat.stall_cycles
        return stat

    def write_region(
        self,
        region_name: str,
        start_elem: int,
        num_elem: int,
        bytes_per_elem: int,
        cur_cycle: int,
    ) -> MemoryAccessResult:
        start_byte = start_elem * bytes_per_elem
        num_bytes = num_elem * bytes_per_elem
        bank_hist = self._bank_access_histogram_from_byte_range(start_byte, num_bytes)
        stat = self._schedule_bank_histogram(bank_hist, cur_cycle, is_write=True)
        stat.bytes_requested = num_bytes

        self.stats.total_write_bytes += num_bytes
        self.stats.total_write_accesses += stat.accesses
        self.stats.total_dynamic_write_energy_nj += stat.dynamic_energy_nj
        self.stats.total_busy_cycles += stat.service_cycles
        self.stats.total_stall_cycles += stat.stall_cycles
        return stat

    def get_summary(self):
        runtime_s = (max(self.bank_busy_until) / self.freq_hz) if self.freq_hz > 0 else 0.0
        leak_energy_nj = 0.0
        if self.result.leak_power_mw is not None:
            leak_energy_mj = float(self.result.leak_power_mw) * runtime_s
            leak_energy_nj = leak_energy_mj * 1e6

        return {
            "name": self.name,
            "num_banks": self.num_banks,
            "capacity_bytes": self.capacity_bytes,
            "block_size_bytes": self.block_size,
            "access_cycles": self.access_cycles,
            "area_mm2": self.result.area_mm2,
            "leak_power_mw": self.result.leak_power_mw,
            "bank_busy_until": self.bank_busy_until,
            "residency": None if self.residency is None else {
                "region_name": self.residency.region_name,
                "start_elem": self.residency.start_elem,
                "num_elem": self.residency.num_elem,
                "ready_cycle": self.residency.ready_cycle,
            },
            "runtime_stats": asdict(self.stats),
            "leak_energy_nj": leak_energy_nj,
            "total_energy_nj": (
                self.stats.total_dynamic_read_energy_nj
                + self.stats.total_dynamic_write_energy_nj
                + leak_energy_nj
            ),
        }