from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class AccessReq:
    cycle: int
    requester: str
    addr: int
    size: int
    stage: str
    tree_level: int = 0
    tag: Optional[str] = None


@dataclass
class SRAMStats:
    reads: int = 0
    writes: int = 0
    conflicts: int = 0
    stalled: int = 0
    elided: int = 0
    served: int = 0
    energy_pj: float = 0.0


class SRAMBuffer:
    def __init__(self, name: str, capacity_bytes: int, num_banks: int,
                 bank_width_bytes: int, ports_per_bank: int,
                 access_latency_cycles: int, read_energy_pj: float,
                 write_energy_pj: float):
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.num_banks = num_banks
        self.bank_width_bytes = bank_width_bytes
        self.ports_per_bank = ports_per_bank
        self.access_latency_cycles = access_latency_cycles
        self.read_energy_pj = read_energy_pj
        self.write_energy_pj = write_energy_pj
        self.stats = SRAMStats()

    def addr_to_bank(self, addr: int) -> int:
        return (addr // self.bank_width_bytes) % self.num_banks

    def process_cycle(self, requests: List[AccessReq], enable_bce: bool, elision_height: int):
        by_bank: Dict[int, List[AccessReq]] = {}
        for r in requests:
            by_bank.setdefault(self.addr_to_bank(r.addr), []).append(r)
        served, stalled, elided = [], [], []
        for bank_id, rs in by_bank.items():
            if len(rs) <= self.ports_per_bank:
                served.extend(rs)
                continue
            self.stats.conflicts += len(rs) - self.ports_per_bank
            keep = rs[:self.ports_per_bank]
            served.extend(keep)
            for r in rs[self.ports_per_bank:]:
                if enable_bce and r.tree_level > elision_height:
                    elided.append(r)
                else:
                    stalled.append(r)
        self.stats.served += len(served)
        self.stats.stalled += len(stalled)
        self.stats.elided += len(elided)
        return served, stalled, elided

    def account_read(self, n: int = 1):
        self.stats.reads += n
        self.stats.energy_pj += n * self.read_energy_pj

    def account_write(self, n: int = 1):
        self.stats.writes += n
        self.stats.energy_pj += n * self.write_energy_pj


@dataclass
class DRAMStats:
    read_bursts: int = 0
    write_bursts: int = 0
    energy_pj: float = 0.0


class DRAMModel:
    def __init__(self, burst_bytes: int, read_energy_pj_per_burst: float, write_energy_pj_per_burst: float):
        self.burst_bytes = burst_bytes
        self.read_energy_pj_per_burst = read_energy_pj_per_burst
        self.write_energy_pj_per_burst = write_energy_pj_per_burst
        self.stats = DRAMStats()

    def stream_read(self, bytes_count: int):
        bursts = math.ceil(bytes_count / self.burst_bytes)
        self.stats.read_bursts += bursts
        self.stats.energy_pj += bursts * self.read_energy_pj_per_burst

    def stream_write(self, bytes_count: int):
        bursts = math.ceil(bytes_count / self.burst_bytes)
        self.stats.write_bursts += bursts
        self.stats.energy_pj += bursts * self.write_energy_pj_per_burst
