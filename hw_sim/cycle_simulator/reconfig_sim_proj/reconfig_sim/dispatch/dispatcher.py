from __future__ import annotations
from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Optional

from reconfig_sim.dispatch.policies import MappingPolicy, Mode
from reconfig_sim.workloads.op_desc import OpDesc


@dataclass(frozen=True)
class HardwareConfig:
    arr_r: int = 8
    arr_c: int = 8
    word_bits: int = 16
    acc_bits: int = 64
    mode_switch_cycles: int = 8
    mode_switch_energy: float = 32.0
    os_clear_cycles: int = 1
    os_drain_extra: int = 4
    ws_preload_cycles: int = 2
    ws_gap_cycles: int = 1
    energy_mac: float = 1.0
    energy_buffer_per_byte: float = 0.02
    energy_noc_per_byte_hop: float = 0.01


@dataclass(frozen=True)
class TileDesc:
    op_name: str
    op_type: str
    mode: Mode
    m0: int
    n0: int
    k0: int
    mt: int
    nt: int
    kt: int


@dataclass
class TileStats:
    cycles: int
    energy: float
    compute_cycles: int
    buffer_bytes: int
    noc_bytes: int
    switch_cycles: int = 0
    switch_energy: float = 0.0
    notes: Dict[str, float] = field(default_factory=dict)


@dataclass
class DispatchResult:
    schedule: List[TileDesc]
    tile_stats: List[TileStats]
    total_cycles: int
    total_energy: float
    mode_hist: Dict[str, int]


class ModeAwareDispatcher:
    def __init__(self, hw: HardwareConfig, policy: MappingPolicy):
        self.hw = hw
        self.policy = policy

    def tile_op(self, op: OpDesc, mode_override: Optional[Mode] = None) -> List[TileDesc]:
        mode = mode_override or self.policy.select_mode(op)
        tiles: List[TileDesc] = []
        mt_full = min(op.m, self.hw.arr_r)
        nt_full = min(op.n, self.hw.arr_c)
        kt_full = min(op.k, self.hw.arr_r if mode == "WS" else op.k)

        if mode == "OS":
            for m0 in range(0, op.m, self.hw.arr_r):
                mt = min(self.hw.arr_r, op.m - m0)
                for n0 in range(0, op.n, self.hw.arr_c):
                    nt = min(self.hw.arr_c, op.n - n0)
                    tiles.append(TileDesc(op.name, op.op_type, mode, m0, n0, 0, mt, nt, op.k))
        else:  # WS
            for m0 in range(0, op.m, 1):
                for n0 in range(0, op.n, self.hw.arr_c):
                    nt = min(self.hw.arr_c, op.n - n0)
                    for k0 in range(0, op.k, self.hw.arr_r):
                        kt = min(self.hw.arr_r, op.k - k0)
                        tiles.append(TileDesc(op.name, op.op_type, mode, m0, n0, k0, 1, nt, kt))
        return tiles

    def estimate_tile_cost(self, tile: TileDesc, prev_mode: Optional[Mode] = None) -> TileStats:
        hw = self.hw
        word_bytes = hw.word_bits // 8
        acc_bytes = hw.acc_bits // 8
        switch_cycles = hw.mode_switch_cycles if prev_mode is not None and prev_mode != tile.mode else 0
        switch_energy = hw.mode_switch_energy if switch_cycles else 0.0

        if tile.mode == "OS":
            compute_cycles = tile.kt + tile.mt + tile.nt - 2
            cycles = hw.os_clear_cycles + compute_cycles + hw.os_drain_extra + switch_cycles
            a_bytes = tile.mt * tile.kt * word_bytes
            b_bytes = tile.kt * tile.nt * word_bytes
            c_bytes = tile.mt * tile.nt * acc_bytes
            buffer_bytes = a_bytes + b_bytes + c_bytes
            noc_bytes = a_bytes + b_bytes
        else:
            compute_cycles = tile.kt + tile.nt + hw.ws_gap_cycles
            cycles = hw.ws_preload_cycles + compute_cycles + switch_cycles
            a_bytes = tile.kt * word_bytes
            b_bytes = tile.kt * tile.nt * word_bytes
            psum_bytes = tile.nt * acc_bytes * 2
            buffer_bytes = a_bytes + b_bytes + psum_bytes
            noc_bytes = a_bytes + psum_bytes

        macs = tile.mt * tile.nt * tile.kt
        energy = (
            macs * hw.energy_mac
            + buffer_bytes * hw.energy_buffer_per_byte
            + noc_bytes * hw.energy_noc_per_byte_hop
            + switch_energy
        )
        return TileStats(
            cycles=cycles,
            energy=energy,
            compute_cycles=compute_cycles,
            buffer_bytes=buffer_bytes,
            noc_bytes=noc_bytes,
            switch_cycles=switch_cycles,
            switch_energy=switch_energy,
            notes={"macs": float(macs)},
        )

    def dispatch(self, ops: List[OpDesc], overrides: Optional[Dict[str, Mode]] = None) -> DispatchResult:
        overrides = overrides or {}
        schedule: List[TileDesc] = []
        tile_stats: List[TileStats] = []
        mode_hist = {"OS": 0, "WS": 0}
        prev_mode: Optional[Mode] = None

        for op in ops:
            mode_override = overrides.get(op.name)
            op_tiles = self.tile_op(op, mode_override=mode_override)
            for tile in op_tiles:
                stats = self.estimate_tile_cost(tile, prev_mode)
                schedule.append(tile)
                tile_stats.append(stats)
                mode_hist[tile.mode] += 1
                prev_mode = tile.mode

        total_cycles = sum(s.cycles for s in tile_stats)
        total_energy = sum(s.energy for s in tile_stats)
        return DispatchResult(schedule, tile_stats, total_cycles, total_energy, mode_hist)
