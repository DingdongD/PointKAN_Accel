# -*- coding: utf-8 -*-
"""
周期-能耗模型：对单个 tile 按 OS/WS 数据流估算计算周期、各阶段能耗，并叠加缓冲与 NoC。
"""
from dataclasses import dataclass, asdict
from .tile import TileDesc


@dataclass
class TileStats:
    """单块（tile）的周期、能耗与各阶段/缓冲/NoC 的分解统计"""
    tile: dict           # TileDesc 的字典形式
    cycles: int          # 本块总周期（计算 + 缓冲 stall + NoC）
    energy: float        # 本块总能耗（相对单位）
    macs: int            # 本块 MAC 数
    buffer: dict         # 缓冲模型输出（BufferStats 转 dict）
    noc: dict            # NoC 模型输出（NoCStats 转 dict）
    phase_cycles: dict   # 各阶段周期：clear/compute/drain（OS）或 preload/compute/gap（WS）
    phase_energy: dict   # 各阶段能耗

    def to_dict(self):
        return asdict(self)


class CycleEnergyModel:
    """将单个 tile 映射为周期与能耗：先按 OS/WS 算计算阶段，再叠加缓冲与 NoC，最后加调度/控制能耗。"""

    def __init__(self, cfg, buffer_model, noc_model):
        self.cfg = cfg
        self.buffer_model = buffer_model
        self.noc_model = noc_model

    def estimate_tile(self, tile: TileDesc, word_bytes: int):
        """
        估计单块周期与能耗。
        word_bytes：单字字节数（如 word_size//8）。
        OS：clear + 脉动计算(k+m+n-2) + drain；C 只写不读。
        WS：preload + 计算(m+n-1) + gap；C 先读后写（部分和累加）。
        """
        m, k, n = tile.m_tile, tile.k_tile, tile.n_tile
        macs = m * k * n

        # 本块 A/B/C 数据量（字节）
        a_bytes = m * k * word_bytes
        b_bytes = k * n * word_bytes
        c_bytes = m * n * (self.cfg.array.acc_width // 8)

        if tile.mode == 'OS':
            compute_cycles = k + m + n - 2
            clear_cycles = self.cfg.timing.os_clear_cycles
            drain_cycles = self.cfg.timing.os_drain_extra
            phase_cycles = {
                'clear': clear_cycles,
                'compute': compute_cycles,
                'drain': drain_cycles,
            }
            phase_energy = {
                'clear': m * n * self.cfg.energy.os_psum_clear_energy,
                'compute': macs * self.cfg.energy.mac_energy + (m * n) * self.cfg.energy.os_psum_write_energy,
                'drain': drain_cycles * self.cfg.energy.control_energy_per_cycle,
            }
            c_read = 0
            c_write = c_bytes
        else:
            preload_cycles = self.cfg.timing.ws_preload_cycles
            compute_cycles = m + n - 1
            gap_cycles = self.cfg.timing.ws_gap_cycles
            phase_cycles = {
                'preload': preload_cycles,
                'compute': compute_cycles,
                'gap': gap_cycles,
            }
            phase_energy = {
                'preload': k * n * self.cfg.energy.ws_weight_load_energy,
                'compute': macs * self.cfg.energy.mac_energy,
                'gap': gap_cycles * self.cfg.energy.control_energy_per_cycle,
            }
            c_read = c_bytes
            c_write = c_bytes

        buf = self.buffer_model.estimate(
            a_bytes=a_bytes,
            b_bytes=b_bytes,
            c_read_bytes=c_read,
            c_write_bytes=c_write,
        )
        noc = self.noc_model.estimate(a_bytes=a_bytes, b_bytes=b_bytes, c_bytes=c_write)

        cycles = sum(phase_cycles.values()) + noc.cycles + buf.stalls
        energy = sum(phase_energy.values()) + buf.energy + noc.energy
        energy += cycles * (self.cfg.energy.dispatcher_energy_per_cycle + self.cfg.energy.control_energy_per_cycle)

        return TileStats(
            tile=tile.to_dict(),
            cycles=cycles,
            energy=energy,
            macs=macs,
            buffer=buf.__dict__,
            noc=noc.__dict__,
            phase_cycles=phase_cycles,
            phase_energy=phase_energy,
        )
