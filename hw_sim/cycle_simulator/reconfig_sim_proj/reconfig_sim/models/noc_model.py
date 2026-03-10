# -*- coding: utf-8 -*-
"""
NoC 解析模型：根据 A/B/C 数据量、平均跳数和链路带宽估算传输周期与能耗（非 BookSim）。
"""
from dataclasses import dataclass
import math


@dataclass
class NoCStats:
    """单块在 NoC 上的传输统计"""
    bytes_moved: int = 0   # 总传输字节数
    hops: int = 0          # 加权总跳数（按 A/B/C 分别乘 hops_per_*）
    cycles: int = 0        # 传输占用周期（带宽与竞争因子）
    energy: float = 0.0    # NoC 传输与路由器能耗（相对单位）


class NoCModel:
    """解析 NoC 模型：基于字节数、平均跳数、链路带宽与 flit 能耗估算，适用于架构探索。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.booksim_available = False

    def estimate(self, *, a_bytes: int, b_bytes: int, c_bytes: int) -> NoCStats:
        """
        根据 A/B/C 字节数估算本块 NoC 周期与能耗。
        周期 = ceil(总字节/链路带宽) * 竞争因子；能耗 = 跳数*每字节每跳 + flit 数*每 flit 路由器能耗。
        """
        bytes_moved = a_bytes + b_bytes + c_bytes
        hops = a_bytes * self.cfg.hops_per_a + b_bytes * self.cfg.hops_per_b + c_bytes * self.cfg.hops_per_c
        cycles = math.ceil((bytes_moved / max(1, self.cfg.link_bandwidth_bytes_per_cycle)) * self.cfg.contention_factor)
        flits = math.ceil(bytes_moved / max(1, self.cfg.flit_bytes))
        energy = hops * self.cfg.energy_per_byte_per_hop + flits * self.cfg.router_energy_per_flit
        return NoCStats(bytes_moved=bytes_moved, hops=hops, cycles=cycles, energy=energy)

    def why_not_booksim(self) -> str:
        """说明为何默认使用解析模型而非 BookSim（可后续用 BookSim 做标定）。"""
        return (
            "This simulator uses an analytic NoC model by default. BookSim is useful when you need packet-level "
            "contention/latency under a concrete topology and routing policy. For systolic-array design-space exploration, "
            "an analytic model is usually faster and easier to calibrate. A practical workflow is: first use the analytic "
            "model for sweeps, then calibrate a few representative traffic points against BookSim."
        )
