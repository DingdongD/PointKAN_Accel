# -*- coding: utf-8 -*-
"""
仿真配置：脉动阵列、缓冲、NoC、时序、能耗等参数的数据类。
"""
from dataclasses import dataclass, asdict, field


@dataclass
class ArrayConfig:
    """脉动阵列硬件规模与数据位宽"""
    arr_r: int = 8          # 阵列行数（PE 行）
    arr_c: int = 8          # 阵列列数（PE 列）
    word_size: int = 16     # 激活/权重量化位宽（bit）
    acc_width: int = 64     # 累加器位宽（bit）


@dataclass
class BufferConfig:
    """片上缓冲与访存能耗模型参数"""
    global_buffer_bytes: int = 512 * 1024   # 全局缓冲总容量（字节）
    a_buffer_bytes: int = 64 * 1024        # A 矩阵缓冲容量
    b_buffer_bytes: int = 64 * 1024       # B 矩阵缓冲容量
    c_buffer_bytes: int = 64 * 1024       # C 矩阵缓冲容量
    line_bytes: int = 64                   # 缓存行/突发传输粒度
    read_energy_per_byte: float = 0.2      # 每字节读能耗（相对单位）
    write_energy_per_byte: float = 0.25   # 每字节写能耗（相对单位）


@dataclass
class NoCConfig:
    """片上网络（NoC）解析模型参数"""
    model: str = "analytic"   # 模型类型：analytic | booksim_stub
    topology: str = "mesh"    # 拓扑（如 mesh）
    link_bandwidth_bytes_per_cycle: int = 32   # 每周期链路带宽（字节）
    hops_per_a: int = 4      # A 数据平均跳数
    hops_per_b: int = 4      # B 数据平均跳数
    hops_per_c: int = 4      # C 数据平均跳数
    energy_per_byte_per_hop: float = 0.02   # 每字节每跳能耗
    router_energy_per_flit: float = 0.1     # 每 flit 路由器能耗
    flit_bytes: int = 16     # 单 flit 大小（字节）
    contention_factor: float = 1.0         # 竞争因子，用于放大周期估计


@dataclass
class TimingConfig:
    """各数据流模式的时序参数（周期数）"""
    os_clear_cycles: int = 1      # OS 模式：累加器清零周期
    os_drain_extra: int = 4      # OS 模式：排空流水线额外周期
    ws_preload_cycles: int = 2   # WS 模式：权重重载周期
    ws_gap_cycles: int = 1       # WS 模式：块间间隔周期
    mode_switch_cycles: int = 2  # 模式切换惩罚周期（OS<->WS）


@dataclass
class EnergyConfig:
    """计算与控制的能耗系数（相对单位）"""
    mac_energy: float = 1.0              # 单次 MAC 运算能耗
    os_psum_write_energy: float = 0.15   # OS：部分和写回能耗
    os_psum_clear_energy: float = 0.05   # OS：部分和清零能耗
    ws_weight_load_energy: float = 0.12  # WS：权重重载能耗
    dispatcher_energy_per_cycle: float = 0.5  # 调度器每周期能耗
    control_energy_per_cycle: float = 0.1    # 控制器每周期能耗


@dataclass
class SimConfig:
    """仿真器总配置：聚合阵列、缓冲、NoC、时序、能耗子配置"""
    array: ArrayConfig = field(default_factory=ArrayConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    noc: NoCConfig = field(default_factory=NoCConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)

    def to_dict(self):
        """转为嵌套字典，便于 JSON 导出或日志"""
        return {
            'array': asdict(self.array),
            'buffer': asdict(self.buffer),
            'noc': asdict(self.noc),
            'timing': asdict(self.timing),
            'energy': asdict(self.energy),
        }
