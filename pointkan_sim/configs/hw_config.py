"""
硬件配置模块（baseline / acc 加速器）
=====================================

面向点云网络（PointKAN / PointNet++ 等）的两种 CIM 加速器扁平配置，
无 Chip→Tile→PE 层级，单芯片由五大宏模块组成：

  1. SRAM-CIM 宏：sram-cim 阵列 + shift_add + 输入/输出 buffer + dist2_merge_adder（交叉项与自平方项距离累加）
  2. RRAM-LUT 宏：rram 查找表
  3. RRAM-CIM 宏：占位，与 MNSIM 集成
  4. 排序电路：baseline 为 Bitonic 排序，acc 为 coarse+fine 两阶段比较器；FPS 最小距离表更新归属本模块
  5. 逻辑电路：BN/LN/ReLU/量化反量化等，acc 增加插值

  FPS 完整流程：接收 SRAM-CIM Array 输出距离 → 最小距离表更新（min_distance_table）→ Sorter（base: BitonicSort MAX，acc: coarse+fine filter MAX）。
  KNN 流程：复用 SRAM-CIM Array 输出距离（存片上 output_buffer）→ Sorter（base: BitonicSort MIN，acc: coarse+fine filter MIN）。
  - base_v1/base_v2：每层都跑 FPS 和 KNN。
  - acc：FPS 仅在「第一个含 FPS 的层」执行，后续层复用第一层 FPS 结果（0 周期）；KNN 每层仍重新计算。

时钟说明（双时钟，端到端延迟按各模块自身时钟换算后累加）：
  - 数字逻辑时钟（SPool/LayerNorm/ReLU/FPS/KNN 等）：由 HardwareConfig.clock_period 表示（单位 ns）。
    接口：脚本通过 --logic_clk_freq / 环境变量 LOGIC_CLK_FREQ 传入频率(Hz)，不传则默认 1 GHz（1 ns）。
  - NeuroSim 时钟（Conv/Linear 的 CIM 仿真）：由脚本 NEUROSIM_CLK_FREQ 传入(Hz)，可变，不限于 100 MHz；
    不传时 NeuroSim 使用自身关键路径频率。Conv/Linear 事件单独带 mnsim_clock_period_ns，不改写 clock_period。

使用：get_accelerator_config("baseline") / get_accelerator_config("acc")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto


# =============================================================================
# 电路模块枚举（用于阶段定义与事件归属）
# =============================================================================

class CircuitModule(Enum):
    """
    电路模块枚举，与 HardwareConfig 中各子模块配置一一对应，
    用于 Event 的 circuit_module 字段及 breakdown 按模块聚合。
    """
    # --- SRAM-CIM Macro ---
    SRAM_CIM_ARRAY = auto()           # SramCimArrayConfig
    SHIFT_ADD = auto()                # ShiftAddConfig
    BUFFER_INPUT = auto()             # BufferConfig (输入 buffer)
    BUFFER_OUTPUT = auto()            # BufferConfig (输出 buffer，存距离等)
    DIST2_MERGE_ADDER = auto()        # Dist2MergeAdderConfig（Array 交叉项 + Buffer 自平方项累加）

    # --- FPS 最小距离表（与排序电路一起完成 FPS 流程，不属于 SRAM-CIM 距离计算）---
    MIN_DISTANCE_TABLE = auto()       # MinDistanceTableConfig，归属 sorting_circuit

    # --- RRAM-LUT Macro ---
    RRAM_LUT = auto()                 # RramLutConfig
    RRAM_LUT_SPECULATIVE = auto()     # RramLutConfig (acc only)

    # --- RRAM-CIM Macro ---
    RRAM_CIM = auto()                 # RramCimMacroConfig

    # --- Sorting Circuit ---
    DIST2_ENGINE = auto()                # BaselineDist2EngineConfig (baseline 距离计算)
    BITONIC_SORT = auto()                # BitonicSortConfig (baseline_v1)
    MERGE_SORT = auto()                 # MergeSortConfig (baseline_v2 流式)
    MERGE_SORT_REG_FILE = auto()        # MergeSort 寄存器组：FPS 存 1 个历史 max，KNN 存 32 个最小距离
    COARSE_FILTER = auto()               # CoarseFilterConfig (acc)
    FINE_FILTER = auto()                 # FineFilterConfig (acc)

    # --- Logic Circuit ---
    BATCHNORM1D = auto()              # BatchNorm1dConfig
    LAYERNORM = auto()                # LayerNormConfig
    RELU = auto()                     # ReLUConfig
    SPOOL = auto()                    # SPoolConfig
    ADAPTIVE_MAX_POOL = auto()        # AdaptiveMaxPoolConfig
    QUANT_DEQUANT = auto()            # quant_pipeline_8bit + quant_pipeline_12bit + dequant_pipeline
    GRF_BASE_ENGINE = auto()          # GRFBaseEngineConfig (baseline only, KAT 因果多项式)
    RESAMPLER = auto()                # ResamplerConfig (acc only, FPS radii 上下采样)


# =============================================================================
# 加速器类型
# =============================================================================

class AcceleratorType(Enum):
    """加速器架构变体：baseline / acc。"""
    BASELINE = auto()
    ACC = auto()


# =============================================================================
# 1. SRAM-CIM 宏 — 子模块配置
# =============================================================================

@dataclass
class SRAMCimArrayConfig:
    """
    SRAM-CIM 阵列配置：128 个 LCU，每 LCU 8 行×36 列，等效 1024 行×36 列，128 路并行。

    两种模式：
      - 写模式（write mode）：推理启动后先将 xyz 从 DRAM 写入 SRAM-CIM array，每次只能写一行，共 1024 次；
        写周期与读周期均为 1 cycle @ 1 GHz；写功耗每行 write_power_per_row_mw。
      - CIM 模式（cim mode）：in-situ 乘加；CIM 功耗每次 cim_parallelism 路计算 cim_power_per_batch_mw。
        shift-add 在阵列算完第一个数据后开始，可与 array 的 in-situ 乘法 overlap。
    """
    rows: int = 1024
    cols: int = 36
    lcu_num: int = 128
    lcu_rows: int = 8                           # rows / lcu_num
    quantization_bits: int = 12
    bit_slice_bits: int = 2
    cim_parallelism: int = 128                  # = lcu_num

    # 写模式：每行 2 mW，每行写/读 1 cycle @ 1 GHz
    write_power_per_row_mw: float = 2.0         # 每写一行功耗 (mW)
    write_cycles_per_row: int = 1               # 每行写周期
    read_cycles_per_row: int = 1                # 每行读周期
    # CIM 模式：每次 cim_parallelism 路计算 6 mW
    cim_power_per_batch_mw: float = 6.0         # 每批（128 路）CIM 计算功耗 (mW)

    # 兼容旧字段：事件生成用 .write_power（每行）/.cim_power（每批）
    @property
    def write_power(self) -> float:
        return self.write_power_per_row_mw

    @property
    def cim_power(self) -> float:
        return self.cim_power_per_batch_mw

    @property
    def bit_serial_cycles(self) -> int:
        """Number of CIM cycles per complete MAC: quantization_bits / bit_slice_bits."""
        return self.quantization_bits // self.bit_slice_bits


@dataclass
class Dist2MergeAdderConfig:
    """
    Dist2 merge adder (acc).
    """
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles

    parallelism: int = 1                        # 设计空间探索参数





@dataclass
class BufferConfig:
    """
    SRAM 写回与中间存储用 buffer。功耗建议由 CACTI 查表得到（见 hw_sim/cycle_simulator/memory_model）。

    若由 CACTI 回填：设置 read_energy_nj_per_access / write_energy_nj_per_access / access_time_ns 后，
    有效 read_power_mw = read_energy_nj_per_access / access_time_ns * 1e-3（1 GHz 时约等于 read_energy_nj * 1e-3）。
    未设置时使用固定 power 作为 fallback。
    """
    power: float = 0.0                          # fallback 固定功耗 (mW)，无 CACTI 时用
    cycles: int = 0                              # 单次访问周期，可由 CACTI access_time 换算
    area: float = 0.0                           # mm²，可由 CACTI 回填
    # CACTI 回填（memory_model CactiResult）：单次访问能耗 (nJ)、访问时间 (ns)
    read_energy_nj_per_access: Optional[float] = None
    write_energy_nj_per_access: Optional[float] = None
    access_time_ns: Optional[float] = None       # 1 GHz 时 1 cycle = 1 ns

    def get_read_power_mw(self, clock_period_ns: float = 1.0) -> float:
        """有效读功耗 (mW)：有 CACTI 数据时用 energy/time，否则用 power。"""
        if self.read_energy_nj_per_access is not None and self.access_time_ns is not None and self.access_time_ns > 0:
            return self.read_energy_nj_per_access / self.access_time_ns * 1e-3
        return self.power

    def get_write_power_mw(self, clock_period_ns: float = 1.0) -> float:
        """有效写功耗 (mW)。"""
        if self.write_energy_nj_per_access is not None and self.access_time_ns is not None and self.access_time_ns > 0:
            return self.write_energy_nj_per_access / self.access_time_ns * 1e-3
        return self.power


@dataclass
class ShiftAddConfig:
    """Shift-and-add：距离累加 k^2 + (-2qk)，与 SRAM-CIM 阵列配合；CIM 算完首数据后可与之 overlap。"""
    power: float = 1.3                          # mW
    cycles: int = 1                              # 每批 1 cycle @ 1 GHz
    area: float = 0.0                           # um^2


@dataclass
class MinDistanceTableConfig:
    """FPS 中用于更新最小距离的查找表。"""
    power: float = 30.0                         # TODO: mW
    cycles: int = 0                             # TODO: execution cycles
    area: float = 26610.0                        # TODO: um^2


@dataclass
class SRAMCimMacroConfig:
    """
    SRAM-CIM 宏：仅负责距离计算相关电路，baseline 与 acc 共用。

    包含：
      - array: SRAM-CIM 阵列（交叉项 -2qk 等）
      - shift_add: 与阵列配合做距离累加（k^2 + (-2qk)）
      - input_buffer: 输入 buffer
      - output_buffer: 输出 buffer（存阵列输出距离等，KNN 复用存片上距离）
      - dist2_merge_adder: 将 SRAM-CIM Array 产生的交叉项距离与 SRAM-Buffer 中的自平方项距离累加（acc 用）

    FPS 最小距离表（min_distance_table）归属排序电路，与 FPS 流程中的「接收 Array 输出 → 最小距离表更新 → Sorter」一致。
    """
    array: SRAMCimArrayConfig = field(default_factory=SRAMCimArrayConfig)
    shift_add: ShiftAddConfig = field(default_factory=ShiftAddConfig)
    input_buffer: BufferConfig = field(default_factory=BufferConfig)
    output_buffer: BufferConfig = field(default_factory=BufferConfig)
    dist2_merge_adder: Dist2MergeAdderConfig = field(
        default_factory=Dist2MergeAdderConfig
    )



# =============================================================================
# 2. RRAM-LUT 宏
# =============================================================================

@dataclass
class RRAMLutConfig:
    """单块 RRAM 查找表单元。"""
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class RRAMLutMacroConfig:
    """
    RRAM-LUT Macro.

    baseline: single rram_lut instance.
    acc:      two instances — one for GRF, one for speculative sampling.
    """
    rram_lut: RRAMLutConfig = field(default_factory=RRAMLutConfig)
    # 用于acc mode中FPS推测采样需要的
    rram_lut_speculative: Optional[RRAMLutConfig] = None


# =============================================================================
# 3. RRAM-CIM 宏（占位，Neurosim/MNSim集成）
# =============================================================================

@dataclass
class RRAMCimMacroConfig:
    """
    RRAM-CIM 宏：功耗/面积由 NeuroSim/MNSIM 等工具按网络与工艺动态计算，此处不填固定值。
    """
    power: float = 0.0                          # 由 NeuroSim/MNSIM 动态回填，mW
    area: float = 0.0                           # 由工具动态回填，mm²
    cycles: int = 0                             # 由工具或公式

    enabled: bool = False                       # Not yet integrated


# =============================================================================
# 4. 排序电路
# =============================================================================

@dataclass
class BaselineDist2EngineConfig:
    """
    Baseline 距离计算引擎（3×64 脉动阵列），与 bitonic max/min 配合完成 FPS/KNN。

    T = M + R + C - 2（M=批内点数，R=3，C=64）；64 点批 129 cycle，1024 点 1089 cycle。
    功耗 94 mW，面积 89884 um^2。仅 baseline 使用；acc 用 SRAM-CIM 阵列。
    """
    power: float = 94.0                          # mW
    area: float = 89884.0                        # um^2
    cycles: int = 0                              # 0 表示按公式 T=M+R+C-2 计算
    m_dim: int = 0                               # 公式用 M（批内点数），0=运行时传入
    r_dim: int = 3
    c_dim: int = 64


@dataclass
class BitonicSortConfig:
    """
    Bitonic 排序电路（baseline_v1）：同时支持 MAX 与 MIN 模式，面积共用，功耗/周期分模式。

    - MAX 模式（FPS）：取全局最大，周期由 FPSParams.bitonic_sort_cycles 决定（默认 10），功耗 power_max_mw。
    - MIN 模式（KNN）：1024→k 最小，周期由 KNNParams.bitonic_sort_cycles 决定（默认 19），功耗 power_min_mw。
    共享模块 bitonic_sort_1024_max_32min，功耗 2097.4mW，面积 2157561 um^2。
    """
    power_max_mw: float = 1570                 # mW，MAX 模式（FPS）
    power_min_mw: float = 1540                 # mW，MIN 模式（KNN）
    area: float = 2157561.0                      # um^2（两种模式共用）


@dataclass
class MergeSortRegFileConfig:
    """
    MergeSort 配套寄存器组：模拟 FPS 历史最大值与 KNN 32 最小距离缓存的读写。

    - 读写周期同理：每读/写 1 个值 cycles_per_read / cycles_per_write 周期，功耗 power_mw。
    - 写可与下一批的 64选1（FPS）或 64选32（KNN）重叠，不占关键路径（overlapped）。
    - FPS：每批读 1、写 1；KNN：每批读 32、写 32（写 overlapped）。
    """
    power_mw: float = 20.0                      # mW，读写时功耗
    cycles_per_read: int = 1                     # 每读 1 个值的周期数
    cycles_per_write: int = 1                    # 每写 1 个值的周期数；写可与下一批 sort 重叠


@dataclass
class MergeSortConfig:
    """
    MergeSort 排序电路（baseline_v2）：配合 dist2_engine 64 点带宽的流式 FPS/KNN。

    - FPS：每批 64 点算距离 → min_dist 更新 → 本批 64 个距离局部 mergesort 取 max（22 cycle，310 mW），
      与全局历史 max 比较更新（reg_file 写 1 个最大值）；16 批后得到下一个最远点。
    - KNN：每批 64 点 → top32 排序（22 cycle，312 mW），再与 buffer 中 32 归并、截断保留前 32，
      归并后 reg_file 写 32 个最小距离；循环至所有批处理完。
    面积 180441 um^2。
    """
    power_max_mw: float = 310.0                  # mW，FPS 局部 max
    power_top32_mw: float = 312.0              # mW，KNN top32
    area: float = 180441.0                      # um^2
    cycles_max: int = 22                        # FPS 每批局部 max 周期
    cycles_top32: int = 22                       # KNN 每批 top32 周期
    reg_file: Optional[MergeSortRegFileConfig] = None  # 寄存器组：FPS 1 写/迭代，KNN 32 写/query


@dataclass
class CoarseFilterConfig:
    """
    Coarse filter (acc)：粗筛，功耗与面积三种模式共用；周期分三种模式。

    - 功耗：38 mW（需求 3.4）
    - 面积：644686 um^2
    - 周期：三种模式可配置 cycles_mode1/2/3；若为 0 则运行时按公式 4*rounds+1 计算。
    """
    power: float = 38.0                          # mW（三种模式相同）
    area: float = 644686.0                       # um^2（三种模式相同）
    cycles: int = 0                              # 通用 override，0 表示按公式
    cycles_mode1: int = 0                         # 模式 1 固定周期，0=用公式
    cycles_mode2: int = 0                         # 模式 2 固定周期，0=用公式
    cycles_mode3: int = 0                         # 模式 3 固定周期，0=用公式
    cycles_per_round: int = 4                     # 公式：cycles = cycles_per_round * rounds + cycles_base
    cycles_base: int = 1


@dataclass
class FineFilterConfig:
    """
    Fine filter (acc).
    """
    power: float = 38.0                          # TODO: mW
    area: float = 53261.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class SortingCircuitConfig:
    """
    Sorting circuit module：仅包含 FPS/KNN 的排序与 FPS 最小距离表，与距离计算无关。

    - min_distance_table: FPS 中用于更新最小距离的查找表；
      FPS 流程 = 接收距离（来自 dist2_engine 或 SRAM-CIM）→ min_distance_table 更新 → Sorter。
    - baseline_v1: bitonic_sort（MAX 用于 FPS，MIN 用于 KNN）
    - baseline_v2: merge_sort（配合 dist2_engine 64 带宽流式 FPS/KNN，FPS 局部 max 22cy/310mW，KNN top32 22cy/312mW）
    - acc:      coarse_filter + fine_filter（FPS 用 coarse+fine 做 max，KNN 同理做 min）

    baseline 的距离计算由 HardwareConfig.baseline_dist2_engine（3×64 脉动）负责，不在此模块。
    """
    # FPS 最小距离表更新（归属 FPS 流程）
    min_distance_table: Optional[MinDistanceTableConfig] = None
    bitonic_sort: Optional[BitonicSortConfig] = None
    merge_sort: Optional[MergeSortConfig] = None   # baseline_v2 流式 FPS/KNN
    coarse_filter: Optional[CoarseFilterConfig] = None
    fine_filter: Optional[FineFilterConfig] = None


# =============================================================================
# 4.1 Baseline 特征计算：im2col + N×N 脉动阵列（仅 base_v1 / base_v2）
# =============================================================================

@dataclass
class Im2ColConfig:
    """
    Conv 的 im2col 阶段（baseline 仅 base_v1/base_v2 使用）。
    将卷积展成矩阵乘前的数据重排与搬移，功耗与周期可标定。
    """
    power: float = 15.0                          # mW
    area: float = 0.0                            # um^2，可选
    cycles_per_output: int = 1                   # 每输出点的等效周期（0=按公式）


@dataclass
class SystolicArrayConfig:
    """
    Baseline 特征计算用 N×N 脉动阵列（Conv/Linear 矩阵乘）。
    仅 base_v1/base_v2 使用；acc 用 MNSIM/NeuroSim（RRAM-CIM）。
    当 M/N/K 超过 rows/cols 时按 Tile 分块：tiles_M=ceil(M/rows), tiles_N=ceil(N/cols),
    tiles_K=ceil(K/rows)，总周期 = tiles_M * tiles_N * tiles_K * 单块延迟（单块 = rows+cols+K_tile-2）。

    MAC 能耗（SCALE-Sim 脉动）：单 PE 功耗 mac_power_mw (mW)，1 GHz 下 1 cycle = 1 ns，
    故单次 MAC 能耗 = mac_power_mw * 1 ns = 7.144 pJ；通用为 mac_power_mw * clock_period_ns (pJ)。
    """
    rows: int = 64                               # 阵列行数，传入 SCALE-Sim 作 ArrayHeight 控制阵列尺寸
    cols: int = 64                               # 阵列列数，传入 SCALE-Sim 作 ArrayWidth 控制阵列尺寸
    mac_power_mw: float = 7.144                  # 单 PE 的 MAC 功耗 (mW)，1 GHz 下 1 cycle 能耗 = 7.144 mW×1 ns = 7.144 pJ/MAC
    area: float = 0.0                            # um^2，可选
    cycles: int = 0                              # 0=按公式+Tile 计算；>0=固定周期覆盖
    data_bytes: int = 2                          # SCALE-Sim 单次访问的字节数（FXP16=2B），用于 trace/CACTI 映射
    sram_size_bytes: int = 32768                 # SCALE-Sim 局部 SRAM统一容量默认值（bytes），若未细分则回退到此值
    ifmap_sram_size_bytes: Optional[int] = None  # IFMAP 本地 SRAM 容量（bytes）；None 时回退到 sram_size_bytes
    filter_sram_size_bytes: Optional[int] = None # FILTER/weight 本地 SRAM 容量（bytes）；None 时回退到 sram_size_bytes
    ofmap_sram_size_bytes: Optional[int] = None  # OFMAP 本地 SRAM 容量（bytes）；None 时回退到 sram_size_bytes
    sram_block_bytes: int = 64                   # SRAM block size，用于 CACTI
    sram_num_banks: int = 1                      # SRAM bank 数，用于 CACTI
    sram_bus_width_bits: int = 128               # SRAM 输出总线位宽，用于 CACTI
    sram_freq_hz: Optional[float] = None         # SCALE-Sim 局部 SRAM 时钟；None 时跟随数字逻辑时钟


# =============================================================================
# 5. Logic Circuit  —  sub-modules (ops + quant/dequant + resampler + grf_base_engine)
# =============================================================================

@dataclass
class BatchNorm1dConfig:
    """BatchNorm1d hardware unit."""
    power: float = 3.1718                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles
    
    area: float = 3228.0                        # um^2

    parallelism: int = 1                        # 设计空间探索参数


@dataclass
class LayerNormConfig:
    """LayerNorm hardware unit. 周期公式：Nvec * (3*C/8 + 17)，C=normalized_shape。"""
    power: float = 56.0                         # mW @ parallelism=1

    area: float = 139582.0                      # um^2

    cycles: int = 0                             # shape-driven, 0 表示运行时按公式 Nvec*(3C/8+17) 估算

    parallelism: int = 1                        # 设计空间探索参数


@dataclass
class ReLUConfig:
    """ReLU activation hardware unit."""

    power: float = 0.45                         # mW @ parallelism=1
    
    area: float = 222.0                          # um^2
    
    cycles: int = 0                             # shape-driven, 0 表示运行时按公式估算
    
    parallelism: int = 1                        # 设计空间探索参数


@dataclass
class SPoolConfig:
    """SPool (spatial pooling) hardware unit."""
    power: float = 65.26 #26.82                         # mW @ parallelism=1

    area: float = 54926.0                       # um^2 @ parallelism=1

    cycles: int = 0                             # shape-driven, 0 表示运行时按公式估算
    
    parallelism: int = 1                        # 设计空间探索参数


@dataclass
class AdaptiveMaxPoolConfig:
    """AdaptiveMaxPool1d / MaxPool 硬件单元（与 SPool 独立）。"""
    power: float = 13.2                         # mW @ parallelism=1
    area: float = 18406                           # TODO: um^2
    cycles: int = 0                             # shape-driven, 运行时按 B×C×L/par 估算
    parallelism: int = 1                        # 设计空间探索参数


@dataclass
class QuantPipeline8bitConfig:
    """FP32→INT8 量化流水线。T = N + 6（N=待量化元素数，6 cycles 流水线填充延迟）。"""
    power: float = 2.234                        # mW @ 1 GHz
    area: float = 2460.0                         # um^2
    pipeline_latency: int = 6                    # 流水线填充延迟 cycles
    parallelism: int = 1


@dataclass
class QuantPipeline12bitConfig:
    """FP32→INT12 量化流水线（如几何坐标）。T = N + 6（N=待量化元素数，6 cycles 流水线填充延迟）。"""
    power: float = 2.28                          # mW @ 1 GHz
    area: float = 2525.0                         # um^2
    pipeline_latency: int = 6                     # 流水线填充延迟 cycles
    parallelism: int = 1


@dataclass
class DequantPipelineConfig:
    """INT8→FP32 反量化流水线。T = N + 4（N=待反量化元素数，4 cycles 流水线填充延迟）。"""
    power: float = 2.30                          # mW @ 1 GHz
    area: float = 2078.0                          # um^2
    pipeline_latency: int = 4                    # 流水线填充延迟 cycles
    parallelism: int = 1


@dataclass
class FP32AddConfig:

    power: float = 1.16                         # mW @ parallelism=1
    area: float = 1273.0                        # um^2
    cycles_per_element: int = 1                 # 每元素周期数
    parallelism: int = 1


@dataclass
class FP32MulConfig:
    
    power: float = 2.5                          # mW @ parallelism=1
    area: float = 2024.0                        # um^2
    cycles_per_element: int = 1                 # 每元素周期数
    parallelism: int = 1


@dataclass
class FP32DivConfig:
    
    power: float = 2.59                         # mW @ parallelism=1
    area: float = 6307.0                        # um^2
    cycles_per_element: int = 10                # 每元素周期数
    parallelism: int = 1


@dataclass
class ResamplerConfig:
    """
    Resampler（acc only）：FPS radii 推测采样的线性插值，分下采样/上采样两种模式。

    - 下采样、上采样：功耗均为 power（3.95mW），核心面积均为 area（4719 um^2）；考虑并行时
      功耗与核心面积乘以 parallelism，即 power_total = power * P，area_core_total = area * P。
    - LUT 面积不乘 parallelism：下采样 LUT lut_area_downsample，上采样 LUT lut_area_upsample，
      只计一次。总面积 = area * parallelism + lut_area_downsample + lut_area_upsample。
    需求 3.4：T=N+2 或 T=N/P+2。
    """
    power: float = 3.95                         # mW @ 每路；总功耗 = power * parallelism
    cycles: int = 0                             # 0 表示 T=N+2 或 T=N/P+2
    area: float = 4719.0                        # um^2 @ 每路；核心总面积 = area * parallelism
    parallelism: int = 1                        # 设计空间探索参数
    lut_area_downsample: float = 13485.0         # um^2，下采样 LUT，不乘 parallelism
    lut_area_upsample: float = 85331.0          # um^2，上采样 LUT，不乘 parallelism


@dataclass
class GRFBaseEngineConfig:
    """
    GRFBaseEngine：baseline（base mode）下 Efficient_KAN 的 KAT 因果多项式计算（数字实现）。

    - 仅 baseline 使用；实现因果有理多项式，与 RRAM-LUT 无关。
    - acc 模式下 KAT 由 RRAM-LUT Macro 的 rram_lut 实现，不用本配置。
    需求（数字 GRF）：约 19 cycle/数据，23.7833 mW，47877 um^2，并行度 P。
    """
    power: float = 23.783                          # TODO: mW（需求约 23.7833）
    area: float = 47887.0                           # TODO: um^2（需求约 47877）
    
    cycles: int = 0                             # 每数据周期，需求约 19；0 表示按公式
    
    parallelism: int = 1                        # 可绑定 KAT_Group 分组并行


@dataclass
class LogicCircuitConfig:
    """
    Logic circuit module.

    Shared ops: batchnorm1d, layernorm, relu, spool, adaptive_max_pool,
               quant_pipeline_8bit, quant_pipeline_12bit, dequant_pipeline（三条量化/反量化流水线），
               fp32_add, fp32_mul, fp32_div（FP32 加减乘除统一配置）。
    baseline: 另有 grf_base_engine（Efficient_KAN 的 KAT 因果多项式，数字）。
    acc:      另有 resampler（FPS radii 上下采样）；KAT 由 rram_lut_macro.rram_lut 实现。

    脚本中设置各模块并行度（设计空间探索）:
      - 直接赋值: hw_config.logic_circuit.layernorm.parallelism = 4
      - 或调用: set_logic_parallelism(hw_config, layernorm=4, relu=8, spool=2, quant_dequant=1)  # quant_dequant 会同时设三条流水线
    """
    # Shared ops
    batchnorm1d: BatchNorm1dConfig = field(default_factory=BatchNorm1dConfig)
    layernorm: LayerNormConfig = field(default_factory=LayerNormConfig)
    relu: ReLUConfig = field(default_factory=ReLUConfig)
    spool: SPoolConfig = field(default_factory=SPoolConfig)
    adaptive_max_pool: AdaptiveMaxPoolConfig = field(default_factory=AdaptiveMaxPoolConfig)
    quant_pipeline_8bit: QuantPipeline8bitConfig = field(default_factory=QuantPipeline8bitConfig)
    quant_pipeline_12bit: QuantPipeline12bitConfig = field(default_factory=QuantPipeline12bitConfig)
    dequant_pipeline: DequantPipelineConfig = field(default_factory=DequantPipelineConfig)
    fp32_add: FP32AddConfig = field(default_factory=FP32AddConfig)
    fp32_mul: FP32MulConfig = field(default_factory=FP32MulConfig)
    fp32_div: FP32DivConfig = field(default_factory=FP32DivConfig)
    # baseline only: KAT 因果多项式（数字），acc 下 KAT 用 rram_lut_macro.rram_lut
    grf_base_engine: Optional[GRFBaseEngineConfig] = None
    # acc only: FPS radii 上下采样（lin_resample_1d 硬件）
    resampler: Optional[ResamplerConfig] = None


# =============================================================================
# DRAM (外部内存，Roofline + CACTI 功耗)
# =============================================================================

@dataclass
class DRAMConfig:
    """
    外部 DRAM：点云加载、查询点读取、结果写回等。
    Roofline: cycles = bytes / bandwidth_bytes_per_cycle；能耗 = cycles * clock * power_mw。
    功耗建议由 DRAMSim3 查表得到（见 hw_sim/cycle_simulator/memory_model）；未回填时使用固定 read_power_mw/write_power_mw。
    SCALE-Sim 脉动：可选两种方式——
    (1) 按访问次数×单次能耗：read_energy_nj_per_access / write_energy_nj_per_access（DRAMSim3 回填）；
    (2) 由 SCALE-Sim 生成 trace，转为 DRAMSim3 格式后跑仿真得能耗（见 demo_with_accelergy.py），需设置 dramsim3_bin 与 dramsim3_config_ini。
    """
    bandwidth_bytes_per_cycle: int = 32
    read_power_mw: float = 50.0                  # 固定 fallback；可由 DRAMSim3 回填
    write_power_mw: float = 50.0                 # 固定 fallback；可由 DRAMSim3 回填
    read_energy_nj_per_access: Optional[float] = None   # DRAMSim3 回填：单次读能耗 (nJ)，SCALE-Sim 按访问计能耗
    write_energy_nj_per_access: Optional[float] = None   # DRAMSim3 回填：单次写能耗 (nJ)
    block_size_bytes: int = 64                          # 访问粒度，用于 num_accesses = ceil(bytes/block_size_bytes)
    dramsim3_bin: Optional[str] = None                  # 可选：trace→DRAMSim3 流程的可执行文件路径
    dramsim3_config_ini: Optional[str] = None            # 可选：DRAMSim3 .ini 配置路径，与 dramsim3_bin 同时设置时用 trace 跑仿真得 DRAM 能耗


# =============================================================================
# SRAM-II（距离计算专用，128 行激活，2bit 距离）
# =============================================================================

@dataclass
class SRAM_IIConfig:
    """
    用于距离计算的 SRAM-II：存储 key 点云，每次激活 128 行，2bit 距离计算。
    每查询点周期数 = (bits_per_axis / bit_slice) * ceil(N / active_rows)。
    例如 12bit 轴、2bit slice、N=1024、128 行 → 6 * 8 = 48 cycles/query。
    """
    active_rows: int = 128
    bits_per_axis: int = 12
    bit_slice: int = 2
    power_mw: float = 0.0  # 距离计算阶段功耗，TODO 填 CACTI/实测


# =============================================================================
# Inter-Layer Buffer (层间激活缓存)
# =============================================================================

@dataclass
class InterLayerBufferConfig:
    """
    层间激活 buffer：模拟上一层输出写回、下一层从 buffer 读入的容量与带宽限制。

    当单层输出数据量超过 size_bytes 时，超出部分按 spill 到 DRAM 建模（可选），
    用于模拟实际内存行为与多次读写。
    SCALE-Sim 脉动 SRAM 能耗：按访问次数×单次能耗计，由 read_energy_nj_per_access / write_energy_nj_per_access 提供（CACTI 回填）。

    Attributes:
        size_bytes: Buffer 容量（字节）；超过则产生 spill 或分块
        bandwidth_bytes_per_cycle: 每周期可读/写字节数（带宽）
        read_power_mw: Buffer 读功率 (mW)
        write_power_mw: Buffer 写功率 (mW)
        read_energy_nj_per_access: CACTI 回填：单次读能耗 (nJ)，用于 SCALE-Sim 等按访问计能耗
        write_energy_nj_per_access: CACTI 回填：单次写能耗 (nJ)
        access_time_ns: CACTI 回填：单次访问延迟 (ns)，用于统一内存接口的 cycle 计算
        block_size_bytes: 访问粒度（字节），用于按访问数计 cycle/能耗，默认 64
        spill_to_dram: 是否将超出 capacity 的部分建模为 DRAM 读写
        dram_bandwidth_bytes_per_cycle: DRAM 带宽（spill 时用）
        dram_read_power_mw: DRAM 读功率 (mW)
        dram_write_power_mw: DRAM 写功率 (mW)
    """
    size_bytes: int = 65536                    # 64 KB 片上 buffer
    bandwidth_bytes_per_cycle: int = 256       # 256 B/cycle
    read_power_mw: float = 2.0                 # 固定 fallback；建议由 CACTI 回填（见 memory_model_adapter）
    write_power_mw: float = 2.0                # 固定 fallback；建议由 CACTI 回填
    read_energy_nj_per_access: Optional[float] = None   # CACTI 回填：单次读能耗 (nJ)
    write_energy_nj_per_access: Optional[float] = None  # CACTI 回填：单次写能耗 (nJ)
    access_time_ns: Optional[float] = None             # CACTI 回填：单次访问延迟 (ns)，统一接口用
    block_size_bytes: int = 64                         # 访问粒度，用于 num_accesses = ceil(bytes/block_size_bytes)
    spill_to_dram: bool = True
    dram_bandwidth_bytes_per_cycle: int = 32
    dram_read_power_mw: float = 50.0
    dram_write_power_mw: float = 50.0
    pipeline_mode: str = "without_pipeline"      # feature SRAM 调度模式: without_pipeline / pipeline
    double_buffering: bool = False             # pipeline 模式下是否为层间 feature SRAM 预留双缓冲
    buffer_slots: int = 0                      # 0=自动推导；>0 时按槽数切分 size_bytes 得到每个 in-flight tensor 的可用容量


# =============================================================================
# 顶层硬件配置（单芯片、五大宏模块扁平组合）
# =============================================================================

@dataclass
class HardwareConfig:
    """
    单芯片加速器顶层配置，无 Chip→Tile→PE 层级，由五大宏 + 层间 buffer + 可选 DRAM/SRAM-II 组成。

    Attributes:
        accelerator_type: BASELINE or ACC
        accelerator_name: Human-readable name string
        clock_period: 数字逻辑时钟周期 (ns)，用于 SPool/LayerNorm/ReLU/FPS/KNN 等数字模块；
                      默认 1.0 ns = 1 GHz；可通过 --logic_clk_freq / LOGIC_CLK_FREQ 从脚本传入覆盖。
        sram_cim_macro: SRAM-CIM macro configuration
        rram_lut_macro: RRAM-LUT macro configuration
        rram_cim_macro: RRAM-CIM macro configuration (placeholder)
        baseline_dist2_engine: Baseline 距离计算（3×64 脉动），仅 baseline 使用；acc 用 SRAM-CIM
        sorting_circuit: 排序电路（bitonic/coarse+fine、FPS 最小距离表），与距离计算无关
        systolic_array: Baseline 特征计算 N×N 脉动阵列（Conv/Linear）；仅 base_v1/base_v2，acc 用 MNSIM/NeuroSim
        im2col: Baseline Conv 的 im2col 阶段；仅 base_v1/base_v2
        logic_circuit: Logic circuit configuration
        inter_layer_buffer: 层间激活 buffer 配置（容量、带宽、spill）
    """
    # Identity
    accelerator_type: AcceleratorType = AcceleratorType.BASELINE
    accelerator_name: str = "baseline"

    # Global timing（数字逻辑时钟，Conv/Linear 若用 NeuroSim 则另有 mnsim_clock_period_ns）
    clock_period: float = 1.0                   # ns，默认 1 GHz；脚本可设 --logic_clk_freq 覆盖

    # Five macro-level modules
    sram_cim_macro: SRAMCimMacroConfig = field(
        default_factory=SRAMCimMacroConfig
    )
    rram_lut_macro: RRAMLutMacroConfig = field(
        default_factory=RRAMLutMacroConfig
    )
    rram_cim_macro: RRAMCimMacroConfig = field(
        default_factory=RRAMCimMacroConfig
    )
    # Baseline 距离计算（3×64 脉动阵列），与排序无关；仅 baseline 使用，acc 用 sram_cim_macro
    baseline_dist2_engine: Optional[BaselineDist2EngineConfig] = None
    sorting_circuit: SortingCircuitConfig = field(
        default_factory=SortingCircuitConfig
    )
    # Baseline 特征计算：im2col + N×N 脉动阵列（仅 base_v1/base_v2；acc 用 MNSIM/NeuroSim）
    systolic_array: Optional['SystolicArrayConfig'] = None
    im2col: Optional['Im2ColConfig'] = None
    logic_circuit: LogicCircuitConfig = field(
        default_factory=LogicCircuitConfig
    )
    inter_layer_buffer: InterLayerBufferConfig = field(
        default_factory=InterLayerBufferConfig
    )
    # DRAM：点云/激活加载与写回，roofline + 功耗（可接 CACTI）
    dram: Optional[DRAMConfig] = None
    # SRAM-II：距离计算专用，128 行激活、2bit 距离
    sram_ii: Optional[SRAM_IIConfig] = None
    # 架构开关
    # distance_in_phase2=True：FPS/KNN 的距离计算(stage1-4)归入 Phase2，引擎只做 bitonic/排序
    distance_in_phase2: bool = True
    # distance_include_query_dram_read=True：Phase2 距离前，建模 query 点从 DRAM 读入
    distance_include_query_dram_read: bool = True
    # FPS 引擎模式：baseline/fuse（当前 fuse 先复用 baseline 排序路径，可继续扩展）
    fps_engine_mode: str = "baseline"
    # KNN 引擎模式：baseline/fuse（fuse 使用 coarse+fine 两级筛选事件模型）
    knn_engine_mode: str = "baseline"
    # bn_fuse_with_conv_linear=True：推理时 BN 紧跟 Conv/Linear 则融合，不计 BN cycle/能耗
    bn_fuse_with_conv_linear: bool = True
    # W8A8：Conv/Linear 前插入量化(FP32→INT8)、后插入反量化(INT8→FP32)，供 BN/ReLU 等浮点非线性使用
    enable_w8a8_quant_dequant: bool = False
    # CIM 后端：mnsim（PE 级/全 NetStruct）、neurosim（DNN_NeuroSim main）、fallback（解析回退）
    cim_backend: str = "mnsim"
    # NeuroSim 根目录（Inference_pytorch），用于 cim_backend=neurosim 时调用 NeuroSIM/main
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"
    # Baseline 脉动阵列：是否用 SCALE-Sim 统计周期与 MAC/访问（能耗仍用本工程内存建模）
    use_scalesim_for_systolic: bool = False
    scalesim_root: str = ""           # 空则用环境变量 SCALE_SIM_ROOT 或默认 /workspace/SCALE-Sim
    scalesim_config_path: str = ""    # 空则用 scalesim_root/configs/demo_64x64_32kb_ws.cfg

    # =========================================================================
    # Derived convenience properties
    # =========================================================================

    @property
    def cim_parallelism(self) -> int:
        """Number of LCUs working in parallel."""
        return self.sram_cim_macro.array.cim_parallelism

    @property
    def bit_serial_cycles(self) -> int:
        """CIM cycles per complete MAC operation."""
        return self.sram_cim_macro.array.bit_serial_cycles

    @property
    def is_baseline(self) -> bool:
        return self.accelerator_type == AcceleratorType.BASELINE

    @property
    def is_acc(self) -> bool:
        return self.accelerator_type == AcceleratorType.ACC

    @property
    def is_baseline_v2(self) -> bool:
        """True 表示 baseline_v2（merge_sort 流式 FPS/KNN），False 表示 baseline_v1（bitonic）或 acc。"""
        return getattr(self, "accelerator_name", "") == "base_v2"

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_all_modules(self) -> Dict[str, Any]:
        """Return a flat dict of all circuit sub-module configs for iteration."""
        modules: Dict[str, Any] = {}

        # SRAM-CIM macro
        modules['sram_cim_array'] = self.sram_cim_macro.array
        modules['shift_add'] = self.sram_cim_macro.shift_add
        modules['input_buffer'] = self.sram_cim_macro.input_buffer
        modules['output_buffer'] = self.sram_cim_macro.output_buffer
        modules['dist2_merge_adder'] = self.sram_cim_macro.dist2_merge_adder
        # FPS 最小距离表归属排序电路
        if self.sorting_circuit.min_distance_table is not None:
            modules['min_distance_table'] = self.sorting_circuit.min_distance_table

        # RRAM-LUT macro
        modules['rram_lut'] = self.rram_lut_macro.rram_lut
        if self.rram_lut_macro.rram_lut_speculative is not None:
            modules['rram_lut_speculative'] = (
                self.rram_lut_macro.rram_lut_speculative
            )

        # RRAM-CIM macro
        modules['rram_cim_macro'] = self.rram_cim_macro

        # Baseline 距离计算（与排序无关）
        if self.baseline_dist2_engine is not None:
            modules['dist2_engine'] = self.baseline_dist2_engine
        # Sorting circuit
        if self.sorting_circuit.bitonic_sort is not None:
            modules['bitonic_sort'] = self.sorting_circuit.bitonic_sort
        if self.sorting_circuit.merge_sort is not None:
            modules['merge_sort'] = self.sorting_circuit.merge_sort
            if getattr(self.sorting_circuit.merge_sort, "reg_file", None) is not None:
                modules['merge_sort_reg_file'] = self.sorting_circuit.merge_sort.reg_file
        if self.sorting_circuit.coarse_filter is not None:
            modules['coarse_filter'] = self.sorting_circuit.coarse_filter
        if self.sorting_circuit.fine_filter is not None:
            modules['fine_filter'] = self.sorting_circuit.fine_filter
        # Baseline 特征计算（仅 base_v1/base_v2）
        if self.systolic_array is not None:
            modules['systolic_array'] = self.systolic_array
        if self.im2col is not None:
            modules['im2col'] = self.im2col

        # Logic circuit
        modules['batchnorm1d'] = self.logic_circuit.batchnorm1d
        modules['layernorm'] = self.logic_circuit.layernorm
        modules['relu'] = self.logic_circuit.relu
        modules['spool'] = self.logic_circuit.spool
        modules['adaptive_max_pool'] = self.logic_circuit.adaptive_max_pool
        modules['quant_pipeline_8bit'] = self.logic_circuit.quant_pipeline_8bit
        modules['quant_pipeline_12bit'] = self.logic_circuit.quant_pipeline_12bit
        modules['dequant_pipeline'] = self.logic_circuit.dequant_pipeline
        modules['fp32_add'] = self.logic_circuit.fp32_add
        modules['fp32_mul'] = self.logic_circuit.fp32_mul
        modules['fp32_div'] = self.logic_circuit.fp32_div
        if self.logic_circuit.grf_base_engine is not None:
            modules['grf_base_engine'] = self.logic_circuit.grf_base_engine
        if self.logic_circuit.resampler is not None:
            modules['resampler'] = self.logic_circuit.resampler

        return modules

    def to_dict(self) -> Dict[str, Any]:
        """Serialise key parameters to a flat dictionary for reporting."""
        return {
            'accelerator_type': self.accelerator_type.name,
            'accelerator_name': self.accelerator_name,
            'clock_period_ns': self.clock_period,
            'cim_parallelism': self.cim_parallelism,
            'bit_serial_cycles': self.bit_serial_cycles,
            'sram_cim_rows': self.sram_cim_macro.array.rows,
            'sram_cim_cols': self.sram_cim_macro.array.cols,
            'lcu_num': self.sram_cim_macro.array.lcu_num,
            'quantization_bits': self.sram_cim_macro.array.quantization_bits,
            'bit_slice_bits': self.sram_cim_macro.array.bit_slice_bits,
            'modules': list(self.get_all_modules().keys()),
        }


# =============================================================================
# Factory Functions — one per accelerator variant
# =============================================================================

def _apply_default_dramsim3_paths(cfg: HardwareConfig) -> HardwareConfig:
    """Populate DRAMSim3 defaults from the in-repo cycle simulator when available."""
    dram = getattr(cfg, "dram", None)
    if dram is None:
        return cfg

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cycle_sim_root = os.path.join(repo_root, "hw_sim", "cycle_simulator")
    dramsim3_bin = os.path.join(cycle_sim_root, "DRAMsim3", "build", "dramsim3main")
    dramsim3_ini = os.path.join(cycle_sim_root, "DRAMsim3", "configs", "LPDDR4_8Gb_x16_2400.ini")

    if getattr(dram, "dramsim3_bin", None) is None and os.path.isfile(dramsim3_bin):
        dram.dramsim3_bin = dramsim3_bin
    if getattr(dram, "dramsim3_config_ini", None) is None and os.path.isfile(dramsim3_ini):
        dram.dramsim3_config_ini = dramsim3_ini
    return cfg


def _build_baseline() -> HardwareConfig:
    """
    Build **baseline** accelerator configuration.

    Modules:
      - SRAM-CIM Macro:  array + shift_add + input_buffer + output_buffer + dist2_merge_adder
      - RRAM-LUT Macro:  single rram_lut
      - RRAM-CIM Macro:  placeholder (TODO)
      - Sorting Circuit:  comp_tree_1024_max + comp_bitonic_1024_min32
      - Logic Circuit:    batchnorm1d, layernorm, relu, spool, quant_pipeline_8/12bit, dequant_pipeline
    """
    return _apply_default_dramsim3_paths(HardwareConfig(
        accelerator_type=AcceleratorType.BASELINE,
        accelerator_name="baseline",
        clock_period=1.0,                       # 1 ns = 1 GHz

        sram_cim_macro=SRAMCimMacroConfig(
            array=SRAMCimArrayConfig(
                rows=1024,
                cols=36,
                lcu_num=128,
                lcu_rows=8,
                quantization_bits=12,
                bit_slice_bits=2,
                cim_parallelism=128,
                write_power_per_row_mw=2.0,    # 每行写 2 mW
                write_cycles_per_row=1,
                read_cycles_per_row=1,
                cim_power_per_batch_mw=6.0,    # 每批 128 路 CIM 6 mW
            ),
            shift_add=ShiftAddConfig(
                power=1.3,                      # mW
                cycles=1,                       # 1 cycle @ 1 GHz
                area=0.0,
            ),
            input_buffer=BufferConfig(
                power=0.5,                      # mW fallback；建议由 CACTI 回填
                cycles=1,
                area=0.0,
            ),
            output_buffer=BufferConfig(
                power=0.5,                      # mW fallback；建议由 CACTI 回填
                cycles=1,
                area=0.0,
            ),
            dist2_merge_adder=Dist2MergeAdderConfig(),  # baseline 可不用，占位
        ),

        rram_lut_macro=RRAMLutMacroConfig(
            rram_lut=RRAMLutConfig(
                power=5.0,                      # mW，标称；可由 LUT 工具标定覆盖
                area=0.0,
                cycles=0,
            ),
            rram_lut_speculative=None,          # baseline: no speculative LUT
        ),

        rram_cim_macro=RRAMCimMacroConfig(
            power=0.0,                          # 由 NeuroSim/MNSIM 动态计算，不填固定
            area=0.0,
            cycles=0,
            enabled=False,
        ),

        baseline_dist2_engine=BaselineDist2EngineConfig(),  # 94 mW, 89884 um^2
        sorting_circuit=SortingCircuitConfig(
            min_distance_table=MinDistanceTableConfig(
                power=30.0,                     # mW
                cycles=0,
                area=26610.0,                   # um^2
            ),
            bitonic_sort=BitonicSortConfig(),   # 1570/1540 mW, 2157561 um^2
            coarse_filter=None,
            fine_filter=None,
        ),
        systolic_array=SystolicArrayConfig(),   # base_v1: 特征计算 N×N 脉动阵列
        im2col=Im2ColConfig(),                 # base_v1: Conv 的 im2col 阶段

        logic_circuit=LogicCircuitConfig(
            batchnorm1d=BatchNorm1dConfig(power=3.1718, cycles=0, area=3228.0),
            layernorm=LayerNormConfig(),       # 56 mW, 139582 um^2
            relu=ReLUConfig(),
            spool=SPoolConfig(),
            adaptive_max_pool=AdaptiveMaxPoolConfig(),
            quant_pipeline_8bit=QuantPipeline8bitConfig(),
            quant_pipeline_12bit=QuantPipeline12bitConfig(),
            dequant_pipeline=DequantPipelineConfig(),
            grf_base_engine=GRFBaseEngineConfig(),  # baseline: KAT 因果多项式（数字）
            resampler=None,                      # baseline: no resampler
        ),

        inter_layer_buffer=InterLayerBufferConfig(
            size_bytes=65536,                   # 64 KB
            bandwidth_bytes_per_cycle=256,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            dram_bandwidth_bytes_per_cycle=32,
            dram_read_power_mw=50.0,
            dram_write_power_mw=50.0,
        ),
        dram=DRAMConfig(
            bandwidth_bytes_per_cycle=32,
            read_power_mw=50.0,
            write_power_mw=50.0,
        ),
        sram_ii=SRAM_IIConfig(
            active_rows=128,
            bits_per_axis=12,
            bit_slice=2,
            power_mw=0.0,
        ),
        distance_in_phase2=True,
        bn_fuse_with_conv_linear=True,
    ))


def _build_base_v2() -> HardwareConfig:
    """
    Build **base_v2** accelerator：与 baseline（base_v1）同构，但排序电路为 MergeSort，
    配合 dist2_engine 64 点带宽做流式 FPS/KNN（FPS 每批 64 局部 max 22cy/310mW，KNN 每批 top32 22cy/312mW）。
    """
    cfg = _build_baseline()
    cfg.accelerator_name = "base_v2"
    cfg.sorting_circuit = SortingCircuitConfig(
        min_distance_table=cfg.sorting_circuit.min_distance_table,
        bitonic_sort=None,
        merge_sort=MergeSortConfig(  # 180441 um^2, FPS max 22cy/310mW, KNN top32 22cy/312mW
            reg_file=MergeSortRegFileConfig(),  # FPS 1 写/迭代 1cy，KNN 32 写/query 32cy，20mW
        ),
        coarse_filter=None,
        fine_filter=None,
    )
    return cfg


def _build_acc() -> HardwareConfig:
    """
    Build **acc** (optimised) accelerator configuration.

    Differences from baseline:
      - RRAM-LUT Macro: two LUTs (GRF + speculative sampling)
      - Sorting Circuit: two-stage reconfigurable comparator
                         (replaces bitonic sort networks)
      - Logic Circuit:   adds resampler unit

    Modules:
      - SRAM-CIM Macro:  same as baseline
      - RRAM-LUT Macro:  rram_lut (GRF) + rram_lut_speculative
      - RRAM-CIM Macro:  placeholder (TODO)
      - Sorting Circuit:  two_stage_reconfigurable_comparator
      - Logic Circuit:    batchnorm1d, layernorm, relu, spool,
                          quant_pipeline_8/12bit, dequant_pipeline, resampler
    """
    return _apply_default_dramsim3_paths(HardwareConfig(
        accelerator_type=AcceleratorType.ACC,
        accelerator_name="acc",
        clock_period=1.0,                       # 1 ns = 1 GHz

        sram_cim_macro=SRAMCimMacroConfig(
            array=SRAMCimArrayConfig(
                rows=1024,
                cols=36,
                lcu_num=128,
                lcu_rows=8,
                quantization_bits=12,
                bit_slice_bits=2,
                cim_parallelism=128,
                write_power_per_row_mw=2.0,
                write_cycles_per_row=1,
                read_cycles_per_row=1,
                cim_power_per_batch_mw=6.0,
            ),
            shift_add=ShiftAddConfig(
                power=1.3,                      # mW
                cycles=1,                       # 1 cycle @ 1 GHz
                area=0.0,
            ),
            input_buffer=BufferConfig(
                power=0.5,                      # mW fallback；建议由 CACTI 回填
                cycles=1,
                area=0.0,
            ),
            output_buffer=BufferConfig(
                power=0.5,                      # mW fallback；建议由 CACTI 回填
                cycles=1,
                area=0.0,
            ),
            dist2_merge_adder=Dist2MergeAdderConfig(),  # acc: 交叉项+自平方项累加，功耗可后续标定
        ),

        rram_lut_macro=RRAMLutMacroConfig(
            rram_lut=RRAMLutConfig(             # for GRF
                power=5.0,                       # mW，标称；可由 LUT 工具标定
                area=0.0,
                cycles=0,
            ),
            rram_lut_speculative=RRAMLutConfig(  # for speculative sampling
                power=5.0,                       # mW，标称
                area=0.0,
                cycles=0,
            ),
        ),

        rram_cim_macro=RRAMCimMacroConfig(
            power=0.0,                          # 由 NeuroSim/MNSIM 动态计算，不填固定
            area=0.0,
            cycles=0,
            enabled=False,
        ),

        baseline_dist2_engine=None,             # acc: 距离由 SRAM-CIM 做，不用脉动
        sorting_circuit=SortingCircuitConfig(
            min_distance_table=MinDistanceTableConfig(
                power=30.0,                     # mW
                cycles=0,
                area=26610.0,                   # um^2
            ),
            bitonic_sort=None,                  # acc: not used
            coarse_filter=CoarseFilterConfig(), # 38 mW, 644686 um^2
            fine_filter=FineFilterConfig(
                power=38.0,                     # mW
                area=53261.0,                   # um^2
                cycles=0,
            ),
        ),

        logic_circuit=LogicCircuitConfig(
            batchnorm1d=BatchNorm1dConfig(power=3.1718, cycles=0, area=3228.0),
            layernorm=LayerNormConfig(),        # 56 mW, 139582 um^2
            relu=ReLUConfig(),
            spool=SPoolConfig(),
            adaptive_max_pool=AdaptiveMaxPoolConfig(),
            quant_pipeline_8bit=QuantPipeline8bitConfig(),
            quant_pipeline_12bit=QuantPipeline12bitConfig(),
            dequant_pipeline=DequantPipelineConfig(),
            grf_base_engine=None,                # acc: KAT 用 rram_lut_macro.rram_lut
            resampler=ResamplerConfig(),         # acc: FPS radii 上下采样
        ),
        inter_layer_buffer=InterLayerBufferConfig(
            size_bytes=65536,
            bandwidth_bytes_per_cycle=256,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            dram_bandwidth_bytes_per_cycle=32,
            dram_read_power_mw=50.0,
            dram_write_power_mw=50.0,
        ),
        dram=DRAMConfig(bandwidth_bytes_per_cycle=32, read_power_mw=50.0, write_power_mw=50.0),
        sram_ii=SRAM_IIConfig(active_rows=128, bits_per_axis=12, bit_slice=2, power_mw=0.0),
        distance_in_phase2=True,
        bn_fuse_with_conv_linear=True,
        knn_engine_mode="fuse",    # acc 使用 coarse+fine 专用路径（KNN MIN）
        fps_engine_mode="fuse",    # acc 使用 coarse+fine 专用路径（FPS MAX）
    ))


# =============================================================================
# Accelerator Registry
# =============================================================================

_ACCELERATOR_REGISTRY: Dict[str, Callable[[], HardwareConfig]] = {
    'baseline': _build_baseline,   # base_v1: bitonic sort
    'base_v2': _build_base_v2,     # MergeSort 流式 FPS/KNN，配合 dist2_engine 64 带宽
    'acc': _build_acc,
}


def get_accelerator_config(name: str) -> HardwareConfig:
    """
    Get a predefined accelerator configuration by name.

    Available:
      - 'baseline': base_v1，bitonic sort，single rram-lut，grf_base_engine，no resampler
      - 'base_v2':  MergeSort 流式 FPS/KNN，配合 dist2_engine 64 带宽（FPS 局部 max 22cy，KNN top32 22cy）
      - 'acc':      two-stage comparator，dual rram-lut，resampler

    Args:
        name: Accelerator name ('baseline', 'base_v2', or 'acc').

    Returns:
        HardwareConfig instance.

    Raises:
        ValueError: If name is not registered.
    """
    if name not in _ACCELERATOR_REGISTRY:
        available = ', '.join(sorted(_ACCELERATOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown accelerator config '{name}'. Available: {available}"
        )
    return _ACCELERATOR_REGISTRY[name]()


def list_accelerator_configs() -> List[str]:
    """List all registered accelerator names."""
    return sorted(_ACCELERATOR_REGISTRY.keys())


def set_digital_clock_freq_hz(hw_config: HardwareConfig, freq_hz: Optional[float] = None) -> None:
    """
    设置数字逻辑时钟频率（SPool/LayerNorm/ReLU/FPS/KNN 等），不改变 NeuroSim(Conv/Linear) 时钟。

    Args:
        hw_config: 硬件配置
        freq_hz: 频率 (Hz)；None 或 ≤0 时设为默认 1e9 (1 GHz)，即 clock_period=1.0 ns
    """
    if freq_hz is None or freq_hz <= 0:
        hw_config.clock_period = 1.0  # 默认 1 GHz
        return
    hw_config.clock_period = 1e9 / float(freq_hz)


def set_logic_parallelism(
    hw_config: HardwareConfig,
    layernorm: Optional[int] = None,
    relu: Optional[int] = None,
    spool: Optional[int] = None,
    quant_dequant: Optional[int] = None,
    quant_pipeline_8bit: Optional[int] = None,
    quant_pipeline_12bit: Optional[int] = None,
    dequant_pipeline: Optional[int] = None,
) -> None:
    """
    设置 logic_circuit 各子模块的并行度；仅对传入的非 None 参数生效。
    quant_dequant 为兼容参数：若传入则同时设 quant_pipeline_8bit/12bit/dequant_pipeline 为同一值。
    """
    lc = getattr(hw_config, "logic_circuit", None)
    if lc is None:
        return
    if layernorm is not None and hasattr(lc, "layernorm") and lc.layernorm is not None:
        lc.layernorm.parallelism = max(1, int(layernorm))
    if relu is not None and hasattr(lc, "relu") and lc.relu is not None:
        lc.relu.parallelism = max(1, int(relu))
    if spool is not None and hasattr(lc, "spool") and lc.spool is not None:
        lc.spool.parallelism = max(1, int(spool))
    par_q = quant_pipeline_8bit if quant_pipeline_8bit is not None else quant_dequant
    if par_q is not None and hasattr(lc, "quant_pipeline_8bit") and lc.quant_pipeline_8bit is not None:
        lc.quant_pipeline_8bit.parallelism = max(1, int(par_q))
    par_q12 = quant_pipeline_12bit if quant_pipeline_12bit is not None else quant_dequant
    if par_q12 is not None and hasattr(lc, "quant_pipeline_12bit") and lc.quant_pipeline_12bit is not None:
        lc.quant_pipeline_12bit.parallelism = max(1, int(par_q12))
    par_d = dequant_pipeline if dequant_pipeline is not None else quant_dequant
    if par_d is not None and hasattr(lc, "dequant_pipeline") and lc.dequant_pipeline is not None:
        lc.dequant_pipeline.parallelism = max(1, int(par_d))
