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
    不传时 NeuroSim 使用自身关键路径频率。Conv/Linear 事件单独带 cim_clock_period_ns，不改写 clock_period。

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
    power: float = 1.3                          # mW
    area: float = 2179.0                        # um^2
    cycles: int = 1                             # execution cycles

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
    power: float = 30.0                         # mW (acc default; baseline overrides separately)
    cycles: int = 1                             # execution cycles
    area: float = 26610.0                       # um^2


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
    post_mac_power: float = 0.0                 # TODO: mW, LUT后乘加模式占位
    post_mac_area: float = 0.0                  # TODO: mm²
    post_mac_cycles: int = 0                    # TODO: execution cycles
    default_mode: str = "rram_pwl"              # acc 默认 KAT 模式
    pwl_power: float = 0.0759164                # mW
    pwl_area: float = 235.199                   # um^2
    pwl_parallelism: int = 1
    pwl_overhead_cycles: int = 1
    rram_word_width_bits: int = 128
    rram_read_parallelism: int = 1
    rram_capacity_kb: int = 4
    rram_optimization_target: str = "ReadEDP"
    rram_read_latency_ns: Optional[float] = None
    rram_read_energy_pj: Optional[float] = None


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
# 3.5 PointCIM 专用：RRAM 距离计算阵列配置
# =============================================================================

@dataclass
class RRAMDistArrayConfig:
    """
    PointCIM 模式：RRAM VMM 实现点云距离计算。

    计算流程：
      ||qi - pj||² = qi² + pj² - 2(qi · pj)
      - pj² (k²)：self-mode VMM，one-time per SA layer，结果缓存到 SRAM
      - -2qi·pj：cross-mode VMM，per query
      - qi² (q²)：数字逻辑，per query，3 次 multiply-add
      以上均由 MICSim（内部调用 NeuroSim .so）建模，与
      hw_sim/cycle_simulator/destiny_3d_cache/matrix_power_estimator.py --distance_backend micsim 结果保持一致。

    MICSim 调用口径：
      self-mode (k²):   _run_micsim_single_fc(cin=M, cout=1, ...)  × self_rounds
      cross-mode (-2qk): _run_micsim_single_fc(cin=M, cout=N, ...)
    """
    # RRAM array 物理参数
    array_rows: int = 128            # RRAM array 行数
    array_cols: int = 128            # RRAM array 列数
    coord_dims: int = 3              # 坐标维度 M（x/y/z=3）
    coord_bits: int = 12             # 坐标量化位宽（权重位宽）
    input_bits: int = 12             # 输入向量位宽（查询坐标位宽）
    mapping_mode: str = "diagonal"   # 映射模式：diagonal（对角 block）/ shared_rows
    word_width: int = 128            # DESTINY word width (bit)，用于写入操作次数估算

    # RRAM 写入原语（DESTINY/静态参数，NeuroSim 不覆盖编程路径）
    write_energy_pj_per_op: float = 0.0    # pJ / write op（一次行写入）
    write_latency_ns_per_op: float = 0.0   # ns / write op
    power_write_mw: float = 0.0    # RRAM 编程期间功耗 (mW)
    area_mm2: float = 0.0          # 总阵列面积 (mm²)，待 VLSI 校准

    # 后处理数字逻辑开销（q² = x²+y²+z² 与 d² 累加）
    self_term_power_mw: float = 0.0   # q² 数字计算功耗 (mW)
    self_term_cycles: int = 3         # q² 计算周期数（每 query，默认 3 次 multiply-add）
    sum_power_mw: float = 0.0         # d² 最终加法功耗 (mW)
    sum_cycles: int = 1               # d² 加法周期数（每 query）

    # ── neurosim_dist pybind .so（MICSim 不可用时的备用路径）──
    # 若为空字符串，则自动尝试 pointkan_sim 包目录下的 neurosim_dist*.so
    neurosim_dist_so_dir: str = ""

    # ── MICSim 参数（与 matrix_power_estimator.py --distance_backend micsim 口径一致）──
    micsim_root: str = "/workspace/MICSim_V1.0"
    micsim_clk_freq_hz: float = 1e9          # 数字外围时钟（Hz）
    micsim_adc_clk_freq_hz: float = 2e8      # SAR ADC 专用时钟（Hz）
    micsim_sar_column_power_uw: float = 128.0  # 单列 SAR ADC 功耗（μW）
    micsim_level_output: int = 64            # ADC 量化级数（64 = 6-bit）
    micsim_static_mem_type: str = "rram"     # 静态阵列器件类型：rram / fefet
    micsim_static_ron_ohm: float = 240000.0  # R_on (Ω)
    micsim_static_roff_ohm: float = 24000000.0  # R_off (Ω)
    micsim_static_cell_bit: int = 1          # 每 cell 位数
    micsim_avg_activity: float = 0.5         # average_activityRowRead
    micsim_avg_conductance: float = 1e-6     # average_Condutance
    micsim_num_col_muxed: int = 0            # 列复用比，0=自动跟随 coord_bits
    micsim_saradc: int = 1                   # SARADC 开关
    micsim_current_mode: int = 1             # 电流模式
    micsim_adc_mode: int = 0                 # ADCmode


# =============================================================================
# 4. 排序电路
# =============================================================================

@dataclass
class BaselineDist2EngineConfig:
    """
    Baseline 距离计算引擎（1×64 FP32 PE），与 bitonic / merge_sort 配合完成 FPS/KNN。

    每个 PE 完成一次距离计算需要 per_distance_cycles 周期，64 个 PE 形成波前，
    单批 64 点周期 = (pe_count - 1) + per_distance_cycles。
    当前默认：64 PE、7 cycle/距离，因此单批固定 70 cycle。
    功耗 95 mW，面积 242750 um^2。仅 baseline 使用；acc 用 SRAM-CIM 阵列。
    """
    power: float = 95.0                          # mW
    area: float = 242750.0                       # um^2
    cycles: int = 0                              # 0 表示按波前公式计算
    pe_count: int = 64
    per_distance_cycles: int = 7
    c_dim: int = 64                              # 兼容旧逻辑：等于 pe_count


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
    sorter_depth: int = 1024                     # 单 pass 最大排序元素数；超出则多 pass + merge


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
    sorter_depth: int = 64                       # 单 pass 最大排序元素数（base_v2 默认 64）
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
    power: float = 38.0                          # mW
    area: float = 53261.0                        # um^2
    cycles: int = 2                              # execution cycles


@dataclass
class ThresholdGenConfig:
    """
    KNN dynamic-threshold generator (acc).

    用于在 blockwise 第一轮粗筛后，根据累计候选数更新下一轮粗筛阈值。
    当前作为独立控制模块建模；若无单独标定，默认复用 1 个 FP32 mul + 1 个 FP32 add 的等效 PPA。
    """
    power: float = 3.66                          # 2.5 (mul) + 1.16 (add) mW
    area: float = 3297.0                         # 2024 (mul) + 1273 (add) um^2
    cycles: int = 1                              # execution cycles


@dataclass
class BallQueryConfig:
    """
    BallQuery 硬件单元（距离过滤 + 半径筛选）。

    - baseline 模式：依托 bitonic 架构实现，每周期处理 1024 个距离（cycles_per_1024=1）。
    - acc 模式：复用 coarse/fine filter 路径（占位）。
    功耗与面积待 VLSI 校准，当前均为占位 0.0。
    """
    power_baseline_mw: float = 0.0   # mW（占位，待 VLSI 校准）
    area_baseline: float = 0.0       # um^2（占位，待 VLSI 校准）
    cycles_per_1024: int = 1         # baseline：每 1024 个距离 1 周期
    power_acc_mw: float = 0.0        # mW（占位，待 VLSI 校准）
    area_acc: float = 0.0            # um^2（占位，待 VLSI 校准）


# ---------------------------------------------------------------------------
# BQ 叶+merge 引擎配置（对应 docs/ball_query_engind.md）
# ---------------------------------------------------------------------------

@dataclass
class BQLeafEngineConfig:
    """
    BQ 叶引擎（Producer）参数。

    每次消费 block_n 条距离，在块内完成筛选与排序，输出 output_k 条已排序槽。
    对应 RTL：rq_*_block64_*_wrap / ball_query_128_tm_top64 等。
    数值与 bq_engine_catalog.py 一致，真值以 DC + PTPX 为准。
    """
    key: str = "bq64_top32"
    block_n: int = 64
    output_k: int = 32
    latency_cycles: int = 22
    power_mw: float = 133.0
    area_um2: float = 0.0


@dataclass
class BQMergeEngineConfig:
    """
    BQ 层间 merge 引擎（Consumer + Producer）。

    二叉树节点：消费两路 K 槽已排序结果，输出一路 K 槽全局序（或 max/top-3）。
    latency_cycles 为单次 merge 操作延迟（所有 merge 类型均为 2 cyc）。
    """
    key: str = "merge32"
    k: int = 32
    latency_cycles: int = 2
    power_mw: float = 87.0
    area_um2: float = 0.0


@dataclass
class BQEngineConfig:
    """
    BallQuery 引擎完整配置：叶引擎 + 层间 merge 引擎 + 调度策略。

    wall-clock 公式（N_b = ceil(N / block_n), d = ceil(log₂(N_b))）:
      TDM + tree_parallel: N_b × L_leaf + d × L_merge
      TDM + chain_serial:  N_b × L_leaf + (N_b - 1) × L_merge
      parallel + tree:     L_leaf + d × L_merge
      parallel + chain:    L_leaf + (N_b - 1) × L_merge
    """
    leaf: BQLeafEngineConfig = field(default_factory=BQLeafEngineConfig)
    merge: BQMergeEngineConfig = field(default_factory=BQMergeEngineConfig)
    leaf_schedule: str = "tdm"             # tdm | parallel
    merge_schedule: str = "tree_parallel"  # tree_parallel | chain_serial


_BQ_ENGINE_CATALOG = {
    "bq": {
        32:  ("bq64_top32",  64, 32, 22, 133.0, "merge32", 32, 87.0),
        64:  ("bq128_top64", 128, 64, 24, 370.0, "merge64", 64, 175.0),
    },
    "fps": {
        1: ("bq64_max1", 64, 1, 22, 133.0, "merge1", 1, 1.2388),
    },
    "knn": {
        3:  ("bq64_knn3",  64,  3, 22, 133.0, "merge3",   3,  8.0),
        24: ("bq64_knn24", 64, 24, 22, 133.0, "merge24", 24, 72.0),
        32: ("bq64_knn32", 64, 32, 22, 133.0, "merge32", 32, 87.0),
    },
}


def select_bq_engine_pair(
    k: int,
    mode: str = "bq",
) -> BQEngineConfig:
    """
    根据 k 值和模式从 catalog 选择匹配的叶+merge 引擎对。

    Args:
        k: 目标 top-k 值（BQ 的 nsample，FPS 为 1，KNN 为 3）。
        mode: "bq" | "fps" | "knn"

    Returns:
        BQEngineConfig 配置实例。

    Raises:
        ValueError: 当 catalog 中无匹配项时。
    """
    mode = str(mode).strip().lower()
    catalog = _BQ_ENGINE_CATALOG.get(mode, {})
    if k in catalog:
        entry = catalog[k]
    elif catalog:
        nearest_k = min(catalog.keys(), key=lambda ck: (abs(ck - k), ck))
        entry = catalog[nearest_k]
    else:
        raise ValueError(
            f"No BQ engine catalog for mode={mode!r}. "
            f"Available modes: {list(_BQ_ENGINE_CATALOG.keys())}"
        )
    lk, lb, lo, ll, lp, mk, mki, mp = entry
    return BQEngineConfig(
        leaf=BQLeafEngineConfig(key=lk, block_n=lb, output_k=lo,
                                latency_cycles=ll, power_mw=lp),
        merge=BQMergeEngineConfig(key=mk, k=mki, latency_cycles=2, power_mw=mp),
    )


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
    threshold_gen: Optional[ThresholdGenConfig] = None
    ball_query: Optional[BallQueryConfig] = None
    bq_engine: Optional[BQEngineConfig] = None     # BQ 叶+merge 引擎（PointCIM/ACC）


# =============================================================================
# 4.1 Baseline 特征计算：im2col + N×N 脉动阵列（仅 base_v1 / base_v2）
# =============================================================================

@dataclass
class Im2ColConfig:
    """
    Conv 的 im2col 前处理引擎（baseline 仅 base_v1/base_v2 使用）。
    当前按显式数据流建模：地址生成 -> 源数据读入 -> pack/reorder -> patch 写入 im2col buffer ->
    systolic 读取 im2col buffer 的矩阵。
    """
    power: float = 15.0                          # 兼容旧字段：默认 pack/reorder 功耗 (mW)
    area: float = 0.0                            # um^2，可选
    cycles_per_output: int = 1                   # 兼容旧字段：每输出点等效周期（旧模型）

    mode: str = "explicit"
    addr_gen_per_cycle: int = 16                 # 每周期可生成的 patch element 地址数
    pack_elems_per_cycle: int = 16               # 每周期可重排的 patch element 数
    overlap_addr_and_read: bool = True           # 地址生成与源读取是否重叠
    overlap_pack_with_write: bool = False        # pack/reorder 与 patch 写 buffer 是否重叠

    addr_gen_power_mw: float = 1.5               # 地址生成控制功耗
    input_read_ctrl_power_mw: float = 2.0        # 输入读取/调度控制功耗（不含存储访问能耗）
    pack_reorder_power_mw: float = 15.0          # pack/reorder 核心功耗
    col_write_ctrl_power_mw: float = 2.0         # patch 写入 col buffer 的控制功耗
    col_read_ctrl_power_mw: float = 2.0          # col buffer 供给 systolic 的读取控制功耗

    input_bus_bits: int = 256                    # 输入侧片上总线位宽
    col_bus_bits: int = 256                      # im2col buffer 总线位宽
    input_from_dram: bool = False                # 输入是否需要先从 DRAM refill 到片上输入 buffer
    dram_burst_bytes: int = 64                   # DRAM burst 大小（用于等待周期/字节估计）


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
    power: float = 3.1718                          # mW @ parallelism=1
    cycles: int = 0                                # shape-driven, 运行时按公式估算

    area: float = 3228.0                           # um^2

    parallelism: int = 1                           # 设计空间探索参数


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
    power: float = 62.26                         # mW @ parallelism=1

    area: float = 54926.0                        # um^2 @ parallelism=1

    cycles: int = 0                              # shape-driven, 0 表示运行时按公式估算

    parallelism: int = 1                         # 设计空间探索参数


@dataclass
class AdaptiveMaxPoolConfig:
    """AdaptiveMaxPool1d / MaxPool 硬件单元（与 SPool 独立）。"""
    power: float = 13.2                          # mW @ parallelism=1
    area: float = 18406.0                        # um^2
    cycles: int = 0                              # shape-driven, 运行时按 B×C×L/par 估算
    parallelism: int = 1                         # 设计空间探索参数


@dataclass
class SAMaxPoolConfig:
    """
    PointNet++ SA MaxPool 专用硬件单元：对 [B, C, nsample, nq] 沿 nsample 维度取 max。

    规格（已验证）：
      - 功耗 8.2 mW，面积 11509 um^2
      - 每 parallelism=32 个邻居消耗 1 个周期
      - 完整层周期 = B × C × nq × ceil(nsample / parallelism)
    """
    power: float = 8.2                           # mW
    area: float = 11509.0                        # um^2
    cycles: int = 0                              # shape-driven
    parallelism: int = 32                        # 每周期处理的邻居数（K维并行度）


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
class InterpN3EngineConfig:
    """
    InterpN3Engine：FP decoder 中 3-NN 插值的专用流水引擎。

    对每个输出点/query：
    - 距离倒数 + 归一化权重生成固定 22 cycles
    - 加权插值阶段按 ceil(4 * D2 / parallelism) cycles 建模

    其中 parallelism 表示 D2 维度并行度，默认 16。
    """
    power: float = 68.0                         # mW，整条引擎忙时总功耗
    area: float = 240294.0                      # um^2
    cycles_per_query_weight: int = 22           # dist_recip / norm / weight 固定周期
    feature_parallelism: int = 16               # D2 维度并行度
    weighted_cycles_numerator: int = 4          # 加权插值阶段 = ceil(4 * D2 / feature_parallelism)


@dataclass
class ResamplerConfig:
    """
    Resampler（acc only）：FPS radii 推测采样的线性插值，分下采样/上采样两种模式。

    - 下采样、上采样核心插值单元功耗均为 power（3.95mW），核心面积均为 area（4719 um^2）；
      考虑并行时功耗与核心面积乘以 parallelism，即 power_total = power * P，
      area_core_total = area * P。
    - LUT 读单独建模：下采样 LUT 与上采样 LUT 读功耗均为 0.094mW、1 cycle，
      面积分别是 13485 / 85331 um^2，不乘 parallelism。
    - 核心插值流水周期仍按需求 3.4：T=N+2 或 T=N/P+2。
    """
    power: float = 3.95                         # mW @ 每路；总功耗 = power * parallelism
    cycles: int = 0                             # 0 表示 T=N+2 或 T=N/P+2
    area: float = 4719.0                        # um^2 @ 每路；核心总面积 = area * parallelism
    parallelism: int = 1                        # 设计空间探索参数
    lut_power_downsample: float = 0.094         # mW，下采样 LUT 读功耗
    lut_cycles_downsample: int = 1              # 下采样 LUT 读周期
    lut_area_downsample: float = 13485.0        # um^2，下采样 LUT，不乘 parallelism
    lut_power_upsample: float = 0.094           # mW，上采样 LUT 读功耗
    lut_cycles_upsample: int = 1                # 上采样 LUT 读周期
    lut_area_upsample: float = 85331.0          # um^2，上采样 LUT，不乘 parallelism


@dataclass
class GRFBaseEngineConfig:
    """
    GRFBaseEngine：Efficient_KAN 的 KAT 因果多项式计算（数字实现）。

    - baseline 与 PointCIM 使用；实现因果有理多项式，与 RRAM-LUT 无关。
    - acc 纯模式下 KAT 由 RRAM-LUT Macro 的 rram_lut 实现，不用本配置。
    规格（数字 GRF，已验证）：约 19 cycle/数据，23.7833 mW，47877 um^2，并行度 P。
    """
    power: float = 23.7833                         # mW @ parallelism=1（已验证）
    area: float = 47877.0                          # um^2（已验证）

    cycles: int = 0                                # 每数据周期约 19；0 表示按公式估算

    parallelism: int = 1                           # 可绑定 KAT_Group 分组并行


@dataclass
class LogicCircuitConfig:
    """
    Logic circuit module.

    Shared ops: batchnorm1d, layernorm, relu, spool, adaptive_max_pool, sa_maxpool,
               quant_pipeline_8bit, quant_pipeline_12bit, dequant_pipeline（三条量化/反量化流水线），
               fp32_add, fp32_mul, fp32_div（FP32 加减乘除统一配置），
               interp_n3_engine（FP decoder 的 3-NN 插值专用引擎）。
    baseline / PointCIM: grf_base_engine（Efficient_KAN 的 KAT 因果多项式，数字 GRF）。
    acc (纯模式):        KAT 由 rram_lut_macro.rram_lut 实现；另有 resampler（FPS radii 上下采样）。

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
    interp_n3_engine: InterpN3EngineConfig = field(default_factory=InterpN3EngineConfig)
    sa_maxpool: SAMaxPoolConfig = field(default_factory=SAMaxPoolConfig)
    # KAT 因果多项式（数字 GRF）：baseline 与 PointCIM 使用；
    # acc 纯模式 (grf_base_engine=None) 下 KAT 走 rram_lut_macro.rram_lut。
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
    严格模型下，周期与能耗均来自 DRAMSim3 trace 回放。
    bandwidth_bytes_per_cycle 仅保留给旧接口/日志兼容，不再作为严格 DRAM latency 的主来源。
    功耗建议由 DRAMSim3 查表得到（见 hw_sim/cycle_simulator/memory_model）；未回填时使用固定 read_power_mw/write_power_mw。
    SCALE-Sim 脉动：可选两种方式——
    (1) 按访问次数×单次能耗：read_energy_nj_per_access / write_energy_nj_per_access（DRAMSim3 回填）；
    (2) 由 SCALE-Sim 生成 trace，转为 DRAMSim3 格式后跑仿真得能耗（见 demo_with_accelergy.py），需设置 dramsim3_bin 与 dramsim3_config_ini。
    """
    bandwidth_bytes_per_cycle: int = 32           # legacy hint for compatibility/logging; strict latency uses DRAMSim3
    read_power_mw: float = 50.0                  # 固定 fallback；可由 DRAMSim3 回填
    write_power_mw: float = 50.0                 # 固定 fallback；可由 DRAMSim3 回填
    read_energy_nj_per_access: Optional[float] = None   # DRAMSim3 回填：单次读能耗 (nJ)，SCALE-Sim 按访问计能耗
    write_energy_nj_per_access: Optional[float] = None   # DRAMSim3 回填：单次写能耗 (nJ)
    block_size_bytes: int = 64                          # 访问粒度，用于 num_accesses = ceil(bytes/block_size_bytes)
    issue_gap: int = 1                                   # 流水线化：连续 DRAM 请求之间最小间隔（周期），与 DRAMSim3 trace 的 issue_gap 对齐
    # Row buffer 参数（用于精细跨步访存建模，对应 DDR4 .ini timing section）
    row_size_bytes: int = 2048                           # DRAM page size = columns * device_width / 8
    tRCD_cycles: int = 17                                # Row activate latency（row buffer miss 时需要）
    tRP_cycles: int = 17                                 # Precharge latency（切换 row 时需要）
    tCAS_cycles: int = 17                                # CAS latency (CL)，row buffer hit/miss 均需
    burst_cycles: int = 4                                # 单次 burst 传输占用周期（BL/2 for DDR）
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
    area: float = 0.0                                 # CACTI 回填：物理面积（mm^2；报告时会统一换算为 um^2）
    block_size_bytes: int = 64                         # 访问粒度，用于 num_accesses = ceil(bytes/block_size_bytes)
    spill_to_dram: bool = True
    dram_bandwidth_bytes_per_cycle: int = 32
    dram_read_power_mw: float = 50.0
    dram_write_power_mw: float = 50.0
    pipeline_mode: str = "without_pipeline"      # feature SRAM 调度模式: without_pipeline / pipeline
    double_buffering: bool = False             # pipeline 模式下是否为层间 feature SRAM 预留双缓冲
    buffer_slots: int = 0                      # 0=自动推导；>0 时按槽数切分 size_bytes 得到每个 in-flight tensor 的可用容量


# =============================================================================
# Feature Computing Engine（FusionArch-inspired multi-slice/core）
# =============================================================================

@dataclass
class FeatureComputingConfig:
    """
    Multi-slice/core feature computing engine config (inspired by FusionArch).

    When num_slices > 1, KNN→Conv pipeline distributes query groups across
    parallel slices; each slice processes its assigned queries sequentially
    using cores_per_slice cores.  coalescing_factor models LHT-based DRAM
    traffic reduction from shared neighbor features within a slice.
    """
    num_slices: int = 1                    # Parallel slices (1 = single-block, multi-slice disabled)
    cores_per_slice: int = 16              # Cores per slice; each core handles one centroid's K neighbors
    muls_per_core: int = 1024              # Multipliers (MAC units) per core
    coalescing_factor: float = 0.0         # DRAM read savings from LHT coalescing (0 = none, 0.7 = 70 % saved)
    normact_cycles_per_neighbor: int = 10  # BN + ReLU latency per neighbor per MLP sub-layer


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

    # Global timing（数字逻辑时钟，Conv/Linear 若用 NeuroSim 全网络注入则另有 cim_clock_period_ns）
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
    # Dedicated SRAM buffers (CACTI-backed) for distance/FPS/KNN paths.
    xyz_sram_buffer: Optional[InterLayerBufferConfig] = None
    query_xyz_sram_buffer: Optional[InterLayerBufferConfig] = None
    dist_sram_buffer: Optional[InterLayerBufferConfig] = None
    self_distance_sram_buffer: Optional[InterLayerBufferConfig] = None
    fps_min_distance_table_sram: Optional[InterLayerBufferConfig] = None
    residual_sram_buffer: Optional[InterLayerBufferConfig] = None
    state_sram_buffer: Optional[InterLayerBufferConfig] = None
    nonlinear_workspace_sram: Optional[InterLayerBufferConfig] = None
    nonlinear_param_sram: Optional[InterLayerBufferConfig] = None
    temp_max_distance_sram: Optional[InterLayerBufferConfig] = None
    topk_sram_buffer: Optional[InterLayerBufferConfig] = None
    fps_index_sram_buffer: Optional[InterLayerBufferConfig] = None
    sort_cache_sram: Optional[InterLayerBufferConfig] = None
    im2col_buffer: Optional[InterLayerBufferConfig] = None
    # Feature Computing multi-slice/core engine (FusionArch-inspired)
    feature_computing: Optional[FeatureComputingConfig] = None
    # DRAM：点云/激活加载与写回，roofline + 功耗（可接 CACTI）
    dram: Optional[DRAMConfig] = None
    # SRAM-II：距离计算专用，128 行激活、2bit 距离
    sram_ii: Optional[SRAM_IIConfig] = None
    # PointCIM 专用：RRAM VMM 距离计算阵列（与 sram_cim_macro 距离路径互斥）
    rram_dist: Optional[RRAMDistArrayConfig] = None
    # 架构开关
    # use_rram_dist=True：距离计算由 RRAM VMM 实现（PointCIM 模式），否则使用 SRAM-CIM
    use_rram_dist: bool = False
    # distance_in_phase2=True：FPS/KNN 的距离计算(stage1-4)归入 Phase2，引擎只做 bitonic/排序
    distance_in_phase2: bool = True
    # distance_include_query_dram_read=True：Phase2 距离前，建模 query 点从 DRAM 读入
    distance_include_query_dram_read: bool = True
    # FPS 引擎模式：baseline/fuse（当前 fuse 先复用 baseline 排序路径，可继续扩展）
    fps_engine_mode: str = "baseline"
    # KNN 引擎模式：baseline/fuse（fuse 使用 coarse+fine 两级筛选事件模型）
    knn_engine_mode: str = "baseline"
    # BQ merge adder 并行度：PointCIM 模式下 merge adder 每批产出距离数，
    # 对齐 BQ 叶引擎 block_n 实现流式 producer-consumer（默认 64 对齐 bq64_top32）
    bq_merge_adder_parallelism: int = 0  # 0 = 使用 cim_parallelism fallback
    # acc KNN 动态阈值：用硬件友好的平均收敛轮数代替软件 while-loop
    # 含义：完成一次 block 粗筛并写入 topk SRAM 后，平均还需要多少轮 threshold refine。
    # 1 表示只做首轮粗筛后直接细筛；2 表示再做 1 轮 refine；依此类推。
    knn_dynamic_threshold_rounds_mean: int = 1
    # bn_fuse_with_conv_linear=True：推理时 BN 紧跟 Conv/Linear 则融合，不计 BN cycle/能耗
    bn_fuse_with_conv_linear: bool = True
    # pointcim_sa_conv_pipeline=True：PointCIM 模式下，每个 query 的 BallQuery 完成后
    # 立即将其 K 个邻居的特征送入后续 Conv2d(kernel_size=1)-BN-ReLU 流水线，
    # 与下一个 query 的 FPS+BQ 并行执行，实现 SA sample_and_group ↔ MLP 流水线。
    # 仅在 use_rram_dist=True 且 is_acc=True（PointCIM 模式）时生效。
    pointcim_sa_conv_pipeline: bool = False
    # W8A8：Conv/Linear 前插入量化(FP32→INT8)、后插入反量化(INT8→FP32)，供 BN/ReLU 等浮点非线性使用
    enable_w8a8_quant_dequant: bool = False
    # CIM 后端：neurosim（DNN_NeuroSim）、fallback（保留枚举；PE 级仍走 NeuroSim 适配器内置回退）
    cim_backend: str = "neurosim"
    # NeuroSim 根目录（Inference_pytorch），用于调用 NeuroSIM/main
    neurosim_root: str = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"
    # NeuroSim 单层/全网络运行频率（Hz）；None 表示未显式指定
    neurosim_clk_freq_hz: Optional[float] = None
    # NeuroSim 时钟模式：
    # - capped（默认）：使用 max(关键路径 clkPeriod, 用户 NEUROSIM_CLK_FREQ 对应周期) 作为工作周期（保守，用户可降频）。
    # - critical：忽略用户频率上限，仅按关键路径 clkPeriod 估算（用于“按最坏关键路径频率”对比）。
    neurosim_clk_mode: str = "capped"
    # NeuroSim ADC column mux（numColMuxed）：多少列共享 1 个 ADC / SenseAmp。
    # None/<=0：使用 NeuroSim 默认（通常为 8），保持旧行为；
    # 典型对齐策略：numColMuxed = numColPerSynapse = ceil(wl_weight / cellBit)，例如 W4A4 时取 4。
    neurosim_num_col_muxed: Optional[int] = None
    # NeuroSim iso-hardware overrides（Conventional Mapping / CM）
    neurosim_force_tile_size_cm: Optional[int] = None
    neurosim_force_pe_size_cm: Optional[int] = None
    neurosim_force_num_tiles_cm: Optional[int] = None
    # NeuroSim 激活相关（仅 subprocess NeuroSIM/main argv 生效；与 _run_neurosim_main 固定 15 项尾部对齐）
    # neurosim_relu_mode: 1=ReLU，0=Sigmoid，None=保持 NeuroSim 默认
    neurosim_relu_mode: Optional[int] = None
    # neurosim_chip_activation_mode: 1=chip-level activation，0=tile-level activation，None=保持默认
    neurosim_chip_activation_mode: Optional[int] = None
    # neurosim_enable_activation: 1=计入片上激活路径，0=关闭，None=保持默认
    neurosim_enable_activation: Optional[int] = None
    # True: Conv/Linear 单层 NeuroSim 优先走 pybind（失败时再回退 subprocess）
    neurosim_pybind: bool = False
    # Baseline 脉动阵列：是否用 SCALE-Sim 统计周期与 MAC/访问（能耗仍用本工程内存建模）
    use_scalesim_for_systolic: bool = False
    scalesim_root: str = ""           # 空则用环境变量 SCALE_SIM_ROOT 或默认 /workspace/SCALE-Sim
    scalesim_config_path: str = ""    # 空则用 scalesim_root/configs/demo_64x64_32kb_ws.cfg
    # base_v3 距离矩阵复用：第一层 SA/GAM 存全局距离矩阵到 DRAM，后续 SA/FP 层读子矩阵
    distance_matrix_reuse: bool = False
    # 转置存储：将距离矩阵按 dist_T[N, Q]（列主序）写入 DRAM，
    # 使 FP 层（上采样方向）读取变为连续行读，代价是写入变为跨步写、SA 层读取变为跨步读。
    # FP 大矩阵转置读是主要瓶颈，此选项以 overlapped 写开销换取关键路径读延迟缩减。
    distance_matrix_transpose_store: bool = False
    # NS 引擎仿真模式："event"=事件驱动估算（默认），"cycle_accurate"=逐周期仿真（FusionArch 风格）
    ns_simulation_mode: str = "event"
    # 兼容旧模型保留；新 InterpN3Engine 默认读取 logic_circuit.interp_n3_engine.feature_parallelism。
    interp_engine_parallelism: int = 128

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
    def is_pointcim(self) -> bool:
        """True 表示 PointCIM 模式：acc 架构 + RRAM VMM 距离计算（use_rram_dist=True）。"""
        return self.accelerator_type == AcceleratorType.ACC and self.use_rram_dist

    @property
    def is_baseline_v2(self) -> bool:
        """True 表示 baseline_v2（merge_sort 流式 FPS/KNN），False 表示 baseline_v1（bitonic）或 acc。"""
        return getattr(self, "accelerator_name", "") == "base_v2"

    @property
    def is_base_v3(self) -> bool:
        """True 表示 base_v3（距离矩阵复用，第一层存 DRAM 后续层读子矩阵）。"""
        return getattr(self, "accelerator_name", "") == "base_v3"

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
        if self.sorting_circuit.threshold_gen is not None:
            modules['threshold_gen'] = self.sorting_circuit.threshold_gen
        if self.sorting_circuit.ball_query is not None:
            modules['ball_query'] = self.sorting_circuit.ball_query
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
        modules['interp_n3_engine'] = self.logic_circuit.interp_n3_engine
        modules['sa_maxpool'] = self.logic_circuit.sa_maxpool
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

        baseline_dist2_engine=BaselineDist2EngineConfig(),  # 95 mW, 242750 um^2
        sorting_circuit=SortingCircuitConfig(
            min_distance_table=MinDistanceTableConfig(
                power=4.3,                      # mW (baseline min-distance update engine)
                cycles=0,
                area=10839.0,                   # um^2
            ),
            bitonic_sort=BitonicSortConfig(),   # 1570/1540 mW, 2157561 um^2
            coarse_filter=None,
            fine_filter=None,
            ball_query=BallQueryConfig(),       # 占位，功耗面积待 VLSI 校准
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
        xyz_sram_buffer=InterLayerBufferConfig(
            size_bytes=5 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        query_xyz_sram_buffer=InterLayerBufferConfig(
            size_bytes=1 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        dist_sram_buffer=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        self_distance_sram_buffer=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        fps_min_distance_table_sram=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        residual_sram_buffer=InterLayerBufferConfig(
            size_bytes=128 * 1024,
            bandwidth_bytes_per_cycle=256,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        state_sram_buffer=InterLayerBufferConfig(
            size_bytes=8 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        nonlinear_workspace_sram=InterLayerBufferConfig(
            size_bytes=32 * 1024,
            bandwidth_bytes_per_cycle=256,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        nonlinear_param_sram=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        temp_max_distance_sram=InterLayerBufferConfig(
            size_bytes=128 * 1024,
            bandwidth_bytes_per_cycle=32,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        topk_sram_buffer=InterLayerBufferConfig(
            size_bytes=2 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        fps_index_sram_buffer=InterLayerBufferConfig(
            size_bytes=2 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        sort_cache_sram=InterLayerBufferConfig(
            size_bytes=2 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
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
    Build base_v2 accelerator：与 baseline（base_v1）同构，但排序电路为 MergeSort，
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


def _build_base_v3() -> HardwareConfig:
    """
    Build base_v3 accelerator：与 baseline（base_v1）同硬件，但启用距离矩阵复用。

    第一层 SA/GAM KNN 计算完距离后流式写入 DRAM，后续 SA/GAM 层和所有 FP 层
    跳过距离计算，改为从 DRAM 读取已存储距离矩阵的子集。
    距离矩阵按转置布局 dist_T[N, Q] 存储，使 FP 层大矩阵读取为连续访问，
    代价是第一层流式写变为跨步写（overlapped，不影响关键路径）。
    """
    cfg = _build_baseline()
    cfg.accelerator_name = "base_v3"
    cfg.distance_matrix_reuse = True
    cfg.distance_matrix_transpose_store = True
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
                cycles=1,
                default_mode="rram_pwl",
                pwl_power=0.5655,
                pwl_area=882.0,
                pwl_parallelism=1,
                pwl_overhead_cycles=1,
            ),
            rram_lut_speculative=RRAMLutConfig(  # for speculative sampling
                power=5.0,                       # mW，标称
                area=0.0,
                cycles=1,
                default_mode="rram_pwl",
                pwl_power=0.5655,
                pwl_area=882.0,
                pwl_parallelism=1,
                pwl_overhead_cycles=1,
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
                cycles=1,
                area=26610.0,                   # um^2
            ),
            bitonic_sort=None,                  # acc: not used
            coarse_filter=CoarseFilterConfig(
                power=81.12,                    # mW, FPS/KNN coarse filter
                area=94204.0,                   # um^2
                cycles=1,
            ),
            fine_filter=FineFilterConfig(
                power=38.0,                     # mW
                area=53261.0,                   # um^2
                cycles=2,
            ),
            threshold_gen=ThresholdGenConfig(
                power=3.66,                     # mW = FP32 mul + add
                area=3297.0,                    # um^2 = 2024 + 1273
                cycles=1,
            ),
            ball_query=BallQueryConfig(),       # 占位，功耗面积待 VLSI 校准
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
        xyz_sram_buffer=InterLayerBufferConfig(
            size_bytes=5 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        query_xyz_sram_buffer=InterLayerBufferConfig(
            size_bytes=1 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        dist_sram_buffer=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        self_distance_sram_buffer=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        fps_min_distance_table_sram=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        residual_sram_buffer=InterLayerBufferConfig(
            size_bytes=128 * 1024,
            bandwidth_bytes_per_cycle=256,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        state_sram_buffer=InterLayerBufferConfig(
            size_bytes=8 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        nonlinear_workspace_sram=InterLayerBufferConfig(
            size_bytes=32 * 1024,
            bandwidth_bytes_per_cycle=256,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
        ),
        nonlinear_param_sram=InterLayerBufferConfig(
            size_bytes=4 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        temp_max_distance_sram=InterLayerBufferConfig(
            size_bytes=128 * 1024,
            bandwidth_bytes_per_cycle=32,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        topk_sram_buffer=InterLayerBufferConfig(
            size_bytes=2 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        fps_index_sram_buffer=InterLayerBufferConfig(
            size_bytes=2 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        sort_cache_sram=InterLayerBufferConfig(
            size_bytes=2 * 1024,
            bandwidth_bytes_per_cycle=64,
            read_power_mw=2.0,
            write_power_mw=2.0,
            spill_to_dram=True,
            block_size_bytes=16,
        ),
        dram=DRAMConfig(bandwidth_bytes_per_cycle=32, read_power_mw=50.0, write_power_mw=50.0),
        sram_ii=SRAM_IIConfig(active_rows=128, bits_per_axis=12, bit_slice=2, power_mw=0.0),
        distance_in_phase2=True,
        bn_fuse_with_conv_linear=True,
        knn_engine_mode="fuse",    # acc 使用 coarse+fine 专用路径（KNN MIN）
        fps_engine_mode="fuse",    # acc 使用 coarse+fine 专用路径（FPS MAX）
    ))


def _build_pointcim() -> HardwareConfig:
    """
    Build **PointCIM** accelerator configuration.

    PointCIM = acc 架构（coarse+fine filter、resampler、GRFEngine KAN）
             + RRAM VMM 距离计算（替代 SRAM-CIM 路径）。

    距离计算流程（per SA layer）：
      1. RRAM 写入（one-time）：将 N 个点的 xyz 坐标编程到 RRAM 阵列
      2. RRAM VMM（per query）：cross-mode 广播查询点，一次读出 N 个距离的 cross-term
      3. 数字后处理：q² + rram_output + k² → 最终 d²
    FPS/KNN 排序、MLP 特征提取与 acc 模式完全相同。

    RRAMDistArrayConfig 中的能耗/延迟原语默认为 0.0（占位），
    需在使用前通过以下命令从 DESTINY/MICSim 提取并填充：
      python hw_sim/cycle_simulator/destiny_3d_cache/matrix_power_estimator.py \\
             --M 3 --K 1024 --bits 12 ...
    """
    cfg = _build_acc()
    cfg.accelerator_name = "pointcim"
    # PointCIM 的 Conv/Linear 单层估算走 NeuroSim。
    cfg.cim_backend = "neurosim"
    cfg.use_rram_dist = True
    # PointCIM 的 KAN/KAT 使用 GRFEngine（数字因果有理多项式），不用 RRAM-LUT。
    cfg.logic_circuit.grf_base_engine = GRFBaseEngineConfig()
    # BQ 叶+merge 引擎（默认 bq64_top32，merge adder 并行度=64 对齐叶引擎 block_n）
    bq_cfg = select_bq_engine_pair(k=32, mode="bq")
    cfg.sorting_circuit.bq_engine = bq_cfg
    cfg.bq_merge_adder_parallelism = bq_cfg.leaf.block_n  # 64
    cfg.rram_dist = RRAMDistArrayConfig(
        # RRAM 物理参数
        array_rows=128,
        array_cols=128,
        coord_dims=3,
        coord_bits=12,
        input_bits=12,
        mapping_mode="diagonal",
        word_width=128,
        # RRAM 写入原语（占位 0.0，可由 DESTINY 填充）
        write_energy_pj_per_op=0.0,
        write_latency_ns_per_op=0.0,
        power_write_mw=0.0,
        area_mm2=0.0,
        # 数字后处理
        self_term_power_mw=0.0,
        self_term_cycles=3,
        sum_power_mw=0.0,
        sum_cycles=1,
        # MICSim 参数（与 matrix_power_estimator.py --distance_backend micsim 口径一致）
        micsim_root="/workspace/MICSim_V1.0",
        micsim_clk_freq_hz=1e9,
        micsim_adc_clk_freq_hz=2e8,
        micsim_sar_column_power_uw=128.0,
        micsim_level_output=64,
        micsim_static_mem_type="rram",
        micsim_static_ron_ohm=240000.0,
        micsim_static_roff_ohm=24000000.0,
        micsim_static_cell_bit=1,
    )
    return cfg


# =============================================================================
# Accelerator Registry
# =============================================================================

_ACCELERATOR_REGISTRY: Dict[str, Callable[[], HardwareConfig]] = {
    'baseline': _build_baseline,   # base_v1: bitonic sort
    'base_v2': _build_base_v2,     # MergeSort 流式 FPS/KNN，配合 dist2_engine 64 带宽
    'base_v3': _build_base_v3,     # base_v1 硬件 + 距离矩阵复用（第一层存 DRAM，后续层读子矩阵）
    'acc': _build_acc,
    'pointcim': _build_pointcim,   # acc 架构 + RRAM VMM 距离 + GRFEngine KAN（PointCIM 模式）
}


def get_accelerator_config(name: str) -> HardwareConfig:
    """
    Get a predefined accelerator configuration by name.

    Available:
      - 'baseline': base_v1，bitonic sort，single rram-lut，grf_base_engine，no resampler
      - 'base_v2':  MergeSort 流式 FPS/KNN，配合 dist2_engine 64 带宽（FPS 局部 max 22cy，KNN top32 22cy）
      - 'base_v3':  base_v1 硬件 + 距离矩阵复用（第一层存 DRAM，后续层读子矩阵，FP 转置跨步读）
      - 'acc':      two-stage comparator，dual rram-lut，resampler
      - 'pointcim': acc 架构 + RRAM VMM 距离 + GRFEngine KAN（SRAM-CIM 距离路径由 RRAM 替代，KAN 走数字 GRF）

    Args:
        name: Accelerator name ('baseline', 'base_v2', 'base_v3', 'acc', or 'pointcim').

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
