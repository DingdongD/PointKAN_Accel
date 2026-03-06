"""
硬件配置模块（baseline / acc 加速器）
=====================================

面向点云网络（PointKAN / PointNet++ 等）的两种 CIM 加速器扁平配置，
无 Chip→Tile→PE 层级，单芯片由五大宏模块组成：

  1. SRAM-CIM 宏：sram-cim 阵列 + shift_add + 最小距离表
  2. RRAM-LUT 宏：rram 查找表
  3. RRAM-CIM 宏：占位，与 MNSIM 集成
  4. 排序电路：baseline 为 Bitonic 排序，acc 为两阶段比较器
  5. 逻辑电路：BN/LN/ReLU/量化反量化等，acc 增加插值

使用：get_accelerator_config("baseline") / get_accelerator_config("acc")
"""

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
    MIN_DISTANCE_TABLE = auto()       # MinDistanceTableConfig
    BUFFER = auto()                   # BufferConfig

    # --- RRAM-LUT Macro ---
    RRAM_LUT = auto()                 # RramLutConfig
    RRAM_LUT_SPECULATIVE = auto()     # RramLutConfig (acc only)

    # --- RRAM-CIM Macro ---
    RRAM_CIM = auto()                 # RramCimMacroConfig

    # --- Sorting Circuit ---
    COMP_TREE_1024_MAX = auto()       # CompTreeMaxConfig (baseline)
    COMP_BITONIC_1024_MIN32 = auto()  # CompBitonicMinConfig (baseline)
    TWO_STAGE_RECONF_COMPARATOR = auto()  # TwoStageReconfComparatorConfig (acc)

    # --- Logic Circuit ---
    BATCHNORM1D = auto()              # BatchNorm1dConfig
    LAYERNORM = auto()                # LayerNormConfig
    RELU = auto()                     # ReLUConfig
    SPOOL = auto()                    # SPoolConfig
    QUANT_DEQUANT = auto()            # QuantDequantConfig
    INTERPOLATION = auto()            # InterpolationConfig (acc only)


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
class SramCimArrayConfig:
    """
    SRAM-CIM 阵列配置：128 个 LCU，每 LCU 8 行×36 列，等效 1024 行×36 列，128 路并行。
    Attributes: rows/cols/lcu_num/lcu_rows, quantization_bits, bit_slice_bits, cim_parallelism, write_power, cim_power.
    """
    rows: int = 1024
    cols: int = 36
    lcu_num: int = 128
    lcu_rows: int = 8                           # rows / lcu_num
    quantization_bits: int = 12
    bit_slice_bits: int = 2
    cim_parallelism: int = 128                  # = lcu_num

    # Power (entire macro, 128 LCUs parallel)
    write_power: float = 0.0                    # TODO: mW
    cim_power: float = 0.0                      # TODO: mW

    @property
    def bit_serial_cycles(self) -> int:
        """Number of CIM cycles per complete MAC: quantization_bits / bit_slice_bits."""
        return self.quantization_bits // self.bit_slice_bits


@dataclass
class ShiftAddConfig:
    """SRAM-CIM 阵列后的移位累加单元，用于 bit-serial 累加。"""
    power: float = 0.0
    cycles: int = 0


@dataclass
class MinDistanceTableConfig:
    """FPS/KNN 中用于记录最小距离的查找表。"""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class BufferConfig:
    """SRAM 写回与中间存储用的 buffer/寄存器。"""
    power: float = 0.0
    cycles: int = 0


@dataclass
class SramCimMacroConfig:
    """
    SRAM-CIM 宏 = array + shift_add + min_distance_table + buffer，baseline 与 acc 共用。
    """
    array: SramCimArrayConfig = field(default_factory=SramCimArrayConfig)
    shift_add: ShiftAddConfig = field(default_factory=ShiftAddConfig)
    min_distance_table: MinDistanceTableConfig = field(
        default_factory=MinDistanceTableConfig
    )
    buffer: BufferConfig = field(default_factory=BufferConfig)


# =============================================================================
# 2. RRAM-LUT 宏
# =============================================================================

@dataclass
class RramLutConfig:
    """单块 RRAM 查找表单元。"""
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class RramLutMacroConfig:
    """
    RRAM-LUT Macro.

    baseline: single rram_lut instance.
    acc:      two instances — one for GRF, one for speculative sampling.
    """
    rram_lut: RramLutConfig = field(default_factory=RramLutConfig)
    # acc-only: additional LUT for speculative sampling (None for baseline)
    rram_lut_speculative: Optional[RramLutConfig] = None


# =============================================================================
# 3. RRAM-CIM 宏（占位，待与 MNSIM 集成）
# =============================================================================

@dataclass
class RramCimMacroConfig:
    """RRAM-CIM 宏，占位，待 MNSIM 集成后填参。"""
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles
    enabled: bool = False                       # Not yet integrated


# =============================================================================
# 4. 排序电路
# =============================================================================

@dataclass
class CompTreeMaxConfig:
    """1024 路比较树取 MAX（baseline），Bitonic 取全局最大值。"""
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class CompBitonicMinConfig:
    """
    Bitonic sort network for 1024-to-32 MIN selection (baseline).

    Selects the 32 smallest elements from 1024 inputs.
    """
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class TwoStageReconfComparatorConfig:
    """
    Two-stage reconfigurable comparator (acc).

    Replaces bitonic sort networks with a coarse-fine screening approach.
    """
    power: float = 0.0                          # TODO: mW
    area: float = 0.0                           # TODO: mm²
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class SortingCircuitConfig:
    """
    Sorting circuit module.

    baseline: comp_tree_1024_max + comp_bitonic_1024_min32
    acc:      two_stage_reconfigurable_comparator
    """
    # baseline components (None for acc)
    comp_tree_1024_max: Optional[CompTreeMaxConfig] = None
    comp_bitonic_1024_min32: Optional[CompBitonicMinConfig] = None
    # acc component (None for baseline)
    two_stage_reconf_comparator: Optional[TwoStageReconfComparatorConfig] = None


# =============================================================================
# 5. Logic Circuit  —  sub-modules (ops + quant/dequant + interpolation)
# =============================================================================

@dataclass
class BatchNorm1dConfig:
    """BatchNorm1d hardware unit."""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class LayerNormConfig:
    """LayerNorm hardware unit."""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class ReLUConfig:
    """ReLU activation hardware unit."""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class SPoolConfig:
    """SPool (spatial pooling) hardware unit."""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class QuantDequantConfig:
    """Quantization / de-quantization hardware unit."""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class InterpolationConfig:
    """Interpolation hardware unit (acc only)."""
    power: float = 0.0                          # TODO: mW
    cycles: int = 0                             # TODO: execution cycles


@dataclass
class LogicCircuitConfig:
    """
    Logic circuit module.

    Contains ops sub-modules + quantize/dequantize.
    acc additionally includes interpolation.

    Shared ops: batchnorm1d, layernorm, relu, spool, quant_dequant.
    acc-only:   interpolation.
    """
    # Shared ops
    batchnorm1d: BatchNorm1dConfig = field(default_factory=BatchNorm1dConfig)
    layernorm: LayerNormConfig = field(default_factory=LayerNormConfig)
    relu: ReLUConfig = field(default_factory=ReLUConfig)
    spool: SPoolConfig = field(default_factory=SPoolConfig)
    quant_dequant: QuantDequantConfig = field(default_factory=QuantDequantConfig)
    # acc-only (None for baseline)
    interpolation: Optional[InterpolationConfig] = None


# =============================================================================
# DRAM (外部内存，Roofline + CACTI 功耗)
# =============================================================================

@dataclass
class DRAMConfig:
    """
    外部 DRAM：点云加载、查询点读取、结果写回等。
    Roofline: cycles = bytes / bandwidth_bytes_per_cycle；能耗 = cycles * clock * power_mw。
    功耗可由 CACTI 或 datasheet 获取。
    """
    bandwidth_bytes_per_cycle: int = 32
    read_power_mw: float = 50.0
    write_power_mw: float = 50.0


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

    Attributes:
        size_bytes: Buffer 容量（字节）；超过则产生 spill 或分块
        bandwidth_bytes_per_cycle: 每周期可读/写字节数（带宽）
        read_power_mw: Buffer 读功率 (mW)
        write_power_mw: Buffer 写功率 (mW)
        spill_to_dram: 是否将超出 capacity 的部分建模为 DRAM 读写
        dram_bandwidth_bytes_per_cycle: DRAM 带宽（spill 时用）
        dram_read_power_mw: DRAM 读功率 (mW)
        dram_write_power_mw: DRAM 写功率 (mW)
    """
    size_bytes: int = 65536                    # 64 KB 片上 buffer
    bandwidth_bytes_per_cycle: int = 256       # 256 B/cycle
    read_power_mw: float = 2.0
    write_power_mw: float = 2.0
    spill_to_dram: bool = True
    dram_bandwidth_bytes_per_cycle: int = 32
    dram_read_power_mw: float = 50.0
    dram_write_power_mw: float = 50.0


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
        clock_period: Clock period in nanoseconds (1 ns = 1 GHz)
        sram_cim_macro: SRAM-CIM macro configuration
        rram_lut_macro: RRAM-LUT macro configuration
        rram_cim_macro: RRAM-CIM macro configuration (placeholder)
        sorting_circuit: Sorting circuit configuration
        logic_circuit: Logic circuit configuration
        inter_layer_buffer: 层间激活 buffer 配置（容量、带宽、spill）
    """
    # Identity
    accelerator_type: AcceleratorType = AcceleratorType.BASELINE
    accelerator_name: str = "baseline"

    # Global timing
    clock_period: float = 1.0                   # ns  (1 GHz)

    # Five macro-level modules
    sram_cim_macro: SramCimMacroConfig = field(
        default_factory=SramCimMacroConfig
    )
    rram_lut_macro: RramLutMacroConfig = field(
        default_factory=RramLutMacroConfig
    )
    rram_cim_macro: RramCimMacroConfig = field(
        default_factory=RramCimMacroConfig
    )
    sorting_circuit: SortingCircuitConfig = field(
        default_factory=SortingCircuitConfig
    )
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

    # =========================================================================
    # Utility methods
    # =========================================================================

    def get_all_modules(self) -> Dict[str, Any]:
        """Return a flat dict of all circuit sub-module configs for iteration."""
        modules: Dict[str, Any] = {}

        # SRAM-CIM macro
        modules['sram_cim_array'] = self.sram_cim_macro.array
        modules['shift_add'] = self.sram_cim_macro.shift_add
        modules['min_distance_table'] = self.sram_cim_macro.min_distance_table
        modules['buffer'] = self.sram_cim_macro.buffer

        # RRAM-LUT macro
        modules['rram_lut'] = self.rram_lut_macro.rram_lut
        if self.rram_lut_macro.rram_lut_speculative is not None:
            modules['rram_lut_speculative'] = (
                self.rram_lut_macro.rram_lut_speculative
            )

        # RRAM-CIM macro
        modules['rram_cim_macro'] = self.rram_cim_macro

        # Sorting circuit
        if self.sorting_circuit.comp_tree_1024_max is not None:
            modules['comp_tree_1024_max'] = (
                self.sorting_circuit.comp_tree_1024_max
            )
        if self.sorting_circuit.comp_bitonic_1024_min32 is not None:
            modules['comp_bitonic_1024_min32'] = (
                self.sorting_circuit.comp_bitonic_1024_min32
            )
        if self.sorting_circuit.two_stage_reconf_comparator is not None:
            modules['two_stage_reconf_comparator'] = (
                self.sorting_circuit.two_stage_reconf_comparator
            )

        # Logic circuit
        modules['batchnorm1d'] = self.logic_circuit.batchnorm1d
        modules['layernorm'] = self.logic_circuit.layernorm
        modules['relu'] = self.logic_circuit.relu
        modules['spool'] = self.logic_circuit.spool
        modules['quant_dequant'] = self.logic_circuit.quant_dequant
        if self.logic_circuit.interpolation is not None:
            modules['interpolation'] = self.logic_circuit.interpolation

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

def _build_baseline() -> HardwareConfig:
    """
    Build **baseline** accelerator configuration.

    Modules:
      - SRAM-CIM Macro:  sram_cim_array + shift_add + min_distance_table
      - RRAM-LUT Macro:  single rram_lut
      - RRAM-CIM Macro:  placeholder (TODO)
      - Sorting Circuit:  comp_tree_1024_max + comp_bitonic_1024_min32
      - Logic Circuit:    batchnorm1d, layernorm, relu, spool, quant_dequant
    """
    return HardwareConfig(
        accelerator_type=AcceleratorType.BASELINE,
        accelerator_name="baseline",
        clock_period=1.0,                       # 1 ns = 1 GHz

        sram_cim_macro=SramCimMacroConfig(
            array=SramCimArrayConfig(
                rows=1024,
                cols=36,
                lcu_num=128,
                lcu_rows=8,
                quantization_bits=12,
                bit_slice_bits=2,
                cim_parallelism=128,
                write_power=0.0,                # TODO
                cim_power=0.0,                  # TODO
            ),
            shift_add=ShiftAddConfig(
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
            min_distance_table=MinDistanceTableConfig(
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
            buffer=BufferConfig(
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
        ),

        rram_lut_macro=RramLutMacroConfig(
            rram_lut=RramLutConfig(
                power=0.0,                      # TODO
                area=0.0,                       # TODO
                cycles=0,                       # TODO
            ),
            rram_lut_speculative=None,          # baseline: no speculative LUT
        ),

        rram_cim_macro=RramCimMacroConfig(
            power=0.0,                          # TODO
            area=0.0,                           # TODO
            cycles=0,                           # TODO
            enabled=False,
        ),

        sorting_circuit=SortingCircuitConfig(
            comp_tree_1024_max=CompTreeMaxConfig(
                power=0.0,                      # TODO
                area=0.0,                       # TODO
                cycles=0,                       # TODO
            ),
            comp_bitonic_1024_min32=CompBitonicMinConfig(
                power=0.0,                      # TODO
                area=0.0,                       # TODO
                cycles=0,                       # TODO
            ),
            two_stage_reconf_comparator=None,   # baseline: not used
        ),

        logic_circuit=LogicCircuitConfig(
            batchnorm1d=BatchNorm1dConfig(power=0.0, cycles=0),    # TODO
            layernorm=LayerNormConfig(power=0.0, cycles=0),        # TODO
            relu=ReLUConfig(power=0.0, cycles=0),                  # TODO
            spool=SPoolConfig(power=0.0, cycles=0),                # TODO
            quant_dequant=QuantDequantConfig(power=0.0, cycles=0), # TODO
            interpolation=None,                 # baseline: no interpolation
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
    )


def _build_acc() -> HardwareConfig:
    """
    Build **acc** (optimised) accelerator configuration.

    Differences from baseline:
      - RRAM-LUT Macro: two LUTs (GRF + speculative sampling)
      - Sorting Circuit: two-stage reconfigurable comparator
                         (replaces bitonic sort networks)
      - Logic Circuit:   adds interpolation unit

    Modules:
      - SRAM-CIM Macro:  same as baseline
      - RRAM-LUT Macro:  rram_lut (GRF) + rram_lut_speculative
      - RRAM-CIM Macro:  placeholder (TODO)
      - Sorting Circuit:  two_stage_reconfigurable_comparator
      - Logic Circuit:    batchnorm1d, layernorm, relu, spool,
                          quant_dequant, interpolation
    """
    return HardwareConfig(
        accelerator_type=AcceleratorType.ACC,
        accelerator_name="acc",
        clock_period=1.0,                       # 1 ns = 1 GHz

        sram_cim_macro=SramCimMacroConfig(
            array=SramCimArrayConfig(
                rows=1024,
                cols=36,
                lcu_num=128,
                lcu_rows=8,
                quantization_bits=12,
                bit_slice_bits=2,
                cim_parallelism=128,
                write_power=0.0,                # TODO
                cim_power=0.0,                  # TODO
            ),
            shift_add=ShiftAddConfig(
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
            min_distance_table=MinDistanceTableConfig(
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
            buffer=BufferConfig(
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
        ),

        rram_lut_macro=RramLutMacroConfig(
            rram_lut=RramLutConfig(              # for GRF
                power=0.0,                      # TODO
                area=0.0,                       # TODO
                cycles=0,                       # TODO
            ),
            rram_lut_speculative=RramLutConfig(  # for speculative sampling
                power=0.0,                      # TODO
                area=0.0,                       # TODO
                cycles=0,                       # TODO
            ),
        ),

        rram_cim_macro=RramCimMacroConfig(
            power=0.0,                          # TODO
            area=0.0,                           # TODO
            cycles=0,                           # TODO
            enabled=False,
        ),

        sorting_circuit=SortingCircuitConfig(
            comp_tree_1024_max=None,            # acc: not used
            comp_bitonic_1024_min32=None,        # acc: not used
            two_stage_reconf_comparator=TwoStageReconfComparatorConfig(
                power=0.0,                      # TODO
                area=0.0,                       # TODO
                cycles=0,                       # TODO
            ),
        ),

        logic_circuit=LogicCircuitConfig(
            batchnorm1d=BatchNorm1dConfig(power=0.0, cycles=0),    # TODO
            layernorm=LayerNormConfig(power=0.0, cycles=0),        # TODO
            relu=ReLUConfig(power=0.0, cycles=0),                  # TODO
            spool=SPoolConfig(power=0.0, cycles=0),                # TODO
            quant_dequant=QuantDequantConfig(power=0.0, cycles=0), # TODO
            interpolation=InterpolationConfig(   # acc: has interpolation
                power=0.0,                      # TODO
                cycles=0,                       # TODO
            ),
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
    )


# =============================================================================
# Accelerator Registry
# =============================================================================

_ACCELERATOR_REGISTRY: Dict[str, Callable[[], HardwareConfig]] = {
    'baseline': _build_baseline,
    'acc': _build_acc,
}


def get_accelerator_config(name: str) -> HardwareConfig:
    """
    Get a predefined accelerator configuration by name.

    Available:
      - 'baseline': bitonic sort, single rram-lut, no interpolation
      - 'acc':      two-stage comparator, dual rram-lut, interpolation

    Args:
        name: Accelerator name ('baseline' or 'acc').

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
