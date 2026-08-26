"""
KNN 算子（K 近邻）
=================

基于 SRAM-CIM 的 K 近邻：k^2、-2qk、距离累加、SRAM 写回距离、Bitonic 排序五阶段，
对每个 query 点按批算距离并写 SRAM，最后 Bitonic 排序取 k 近邻。
生成事件时 operator 统一为 "KNN"（PascalCase），供 operator_sequence.txt 使用。

Uses 5 sequential stages:
1. k^2 computation (SRAM-CIM-array + shift-and-add, bit-serial)
2. -2qk cross-term computation (SRAM-CIM-array + shift-and-add, bit-serial)
3. Distance accumulation (shift-and-add)
4. SRAM write (store computed distances)
5. Bitonic sort (1024->24, find k nearest neighbors)

Execution flow:
    For each query q = 0 .. query_points - 1:
        batches = ceil(input_points / cim_parallelism)   (constant per query)
        For each batch:
            Stage 1: k^2             (dist_compute_cycles)
            Stage 2: -2qk            (dist_compute_cycles)
            Stage 3: accumulate      (1 cycle)
            Stage 4: SRAM write      (1 cycle)
        Stage 5: bitonic sort        (fixed cycles, after all batches)

    dist_compute_cycles = points_per_axis_quant_width / bit_slice

Cycle formula:
    per_batch = 2 * dist_compute_cycles + 1 + 1
    batches = ceil(input_points / cim_parallelism)        (constant per query)
    per_query = batches * per_batch + bitonic_sort_cycles
    total = query_points * per_query
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import math

from .base_operator import (
    BaseOperator, Stage, SubStage,
    OperatorParams,
)
from ..configs.hw_config import CircuitModule
from ..simulator.event import Event
from .distance_ops import (
    gen_distance_events_one_batch,
    gen_rram_dist_write_events,
    gen_rram_dist_self_k2_events,
)
from ..integration.memory_model_adapter import (
    query_sram_cycles_energy,
    query_dram_batch_cycles_energy,
    query_dram_cycles_energy,
    query_dram_matrix_cycles_energy,
)

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


@dataclass
class KNNParams(OperatorParams):
    """
    KNN 专用参数：继承 input_points（搜索空间点数）；额外包括 query_points、k、
    sampled_points_num、cim_parallelism、bit_slice、points_per_axis_quant_width、bitonic_sort_cycles。

    distance_reuse: 跳过距离计算（Stage 1-4），从 DRAM 读取已存储的距离矩阵子集 + sort top-k。
    dist_matrix_dram_write: 距离计算完成后额外将 [query_points, input_points] 距离矩阵写入 DRAM。
    global_input_points: 全局输入点数，用于 distance_reuse 时计算 DRAM 偏移（默认与 input_points 相同）。
    is_transpose_read: FP 层读取时为 True（读矩阵列，跨步访问，row buffer miss 代价高）。
    global_dist_cols: 全局距离矩阵列数（SA 的 input_points），用于计算转置读 stride。
    """
    query_points: int = 512
    k: int = 24
    sampled_points_num: int = 512
    cim_parallelism: int = 128
    bit_slice: int = 2
    points_per_axis_quant_width: int = 28
    bitonic_sort_cycles: int = 19   # MIN 模式（KNN 1024→k）：19 周期
    distance_reuse: bool = False
    dist_matrix_dram_write: bool = False
    global_input_points: int = 0
    is_transpose_read: bool = False
    global_dist_cols: int = 0


class KNNOperator(BaseOperator):
    """
    K-Nearest Neighbors (KNN) Operator.

    SRAM-CIM based KNN with fixed parallelism and bitonic sort.

    Processing flow (per query point):

    [Stages 1-4: sequential, repeated for each batch]

    1. k^2 computation
       SRAM-CIM array read + shift-and-add, bit-serial.
       Duration = points_per_axis_quant_width / bit_slice cycles.
       Circuit modules: SRAM_CIM_ARRAY, SHIFT_ADD

    2. -2qk cross-term
       SRAM-CIM array MAC + shift-and-add, bit-serial.
       Duration = points_per_axis_quant_width / bit_slice cycles.
       Circuit modules: SRAM_CIM_ARRAY, SHIFT_ADD

    3. Distance accumulation
       Shift-and-add: distance = k^2 + (-2qk).
       Duration = 1 cycle.
       Circuit module: SHIFT_ADD

    4. SRAM write
       Write computed distances to SRAM for sorting.
       Duration = 1 cycle.
       Circuit module: BUFFER_OUTPUT

    [Stage 5: runs once after all batches complete]

    5. Bitonic sort (1024->24)
       Full bitonic sort to find k nearest neighbors (min distances).
       Duration = bitonic_sort_cycles (fixed constant).
       Circuit modules: BITONIC_SORT (output buffer: BUFFER_OUTPUT)
    """

    def __init__(
        self,
        params: Optional[KNNParams] = None,
        hw_config: Optional['HardwareConfig'] = None
    ):
        if params is None:
            params = KNNParams()
        self.knn_params: KNNParams = params
        super().__init__("KNN", params, hw_config)

    # --- Derived parameter helpers ---

    def _get_dist_compute_cycles(self) -> int:
        """Cycles for distance computation (k^2 or -2qk) per batch."""
        p = self.knn_params
        return p.points_per_axis_quant_width // p.bit_slice

    def _get_per_batch_cycles(self) -> int:
        """Total cycles per batch (stages 1-4)."""
        return 2 * self._get_dist_compute_cycles() + 1 + 1

    def _get_sort_cycles(self) -> int:
        """
        计算排序周期，考虑 sorter_depth 多 pass + merge 开销。

        当 input_points <= sorter_depth 时退化为基准 bitonic_sort_cycles；
        超出时需 ceil(input_points/sorter_depth) 趟局部排序 + log2 阶 merge。
        """
        p = self.knn_params
        hw = self.hw_config
        base_cycles = p.bitonic_sort_cycles
        if hw is None:
            return base_cycles
        sort_cfg = getattr(getattr(hw, "sorting_circuit", None), "bitonic_sort", None)
        if sort_cfg is None:
            return base_cycles
        depth = int(getattr(sort_cfg, "sorter_depth", 0) or 0)
        if depth <= 0 or depth >= p.input_points:
            return base_cycles
        num_passes = math.ceil(p.input_points / depth)
        merge_stages = max(0, int(math.ceil(math.log2(num_passes))))
        return num_passes * base_cycles + merge_stages * max(1, base_cycles // 2)

    # --- Pipeline helpers (供 dispatcher 做 FPS→KNN 流水线重叠) ---

    def get_per_query_cycles(self) -> int:
        """
        返回单个 KNN 查询的周期数（含距离计算 + 排序），
        用于 dispatcher 计算 producer-consumer 流水线调度。
        """
        p = self.knn_params
        per_batch = self._get_per_batch_cycles()
        batches = math.ceil(p.input_points / p.cim_parallelism)
        sort_cycles = self._get_sort_cycles()
        return batches * per_batch + sort_cycles

    # --- Stage definition ---

    def _define_stages(self) -> None:
        """
        Define KNN hardware processing stages.

        All 5 stages are sequential (is_pipeline=False).
        Stages 1-4 form a loop body executed per batch;
        Stage 5 runs once after all batches complete.
        """
        # Stage 1: k^2 computation
        stage1 = Stage(
            name="k_square",
            is_pipeline=False,
            description="k^2 via SRAM-CIM array + shift-and-add (bit-serial)"
        )
        stage1.add_sub_stage(SubStage(
            name="k_square_compute",
            circuit_modules=[
                CircuitModule.SRAM_CIM_ARRAY,
                CircuitModule.SHIFT_ADD,
            ],
            is_pipeline=False,
            description="SRAM-CIM array read + shift-and-add for k^2 (bit-serial)"
        ))
        self.stages.append(stage1)

        # Stage 2: -2qk cross-term
        stage2 = Stage(
            name="cross_term",
            is_pipeline=False,
            description="-2qk via SRAM-CIM array + shift-and-add (bit-serial)"
        )
        stage2.add_sub_stage(SubStage(
            name="cross_term_compute",
            circuit_modules=[
                CircuitModule.SRAM_CIM_ARRAY,
                CircuitModule.SHIFT_ADD,
            ],
            is_pipeline=False,
            description="SRAM-CIM array MAC + shift-and-add for -2qk (bit-serial)"
        ))
        self.stages.append(stage2)

        # Stage 3: Distance accumulation
        stage3 = Stage(
            name="accumulate",
            is_pipeline=False,
            description="Accumulate k^2 + (-2qk) via shift-and-add"
        )
        stage3.add_sub_stage(SubStage(
            name="shift_add_accumulate",
            circuit_modules=[CircuitModule.SHIFT_ADD],
            is_pipeline=False,
            description="Shift-and-add: distance = k^2 + (-2qk)"
        ))
        self.stages.append(stage3)

        # Stage 4: SRAM write
        stage4 = Stage(
            name="sram_write",
            is_pipeline=False,
            description="Write computed distances to SRAM"
        )
        stage4.add_sub_stage(SubStage(
            name="distance_write",
            circuit_modules=[CircuitModule.BUFFER_OUTPUT],
            is_pipeline=False,
            description="Write distance values to SRAM for sorting"
        ))
        self.stages.append(stage4)

        # Stage 5: Bitonic sort (1024->24)
        stage5 = Stage(
            name="bitonic_sort",
            is_pipeline=False,
            description="Bitonic sort (1024->24) to find k nearest neighbors"
        )
        stage5.add_sub_stage(SubStage(
            name="bitonic_sort_1024_to_k",
            circuit_modules=[CircuitModule.BITONIC_SORT],
            is_pipeline=False,
            description="Bitonic sort 1024->24: find k nearest neighbors"
        ))
        self.stages.append(stage5)

    # --- Cycle calculation ---

    def calculate_cycles(self, params: Optional[KNNParams] = None) -> int:
        """
        Calculate total KNN execution cycles.

        KNN search space is constant (input_points) across all queries.

        Formula:
            per_batch = k2_width / bit_slice + two_qk_width / bit_slice + 1 + 1
            batches = ceil(input_points / cim_parallelism)
            per_query = batches * per_batch + bitonic_sort_cycles
            total = query_points * per_query

        Returns:
            Total execution cycles.
        """
        if params is not None:
            self.knn_params = params

        p = self.knn_params
        per_batch = self._get_per_batch_cycles()

        batches = math.ceil(p.input_points / p.cim_parallelism)
        sort_cycles = self._get_sort_cycles()
        per_query = batches * per_batch + sort_cycles
        total_cycles = p.query_points * per_query

        return total_cycles

    # --- Energy calculation ---

    def calculate_energy(self, params: Optional[KNNParams] = None) -> float:
        """Calculate total energy for KNN (nJ). Computed from generated events."""
        events = self.generate_events(params)
        return sum(e.energy for e in events)

    # --- Event generation ---

    def _generate_single_merge_sort_event(self, params: Optional[KNNParams] = None) -> Event:
        """base_v2：单次 KNN 查询 MergeSort top32 事件（每批 64 做 top32 共 22cy，batches 批）。"""
        if params is not None:
            self.knn_params = params
        p = self.knn_params
        hw = self.hw_config
        ms = hw.sorting_circuit.merge_sort
        batches = math.ceil(p.input_points / p.cim_parallelism)  # 1024/64=16
        duration = batches * ms.cycles_top32
        return Event(
            event_id=0,
            cycle=0,
            operator="KNN",
            stage="merge_sort",
            sub_stage="merge_sort_top32_merge",
            circuit_module="merge_sort",
            operation="merge_sort_top32",
            power=ms.power_top32_mw,
            duration=duration,
            clock_period_ns=hw.clock_period,
            metadata={"phase": "knn_engine_merge_sort_only", "batches": batches},
        )

    def _generate_merge_sort_reg_file_read_event(
        self, cycle: int, event_id: int, num_reads: int = 32, params: Optional[KNNParams] = None
    ) -> Optional[Event]:
        """base_v2：MergeSort 寄存器组读事件。KNN 每 query 读 32 个最小距离缓存，占关键路径。"""
        if params is not None:
            self.knn_params = params
        hw = self.hw_config
        ms = getattr(hw.sorting_circuit, "merge_sort", None)
        reg = getattr(ms, "reg_file", None) if ms is not None else None
        if reg is None:
            return None
        clock_period = hw.clock_period
        num_bytes = int(num_reads) * 4
        duration, energy_nj = query_sram_cycles_energy(
            num_bytes,
            is_read=True,
            hw_config=hw,
            clock_period_ns=clock_period,
            buffer_name="sort_cache_sram",
        )
        power = (energy_nj * 1000.0 / (max(1, duration) * clock_period)) if duration > 0 and clock_period > 0 else 0.0
        return Event(
            event_id=event_id,
            cycle=cycle,
            operator="SRAM",
            stage="merge_sort_reg_file",
            sub_stage="reg_file_read_32min",
            circuit_module="sort_cache_sram",
            operation="reg_file_read",
            power=power,
            duration=duration,
            clock_period_ns=clock_period,
            metadata={"num_reads": num_reads, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": num_bytes, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
        )

    def _generate_merge_sort_reg_file_event(
        self, cycle: int, event_id: int, num_writes: int = 32, overlapped: bool = False,
        params: Optional[KNNParams] = None,
    ) -> Optional[Event]:
        """base_v2：MergeSort 寄存器组写事件。KNN 写 32 个最小距离；overlapped 时与下一批 64选32 重叠。"""
        if params is not None:
            self.knn_params = params
        hw = self.hw_config
        ms = getattr(hw.sorting_circuit, "merge_sort", None)
        reg = getattr(ms, "reg_file", None) if ms is not None else None
        if reg is None:
            return None
        clock_period = hw.clock_period
        num_bytes = int(num_writes) * 4
        duration, energy_nj = query_sram_cycles_energy(
            num_bytes,
            is_read=False,
            hw_config=hw,
            clock_period_ns=clock_period,
            buffer_name="sort_cache_sram",
        )
        power = (energy_nj * 1000.0 / (max(1, duration) * clock_period)) if duration > 0 and clock_period > 0 else 0.0
        ev = Event(
            event_id=event_id,
            cycle=cycle,
            operator="SRAM",
            stage="merge_sort_reg_file",
            sub_stage="reg_file_write_32min",
            circuit_module="sort_cache_sram",
            operation="reg_file_write",
            power=power,
            duration=duration,
            clock_period_ns=clock_period,
            metadata={"num_writes": num_writes, "buffer_name": "sort_cache_sram", "buffer_phase": "write", "bytes": num_bytes, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
        )
        ev.overlapped = overlapped
        return ev

    def generate_single_bitonic_event(self, params: Optional[KNNParams] = None) -> Event:
        """返回单次 KNN 查询的 baseline bitonic top-k 事件（cycle=0）。"""
        if params is not None:
            self.knn_params = params
        p = self.knn_params
        hw = self.hw_config
        bitonic = hw.sorting_circuit.bitonic_sort
        sort_power = bitonic.power_min_mw
        return Event(
            event_id=0,
            cycle=0,
            operator="KNN",
            stage="bitonic_sort",
            sub_stage="bitonic_sort_1024_to_k",
            circuit_module="comp_bitonic_1024_min32",
            operation="bitonic_sort_min_k",
            power=sort_power,
            duration=p.bitonic_sort_cycles,
            clock_period_ns=hw.clock_period,
            metadata={
                "phase": "knn_engine_bitonic_only",
                # baseline 下可与 FPS 共享同一套 bitonic 比较器（由 Scheduler 做互斥仲裁）
                "resource_id": "bitonic_shared",
                "exclusive": True,
            },
        )

    def generate_single_fuse_events(self, params: Optional[KNNParams] = None) -> List[Event]:
        """
        返回单次 KNN 查询的 fuse 模式事件（粗筛 + 细筛）。

        说明：
        - fuse 模式下不再使用 baseline bitonic 网络；
        - 采用 coarse_filter + fine_filter 配置。
        """
        if params is not None:
            self.knn_params = params

        p = self.knn_params
        hw = self.hw_config
        coarse_cfg = getattr(hw.sorting_circuit, "coarse_filter", None)
        fine_cfg = getattr(hw.sorting_circuit, "fine_filter", None)
        coarse_power = float(getattr(coarse_cfg, "power", 38.0) if coarse_cfg is not None else 38.0)
        fine_power = float(getattr(fine_cfg, "power", 30.0) if fine_cfg is not None else 30.0)
        coarse_rounds = max(1, int(getattr(hw, "knn_dynamic_threshold_rounds_mean", getattr(hw, "knn_coarse_rounds_mean", 1)) or 1))
        explicit_coarse_cycles = int(getattr(coarse_cfg, "cycles", 0) if coarse_cfg is not None else 0)
        coarse_cycles = explicit_coarse_cycles if explicit_coarse_cycles > 0 else (4 * coarse_rounds + 1)
        fine_cycles = int(getattr(fine_cfg, "cycles", 2) if fine_cfg is not None else 2)

        return [
            Event(
                event_id=0,
                cycle=0,
                operator="KNN",
                stage="fuse_filter",
                sub_stage="coarse_filter",
                circuit_module="coarse_filter",
                operation="knn_coarse_filter",
                power=coarse_power,
                duration=coarse_cycles,
                clock_period_ns=hw.clock_period,
                metadata={
                    "phase": "knn_engine_fuse",
                    "resource_id": "comparator_shared",
                    "exclusive": True,
                    "coarse_rounds_mean": coarse_rounds,
                },
            ),
            Event(
                event_id=1,
                cycle=coarse_cycles,
                operator="KNN",
                stage="fuse_filter",
                sub_stage="fine_filter",
                circuit_module="fine_filter",
                operation="knn_fine_filter_topk",
                power=fine_power,
                duration=fine_cycles,
                clock_period_ns=hw.clock_period,
                metadata={
                    "phase": "knn_engine_fuse",
                    "resource_id": "comparator_shared",
                    "exclusive": True,
                },
            ),
        ]

    def generate_events(self, params: Optional[KNNParams] = None, rram_already_loaded: bool = False) -> List[Event]:
        """
        Generate cycle-level events for KNN.

        当 hw_config.distance_in_phase2=True 时，仅返回 bitonic 事件（stage5），
        距离计算（stage1-4）由 Phase2 在 dispatcher 中统一插入。

        Args:
            rram_already_loaded: PointCIM 模式下，若同层 FPS 已完成 RRAM 写入 + self_k2 初始化，
                置 True 可跳过 KNN 的重复初始化事件（FPS 与 KNN 共享同一点云的 RRAM 阵列）。
        """
        if params is not None:
            self.knn_params = params

        p = self.knn_params
        hw = self.hw_config
        if getattr(hw, "distance_in_phase2", False):
            if getattr(hw, "is_baseline_v2", False):
                return [self._generate_single_merge_sort_event() for _ in range(p.query_points)]
            return [self.generate_single_bitonic_event() for _ in range(p.query_points)]

        # 完整 5 阶段：Stage 1–3 复用共用距离计算，Stage 4/5 为 KNN 专属
        batches = math.ceil(p.input_points / p.cim_parallelism)
        if getattr(hw, "is_baseline_v2", False) and getattr(hw.sorting_circuit, "merge_sort", None) is not None:
            ms = hw.sorting_circuit.merge_sort
            sort_power = ms.power_top32_mw
            sort_cycles_per_batch = ms.cycles_top32  # 22，每批 64 的 top32
            sort_cycles_total = batches * sort_cycles_per_batch  # 流式：每批 top32 后归并截断
            sort_stage = "merge_sort"
            sort_circuit = "merge_sort"
        else:
            bitonic = hw.sorting_circuit.bitonic_sort
            sort_power = bitonic.power_min_mw
            sort_cycles_total = self._get_sort_cycles()
            sort_stage = "bitonic_sort"
            sort_circuit = "comp_bitonic_1024_min32"
        clock_period = hw.clock_period
        dist_cycles = self._get_dist_compute_cycles()

        events: List[Event] = []
        event_id = 0
        cycle = 0

        # PointCIM: one-time initialization（rram_write → rram_self_k2）
        # 若 rram_already_loaded=True（同层 FPS 已初始化），跳过，避免对同一点云写入两次 RRAM
        if getattr(hw, "use_rram_dist", False) and getattr(hw, "rram_dist", None) is not None and not rram_already_loaded:
            init_meta = {"input_points": p.input_points, "operator": "KNN", "phase": "init"}
            write_evs, event_id, cycle = gen_rram_dist_write_events(
                "KNN", hw, event_id, cycle, p.input_points, init_meta,
            )
            events.extend(write_evs)
            k2_evs, event_id, cycle = gen_rram_dist_self_k2_events(
                "KNN", hw, event_id, cycle, p.input_points, init_meta,
            )
            events.extend(k2_evs)

        for q in range(p.query_points):
            for b in range(batches):
                # Stage 1–3（distance）：
                #   baseline → dist2_engine 脉动阵列
                #   acc      → SRAM-CIM k²/-2qk/shift_add
                #   pointcim → rram_vmm(cross)/rram_self_q/rram_sum
                batch_m = min(p.cim_parallelism, p.input_points - b * p.cim_parallelism)
                dist_meta = {"query": q, "batch": b, "input_points": p.input_points}
                dist_events, event_id, cycle = gen_distance_events_one_batch(
                    "KNN", hw, event_id, cycle, dist_meta,
                    dist_cycles=dist_cycles, batch_size_m=batch_m,
                )
                events.extend(dist_events)
                # Stage 4：KNN 专属 — 距离写回 output_buffer
                sram_cycles, sram_energy = query_sram_cycles_energy(
                    int(batch_m) * 4,
                    is_read=False,
                    hw_config=hw,
                    clock_period_ns=clock_period,
                    buffer_name="dist_sram_buffer",
                )
                sram_power = (sram_energy * 1000.0 / (max(1, sram_cycles) * clock_period)) if sram_cycles > 0 and clock_period > 0 else 0.0
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="SRAM",
                    stage="sram_write",
                    sub_stage="distance_write",
                    circuit_module="dist_sram_buffer",
                    operation="distance_write",
                    power=sram_power,
                    duration=sram_cycles,
                    clock_period_ns=clock_period,
                    metadata={"query": q, "batch": b, "buffer_name": "dist_sram_buffer", "buffer_phase": "write", "bytes": int(batch_m) * 4, "resource_id": "dist_sram_shared", "exclusive": True, "reset_scope": "query_boundary"},
                ))
                event_id += 1
                cycle += sram_cycles

            # base_v2：reg_read(32) 再 merge_sort，再 reg_write(32, overlapped)
            if getattr(hw, "is_baseline_v2", False) and getattr(hw.sorting_circuit, "merge_sort", None) is not None:
                sort_read_cycles, sort_read_energy = query_sram_cycles_energy(
                    int(p.input_points) * 4,
                    is_read=True,
                    hw_config=hw,
                    clock_period_ns=clock_period,
                    buffer_name="dist_sram_buffer",
                )
                sort_read_power = (sort_read_energy * 1000.0 / (max(1, sort_read_cycles) * clock_period)) if sort_read_cycles > 0 and clock_period > 0 else 0.0
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="SRAM",
                    stage="merge_sort",
                    sub_stage="sort_input_read",
                    circuit_module="dist_sram_buffer",
                    operation="dist_sram_sort_read",
                    power=sort_read_power,
                    duration=sort_read_cycles,
                    clock_period_ns=clock_period,
                    metadata={"query": q, "buffer_name": "dist_sram_buffer", "buffer_phase": "sort_read", "bytes": int(p.input_points) * 4, "resource_id": "dist_sram_shared", "exclusive": True, "reset_scope": "query_boundary"},
                ))
                event_id += 1
                cycle += sort_read_cycles
                reg_read_ev = self._generate_merge_sort_reg_file_read_event(cycle, event_id, num_reads=32)
                if reg_read_ev is not None:
                    events.append(reg_read_ev)
                    event_id += 1
                    cycle += reg_read_ev.duration
            # Stage 5：排序（每 query 一次；base_v1=Bitonic，base_v2=MergeSort 每批 top32 后归并）
            if sort_stage == "bitonic_sort":
                sort_read_cycles, sort_read_energy = query_sram_cycles_energy(
                    int(p.input_points) * 4,
                    is_read=True,
                    hw_config=hw,
                    clock_period_ns=clock_period,
                    buffer_name="dist_sram_buffer",
                )
                sort_read_power = (sort_read_energy * 1000.0 / (max(1, sort_read_cycles) * clock_period)) if sort_read_cycles > 0 and clock_period > 0 else 0.0
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="SRAM",
                    stage="bitonic_sort",
                    sub_stage="sort_input_read",
                    circuit_module="dist_sram_buffer",
                    operation="dist_sram_sort_read",
                    power=sort_read_power,
                    duration=sort_read_cycles,
                    clock_period_ns=clock_period,
                    metadata={"query": q, "buffer_name": "dist_sram_buffer", "buffer_phase": "sort_read", "bytes": int(p.input_points) * 4, "resource_id": "dist_sram_shared", "exclusive": True, "reset_scope": "query_boundary"},
                ))
                event_id += 1
                cycle += sort_read_cycles
            events.append(Event(
                event_id=event_id,
                cycle=cycle,
                operator="KNN",
                stage=sort_stage,
                sub_stage="bitonic_sort_1024_to_k" if sort_stage == "bitonic_sort" else "merge_sort_top32_merge",
                circuit_module=sort_circuit,
                operation="bitonic_sort_min_k" if sort_stage == "bitonic_sort" else "merge_sort_top32",
                power=sort_power,
                duration=sort_cycles_total,
                clock_period_ns=clock_period,
                metadata={"query": q, "buffer_name": "sort_cache_sram" if sort_stage == "merge_sort" else "dist_sram_buffer", "buffer_phase": "sort_read", "bytes": (32 * 4) if sort_stage == "merge_sort" else (int(p.input_points) * 4), "resource_id": "merge_sort_shared" if sort_stage == "merge_sort" else "bitonic_shared", "exclusive": True},
            ))
            event_id += 1
            cycle += sort_cycles_total
            # base_v2：寄存器组写 32 个最小距离（与下一 query 或下一批重叠，不占关键路径）
            reg_ev = self._generate_merge_sort_reg_file_event(
                cycle, event_id, num_writes=32, overlapped=True,
            )
            if reg_ev is not None:
                events.append(reg_ev)
                event_id += 1
                # 写 overlapped，不推进 cycle

        return events

    # --- Distance matrix DRAM writeback ---

    def generate_dram_writeback_events(
        self, start_cycle: int = 0, start_event_id: int = 0, params: Optional[KNNParams] = None,
        streaming: bool = False,
        per_query_ready_cycles: Optional[List[int]] = None,
        transpose_store: bool = False,
    ) -> List[Event]:
        """
        生成距离矩阵 DRAM 写回事件。

        两种模式：
        - streaming=False (legacy): 所有 query 完成后一次性写入整个矩阵（bulk tail）。
        - streaming=True: 逐 query 流式写入，每个 query 的距离行在该 query 计算完成后
          立即发起 DRAM 写入。写入事件标记为 overlapped，可与下一个 query 的距离计算并行。
          per_query_ready_cycles[q] 给出第 q 个 query 距离就绪的周期。

        transpose_store:
          为 True 时按列主序 dist_T[N, Q] 写入 DRAM，即每个 query q 的距离值
          写到 (n, q) 位置而非 (q, n)。这使 FP 层（上采样方向，N>Q）的后续读取
          变为连续行读，消除 row buffer miss 开销。
          代价是每个 query 的 N 个写入跨 N 个 DRAM 行（stride = Q*4 bytes），
          需要额外的 row switch penalty。由于写入是 overlapped（不在关键路径上），
          增加的写延迟不影响端到端 latency，仅影响写能耗。
        """
        if params is not None:
            self.knn_params = params
        p = self.knn_params
        hw = self.hw_config
        clock_period = hw.clock_period
        row_bytes = p.input_points * 4
        total_bytes = p.query_points * row_bytes

        if not streaming:
            if transpose_store:
                dur, energy_nj = query_dram_matrix_cycles_energy(
                    p.query_points,
                    p.input_points,
                    element_bytes=4,
                    is_read=False,
                    hw_config=hw,
                    clock_period_ns=clock_period,
                    transpose_access=True,
                )
            else:
                dur, energy_nj = query_dram_cycles_energy(
                    total_bytes, is_read=False, hw_config=hw, clock_period_ns=clock_period,
                )
            dur = max(1, int(dur))
            power = (energy_nj * 1e3 / (dur * clock_period)) if dur > 0 and clock_period > 0 else 0.0
            return [Event(
                event_id=start_event_id,
                cycle=start_cycle,
                operator="DRAM",
                stage="dist_matrix_dram_write",
                sub_stage="distance_matrix_writeback",
                circuit_module="dram",
                operation="dist_matrix_dram_write",
                power=power,
                duration=dur,
                clock_period_ns=clock_period,
                metadata={
                    "phase": "dist_matrix_store" + ("_transposed" if transpose_store else ""),
                    "query_points": p.query_points,
                    "input_points": p.input_points,
                    "bytes": total_bytes,
                    "transpose_store": transpose_store,
                    "buffer_name": "offchip_dram",
                    "buffer_phase": "dist_matrix_write",
                    "resource_id": "dram_port",
                },
            )]

        # --- 流式写入 ---
        events: List[Event] = []
        eid = start_event_id
        batch_requests: List[Dict[str, Any]] = []
        for q in range(p.query_points):
            ready = per_query_ready_cycles[q] if per_query_ready_cycles and q < len(per_query_ready_cycles) else start_cycle
            if transpose_store:
                batch_requests.append({
                    "kind": "matrix",
                    "rows": 1,
                    "cols": p.input_points,
                    "element_bytes": 4,
                    "is_read": False,
                    "transpose_access": True,
                    "row_offset": q,
                    "physical_row_stride_elems": p.query_points,
                    "start_cycle": ready,
                })
            else:
                batch_requests.append({
                    "num_bytes": row_bytes,
                    "is_read": False,
                    "start_cycle": ready,
                })
        batch_results = query_dram_batch_cycles_energy(batch_requests, hw, clock_period)
        for q in range(p.query_points):
            ready = per_query_ready_cycles[q] if per_query_ready_cycles and q < len(per_query_ready_cycles) else start_cycle
            dur, energy_nj = batch_results[q]
            dur = max(1, int(dur))
            power = (energy_nj * 1e3 / (dur * clock_period)) if dur > 0 and clock_period > 0 else 0.0
            ev = Event(
                event_id=eid,
                cycle=ready,
                operator="DRAM",
                stage="dist_matrix_dram_write",
                sub_stage="distance_row_writeback",
                circuit_module="dram",
                operation="dist_matrix_dram_write",
                power=power,
                duration=dur,
                clock_period_ns=clock_period,
                metadata={
                    "phase": "dist_matrix_store_streaming" + ("_transposed" if transpose_store else ""),
                    "query": q,
                    "input_points": p.input_points,
                    "bytes": row_bytes,
                    "transpose_store": transpose_store,
                    "buffer_name": "offchip_dram",
                    "buffer_phase": "dist_matrix_write",
                    "resource_id": "dram_port",
                    "dram_pipelined": True,
                },
            )
            ev.overlapped = True
            events.append(ev)
            eid += 1

        return events

    def generate_distance_reuse_events(self, params: Optional[KNNParams] = None,
                                       transpose_store: bool = False) -> List[Event]:
        """
        距离矩阵复用模式：跳过 Stage 1-4 距离计算，从 DRAM 读取已存储的距离矩阵子集，
        然后仅做 sort top-k。

        两阶段模型：
        Phase A - 批量 DRAM 读取：将需要的距离子矩阵一次性读入片上 buffer。
        Phase B - 逐 query sort：从片上 buffer 取每 query 的距离行做 sort top-k。

        transpose_store=True 时，距离矩阵以 dist_T[N, Q]（列主序）存储在 DRAM 中。
        此时原本需要转置读的方向（FP 层）变为连续行读，
        而原本连续的方向（SA 后续层）变为需要跨步读。
        即 is_transpose_read 标志的含义翻转：
          - FP 层 (is_transpose_read=True) → 转置存储后变连续读（无 penalty）
          - SA 层 (is_transpose_read=False) → 转置存储后变跨步读（有 penalty，但数据量小）
        """
        if params is not None:
            self.knn_params = params
        p = self.knn_params
        hw = self.hw_config
        clock_period = hw.clock_period

        ms_cfg = getattr(hw.sorting_circuit, "merge_sort", None)
        bitonic_cfg = getattr(hw.sorting_circuit, "bitonic_sort", None)
        if getattr(hw, "is_baseline_v2", False) and ms_cfg is not None:
            sort_power = ms_cfg.power_top32_mw
            batches = math.ceil(p.input_points / p.cim_parallelism)
            sort_cycles_per_query = batches * ms_cfg.cycles_top32
            sort_stage = "merge_sort"
            sort_circuit = "merge_sort"
        elif bitonic_cfg is not None:
            sort_power = bitonic_cfg.power_min_mw
            sort_cycles_per_query = self._get_sort_cycles()
            sort_stage = "bitonic_sort"
            sort_circuit = "comp_bitonic_1024_min32"
        else:
            sort_power = 1540.0
            sort_cycles_per_query = p.bitonic_sort_cycles
            sort_stage = "bitonic_sort"
            sort_circuit = "comp_bitonic_1024_min32"

        # 转置存储时翻转读取模式：原本 FP 的转置读 → 连续；原本 SA 的连续读 → 转置
        is_transpose = p.is_transpose_read
        if transpose_store:
            is_transpose = not is_transpose
        total_bytes = p.query_points * p.input_points * 4

        events: List[Event] = []
        event_id = 0
        cycle = 0

        # --- Phase A: bulk DRAM read of the distance sub-matrix ---
        if is_transpose:
            dram_dur, dram_nj = query_dram_matrix_cycles_energy(
                p.query_points,
                p.input_points,
                element_bytes=4,
                is_read=True,
                hw_config=hw,
                clock_period_ns=clock_period,
                transpose_access=True,
            )
        else:
            dram_dur, dram_nj = query_dram_cycles_energy(
                total_bytes, is_read=True, hw_config=hw, clock_period_ns=clock_period,
            )
        dram_dur = max(1, int(dram_dur))

        dram_power = (dram_nj * 1e3 / (dram_dur * clock_period)) if dram_dur > 0 and clock_period > 0 else 0.0
        events.append(Event(
            event_id=event_id,
            cycle=cycle,
            operator="DRAM",
            stage="dist_reuse_dram_read",
            sub_stage="bulk_matrix_read" if is_transpose else "bulk_row_read",
            circuit_module="dram",
            operation="dist_reuse_dram_read",
            power=dram_power,
            duration=dram_dur,
            clock_period_ns=clock_period,
            metadata={
                "phase": "distance_reuse",
                "access_pattern": "bulk_transpose" if is_transpose else "bulk_contiguous",
                "transpose_store": transpose_store,
                "total_bytes": total_bytes,
                "query_points": p.query_points,
                "input_points": p.input_points,
                "buffer_name": "offchip_dram",
                "buffer_phase": "dist_reuse_read",
                "resource_id": "dram_port",
            },
        ))
        event_id += 1
        cycle += dram_dur

        # --- Phase B: per-query sort from on-chip buffer ---
        for q in range(p.query_points):
            events.append(Event(
                event_id=event_id,
                cycle=cycle,
                operator="KNN",
                stage=sort_stage,
                sub_stage="bitonic_sort_reuse_topk" if sort_stage == "bitonic_sort" else "merge_sort_reuse_topk",
                circuit_module=sort_circuit,
                operation="bitonic_sort_min_k" if sort_stage == "bitonic_sort" else "merge_sort_top32",
                power=sort_power,
                duration=sort_cycles_per_query,
                clock_period_ns=clock_period,
                metadata={
                    "query": q,
                    "phase": "distance_reuse",
                    "resource_id": "merge_sort_shared" if sort_stage == "merge_sort" else "bitonic_shared",
                    "exclusive": True,
                },
            ))
            event_id += 1
            cycle += sort_cycles_per_query

        return events

    # --- Reporting helpers ---

    def get_cycle_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed cycle breakdown for KNN.

        Returns:
            Dictionary with per-batch, per-query, and total cycle info.
        """
        p = self.knn_params
        dist_c = self._get_dist_compute_cycles()
        per_batch = self._get_per_batch_cycles()

        batches = math.ceil(p.input_points / p.cim_parallelism)
        stages_1_4_cycles = batches * per_batch
        sort_cycles = self._get_sort_cycles()
        per_query = stages_1_4_cycles + sort_cycles
        total = p.query_points * per_query

        return {
            'params': {
                'input_points': p.input_points,
                'query_points': p.query_points,
                'k': p.k,
                'sampled_points_num': p.sampled_points_num,
                'cim_parallelism': p.cim_parallelism,
                'bit_slice': p.bit_slice,
                'points_per_axis_quant_width': p.points_per_axis_quant_width,
                'bitonic_sort_cycles': p.bitonic_sort_cycles,
            },
            'per_batch': {
                'dist_compute_cycles': dist_c,
                'accumulate_cycles': 1,
                'sram_write_cycles': 1,
                'total': per_batch,
            },
            'per_query': {
                'search_points': p.input_points,
                'batches': batches,
                'stages_1_4': stages_1_4_cycles,
                'bitonic_sort': sort_cycles,
                'total': per_query,
            },
            'total': {
                'queries': p.query_points,
                'cycles': total,
            },
        }
