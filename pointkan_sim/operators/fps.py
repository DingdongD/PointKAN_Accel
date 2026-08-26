"""
FPS 算子（最远点采样）
======================

基于 SRAM-CIM 的最远点采样：k^2、-2qk、距离累加、最小距离表更新、Bitonic 排序五阶段，
每轮迭代对候选点按批做距离计算，最后做一次 Bitonic 排序选出最远点。
生成事件时 operator 统一为 "FPS"（PascalCase），供 operator_sequence.txt 使用。

Uses 5 sequential stages:
1. k^2 computation (SRAM-CIM-array + shift-and-add, bit-serial)
2. -2qk cross-term computation (SRAM-CIM-array + shift-and-add, bit-serial)
3. Distance accumulation (shift-and-add)
4. Min-distance table update (comparator + buffer)
5. Bitonic sort (1024->1, find the farthest point)

Execution flow:
    For each iteration i = 0 .. sampled_points_num - 1:
        candidate_points = input_points - i
        batches = ceil(candidate_points / cim_parallelism)
        For each batch:
            Stage 1: k^2             (dist_compute_cycles)
            Stage 2: -2qk            (dist_compute_cycles)
            Stage 3: accumulate      (1 cycle)
            Stage 4: table update    (1 cycle)
        Stage 5: bitonic sort        (fixed cycles, after all batches)

    dist_compute_cycles = points_per_axis_quant_width / bit_slice

Cycle formula:
    per_batch = 2 * dist_compute_cycles + 1 + 1
    For iteration i:
        candidate_points = input_points - i
        batches = ceil(candidate_points / cim_parallelism)
        iteration_cycles = batches * per_batch + bitonic_sort_cycles
    total = sum over sampled_points_num iterations
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import math

from .base_operator import (
    BaseOperator, Stage, SubStage,
    OperatorParams,
)
from ..configs.hw_config import CircuitModule, MinDistanceTableConfig
from ..simulator.event import Event
from .distance_ops import (
    gen_distance_events_one_batch,
    gen_rram_dist_write_events,
    gen_rram_dist_self_k2_events,
)
from ..integration.memory_model_adapter import query_sram_cycles_energy

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


@dataclass
class FPSParams(OperatorParams):
    """
    FPS 专用参数：继承 input_points；额外包括 sampled_points_num（采样点数/迭代次数）、
    cim_parallelism、bit_slice、points_per_axis_quant_width、bitonic_sort_cycles。
    """
    sampled_points_num: int = 512
    cim_parallelism: int = 128
    bit_slice: int = 2
    points_per_axis_quant_width: int = 28
    bitonic_sort_cycles: int = 10   # MAX 模式（FPS 1024→1）：10 周期


class FPSOperator(BaseOperator):
    """
    Farthest Point Sampling (FPS) Operator.

    SRAM-CIM based FPS with fixed parallelism and bitonic sort.
    Stage 1–3（距离计算）与 KNN 共用 distance_ops.gen_sram_cim_distance_events_one_batch；
    本算子仅负责 Stage 4（最小距离表更新）与 Stage 5（Bitonic 排序）。

    Processing flow (per iteration, one per sampled point):

    [Stages 1-4: sequential, repeated for each batch]

    1–3. 距离计算（共用）
       k^2、-2qk、shift_add 累加，由 distance_ops 生成。
       Circuit modules: SRAM_CIM_ARRAY, SHIFT_ADD

    4. Min-distance table update（FPS 专属）
       Compare and update per-point minimum distance.
       Duration = 1 cycle. Circuit: min_distance_table

    [Stage 5: runs once after all batches complete]

    5. Bitonic sort (1024->1), MAX 模式
       Find the farthest point (max distance).
       Duration = bitonic_sort_cycles (default 10). Circuit: comp_tree_1024_max
    """

    def __init__(
        self,
        params: Optional[FPSParams] = None,
        hw_config: Optional['HardwareConfig'] = None
    ):
        if params is None:
            params = FPSParams()
        self.fps_params: FPSParams = params
        super().__init__("FPS", params, hw_config)

    # --- Derived parameter helpers ---

    def _get_dist_compute_cycles(self) -> int:
        """Cycles for distance computation (k^2 or -2qk) per batch."""
        p = self.fps_params
        return p.points_per_axis_quant_width // p.bit_slice

    def _get_per_batch_cycles(self) -> int:
        """Total cycles per batch (stages 1-4)."""
        return 2 * self._get_dist_compute_cycles() + 1 + 1

    def _get_sort_cycles(self, candidate_points: int) -> int:
        """
        计算排序周期，考虑 sorter_depth 多 pass + merge 开销。

        当 candidate_points <= sorter_depth 时退化为基准 bitonic_sort_cycles；
        超出时需 ceil(candidate_points/sorter_depth) 趟局部排序 + log2 阶 merge。
        """
        p = self.fps_params
        hw = self.hw_config
        base_cycles = p.bitonic_sort_cycles
        if hw is None:
            return base_cycles
        sort_cfg = getattr(getattr(hw, "sorting_circuit", None), "bitonic_sort", None)
        if sort_cfg is None:
            return base_cycles
        depth = int(getattr(sort_cfg, "sorter_depth", 0) or 0)
        if depth <= 0 or depth >= candidate_points:
            return base_cycles
        num_passes = math.ceil(candidate_points / depth)
        merge_stages = max(0, int(math.ceil(math.log2(num_passes))))
        # 每 pass 基准排序 + merge 每阶取半个基准排序周期
        return num_passes * base_cycles + merge_stages * max(1, base_cycles // 2)

    # --- Pipeline helpers (供 dispatcher 做 FPS→KNN 流水线重叠) ---

    def get_per_iteration_cycles_list(self) -> 'List[int]':
        """
        返回每轮 FPS 迭代的周期数列表（含距离计算 + 排序），
        用于 dispatcher 计算 producer-consumer 流水线调度。
        """
        p = self.fps_params
        per_batch = self._get_per_batch_cycles()
        result: 'List[int]' = []
        for i in range(p.sampled_points_num):
            candidate_points = p.input_points - i
            batches = math.ceil(candidate_points / p.cim_parallelism)
            sort_cycles = self._get_sort_cycles(candidate_points)
            result.append(batches * per_batch + sort_cycles)
        return result

    # --- Stage definition ---

    def _define_stages(self) -> None:
        """
        Define FPS hardware processing stages.

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

        # Stage 4: Min-distance table update
        stage4 = Stage(
            name="min_distance_update",
            is_pipeline=False,
            description="Update per-point minimum distance table"
        )
        stage4.add_sub_stage(SubStage(
            name="table_update",
            circuit_modules=[CircuitModule.MIN_DISTANCE_TABLE],
            is_pipeline=False,
            description="Compare and update per-point min-distance table"
        ))
        self.stages.append(stage4)

        # Stage 5: Bitonic sort (1024->1)
        stage5 = Stage(
            name="bitonic_sort",
            is_pipeline=False,
            description="Bitonic sort (1024->1) to find farthest point"
        )
        stage5.add_sub_stage(SubStage(
            name="bitonic_sort_1024_to_1",
            circuit_modules=[CircuitModule.BITONIC_SORT],
            is_pipeline=False,
            description="Bitonic sort 1024->1: find global maximum distance"
        ))
        self.stages.append(stage5)

    # --- Cycle calculation ---

    def calculate_cycles(self, params: Optional[FPSParams] = None) -> int:
        """
        Calculate total FPS execution cycles.

        For each iteration i (0 to sampled_points_num - 1):
            candidate_points = input_points - i
            batches = ceil(candidate_points / cim_parallelism)
            iteration_cycles = batches * per_batch + bitonic_sort_cycles
        total = sum of all iteration_cycles

        Returns:
            Total execution cycles.
        """
        if params is not None:
            self.fps_params = params

        p = self.fps_params
        per_batch = self._get_per_batch_cycles()

        total_cycles = 0
        for i in range(p.sampled_points_num):
            candidate_points = p.input_points - i
            batches = math.ceil(candidate_points / p.cim_parallelism)
            sort_cycles = self._get_sort_cycles(candidate_points)
            total_cycles += batches * per_batch + sort_cycles

        return total_cycles

    # --- Energy calculation ---

    def calculate_energy(self, params: Optional[FPSParams] = None) -> float:
        """Calculate total energy for FPS (nJ). Computed from generated events."""
        events = self.generate_events(params)
        return sum(e.energy for e in events)

    # --- Event generation ---

    def _generate_single_merge_sort_event(self, params: Optional[FPSParams] = None) -> Event:
        """base_v2：单次 FPS 迭代 MergeSort 局部 max 事件（22 cycle）。仅用于无 reg_file 或兼容。"""
        if params is not None:
            self.fps_params = params
        p = self.fps_params
        hw = self.hw_config
        ms = hw.sorting_circuit.merge_sort
        return Event(
            event_id=0,
            cycle=0,
            operator="FPS",
            stage="merge_sort",
            sub_stage="merge_sort_local_max",
            circuit_module="merge_sort",
            operation="merge_sort_max",
            power=ms.power_max_mw,
            duration=ms.cycles_max,
            clock_period_ns=hw.clock_period,
            metadata={"phase": "fps_engine_merge_sort_only"},
        )

    def _generate_single_merge_sort_iteration_events(self, params: Optional[FPSParams] = None) -> List[Event]:
        """base_v2 Phase2：单次 FPS 迭代的 merge_sort + reg_read + reg_write(overlapped) 事件列表（cycle 相对 0）。"""
        if params is not None:
            self.fps_params = params
        p = self.fps_params
        hw = self.hw_config
        ms = getattr(hw.sorting_circuit, "merge_sort", None)
        reg = getattr(ms, "reg_file", None) if ms is not None else None
        batches = math.ceil(p.input_points / p.cim_parallelism)
        events: List[Event] = []
        event_id = 0
        cycle = 0
        for b in range(batches):
            if ms is not None:
                batch_bytes = int(p.cim_parallelism) * 4
                read_duration, read_energy_nj = query_sram_cycles_energy(
                    batch_bytes,
                    is_read=True,
                    hw_config=hw,
                    clock_period_ns=hw.clock_period,
                    buffer_name="sort_cache_sram",
                )
                read_power = (read_energy_nj * 1000.0 / (max(1, read_duration) * hw.clock_period)) if read_duration > 0 and hw.clock_period > 0 else 0.0
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="SRAM",
                    stage="merge_sort",
                    sub_stage="sort_input_read",
                    circuit_module="sort_cache_sram",
                    operation="merge_sort_input_read",
                    power=read_power,
                    duration=read_duration,
                    clock_period_ns=hw.clock_period,
                    metadata={"batch": b, "buffer_name": "sort_cache_sram", "buffer_phase": "sort_read", "bytes": batch_bytes, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
                ))
                event_id += 1
                cycle += read_duration
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="FPS",
                    stage="merge_sort",
                    sub_stage="merge_sort_local_max",
                    circuit_module="merge_sort",
                    operation="merge_sort_max",
                    power=ms.power_max_mw,
                    duration=ms.cycles_max,
                    clock_period_ns=hw.clock_period,
                    metadata={"batch": b, "buffer_name": "sort_cache_sram", "buffer_phase": "sort_read", "bytes": int(p.cim_parallelism) * 4, "resource_id": "merge_sort_shared", "exclusive": True, "reset_scope": "batch_boundary"},
                ))
                event_id += 1
                cycle += ms.cycles_max
            if reg is not None:
                read_ev = self._generate_merge_sort_reg_file_read_event(cycle, event_id, num_reads=1)
                if read_ev is not None:
                    events.append(read_ev)
                    event_id += 1
                    cycle += read_ev.duration
                write_ev = self._generate_merge_sort_reg_file_event(cycle, event_id, num_writes=1, overlapped=True)
                if write_ev is not None:
                    events.append(write_ev)
                    event_id += 1
                    # 写与下一批重叠，不推进 cycle
        return events

    def _generate_merge_sort_reg_file_read_event(
        self, cycle: int, event_id: int, num_reads: int = 1, params: Optional[FPSParams] = None
    ) -> Optional[Event]:
        """base_v2：MergeSort 寄存器组读事件。FPS 每批读 1 个历史 max，占关键路径。"""
        if params is not None:
            self.fps_params = params
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
            sub_stage="reg_file_read_max",
            circuit_module="sort_cache_sram",
            operation="reg_file_read",
            power=power,
            duration=duration,
            clock_period_ns=clock_period,
            metadata={"num_reads": num_reads, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": num_bytes, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
        )

    def _generate_merge_sort_reg_file_event(
        self, cycle: int, event_id: int, num_writes: int = 1, overlapped: bool = False,
        params: Optional[FPSParams] = None,
    ) -> Optional[Event]:
        """base_v2：MergeSort 寄存器组写事件。FPS 写 1 个历史 max；overlapped 时与下一批 64选1 重叠。"""
        if params is not None:
            self.fps_params = params
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
            sub_stage="reg_file_write_max",
            circuit_module="sort_cache_sram",
            operation="reg_file_write",
            power=power,
            duration=duration,
            clock_period_ns=clock_period,
            metadata={"num_writes": num_writes, "buffer_name": "sort_cache_sram", "buffer_phase": "write", "bytes": num_bytes, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
        )
        ev.overlapped = overlapped
        return ev

    def generate_single_bitonic_event(self, params: Optional[FPSParams] = None) -> Event:
        """返回单次 FPS 迭代排序事件（cycle=0）。base_v1 用 Bitonic。"""
        if params is not None:
            self.fps_params = params
        p = self.fps_params
        hw = self.hw_config
        bitonic = hw.sorting_circuit.bitonic_sort
        sort_power = bitonic.power_max_mw
        return Event(
            event_id=0,
            cycle=0,
            operator="FPS",
            stage="bitonic_sort",
            sub_stage="bitonic_sort_1024_to_1",
            circuit_module="comp_tree_1024_max",
            operation="bitonic_sort_max",
            power=sort_power,
            duration=p.bitonic_sort_cycles,
            clock_period_ns=hw.clock_period,
            metadata={
                "phase": "fps_engine_bitonic_only",
                "resource_id": "bitonic_shared",
                "exclusive": True,
            },
        )

    def generate_single_fuse_events(self, params: Optional[FPSParams] = None) -> List[Event]:
        """
        返回单次 FPS 迭代的 fuse 模式事件（coarse+fine MAX）。

        acc 模式下使用 coarse_filter + fine_filter 做最远点选取（MAX），
        与 KNN 共用同一套 coarse/fine 硬件，仅比较方向不同（MAX vs MIN）。
        """
        if params is not None:
            self.fps_params = params
        hw = self.hw_config
        coarse_cfg = getattr(hw.sorting_circuit, "coarse_filter", None)
        fine_cfg = getattr(hw.sorting_circuit, "fine_filter", None)
        coarse_power = float(getattr(coarse_cfg, "power", 38.0) if coarse_cfg is not None else 38.0)
        fine_power = float(getattr(fine_cfg, "power", 30.0) if fine_cfg is not None else 30.0)
        coarse_rounds = max(1, int(getattr(hw, "fps_coarse_rounds_mean", 1)))
        explicit_coarse_cycles = int(getattr(coarse_cfg, "cycles", 0) if coarse_cfg is not None else 0)
        coarse_cycles = explicit_coarse_cycles if explicit_coarse_cycles > 0 else (4 * coarse_rounds + 1)
        fine_cycles = int(getattr(fine_cfg, "cycles", 2) if fine_cfg is not None else 2)

        return [
            Event(
                event_id=0,
                cycle=0,
                operator="FPS",
                stage="fuse_filter",
                sub_stage="coarse_filter",
                circuit_module="coarse_filter",
                operation="fps_coarse_filter_max",
                power=coarse_power,
                duration=coarse_cycles,
                clock_period_ns=hw.clock_period,
                metadata={
                    "phase": "fps_engine_fuse",
                    "resource_id": "comparator_shared",
                    "exclusive": True,
                    "coarse_rounds_mean": coarse_rounds,
                },
            ),
            Event(
                event_id=1,
                cycle=coarse_cycles,
                operator="FPS",
                stage="fuse_filter",
                sub_stage="fine_filter",
                circuit_module="fine_filter",
                operation="fps_fine_filter_max",
                power=fine_power,
                duration=fine_cycles,
                clock_period_ns=hw.clock_period,
                metadata={
                    "phase": "fps_engine_fuse",
                    "resource_id": "comparator_shared",
                    "exclusive": True,
                },
            ),
        ]

    def generate_events(self, params: Optional[FPSParams] = None) -> List[Event]:
        """
        Generate cycle-level events for FPS.

        当 hw_config.distance_in_phase2=True 时，仅返回 bitonic 事件（stage5），
        距离计算（stage1-4）由 Phase2 在 dispatcher 中统一插入。
        否则返回完整 5 阶段事件。
        """
        if params is not None:
            self.fps_params = params

        p = self.fps_params
        hw = self.hw_config
        if getattr(hw, "distance_in_phase2", False):
            if getattr(hw, "is_baseline_v2", False):
                return [self._generate_single_merge_sort_event() for _ in range(p.sampled_points_num)]
            return [self.generate_single_bitonic_event() for _ in range(p.sampled_points_num)]

        # 完整 5 阶段：Stage 1–3 复用共用距离计算，Stage 4/5 为 FPS 专属
        mdt_power = (hw.sorting_circuit.min_distance_table or MinDistanceTableConfig()).power
        if getattr(hw, "is_baseline_v2", False) and getattr(hw.sorting_circuit, "merge_sort", None) is not None:
            ms = hw.sorting_circuit.merge_sort
            sort_power = ms.power_max_mw
            sort_cycles = ms.cycles_max  # 22，每迭代局部 max
            sort_stage = "merge_sort"
            sort_circuit = "merge_sort"
        else:
            bitonic = hw.sorting_circuit.bitonic_sort
            sort_power = bitonic.power_max_mw
            sort_cycles = p.bitonic_sort_cycles
            sort_stage = "bitonic_sort"
            sort_circuit = "comp_tree_1024_max"
        clock_period = hw.clock_period
        dist_cycles = self._get_dist_compute_cycles()

        events: List[Event] = []
        event_id = 0
        cycle = 0

        # PointCIM: one-time initialization（rram_write → rram_self_k2）
        # Producer-Consumer: write 产生 rram_content，self_k2 消耗 rram_content 产生 k2_sram_cache
        if getattr(hw, "use_rram_dist", False) and getattr(hw, "rram_dist", None) is not None:
            init_meta = {"input_points": p.input_points, "operator": "FPS", "phase": "init"}
            write_evs, event_id, cycle = gen_rram_dist_write_events(
                "FPS", hw, event_id, cycle, p.input_points, init_meta,
            )
            events.extend(write_evs)
            k2_evs, event_id, cycle = gen_rram_dist_self_k2_events(
                "FPS", hw, event_id, cycle, p.input_points, init_meta,
            )
            events.extend(k2_evs)

        for i in range(p.sampled_points_num):
            candidate_points = p.input_points - i
            batches = math.ceil(candidate_points / p.cim_parallelism)

            for b in range(batches):
                # Stage 1–3（distance）：
                #   baseline → dist2_engine 脉动阵列
                #   acc      → SRAM-CIM k²/-2qk/shift_add
                #   pointcim → rram_vmm(cross)/rram_self_q/rram_sum，input_points 注入 metadata
                batch_m = min(p.cim_parallelism, candidate_points - b * p.cim_parallelism)
                dist_meta = {"iteration": i, "batch": b, "input_points": p.input_points}
                dist_events, event_id, cycle = gen_distance_events_one_batch(
                    "FPS", hw, event_id, cycle, dist_meta,
                    dist_cycles=dist_cycles, batch_size_m=batch_m,
                )
                events.extend(dist_events)
                # Stage 4：FPS 专属 — 最小距离表更新
                table_bytes = int(batch_m) * 4
                if i != 0:
                    table_read_cycles, table_read_energy = query_sram_cycles_energy(
                        table_bytes,
                        is_read=True,
                        hw_config=hw,
                        clock_period_ns=clock_period,
                        buffer_name="fps_min_distance_table_sram",
                    )
                    table_read_power = (table_read_energy * 1000.0 / (max(1, table_read_cycles) * clock_period)) if table_read_cycles > 0 and clock_period > 0 else 0.0
                    events.append(Event(
                        event_id=event_id,
                        cycle=cycle,
                        operator="SRAM",
                        stage="min_distance_update",
                        sub_stage="table_read",
                        circuit_module="fps_min_distance_table_sram",
                        operation="fps_min_distance_table_read",
                        power=table_read_power,
                        duration=table_read_cycles,
                        clock_period_ns=clock_period,
                        metadata={"iteration": i, "batch": b, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "read", "bytes": table_bytes, "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
                    ))
                    event_id += 1
                    cycle += table_read_cycles
                    events.append(Event(
                        event_id=event_id,
                        cycle=cycle,
                        operator="FPS",
                        stage="min_distance_update",
                        sub_stage="table_update",
                        circuit_module="min_distance_table",
                        operation="min_distance_compare",
                        power=mdt_power,
                        duration=1,
                        clock_period_ns=clock_period,
                        metadata={"iteration": i, "batch": b, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "update", "bytes": table_bytes, "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
                    ))
                    event_id += 1
                    cycle += 1
                sram_cycles, sram_energy = query_sram_cycles_energy(
                    table_bytes,
                    is_read=False,
                    hw_config=hw,
                    clock_period_ns=clock_period,
                    buffer_name="fps_min_distance_table_sram",
                )
                sram_power = (sram_energy * 1000.0 / (max(1, sram_cycles) * clock_period)) if sram_cycles > 0 and clock_period > 0 else 0.0
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="SRAM",
                    stage="min_distance_update",
                    sub_stage="table_write",
                    circuit_module="fps_min_distance_table_sram",
                    operation="fps_min_distance_table_write",
                    power=sram_power,
                    duration=sram_cycles,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i, "batch": b, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "update", "bytes": table_bytes, "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
                ))
                event_id += 1
                cycle += sram_cycles

                # base_v2：每批 64选1(merge_sort 22cy) + reg 读 1 + reg 写 1（写与下一批重叠）
                if getattr(hw, "is_baseline_v2", False) and getattr(hw.sorting_circuit, "merge_sort", None) is not None:
                    ms = hw.sorting_circuit.merge_sort
                    events.append(Event(
                        event_id=event_id,
                        cycle=cycle,
                        operator="FPS",
                        stage="merge_sort",
                        sub_stage="merge_sort_local_max",
                        circuit_module="merge_sort",
                        operation="merge_sort_max",
                        power=ms.power_max_mw,
                        duration=ms.cycles_max,
                        clock_period_ns=clock_period,
                        metadata={"iteration": i, "batch": b, "buffer_name": "sort_cache_sram", "buffer_phase": "sort", "bytes": int(batch_m) * 4, "resource_id": "merge_sort_shared", "exclusive": True},
                    ))
                    event_id += 1
                    cycle += ms.cycles_max
                    reg_read_ev = self._generate_merge_sort_reg_file_read_event(cycle, event_id, num_reads=1)
                    if reg_read_ev is not None:
                        events.append(reg_read_ev)
                        event_id += 1
                        cycle += reg_read_ev.duration
                    reg_write_ev = self._generate_merge_sort_reg_file_event(
                        cycle, event_id, num_writes=1, overlapped=True,
                    )
                    if reg_write_ev is not None:
                        events.append(reg_write_ev)
                        event_id += 1
                        # 写与下一批 64选1 重叠，不推进 cycle

            # Stage 5：排序（仅 base_v1 每迭代一次 Bitonic；base_v2 已在每批内 merge_sort+reg）
            if not (getattr(hw, "is_baseline_v2", False) and getattr(hw.sorting_circuit, "merge_sort", None) is not None):
                iter_sort_cycles = self._get_sort_cycles(candidate_points)
                sort_read_cycles, sort_read_energy = query_sram_cycles_energy(
                    int(candidate_points) * 4,
                    is_read=True,
                    hw_config=hw,
                    clock_period_ns=clock_period,
                    buffer_name="fps_min_distance_table_sram",
                )
                sort_read_power = (sort_read_energy * 1000.0 / (max(1, sort_read_cycles) * clock_period)) if sort_read_cycles > 0 and clock_period > 0 else 0.0
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="SRAM",
                    stage="bitonic_sort",
                    sub_stage="sort_input_read",
                    circuit_module="fps_min_distance_table_sram",
                    operation="fps_min_distance_table_sort_read",
                    power=sort_read_power,
                    duration=sort_read_cycles,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i, "buffer_name": "fps_min_distance_table_sram", "buffer_phase": "sort_read", "bytes": int(candidate_points) * 4, "resource_id": "fps_min_table_shared", "exclusive": True, "reset_scope": "fps_iteration"},
                ))
                event_id += 1
                cycle += sort_read_cycles
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="FPS",
                    stage=sort_stage,
                    sub_stage="bitonic_sort_1024_to_1",
                    circuit_module=sort_circuit,
                    operation="bitonic_sort_max",
                    power=sort_power,
                    duration=iter_sort_cycles,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i},
                ))
                event_id += 1
                cycle += iter_sort_cycles

        return events

    # --- Reporting helpers ---

    def get_cycle_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed cycle breakdown for FPS.

        Returns:
            Dictionary with per-batch, first/last iteration, and total info.
        """
        p = self.fps_params
        dist_c = self._get_dist_compute_cycles()
        per_batch = self._get_per_batch_cycles()

        # First iteration (most candidates)
        first_candidates = p.input_points
        first_batches = math.ceil(first_candidates / p.cim_parallelism)
        first_sort = self._get_sort_cycles(first_candidates)
        first_iter = first_batches * per_batch + first_sort

        # Last iteration (fewest candidates)
        last_candidates = p.input_points - (p.sampled_points_num - 1)
        last_batches = math.ceil(last_candidates / p.cim_parallelism)
        last_sort = self._get_sort_cycles(last_candidates)
        last_iter = last_batches * per_batch + last_sort

        total = self.calculate_cycles()

        return {
            'params': {
                'input_points': p.input_points,
                'sampled_points_num': p.sampled_points_num,
                'cim_parallelism': p.cim_parallelism,
                'bit_slice': p.bit_slice,
                'points_per_axis_quant_width': p.points_per_axis_quant_width,
                'bitonic_sort_cycles': p.bitonic_sort_cycles,
            },
            'per_batch': {
                'dist_compute_cycles': dist_c,
                'accumulate_cycles': 1,
                'table_update_cycles': 1,
                'total': per_batch,
            },
            'first_iteration': {
                'candidate_points': first_candidates,
                'batches': first_batches,
                'stages_1_4': first_batches * per_batch,
                'bitonic_sort': first_sort,
                'total': first_iter,
            },
            'last_iteration': {
                'candidate_points': last_candidates,
                'batches': last_batches,
                'stages_1_4': last_batches * per_batch,
                'bitonic_sort': last_sort,
                'total': last_iter,
            },
            'total': {
                'iterations': p.sampled_points_num,
                'cycles': total,
            },
        }
