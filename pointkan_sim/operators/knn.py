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
from .distance_ops import gen_distance_events_one_batch

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


@dataclass
class KNNParams(OperatorParams):
    """
    KNN 专用参数：继承 input_points（搜索空间点数）；额外包括 query_points、k、
    sampled_points_num、cim_parallelism、bit_slice、points_per_axis_quant_width、bitonic_sort_cycles。
    """
    query_points: int = 512
    k: int = 24
    sampled_points_num: int = 512
    cim_parallelism: int = 128
    bit_slice: int = 2
    points_per_axis_quant_width: int = 28
    bitonic_sort_cycles: int = 19   # MIN 模式（KNN 1024→k）：19 周期


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
        per_query = batches * per_batch + p.bitonic_sort_cycles
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
        duration = num_reads * getattr(reg, "cycles_per_read", 1)
        return Event(
            event_id=event_id,
            cycle=cycle,
            operator="KNN",
            stage="merge_sort_reg_file",
            sub_stage="reg_file_read_32min",
            circuit_module="merge_sort_reg_file",
            operation="reg_file_read",
            power=reg.power_mw,
            duration=duration,
            clock_period_ns=hw.clock_period,
            metadata={"num_reads": num_reads, "buffer_name": "sort_cache_sram", "buffer_phase": "read", "bytes": int(num_reads) * 4, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
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
        duration = num_writes * reg.cycles_per_write
        ev = Event(
            event_id=event_id,
            cycle=cycle,
            operator="KNN",
            stage="merge_sort_reg_file",
            sub_stage="reg_file_write_32min",
            circuit_module="merge_sort_reg_file",
            operation="reg_file_write",
            power=reg.power_mw,
            duration=duration,
            clock_period_ns=hw.clock_period,
            metadata={"num_writes": num_writes, "buffer_name": "sort_cache_sram", "buffer_phase": "write", "bytes": int(num_writes) * 4, "resource_id": "sort_cache_shared", "exclusive": True, "reset_scope": "batch_boundary"},
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
        coarse_rounds = max(1, int(getattr(hw, "knn_coarse_rounds_mean", 1)))
        coarse_cycles = 4 * coarse_rounds + 1
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

    def generate_events(self, params: Optional[KNNParams] = None) -> List[Event]:
        """
        Generate cycle-level events for KNN.

        当 hw_config.distance_in_phase2=True 时，仅返回 bitonic 事件（stage5），
        距离计算（stage1-4）由 Phase2 在 dispatcher 中统一插入。
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
        ob = hw.sram_cim_macro.output_buffer
        write_power = ob.get_write_power_mw(hw.clock_period) if hasattr(ob, "get_write_power_mw") else getattr(ob, "power", 0.5)
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
            sort_cycles_total = p.bitonic_sort_cycles
            sort_stage = "bitonic_sort"
            sort_circuit = "comp_bitonic_1024_min32"
        clock_period = hw.clock_period
        dist_cycles = self._get_dist_compute_cycles()

        events: List[Event] = []
        event_id = 0
        cycle = 0

        for q in range(p.query_points):
            for b in range(batches):
                # Stage 1–3：共用距离计算（baseline=dist2_engine 脉动，acc=SRAM-CIM k²/-2qk/shift_add）
                batch_m = min(p.cim_parallelism, p.input_points - b * p.cim_parallelism)
                dist_events, event_id, cycle = gen_distance_events_one_batch(
                    "KNN", hw, event_id, cycle, {"query": q, "batch": b},
                    dist_cycles=dist_cycles, batch_size_m=batch_m,
                )
                events.extend(dist_events)
                # Stage 4：KNN 专属 — 距离写回 output_buffer
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="KNN",
                    stage="sram_write",
                    sub_stage="distance_write",
                    circuit_module="output_buffer",
                    operation="sram_write",
                    power=write_power,
                    duration=1,
                    clock_period_ns=clock_period,
                    metadata={"query": q, "batch": b, "buffer_name": "dist_sram_buffer", "buffer_phase": "write", "bytes": int(batch_m) * 4, "resource_id": "dist_sram_shared", "exclusive": True, "reset_scope": "query_boundary"},
                ))
                event_id += 1
                cycle += 1

            # base_v2：reg_read(32) 再 merge_sort，再 reg_write(32, overlapped)
            if getattr(hw, "is_baseline_v2", False) and getattr(hw.sorting_circuit, "merge_sort", None) is not None:
                reg_read_ev = self._generate_merge_sort_reg_file_read_event(cycle, event_id, num_reads=32)
                if reg_read_ev is not None:
                    events.append(reg_read_ev)
                    event_id += 1
                    cycle += reg_read_ev.duration
            # Stage 5：排序（每 query 一次；base_v1=Bitonic，base_v2=MergeSort 每批 top32 后归并）
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
        per_query = stages_1_4_cycles + p.bitonic_sort_cycles
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
                'bitonic_sort': p.bitonic_sort_cycles,
                'total': per_query,
            },
            'total': {
                'queries': p.query_points,
                'cycles': total,
            },
        }
