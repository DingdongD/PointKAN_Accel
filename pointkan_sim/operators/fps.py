"""
FPS Operator Model
==================

Farthest Point Sampling (FPS) using SRAM-CIM processing.

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
from ..configs.hw_config import CircuitModule
from ..simulator.event import Event

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


@dataclass
class FPSParams(OperatorParams):
    """
    FPS-specific parameters.

    Inherits from OperatorParams:
        input_points: Total input points (N), default 1024

    Additional attributes:
        sampled_points_num: Number of points to sample (FPS iterations)
        cim_parallelism: CIM array parallelism per batch (default 128)
        bit_slice: Bit-slice width for bit-serial computation (default 2)
        points_per_axis_quant_width: Quantization bit width per coordinate axis
        bitonic_sort_cycles: Fixed cycle count for bitonic sort stage
    """
    sampled_points_num: int = 512
    cim_parallelism: int = 128
    bit_slice: int = 2
    points_per_axis_quant_width: int = 28
    bitonic_sort_cycles: int = 2


class FPSOperator(BaseOperator):
    """
    Farthest Point Sampling (FPS) Operator.

    SRAM-CIM based FPS with fixed parallelism and bitonic sort.

    Processing flow (per iteration, one per sampled point):

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

    4. Min-distance table update
       Compare and update per-point minimum distance.
       Duration = 1 cycle.
       Circuit modules: COMPARATOR, BUFFER

    [Stage 5: runs once after all batches complete]

    5. Bitonic sort (1024->1)
       Find the farthest point (max distance).
       Duration = bitonic_sort_cycles (fixed constant).
       Circuit modules: COMPARATOR, BUFFER
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
            circuit_modules=[CircuitModule.COMP_TREE_1024_MAX],
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
            total_cycles += batches * per_batch + p.bitonic_sort_cycles

        return total_cycles

    # --- Energy calculation ---

    def calculate_energy(self, params: Optional[FPSParams] = None) -> float:
        """Calculate total energy for FPS (nJ). Computed from generated events."""
        events = self.generate_events(params)
        return sum(e.energy for e in events)

    # --- Event generation ---

    def generate_single_bitonic_event(self, params: Optional[FPSParams] = None) -> Event:
        """返回单次 FPS 迭代排序事件（cycle=0）。"""
        if params is not None:
            self.fps_params = params
        p = self.fps_params
        hw = self.hw_config
        sort_power = hw.sorting_circuit.comp_tree_1024_max.power
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
                # baseline 下与 KNN 共用 bitonic，比对器互斥由 Scheduler 统一处理
                "resource_id": "bitonic_shared",
                "exclusive": True,
            },
        )

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
            return [self.generate_single_bitonic_event() for _ in range(p.sampled_points_num)]

        # 完整 5 阶段
        cim_power = hw.sram_cim_macro.array.cim_power
        shift_add_power = hw.sram_cim_macro.shift_add.power
        mdt_power = hw.sram_cim_macro.min_distance_table.power
        sort_power = hw.sorting_circuit.comp_tree_1024_max.power
        clock_period = hw.clock_period

        dist_cycles = self._get_dist_compute_cycles()

        events: List[Event] = []
        event_id = 0
        cycle = 0

        for i in range(p.sampled_points_num):
            candidate_points = p.input_points - i
            batches = math.ceil(candidate_points / p.cim_parallelism)

            for b in range(batches):
                # Stage 1: k^2
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="FPS",
                    stage="k_square",
                    sub_stage="k_square_compute",
                    circuit_module="sram_cim_array",
                    operation="k2_bit_serial",
                    power=cim_power,
                    duration=dist_cycles,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i, "batch": b},
                ))
                event_id += 1
                cycle += dist_cycles

                # Stage 2: -2qk
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="FPS",
                    stage="cross_term",
                    sub_stage="cross_term_compute",
                    circuit_module="sram_cim_array",
                    operation="two_qk_bit_serial",
                    power=cim_power,
                    duration=dist_cycles,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i, "batch": b},
                ))
                event_id += 1
                cycle += dist_cycles

                # Stage 3: accumulate
                events.append(Event(
                    event_id=event_id,
                    cycle=cycle,
                    operator="FPS",
                    stage="accumulate",
                    sub_stage="shift_add_accumulate",
                    circuit_module="shift_add",
                    operation="distance_accumulate",
                    power=shift_add_power,
                    duration=1,
                    clock_period_ns=clock_period,
                    metadata={"iteration": i, "batch": b},
                ))
                event_id += 1
                cycle += 1

                # Stage 4: min-distance table update
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
                    metadata={"iteration": i, "batch": b},
                ))
                event_id += 1
                cycle += 1

            # Stage 5: bitonic sort (once per iteration, after all batches)
            events.append(Event(
                event_id=event_id,
                cycle=cycle,
                operator="FPS",
                stage="bitonic_sort",
                sub_stage="bitonic_sort_1024_to_1",
                circuit_module="comp_tree_1024_max",
                operation="bitonic_sort_max",
                power=sort_power,
                duration=p.bitonic_sort_cycles,
                clock_period_ns=clock_period,
                metadata={"iteration": i},
            ))
            event_id += 1
            cycle += p.bitonic_sort_cycles

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
        first_iter = first_batches * per_batch + p.bitonic_sort_cycles

        # Last iteration (fewest candidates)
        last_candidates = p.input_points - (p.sampled_points_num - 1)
        last_batches = math.ceil(last_candidates / p.cim_parallelism)
        last_iter = last_batches * per_batch + p.bitonic_sort_cycles

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
                'bitonic_sort': p.bitonic_sort_cycles,
                'total': first_iter,
            },
            'last_iteration': {
                'candidate_points': last_candidates,
                'batches': last_batches,
                'stages_1_4': last_batches * per_batch,
                'bitonic_sort': p.bitonic_sort_cycles,
                'total': last_iter,
            },
            'total': {
                'iterations': p.sampled_points_num,
                'cycles': total,
            },
        }
