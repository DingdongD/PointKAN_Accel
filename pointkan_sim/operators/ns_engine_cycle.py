"""
Cycle-accurate Neighbor Search (NS) Engine, inspired by FusionArch.

Models DC (Distance Calculator) + SA (Sorter Array) pipeline with DRAM
request/response semantics. Produces a list of ``Event`` objects compatible
with the event-driven simulator.

Two operating modes:
  - ``compute``:  DC reads centroid/neighbor xyz from DRAM, computes L2
                  distance, SA inserts into top-k.
  - ``reuse``:    DC reads a pre-computed distance row/column from DRAM
                  (distance_matrix_reuse), SA inserts into top-k.

Usage::

    engine = NSEngineCycleAccurate(hw_config, knn_params)
    events = engine.run()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..simulator.event import Event
from ..integration.memory_model_adapter import (
    query_dram_cycles_energy,
    query_dram_matrix_cycles_energy,
    query_dram_strided_cycles_energy,
)

__all__ = ["NSEngineCycleAccurate"]


@dataclass
class _DRAMRequest:
    """Internal DRAM request descriptor."""
    addr_tag: str           # human-readable tag (e.g. "centroid_42_xyz")
    issue_cycle: int
    latency: int            # cycles until response
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _DCState:
    """Per-DC (Distance Calculator) register state."""
    busy: bool = False
    centroid_id: int = -1
    centroid_ready: bool = False
    neighbor_ptr: int = 0       # next neighbor to scan
    total_neighbors: int = 0
    pending_reads: int = 0      # outstanding DRAM reads for this DC
    dc_latency_counter: int = 0


class NSEngineCycleAccurate:
    """
    Cycle-accurate NS engine producing Event list.

    Parameters are extracted from ``KNNParams`` and ``HardwareConfig``.
    """

    def __init__(
        self,
        hw_config: Any,
        knn_params: Any,
        *,
        num_dcs: int = 0,
        num_sas: int = 0,
    ):
        self.hw = hw_config
        self.p = knn_params

        self.query_points = self.p.query_points
        self.input_points = self.p.input_points
        self.k = self.p.k

        self.distance_reuse = getattr(self.p, "distance_reuse", False)
        self.is_transpose_read = getattr(self.p, "is_transpose_read", False)
        self.global_dist_cols = getattr(self.p, "global_dist_cols", 0) or self.input_points

        n = num_dcs if num_dcs > 0 else getattr(hw_config, "ns_num_dcs", 4)
        self.num_dcs = n
        self.num_sas = num_sas if num_sas > 0 else n

        self.dc_pipeline_interval = 3
        dram_cfg = getattr(hw_config, "dram", None)
        self.dram_latency = 50
        if dram_cfg is not None:
            tRCD = getattr(dram_cfg, "tRCD_cycles", 17)
            tCAS = getattr(dram_cfg, "tCAS_cycles", 17)
            burst = getattr(dram_cfg, "burst_cycles", 4)
            self.dram_latency = tRCD + tCAS + burst

        self.clock_ns = getattr(hw_config, "clock_period", 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[Event]:
        """Run full cycle-accurate simulation and return Event list."""
        if self.distance_reuse:
            return self._run_reuse_mode()
        return self._run_compute_mode()

    # ------------------------------------------------------------------
    # Compute mode (full distance calculation)
    # ------------------------------------------------------------------

    def _run_compute_mode(self) -> List[Event]:
        dcs = [_DCState(total_neighbors=self.input_points) for _ in range(self.num_dcs)]
        pending_centroids = list(range(self.query_points))
        in_flight: List[_DRAMRequest] = []
        events: List[Event] = []
        eid = 0
        clk = 0
        completed = 0
        max_cycles = self.query_points * self.input_points * (self.dram_latency + 10) + 1000

        while completed < self.query_points and clk < max_cycles:
            # 1. Process DRAM responses
            still_pending = []
            for req in in_flight:
                if clk >= req.issue_cycle + req.latency:
                    dc_idx = req.payload.get("dc", -1)
                    kind = req.payload.get("kind", "")
                    if 0 <= dc_idx < len(dcs):
                        dcs[dc_idx].pending_reads -= 1
                        if kind == "centroid" and dcs[dc_idx].pending_reads <= 0:
                            dcs[dc_idx].centroid_ready = True
                else:
                    still_pending.append(req)
            in_flight = still_pending

            # 2. Per-DC logic
            for dc_idx, dc in enumerate(dcs):
                # Allocate centroid
                if not dc.busy and pending_centroids:
                    cid = pending_centroids.pop(0)
                    dc.busy = True
                    dc.centroid_id = cid
                    dc.centroid_ready = False
                    dc.neighbor_ptr = 0
                    dc.pending_reads = 1
                    dc.dc_latency_counter = 0
                    in_flight.append(_DRAMRequest(
                        addr_tag=f"centroid_{cid}_xyz",
                        issue_cycle=clk,
                        latency=self.dram_latency,
                        payload={"dc": dc_idx, "kind": "centroid"},
                    ))
                    ev = Event(
                        event_id=eid, cycle=clk, operator="DRAM",
                        stage="ns_cycle_centroid_load",
                        sub_stage=f"dc{dc_idx}_centroid{cid}",
                        circuit_module="dram",
                        operation="centroid_xyz_read",
                        power=0.0, duration=self.dram_latency,
                        clock_period_ns=self.clock_ns,
                        metadata={"query": cid, "dc": dc_idx, "kind": "centroid_load"},
                    )
                    events.append(ev)
                    eid += 1

                # Issue neighbor reads
                if dc.busy and dc.centroid_ready and dc.pending_reads <= 0 and dc.neighbor_ptr < dc.total_neighbors:
                    if dc.dc_latency_counter <= 0:
                        nidx = dc.neighbor_ptr
                        dc.pending_reads = 1
                        in_flight.append(_DRAMRequest(
                            addr_tag=f"neighbor_{dc.centroid_id}_{nidx}_xyz",
                            issue_cycle=clk,
                            latency=self.dram_latency,
                            payload={"dc": dc_idx, "kind": "neighbor", "nidx": nidx, "cid": dc.centroid_id},
                        ))
                        dc.neighbor_ptr += 1
                        dc.dc_latency_counter = self.dc_pipeline_interval
                    else:
                        dc.dc_latency_counter -= 1

                # Check completion
                if dc.busy and dc.neighbor_ptr >= dc.total_neighbors and dc.pending_reads <= 0:
                    cid = dc.centroid_id
                    sort_dur = self._get_sort_duration()
                    sort_power = self._get_sort_power()
                    ev = Event(
                        event_id=eid, cycle=clk, operator="KNN",
                        stage="ns_cycle_sort",
                        sub_stage=f"dc{dc_idx}_sort_q{cid}",
                        circuit_module="sorting",
                        operation="top_k_sort",
                        power=sort_power, duration=sort_dur,
                        clock_period_ns=self.clock_ns,
                        metadata={"query": cid, "dc": dc_idx, "kind": "sort_topk"},
                    )
                    events.append(ev)
                    eid += 1
                    dc.busy = False
                    dc.centroid_id = -1
                    dc.centroid_ready = False
                    completed += 1

            clk += 1

        return events

    # ------------------------------------------------------------------
    # Reuse mode (read pre-computed distances)
    # ------------------------------------------------------------------

    def _run_reuse_mode(self) -> List[Event]:
        """
        Reuse mode: two-phase model.
        Phase A: bulk DRAM read of the distance sub-matrix (shared across all queries).
        Phase B: per-query sort from on-chip buffer, multiple DCs in parallel.

        transpose_store (from hw_config) flips the is_transpose_read flag:
        FP layers become contiguous reads, SA layers become strided reads.
        """
        events: List[Event] = []
        eid = 0

        transpose_store = bool(getattr(self.hw, "distance_matrix_transpose_store", False))
        is_transpose = self.is_transpose_read
        if transpose_store:
            is_transpose = not is_transpose

        total_bytes = self.query_points * self.input_points * 4
        if is_transpose:
            bulk_dur, bulk_nj = query_dram_matrix_cycles_energy(
                self.query_points,
                self.input_points,
                element_bytes=4,
                is_read=True,
                hw_config=self.hw,
                clock_period_ns=self.clock_ns,
                transpose_access=True,
            )
        else:
            bulk_dur, bulk_nj = query_dram_cycles_energy(
                total_bytes, is_read=True, hw_config=self.hw, clock_period_ns=self.clock_ns,
            )
        bulk_dur = max(1, int(bulk_dur))

        bulk_power = (bulk_nj * 1e3 / (bulk_dur * self.clock_ns)) if bulk_dur > 0 and self.clock_ns > 0 else 0.0
        ev_bulk = Event(
            event_id=eid, cycle=0, operator="DRAM",
            stage="ns_cycle_dist_reuse_read",
            sub_stage="bulk_matrix_read",
            circuit_module="dram",
            operation="dist_reuse_dram_read",
            power=bulk_power, duration=bulk_dur,
            clock_period_ns=self.clock_ns,
            metadata={
                "phase": "distance_reuse",
                "transpose_store": transpose_store,
                "access_pattern": "bulk_transpose" if is_transpose else "bulk_contiguous",
                "total_bytes": total_bytes,
            },
        )
        events.append(ev_bulk)
        eid += 1

        sort_dur = self._get_sort_duration()
        sort_power = self._get_sort_power()
        dc_free_at = [bulk_dur] * self.num_dcs

        for q in range(self.query_points):
            dc_idx = min(range(self.num_dcs), key=lambda i: dc_free_at[i])
            start = dc_free_at[dc_idx]

            ev_sort = Event(
                event_id=eid, cycle=start, operator="KNN",
                stage="ns_cycle_sort",
                sub_stage=f"dc{dc_idx}_sort_q{q}",
                circuit_module="sorting",
                operation="top_k_sort",
                power=sort_power, duration=sort_dur,
                clock_period_ns=self.clock_ns,
                metadata={"query": q, "dc": dc_idx, "kind": "sort_topk"},
            )
            events.append(ev_sort)
            eid += 1
            dc_free_at[dc_idx] = start + sort_dur

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_sort_duration(self) -> int:
        sc = getattr(self.hw, "sorting_circuit", None)
        if sc is None:
            return 19
        ms = getattr(sc, "merge_sort", None)
        if getattr(self.hw, "is_baseline_v2", False) and ms is not None:
            batches = math.ceil(self.input_points / getattr(self.p, "cim_parallelism", 128))
            return max(1, batches * getattr(ms, "cycles_top32", 22))
        bt = getattr(sc, "bitonic_sort", None)
        if bt is not None:
            return max(1, getattr(self.p, "bitonic_sort_cycles", 19))
        return 19

    def _get_sort_power(self) -> float:
        sc = getattr(self.hw, "sorting_circuit", None)
        if sc is None:
            return 1540.0
        ms = getattr(sc, "merge_sort", None)
        if getattr(self.hw, "is_baseline_v2", False) and ms is not None:
            return float(getattr(ms, "power_top32_mw", 312.0))
        bt = getattr(sc, "bitonic_sort", None)
        if bt is not None:
            return float(getattr(bt, "power_min_mw", 1540.0))
        return 1540.0
