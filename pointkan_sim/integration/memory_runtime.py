from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .memory_model_adapter import get_dram_trace_parameters, get_memory_model_roots, resolve_dramsim3_config_path

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


@dataclass
class MemoryRuntimeContext:
    """Persistent CACTI/DRAMSim3-backed memory context for one simulation run."""

    memsys: Any
    inter_layer_buffer_name: str
    dramsim3_config_path: str
    runtime_output_dir: str
    dram_burst_bytes: int = 128
    dram_issue_gap: int = 1
    transfer_counter: int = 0
    batch_transfer_results: List[Dict[str, Any]] = field(default_factory=list)
    batch_transfer_cursor: int = 0
    enable_dram_transfer_cache: bool = True
    dram_transfer_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dram_transfer_cache_hits: int = 0
    dram_transfer_cache_misses: int = 0

    def sram_read(self, num_bytes: int, cur_cycle: int):
        return self.memsys.sram_read(self.inter_layer_buffer_name, num_bytes, cur_cycle)

    def sram_write(self, num_bytes: int, cur_cycle: int):
        return self.memsys.sram_write(self.inter_layer_buffer_name, num_bytes, cur_cycle)

    def _run_batched_dram_requests(
        self,
        request_meta: List[Dict[str, Any]],
        *,
        burst_bytes: Optional[int] = None,
        issue_gap: Optional[int] = None,
        dir_prefix: str = "batch",
        store_results: bool = True,
    ) -> List[Dict[str, Any]]:
        from memory_model.dram_trace import DRAMTrace

        self.batch_transfer_results = []
        self.batch_transfer_cursor = 0
        if not request_meta:
            return []

        trace = DRAMTrace()
        if burst_bytes is None:
            burst_bytes = self.dram_burst_bytes
        if issue_gap is None:
            issue_gap = self.dram_issue_gap

        total_read_bytes = 0
        total_write_bytes = 0
        last_issue_cycle = 0
        base_addr = 0x10000000 + self.transfer_counter * 0x01000000

        for req in request_meta:
            num_bytes = max(0, int(req.get("num_bytes", 0) or 0))
            is_read = bool(req.get("is_read", True))
            start_cycle = int(req.get("start_cycle", 0) or 0)
            req["start_cycle"] = start_cycle
            if num_bytes <= 0:
                req["last_issue_cycle"] = start_cycle
                continue
            trace.extend_burst(
                base_addr=base_addr,
                num_bytes=num_bytes,
                burst_bytes=burst_bytes,
                is_write=not is_read,
                start_cycle=start_cycle,
                issue_gap=issue_gap,
            )
            num_txn = (num_bytes + burst_bytes - 1) // burst_bytes
            req_last_issue = start_cycle + max(0, num_txn - 1) * issue_gap
            req["last_issue_cycle"] = req_last_issue
            last_issue_cycle = max(last_issue_cycle, req_last_issue)
            if is_read:
                total_read_bytes += num_bytes
            else:
                total_write_bytes += num_bytes
            base_addr += max(num_bytes, burst_bytes)

        batch_dir = os.path.join(self.runtime_output_dir, f"{dir_prefix}_{self.transfer_counter:05d}")
        os.makedirs(batch_dir, exist_ok=True)
        trace_path = os.path.join(batch_dir, f"{dir_prefix}.trace")
        trace.dump(trace_path)

        sim_cycles = max(100000, last_issue_cycle + 10000)
        result = self.memsys.run_dram_trace(
            config_path=self.dramsim3_config_path,
            trace_path=trace_path,
            num_cycles=sim_cycles,
            output_dir=batch_dir,
        )
        metrics = self.memsys._extract_dram_metrics(result)
        js = result.json_stats if result.json_stats else {}
        if isinstance(js, dict) and "0" in js and isinstance(js["0"], dict):
            js = js["0"]
        read_pj = float(js.get("read_energy") or 0.0)
        write_pj = float(js.get("write_energy") or 0.0)
        act_pj = float(js.get("act_energy") or 0.0)
        total_bytes = total_read_bytes + total_write_bytes

        out: List[Dict[str, Any]] = []
        for req in request_meta:
            num_bytes = max(0, int(req.get("num_bytes", 0) or 0))
            is_read = bool(req.get("is_read", True))
            start_cycle = int(req.get("start_cycle", 0) or 0)
            if num_bytes <= 0:
                out.append({
                    "cycles": 0,
                    "energy_nj": 0.0,
                    "start_cycle": start_cycle,
                    "end_cycle": start_cycle,
                    "metrics": metrics,
                })
                continue
            done_cycle = self.memsys._estimate_refill_done_cycle(int(req["last_issue_cycle"]), result)
            if is_read and total_read_bytes > 0:
                rw_energy_nj = (read_pj / 1000.0) * (num_bytes / total_read_bytes)
            elif (not is_read) and total_write_bytes > 0:
                rw_energy_nj = (write_pj / 1000.0) * (num_bytes / total_write_bytes)
            else:
                rw_energy_nj = 0.0
            # Keep batch replay on the same energy footing as single-transfer replay:
            # dynamic read/write energy stays direction-specific, while ACT energy is
            # apportioned across all requests by byte share.
            act_energy_nj = (act_pj / 1000.0) * (num_bytes / total_bytes) if total_bytes > 0 else 0.0
            energy_nj = rw_energy_nj + act_energy_nj
            out.append({
                "cycles": max(1, done_cycle - start_cycle),
                "energy_nj": float(energy_nj),
                "start_cycle": start_cycle,
                "end_cycle": done_cycle,
                "metrics": metrics,
            })

        self.transfer_counter += 1
        if store_results:
            self.batch_transfer_results = out
        else:
            self.batch_transfer_results = []
            self.batch_transfer_cursor = 0
        return out

    def prime_batch_transfers(self, transfers: List[tuple], *, burst_bytes: Optional[int] = None, issue_gap: Optional[int] = None) -> List[Dict[str, Any]]:
        request_meta: List[Dict[str, Any]] = []
        start_cycle = 0
        burst = self.dram_burst_bytes if burst_bytes is None else burst_bytes
        gap = self.dram_issue_gap if issue_gap is None else issue_gap
        for num_bytes, is_read in transfers:
            request_meta.append({
                "num_bytes": max(0, int(num_bytes)),
                "is_read": bool(is_read),
                "start_cycle": start_cycle,
            })
            if num_bytes > 0:
                num_txn = (int(num_bytes) + int(burst) - 1) // int(burst)
                start_cycle += num_txn * int(gap)
        return self._run_batched_dram_requests(
            request_meta,
            burst_bytes=burst_bytes,
            issue_gap=issue_gap,
            dir_prefix="batch",
            store_results=True,
        )

    def prime_explicit_batch_transfers(
        self,
        requests: List[Dict[str, Any]],
        *,
        burst_bytes: Optional[int] = None,
        issue_gap: Optional[int] = None,
        dir_prefix: str = "transfer_batch",
    ) -> List[Dict[str, Any]]:
        request_meta = [
            {
                "num_bytes": max(0, int(req.get("num_bytes", 0) or 0)),
                "is_read": bool(req.get("is_read", True)),
                "start_cycle": int(req.get("start_cycle", 0) or 0),
            }
            for req in requests
        ]
        return self._run_batched_dram_requests(
            request_meta,
            burst_bytes=burst_bytes,
            issue_gap=issue_gap,
            dir_prefix=dir_prefix,
            store_results=False,
        )

    def next_batch_transfer(self) -> Optional[Dict[str, Any]]:
        if self.batch_transfer_cursor >= len(self.batch_transfer_results):
            return None
        item = self.batch_transfer_results[self.batch_transfer_cursor]
        self.batch_transfer_cursor += 1
        return item

    def dram_transfer(self, num_bytes: int, is_read: bool, cur_cycle: int):
        if self.batch_transfer_cursor < len(self.batch_transfer_results):
            item = self.next_batch_transfer()
            if item is not None:
                return item

        from memory_model.dram_trace import DRAMTrace

        if num_bytes <= 0:
            return {"cycles": 0, "energy_nj": 0.0, "start_cycle": cur_cycle, "end_cycle": cur_cycle}

        burst_bytes = self.dram_burst_bytes
        issue_gap = self.dram_issue_gap

        # Optional memoization: when we fall back to per-transfer DRAMSim3 runs, many
        # transfers repeat (same num_bytes and direction). Since this path already
        # assumes transfers are independent (each trace is replayed in isolation),
        # caching does not change numerical results but can drastically reduce
        # subprocess invocations.
        if self.enable_dram_transfer_cache:
            key = f"{self.dramsim3_config_path}|b{burst_bytes}|g{issue_gap}|n{int(num_bytes)}|r{1 if is_read else 0}"
            cached = self.dram_transfer_cache.get(key)
            if cached is not None:
                self.dram_transfer_cache_hits += 1
                cyc = int(cached.get("cycles", 0) or 0)
                en = float(cached.get("energy_nj", 0.0) or 0.0)
                metrics = cached.get("metrics", None)
                end_cycle = cur_cycle + max(0, cyc)
                out = {"cycles": cyc, "energy_nj": en, "start_cycle": cur_cycle, "end_cycle": end_cycle}
                if metrics is not None:
                    out["metrics"] = metrics
                return out
            self.dram_transfer_cache_misses += 1

        trace = DRAMTrace()
        base_addr = 0x10000000 + self.transfer_counter * 0x01000000
        trace.extend_burst(
            base_addr=base_addr,
            num_bytes=num_bytes,
            burst_bytes=burst_bytes,
            is_write=not is_read,
            start_cycle=cur_cycle,
            issue_gap=issue_gap,
        )

        transfer_dir = os.path.join(self.runtime_output_dir, f"transfer_{self.transfer_counter:05d}")
        os.makedirs(transfer_dir, exist_ok=True)
        trace_path = os.path.join(transfer_dir, "transfer.trace")
        trace.dump(trace_path)

        num_txn = (num_bytes + burst_bytes - 1) // burst_bytes
        last_issue_cycle = cur_cycle + max(0, num_txn - 1) * issue_gap
        sim_cycles = max(100000, last_issue_cycle + 10000)
        result = self.memsys.run_dram_trace(
            config_path=self.dramsim3_config_path,
            trace_path=trace_path,
            num_cycles=sim_cycles,
            output_dir=transfer_dir,
        )

        done_cycle = self.memsys._estimate_refill_done_cycle(last_issue_cycle, result)
        metrics = self.memsys._extract_dram_metrics(result)
        energy_nj = float(metrics.get("dram_energy_nj") or 0.0)

        self.transfer_counter += 1
        out = {
            "cycles": max(1, done_cycle - cur_cycle),
            "energy_nj": energy_nj,
            "start_cycle": cur_cycle,
            "end_cycle": done_cycle,
            "metrics": metrics,
        }
        if self.enable_dram_transfer_cache:
            try:
                key = f"{self.dramsim3_config_path}|b{burst_bytes}|g{issue_gap}|n{int(num_bytes)}|r{1 if is_read else 0}"
                self.dram_transfer_cache[key] = {
                    "cycles": int(out["cycles"]),
                    "energy_nj": float(out["energy_nj"]),
                    "metrics": metrics,
                }
            except Exception:
                pass
        return out

    def summary(self) -> Dict[str, Any]:
        out = self.memsys.summary()
        out["runtime_output_dir"] = self.runtime_output_dir
        out["transfer_counter"] = self.transfer_counter
        out["primed_batch_transfers"] = len(self.batch_transfer_results)
        out["dram_transfer_cache"] = {
            "enabled": bool(self.enable_dram_transfer_cache),
            "entries": len(self.dram_transfer_cache),
            "hits": int(self.dram_transfer_cache_hits),
            "misses": int(self.dram_transfer_cache_misses),
        }
        return out


def create_memory_runtime_context(
    hw_config: "HardwareConfig",
    repo_root: Optional[str],
    *,
    inter_layer_buffer_name: str = "inter_layer_buffer",
) -> Optional[MemoryRuntimeContext]:
    roots = get_memory_model_roots(repo_root)
    if roots is None:
        return None
    memory_model_repo, cycle_sim_abs = roots

    if memory_model_repo not in sys.path:
        sys.path.insert(0, memory_model_repo)

    try:
        from memory_model.cacti_runner import CactiRunner
        from memory_model.dramsim3_runner import DRAMSim3Runner
        from memory_model.memory_system import MemorySystem
        from memory_model.sram_buffer import SRAMBuffer
        from memory_model.types import CactiInput
    except Exception:
        return None

    buf_cfg = getattr(hw_config, "inter_layer_buffer", None)
    dram_cfg = getattr(hw_config, "dram", None)
    if buf_cfg is None or dram_cfg is None:
        return None
    if not getattr(dram_cfg, "dramsim3_bin", None) or not getattr(dram_cfg, "dramsim3_config_ini", None):
        return None

    cacti_runner = CactiRunner(
        cacti_bin=os.path.join(cycle_sim_abs, "cacti", "cacti"),
        template_cfg=os.path.join(cycle_sim_abs, "cacti", "cache.cfg"),
        generated_cfg_dir=os.path.join(memory_model_repo, "configs", "cacti", "generated"),
        output_dir=os.path.join(memory_model_repo, "outputs", "cacti"),
        reuse_if_exists=True,
    )

    size_bytes = max(1, int(getattr(buf_cfg, "size_bytes", 65536) or 65536))
    block_size_bytes = max(1, int(getattr(buf_cfg, "block_size_bytes", 64) or 64))
    num_banks = max(1, int(getattr(buf_cfg, "num_banks", 1) or 1))
    freq_hz = int(round(1e9 / max(1e-6, float(getattr(hw_config, "clock_period", 1.0) or 1.0))))

    inter_layer_sram = SRAMBuffer(
        name=inter_layer_buffer_name,
        cacti_runner=cacti_runner,
        cacti_cfg=CactiInput(
            size_bytes=size_bytes,
            block_size_bytes=block_size_bytes,
            output_bus_width_bits=128,
            cache_type="ram",
            technology_u=0.028,
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=num_banks,
        ),
        freq_hz=freq_hz,
        num_banks=num_banks,
    )

    runtime_output_dir = os.path.join(
        memory_model_repo,
        "outputs",
        "dramsim3",
        "pointkan_runtime",
        f"pid_{os.getpid()}",
    )
    os.makedirs(runtime_output_dir, exist_ok=True)

    resolved_config = resolve_dramsim3_config_path(hw_config, runtime_output_dir)
    if resolved_config is None:
        return None

    dramsim3_runner = DRAMSim3Runner(
        dramsim3_bin=os.path.abspath(dram_cfg.dramsim3_bin),
        work_root=os.path.join(memory_model_repo, "outputs", "dramsim3", "pointkan_runtime"),
    )
    dram_burst_bytes, dram_issue_gap = get_dram_trace_parameters(hw_config, config_path=os.path.abspath(resolved_config))

    ctx = MemoryRuntimeContext(
        memsys=MemorySystem(
            sram_buffers={inter_layer_buffer_name: inter_layer_sram},
            dramsim3_runner=dramsim3_runner,
        ),
        inter_layer_buffer_name=inter_layer_buffer_name,
        dramsim3_config_path=os.path.abspath(resolved_config),
        runtime_output_dir=runtime_output_dir,
        dram_burst_bytes=dram_burst_bytes,
        dram_issue_gap=dram_issue_gap,
    )
    # Default: enable memoization for per-transfer DRAMSim3 fallback.
    # Set DRAM_TRANSFER_CACHE=0 to disable (for strict A/B comparisons).
    v = os.getenv("DRAM_TRANSFER_CACHE", "1").strip().lower()
    ctx.enable_dram_transfer_cache = v not in ("0", "false", "no", "off")
    return ctx
