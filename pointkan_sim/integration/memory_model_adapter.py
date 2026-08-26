"""
与 hw_sim/cycle_simulator/memory_model 的对接：DRAM/SRAM（除 SRAM CIM Array 用用户数据外）均通过 CACTI/DRAMSim3 仿真得到 energy 与 cycle，参考 demo_pointcloud_pipeline.py 接入 trace。

- **SRAM（层间 buffer）**：通过 CACTI 仿真（fill_inter_layer_buffer_from_cacti）回填 access_time_ns、read/write_energy_nj_per_access；query_sram_cycles_energy 仅用回填值。
- **DRAM（首层加载与 spill）**：通过生成 trace 调用 DRAMSim3（run_dram_trace_for_layer_buffer_dram），周期与能耗都来自 DRAMSim3 返回结果。
- **无带宽功率回退**：所有 LayerBuffer/DRAM 的 energy 与 cycle 仅来自 CACTI/DRAMSim3 调用；缺配置时由调用方显式报错。
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig

from .dramsim_utils import parse_dramsim3_burst_bytes as _parse_dramsim3_burst_bytes


def _ceil_div(a: int, b: int) -> int:
    return max(0, (a + b - 1) // b) if b > 0 else 0


def _make_unique_dramsim3_output_dir(work_root: str, prefix: str) -> str:
    os.makedirs(work_root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{prefix}_", dir=work_root)


def _extract_dramsim3_channel_stats(dram_result: Any) -> Dict[str, Any]:
    js_all = getattr(dram_result, "json_stats", None) or {}
    if isinstance(js_all, dict) and "0" in js_all and isinstance(js_all["0"], dict):
        return js_all["0"]
    return js_all if isinstance(js_all, dict) else {}


def _estimate_dramsim3_done_cycle(last_issue_cycle: int, dram_result: Any) -> int:
    js = _extract_dramsim3_channel_stats(dram_result)

    read_latency_map = js.get("read_latency", None)
    if isinstance(read_latency_map, dict) and read_latency_map:
        numeric_keys = []
        for key in read_latency_map.keys():
            try:
                numeric_keys.append(int(key))
            except Exception:
                pass
        if numeric_keys:
            return last_issue_cycle + max(numeric_keys)

    avg_read_latency = js.get("average_read_latency", None)
    if avg_read_latency is not None:
        try:
            return last_issue_cycle + int(math.ceil(float(avg_read_latency)))
        except Exception:
            pass

    return last_issue_cycle + 200


def _strict_dram_cache(hw_config: "HardwareConfig") -> Dict[Tuple[Any, ...], Tuple[int, float]]:
    cache = getattr(hw_config, "_strict_dram_transfer_cache", None)
    if cache is None:
        cache = {}
        setattr(hw_config, "_strict_dram_transfer_cache", cache)
    return cache


def _strict_dram_repo_root(hw_config: "HardwareConfig") -> Optional[str]:
    repo_root = getattr(hw_config, "_pointkan_repo_root", None)
    if repo_root:
        return str(repo_root)
    dram = getattr(hw_config, "dram", None)
    if dram is None:
        return None
    repo_root = getattr(dram, "_pointkan_repo_root", None)
    return str(repo_root) if repo_root else None


def _resolve_dramsim3_runtime(
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
):
    dram = getattr(hw_config, "dram", None)
    if dram is None or not getattr(dram, "dramsim3_bin", None) or not getattr(dram, "dramsim3_config_ini", None):
        return None

    dramsim3_bin = os.path.abspath(dram.dramsim3_bin)
    base_config_path = os.path.abspath(dram.dramsim3_config_ini)
    if not os.path.isfile(dramsim3_bin) or not os.path.isfile(base_config_path):
        return None

    roots = get_memory_model_roots(repo_root, cycle_sim_root)
    if roots is None:
        return None
    memory_model_repo, _cycle_sim_abs = roots
    config_path = resolve_dramsim3_config_path(
        hw_config,
        os.path.join(memory_model_repo, "outputs", "dramsim3", "layer_buffer_dram_cfg"),
    ) or base_config_path
    dram_cfg = _parse_dramsim3_config(config_path)
    tCK_ns = float(dram_cfg["tCK"])
    burst_bytes, issue_gap = get_dram_trace_parameters(hw_config, config_path=config_path)

    if memory_model_repo not in __import__("sys").path:
        __import__("sys").path.insert(0, memory_model_repo)
    try:
        from memory_model.dram_trace import DRAMTrace
        from memory_model.dramsim3_runner import DRAMSim3Runner
        from memory_model.types import DRAMSim3Input
    except Exception:
        return None

    return {
        "dram": dram,
        "memory_model_repo": memory_model_repo,
        "dramsim3_bin": dramsim3_bin,
        "config_path": config_path,
        "tCK_ns": tCK_ns,
        "burst_bytes": int(burst_bytes),
        "issue_gap": int(issue_gap),
        "DRAMTrace": DRAMTrace,
        "DRAMSim3Runner": DRAMSim3Runner,
        "DRAMSim3Input": DRAMSim3Input,
    }


def _run_dramsim3_trace_events(
    trace_events: List[Tuple[int, bool, int]],
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
) -> Optional[Tuple[int, float]]:
    if not trace_events:
        return 0, 0.0

    runtime = _resolve_dramsim3_runtime(hw_config, repo_root, cycle_sim_root)
    if runtime is None:
        return None

    trace = runtime["DRAMTrace"]()
    min_cycle = min(int(cycle) for _, _, cycle in trace_events)
    last_issue_cycle = max(int(cycle) for _, _, cycle in trace_events)
    any_read = False
    any_write = False
    for address, is_write, cycle in trace_events:
        addr = int(address)
        cyc = int(cycle)
        if is_write:
            trace.add_write(addr, cyc)
            any_write = True
        else:
            trace.add_read(addr, cyc)
            any_read = True

    work_root = os.path.join(runtime["memory_model_repo"], "outputs", "dramsim3")
    runner = runtime["DRAMSim3Runner"](dramsim3_bin=runtime["dramsim3_bin"], work_root=work_root)
    with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
        trace_path = f.name
    try:
        trace.dump(trace_path)
        num_cycles = max(100000, last_issue_cycle + 10000)
        out_dir = _make_unique_dramsim3_output_dir(work_root, "explicit_trace")
        result = runner.run(
            runtime["DRAMSim3Input"](
                config_path=runtime["config_path"],
                trace_path=trace_path,
                num_cycles=num_cycles,
                output_dir=out_dir,
            )
        )
    finally:
        try:
            os.unlink(trace_path)
        except Exception:
            pass

    done_cycle = _estimate_dramsim3_done_cycle(last_issue_cycle, result)
    js = _extract_dramsim3_channel_stats(result)
    read_pj = float(js.get("read_energy") or 0.0)
    write_pj = float(js.get("write_energy") or 0.0)
    if any_read and any_write:
        energy_nj = (read_pj + write_pj) * runtime["tCK_ns"] / 1000.0
    elif any_read:
        energy_nj = read_pj * runtime["tCK_ns"] / 1000.0
    else:
        energy_nj = write_pj * runtime["tCK_ns"] / 1000.0
    return max(1, int(done_cycle - min_cycle)), max(0.0, float(energy_nj))


def _normalize_strict_dram_request(req: Dict[str, Any]) -> Tuple[Any, ...]:
    kind = str(req.get("kind", "bytes") or "bytes").strip().lower()
    base = (
        kind,
        bool(req.get("is_read", True)),
        int(req.get("start_cycle", 0) or 0),
    )
    if kind == "matrix":
        return base + (
            int(req.get("rows", 0) or 0),
            int(req.get("cols", 0) or 0),
            int(req.get("element_bytes", 4) or 4),
            bool(req.get("transpose_access", False)),
            int(req.get("row_offset", 0) or 0),
            int(req.get("physical_row_stride_elems", 0) or 0),
        )
    return base + (int(req.get("num_bytes", 0) or 0),)


def _run_strict_dram_request_batch(
    requests: List[Dict[str, Any]],
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
) -> List[Tuple[int, float]]:
    if not requests:
        return []

    runtime = _resolve_dramsim3_runtime(hw_config, repo_root, cycle_sim_root)
    if runtime is None:
        raise RuntimeError("Strict DRAM batch query requires a valid DRAMSim3 runtime.")

    trace = runtime["DRAMTrace"]()
    request_specs: List[Dict[str, Any]] = []
    base_addr = 0x50000000

    for req in requests:
        kind = str(req.get("kind", "bytes") or "bytes").strip().lower()
        start_cycle = int(req.get("start_cycle", 0) or 0)
        is_read = bool(req.get("is_read", True))

        if kind == "matrix":
            rows = max(0, int(req.get("rows", 0) or 0))
            cols = max(0, int(req.get("cols", 0) or 0))
            element_bytes = max(1, int(req.get("element_bytes", 4) or 4))
            total_bytes = rows * cols * element_bytes
            transpose_access = bool(req.get("transpose_access", False))
            if transpose_access:
                trace_events = _build_transposed_burst_trace_events(
                    num_rows=rows,
                    num_cols=cols,
                    element_bytes=element_bytes,
                    burst_bytes=runtime["burst_bytes"],
                    issue_gap=runtime["issue_gap"],
                    row_offset=int(req.get("row_offset", 0) or 0),
                    physical_row_stride_elems=req.get("physical_row_stride_elems", None),
                    is_write=not is_read,
                    base_addr=base_addr,
                )
            else:
                num_txn = _ceil_div(total_bytes, runtime["burst_bytes"])
                trace_events = [
                    (
                        int(base_addr + txn_idx * runtime["burst_bytes"]),
                        not is_read,
                        int(start_cycle + txn_idx * runtime["issue_gap"]),
                    )
                    for txn_idx in range(num_txn)
                ]
            trace_events = [(addr, is_write, cyc + (start_cycle if transpose_access else 0)) for addr, is_write, cyc in trace_events] if transpose_access else trace_events
            last_issue_cycle = max((cyc for _addr, _isw, cyc in trace_events), default=start_cycle)
            span_bytes = max(total_bytes, max(1, len(trace_events)) * int(runtime["burst_bytes"]))
        else:
            total_bytes = max(0, int(req.get("num_bytes", 0) or 0))
            num_txn = _ceil_div(total_bytes, runtime["burst_bytes"])
            trace_events = [
                (
                    int(base_addr + txn_idx * runtime["burst_bytes"]),
                    not is_read,
                    int(start_cycle + txn_idx * runtime["issue_gap"]),
                )
                for txn_idx in range(num_txn)
            ]
            last_issue_cycle = start_cycle + max(0, num_txn - 1) * runtime["issue_gap"]
            span_bytes = max(total_bytes, max(1, num_txn) * int(runtime["burst_bytes"]))

        for addr, is_write, cyc in trace_events:
            if is_write:
                trace.add_write(addr, cyc)
            else:
                trace.add_read(addr, cyc)

        request_specs.append(
            {
                "start_cycle": start_cycle,
                "last_issue_cycle": last_issue_cycle,
                "is_read": is_read,
                "bytes": total_bytes,
            }
        )
        base_addr += max(0x1000, span_bytes)

    work_root = os.path.join(runtime["memory_model_repo"], "outputs", "dramsim3")
    runner = runtime["DRAMSim3Runner"](dramsim3_bin=runtime["dramsim3_bin"], work_root=work_root)
    with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
        trace_path = f.name
    try:
        trace.dump(trace_path)
        max_issue = max(int(spec["last_issue_cycle"]) for spec in request_specs) if request_specs else 0
        num_cycles = max(100000, max_issue + 10000)
        out_dir = _make_unique_dramsim3_output_dir(work_root, "strict_batch")
        result = runner.run(
            runtime["DRAMSim3Input"](
                config_path=runtime["config_path"],
                trace_path=trace_path,
                num_cycles=num_cycles,
                output_dir=out_dir,
            )
        )
    finally:
        try:
            os.unlink(trace_path)
        except Exception:
            pass

    js = _extract_dramsim3_channel_stats(result)
    read_pj = float(js.get("read_energy") or 0.0)
    write_pj = float(js.get("write_energy") or 0.0)
    act_pj = float(js.get("act_energy") or 0.0)
    total_read_bytes = sum(int(spec["bytes"]) for spec in request_specs if spec["is_read"])
    total_write_bytes = sum(int(spec["bytes"]) for spec in request_specs if not spec["is_read"])
    total_bytes = total_read_bytes + total_write_bytes

    out: List[Tuple[int, float]] = []
    for spec in request_specs:
        req_bytes = int(spec["bytes"])
        if req_bytes <= 0:
            out.append((0, 0.0))
            continue
        done_cycle = _estimate_dramsim3_done_cycle(int(spec["last_issue_cycle"]), result)
        if spec["is_read"] and total_read_bytes > 0:
            rw_energy_nj = (read_pj * runtime["tCK_ns"] / 1000.0) * (req_bytes / total_read_bytes)
        elif (not spec["is_read"]) and total_write_bytes > 0:
            rw_energy_nj = (write_pj * runtime["tCK_ns"] / 1000.0) * (req_bytes / total_write_bytes)
        else:
            rw_energy_nj = 0.0
        act_energy_nj = (act_pj * runtime["tCK_ns"] / 1000.0) * (req_bytes / total_bytes) if total_bytes > 0 else 0.0
        out.append((max(1, int(done_cycle - int(spec["start_cycle"]))), max(0.0, rw_energy_nj + act_energy_nj)))
    return out


def _build_transposed_burst_trace_events(
    *,
    num_rows: int,
    num_cols: int,
    element_bytes: int,
    burst_bytes: int,
    issue_gap: int,
    row_offset: int = 0,
    physical_row_stride_elems: Optional[int] = None,
    is_write: bool = False,
    base_addr: int = 0x30000000,
) -> List[Tuple[int, bool, int]]:
    rows = max(0, int(num_rows))
    cols = max(0, int(num_cols))
    elem_bytes = max(1, int(element_bytes))
    if rows <= 0 or cols <= 0:
        return []

    elems_per_burst = max(1, int(burst_bytes) // elem_bytes)
    stride_elems = max(rows + int(row_offset), int(physical_row_stride_elems or rows))
    trace_events: List[Tuple[int, bool, int]] = []
    issue_cycle = 0

    # Common full-matrix case: reading every logical row of a transposed [cols, rows] matrix.
    # First touch of each burst happens once per burst-row group, not once per element.
    if row_offset == 0 and stride_elems == rows:
        groups_per_col = int(math.ceil(rows / float(elems_per_burst)))
        for group_idx in range(groups_per_col):
            r_base = group_idx * elems_per_burst
            for c in range(cols):
                linear_idx = c * stride_elems + r_base
                burst_addr = int(base_addr + (linear_idx * elem_bytes // burst_bytes) * burst_bytes)
                trace_events.append((burst_addr, bool(is_write), issue_cycle))
                issue_cycle += int(issue_gap)
        return trace_events

    seen = set()
    for r in range(rows):
        physical_row = int(row_offset) + r
        for c in range(cols):
            linear_idx = c * stride_elems + physical_row
            burst_addr = int(base_addr + (linear_idx * elem_bytes // burst_bytes) * burst_bytes)
            if burst_addr in seen:
                continue
            seen.add(burst_addr)
            trace_events.append((burst_addr, bool(is_write), issue_cycle))
            issue_cycle += int(issue_gap)
    return trace_events


# -----------------------------------------------------------------------------
# 统一 SRAM/DRAM 周期与能耗查询（仅 CACTI/DRAMSim3 回填，无带宽功率回退）
# -----------------------------------------------------------------------------

def query_sram_cycles_energy(
    num_bytes: int,
    is_read: bool,
    hw_config: "HardwareConfig",
    clock_period_ns: Optional[float] = None,
    *,
    buffer_name: Optional[str] = None,
    buffer_cfg: Optional[Any] = None,
) -> Tuple[int, float]:
    """
    层间 SRAM（InterLayerBuffer）单次读或写的周期与能耗。
    统一要求使用 CACTI 回填的 access_time_ns、read/write_energy_nj_per_access；
    不再允许静默退回带宽估算/零能耗。

    Returns:
        (cycles, energy_nj)
    """
    if num_bytes <= 0:
        return 0, 0.0

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = buffer_cfg
    if buf is None and buffer_name:
        buf = getattr(hw_config, str(buffer_name), None)
    if buf is None:
        buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        raise RuntimeError(
            f"SRAM query requires a configured buffer (buffer_name={buffer_name!r}) before accessing {num_bytes} bytes."
        )

    block = max(1, getattr(buf, "block_size_bytes", 64))
    num_accesses = _ceil_div(num_bytes, block)

    access_ns = getattr(buf, "access_time_ns", None)
    read_nj = getattr(buf, "read_energy_nj_per_access", None)
    write_nj = getattr(buf, "write_energy_nj_per_access", None)
    if access_ns is None or access_ns <= 0:
        raise RuntimeError(
            f"SRAM buffer {buffer_name or getattr(buf, 'name', 'inter_layer_buffer')} is missing CACTI access_time_ns."
        )
    if is_read and read_nj is None:
        raise RuntimeError(
            f"SRAM buffer {buffer_name or getattr(buf, 'name', 'inter_layer_buffer')} is missing CACTI read_energy_nj_per_access."
        )
    if (not is_read) and write_nj is None:
        raise RuntimeError(
            f"SRAM buffer {buffer_name or getattr(buf, 'name', 'inter_layer_buffer')} is missing CACTI write_energy_nj_per_access."
        )

    access_cycles = max(1, int(math.ceil(access_ns / clk)))
    cycles = num_accesses * access_cycles
    energy_nj = num_accesses * float(read_nj if is_read else write_nj)

    return max(1, cycles), max(0.0, energy_nj)


def fill_named_sram_buffers_from_cacti(
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
    buffer_names: Optional[List[str]] = None,
) -> bool:
    """
    Fill dedicated SRAM buffer configs from CACTI using the same memory_model path
    as inter_layer_buffer. Each named buffer must exist on hw_config and expose the
    InterLayerBufferConfig-compatible fields.
    """
    roots = get_memory_model_roots(repo_root, cycle_sim_root)
    if roots is None:
        return False
    memory_model_repo, cycle_sim_abs = roots
    out = _get_memory_model_runner(memory_model_repo, cycle_sim_abs)
    if out is None:
        return False
    runner, CactiInput = out

    names = list(buffer_names or [
        "xyz_sram_buffer",
        "query_xyz_sram_buffer",
        "dist_sram_buffer",
        "self_distance_sram_buffer",
        "fps_min_distance_table_sram",
        "residual_sram_buffer",
        "state_sram_buffer",
        "nonlinear_workspace_sram",
        "nonlinear_param_sram",
        "temp_max_distance_sram",
        "topk_sram_buffer",
        "fps_index_sram_buffer",
        "sort_cache_sram",
        "im2col_buffer",
    ])
    filled = False
    for name in names:
        buf = getattr(hw_config, name, None)
        if buf is None:
            continue
        size_bytes = max(1, int(getattr(buf, "size_bytes", 1024) or 1024))
        # CACTI 对过小或非规则容量更敏感，统一向上规整到不小于 2KB 的 2 次幂容量，
        # 让 query_xyz / xyz / temp_max 这类小而碎的 SRAM 也能稳定拿到回填值。
        cacti_size_bytes = max(2048, 1 << int(math.ceil(math.log2(float(size_bytes)))))
        default_block = 16 if cacti_size_bytes <= 2048 else 64
        block_size_bytes = max(1, int(getattr(buf, "block_size_bytes", min(cacti_size_bytes, default_block)) or min(cacti_size_bytes, default_block)))
        bw_bytes = max(1, int(getattr(buf, "bandwidth_bytes_per_cycle", 64) or 64))
        if cacti_size_bytes >= 4096:
            output_bus_width_bits = 512
        else:
            output_bus_width_bits = min(128, max(8, min(int(bw_bytes * 8), int(cacti_size_bytes * 8))))
        result = None
        candidate_cfgs = [
            (cacti_size_bytes, block_size_bytes, output_bus_width_bits),
            (cacti_size_bytes, block_size_bytes, min(128, output_bus_width_bits)),
            (cacti_size_bytes, min(block_size_bytes, 16), 128),
            (cacti_size_bytes, min(block_size_bytes, 16), 64),
            (max(2048, min(cacti_size_bytes, 4096)), min(block_size_bytes, 64), 128),
            (max(2048, min(cacti_size_bytes, 4096)), min(block_size_bytes, 16), 128),
        ]
        for cand_size, cand_block, cand_bus in candidate_cfgs:
            try:
                result = runner.run(
                    CactiInput(
                        size_bytes=int(cand_size),
                        block_size_bytes=max(1, int(cand_block)),
                        output_bus_width_bits=max(8, int(cand_bus)),
                        cache_type="ram",
                        technology_u=0.028,
                        associativity=1,
                        read_write_port=1,
                        exclusive_read_port=0,
                        exclusive_write_port=0,
                        single_ended_read_ports=0,
                        uca_bank_count=1,
                    )
                )
            except Exception:
                result = None
            if result is not None and result.read_energy_nj is not None and result.write_energy_nj is not None and result.access_time_ns is not None:
                break
        if result is None or result.read_energy_nj is None or result.write_energy_nj is None or result.access_time_ns is None:
            return False
        buf.read_energy_nj_per_access = result.read_energy_nj
        buf.write_energy_nj_per_access = result.write_energy_nj
        buf.access_time_ns = result.access_time_ns
        if getattr(result, "area_mm2", None) is not None:
            buf.area = float(result.area_mm2)
        buf.block_size_bytes = block_size_bytes
        filled = True
    return filled


def query_dram_cycles_energy(
    num_bytes: int,
    is_read: bool,
    hw_config: "HardwareConfig",
    clock_period_ns: Optional[float] = None,
    *,
    bandwidth_bytes_per_cycle: Optional[int] = None,
    read_power_mw: Optional[float] = None,
    write_power_mw: Optional[float] = None,
) -> Tuple[int, float]:
    """
    DRAM 单次传输的周期与能耗。

    统一要求使用 DRAMSim3 trace 返回的严格周期与能耗；
    不再允许静默退回 bytes/bandwidth 近似。

    Returns:
        (cycles, energy_nj)
    """
    if num_bytes <= 0:
        return 0, 0.0

    dram = getattr(hw_config, "dram", None)
    if dram is None:
        raise RuntimeError("DRAM query requires hw_config.dram to be configured.")
    if not getattr(dram, "dramsim3_bin", None) or not getattr(dram, "dramsim3_config_ini", None):
        raise RuntimeError(
            f"Strict DRAM query requires dramsim3_bin/config_ini before querying {num_bytes} bytes."
        )

    repo_root = _strict_dram_repo_root(hw_config)
    cache = _strict_dram_cache(hw_config)
    cache_key = (
        os.path.abspath(str(getattr(dram, "dramsim3_config_ini", ""))),
        int(getattr(dram, "dramsim3_channels_override", 0) or 0),
        int(num_bytes),
        bool(is_read),
        repo_root,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    results = run_dram_trace_for_layer_buffer_dram(
        [(int(num_bytes), bool(is_read))],
        hw_config,
        repo_root,
        "hw_sim/cycle_simulator",
    )
    if not results or len(results) != 1:
        raise RuntimeError(
            f"Strict DRAM query failed to obtain DRAMSim3 latency for {num_bytes} bytes."
        )
    cycles, energy_nj = results[0]
    out = (max(1, int(cycles)), max(0.0, float(energy_nj)))
    cache[cache_key] = out
    return out


def query_dram_matrix_cycles_energy(
    num_rows: int,
    num_cols: int,
    *,
    element_bytes: int,
    is_read: bool,
    hw_config: "HardwareConfig",
    clock_period_ns: Optional[float] = None,
    transpose_access: bool = False,
    row_offset: int = 0,
    physical_row_stride_elems: Optional[int] = None,
) -> Tuple[int, float]:
    """
    Strict DRAMSim3 query for matrix transfers.

    - transpose_access=False: treat the matrix as contiguous row-major bytes.
    - transpose_access=True: generate a burst-level trace that reflects logical row-wise
      traversal over a physically transposed layout.
    """
    rows = max(0, int(num_rows))
    cols = max(0, int(num_cols))
    elem_bytes = max(1, int(element_bytes))
    if rows <= 0 or cols <= 0:
        return 0, 0.0

    total_bytes = rows * cols * elem_bytes
    if not transpose_access:
        return query_dram_cycles_energy(
            total_bytes,
            is_read=is_read,
            hw_config=hw_config,
            clock_period_ns=clock_period_ns,
        )

    repo_root = _strict_dram_repo_root(hw_config)
    runtime = _resolve_dramsim3_runtime(hw_config, repo_root, "hw_sim/cycle_simulator")
    if runtime is None:
        raise RuntimeError(
            f"Strict DRAM matrix query requires DRAMSim3 runtime before querying {rows}x{cols}x{elem_bytes}."
        )

    elems_per_burst = max(1, runtime["burst_bytes"] // elem_bytes)
    cache = _strict_dram_cache(hw_config)
    cache_key = (
        "matrix",
        os.path.abspath(str(runtime["config_path"])),
        int(getattr(runtime["dram"], "dramsim3_channels_override", 0) or 0),
        rows,
        cols,
        elem_bytes,
        bool(is_read),
        bool(transpose_access),
        int(physical_row_stride_elems or rows),
        int(row_offset) % elems_per_burst,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    trace_events = _build_transposed_burst_trace_events(
        num_rows=rows,
        num_cols=cols,
        element_bytes=elem_bytes,
        burst_bytes=runtime["burst_bytes"],
        issue_gap=runtime["issue_gap"],
        row_offset=int(row_offset),
        physical_row_stride_elems=physical_row_stride_elems,
        is_write=not is_read,
    )
    result = _run_dramsim3_trace_events(
        trace_events,
        hw_config,
        repo_root,
        "hw_sim/cycle_simulator",
    )
    if result is None:
        raise RuntimeError(
            f"Strict DRAM matrix query failed to obtain DRAMSim3 latency for {rows}x{cols}x{elem_bytes}."
        )
    cache[cache_key] = result
    return result


def query_dram_batch_cycles_energy(
    requests: List[Dict[str, Any]],
    hw_config: "HardwareConfig",
    clock_period_ns: Optional[float] = None,
) -> List[Tuple[int, float]]:
    if not requests:
        return []

    dram = getattr(hw_config, "dram", None)
    if dram is None:
        raise RuntimeError("DRAM batch query requires hw_config.dram to be configured.")
    if not getattr(dram, "dramsim3_bin", None) or not getattr(dram, "dramsim3_config_ini", None):
        raise RuntimeError("Strict DRAM batch query requires dramsim3_bin/config_ini.")

    repo_root = _strict_dram_repo_root(hw_config)
    cache = _strict_dram_cache(hw_config)
    normalized = tuple(_normalize_strict_dram_request(req) for req in requests)
    cache_key = (
        "batch",
        os.path.abspath(str(getattr(dram, "dramsim3_config_ini", ""))),
        int(getattr(dram, "dramsim3_channels_override", 0) or 0),
        repo_root,
        normalized,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return list(cached)

    out = _run_strict_dram_request_batch(
        requests,
        hw_config,
        repo_root,
        "hw_sim/cycle_simulator",
    )
    cache[cache_key] = tuple(out)
    return out


def query_dram_strided_cycles_energy(
    num_elements: int,
    element_bytes: int,
    stride_bytes: int,
    is_read: bool,
    hw_config: "HardwareConfig",
    clock_period_ns: Optional[float] = None,
) -> Tuple[int, float]:
    """
    非连续（跨步）DRAM 访问的 cycle/energy 估算。

    用于距离矩阵转置读取：SA 存 dist[Q, N] 行主序，FP 按列读取时
    每个元素间隔 stride_bytes = N * element_bytes，远大于 DRAM page size，
    导致每次访问都是 row buffer miss。

    Row buffer hit:  tCAS + burst_cycles
    Row buffer miss: tRP + tRCD + tCAS + burst_cycles

    当 stride >= row_size 时认为 100% miss；
    当 stride < row_size 时按 burst 内连续判断 hit 率。
    """
    if num_elements <= 0:
        return 0, 0.0

    dram = getattr(hw_config, "dram", None)
    if dram is None:
        raise RuntimeError("Strided DRAM query requires hw_config.dram to be configured.")

    row_size = max(1, getattr(dram, "row_size_bytes", 2048))
    tRCD = max(0, getattr(dram, "tRCD_cycles", 17))
    tRP = max(0, getattr(dram, "tRP_cycles", 17))
    tCAS = max(0, getattr(dram, "tCAS_cycles", 17))
    burst_cy = max(1, getattr(dram, "burst_cycles", 4))
    block = max(1, getattr(dram, "block_size_bytes", 64))

    hit_latency = tCAS + burst_cy
    miss_latency = tRP + tRCD + tCAS + burst_cy

    if stride_bytes >= row_size:
        cycles_per_access = miss_latency
    elif stride_bytes <= block:
        bursts = _ceil_div(num_elements * element_bytes, block)
        first_miss = miss_latency
        hit_total = max(0, bursts - 1) * hit_latency
        cycles = first_miss + hit_total
        nj = _strided_energy(num_elements, element_bytes, block, dram, is_read)
        return max(1, cycles), max(0.0, nj)
    else:
        elements_per_row = max(1, row_size // stride_bytes)
        bursts_per_group = max(1, elements_per_row)
        num_groups = _ceil_div(num_elements, bursts_per_group)
        cycles = num_groups * miss_latency + max(0, num_elements - num_groups) * hit_latency
        nj = _strided_energy(num_elements, element_bytes, block, dram, is_read)
        return max(1, cycles), max(0.0, nj)

    total_cycles = num_elements * cycles_per_access
    nj = _strided_energy(num_elements, element_bytes, block, dram, is_read)
    return max(1, total_cycles), max(0.0, nj)


def _strided_energy(
    num_elements: int,
    element_bytes: int,
    block_size: int,
    dram: Any,
    is_read: bool,
) -> float:
    """Helper: per-access energy for strided DRAM access."""
    num_accesses = max(1, num_elements)
    if is_read:
        nj_per = getattr(dram, "read_energy_nj_per_access", None)
    else:
        nj_per = getattr(dram, "write_energy_nj_per_access", None)
    if nj_per is None:
        kind = "read" if is_read else "write"
        raise RuntimeError(
            f"Strided DRAM query requires DRAMSim3-derived {kind}_energy_nj_per_access on hw_config.dram."
        )
    return num_accesses * float(nj_per)


def query_dram_issue_cycles(
    num_bytes: int,
    hw_config: "HardwareConfig",
) -> int:
    """
    DRAM 命令总线占用周期数（issue window）。
    = ceil(num_bytes / burst_bytes) * issue_gap。
    burst_bytes 尽量从 DRAMSim3 config 派生，issue_gap 来自 hw_config.dram，
    与实际 trace 生成保持一致。
    """
    if num_bytes <= 0:
        return 0
    burst_bytes, gap = get_dram_trace_parameters(hw_config)
    return max(1, _ceil_div(num_bytes, burst_bytes) * gap)


# -----------------------------------------------------------------------------
# repo_root 与 demo_pointcloud_pipeline 对齐
# -----------------------------------------------------------------------------
# - demo 中 repo_root = dirname(dirname(__file__)) = memory_model 目录（即 hw_sim/cycle_simulator/memory_model），
#   configs/outputs/traces 均在该目录下；cycle_sim_root = dirname(repo_root) = cycle_simulator。
# - pointkan_sim 调用方传入的 repo_root 约定为「项目根目录」（PointKAN-pytorch），此处统一推导出
#   memory_model_repo = repo_root/hw_sim/cycle_simulator/memory_model，与 demo 的 config/output 路径一致。
# - 原始 memory_model 管理路径：<项目根>/hw_sim/cycle_simulator/memory_model/
# -----------------------------------------------------------------------------


def get_memory_model_roots(
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
) -> Optional[Tuple[str, str]]:
    """
    解析 memory_model 的 repo 根目录与 cycle_simulator 绝对路径，与 demo_pointcloud_pipeline 的路径约定一致。

    - 若传入 repo_root（项目根，即含 hw_sim 的 PointKAN-pytorch 根）：
      memory_model_repo = repo_root/hw_sim/cycle_simulator/memory_model，
      cycle_sim_abs = repo_root/hw_sim/cycle_simulator。
    - 若 repo_root 为 None：尝试从 import memory_model 反推（即从已安装或 sys.path 中的 memory_model 包位置
      得到其所在目录，再上一层为 memory_model_repo，再上一层为 cycle_sim_abs）。

    Returns:
        (memory_model_repo, cycle_sim_abs) 或 None（无法解析时）。
    """
    if repo_root and os.path.isdir(repo_root):
        repo_root_abs = os.path.abspath(repo_root)
        cycle_sim_abs = os.path.abspath(os.path.join(repo_root_abs, cycle_sim_root))
        memory_model_repo = os.path.join(cycle_sim_abs, "memory_model")
        if os.path.isdir(memory_model_repo):
            return memory_model_repo, cycle_sim_abs
    try:
        import memory_model as _mm
        pkg_dir = os.path.dirname(os.path.abspath(getattr(_mm, "__file__", "")))
        if not pkg_dir:
            return None
        memory_model_repo = os.path.dirname(pkg_dir)
        cycle_sim_abs = os.path.dirname(memory_model_repo)
        if os.path.isdir(memory_model_repo) and os.path.isdir(cycle_sim_abs):
            return memory_model_repo, cycle_sim_abs
    except Exception:
        pass
    return None


def resolve_dramsim3_config_path(
    hw_config: "HardwareConfig",
    output_dir: str,
) -> Optional[str]:
    """Return a DRAMSim3 config path, creating a temporary override file when needed."""
    dram = getattr(hw_config, "dram", None)
    if dram is None:
        return None
    base_config = getattr(dram, "dramsim3_config_ini", None)
    if not base_config:
        return None
    base_config = os.path.abspath(base_config)
    if not os.path.isfile(base_config):
        return None

    channels_override = getattr(dram, "dramsim3_channels_override", None)
    if channels_override is None:
        return base_config

    try:
        channels_override = max(1, int(channels_override))
    except Exception:
        return base_config

    os.makedirs(output_dir, exist_ok=True)
    override_path = os.path.join(output_dir, f"dramsim3_channels_{channels_override}.ini")
    try:
        lines = open(base_config, "r", encoding="utf-8").read().splitlines()
    except Exception:
        return base_config

    out_lines = []
    in_system = False
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_system = stripped.lower() == "[system]"
        if in_system and stripped.startswith("channels"):
            prefix = line.split("=")[0] if "=" in line else "channels "
            out_lines.append(f"{prefix}= {channels_override}")
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        out_lines.append("[system]")
        out_lines.append(f"channels = {channels_override}")

    with open(override_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return override_path


def _parse_dramsim3_config(config_path: str) -> dict:
    """
    Parse DRAMSim3 INI config to extract key parameters.

    Returns dict with:
      - request_size_bytes: bus_width/8 × BL (bytes per DRAM request, used as burst_bytes for trace)
      - tCK: clock period in ns (needed to convert mW·cycles energy to pJ)
      - bus_width, device_width, BL: raw config values

    DRAMSim3 energy values are in mW·cycles (V × mA × cycles). To get pJ,
    multiply by tCK. The source comment claiming "exactly pJ" is only correct
    when tCK = 1 ns.  See configuration.cc line 200 vs simple_stats.cc line 407
    where average_power = total_energy / num_cycles (not / (num_cycles × tCK)).
    """
    parsed: dict = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("["):
                    continue
                if "=" not in stripped:
                    continue
                key, _, val = stripped.partition("=")
                key = key.strip()
                val = val.strip().split("#")[0].strip()
                if key == "BL":
                    parsed["BL"] = int(val)
                elif key == "device_width":
                    parsed["device_width"] = int(val)
                elif key == "bus_width":
                    parsed["bus_width"] = int(val)
                elif key == "tCK":
                    parsed["tCK"] = float(val)
    except Exception:
        pass

    bl = parsed.get("BL", 8)
    bus_width = parsed.get("bus_width", 64)
    tCK = parsed.get("tCK", 1.0)
    request_size_bytes = max(1, bus_width * bl // 8)

    return {
        "request_size_bytes": request_size_bytes,
        "tCK": tCK,
        "bus_width": bus_width,
        "device_width": parsed.get("device_width", 16),
        "BL": bl,
    }


def get_dram_trace_parameters(hw_config: "HardwareConfig", *, config_path: Optional[str] = None) -> Tuple[int, int]:
    """
    Return (burst_bytes, issue_gap) used by PointKAN DRAM trace generation.
    burst_bytes is derived from the DRAMSim3 config when available; issue_gap is taken from hw_config.dram.
    """
    dram = getattr(hw_config, "dram", None)
    issue_gap = max(1, int(getattr(dram, "issue_gap", 1) or 1)) if dram is not None else 1
    cfg_path = config_path
    if not cfg_path and dram is not None:
        cfg_path = getattr(dram, "dramsim3_config_ini", None)
    if cfg_path:
        try:
            burst_bytes = max(1, int(_parse_dramsim3_burst_bytes(str(cfg_path))))
            return burst_bytes, issue_gap
        except Exception:
            pass
    block = max(1, int(getattr(dram, "block_size_bytes", 64) or 64)) if dram is not None else 64
    return block, issue_gap


def _get_memory_model_runner(
    memory_model_repo: str,
    cycle_sim_abs: str,
):
    """可选导入 memory_model，返回 (CactiRunner, CactiInput 类) 或 None。configs/outputs 使用 memory_model_repo（与 demo 一致）。"""
    try:
        # Match the demo import path: add the memory_model repo root to sys.path.
        if memory_model_repo not in __import__("sys").path:
            __import__("sys").path.insert(0, memory_model_repo)
        from memory_model.cacti_runner import CactiRunner
        from memory_model.types import CactiInput
        cacti_bin = os.path.join(cycle_sim_abs, "cacti", "cacti")
        template_cfg = os.path.join(cycle_sim_abs, "cacti", "cache.cfg")
        generated_cfg_dir = os.path.join(memory_model_repo, "configs", "cacti", "generated")
        output_dir = os.path.join(memory_model_repo, "outputs", "cacti")
        if not os.path.isfile(cacti_bin) or not os.path.isfile(template_cfg):
            return None
        os.makedirs(generated_cfg_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        runner = CactiRunner(
            cacti_bin=cacti_bin,
            template_cfg=template_cfg,
            generated_cfg_dir=generated_cfg_dir,
            output_dir=output_dir,
            reuse_if_exists=True,
        )
        return runner, CactiInput
    except Exception:
        return None


def fill_buffers_from_cacti(
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
    input_buffer_size_bytes: int = 4096,
    output_buffer_size_bytes: int = 2048,
    block_size_bytes: int = 16,
    freq_hz: int = 1_000_000_000,
) -> bool:
    """
    使用 memory_model 的 CACTI 对 input_buffer / output_buffer 查表，将 read_energy_nj、
    write_energy_nj、access_time_ns 回填到 hw_config.sram_cim_macro 的 buffer 配置中。

    Args:
        hw_config: 当前硬件配置，会被原地更新。
        repo_root: 项目根目录（PointKAN-pytorch）；None 时尝试从 memory_model 包位置自动解析（与 demo 的 memory_model 目录一致）。
        cycle_sim_root: 相对 repo_root 的 cycle_simulator 目录。
        input_buffer_size_bytes: 输入 buffer 容量（字节），与 memory_model demo 中 xyz_buffer 一致时可取 4096。
        output_buffer_size_bytes: 输出 buffer 容量（字节），与 dist_buffer 一致时可取 2048。
        block_size_bytes: CACTI block size。
        freq_hz: 时钟频率，用于将 access_time_ns 转为 cycles。

    Returns:
        True 若成功回填至少一个 buffer，否则 False（保留原固定功耗）。
    """
    roots = get_memory_model_roots(repo_root, cycle_sim_root)
    if roots is None:
        return False
    memory_model_repo, cycle_sim_abs = roots
    out = _get_memory_model_runner(memory_model_repo, cycle_sim_abs)
    if out is None:
        return False
    runner, CactiInput = out

    filled = False
    for name, size_bytes, target in [
        ("input_buffer", input_buffer_size_bytes, "input_buffer"),
        ("output_buffer", output_buffer_size_bytes, "output_buffer"),
    ]:
        cfg = CactiInput(
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
            uca_bank_count=2,
        )
        try:
            result = runner.run(cfg)
        except Exception:
            continue
        if result.read_energy_nj is None or result.write_energy_nj is None or result.access_time_ns is None:
            continue
        buf = getattr(hw_config.sram_cim_macro, target, None)
        if buf is None:
            continue
        buf.read_energy_nj_per_access = result.read_energy_nj
        buf.write_energy_nj_per_access = result.write_energy_nj
        buf.access_time_ns = result.access_time_ns
        if getattr(buf, "cycles", None) is not None and result.access_time_ns > 0 and freq_hz > 0:
            buf.cycles = max(1, int(round(result.access_time_ns * freq_hz / 1e9)))
        filled = True
    return filled


def fill_dram_per_access_from_dramsim3(
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
) -> bool:
    """
    Run DRAMSim3 calibration traces and back out read/write_energy_nj_per_access on hw_config.dram.
    We use a sufficiently large transfer so DRAMSim3 reports stable read/write energy, then divide by
    the number of DRAM accesses at the configured block granularity.
    """
    dram = getattr(hw_config, "dram", None)
    if dram is None or not getattr(dram, "dramsim3_bin", None) or not getattr(dram, "dramsim3_config_ini", None):
        return False
    if repo_root:
        setattr(hw_config, "_pointkan_repo_root", repo_root)
        setattr(dram, "_pointkan_repo_root", repo_root)

    block = max(1, int(getattr(dram, "block_size_bytes", 64) or 64))
    calibration_bytes = max(4096, block * 64)
    num_accesses = max(1, _ceil_div(calibration_bytes, block))

    read_stats = run_dram_trace_for_layer_buffer_dram([(calibration_bytes, True)], hw_config, repo_root, cycle_sim_root)
    write_stats = run_dram_trace_for_layer_buffer_dram([(calibration_bytes, False)], hw_config, repo_root, cycle_sim_root)
    if not read_stats or not write_stats:
        return False
    read_energy_total = float(read_stats[0][1] or 0.0)
    write_energy_total = float(write_stats[0][1] or 0.0)
    if read_energy_total <= 0 or write_energy_total <= 0:
        return False
    dram.read_energy_nj_per_access = read_energy_total / float(num_accesses)
    dram.write_energy_nj_per_access = write_energy_total / float(num_accesses)
    return True


def fill_inter_layer_buffer_from_cacti(
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
    size_bytes: Optional[int] = None,
    block_size_bytes: int = 64,
) -> bool:
    """
    使用 memory_model 的 CACTI 对层间 buffer（inter_layer_buffer）查表，将 read_energy_nj_per_access、
    write_energy_nj_per_access、access_time_ns 回填到 hw_config.inter_layer_buffer。
    与 demo_pointcloud_pipeline 的 CACTI 调用方式一致；回填后由本模块 query_sram_cycles_energy 按访问级模型计算周期与能耗。

    Args:
        hw_config: 当前硬件配置，会被原地更新。
        repo_root: 项目根目录（PointKAN-pytorch）；None 时尝试从 memory_model 包位置自动解析。
        cycle_sim_root: 相对 repo_root 的 cycle_simulator 目录。
        size_bytes: 层间 buffer 容量（字节）；None 时使用 hw_config.inter_layer_buffer.size_bytes。
        block_size_bytes: CACTI 访问粒度，回填到 inter_layer_buffer.block_size_bytes。

    Returns:
        True 若成功回填，否则 False。
    """
    roots = get_memory_model_roots(repo_root, cycle_sim_root)
    if roots is None:
        return False
    memory_model_repo, cycle_sim_abs = roots
    out = _get_memory_model_runner(memory_model_repo, cycle_sim_abs)
    if out is None:
        return False
    runner, CactiInput = out

    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return False
    sz = size_bytes if size_bytes is not None else max(1, getattr(buf, "size_bytes", 65536))
    block = max(1, block_size_bytes)

    cfg = CactiInput(
        size_bytes=sz,
        block_size_bytes=block,
        output_bus_width_bits=128,
        cache_type="ram",
        technology_u=0.028,
        associativity=1,
        read_write_port=1,
        exclusive_read_port=0,
        exclusive_write_port=0,
        single_ended_read_ports=0,
        uca_bank_count=2,
    )
    try:
        result = runner.run(cfg)
    except Exception:
        return False
    if result.read_energy_nj is None or result.write_energy_nj is None or result.access_time_ns is None:
        return False
    buf.read_energy_nj_per_access = result.read_energy_nj
    buf.write_energy_nj_per_access = result.write_energy_nj
    buf.access_time_ns = result.access_time_ns
    if getattr(result, "area_mm2", None) is not None:
        buf.area = float(result.area_mm2)
    buf.block_size_bytes = block
    return True


def run_dram_trace_for_layer_buffer_dram(
    dram_transfers: List[Tuple[int, bool]],
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
    burst_bytes: Optional[int] = None,
    issue_gap: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    按 demo_pointcloud_pipeline 方式：为首层加载与 spill 的 DRAM 传输生成 trace，调用 DRAMSim3，
    按字节比例分配 read_energy/write_energy，返回与 dram_transfers 同序的 (cycles, energy_nj) 列表。
    configs/outputs 使用与 demo 一致的 memory_model 目录（get_memory_model_roots 解析）。

    dram_transfers: 与 main 中 DRAM 事件顺序一致，每项 (num_bytes, is_read)。
    repo_root: 项目根目录（PointKAN-pytorch）；None 时尝试从 memory_model 包位置自动解析。
    Returns:
        与 dram_transfers 等长的 [(cycles, energy_nj), ...]，cycles 与 energy 均来自 DRAMSim3。
        若未配置 dramsim3_bin/config 或仿真失败，返回空列表。
    """
    if not dram_transfers:
        return []
    dram = getattr(hw_config, "dram", None)
    if not dram or not getattr(dram, "dramsim3_bin", None) or not getattr(dram, "dramsim3_config_ini", None):
        return []
    dramsim3_bin = os.path.abspath(dram.dramsim3_bin)
    base_config_path = os.path.abspath(dram.dramsim3_config_ini)
    if not os.path.isfile(dramsim3_bin) or not os.path.isfile(base_config_path):
        return []

    roots = get_memory_model_roots(repo_root, cycle_sim_root)
    if roots is None:
        return []
    memory_model_repo, cycle_sim_abs = roots
    config_path = resolve_dramsim3_config_path(hw_config, os.path.join(memory_model_repo, "outputs", "dramsim3", "layer_buffer_dram_cfg")) or base_config_path
    dram_cfg = _parse_dramsim3_config(config_path)
    tCK_ns = dram_cfg["tCK"]
    derived_burst_bytes, derived_issue_gap = get_dram_trace_parameters(hw_config, config_path=config_path)
    if burst_bytes is None:
        burst_bytes = derived_burst_bytes
    if issue_gap is None:
        issue_gap = derived_issue_gap
    if memory_model_repo not in __import__("sys").path:
        __import__("sys").path.insert(0, memory_model_repo)
    try:
        from memory_model.dram_trace import DRAMTrace
        from memory_model.dramsim3_runner import DRAMSim3Runner
        from memory_model.types import DRAMSim3Input
    except Exception:
        return []

    trace = DRAMTrace()
    base_addr = 0x10000000
    start_cycle = 0
    for num_bytes, is_read in dram_transfers:
        if num_bytes <= 0:
            continue
        trace.extend_burst(
            base_addr=base_addr,
            num_bytes=num_bytes,
            burst_bytes=burst_bytes,
            is_write=not is_read,
            start_cycle=start_cycle,
            issue_gap=issue_gap,
        )
        num_txn = _ceil_div(num_bytes, burst_bytes)
        start_cycle += num_txn * issue_gap
        base_addr += num_bytes

    work_root = os.path.join(memory_model_repo, "outputs", "dramsim3")
    runner = DRAMSim3Runner(dramsim3_bin=dramsim3_bin, work_root=work_root)
    with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
        trace_path = f.name
    try:
        trace.dump(trace_path)
        num_cycles = max(100000, start_cycle + 10000)
        out_dir = _make_unique_dramsim3_output_dir(work_root, "layer_buffer_dram")
        result = runner.run(
            DRAMSim3Input(
                config_path=config_path,
                trace_path=trace_path,
                num_cycles=num_cycles,
                output_dir=out_dir,
            )
        )
    except Exception:
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
        return [(max(1, _ceil_div(b, bw)), 0.0) for b, _ in dram_transfers]
    finally:
        try:
            os.unlink(trace_path)
        except Exception:
            pass

    js = result.json_stats if result.json_stats else {}
    if isinstance(js, dict) and "0" in js and isinstance(js["0"], dict):
        js = js["0"]
    read_mw_cy = float(js.get("read_energy") or 0)
    write_mw_cy = float(js.get("write_energy") or 0)
    read_nj = read_mw_cy * tCK_ns / 1000.0
    write_nj = write_mw_cy * tCK_ns / 1000.0
    total_read_bytes = sum(b for b, is_r in dram_transfers if is_r and b > 0)
    total_write_bytes = sum(b for b, is_r in dram_transfers if not is_r and b > 0)
    energy_nj_list: List[float] = []
    cycles_list: List[int] = []
    start_cycle = 0
    for num_bytes, is_read in dram_transfers:
        if num_bytes <= 0:
            cycles_list.append(0)
            energy_nj_list.append(0.0)
            continue
        num_txn = _ceil_div(num_bytes, burst_bytes)
        last_issue_cycle = start_cycle + max(0, num_txn - 1) * issue_gap
        done_cycle = _estimate_dramsim3_done_cycle(last_issue_cycle, result)
        cycles_list.append(max(1, done_cycle - start_cycle))
        start_cycle += num_txn * issue_gap
        if is_read and total_read_bytes > 0:
            energy_nj_list.append(read_nj * (num_bytes / total_read_bytes))
        elif not is_read and total_write_bytes > 0:
            energy_nj_list.append(write_nj * (num_bytes / total_write_bytes))
        else:
            energy_nj_list.append(0.0)
    return list(zip(cycles_list, energy_nj_list))
