"""
与 hw_sim/cycle_simulator/memory_model 的对接：DRAM/SRAM（除 SRAM CIM Array 用用户数据外）均通过 CACTI/DRAMSim3 仿真得到 energy 与 cycle，参考 demo_pointcloud_pipeline.py 接入 trace。

- **SRAM（层间 buffer）**：通过 CACTI 仿真（fill_inter_layer_buffer_from_cacti）回填 access_time_ns、read/write_energy_nj_per_access；query_sram_cycles_energy 仅用回填值，未回填时能耗为 0。
- **DRAM（首层加载与 spill）**：通过生成 trace 调用 DRAMSim3（run_dram_trace_for_layer_buffer_dram），按字节分配能耗；未配置 dramsim3_bin 或未跑 trace 时能耗为 0。
- **无带宽功率回退**：所有 LayerBuffer/DRAM 的 energy 与 cycle 仅来自 CACTI/DRAMSim3 调用。
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..configs.hw_config import HardwareConfig


def _ceil_div(a: int, b: int) -> int:
    return max(0, (a + b - 1) // b) if b > 0 else 0


# -----------------------------------------------------------------------------
# 统一 SRAM/DRAM 周期与能耗查询（仅 CACTI/DRAMSim3 回填，无带宽功率回退）
# -----------------------------------------------------------------------------

def query_sram_cycles_energy(
    num_bytes: int,
    is_read: bool,
    hw_config: "HardwareConfig",
    clock_period_ns: Optional[float] = None,
) -> Tuple[int, float]:
    """
    层间 SRAM（InterLayerBuffer）单次读或写的周期与能耗。仅使用 CACTI 仿真回填的 access_time_ns、
    read_energy_nj_per_access、write_energy_nj_per_access；未回填时周期用 ceil(bytes/bandwidth)，能耗为 0。

    Returns:
        (cycles, energy_nj)
    """
    if num_bytes <= 0:
        return 0, 0.0

    clk = float(clock_period_ns if clock_period_ns is not None else getattr(hw_config, "clock_period", 1.0))
    buf = getattr(hw_config, "inter_layer_buffer", None)
    if buf is None:
        return 1, 0.0

    bw = max(1, getattr(buf, "bandwidth_bytes_per_cycle", 256))
    block = max(1, getattr(buf, "block_size_bytes", 64))
    num_accesses = _ceil_div(num_bytes, block)

    access_ns = getattr(buf, "access_time_ns", None)
    read_nj = getattr(buf, "read_energy_nj_per_access", None)
    write_nj = getattr(buf, "write_energy_nj_per_access", None)

    if access_ns is not None and access_ns > 0:
        access_cycles = max(1, int(math.ceil(access_ns / clk)))
        cycles = num_accesses * access_cycles
    else:
        cycles = _ceil_div(num_bytes, bw)

    if is_read and read_nj is not None:
        energy_nj = num_accesses * float(read_nj)
    elif not is_read and write_nj is not None:
        energy_nj = num_accesses * float(write_nj)
    else:
        energy_nj = 0.0

    return max(1, cycles), max(0.0, energy_nj)


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
    DRAM 单次传输（首层加载或 spill）的周期与能耗。仅使用 DRAMSim3 仿真结果（trace 跑完后按字节分配或 per_access 回填）；
    未回填时周期用 ceil(bytes/bandwidth)，能耗为 0。可选参数用于 spill 时覆盖 hw_config.dram。

    Returns:
        (cycles, energy_nj)
    """
    if num_bytes <= 0:
        return 0, 0.0

    dram = getattr(hw_config, "dram", None)
    if bandwidth_bytes_per_cycle is not None:
        bw = max(1, bandwidth_bytes_per_cycle)
    elif dram is not None:
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
    else:
        bw = 32

    cycles = max(1, _ceil_div(num_bytes, bw))

    block = 64
    if dram is not None:
        block = max(1, getattr(dram, "block_size_bytes", 64))
    num_accesses = _ceil_div(num_bytes, block)

    read_nj = write_nj = None
    if dram is not None:
        read_nj = getattr(dram, "read_energy_nj_per_access", None)
        write_nj = getattr(dram, "write_energy_nj_per_access", None)

    if is_read and read_nj is not None:
        energy_nj = num_accesses * float(read_nj)
    elif not is_read and write_nj is not None:
        energy_nj = num_accesses * float(write_nj)
    else:
        energy_nj = 0.0

    return cycles, max(0.0, energy_nj)


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
    buf.block_size_bytes = block
    return True


def run_dram_trace_for_layer_buffer_dram(
    dram_transfers: List[Tuple[int, bool]],
    hw_config: "HardwareConfig",
    repo_root: Optional[str] = None,
    cycle_sim_root: str = "hw_sim/cycle_simulator",
    burst_bytes: Optional[int] = None,
    issue_gap: int = 1,
) -> List[Tuple[int, float]]:
    """
    按 demo_pointcloud_pipeline 方式：为首层加载与 spill 的 DRAM 传输生成 trace，调用 DRAMSim3，
    按字节比例分配 read_energy/write_energy，返回与 dram_transfers 同序的 (cycles, energy_nj) 列表。
    configs/outputs 使用与 demo 一致的 memory_model 目录（get_memory_model_roots 解析）。

    dram_transfers: 与 main 中 DRAM 事件顺序一致，每项 (num_bytes, is_read)。
    repo_root: 项目根目录（PointKAN-pytorch）；None 时尝试从 memory_model 包位置自动解析。
    Returns:
        与 dram_transfers 等长的 [(cycles, energy_nj), ...]，cycles 按 bandwidth 算，energy 来自 DRAMSim3 分配。
        若未配置 dramsim3_bin/config 或仿真失败，返回全 (cycles, 0.0)。
    """
    if not dram_transfers:
        return []
    dram = getattr(hw_config, "dram", None)
    if not dram or not getattr(dram, "dramsim3_bin", None) or not getattr(dram, "dramsim3_config_ini", None):
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32)) if dram else 32
        return [(_ceil_div(b, bw), 0.0) for b, _ in dram_transfers]
    dramsim3_bin = os.path.abspath(dram.dramsim3_bin)
    base_config_path = os.path.abspath(dram.dramsim3_config_ini)
    if not os.path.isfile(dramsim3_bin) or not os.path.isfile(base_config_path):
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
        return [(_ceil_div(b, bw), 0.0) for b, _ in dram_transfers]

    roots = get_memory_model_roots(repo_root, cycle_sim_root)
    if roots is None:
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
        return [(max(1, _ceil_div(b, bw)), 0.0) for b, _ in dram_transfers]
    memory_model_repo, cycle_sim_abs = roots
    config_path = resolve_dramsim3_config_path(hw_config, os.path.join(memory_model_repo, "outputs", "dramsim3", "layer_buffer_dram_cfg")) or base_config_path
    if burst_bytes is None:
        burst_bytes = _parse_dramsim3_burst_bytes(config_path)
    if memory_model_repo not in __import__("sys").path:
        __import__("sys").path.insert(0, memory_model_repo)
    try:
        from memory_model.dram_trace import DRAMTrace
        from memory_model.dramsim3_runner import DRAMSim3Runner
        from memory_model.types import DRAMSim3Input
    except Exception:
        bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
        return [(_ceil_div(b, bw), 0.0) for b, _ in dram_transfers]

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
    os.makedirs(work_root, exist_ok=True)
    runner = DRAMSim3Runner(dramsim3_bin=dramsim3_bin, work_root=work_root)
    with tempfile.NamedTemporaryFile(suffix=".trace", delete=False) as f:
        trace_path = f.name
    try:
        trace.dump(trace_path)
        num_cycles = max(100000, start_cycle + 10000)
        out_dir = os.path.join(work_root, "layer_buffer_dram")
        os.makedirs(out_dir, exist_ok=True)
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

    bw = max(1, getattr(dram, "bandwidth_bytes_per_cycle", 32))
    cycles_list = [max(1, _ceil_div(b, bw)) for b, _ in dram_transfers]
    js = result.json_stats if result.json_stats else {}
    if isinstance(js, dict) and "0" in js and isinstance(js["0"], dict):
        js = js["0"]
    read_pj = float(js.get("read_energy") or 0)
    write_pj = float(js.get("write_energy") or 0)
    total_read_bytes = sum(b for b, is_r in dram_transfers if is_r and b > 0)
    total_write_bytes = sum(b for b, is_r in dram_transfers if not is_r and b > 0)
    energy_nj_list: List[float] = []
    for num_bytes, is_read in dram_transfers:
        if num_bytes <= 0:
            energy_nj_list.append(0.0)
            continue
        if is_read and total_read_bytes > 0:
            energy_nj_list.append((read_pj / 1000.0) * (num_bytes / total_read_bytes))
        elif not is_read and total_write_bytes > 0:
            energy_nj_list.append((write_pj / 1000.0) * (num_bytes / total_write_bytes))
        else:
            energy_nj_list.append(0.0)
    return list(zip(cycles_list, energy_nj_list))
