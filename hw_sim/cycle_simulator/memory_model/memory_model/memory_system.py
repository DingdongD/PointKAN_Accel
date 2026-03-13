# import json
# import math
# import os
# from typing import Dict, Any, Optional

# from .sram_buffer import SRAMBuffer
# from .dramsim3_runner import DRAMSim3Runner
# from .types import DRAMSim3Input
# from .dram_trace import DRAMTrace
# from .pointcloud_trace_gen import TensorRegion


# class MemorySystem:
#     def __init__(self, sram_buffers: Dict[str, SRAMBuffer], dramsim3_runner: DRAMSim3Runner):
#         self.sram_buffers = sram_buffers
#         self.dramsim3_runner = dramsim3_runner

#         # record all refill events for debug / analysis
#         self.refill_history = []

#     # ---------------------------------------------------------
#     # SRAM direct access
#     # ---------------------------------------------------------
#     def sram_read(self, name: str, num_bytes: int, cur_cycle: int):
#         return self.sram_buffers[name].read(num_bytes, cur_cycle)

#     def sram_write(self, name: str, num_bytes: int, cur_cycle: int):
#         return self.sram_buffers[name].write(num_bytes, cur_cycle)

#     # ---------------------------------------------------------
#     # DRAM trace execution
#     # ---------------------------------------------------------
#     def run_dram_trace(self, config_path: str, trace_path: str, num_cycles: int, output_dir: str):
#         return self.dramsim3_runner.run(
#             DRAMSim3Input(
#                 config_path=config_path,
#                 trace_path=trace_path,
#                 num_cycles=num_cycles,
#                 output_dir=output_dir,
#             )
#         )

#     def _run_single_refill_trace(
#         self,
#         config_path: str,
#         tensor_region: TensorRegion,
#         start_elem: int,
#         num_elem: int,
#         burst_bytes: int,
#         start_cycle: int,
#         issue_gap: int,
#         output_dir: str,
#         trace_name: str,
#     ):
#         """
#         Generate and run a single DRAM refill trace for a continuous tensor slice.
#         """
#         trace = DRAMTrace()

#         start_addr = tensor_region.addr_of(start_elem)
#         total_bytes = num_elem * tensor_region.bytes_per_element

#         trace.extend_burst(
#             base_addr=start_addr,
#             num_bytes=total_bytes,
#             burst_bytes=burst_bytes,
#             is_write=False,
#             start_cycle=start_cycle,
#             issue_gap=issue_gap,
#         )

#         os.makedirs(output_dir, exist_ok=True)
#         trace_path = os.path.join(output_dir, f"{trace_name}.trace")
#         trace.dump(trace_path)

#         num_txn = (total_bytes + burst_bytes - 1) // burst_bytes
#         sim_cycles = max(100000, start_cycle + num_txn * max(1, issue_gap) + 10000)

#         result = self.run_dram_trace(
#             config_path=config_path,
#             trace_path=trace_path,
#             num_cycles=sim_cycles,
#             output_dir=output_dir,
#         )

#         return trace, result
#     def _estimate_refill_done_cycle(
#         self,
#         start_cycle: int,
#         total_bytes: int,
#         burst_bytes: int,
#         issue_gap: int,
#         dram_result,
#     ) -> int:
#         """
#         Estimate when a DRAM refill is completed, using local DRAMsim3 JSON fields.

#         Priority:
#         1) Use max read latency bucket if available
#         2) Fall back to average_read_latency
#         3) Fall back to conservative constant
#         """
#         num_txn = (total_bytes + burst_bytes - 1) // burst_bytes
#         last_issue_cycle = start_cycle + max(0, num_txn - 1) * issue_gap

#         js_all = dram_result.json_stats if dram_result.json_stats else {}

#         # Your DRAMsim3 JSON is of form:
#         # {
#         #   "0": { ... channel stats ... }
#         # }
#         if "0" in js_all and isinstance(js_all["0"], dict):
#             js = js_all["0"]
#         else:
#             js = js_all

#         # ---------------------------------------------------------
#         # Preferred: max read latency from detailed histogram
#         # Example:
#         # "read_latency": {"49":1,"56":1,...,"98":1}
#         # ---------------------------------------------------------
#         read_latency_map = js.get("read_latency", None)
#         if isinstance(read_latency_map, dict) and len(read_latency_map) > 0:
#             numeric_keys = []
#             for k in read_latency_map.keys():
#                 try:
#                     numeric_keys.append(int(k))
#                 except Exception:
#                     pass
#             if numeric_keys:
#                 max_read_latency = max(numeric_keys)
#                 return last_issue_cycle + max_read_latency

#         # ---------------------------------------------------------
#         # Fallback: average read latency
#         # ---------------------------------------------------------
#         avg_read_latency = js.get("average_read_latency", None)
#         if avg_read_latency is not None:
#             try:
#                 return last_issue_cycle + math.ceil(float(avg_read_latency))
#             except Exception:
#                 pass

#         # ---------------------------------------------------------
#         # Last fallback: conservative constant
#         # ---------------------------------------------------------
#         return last_issue_cycle + 200


#     # ---------------------------------------------------------
#     # Refill logic
#     # ---------------------------------------------------------
#     def refill_if_needed(
#         self,
#         buffer_name: str,
#         tensor_region: TensorRegion,
#         start_elem: int,
#         num_elem: int,
#         cur_cycle: int,
#         dram_config_path: str,
#         output_dir: str,
#         burst_bytes: int = 64,
#         issue_gap: int = 1,
#     ) -> int:
#         buf = self.sram_buffers[buffer_name]

#         if buf.contains(tensor_region.name, start_elem, num_elem, cur_cycle):
#             return cur_cycle

#         total_bytes = num_elem * tensor_region.bytes_per_element
#         if not buf.can_hold(total_bytes):
#             raise RuntimeError(
#                 f"{buffer_name} cannot hold refill bytes={total_bytes}, "
#                 f"capacity={buf.capacity_bytes}"
#             )

#         trace_name = (
#             f"refill_{buffer_name}_{tensor_region.name}_"
#             f"{start_elem}_{start_elem + num_elem}_cyc{cur_cycle}"
#         )

#         trace, dram_result = self._run_single_refill_trace(
#             config_path=dram_config_path,
#             tensor_region=tensor_region,
#             start_elem=start_elem,
#             num_elem=num_elem,
#             burst_bytes=burst_bytes,
#             start_cycle=cur_cycle,
#             issue_gap=issue_gap,
#             output_dir=output_dir,
#             trace_name=trace_name,
#         )

#         dram_done_cycle = self._estimate_refill_done_cycle(
#             start_cycle=cur_cycle,
#             total_bytes=total_bytes,
#             burst_bytes=burst_bytes,
#             issue_gap=issue_gap,
#             dram_result=dram_result,
#         )

#         write_stat = buf.write(total_bytes, dram_done_cycle)
#         ready_cycle = write_stat.end_cycle

#         buf.fill(
#             region_name=tensor_region.name,
#             start_elem=start_elem,
#             num_elem=num_elem,
#             ready_cycle=ready_cycle,
#         )

        

#         read_latency_map = {}
#         avg_read_latency = None
#         max_read_latency = None
#         row_hit_rate = None
#         num_read_row_hits = 0
#         num_read_cmds = 0
#         num_reads_done = 0
#         num_act_cmds = 0
#         num_pre_cmds = 0
#         dram_energy_nj = None

#         js_all = dram_result.json_stats if dram_result.json_stats else {}
#         js = js_all.get("0", js_all) if isinstance(js_all, dict) else {}

#         if isinstance(js, dict):
#             read_latency_map = js.get("read_latency", {})
#             avg_read_latency = js.get("average_read_latency", None)

#             num_read_row_hits = js.get("num_read_row_hits", 0)
#             num_read_cmds = js.get("num_read_cmds", 0)
#             num_reads_done = js.get("num_reads_done", 0)
#             num_act_cmds = js.get("num_act_cmds", 0)
#             num_pre_cmds = js.get("num_pre_cmds", 0)

#             if num_read_cmds > 0:
#                 row_hit_rate = num_read_row_hits / num_read_cmds

#             if isinstance(read_latency_map, dict) and len(read_latency_map) > 0:
#                 numeric_keys = []
#                 for k in read_latency_map.keys():
#                     try:
#                         numeric_keys.append(int(k))
#                     except Exception:
#                         pass
#                 if numeric_keys:
#                     max_read_latency = max(numeric_keys)

#             # DRAMsim3 常见输出：read_energy/write_energy 为读写动态能耗(pJ)
#             # total_energy 含刷新/静态，通常不适合直接当单次 refill 能耗
#             read_pj = js.get("read_energy")
#             write_pj = js.get("write_energy")
#             act_pj = js.get("act_energy")

#             if read_pj is not None or write_pj is not None or act_pj is not None:
#                 dram_energy_nj = (
#                     float(read_pj or 0) +
#                     float(write_pj or 0) +
#                     float(act_pj or 0)
#                 ) / 1000.0  # pJ -> nJ
#             else:
#                 total_pj = js.get("total_energy")
#                 if total_pj is not None:
#                     dram_energy_nj = float(total_pj) / 1000.0

#         self.refill_history.append({
#             "buffer_name": buffer_name,
#             "tensor_region": tensor_region.name,
#             "start_elem": start_elem,
#             "num_elem": num_elem,
#             "bytes": total_bytes,

#             "request_cycle": cur_cycle,
#             "dram_done_cycle": dram_done_cycle,
#             "dram_latency_cycles": dram_done_cycle - cur_cycle,
#             "ready_cycle": ready_cycle,

#             "dram_energy_nj": dram_energy_nj,

#             "avg_read_latency": avg_read_latency,
#             "max_read_latency": max_read_latency,
#             "row_hit_rate": row_hit_rate,
#             "num_read_row_hits": num_read_row_hits,
#             "num_read_cmds": num_read_cmds,
#             "num_reads_done": num_reads_done,
#             "num_act_cmds": num_act_cmds,
#             "num_pre_cmds": num_pre_cmds,

#             "read_latency_map": read_latency_map,
#             "trace_stats": trace.get_stats(),
#             "dram_json_keys": list(js.keys()) if isinstance(js, dict) else [],
#             "output_dir": output_dir,
#         })

#         return ready_cycle

#     def load_region_to_buffer(
#         self,
#         buffer_name: str,
#         tensor_region: TensorRegion,
#         start_elem: int,
#         num_elem: int,
#         cur_cycle: int,
#         dram_config_path: str,
#         output_dir: str,
#         burst_bytes: int = 64,
#         issue_gap: int = 1,
#     ) -> int:
#         return self.refill_if_needed(
#             buffer_name=buffer_name,
#             tensor_region=tensor_region,
#             start_elem=start_elem,
#             num_elem=num_elem,
#             cur_cycle=cur_cycle,
#             dram_config_path=dram_config_path,
#             output_dir=output_dir,
#             burst_bytes=burst_bytes,
#             issue_gap=issue_gap,
#         )

#     # ---------------------------------------------------------
#     # Batch refill: 合并多个 refill 请求为一条 trace，跑一次 DRAMsim3，再按结果给各请求分配完成时间并 fill
#     # ---------------------------------------------------------
#     def refill_batch(
#         self,
#         refill_requests: list,
#         dram_config_path: str,
#         output_dir: str,
#         burst_bytes: int = 64,
#         issue_gap: int = 1,
#     ) -> int:
#         """
#         批量 refill：将多个 refill 请求合成一条 DRAM trace，跑一次 DRAMsim3，
#         从输出得到平均读延迟后为每个请求分配完成时间，再分别 write + fill 对应 buffer。

#         refill_requests: 列表，每项为 dict，例如
#           [
#             {"buffer_name": "xyz_buffer", "tensor_region": xyz_region, "start_elem": 0, "num_elem": 384, "request_cycle": 0},
#             {"buffer_name": "dist_buffer", "tensor_region": dist_region, "start_elem": 0, "num_elem": 128, "request_cycle": 0},
#           ]
#         可选键：burst_bytes, issue_gap（覆盖默认参数）。

#         返回：所有请求中最大的 ready_cycle（即 batch 全部就绪的时刻）。
#         """
#         if not refill_requests:
#             return 0

#         all_events = []
#         per_request = []

#         for r in refill_requests:
#             buffer_name = r["buffer_name"]
#             tensor_region = r["tensor_region"]
#             start_elem = int(r["start_elem"])
#             num_elem = int(r["num_elem"])
#             request_cycle = int(r["request_cycle"])
#             bb = int(r.get("burst_bytes", burst_bytes))
#             ig = int(r.get("issue_gap", issue_gap))

#             buf = self.sram_buffers[buffer_name]
#             total_bytes = num_elem * tensor_region.bytes_per_element
#             if not buf.can_hold(total_bytes):
#                 raise RuntimeError(
#                     f"{buffer_name} cannot hold refill bytes={total_bytes}, capacity={buf.capacity_bytes}"
#                 )

#             trace = DRAMTrace()
#             start_addr = tensor_region.addr_of(start_elem)
#             trace.extend_burst(
#                 base_addr=start_addr,
#                 num_bytes=total_bytes,
#                 burst_bytes=bb,
#                 is_write=False,
#                 start_cycle=request_cycle,
#                 issue_gap=ig,
#             )
#             num_txn = (total_bytes + bb - 1) // bb
#             per_request.append({
#                 "buffer_name": buffer_name,
#                 "tensor_region": tensor_region,
#                 "start_elem": start_elem,
#                 "num_elem": num_elem,
#                 "request_cycle": request_cycle,
#                 "total_bytes": total_bytes,
#                 "num_txn": num_txn,
#                 "burst_bytes": bb,
#                 "issue_gap": ig,
#             })
#             all_events.extend(trace.events)

#         all_events.sort(key=lambda e: (e.cycle, e.address))
#         merged = DRAMTrace()
#         merged.events = all_events

#         os.makedirs(output_dir, exist_ok=True)
#         trace_path = os.path.join(output_dir, "refill_batch.trace")
#         merged.dump(trace_path)

#         total_txn = len(all_events)
#         sim_cycles = max(100000, max(e.cycle for e in all_events) + total_txn * max(1, issue_gap) + 10000)
#         dram_result = self.run_dram_trace(
#             config_path=dram_config_path,
#             trace_path=trace_path,
#             num_cycles=sim_cycles,
#             output_dir=output_dir,
#         )

#         js_all = dram_result.json_stats or {}
#         js = js_all.get("0", js_all) if isinstance(js_all, dict) else {}
#         avg_read_latency = 200
#         if isinstance(js, dict) and js.get("average_read_latency") is not None:
#             try:
#                 avg_read_latency = math.ceil(float(js["average_read_latency"]))
#             except Exception:
#                 pass

#         batch_dram_energy_nj = None
#         if isinstance(js, dict):
#             read_pj = js.get("read_energy")
#             write_pj = js.get("write_energy")
#             act_pj = js.get("act_energy")
#             if read_pj is not None or write_pj is not None or act_pj is not None:
#                 batch_dram_energy_nj = (float(read_pj or 0) + float(write_pj or 0) + float(act_pj or 0)) / 1000.0

#         max_ready = 0
#         for idx, pr in enumerate(per_request):
#             last_issue = pr["request_cycle"] + (pr["num_txn"] - 1) * pr["issue_gap"]
#             dram_done_cycle = last_issue + avg_read_latency

#             buf = self.sram_buffers[pr["buffer_name"]]
#             write_stat = buf.write(pr["total_bytes"], dram_done_cycle)
#             ready_cycle = write_stat.end_cycle

#             buf.fill(
#                 region_name=pr["tensor_region"].name,
#                 start_elem=pr["start_elem"],
#                 num_elem=pr["num_elem"],
#                 ready_cycle=ready_cycle,
#             )

#             # 整条 batch trace 的 DRAM 能耗只记在第一个请求上，避免 refill_history 加总时重复
#             dram_energy_nj = batch_dram_energy_nj if idx == 0 else None

#             self.refill_history.append({
#                 "buffer_name": pr["buffer_name"],
#                 "tensor_region": pr["tensor_region"].name,
#                 "start_elem": pr["start_elem"],
#                 "num_elem": pr["num_elem"],
#                 "bytes": pr["total_bytes"],
#                 "request_cycle": pr["request_cycle"],
#                 "dram_done_cycle": dram_done_cycle,
#                 "dram_latency_cycles": dram_done_cycle - pr["request_cycle"],
#                 "ready_cycle": ready_cycle,
#                 "dram_energy_nj": dram_energy_nj,
#                 "batch": True,
#             })
#             max_ready = max(max_ready, ready_cycle)

#         return max_ready

#     def summary(self) -> Dict[str, Any]:
#         return {
#             "buffers": {
#                 name: buf.get_summary()
#                 for name, buf in self.sram_buffers.items()
#             },
#             "refill_history": self.refill_history,
#         }





import math
import os
from typing import Dict, Any, List

from .sram_buffer import SRAMBuffer
from .dramsim3_runner import DRAMSim3Runner
from .types import DRAMSim3Input
from .dram_trace import DRAMTrace
from .pointcloud_trace_gen import TensorRegion


class MemorySystem:
    def __init__(self, sram_buffers: Dict[str, SRAMBuffer], dramsim3_runner: DRAMSim3Runner):
        self.sram_buffers = sram_buffers
        self.dramsim3_runner = dramsim3_runner
        self.refill_history = []

    # ---------------------------------------------------------
    # SRAM direct access
    # ---------------------------------------------------------
    def sram_read(self, name: str, num_bytes: int, cur_cycle: int):
        return self.sram_buffers[name].read(num_bytes, cur_cycle)

    def sram_write(self, name: str, num_bytes: int, cur_cycle: int):
        return self.sram_buffers[name].write(num_bytes, cur_cycle)

    # ---------------------------------------------------------
    # DRAM execution
    # ---------------------------------------------------------
    def run_dram_trace(self, config_path: str, trace_path: str, num_cycles: int, output_dir: str):
        return self.dramsim3_runner.run(
            DRAMSim3Input(
                config_path=config_path,
                trace_path=trace_path,
                num_cycles=num_cycles,
                output_dir=output_dir,
            )
        )

    def _extract_channel_stats(self, dram_result):
        js_all = dram_result.json_stats if dram_result.json_stats else {}
        if isinstance(js_all, dict) and "0" in js_all and isinstance(js_all["0"], dict):
            return js_all["0"]
        return js_all if isinstance(js_all, dict) else {}

    def _estimate_refill_done_cycle(
        self,
        last_issue_cycle: int,
        dram_result,
    ) -> int:
        js = self._extract_channel_stats(dram_result)

        read_latency_map = js.get("read_latency", None)
        if isinstance(read_latency_map, dict) and len(read_latency_map) > 0:
            numeric_keys = []
            for k in read_latency_map.keys():
                try:
                    numeric_keys.append(int(k))
                except Exception:
                    pass
            if numeric_keys:
                max_read_latency = max(numeric_keys)
                return last_issue_cycle + max_read_latency

        avg_read_latency = js.get("average_read_latency", None)
        if avg_read_latency is not None:
            try:
                return last_issue_cycle + math.ceil(float(avg_read_latency))
            except Exception:
                pass

        return last_issue_cycle + 200

    def _extract_dram_metrics(self, dram_result):
        js = self._extract_channel_stats(dram_result)

        read_latency_map = js.get("read_latency", {}) if isinstance(js, dict) else {}
        avg_read_latency = js.get("average_read_latency", None) if isinstance(js, dict) else None
        num_read_row_hits = js.get("num_read_row_hits", 0) if isinstance(js, dict) else 0
        num_read_cmds = js.get("num_read_cmds", 0) if isinstance(js, dict) else 0
        num_reads_done = js.get("num_reads_done", 0) if isinstance(js, dict) else 0
        num_act_cmds = js.get("num_act_cmds", 0) if isinstance(js, dict) else 0
        num_pre_cmds = js.get("num_pre_cmds", 0) if isinstance(js, dict) else 0

        row_hit_rate = None
        if num_read_cmds > 0:
            row_hit_rate = num_read_row_hits / num_read_cmds

        max_read_latency = None
        if isinstance(read_latency_map, dict) and len(read_latency_map) > 0:
            numeric_keys = []
            for k in read_latency_map.keys():
                try:
                    numeric_keys.append(int(k))
                except Exception:
                    pass
            if numeric_keys:
                max_read_latency = max(numeric_keys)

        dram_energy_nj = None
        if isinstance(js, dict):
            read_pj = js.get("read_energy")
            write_pj = js.get("write_energy")
            act_pj = js.get("act_energy")
            if read_pj is not None or write_pj is not None or act_pj is not None:
                dram_energy_nj = (
                    float(read_pj or 0) +
                    float(write_pj or 0) +
                    float(act_pj or 0)
                ) / 1000.0  # pJ -> nJ

        return {
            "avg_read_latency": avg_read_latency,
            "max_read_latency": max_read_latency,
            "row_hit_rate": row_hit_rate,
            "num_read_row_hits": num_read_row_hits,
            "num_read_cmds": num_read_cmds,
            "num_reads_done": num_reads_done,
            "num_act_cmds": num_act_cmds,
            "num_pre_cmds": num_pre_cmds,
            "dram_energy_nj": dram_energy_nj,
            "read_latency_map": read_latency_map,
            "dram_json_keys": list(js.keys()) if isinstance(js, dict) else [],
        }

    # ---------------------------------------------------------
    # Single refill
    # ---------------------------------------------------------
    def refill_if_needed(
        self,
        buffer_name: str,
        tensor_region: TensorRegion,
        start_elem: int,
        num_elem: int,
        cur_cycle: int,
        dram_config_path: str,
        output_dir: str,
        burst_bytes: int = 64,
        issue_gap: int = 1,
    ) -> int:
        buf = self.sram_buffers[buffer_name]

        if buf.contains(tensor_region.name, start_elem, num_elem, cur_cycle):
            return cur_cycle

        total_bytes = num_elem * tensor_region.bytes_per_element
        if not buf.can_hold(total_bytes):
            raise RuntimeError(
                f"{buffer_name} cannot hold refill bytes={total_bytes}, "
                f"capacity={buf.capacity_bytes}"
            )

        trace = DRAMTrace()
        start_addr = tensor_region.addr_of(start_elem)
        trace.extend_burst(
            base_addr=start_addr,
            num_bytes=total_bytes,
            burst_bytes=burst_bytes,
            is_write=False,
            start_cycle=cur_cycle,
            issue_gap=issue_gap,
        )

        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, f"{buffer_name}_{tensor_region.name}_{start_elem}_{num_elem}.trace")
        trace.dump(trace_path)

        num_txn = (total_bytes + burst_bytes - 1) // burst_bytes
        last_issue_cycle = cur_cycle + max(0, num_txn - 1) * issue_gap
        sim_cycles = max(100000, last_issue_cycle + 10000)

        dram_result = self.run_dram_trace(
            config_path=dram_config_path,
            trace_path=trace_path,
            num_cycles=sim_cycles,
            output_dir=output_dir,
        )

        dram_done_cycle = self._estimate_refill_done_cycle(last_issue_cycle, dram_result)

        write_stat = buf.write(total_bytes, dram_done_cycle)
        ready_cycle = write_stat.end_cycle

        buf.fill(
            region_name=tensor_region.name,
            start_elem=start_elem,
            num_elem=num_elem,
            ready_cycle=ready_cycle,
        )

        metrics = self._extract_dram_metrics(dram_result)

        self.refill_history.append({
            "kind": "single",
            "buffer_name": buffer_name,
            "tensor_region": tensor_region.name,
            "start_elem": start_elem,
            "num_elem": num_elem,
            "bytes": total_bytes,
            "request_cycle": cur_cycle,
            "dram_done_cycle": dram_done_cycle,
            "dram_latency_cycles": dram_done_cycle - cur_cycle,
            "ready_cycle": ready_cycle,
            **metrics,
            "trace_stats": trace.get_stats(),
            "output_dir": output_dir,
        })

        return ready_cycle

    # ---------------------------------------------------------
    # Batch refill: multiple requests in one DRAM trace
    # ---------------------------------------------------------
    def refill_batch(
        self,
        refill_requests: List[Dict[str, Any]],
        dram_config_path: str,
        output_dir: str,
        burst_bytes: int = 64,
        issue_gap: int = 1,
    ) -> int:
        """
        将多个 refill 请求合并成同一条 DRAM trace，交给一次 DRAMsim3 运行。
        这才是真正开始利用 DRAMsim3 bank/channel 并行能力的方式。
        """
        if len(refill_requests) == 0:
            return 0

        # 先过滤命中请求
        pending = []
        for req in refill_requests:
            buf = self.sram_buffers[req["buffer_name"]]
            region = req["tensor_region"]
            start_elem = req["start_elem"]
            num_elem = req["num_elem"]
            req_cycle = req["request_cycle"]

            if buf.contains(region.name, start_elem, num_elem, req_cycle):
                continue

            total_bytes = num_elem * region.bytes_per_element
            if not buf.can_hold(total_bytes):
                raise RuntimeError(
                    f"{req['buffer_name']} cannot hold refill bytes={total_bytes}, "
                    f"capacity={buf.capacity_bytes}"
                )

            pending.append(req)

        if len(pending) == 0:
            return max(r["request_cycle"] for r in refill_requests)

        trace = DRAMTrace()
        os.makedirs(output_dir, exist_ok=True)

        batch_start_cycle = min(req["request_cycle"] for req in pending)
        batch_last_issue_cycle = batch_start_cycle
        total_batch_bytes = 0

        # 关键点：
        # 每个请求都按自己的 request_cycle 注入到同一条 trace 中，
        # 这样 DRAMsim3 才能在同一时间线上调度它们。
        for req in pending:
            region = req["tensor_region"]
            start_elem = req["start_elem"]
            num_elem = req["num_elem"]
            req_cycle = req["request_cycle"]

            start_addr = region.addr_of(start_elem)
            total_bytes = num_elem * region.bytes_per_element
            total_batch_bytes += total_bytes
            num_txn = (total_bytes + burst_bytes - 1) // burst_bytes
            last_issue = req_cycle + max(0, num_txn - 1) * issue_gap
            batch_last_issue_cycle = max(batch_last_issue_cycle, last_issue)

            trace.extend_burst(
                base_addr=start_addr,
                num_bytes=total_bytes,
                burst_bytes=burst_bytes,
                is_write=False,
                start_cycle=req_cycle,
                issue_gap=issue_gap,
            )

        trace_path = os.path.join(output_dir, "batch_refill.trace")
        trace.dump(trace_path)

        sim_cycles = max(100000, batch_last_issue_cycle + 10000)
        dram_result = self.run_dram_trace(
            config_path=dram_config_path,
            trace_path=trace_path,
            num_cycles=sim_cycles,
            output_dir=output_dir,
        )

        batch_dram_done_cycle = self._estimate_refill_done_cycle(batch_last_issue_cycle, dram_result)
        metrics = self._extract_dram_metrics(dram_result)

        # energy 按 bytes 比例分摊，避免双计数
        total_batch_energy = metrics["dram_energy_nj"]
        if total_batch_energy is None:
            total_batch_energy = 0.0

        batch_ready_cycle = batch_dram_done_cycle

        for req in pending:
            buf = self.sram_buffers[req["buffer_name"]]
            region = req["tensor_region"]
            start_elem = req["start_elem"]
            num_elem = req["num_elem"]
            req_cycle = req["request_cycle"]

            total_bytes = num_elem * region.bytes_per_element
            ratio = total_bytes / total_batch_bytes if total_batch_bytes > 0 else 0.0
            req_dram_energy_nj = total_batch_energy * ratio

            # DRAM 到达后，还要写入片上 SRAM buffer
            write_stat = buf.write(total_bytes, batch_dram_done_cycle)
            ready_cycle = write_stat.end_cycle
            batch_ready_cycle = max(batch_ready_cycle, ready_cycle)

            buf.fill(
                region_name=region.name,
                start_elem=start_elem,
                num_elem=num_elem,
                ready_cycle=ready_cycle,
            )

            self.refill_history.append({
                "kind": "batch",
                "buffer_name": req["buffer_name"],
                "tensor_region": region.name,
                "start_elem": start_elem,
                "num_elem": num_elem,
                "bytes": total_bytes,
                "request_cycle": req_cycle,
                "dram_done_cycle": batch_dram_done_cycle,
                "dram_latency_cycles": batch_dram_done_cycle - req_cycle,
                "ready_cycle": ready_cycle,
                "batch_start_cycle": batch_start_cycle,
                "batch_last_issue_cycle": batch_last_issue_cycle,
                "batch_total_bytes": total_batch_bytes,
                "dram_energy_nj": req_dram_energy_nj,
                **metrics,
                "trace_stats": trace.get_stats(),
                "output_dir": output_dir,
            })

        return batch_ready_cycle

    def summary(self) -> Dict[str, Any]:
        return {
            "buffers": {name: buf.get_summary() for name, buf in self.sram_buffers.items()},
            "refill_history": self.refill_history,
        }