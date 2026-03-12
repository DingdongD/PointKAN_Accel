import math
import random
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Optional, List, Deque, Tuple, Dict

import matplotlib.pyplot as plt

# 脚本所在目录的上级 = cacti 根目录；图表输出到 outputs/figures
CACTI_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = CACTI_DIR / "outputs" / "figures"


# ============================================================
# 工具函数
# ============================================================

def mw_cycle_to_nj(power_mw: float, cycle_time_ns: float) -> float:
    return power_mw * cycle_time_ns / 1e3


# ============================================================
# DRAM 校准与时序
# ============================================================

@dataclass
class DramIOCalib:
    bus_freq_mhz: float
    num_dq: int
    burst_bytes: int

    io_dynamic_power_mw: float
    phy_power_mw: float
    io_termination_power_mw: float
    phy_wakeup_time_us: float = 0.0

    def peak_bandwidth_Bps(self) -> float:
        transfers_per_sec = 2.0 * self.bus_freq_mhz * 1e6
        bytes_per_transfer = self.num_dq / 8.0
        return transfers_per_sec * bytes_per_transfer


@dataclass
class DramTiming:
    cycle_time_ns: float = 1.25
    tRCD_ns: float = 13.75
    tCAS_ns: float = 13.75
    tRP_ns: float = 13.75
    tBURST_ns: float = 5.0
    tREFI_ns: float = 7800.0
    tRFC_ns: float = 160.0

    @property
    def tRCD_ck(self) -> int:
        return math.ceil(self.tRCD_ns / self.cycle_time_ns)

    @property
    def tCAS_ck(self) -> int:
        return math.ceil(self.tCAS_ns / self.cycle_time_ns)

    @property
    def tRP_ck(self) -> int:
        return math.ceil(self.tRP_ns / self.cycle_time_ns)

    @property
    def tBURST_ck(self) -> int:
        return math.ceil(self.tBURST_ns / self.cycle_time_ns)

    @property
    def tREFI_ck(self) -> int:
        return math.ceil(self.tREFI_ns / self.cycle_time_ns)

    @property
    def tRFC_ck(self) -> int:
        return math.ceil(self.tRFC_ns / self.cycle_time_ns)


# ============================================================
# DRAM 请求与状态
# ============================================================

@dataclass
class DramRequest:
    req_id: int
    op: str
    addr: int
    nbytes: int
    arrive_cycle: int

    channel: Optional[int] = None
    bank: Optional[int] = None
    row: Optional[int] = None
    col: Optional[int] = None

    start_cycle: Optional[int] = None
    finish_cycle: Optional[int] = None
    is_row_hit: Optional[bool] = None
    tag: str = ""


@dataclass
class BankState:
    open_row: Optional[int] = None
    busy_until: int = 0
    refresh_until: int = 0


@dataclass
class DramStats:
    cycles: int = 0
    busy_cycles: int = 0
    idle_cycles: int = 0
    refresh_cycles: int = 0

    read_reqs: int = 0
    write_reqs: int = 0
    finished_reqs: int = 0

    row_hits: int = 0
    row_misses: int = 0
    row_closed: int = 0
    bank_conflicts: int = 0

    total_read_bytes: int = 0
    total_write_bytes: int = 0
    total_latency_cycles: int = 0

    dynamic_energy_nj: float = 0.0
    background_energy_nj: float = 0.0
    refresh_energy_nj: float = 0.0

    @property
    def total_energy_nj(self) -> float:
        return self.dynamic_energy_nj + self.background_energy_nj + self.refresh_energy_nj

    @property
    def avg_latency_cycles(self) -> float:
        return self.total_latency_cycles / self.finished_reqs if self.finished_reqs else 0.0


# ============================================================
# DRAM Backend
# ============================================================

class DramBackend:
    def __init__(
        self,
        calib: DramIOCalib,
        timing: DramTiming,
        n_channels: int = 1,
        n_banks_per_channel: int = 8,
        queue_size: int = 1024,
        enable_refresh: bool = True,
        reorder_policy: str = "fr_fcfs",
        write_drain_threshold: int = 32,
    ):
        self.calib = calib
        self.timing = timing
        self.n_channels = n_channels
        self.n_banks_per_channel = n_banks_per_channel
        self.queue_size = queue_size
        self.enable_refresh = enable_refresh
        self.reorder_policy = reorder_policy
        self.write_drain_threshold = write_drain_threshold

        self.cycle = 0
        self.next_req_id = 0
        self.last_refresh_cycle = 0

        self.banks: List[List[BankState]] = [
            [BankState() for _ in range(n_banks_per_channel)]
            for _ in range(n_channels)
        ]

        self.readq: Deque[DramRequest] = deque()
        self.writeq: Deque[DramRequest] = deque()
        self.inflight: List[DramRequest] = []
        self.completed: List[DramRequest] = []

        self.stats = DramStats()

    def map_address(self, addr: int) -> Tuple[int, int, int, int]:
        burst_bits = int(math.log2(self.calib.burst_bytes))
        bank_bits = int(math.ceil(math.log2(self.n_banks_per_channel)))
        chan_bits = int(math.ceil(math.log2(self.n_channels))) if self.n_channels > 1 else 0

        burst_offset = addr & ((1 << burst_bits) - 1)
        x = addr >> burst_bits

        bank = x & ((1 << bank_bits) - 1)
        x >>= bank_bits

        if chan_bits > 0:
            channel = x & ((1 << chan_bits) - 1)
            x >>= chan_bits
        else:
            channel = 0

        row = x
        col = burst_offset
        return channel, bank, row, col

    def submit(self, op: str, addr: int, nbytes: int, tag: str = "") -> bool:
        assert op in ("read", "write")
        q = self.readq if op == "read" else self.writeq
        if len(q) >= self.queue_size:
            return False

        req = DramRequest(
            req_id=self.next_req_id,
            op=op,
            addr=addr,
            nbytes=nbytes,
            arrive_cycle=self.cycle,
            tag=tag,
        )
        self.next_req_id += 1

        ch, bk, row, col = self.map_address(addr)
        req.channel, req.bank, req.row, req.col = ch, bk, row, col
        q.append(req)

        if op == "read":
            self.stats.read_reqs += 1
            self.stats.total_read_bytes += nbytes
        else:
            self.stats.write_reqs += 1
            self.stats.total_write_bytes += nbytes

        return True

    def classify_row(self, req: DramRequest) -> str:
        bank = self.banks[req.channel][req.bank]
        if bank.open_row is None:
            return "closed"
        elif bank.open_row == req.row:
            return "hit"
        else:
            return "miss"

    def service_cycles(self, req: DramRequest) -> Tuple[int, str]:
        row_state = self.classify_row(req)

        transfer_cycles = math.ceil(
            (req.nbytes / self.calib.peak_bandwidth_Bps()) * 1e9 / self.timing.cycle_time_ns
        )
        transfer_cycles = max(transfer_cycles, self.timing.tBURST_ck)

        if row_state == "hit":
            req.is_row_hit = True
            self.stats.row_hits += 1
            return self.timing.tCAS_ck + transfer_cycles, row_state
        elif row_state == "miss":
            req.is_row_hit = False
            self.stats.row_misses += 1
            return self.timing.tRP_ck + self.timing.tRCD_ck + self.timing.tCAS_ck + transfer_cycles, row_state
        else:
            req.is_row_hit = False
            self.stats.row_closed += 1
            return self.timing.tRCD_ck + self.timing.tCAS_ck + transfer_cycles, row_state

    def pick_request_for_channel(self, channel: int, q: Deque[DramRequest]) -> Optional[int]:
        if not q:
            return None

        candidates = []
        for i, req in enumerate(q):
            if req.channel != channel:
                continue

            bank = self.banks[req.channel][req.bank]
            if self.cycle < bank.busy_until or self.cycle < bank.refresh_until:
                continue

            row_state = self.classify_row(req)
            ready_rank = 0 if row_state == "hit" else 1
            candidates.append((ready_rank, req.arrive_cycle, i))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    def maybe_issue_refresh(self):
        if not self.enable_refresh:
            return

        if self.cycle - self.last_refresh_cycle >= self.timing.tREFI_ck:
            until = self.cycle + self.timing.tRFC_ck
            for ch in range(self.n_channels):
                for bk in range(self.n_banks_per_channel):
                    self.banks[ch][bk].refresh_until = max(self.banks[ch][bk].refresh_until, until)
            self.last_refresh_cycle = self.cycle

    def issue_one_cycle(self):
        prefer_write = len(self.writeq) >= self.write_drain_threshold

        for ch in range(self.n_channels):
            selected = None
            selected_q = None

            if prefer_write:
                idx = self.pick_request_for_channel(ch, self.writeq)
                if idx is not None:
                    selected = idx
                    selected_q = self.writeq
                else:
                    idx = self.pick_request_for_channel(ch, self.readq)
                    if idx is not None:
                        selected = idx
                        selected_q = self.readq
            else:
                idx = self.pick_request_for_channel(ch, self.readq)
                if idx is not None:
                    selected = idx
                    selected_q = self.readq
                else:
                    idx = self.pick_request_for_channel(ch, self.writeq)
                    if idx is not None:
                        selected = idx
                        selected_q = self.writeq

            if selected is None:
                continue

            req = selected_q[selected]
            del selected_q[selected]

            bank = self.banks[req.channel][req.bank]
            if self.cycle < bank.busy_until:
                self.stats.bank_conflicts += 1
                continue

            req.start_cycle = self.cycle
            serv_ck, row_state = self.service_cycles(req)
            req.finish_cycle = self.cycle + serv_ck

            bank.busy_until = req.finish_cycle
            bank.open_row = req.row
            self.inflight.append(req)

    def retire_finished(self):
        remain = []
        for req in self.inflight:
            if req.finish_cycle is not None and req.finish_cycle <= self.cycle:
                self.completed.append(req)
                self.stats.finished_reqs += 1
                self.stats.total_latency_cycles += (req.finish_cycle - req.arrive_cycle)
            else:
                remain.append(req)
        self.inflight = remain

    def accumulate_energy_one_cycle(self):
        active_channels = 0
        any_refresh = False

        for ch in range(self.n_channels):
            ch_active = False
            for bk in range(self.n_banks_per_channel):
                bank = self.banks[ch][bk]
                if self.cycle < bank.refresh_until:
                    any_refresh = True
                if self.cycle < bank.busy_until:
                    ch_active = True
            if ch_active:
                active_channels += 1

        bg_power = self.n_channels * (self.calib.phy_power_mw + self.calib.io_termination_power_mw)
        self.stats.background_energy_nj += mw_cycle_to_nj(bg_power, self.timing.cycle_time_ns)

        dyn_power = active_channels * self.calib.io_dynamic_power_mw
        self.stats.dynamic_energy_nj += mw_cycle_to_nj(dyn_power, self.timing.cycle_time_ns)

        if any_refresh:
            self.stats.refresh_energy_nj += mw_cycle_to_nj(
                0.2 * self.n_channels * self.calib.io_dynamic_power_mw,
                self.timing.cycle_time_ns
            )

    def tick(self, n: int = 1):
        for _ in range(n):
            self.retire_finished()
            self.maybe_issue_refresh()
            self.issue_one_cycle()
            self.accumulate_energy_one_cycle()

            any_busy = False
            any_refresh = False
            for ch in range(self.n_channels):
                for bk in range(self.n_banks_per_channel):
                    bank = self.banks[ch][bk]
                    if self.cycle < bank.busy_until:
                        any_busy = True
                    if self.cycle < bank.refresh_until:
                        any_refresh = True

            if any_refresh:
                self.stats.refresh_cycles += 1
            elif any_busy:
                self.stats.busy_cycles += 1
            else:
                self.stats.idle_cycles += 1

            self.stats.cycles += 1
            self.cycle += 1

    def run_until_empty(self, max_cycles: int = 10_000_000):
        steps = 0
        while (self.readq or self.writeq or self.inflight) and steps < max_cycles:
            self.tick(1)
            steps += 1
        return steps

    def summary(self) -> Dict[str, float]:
        return {
            "cycles": self.stats.cycles,
            "busy_cycles": self.stats.busy_cycles,
            "idle_cycles": self.stats.idle_cycles,
            "refresh_cycles": self.stats.refresh_cycles,
            "read_reqs": self.stats.read_reqs,
            "write_reqs": self.stats.write_reqs,
            "finished_reqs": self.stats.finished_reqs,
            "row_hits": self.stats.row_hits,
            "row_misses": self.stats.row_misses,
            "row_closed": self.stats.row_closed,
            "bank_conflicts": self.stats.bank_conflicts,
            "total_read_bytes": self.stats.total_read_bytes,
            "total_write_bytes": self.stats.total_write_bytes,
            "avg_latency_cycles": self.stats.avg_latency_cycles,
            "dynamic_energy_nj": self.stats.dynamic_energy_nj,
            "background_energy_nj": self.stats.background_energy_nj,
            "refresh_energy_nj": self.stats.refresh_energy_nj,
            "total_energy_nj": self.stats.total_energy_nj,
        }


# ============================================================
# 点云流量生成器
# ============================================================

class PointCloudTrafficGenerator:
    def __init__(
        self,
        n_points: int,
        point_bytes: int = 16,
        feature_bytes: int = 64,
        burst_bytes: int = 64,
        seed: int = 0,
    ):
        self.n_points = n_points
        self.point_bytes = point_bytes
        self.feature_bytes = feature_bytes
        self.burst_bytes = burst_bytes
        self.rng = random.Random(seed)

        self.xyz_base = 0x10000000
        self.feature_base = 0x20000000
        self.reordered_feature_base = 0x30000000

    def align_down(self, addr: int) -> int:
        return addr - (addr % self.burst_bytes)

    def gen_stream(self, backend: DramBackend, tag: str = "stream"):
        total_bytes = self.n_points * self.point_bytes
        for offset in range(0, total_bytes, self.burst_bytes):
            backend.submit("read", self.xyz_base + offset, self.burst_bytes, tag=tag)

    def gen_unordered_gather(
        self,
        backend: DramBackend,
        n_queries: int,
        k: int,
        tag: str = "unordered_gather"
    ):
        for _ in range(n_queries):
            neigh = [self.rng.randrange(0, self.n_points) for _ in range(k)]
            for idx in neigh:
                addr = self.feature_base + idx * self.feature_bytes
                addr = self.align_down(addr)
                backend.submit("read", addr, self.burst_bytes, tag=tag)

    def gen_reordered_gather(
        self,
        backend: DramBackend,
        n_queries: int,
        k: int,
        cluster_span: int = 64,
        tag: str = "reordered_gather"
    ):
        for _ in range(n_queries):
            center = self.rng.randrange(0, self.n_points)
            start = max(0, center - cluster_span // 2)
            end = min(self.n_points, center + cluster_span // 2)

            neigh = [self.rng.randrange(start, end) for _ in range(k)]
            for idx in neigh:
                # 重排后相邻 idx 的地址更局部
                addr = self.reordered_feature_base + idx * self.feature_bytes
                addr = self.align_down(addr)
                backend.submit("read", addr, self.burst_bytes, tag=tag)

    def gen_writeback(
        self,
        backend: DramBackend,
        n_items: int,
        item_bytes: int,
        base: int = 0x40000000,
        tag: str = "writeback"
    ):
        total_bytes = n_items * item_bytes
        for offset in range(0, total_bytes, self.burst_bytes):
            backend.submit("write", base + offset, self.burst_bytes, tag=tag)


# ============================================================
# 单次实验
# ============================================================

def run_one_case(
    case_name: str,
    traffic_mode: str,
    n_points: int = 8192,
    n_queries: int = 512,
    k: int = 16,
    feature_bytes: int = 64,
    point_bytes: int = 16,
    burst_bytes: int = 64,
    seed: int = 0,
):
    calib = DramIOCalib(
        bus_freq_mhz=800,
        num_dq=64,
        burst_bytes=burst_bytes,
        io_dynamic_power_mw=1749.24,
        phy_power_mw=232.752,
        io_termination_power_mw=3785.43,
        phy_wakeup_time_us=27.503,
    )

    timing = DramTiming(
        cycle_time_ns=1.25,
        tRCD_ns=13.75,
        tCAS_ns=13.75,
        tRP_ns=13.75,
        tBURST_ns=5.0,
        tREFI_ns=7800.0,
        tRFC_ns=160.0,
    )

    backend = DramBackend(
        calib=calib,
        timing=timing,
        n_channels=2,
        n_banks_per_channel=8,
        queue_size=200000,
        enable_refresh=True,
        reorder_policy="fr_fcfs",
        write_drain_threshold=64,
    )

    gen = PointCloudTrafficGenerator(
        n_points=n_points,
        point_bytes=point_bytes,
        feature_bytes=feature_bytes,
        burst_bytes=burst_bytes,
        seed=seed,
    )

    # 先顺序读入 xyz
    gen.gen_stream(backend, tag=f"{traffic_mode}_input_stream")

    # 主访问模式
    if traffic_mode == "stream":
        # 再顺序读 feature
        total_feature_bytes = n_points * feature_bytes
        for offset in range(0, total_feature_bytes, burst_bytes):
            backend.submit("read", gen.feature_base + offset, burst_bytes, tag="stream_feature")
    elif traffic_mode == "unordered_gather":
        gen.gen_unordered_gather(backend, n_queries=n_queries, k=k, tag="unordered_gather")
    elif traffic_mode == "reordered_gather":
        gen.gen_reordered_gather(backend, n_queries=n_queries, k=k, cluster_span=64, tag="reordered_gather")
    else:
        raise ValueError(f"Unknown traffic_mode: {traffic_mode}")

    # 写回部分结果
    gen.gen_writeback(backend, n_items=n_queries, item_bytes=feature_bytes, tag="writeback")

    backend.run_until_empty()
    summary = backend.summary()
    summary["case_name"] = case_name
    summary["traffic_mode"] = traffic_mode
    return summary


# ============================================================
# 多组实验 + 可视化
# ============================================================

def plot_bar(names, values, ylabel, title, out_png):
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        ("Sequential", "stream"),
        ("UnorderedGather", "unordered_gather"),
        ("ReorderedGather", "reordered_gather"),
    ]

    results = []
    for name, mode in cases:
        res = run_one_case(
            case_name=name,
            traffic_mode=mode,
            n_points=8192,
            n_queries=512,
            k=16,
            feature_bytes=64,
            point_bytes=16,
            burst_bytes=64,
            seed=42,
        )
        results.append(res)
        print(f"\n=== {name} ===")
        for k, v in res.items():
            print(f"{k}: {v}")

    names = [r["case_name"] for r in results]

    plot_bar(
        names,
        [r["cycles"] for r in results],
        ylabel="Total cycles",
        title="Point Cloud DRAM Simulation: Total Cycles",
        out_png=FIGURES_DIR / "pc_dram_total_cycles.png",
    )

    plot_bar(
        names,
        [r["total_energy_nj"] for r in results],
        ylabel="Total energy (nJ)",
        title="Point Cloud DRAM Simulation: Total Energy",
        out_png=FIGURES_DIR / "pc_dram_total_energy.png",
    )

    plot_bar(
        names,
        [r["row_hits"] for r in results],
        ylabel="Row hits",
        title="Point Cloud DRAM Simulation: Row Hits",
        out_png=FIGURES_DIR / "pc_dram_row_hits.png",
    )

    plot_bar(
        names,
        [r["row_misses"] for r in results],
        ylabel="Row misses",
        title="Point Cloud DRAM Simulation: Row Misses",
        out_png=FIGURES_DIR / "pc_dram_row_misses.png",
    )

    plot_bar(
        names,
        [r["bank_conflicts"] for r in results],
        ylabel="Bank conflicts",
        title="Point Cloud DRAM Simulation: Bank Conflicts",
        out_png=FIGURES_DIR / "pc_dram_bank_conflicts.png",
    )

    print("\nSaved figures:")
    print(f"  {FIGURES_DIR}/pc_dram_total_cycles.png")
    print(f"  {FIGURES_DIR}/pc_dram_total_energy.png")
    print(f"  {FIGURES_DIR}/pc_dram_row_hits.png")
    print(f"  {FIGURES_DIR}/pc_dram_row_misses.png")
    print(f"  {FIGURES_DIR}/pc_dram_bank_conflicts.png")


if __name__ == "__main__":
    main()
