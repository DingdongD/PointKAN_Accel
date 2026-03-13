class DRAMTraceEvent:
    def __init__(self, address: int, is_write: bool, cycle: int = 0):
        self.address = address
        self.is_write = is_write
        self.cycle = cycle

    def to_line(self) -> str:
        op = "WRITE" if self.is_write else "READ"
        return f"0x{self.address:x} {op} {self.cycle}"


class DRAMTrace:
    def __init__(self):
        self.events = []

    def add_read(self, address: int, cycle: int = 0):
        self.events.append(DRAMTraceEvent(address, False, cycle))

    def add_write(self, address: int, cycle: int = 0):
        self.events.append(DRAMTraceEvent(address, True, cycle))

    def extend_burst(self, base_addr: int, num_bytes: int, burst_bytes: int, is_write: bool, start_cycle: int = 0, issue_gap: int = 1):
        num_txn = (num_bytes + burst_bytes - 1) // burst_bytes
        for i in range(num_txn):
            addr = base_addr + i * burst_bytes
            cycle = start_cycle + i * issue_gap
            if is_write:
                self.add_write(addr, cycle)
            else:
                self.add_read(addr, cycle)

    def dump(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(ev.to_line() + "\n")

    def get_stats(self):
        num_read = sum(1 for e in self.events if not e.is_write)
        num_write = sum(1 for e in self.events if e.is_write)
        min_cycle = min((e.cycle for e in self.events), default=0)
        max_cycle = max((e.cycle for e in self.events), default=0)
        min_addr = min((e.address for e in self.events), default=0)
        max_addr = max((e.address for e in self.events), default=0)
        return {
            "num_events": len(self.events),
            "num_read": num_read,
            "num_write": num_write,
            "min_cycle": min_cycle,
            "max_cycle": max_cycle,
            "min_addr": hex(min_addr),
            "max_addr": hex(max_addr),
        }