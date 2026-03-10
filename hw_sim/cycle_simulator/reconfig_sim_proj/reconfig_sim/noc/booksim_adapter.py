from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from reconfig_sim.noc.traffic_trace import TrafficTrace


class BookSimAdapter:
    """Serialize internal traffic trace into BookSim-friendly text artifacts."""

    def __init__(self, packet_bytes: int = 16):
        if packet_bytes <= 0:
            raise ValueError("packet_bytes must be positive")
        self.packet_bytes = packet_bytes

    def to_packet_trace_lines(self, trace: TrafficTrace) -> List[str]:
        """
        Return line-based packet trace.

        Format per line: `<cycle> <src> <dst> <packets> <flow>`
        """
        lines: List[str] = []
        for event in trace.events:
            packets = (event.bytes + self.packet_bytes - 1) // self.packet_bytes
            lines.append(f"{event.cycle} {event.src} {event.dst} {packets} {event.flow}")
        return lines

    def write_packet_trace(self, trace: TrafficTrace, out_path: str) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(self.to_packet_trace_lines(trace)) + "\n", encoding="utf-8")
        return out

    def write_trace_json(self, trace: TrafficTrace, out_path: str) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(trace.to_dict(), indent=2), encoding="utf-8")
        return out

    def write_trace_csv(self, trace: TrafficTrace, out_path: str) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["cycle", "src", "dst", "bytes", "flow", "op_name", "tile_index", "meta"],
            )
            writer.writeheader()
            for e in trace.events:
                writer.writerow(
                    {
                        "cycle": e.cycle,
                        "src": e.src,
                        "dst": e.dst,
                        "bytes": e.bytes,
                        "flow": e.flow,
                        "op_name": e.op_name,
                        "tile_index": e.tile_index,
                        "meta": json.dumps(e.meta, ensure_ascii=False),
                    }
                )
        return out

    def export_trace(self, trace: TrafficTrace) -> Dict[str, object]:
        """Return in-memory BookSim export payload for quick integration."""
        return {
            "packet_trace_lines": self.to_packet_trace_lines(trace),
            "summary": self.summarize(trace),
        }

    def summarize(self, trace: TrafficTrace) -> Dict[str, int]:
        total_bytes = sum(e.bytes for e in trace.events)
        total_packets = sum((e.bytes + self.packet_bytes - 1) // self.packet_bytes for e in trace.events)
        return {
            "events": len(trace.events),
            "bytes": total_bytes,
            "packets": total_packets,
            "packet_bytes": self.packet_bytes,
        }
