"""
Reporter Module
===============

Report generation for PPA evaluation results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum, auto
from datetime import datetime
import json
from pathlib import Path

if TYPE_CHECKING:
    from ..simulator.scheduler import SimulationResult


class ReportFormat(Enum):
    """Report format enumeration."""
    CONSOLE = auto()       # Console output
    TEXT = auto()          # Plain text file
    JSON = auto()          # JSON file
    CSV = auto()           # CSV file


@dataclass
class PPAResult:
    """
    Complete PPA evaluation result.
    
    Attributes:
        latency_cycles: Total latency in cycles
        latency_ns: Total latency in nanoseconds
        latency_ms: Total latency in milliseconds
        power_mW: Average power in mW
        area_mm2: Total area in mm^2
        energy_nJ: Total energy in nJ
        energy_uJ: Total energy in µJ
    """
    latency_cycles: int = 0
    latency_ns: float = 0.0
    latency_ms: float = 0.0
    power_mW: float = 0.0
    area_mm2: float = 0.0
    energy_nJ: float = 0.0
    energy_uJ: float = 0.0
    
    # Breakdowns
    latency_breakdown: Dict[str, Any] = field(default_factory=dict)
    power_breakdown: Dict[str, float] = field(default_factory=dict)
    area_breakdown: Dict[str, float] = field(default_factory=dict)
    energy_breakdown: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'latency': {
                'cycles': self.latency_cycles,
                'ns': self.latency_ns,
                'ms': self.latency_ms,
            },
            'power_mW': self.power_mW,
            'area_mm2': self.area_mm2,
            'energy': {
                'nJ': self.energy_nJ,
                'uJ': self.energy_uJ,
            },
            'breakdown': {
                'latency': self.latency_breakdown,
                'power': self.power_breakdown,
                'area': self.area_breakdown,
                'energy': self.energy_breakdown,
            }
        }


class Reporter:
    """
    Report generator for PPA evaluation.
    
    Generates formatted reports in various formats.
    
    Attributes:
        title: Report title
        results: PPA results
        metadata: Additional metadata
    """
    
    def __init__(
        self,
        title: str = "PPA Evaluation Report"
    ):
        """
        Initialize reporter.
        
        Args:
            title: Report title
        """
        self.title = title
        self.results: Optional[PPAResult] = None
        self.metadata: Dict[str, Any] = {
            'generated_at': datetime.now().isoformat(),
            'version': '0.1.0',
        }
        self._sections: List[Dict[str, Any]] = []
    
    def set_results(self, results: PPAResult) -> None:
        """
        Set PPA results.
        
        Args:
            results: PPA evaluation results
        """
        self.results = results
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to report.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def add_section(self, title: str, content: Dict[str, Any]) -> None:
        """
        Add a custom section to the report.
        
        Args:
            title: Section title
            content: Section content
        """
        self._sections.append({
            'title': title,
            'content': content
        })
    
    def generate_console_report(self) -> str:
        """
        Generate console-formatted report.
        
        Returns:
            Formatted report string.
        """
        lines = []
        separator = "=" * 60
        
        # Header
        lines.append(separator)
        lines.append(f" {self.title}")
        lines.append(separator)
        lines.append("")
        
        # Metadata
        lines.append("Metadata:")
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # PPA Summary
        if self.results:
            lines.append("-" * 60)
            lines.append(" PPA Summary")
            lines.append("-" * 60)
            lines.append(f"  Latency:  {self.results.latency_cycles:,} cycles")
            lines.append(f"            {self.results.latency_ms:.4f} ms")
            lines.append(f"  Power:    {self.results.power_mW:.2f} mW")
            lines.append(f"  Area:     {self.results.area_mm2:.4f} mm^2")
            lines.append(f"  Energy:   {self.results.energy_nJ:.2f} nJ")
            lines.append(f"            {self.results.energy_uJ:.4f} uJ")
            lines.append("")
            
            # Latency breakdown
            if self.results.latency_breakdown:
                lines.append("-" * 60)
                lines.append(" Latency Breakdown")
                lines.append("-" * 60)
                for stage, cycles in self.results.latency_breakdown.items():
                    lines.append(f"  {stage}: {cycles:,} cycles")
                lines.append("")
            
            # Power breakdown
            if self.results.power_breakdown:
                lines.append("-" * 60)
                lines.append(" Power Breakdown")
                lines.append("-" * 60)
                for comp, power in self.results.power_breakdown.items():
                    lines.append(f"  {comp}: {power:.2f} mW")
                lines.append("")
            
            # Area breakdown
            if self.results.area_breakdown:
                lines.append("-" * 60)
                lines.append(" Area Breakdown")
                lines.append("-" * 60)
                for comp, area in self.results.area_breakdown.items():
                    lines.append(f"  {comp}: {area:.6f} mm^2")
                lines.append("")
        
        # Custom sections
        for section in self._sections:
            lines.append("-" * 60)
            lines.append(f" {section['title']}")
            lines.append("-" * 60)
            for key, value in section['content'].items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def generate_json_report(self) -> str:
        """
        Generate JSON-formatted report.
        
        Returns:
            JSON string.
        """
        report_data = {
            'title': self.title,
            'metadata': self.metadata,
            'results': self.results.to_dict() if self.results else None,
            'sections': self._sections,
        }
        
        return json.dumps(report_data, indent=2)
    
    def generate_csv_header(self) -> str:
        """
        Generate CSV header row.
        
        Returns:
            CSV header string.
        """
        headers = [
            'timestamp',
            'latency_cycles',
            'latency_ms',
            'power_mW',
            'area_mm2',
            'energy_nJ',
            'energy_uJ'
        ]
        return ','.join(headers)
    
    def generate_csv_row(self) -> str:
        """
        Generate CSV data row.
        
        Returns:
            CSV row string.
        """
        if not self.results:
            return ""
        
        values = [
            self.metadata.get('generated_at', ''),
            str(self.results.latency_cycles),
            f"{self.results.latency_ms:.6f}",
            f"{self.results.power_mW:.2f}",
            f"{self.results.area_mm2:.6f}",
            f"{self.results.energy_nJ:.2f}",
            f"{self.results.energy_uJ:.6f}"
        ]
        return ','.join(values)
    
    def print_report(self) -> None:
        """Print report to console."""
        print(self.generate_console_report())
    
    def save_report(
        self,
        filepath: str,
        format: ReportFormat = ReportFormat.TEXT
    ) -> None:
        """
        Save report to file.
        
        Args:
            filepath: Output file path
            format: Report format
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == ReportFormat.JSON:
            content = self.generate_json_report()
        elif format == ReportFormat.CSV:
            content = self.generate_csv_header() + "\n" + self.generate_csv_row()
        else:
            content = self.generate_console_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'title': self.title,
            'metadata': self.metadata,
            'results': self.results.to_dict() if self.results else None,
            'sections': self._sections,
        }


def create_ppa_result(
    latency_cycles: int,
    power_mW: float,
    area_mm2: float,
    clock_period_ns: float = 100.0,
    latency_breakdown: Optional[Dict] = None,
    power_breakdown: Optional[Dict] = None,
    area_breakdown: Optional[Dict] = None,
    energy_breakdown: Optional[Dict] = None
) -> PPAResult:
    """
    Create PPAResult from basic metrics.
    
    Args:
        latency_cycles: Total cycles
        power_mW: Average power
        area_mm2: Total area
        clock_period_ns: Clock period
        latency_breakdown: Latency breakdown
        power_breakdown: Power breakdown
        area_breakdown: Area breakdown
        energy_breakdown: Energy breakdown
        
    Returns:
        PPAResult instance.
    """
    latency_ns = latency_cycles * clock_period_ns
    latency_ms = latency_ns / 1e6
    energy_nJ = power_mW * latency_ns / 1e3
    energy_uJ = energy_nJ / 1000
    
    return PPAResult(
        latency_cycles=latency_cycles,
        latency_ns=latency_ns,
        latency_ms=latency_ms,
        power_mW=power_mW,
        area_mm2=area_mm2,
        energy_nJ=energy_nJ,
        energy_uJ=energy_uJ,
        latency_breakdown=latency_breakdown or {},
        power_breakdown=power_breakdown or {},
        area_breakdown=area_breakdown or {},
        energy_breakdown=energy_breakdown or {},
    )
