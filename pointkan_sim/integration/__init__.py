"""
Integration with external simulators (MNSIM-2.0).

- mnsim_netstruct: Build NetStruct from PointKAN conv/linear ops and run
  Model_latency / Model_energy for Tile/NoC-level results.
"""

from .mnsim_netstruct import (
    build_netstruct_from_layers,
    run_mnsim_full,
    MNSIMFullResult,
)

__all__ = [
    "build_netstruct_from_layers",
    "run_mnsim_full",
    "MNSIMFullResult",
]
