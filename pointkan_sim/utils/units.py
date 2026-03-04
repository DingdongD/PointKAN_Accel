"""
Unit Conversion Utilities
=========================

Unified energy, power, time, area conversion functions.
All modules should use these functions instead of inline conversion.

Key relationships:
    Energy = Power × Time
    mW × ns = pJ (picojoules)
    1 nJ = 1000 pJ

Standard units in this project:
    - Power: mW (milliwatts)
    - Time/Latency: ns (nanoseconds), cycles
    - Energy: nJ (nanojoules)
    - Area: mm²
"""


# =============================================================================
# Energy
# =============================================================================

def energy_nJ_from_ns(power_mW: float, time_ns: float) -> float:
    """
    Calculate energy in nanojoules from power and time.

    Args:
        power_mW: Power in milliwatts.
        time_ns: Time in nanoseconds.

    Returns:
        Energy in nanojoules (nJ).
    """
    # mW * ns = pJ; /1e3 => nJ
    return power_mW * time_ns / 1e3


def energy_to_nj(value: float, unit: str) -> float:
    """
    Convert energy value to nanojoules.

    Args:
        value: Energy value.
        unit: Source unit ('pj', 'nj', 'uj', 'mj', 'j').

    Returns:
        Energy in nanojoules.
    """
    conversions = {
        'pj': 0.001,
        'nj': 1.0,
        'uj': 1000.0,
        'mj': 1000000.0,
        'j': 1000000000.0,
    }
    return value * conversions.get(unit.lower(), 1.0)


# =============================================================================
# Power
# =============================================================================

def power_from_energy_nJ(energy_nJ_val: float, time_ns: float) -> float:
    """
    Calculate average power from energy and time.

    Args:
        energy_nJ_val: Energy in nanojoules.
        time_ns: Time in nanoseconds.

    Returns:
        Power in milliwatts (mW).
    """
    if time_ns <= 0:
        return 0.0
    # nJ / ns = mW  (1e-9 J / 1e-9 s = 1 W = 1000 mW)
    return energy_nJ_val / time_ns * 1e3


def power_to_mw(value: float, unit: str) -> float:
    """
    Convert power value to milliwatts.

    Args:
        value: Power value.
        unit: Source unit ('uw', 'mw', 'w').

    Returns:
        Power in milliwatts.
    """
    conversions = {
        'uw': 0.001,
        'mw': 1.0,
        'w': 1000.0,
    }
    return value * conversions.get(unit.lower(), 1.0)


# =============================================================================
# Time / Latency
# =============================================================================

def cycles_to_ns(cycles: int, clock_period_ns: float) -> float:
    """Convert cycles to nanoseconds."""
    return cycles * clock_period_ns


def cycles_to_ms(cycles: int, clock_period_ns: float) -> float:
    """Convert cycles to milliseconds."""
    return cycles * clock_period_ns / 1e6


def time_to_ns(value: float, unit: str) -> float:
    """
    Convert time value to nanoseconds.

    Args:
        value: Time value.
        unit: Source unit ('ps', 'ns', 'us', 'ms', 's').

    Returns:
        Time in nanoseconds.
    """
    conversions = {
        'ps': 0.001,
        'ns': 1.0,
        'us': 1000.0,
        'ms': 1000000.0,
        's': 1000000000.0,
    }
    return value * conversions.get(unit.lower(), 1.0)


def freq_mhz_from_period_ns(clock_period_ns: float) -> float:
    """Convert clock period (ns) to frequency (MHz)."""
    if clock_period_ns <= 0:
        return 0.0
    return 1e3 / clock_period_ns


# =============================================================================
# Area
# =============================================================================

def area_to_mm2(value: float, unit: str) -> float:
    """
    Convert area value to mm².

    Args:
        value: Area value.
        unit: Source unit ('um2', 'mm2', 'cm2').

    Returns:
        Area in mm².
    """
    conversions = {
        'um2': 1e-6,
        'mm2': 1.0,
        'cm2': 100.0,
    }
    return value * conversions.get(unit.lower(), 1.0)
