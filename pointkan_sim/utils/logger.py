"""
Logger Module
=============

Logging utilities for the simulator.
"""

import logging
import sys
from typing import Optional
from datetime import datetime
from pathlib import Path
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Global logger instance
_logger: Optional[logging.Logger] = None
_log_file_path: Optional[str] = None


def setup_logger(
    name: str = "pointcloud_ppa_sim",
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure the logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (auto-generated if None)
        console_output: Whether to output to console
        file_output: Whether to output to file
        format_string: Custom format string
        
    Returns:
        Configured logger instance.
    """
    global _logger, _log_file_path
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level.value)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)-8s | %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"simulation_{timestamp}.log"
        
        _log_file_path = log_file
        
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(level.value)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    
    # Log initial message
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log level: {level.name}")
    if file_output and log_file:
        logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    Returns:
        Logger instance (creates default if not initialized).
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logger()
    
    return _logger


def get_log_file_path() -> Optional[str]:
    """
    Get the current log file path.
    
    Returns:
        Log file path or None if not logging to file.
    """
    return _log_file_path


def set_log_level(level: LogLevel) -> None:
    """
    Set the logging level.
    
    Args:
        level: New logging level
    """
    logger = get_logger()
    logger.setLevel(level.value)
    
    for handler in logger.handlers:
        handler.setLevel(level.value)


def log_debug(message: str) -> None:
    """Log debug message."""
    get_logger().debug(message)


def log_info(message: str) -> None:
    """Log info message."""
    get_logger().info(message)


def log_warning(message: str) -> None:
    """Log warning message."""
    get_logger().warning(message)


def log_error(message: str) -> None:
    """Log error message."""
    get_logger().error(message)


def log_critical(message: str) -> None:
    """Log critical message."""
    get_logger().critical(message)


def log_separator(char: str = "=", length: int = 60) -> None:
    """Log a separator line."""
    get_logger().info(char * length)


def log_section(title: str, char: str = "=", length: int = 60) -> None:
    """Log a section header."""
    logger = get_logger()
    logger.info(char * length)
    logger.info(f" {title}")
    logger.info(char * length)


def log_dict(data: dict, title: str = "", indent: int = 2) -> None:
    """
    Log a dictionary with formatting.
    
    Args:
        data: Dictionary to log
        title: Optional title
        indent: Indentation spaces
    """
    logger = get_logger()
    
    if title:
        logger.info(f"{title}:")
    
    def _log_dict_recursive(d: dict, level: int = 0):
        prefix = " " * (indent * level)
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                _log_dict_recursive(value, level + 1)
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    _log_dict_recursive(data)


def log_table(headers: list, rows: list, title: str = "") -> None:
    """
    Log data as a formatted table.
    
    Args:
        headers: List of column headers
        rows: List of row data (lists)
        title: Optional table title
    """
    logger = get_logger()
    
    if title:
        logger.info(f"\n{title}")
    
    # Calculate column widths
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    # Format header
    header_line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in widths)
    
    logger.info(header_line)
    logger.info(separator)
    
    # Format rows
    for row in rows:
        row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        logger.info(row_line)


def log_ppa_summary(
    latency_cycles: int,
    latency_ns: float,
    power_mW: float,
    area_mm2: float,
    energy_nJ: float
) -> None:
    """
    Log PPA summary in a formatted way.
    
    Args:
        latency_cycles: Total cycles
        latency_ns: Total time in ns
        power_mW: Average power in mW
        area_mm2: Total area in mm^2
        energy_nJ: Total energy in nJ
    """
    logger = get_logger()
    
    log_section("PPA Summary")
    
    logger.info(f"  Latency:  {latency_cycles:,} cycles ({latency_ns/1e6:.3f} ms)")
    logger.info(f"  Power:    {power_mW:.2f} mW")
    logger.info(f"  Area:     {area_mm2:.4f} mm^2")
    logger.info(f"  Energy:   {energy_nJ:.2f} nJ ({energy_nJ/1000:.4f} uJ)")
    
    log_separator()


class SimulationLogger:
    """
    Context manager for simulation logging.
    
    Provides structured logging for simulation runs.
    """
    
    def __init__(
        self,
        name: str = "simulation",
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None
    ):
        """
        Initialize simulation logger.
        
        Args:
            name: Simulation name
            level: Log level
            log_file: Log file path
        """
        self.name = name
        self.level = level
        self.log_file = log_file
        self.start_time: Optional[datetime] = None
        self.logger: Optional[logging.Logger] = None
    
    def __enter__(self):
        """Enter context."""
        self.start_time = datetime.now()
        self.logger = setup_logger(
            name=self.name,
            level=self.level,
            log_file=self.log_file
        )
        
        log_section(f"Simulation: {self.name}")
        log_info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        log_separator()
        log_info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_info(f"Duration: {duration}")
        
        if exc_type is not None:
            log_error(f"Simulation failed with error: {exc_val}")
        else:
            log_info("Simulation completed successfully")
        
        log_separator()
        
        return False  # Don't suppress exceptions
