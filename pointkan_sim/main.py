"""
Point Cloud CIM PPA Simulator - Main Entry Point
=================================================

Simulates PointKAN / PointKAN-Elite network inference on
baseline and acc CIM accelerator architectures.

Usage:
    python main.py --help
    python main.py --demo
    python main.py --network pointkan --accelerator baseline
    python main.py --network pointkan-elite --accelerator acc
    python main.py --list_accelerators
"""

import argparse
import sys
import os
from typing import Optional, List, Dict, Any

# Fix import paths when running as script
if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from pointkan_sim.configs.hw_config import (
    get_accelerator_config, list_accelerator_configs, HardwareConfig,
)
from pointkan_sim.network.pointkan_net import (
    OperatorRef, LayerDef,
    create_pointkan_classification,
    POINTKAN_LAYERS,
)
from pointkan_sim.network.pointkan_elite_net import (
    create_pointkan_elite_classification,
    POINTKAN_ELITE_LAYERS,
)
from pointkan_sim.operators.dispatcher import (
    dispatch_operator, dispatch_layer, list_supported_op_types,
)
from pointkan_sim.simulator.scheduler import Scheduler
from pointkan_sim.simulator.event import Event
from pointkan_sim.report.reporter import Reporter, PPAResult
from pointkan_sim.utils.logger import (
    setup_logger, log_info, log_section, log_separator,
    LogLevel,
)
from pointkan_sim.utils.units import (
    cycles_to_ns, power_from_energy_nJ,
)


# =============================================================================
# CLI argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Point Cloud CIM PPA Simulator (new architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --network pointkan --accelerator baseline
  python main.py --network pointkan-elite --accelerator acc
  python main.py --list_accelerators
        """
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo simulation (PointKAN on baseline)")
    parser.add_argument(
        "--network", type=str, default=None,
        choices=["pointkan", "pointkan-elite"],
        help="Network to simulate")
    parser.add_argument(
        "--accelerator", type=str, default="baseline",
        help="Accelerator config name (default: baseline)")
    parser.add_argument(
        "--list_accelerators", action="store_true",
        help="List available accelerator configurations and exit")
    parser.add_argument(
        "--input_points", type=int, default=1024,
        help="Number of input points (default: 1024)")
    parser.add_argument(
        "--num_classes", type=int, default=40,
        help="Number of output classes (default: 40)")
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)")
    parser.add_argument(
        "--show_events", action="store_true",
        help="Show detailed event statistics")
    parser.add_argument(
        "--show_layers", action="store_true",
        help="Show per-layer breakdown")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output report file path (JSON)")

    return parser.parse_args()


# =============================================================================
# Network loading
# =============================================================================

def load_network(
    name: str,
    input_points: int = 1024,
    num_classes: int = 40,
) -> List[LayerDef]:
    """
    Load network layer definitions.

    Args:
        name: "pointkan" or "pointkan-elite"
        input_points: Number of input points
        num_classes: Number of output classes

    Returns:
        Ordered list of LayerDef.
    """
    if name == "pointkan":
        return create_pointkan_classification(
            input_points=input_points,
            num_classes=num_classes,
        )
    elif name == "pointkan-elite":
        return create_pointkan_elite_classification(
            input_points=input_points,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown network: {name}")


# =============================================================================
# Simulation core
# =============================================================================

def simulate_network(
    layers: List[LayerDef],
    hw_config: HardwareConfig,
    show_layers: bool = False,
) -> Dict[str, Any]:
    """
    Simulate a full network using the dispatcher and scheduler.

    For each layer, dispatches all OperatorRefs to generate events,
    then feeds them to the scheduler for cycle-accurate simulation.

    Args:
        layers: Network layer definitions
        hw_config: Hardware configuration
        show_layers: Print per-layer info

    Returns:
        Dictionary with simulation results.
    """
    scheduler = Scheduler(hw_config=hw_config)
    layer_results: List[Dict[str, Any]] = []
    current_cycle = 0

    for layer in layers:
        if not layer.operators:
            # Skip layers with no CIM-mappable operators (e.g., global_maxpool)
            layer_results.append({
                'name': layer.name,
                'type': layer.layer_type,
                'cycles': 0,
                'events': 0,
                'description': layer.description,
            })
            continue

        # Dispatch all operators in this layer
        layer_events = dispatch_layer(layer.operators, hw_config)

        # Add events to scheduler with cycle offset
        for event in layer_events:
            scheduler.event_queue.add_event(
                cycle=event.cycle + current_cycle,
                operator=event.operator,
                stage=event.stage,
                sub_stage=event.sub_stage,
                circuit_module=event.circuit_module,
                operation=event.operation,
                power=event.power,
                duration=event.duration,
                clock_period_ns=event.clock_period_ns,
                metadata=event.metadata,
            )

        # Calculate layer span
        if layer_events:
            layer_span = max(e.cycle + e.duration for e in layer_events)
        else:
            layer_span = 0

        layer_results.append({
            'name': layer.name,
            'type': layer.layer_type,
            'cycles': layer_span,
            'events': len(layer_events),
            'description': layer.description,
        })

        if show_layers:
            log_info(f"  {layer.name:<25s}  {layer_span:>10,} cycles  "
                     f"{len(layer_events):>4} events  [{layer.layer_type}]")

        current_cycle += layer_span

    # Run simulation
    log_info(f"\nRunning simulation: {len(scheduler.event_queue)} events, "
             f"{current_cycle:,} estimated cycles...")
    sim_result = scheduler.run_simulation()

    total_cycles = sim_result.total_cycles
    total_energy = sim_result.total_energy
    clock_period = hw_config.clock_period

    latency_ns = total_cycles * clock_period
    latency_ms = latency_ns / 1e6
    avg_power = total_energy / latency_ns * 1e3 if latency_ns > 0 else 0.0

    return {
        'total_cycles': total_cycles,
        'total_energy_nJ': total_energy,
        'latency_ns': latency_ns,
        'latency_ms': latency_ms,
        'avg_power_mW': avg_power,
        'peak_power_mW': sim_result.peak_power,
        'total_events': len(sim_result.events),
        'layer_results': layer_results,
        'sim_result': sim_result,
    }


# =============================================================================
# Printing
# =============================================================================

def print_simulation_summary(
    results: Dict[str, Any],
    hw_config: HardwareConfig,
    network_name: str,
):
    """Print simulation summary."""
    log_section("Simulation Results")
    log_info(f"  Network:     {network_name}")
    log_info(f"  Accelerator: {hw_config.accelerator_name} "
             f"({hw_config.accelerator_type.name})")
    log_info(f"  Clock:       {hw_config.clock_period} ns "
             f"({1000/hw_config.clock_period:.0f} MHz)")
    log_separator("-")
    log_info(f"  Latency:     {results['total_cycles']:,} cycles "
             f"({results['latency_ms']:.3f} ms)")
    log_info(f"  Energy:      {results['total_energy_nJ']:.2f} nJ "
             f"({results['total_energy_nJ']/1000:.4f} uJ)")
    log_info(f"  Avg Power:   {results['avg_power_mW']:.2f} mW")
    log_info(f"  Peak Power:  {results['peak_power_mW']:.2f} mW")
    log_info(f"  Events:      {results['total_events']:,}")
    log_separator()


def print_event_statistics(sim_result, hw_config: HardwareConfig):
    """Print detailed event statistics."""
    stats = sim_result.statistics

    if 'events_by_operator' in stats:
        log_info("\n  Events by Operator:")
        for op_name, count in sorted(stats['events_by_operator'].items()):
            log_info(f"    {op_name}: {count:,}")

    if 'events_by_stage' in stats:
        log_info("\n  Events by Stage:")
        for stage_name, count in sorted(stats['events_by_stage'].items()):
            log_info(f"    {stage_name}: {count:,}")

    if 'power' in stats:
        log_info("\n  Power Statistics:")
        log_info(f"    Average: {stats['power']['average_mW']:.2f} mW")
        log_info(f"    Peak:    {stats['power']['peak_mW']:.2f} mW")
        log_info(f"    Min:     {stats['power']['min_mW']:.2f} mW")


# =============================================================================
# Main workflows
# =============================================================================

def run_demo(args: argparse.Namespace) -> Dict[str, Any]:
    """Run demo simulation."""
    log_section("Demo: PointKAN on baseline")
    hw_config = get_accelerator_config(args.accelerator)
    layers = load_network("pointkan", args.input_points, args.num_classes)

    log_info(f"Network: PointKAN ({len(layers)} layers)")
    log_info(f"Accelerator: {hw_config.accelerator_name}")
    log_info(f"CIM parallelism: {hw_config.cim_parallelism}")
    log_info("")

    results = simulate_network(layers, hw_config, show_layers=True)
    print_simulation_summary(results, hw_config, "PointKAN")

    if args.show_events:
        print_event_statistics(results['sim_result'], hw_config)

    return results


def run_network_simulation(args: argparse.Namespace) -> Dict[str, Any]:
    """Run full network simulation."""
    network_name = args.network
    log_section(f"Network Simulation: {network_name}")

    hw_config = get_accelerator_config(args.accelerator)
    layers = load_network(network_name, args.input_points, args.num_classes)

    log_info(f"Network: {network_name} ({len(layers)} layers)")
    log_info(f"Accelerator: {hw_config.accelerator_name} "
             f"({hw_config.accelerator_type.name})")
    log_info(f"CIM parallelism: {hw_config.cim_parallelism}")
    log_info(f"Bit-serial cycles: {hw_config.bit_serial_cycles}")
    log_info("")

    results = simulate_network(
        layers, hw_config, show_layers=args.show_layers)
    print_simulation_summary(results, hw_config, network_name)

    if args.show_events:
        print_event_statistics(results['sim_result'], hw_config)

    return results


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = LogLevel[args.log_level]
    setup_logger(level=log_level)

    # Handle --list_accelerators
    if args.list_accelerators:
        print("Available accelerator configurations:")
        for name in list_accelerator_configs():
            cfg = get_accelerator_config(name)
            print(f"  {name:<15s}  [{cfg.accelerator_type.name}, "
                  f"CIM parallelism={cfg.cim_parallelism}, "
                  f"modules: {', '.join(cfg.get_all_modules().keys())}]")
        return 0

    log_section("Point Cloud CIM PPA Simulator")
    log_info(f"Supported op_types: {list_supported_op_types()}")
    log_info("")

    try:
        if args.demo:
            results = run_demo(args)
        elif args.network:
            results = run_network_simulation(args)
        else:
            log_info("No operation specified. Running demo...")
            results = run_demo(args)

        # Save output if specified
        if args.output and results:
            import json
            output_data = {
                k: v for k, v in results.items()
                if k != 'sim_result'  # not serializable
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            log_info(f"\nResults saved to: {args.output}")

        return 0

    except Exception as e:
        log_info(f"ERROR: Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
