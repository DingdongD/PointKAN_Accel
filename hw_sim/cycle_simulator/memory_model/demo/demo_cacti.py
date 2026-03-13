import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
from memory_model.cacti_runner import CactiRunner
from memory_model.types import CactiInput


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cycle_sim_root = os.path.dirname(repo_root)

    runner = CactiRunner(
        cacti_bin=os.path.join(cycle_sim_root, "cacti", "cacti"),
        template_cfg=os.path.join(cycle_sim_root, "cacti", "cache.cfg"),
        generated_cfg_dir=os.path.join(repo_root, "configs", "cacti", "generated"),
        output_dir=os.path.join(repo_root, "outputs", "cacti"),
        reuse_if_exists=True,
    )

    result = runner.run(
        CactiInput(
            size_bytes=16 * 1024,
            block_size_bytes=16,
            technology_u=0.028,
            output_bus_width_bits=128,
            cache_type="ram",
            associativity=1,
            read_write_port=1,
            exclusive_read_port=0,
            exclusive_write_port=0,
            single_ended_read_ports=0,
            uca_bank_count=1,
        )
    )

    print(json.dumps({
        "cfg_path": result.cfg_path,
        "access_time_ns": result.access_time_ns,
        "cycle_time_ns": result.cycle_time_ns,
        "read_energy_nj": result.read_energy_nj,
        "write_energy_nj": result.write_energy_nj,
        "leak_power_mw": result.leak_power_mw,
        "area_mm2": result.area_mm2,
    }, indent=2))


if __name__ == "__main__":
    main()