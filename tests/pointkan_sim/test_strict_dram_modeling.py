import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from pointkan_sim.integration import memory_model_adapter as mma


class StrictDramModelingTest(unittest.TestCase):
    def test_query_dram_cycles_energy_uses_strict_dramsim3_cycles(self):
        hw = SimpleNamespace(
            dram=SimpleNamespace(
                dramsim3_bin="/tmp/fake_dramsim3",
                dramsim3_config_ini="/tmp/fake.ini",
                bandwidth_bytes_per_cycle=1,
                block_size_bytes=64,
                read_energy_nj_per_access=2.0,
                write_energy_nj_per_access=3.0,
            )
        )

        with mock.patch.object(
            mma,
            "run_dram_trace_for_layer_buffer_dram",
            return_value=[(17, 9.5)],
        ) as mocked_run:
            cycles, energy_nj = mma.query_dram_cycles_energy(
                4096,
                is_read=True,
                hw_config=hw,
                clock_period_ns=1.0,
            )

        self.assertEqual(cycles, 17)
        self.assertEqual(energy_nj, 9.5)
        mocked_run.assert_called_once_with([(4096, True)], hw, None, "hw_sim/cycle_simulator")

    def test_query_dram_cycles_energy_caches_identical_transfers(self):
        hw = SimpleNamespace(
            dram=SimpleNamespace(
                dramsim3_bin="/tmp/fake_dramsim3",
                dramsim3_config_ini="/tmp/fake.ini",
                bandwidth_bytes_per_cycle=1,
                block_size_bytes=64,
                read_energy_nj_per_access=2.0,
                write_energy_nj_per_access=3.0,
            )
        )

        with mock.patch.object(
            mma,
            "run_dram_trace_for_layer_buffer_dram",
            return_value=[(11, 4.0)],
        ) as mocked_run:
            first = mma.query_dram_cycles_energy(2048, is_read=True, hw_config=hw, clock_period_ns=1.0)
            second = mma.query_dram_cycles_energy(2048, is_read=True, hw_config=hw, clock_period_ns=1.0)

        self.assertEqual(first, (11, 4.0))
        self.assertEqual(second, (11, 4.0))
        self.assertEqual(mocked_run.call_count, 1)

    def test_run_dram_trace_for_layer_buffer_dram_uses_dramsim3_done_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_model_repo = os.path.join(tmpdir, "memory_model")
            cycle_sim_abs = os.path.join(tmpdir, "cycle_simulator")
            os.makedirs(memory_model_repo, exist_ok=True)
            os.makedirs(cycle_sim_abs, exist_ok=True)
            dramsim3_bin = os.path.join(tmpdir, "dramsim3main")
            config_ini = os.path.join(tmpdir, "dram.ini")
            open(dramsim3_bin, "w", encoding="utf-8").close()
            open(config_ini, "w", encoding="utf-8").close()

            class FakeTrace:
                def extend_burst(self, **kwargs):
                    return None

                def dump(self, path):
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write("trace")

            class FakeRunner:
                def __init__(self, dramsim3_bin, work_root):
                    self.dramsim3_bin = dramsim3_bin
                    self.work_root = work_root

                def run(self, dramsim_input):
                    return SimpleNamespace(
                        json_stats={
                            "0": {
                                "average_read_latency": 37,
                                "read_energy": 8000.0,
                                "write_energy": 0.0,
                            }
                        }
                    )

            class FakeInput:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            fake_dram_trace_module = SimpleNamespace(DRAMTrace=FakeTrace)
            fake_runner_module = SimpleNamespace(DRAMSim3Runner=FakeRunner)
            fake_types_module = SimpleNamespace(DRAMSim3Input=FakeInput)

            hw = SimpleNamespace(
                dram=SimpleNamespace(
                    dramsim3_bin=dramsim3_bin,
                    dramsim3_config_ini=config_ini,
                    bandwidth_bytes_per_cycle=1,
                    block_size_bytes=64,
                    issue_gap=1,
                )
            )

            with mock.patch.object(mma, "get_memory_model_roots", return_value=(memory_model_repo, cycle_sim_abs)), \
                 mock.patch.object(mma, "resolve_dramsim3_config_path", return_value=config_ini), \
                 mock.patch.object(mma, "_parse_dramsim3_config", return_value={"tCK": 1.0}), \
                 mock.patch.object(mma, "get_dram_trace_parameters", return_value=(128, 1)), \
                 mock.patch.dict("sys.modules", {
                     "memory_model.dram_trace": fake_dram_trace_module,
                     "memory_model.dramsim3_runner": fake_runner_module,
                     "memory_model.types": fake_types_module,
                 }):
                out = mma.run_dram_trace_for_layer_buffer_dram(
                    [(4096, True)],
                    hw,
                    repo_root=tmpdir,
                )

        self.assertEqual(out, [(68, 8.0)])

    def test_run_dram_trace_for_layer_buffer_dram_uses_unique_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_model_repo = os.path.join(tmpdir, "memory_model")
            cycle_sim_abs = os.path.join(tmpdir, "cycle_simulator")
            os.makedirs(memory_model_repo, exist_ok=True)
            os.makedirs(cycle_sim_abs, exist_ok=True)
            dramsim3_bin = os.path.join(tmpdir, "dramsim3main")
            config_ini = os.path.join(tmpdir, "dram.ini")
            open(dramsim3_bin, "w", encoding="utf-8").close()
            open(config_ini, "w", encoding="utf-8").close()

            seen_output_dirs = []

            class FakeTrace:
                def extend_burst(self, **kwargs):
                    return None

                def dump(self, path):
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write("trace")

            class FakeRunner:
                def __init__(self, dramsim3_bin, work_root):
                    self.dramsim3_bin = dramsim3_bin
                    self.work_root = work_root

                def run(self, dramsim_input):
                    seen_output_dirs.append(dramsim_input.output_dir)
                    return SimpleNamespace(
                        json_stats={
                            "0": {
                                "average_read_latency": 5,
                                "read_energy": 128.0,
                                "write_energy": 0.0,
                            }
                        }
                    )

            class FakeInput:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            hw = SimpleNamespace(
                dram=SimpleNamespace(
                    dramsim3_bin=dramsim3_bin,
                    dramsim3_config_ini=config_ini,
                    bandwidth_bytes_per_cycle=1,
                    block_size_bytes=64,
                    issue_gap=1,
                )
            )

            with mock.patch.object(mma, "get_memory_model_roots", return_value=(memory_model_repo, cycle_sim_abs)), \
                 mock.patch.object(mma, "resolve_dramsim3_config_path", return_value=config_ini), \
                 mock.patch.object(mma, "_parse_dramsim3_config", return_value={"tCK": 1.0}), \
                 mock.patch.object(mma, "get_dram_trace_parameters", return_value=(128, 1)), \
                 mock.patch.dict("sys.modules", {
                     "memory_model.dram_trace": SimpleNamespace(DRAMTrace=FakeTrace),
                     "memory_model.dramsim3_runner": SimpleNamespace(DRAMSim3Runner=FakeRunner),
                     "memory_model.types": SimpleNamespace(DRAMSim3Input=FakeInput),
                 }):
                mma.run_dram_trace_for_layer_buffer_dram([(4096, True)], hw, repo_root=tmpdir)
                mma.run_dram_trace_for_layer_buffer_dram([(4096, True)], hw, repo_root=tmpdir)

        self.assertEqual(len(seen_output_dirs), 2)
        self.assertNotEqual(seen_output_dirs[0], seen_output_dirs[1])
        for outdir in seen_output_dirs:
            self.assertEqual(os.path.basename(outdir).startswith("layer_buffer_dram_"), True)

    def test_strict_dram_batch_uses_unique_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seen_output_dirs = []

            class FakeTrace:
                def add_read(self, addr, cyc):
                    return None

                def add_write(self, addr, cyc):
                    return None

                def dump(self, path):
                    with open(path, "w", encoding="utf-8") as fh:
                        fh.write("trace")

            class FakeRunner:
                def __init__(self, dramsim3_bin, work_root):
                    self.dramsim3_bin = dramsim3_bin
                    self.work_root = work_root

                def run(self, dramsim_input):
                    seen_output_dirs.append(dramsim_input.output_dir)
                    return SimpleNamespace(
                        json_stats={
                            "0": {
                                "average_read_latency": 9,
                                "read_energy": 256.0,
                                "write_energy": 0.0,
                                "act_energy": 0.0,
                            }
                        }
                    )

            class FakeInput:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)

            runtime = {
                "dram": SimpleNamespace(bandwidth_bytes_per_cycle=1),
                "memory_model_repo": tmpdir,
                "dramsim3_bin": "/tmp/fake_dramsim3",
                "config_path": "/tmp/fake.ini",
                "tCK_ns": 1.0,
                "burst_bytes": 128,
                "issue_gap": 1,
                "DRAMTrace": FakeTrace,
                "DRAMSim3Runner": FakeRunner,
                "DRAMSim3Input": FakeInput,
            }

            requests = [{"num_bytes": 1024, "is_read": True, "start_cycle": 0}]

            with mock.patch.object(mma, "_resolve_dramsim3_runtime", return_value=runtime):
                mma._run_strict_dram_request_batch(requests, SimpleNamespace(), repo_root=tmpdir)
                mma._run_strict_dram_request_batch(requests, SimpleNamespace(), repo_root=tmpdir)

        self.assertEqual(len(seen_output_dirs), 2)
        self.assertNotEqual(seen_output_dirs[0], seen_output_dirs[1])
        for outdir in seen_output_dirs:
            self.assertEqual(os.path.basename(outdir).startswith("strict_batch_"), True)


if __name__ == "__main__":
    unittest.main()
