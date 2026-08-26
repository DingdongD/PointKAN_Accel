import json
import os
import sys
import tempfile
import unittest
from unittest import mock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEMORY_MODEL_ROOT = os.path.join(REPO_ROOT, "hw_sim", "cycle_simulator", "memory_model")
if MEMORY_MODEL_ROOT not in sys.path:
    sys.path.insert(0, MEMORY_MODEL_ROOT)

from memory_model.dramsim3_runner import DRAMSim3Runner
from memory_model.types import DRAMSim3Input


class DRAMSim3RunnerEpochJsonTest(unittest.TestCase):
    def test_run_accepts_concatenated_epoch_json_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dramsim3_bin = os.path.join(tmpdir, "dramsim3main")
            with open(dramsim3_bin, "w", encoding="utf-8") as f:
                f.write("")

            outdir = os.path.join(tmpdir, "out")
            runner = DRAMSim3Runner(dramsim3_bin=dramsim3_bin, work_root=tmpdir)

            def _fake_popen(cmd, stdout, stderr, text, cwd):
                with open(os.path.join(cwd, "dramsim3.json"), "w", encoding="utf-8") as f:
                    json.dump({"0": {"average_read_latency": 12}}, f)
                with open(os.path.join(cwd, "dramsim3epoch.json"), "w", encoding="utf-8") as f:
                    f.write('[{"epoch_num": 1}]\n')
                    f.write('{"epoch_num": 2}\n')

                class _Proc:
                    returncode = 0

                    @staticmethod
                    def poll():
                        return 0

                return _Proc()

            with mock.patch(
                "memory_model.dramsim3_runner.subprocess.Popen",
                side_effect=_fake_popen,
            ):
                result = runner.run(
                    DRAMSim3Input(
                        config_path="/tmp/fake.ini",
                        trace_path="/tmp/fake.trace",
                        num_cycles=1000,
                        output_dir=outdir,
                    )
                )

        self.assertEqual(result.json_stats["0"]["average_read_latency"], 12)
        self.assertEqual(result.epoch_stats, [{"epoch_num": 1}, {"epoch_num": 2}])

    def test_run_ignores_trailing_bracket_after_concatenated_epoch_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dramsim3_bin = os.path.join(tmpdir, "dramsim3main")
            with open(dramsim3_bin, "w", encoding="utf-8") as f:
                f.write("")

            outdir = os.path.join(tmpdir, "out")
            runner = DRAMSim3Runner(dramsim3_bin=dramsim3_bin, work_root=tmpdir)

            def _fake_popen(cmd, stdout, stderr, text, cwd):
                with open(os.path.join(cwd, "dramsim3.json"), "w", encoding="utf-8") as f:
                    json.dump({"0": {"average_read_latency": 12}}, f)
                with open(os.path.join(cwd, "dramsim3epoch.json"), "w", encoding="utf-8") as f:
                    f.write('[{"epoch_num": 1}]\n')
                    f.write('{"epoch_num": 2}]')

                class _Proc:
                    returncode = 0

                    @staticmethod
                    def poll():
                        return 0

                return _Proc()

            with mock.patch(
                "memory_model.dramsim3_runner.subprocess.Popen",
                side_effect=_fake_popen,
            ):
                result = runner.run(
                    DRAMSim3Input(
                        config_path="/tmp/fake.ini",
                        trace_path="/tmp/fake.trace",
                        num_cycles=1000,
                        output_dir=outdir,
                    )
                )

        self.assertEqual(result.epoch_stats, [{"epoch_num": 1}, {"epoch_num": 2}])

    def test_load_json_retries_when_first_read_is_transiently_malformed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "dramsim3.json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write('{"0": {"average_read_latency": 12}}')

            real_open = open
            call_count = {"n": 0}

            def _fake_open(path, mode="r", *args, **kwargs):
                if path == json_path and "r" in mode:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        return mock.mock_open(read_data='{"0": {"average_read_latency": 12]').return_value
                return real_open(path, mode, *args, **kwargs)

            with mock.patch("builtins.open", side_effect=_fake_open):
                out = DRAMSim3Runner._load_json_allow_concatenated(json_path)

        self.assertEqual(out, {"0": {"average_read_latency": 12}})


if __name__ == "__main__":
    unittest.main()
