import tempfile
import unittest

import numpy as np
import pointkan_sim.report.reporter as reporter_module

from pointkan_sim.main import _sanitize_log_tag as main_sanitize_log_tag
from pointkan_sim.run_sim_with_real_eval import _sanitize_log_tag as real_eval_sanitize_log_tag
from pointkan_sim.operators.dispatcher import _retag_events as dispatcher_retag_events
from pointkan_sim.operators.dispatcher import _split_bytes as dispatcher_split_bytes
from pointkan_sim.operators.layer_buffer_ops import _split_bytes as layer_buffer_split_bytes
from pointkan_sim.operators.common_ops import _retag_events as common_retag_events
from pointkan_sim.operators.neurosim_adapter import _stretch_input_conv1d as neurosim_stretch_input_conv1d
from pointkan_sim.integration.neurosim_full import _stretch_input_conv1d as neurosim_full_stretch_input_conv1d
from pointkan_sim.integration.memory_model_adapter import _parse_dramsim3_burst_bytes as memory_model_burst_bytes
from pointkan_sim.integration.scalesim_adapter import _parse_dramsim3_burst_bytes as scalesim_burst_bytes
from pointkan_sim.memory.memory_contract import _index_bytes_for_points as contract_index_bytes_for_points
from pointkan_sim.operators.dispatcher import _index_bytes_for_points as dispatcher_index_bytes_for_points
from pointkan_sim.simulator.event import Event


class DedupRegressionTest(unittest.TestCase):
    def test_log_tag_sanitizer_is_shared_implementation(self):
        self.assertIs(main_sanitize_log_tag, real_eval_sanitize_log_tag)
        self.assertEqual(main_sanitize_log_tag(" pointkan elite/semseg "), "pointkan_elite_semseg")
        self.assertEqual(main_sanitize_log_tag("..."), "sim")

    def test_split_bytes_is_shared_implementation(self):
        self.assertIs(dispatcher_split_bytes, layer_buffer_split_bytes)
        self.assertEqual(dispatcher_split_bytes(10, 4), [4, 4, 2])
        self.assertEqual(dispatcher_split_bytes(0, 4), [])

    def test_retag_events_is_shared_implementation(self):
        self.assertIs(dispatcher_retag_events, common_retag_events)

        events = [
            Event(
                event_id=1,
                cycle=0,
                operator="KNN",
                stage="distance",
                sub_stage="query",
                circuit_module="SRAM",
                operation="knn_distance",
                power=1.0,
                duration=2,
                metadata={"layer_name": "sa1"},
            )
        ]

        out = dispatcher_retag_events(
            events,
            operator="BallQuery",
            operation_replace={"knn": "ballquery"},
            metadata_updates={"native_op": "pointnet2_ballquery"},
        )

        self.assertIs(out, events)
        self.assertEqual(events[0].operator, "BallQuery")
        self.assertEqual(events[0].operation, "ballquery_distance")
        self.assertEqual(events[0].metadata["layer_name"], "sa1")
        self.assertEqual(events[0].metadata["native_op"], "pointnet2_ballquery")

    def test_dramsim3_burst_parser_is_shared_implementation(self):
        self.assertIs(memory_model_burst_bytes, scalesim_burst_bytes)

        with tempfile.TemporaryDirectory() as tmpdir:
            ini_path = f"{tmpdir}/dram.ini"
            with open(ini_path, "w", encoding="utf-8") as fh:
                fh.write(
                    "[system]\n"
                    "bus_width = 128\n"
                    "\n"
                    "[dram_structure]\n"
                    "BL = 16\n"
                )

            self.assertEqual(memory_model_burst_bytes(ini_path), 256)

    def test_neurosim_conv1d_stretch_is_shared_implementation(self):
        self.assertIs(neurosim_stretch_input_conv1d, neurosim_full_stretch_input_conv1d)

        np.random.seed(7)
        a = neurosim_stretch_input_conv1d(8, 32, 3, 2)
        np.random.seed(7)
        b = neurosim_full_stretch_input_conv1d(8, 32, 3, 2)

        self.assertEqual(a.shape, (1, 15, 24))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, b))

    def test_index_bytes_for_points_is_shared_implementation(self):
        self.assertIs(dispatcher_index_bytes_for_points, contract_index_bytes_for_points)
        self.assertEqual(dispatcher_index_bytes_for_points(2), 1)
        self.assertEqual(dispatcher_index_bytes_for_points(1024), 2)
        self.assertEqual(contract_index_bytes_for_points(1024, 2), 2)
        self.assertEqual(contract_index_bytes_for_points(1024, 1), 1)

    def test_legacy_report_wrapper_is_removed(self):
        self.assertFalse(hasattr(reporter_module, "create_report_from_simulation"))


if __name__ == "__main__":
    unittest.main()
