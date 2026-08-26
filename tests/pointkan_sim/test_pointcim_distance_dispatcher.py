import math
import unittest
from unittest import mock

from pointkan_sim.configs.hw_config import get_accelerator_config
from pointkan_sim.operators import dispatcher as dispatcher_module


def _fill_test_memory_models(hw_config):
    for name in [
        "xyz_sram_buffer",
        "query_xyz_sram_buffer",
        "dist_sram_buffer",
        "self_distance_sram_buffer",
        "fps_min_distance_table_sram",
    ]:
        buf = getattr(hw_config, name, None)
        if buf is None:
            continue
        buf.access_time_ns = 1.0
        buf.read_energy_nj_per_access = 1.0
        buf.write_energy_nj_per_access = 1.0
        buf.block_size_bytes = max(1, int(getattr(buf, "block_size_bytes", 16) or 16))

    hw_config.dram.block_size_bytes = 64
    hw_config.dram.read_energy_nj_per_access = 2.0
    hw_config.dram.write_energy_nj_per_access = 2.0


def _fake_rram_primitives(_cfg, num_points):
    pts = max(1, int(num_points))
    points_per_array = min(32, pts)
    num_arrays = max(1, int(math.ceil(pts / 32.0)))
    write_latency_ns = float(50 + pts)
    vmm_latency_ns = float(20 + points_per_array)
    self_k2_latency_ns = float(7 + 1)
    native_clk_ns = 10.0
    single_array_energy_nj = float(points_per_array) * 0.1
    return {
        "write_cycles": max(1, int(math.ceil(write_latency_ns / native_clk_ns))),
        "write_latency_ns": write_latency_ns,
        "write_energy_nj": 5.0,
        "vmm_cycles": max(1, int(math.ceil(vmm_latency_ns / native_clk_ns))),
        "vmm_energy_nj": single_array_energy_nj * num_arrays,
        "vmm_latency_ns": vmm_latency_ns,
        "vmm_power_mw": 10.0,
        "self_k2_cycles": max(1, int(math.ceil(self_k2_latency_ns / native_clk_ns))),
        "self_k2_energy_nj": 3.0 * pts,
        "self_k2_latency_ns": self_k2_latency_ns,
        "self_term_cycles": 3,
        "sum_cycles": 1,
        "num_arrays_total": num_arrays,
        "points_per_array": points_per_array,
        "array_point_counts": [
            min(32, max(0, pts - idx * 32))
            for idx in range(num_arrays)
            if min(32, max(0, pts - idx * 32)) > 0
        ],
        "vmm_source": "fake_neurosim_dist",
    }


class PointCimDispatcherDistanceTest(unittest.TestCase):
    def _build_hw(self):
        hw = get_accelerator_config("pointcim")
        hw.clock_period = 1.0
        hw.neurosim_clk_freq_hz = 100_000_000.0
        _fill_test_memory_models(hw)
        return hw

    @mock.patch(
        "pointkan_sim.operators.dispatcher.query_dram_cycles_energy",
        return_value=(10, 2.0),
    )
    @mock.patch(
        "pointkan_sim.operators.dispatcher.compute_rram_dist_primitives",
        side_effect=_fake_rram_primitives,
    )
    def test_pointcim_distance_chain_uses_dependency_tags(self, _primitive_mock, _dram_mock):
        hw = self._build_hw()

        events, next_eid, next_cycle, block_ready_cycles, block_prefetch_cycles, block_ready_tags, block_prefetch_tags = (
            dispatcher_module._gen_acc_distance_query_events(
                start_event_id=0,
                base_cycle=0,
                operator_name="FPS",
                num_keys=128,
                hw_config=hw,
                query_index=0,
                startup_once=True,
                write_dist_buffer=False,
                tag_scope="fps_stage0_test",
            )
        )

        self.assertGreater(next_eid, 0)
        self.assertGreater(next_cycle, 0)
        # merge_parallelism = bq_merge_adder_parallelism = 64 → 128/64 = 2 batches
        expected_batches = max(1, -(-128 // max(1, int(getattr(hw, "bq_merge_adder_parallelism", 0) or 128))))
        self.assertEqual(len(block_ready_cycles), expected_batches)
        self.assertEqual(len(block_prefetch_cycles), expected_batches)
        self.assertEqual(len(block_ready_tags), expected_batches)
        self.assertEqual(len(block_prefetch_tags), expected_batches)

        ops = [str(getattr(e, "operation", "") or "") for e in events]
        expected_ops = [
            "xyz_refill",
            "xyz_quantize",
            "key_xyz_buffer_write",
            "key_xyz_buffer_read",
            "rram_program_keyset",
            "rram_self_k2_init",
            "self_distance_write",
            "xyz_buffer_query_read",
            "query_xyz_buffer_write",
            "query_xyz_buffer_read",
            "query_self_q",
            "rram_cross_term_distance",
            "self_distance_read",
            "distance_merge_accumulate",
        ]
        for op in expected_ops:
            self.assertIn(op, ops)

        event_by_op = {str(getattr(e, "operation", "") or ""): e for e in events}
        self.assertEqual(
            event_by_op["xyz_quantize"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_key_dram_ready",
        )
        self.assertEqual(
            event_by_op["key_xyz_buffer_write"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_key_quant_ready",
        )
        self.assertEqual(
            event_by_op["key_xyz_buffer_read"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_key_sram_ready",
        )
        self.assertEqual(
            event_by_op["rram_program_keyset"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_key_sram_read_ready",
        )
        self.assertEqual(
            event_by_op["rram_self_k2_init"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_rram_program_ready",
        )
        self.assertEqual(
            event_by_op["self_distance_write"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_self_k2_ready",
        )
        self.assertEqual(
            event_by_op["query_xyz_buffer_write"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_query_src_ready_q0",
        )
        self.assertEqual(
            event_by_op["query_xyz_buffer_read"].metadata.get("depends_on_tag"),
            "fps_stage0_test_pointcim_query_sram_ready_q0",
        )

        merge_events_all = [
            e for e in events
            if str(getattr(e, "operation", "") or "") == "distance_merge_accumulate"
        ]
        self.assertGreaterEqual(len(merge_events_all), 1)
        first_merge = merge_events_all[0]
        merge_tags = first_merge.metadata.get("depends_on_tags")
        self.assertIsInstance(merge_tags, list)
        self.assertIn("fps_stage0_test_pointcim_query_self_ready_q0", merge_tags)
        self.assertIn("fps_stage0_test_pointcim_cross_ready_q0", merge_tags)
        self.assertIn("fps_stage0_test_pointcim_self_read_ready_q0_b0", merge_tags)
        self.assertEqual(block_ready_tags[0], "fps_stage0_test_pointcim_merge_ready_q0_b0")
        self.assertEqual(block_prefetch_tags[0], "fps_stage0_test_pointcim_self_read_ready_q0_b0")

    @mock.patch(
        "pointkan_sim.operators.dispatcher.query_dram_cycles_energy",
        return_value=(10, 2.0),
    )
    @mock.patch(
        "pointkan_sim.operators.dispatcher.compute_rram_dist_primitives",
        side_effect=_fake_rram_primitives,
    )
    def test_pointcim_cross_uses_physical_arrays_and_merge_uses_consumer_batches(self, _primitive_mock, _dram_mock):
        hw = self._build_hw()

        events, *_rest = dispatcher_module._gen_acc_distance_query_events(
            start_event_id=0,
            base_cycle=0,
            operator_name="KNN",
            num_keys=130,
            hw_config=hw,
            query_index=0,
            startup_once=True,
            write_dist_buffer=True,
            tag_scope="knn_stage0_test",
        )

        cross_events = [
            e for e in events
            if str(getattr(e, "operation", "") or "") == "rram_cross_term_distance"
        ]
        merge_events = [
            e for e in events
            if str(getattr(e, "operation", "") or "") == "distance_merge_accumulate"
        ]

        self.assertEqual(len(cross_events), 1, "parallel arrays consolidated into 1 event per query")
        ce = cross_events[0]
        self.assertEqual(ce.metadata.get("num_arrays"), 5)
        self.assertEqual(ce.metadata.get("array_point_counts"), [32, 32, 32, 32, 2])
        self.assertTrue(ce.metadata.get("consolidated_arrays"))
        self.assertEqual(ce.clock_period_ns, 1.0)
        self.assertEqual(ce.metadata.get("native_clock_period_ns"), 10.0)
        expected_vmm_duration = max(1, int(math.ceil(float(20 + min(32, 130)) / 1.0)))
        self.assertEqual(ce.duration, expected_vmm_duration)
        # merge_parallelism = bq_merge_adder_parallelism = 64 → ceil(130/64) = 3 batches
        merge_par = max(1, int(getattr(hw, "bq_merge_adder_parallelism", 0) or 128))
        expected_merge_batches = max(1, int(math.ceil(130 / float(merge_par))))
        self.assertEqual(len(merge_events), expected_merge_batches)
        expected_merge_points = [min(merge_par, max(0, 130 - b * merge_par)) for b in range(expected_merge_batches)]
        self.assertEqual([e.metadata.get("merge_points") for e in merge_events], expected_merge_points)
        self.assertEqual([e.metadata.get("consumer_batch_idx") for e in merge_events], list(range(expected_merge_batches)))
        self.assertTrue(all(e.duration == 1 for e in merge_events))
        self.assertGreaterEqual(len(merge_events[0].metadata.get("depends_on_tags") or []), 2)
        self.assertGreaterEqual(len(merge_events[-1].metadata.get("depends_on_tags") or []), 2)


if __name__ == "__main__":
    unittest.main()
