import unittest
from types import SimpleNamespace
from unittest import mock

from pointkan_sim.integration import memory_model_adapter as mma
from pointkan_sim.operators.knn import KNNOperator, KNNParams
from pointkan_sim.operators.layer_buffer_ops import gen_phase2_distance_events
from pointkan_sim.operators import common_ops


class DramBatchPreplayTest(unittest.TestCase):
    def test_query_dram_batch_cycles_energy_matches_single_query_shape(self):
        hw = SimpleNamespace(
            dram=SimpleNamespace(
                dramsim3_bin="/tmp/fake_dramsim3",
                dramsim3_config_ini="/tmp/fake.ini",
                dramsim3_channels_override=0,
            )
        )

        with mock.patch.object(
            mma,
            "_run_strict_dram_request_batch",
            return_value=[(13, 4.5), (17, 6.5)],
        ) as mocked_batch:
            out = mma.query_dram_batch_cycles_energy(
                [
                    {"num_bytes": 1024, "is_read": True, "start_cycle": 0},
                    {"num_bytes": 2048, "is_read": False, "start_cycle": 8},
                ],
                hw,
            )

        self.assertEqual(out, [(13, 4.5), (17, 6.5)])
        mocked_batch.assert_called_once()

    def test_knn_streaming_writeback_uses_batch_dram_query(self):
        hw = SimpleNamespace(clock_period=1.0)
        engine = KNNOperator(
            hw_config=hw,
            params=KNNParams(input_points=16, query_points=4, k=4),
        )

        with mock.patch(
            "pointkan_sim.operators.knn.query_dram_batch_cycles_energy",
            return_value=[(11, 1.1), (12, 1.2), (13, 1.3), (14, 1.4)],
        ) as mocked_batch, mock.patch(
            "pointkan_sim.operators.knn.query_dram_cycles_energy",
            side_effect=AssertionError("single-transfer query should not be used"),
        ):
            events = engine.generate_dram_writeback_events(
                start_cycle=100,
                start_event_id=7,
                streaming=True,
                per_query_ready_cycles=[100, 110, 120, 130],
                transpose_store=False,
            )

        self.assertEqual(len(events), 4)
        self.assertEqual([ev.duration for ev in events], [11, 12, 13, 14])
        mocked_batch.assert_called_once()

    def test_phase2_distance_query_reads_use_batch_dram_query(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            sram_ii=SimpleNamespace(active_rows=128, bits_per_axis=12, bit_slice=2, power_mw=1.0),
        )

        with mock.patch(
            "pointkan_sim.operators.layer_buffer_ops.query_dram_batch_cycles_energy",
            return_value=[(9, 0.9), (9, 0.9), (9, 0.9)],
        ) as mocked_batch:
            events = gen_phase2_distance_events(
                num_queries=3,
                num_keys=256,
                hw_config=hw,
                include_query_dram_read=True,
            )

        self.assertEqual(len([ev for ev in events if ev.circuit_module == "dram"]), 3)
        self.assertEqual([ev.duration for ev in events if ev.circuit_module == "dram"], [9, 9, 9])
        mocked_batch.assert_called_once()

    def test_interp_n3_gather_reads_use_batch_dram_query(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            logic_circuit=SimpleNamespace(
                interp_n3_engine=SimpleNamespace(
                    power=2.0,
                    weight_cycles=2,
                    weighted_sum_cycles_per_channel=3,
                    feature_parallelism=4,
                ),
            ),
        )
        params = {"src_channels": 8, "k": 3}
        query_ready_cycles = {0: 5, 1: 11}

        with mock.patch(
            "pointkan_sim.operators.common_ops.query_dram_batch_cycles_energy",
            return_value=[(7, 0.7), (8, 0.8)],
        ) as mocked_batch:
            events, _eid, _ready = common_ops._build_interp_n3_engine_events(
                params,
                hw,
                num_queries=2,
                query_ready_cycles=query_ready_cycles,
            )

        gather_events = [ev for ev in events if ev.circuit_module == "dram"]
        self.assertEqual([ev.duration for ev in gather_events], [7, 8])
        mocked_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
