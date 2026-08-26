import unittest
from unittest import mock

from pointkan_sim.configs.hw_config import get_accelerator_config
from pointkan_sim.operators.knn import KNNOperator, KNNParams
from pointkan_sim.operators.ns_engine_cycle import NSEngineCycleAccurate


class BaseV3StrictDramTest(unittest.TestCase):
    def test_distance_reuse_transpose_read_uses_matrix_trace_query(self):
        hw = get_accelerator_config("base_v3")
        params = KNNParams(query_points=4, input_points=8, k=3, is_transpose_read=False)
        op = KNNOperator(params=params, hw_config=hw)

        with mock.patch(
            "pointkan_sim.operators.knn.query_dram_matrix_cycles_energy",
            return_value=(123, 4.5),
        ) as mocked_matrix_query:
            events = op.generate_distance_reuse_events(transpose_store=True)

        self.assertEqual(events[0].duration, 123)
        self.assertEqual(events[0].metadata["access_pattern"], "bulk_transpose")
        mocked_matrix_query.assert_called_once_with(
            4,
            8,
            element_bytes=4,
            is_read=True,
            hw_config=hw,
            clock_period_ns=hw.clock_period,
            transpose_access=True,
        )

    def test_distance_reuse_transpose_writeback_uses_matrix_trace_query(self):
        hw = get_accelerator_config("base_v3")
        params = KNNParams(query_points=4, input_points=8, k=3)
        op = KNNOperator(params=params, hw_config=hw)

        with mock.patch(
            "pointkan_sim.operators.knn.query_dram_matrix_cycles_energy",
            return_value=(211, 8.0),
        ) as mocked_matrix_query:
            events = op.generate_dram_writeback_events(
                start_cycle=0,
                start_event_id=0,
                streaming=False,
                transpose_store=True,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].duration, 211)
        mocked_matrix_query.assert_called_once_with(
            4,
            8,
            element_bytes=4,
            is_read=False,
            hw_config=hw,
            clock_period_ns=hw.clock_period,
            transpose_access=True,
        )

    def test_ns_engine_reuse_transpose_read_uses_matrix_trace_query(self):
        hw = get_accelerator_config("base_v3")
        params = KNNParams(query_points=4, input_points=8, k=3, distance_reuse=True, is_transpose_read=False)
        engine = NSEngineCycleAccurate(hw, params)

        with mock.patch(
            "pointkan_sim.operators.ns_engine_cycle.query_dram_matrix_cycles_energy",
            return_value=(77, 2.0),
        ) as mocked_matrix_query:
            events = engine.run()

        self.assertEqual(events[0].duration, 77)
        self.assertEqual(events[0].metadata["access_pattern"], "bulk_transpose")
        mocked_matrix_query.assert_called_once_with(
            4,
            8,
            element_bytes=4,
            is_read=True,
            hw_config=hw,
            clock_period_ns=engine.clock_ns,
            transpose_access=True,
        )


if __name__ == "__main__":
    unittest.main()
