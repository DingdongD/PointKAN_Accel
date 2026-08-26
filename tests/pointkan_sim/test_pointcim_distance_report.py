import unittest

from pointkan_sim.main import _summarize_pointcim_distance_events
from pointkan_sim.simulator.event import Event


class PointCimDistanceReportTest(unittest.TestCase):
    def test_parallel_cross_events_use_makespan_not_sum(self):
        events = [
            Event(
                event_id=0,
                cycle=100,
                duration=10,
                operator="FPS",
                stage="distance",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                metadata={"layer_name": "SA1", "query_index": 0, "array_idx": 0},
            ),
            Event(
                event_id=1,
                cycle=100,
                duration=10,
                operator="FPS",
                stage="distance",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                metadata={"layer_name": "SA1", "query_index": 0, "array_idx": 1},
            ),
            Event(
                event_id=2,
                cycle=112,
                duration=4,
                operator="FPS",
                stage="distance",
                operation="distance_merge_accumulate",
                clock_period_ns=1.0,
                metadata={"layer_name": "SA1", "query_index": 0},
            ),
        ]

        summary = _summarize_pointcim_distance_events(events)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["overall_makespan_cycles"], 16)
        self.assertEqual(summary["overall_makespan_ns"], 16.0)
        self.assertEqual(summary["sum_layer_makespan_ns"], 16.0)
        self.assertEqual(len(summary["layers"]), 1)
        layer = summary["layers"][0]
        self.assertEqual(layer["layer_name"], "SA1")
        self.assertEqual(layer["makespan_cycles"], 16)
        self.assertEqual(layer["query_count"], 1)
        self.assertEqual(layer["array_count"], 2)
        op_breakdown = {row["operation"]: row for row in layer["operation_breakdown"]}
        self.assertEqual(op_breakdown["rram_cross_term_distance"]["makespan_cycles"], 10)
        self.assertEqual(op_breakdown["distance_merge_accumulate"]["makespan_cycles"], 4)


if __name__ == "__main__":
    unittest.main()
