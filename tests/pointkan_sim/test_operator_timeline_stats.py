import unittest

from pointkan_sim.main import _summarize_operator_timeline_stats
from pointkan_sim.simulator.event import Event


class OperatorTimelineStatsTest(unittest.TestCase):
    def test_parallel_events_use_makespan_instead_of_duration_sum(self):
        events = [
            Event(
                event_id=0,
                cycle=100,
                duration=10,
                operator="FPS",
                stage="distance",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                metadata={"layer_name": "SA1_xyz", "array_idx": 0},
            ),
            Event(
                event_id=1,
                cycle=100,
                duration=10,
                operator="FPS",
                stage="distance",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                metadata={"layer_name": "SA1_xyz", "array_idx": 1},
            ),
            Event(
                event_id=2,
                cycle=112,
                duration=4,
                operator="FPS",
                stage="distance",
                operation="distance_merge_accumulate",
                clock_period_ns=1.0,
                metadata={"layer_name": "SA1_xyz"},
            ),
        ]

        operator_breakdown, operator_cycle_detail = _summarize_operator_timeline_stats(events)

        self.assertEqual(operator_breakdown["FPS"]["cycles"], 14)
        self.assertEqual(operator_breakdown["FPS"]["latency_ns"], 14.0)
        self.assertAlmostEqual(operator_breakdown["FPS"]["energy_nj"], 0.0)

        self.assertEqual(len(operator_cycle_detail), 1)
        row = operator_cycle_detail[0]
        self.assertEqual(row["operator"], "FPS")
        self.assertEqual(row["layer_name"], "SA1_xyz")
        self.assertEqual(row["cycles"], 14)
        self.assertEqual(row["latency_ns"], 14.0)
        self.assertEqual(row["event_count"], 3)

    def test_cycles_use_layer_makespan_while_energy_still_sums(self):
        events = [
            Event(
                event_id=0,
                cycle=0,
                duration=8,
                operator="FPS",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                power=187.5,
                metadata={"layer_name": "SA1_xyz", "array_idx": 0},
            ),
            Event(
                event_id=1,
                cycle=0,
                duration=8,
                operator="FPS",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                power=312.5,
                metadata={"layer_name": "SA1_xyz", "array_idx": 1},
            ),
            Event(
                event_id=2,
                cycle=9,
                duration=3,
                operator="FPS",
                operation="distance_merge_accumulate",
                clock_period_ns=1.0,
                power=166.66666666666666,
                metadata={"layer_name": "SA1_xyz"},
            ),
            Event(
                event_id=3,
                cycle=20,
                duration=5,
                operator="FPS",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                power=800.0,
                metadata={"layer_name": "SA2_xyz", "array_idx": 0},
            ),
            Event(
                event_id=4,
                cycle=20,
                duration=5,
                operator="FPS",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                power=1200.0,
                metadata={"layer_name": "SA2_xyz", "array_idx": 1},
            ),
            Event(
                event_id=5,
                cycle=26,
                duration=2,
                operator="FPS",
                operation="distance_merge_accumulate",
                clock_period_ns=1.0,
                power=125.0,
                metadata={"layer_name": "SA2_xyz"},
            ),
            Event(
                event_id=6,
                cycle=30,
                duration=100,
                operator="FPS",
                operation="rram_cross_term_distance",
                clock_period_ns=1.0,
                power=70.0,
                overlapped=True,
                metadata={"layer_name": "SA2_xyz", "array_idx": 2},
            ),
        ]

        operator_breakdown, operator_cycle_detail = _summarize_operator_timeline_stats(events)

        self.assertEqual(operator_breakdown["FPS"]["cycles"], 18)
        self.assertEqual(operator_breakdown["FPS"]["latency_ns"], 18.0)
        self.assertAlmostEqual(operator_breakdown["FPS"]["energy_nj"], 21.75)

        by_layer = {(row["operator"], row["layer_name"]): row for row in operator_cycle_detail}
        self.assertEqual(by_layer[("FPS", "SA1_xyz")]["cycles"], 11)
        self.assertEqual(by_layer[("FPS", "SA2_xyz")]["cycles"], 7)
        self.assertAlmostEqual(by_layer[("FPS", "SA2_xyz")]["energy_nj"], 17.25)
        self.assertEqual(by_layer[("FPS", "SA2_xyz")]["event_count"], 4)


if __name__ == "__main__":
    unittest.main()
