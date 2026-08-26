import unittest

from pointkan_sim.configs.hw_config import get_accelerator_config
from pointkan_sim.operators.distance_ops import gen_distance_events_one_batch


class AccModeRegressionTest(unittest.TestCase):
    def test_acc_distance_events_still_use_sram_cim_path(self):
        hw = get_accelerator_config("acc")

        events, next_event_id, next_cycle = gen_distance_events_one_batch(
            "KNN",
            hw,
            start_event_id=1,
            start_cycle=10,
            metadata={"query": 0, "num_points": 1024},
            dist_cycles=7,
            batch_size_m=128,
        )

        self.assertEqual([event.circuit_module for event in events], [
            "sram_cim_array",
            "sram_cim_array",
            "dist2_merge_adder",
        ])
        self.assertEqual([event.operation for event in events], [
            "k2_bit_serial",
            "two_qk_bit_serial",
            "distance_merge_accumulate",
        ])
        self.assertEqual(next_event_id, 4)
        self.assertEqual(next_cycle, 25)
        self.assertFalse(hw.use_rram_dist)
        self.assertIsNone(hw.rram_dist)


if __name__ == "__main__":
    unittest.main()
