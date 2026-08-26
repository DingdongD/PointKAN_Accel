import unittest
from unittest import mock

from pointkan_sim.configs.hw_config import get_accelerator_config
from pointkan_sim.integration import rram_matrix_estimator as estimator


class RramDistEstimatorDestinyTest(unittest.TestCase):
    @mock.patch("pointkan_sim.integration.rram_matrix_estimator._estimate_destiny_write_cost")
    @mock.patch("pointkan_sim.integration.rram_matrix_estimator._get_neurosim_dist")
    def test_write_cost_uses_destiny_helper_when_write_placeholders_are_zero(self, nd_mock, destiny_write_mock):
        hw = get_accelerator_config("pointcim")
        cfg = hw.rram_dist
        cfg.write_energy_pj_per_op = 0.0
        cfg.write_latency_ns_per_op = 0.0

        nd_mock.return_value.run_neurosim_fc.side_effect = [
            {"layer_read_latency_ns": 100.0, "layer_read_energy_pj": 1000.0},
            {"layer_read_latency_ns": 20.0, "layer_read_energy_pj": 200.0},
        ]
        destiny_write_mock.return_value = {
            "total_write_ops": 309,
            "write_energy_per_op_pj": 65.425,
            "total_write_energy_pj": 20216.325,
            "total_write_energy_nj": 20.216325,
            "total_write_energy_uj": 0.020216325,
            "total_write_latency_ns": 3157.98,
            "total_write_latency_us": 3.15798,
        }

        prim = estimator.compute_rram_dist_primitives(cfg, 1024)

        self.assertEqual(prim["write_cycles"], 3158)
        self.assertAlmostEqual(prim["write_energy_nj"], 20.216325, places=6)
        self.assertEqual(prim["write_cost"]["total_write_ops"], 309)
        self.assertEqual(prim["write_cost"]["write_energy_per_op_pj"], 65.425)
        self.assertEqual(prim["write_source"], "destiny_helper")
        destiny_write_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
