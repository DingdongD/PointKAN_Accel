import unittest

from pointkan_sim.configs.hw_config import get_accelerator_config
from pointkan_sim.network.pointkan_elite_net import create_pointkan_elite_classification
from pointkan_sim.operators.common_ops import gen_kan_rram_lut_events


class StrictKATModelTest(unittest.TestCase):
    def _prepare_param_sram(self, hw):
        buf = hw.nonlinear_param_sram
        buf.block_size_bytes = 16
        buf.access_time_ns = 1.0
        buf.read_energy_nj_per_access = 0.01
        buf.write_energy_nj_per_access = 0.01
        return hw

    def _kan_ops_for_stage0(self):
        layers = create_pointkan_elite_classification()
        stage0 = next(layer for layer in layers if layer.name == "LFP-stage0")
        return [op for op in stage0.operators if op.op_type == "kan"]

    def test_elite_stage0_marks_local_and_center_kat_shapes(self):
        local, center = self._kan_ops_for_stage0()

        self.assertEqual(local.params["kat_path"], "local")
        self.assertEqual(local.params["num_points"], 512)
        self.assertEqual(local.params["sequence_len"], 512 * 24)
        self.assertEqual(local.params["k_neighbors"], 24)

        self.assertEqual(center.params["kat_path"], "center")
        self.assertEqual(center.params["num_points"], 512)
        self.assertEqual(center.params["sequence_len"], 512)
        self.assertNotIn("k_neighbors", center.params)

    def test_baseline_grf_uses_local_sequence_len_and_groupwise_coefficients(self):
        hw = self._prepare_param_sram(get_accelerator_config("baseline"))
        params = {
            "variant": "elite",
            "kat_path": "local",
            "num_points": 512,
            "sequence_len": 512 * 24,
            "in_features": 64,
            "out_features": 64,
            "num_groups": 8,
        }

        events = gen_kan_rram_lut_events(params, hw)

        self.assertEqual([event.operation for event in events], ["kan_param_read", "kan_grf_causal"])
        self.assertEqual(events[0].metadata["bytes"], 8 * 10 * 4)
        self.assertEqual(events[0].duration, 20)
        self.assertEqual(events[1].duration, 512 * 24 * 64 * 19)
        self.assertEqual(events[1].metadata["activation_input_shape"], "512×24×64")
        self.assertEqual(events[1].metadata["kat_path"], "local")

    def test_acc_rram_pwl_uses_local_sequence_len_without_sram_coeff_read(self):
        hw = self._prepare_param_sram(get_accelerator_config("acc"))
        params = {
            "variant": "elite",
            "kat_path": "local",
            "num_points": 512,
            "sequence_len": 512 * 24,
            "in_features": 64,
            "out_features": 64,
            "num_groups": 8,
            "rram_read_latency_ns": 1.6,
            "rram_read_energy_pj": 20.0,
            "rram_word_width_bits": 128,
        }

        events = gen_kan_rram_lut_events(params, hw)

        self.assertEqual([event.operation for event in events], ["kan_rram_table_read", "kan_rram_pwl"])
        self.assertEqual(events[0].duration, 40)
        self.assertAlmostEqual(events[0].energy, 0.4)
        self.assertEqual(events[0].metadata["coeff_bytes"], 8 * 10 * 4)
        self.assertEqual(events[0].metadata["rram_read_accesses"], 20)
        self.assertEqual(events[1].duration, 512 * 24 * 64 + 1)
        self.assertEqual(events[1].metadata["activation_input_shape"], "512×24×64")
        self.assertEqual(events[1].metadata["coeff_source"], "rram_resident")


if __name__ == "__main__":
    unittest.main()
