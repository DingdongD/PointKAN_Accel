import math
import unittest
from types import SimpleNamespace
from unittest import mock

from pointkan_sim.integration import rram_matrix_estimator as rme


class _BadNeuroSimDist:
    def run_neurosim_fc(self, cin, cout, **kwargs):
        if cout == 1:
            return {
                "layer_read_latency_ns": 0.0,
                "layer_read_energy_pj": float("nan"),
            }
        return {
            "layer_read_latency_ns": 0.0,
            "layer_read_energy_pj": float("nan"),
        }


class RramDistNonFiniteTest(unittest.TestCase):
    def _cfg(self):
        return SimpleNamespace(
            coord_dims=3,
            coord_bits=12,
            input_bits=12,
            array_rows=128,
            array_cols=128,
            mapping_mode="diagonal",
            word_width=128,
            _clock_period_ns=1.0,
            write_energy_pj_per_op=0.0,
            write_latency_ns_per_op=0.0,
            destiny_optimization_target="WriteEDP",
            self_term_cycles=3,
            sum_cycles=1,
            micsim_root="/workspace/MICSim_V1.0",
            micsim_clk_freq_hz=1e9,
            micsim_adc_clk_freq_hz=2e8,
            micsim_sar_column_power_uw=128.0,
            micsim_level_output=64,
            micsim_static_mem_type="rram",
            micsim_static_ron_ohm=240000.0,
            micsim_static_roff_ohm=24000000.0,
            micsim_static_cell_bit=1,
            micsim_avg_activity=0.5,
            micsim_avg_conductance=1e-6,
            micsim_num_col_muxed=12,
            micsim_saradc=1,
            micsim_current_mode=1,
            micsim_adc_mode=0,
            neurosim_dist_so_dir="/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM",
        )

    @mock.patch("pointkan_sim.integration.rram_matrix_estimator._estimate_destiny_write_cost")
    def test_compute_rram_dist_primitives_raises_on_nonfinite_neurosim_outputs(self, destiny_mock):
        destiny_mock.return_value = {
            "total_write_ops": 1,
            "write_energy_per_op_pj": 1.0,
            "total_write_energy_pj": 1.0,
            "total_write_energy_nj": 0.001,
            "total_write_energy_uj": 0.0,
            "total_write_latency_ns": 1.0,
            "total_write_latency_us": 0.0,
        }
        cfg = self._cfg()
        with mock.patch.object(rme, "_get_neurosim_dist", return_value=_BadNeuroSimDist()):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                rme.compute_rram_dist_primitives(cfg, num_points=1024)


if __name__ == "__main__":
    unittest.main()
