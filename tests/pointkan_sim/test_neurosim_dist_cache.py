import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock


class NeuroSimDistCacheTest(unittest.TestCase):
    def test_run_neurosim_fc_reuses_cached_subprocess_result(self):
        import sys
        import types

        fake_core = types.SimpleNamespace(
            MemCell=type("MemCell", (), {})(),
            SubArray=type("SubArray", (), {})(),
        )
        sys.modules["neurosim_core"] = fake_core
        sys.path.insert(0, "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM")
        import neurosim_dist

        if hasattr(neurosim_dist, "_run_micsim_single_fc_ported_cached"):
            neurosim_dist._run_micsim_single_fc_ported_cached.cache_clear()

        payload = {
            "layer_read_latency_ns": 1.0,
            "layer_read_energy_pj": 2.0,
            "layer_subarray_latency_ns": 0.1,
            "layer_subarray_energy_pj": 0.2,
            "layer_adc_latency_ns": 0.3,
            "layer_adc_energy_pj": 0.4,
        }

        proc = SimpleNamespace(stdout=json.dumps(payload), stderr="", returncode=0)

        with mock.patch.object(neurosim_dist.subprocess, "run", return_value=proc) as run_mock:
            kwargs = dict(
                cin=3,
                cout=1024,
                weight_bits=12,
                input_bits=12,
                array_rows=64,
                array_cols=64,
                clk_freq_hz=1e9,
                adc_clk_freq_hz=2e8,
                sar_col_power_w=128e-6,
                avg_activity=0.5,
                avg_conductance=1e-6,
                level_output=64,
                num_col_muxed=12,
                saradc=1,
                current_mode=1,
                adc_mode=0,
                r_on_ohm=240000.0,
                r_off_ohm=24000000.0,
                cell_bit=1,
                memcell_type=2,
            )
            out1 = neurosim_dist.run_neurosim_fc(**kwargs)
            out2 = neurosim_dist.run_neurosim_fc(**kwargs)

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
