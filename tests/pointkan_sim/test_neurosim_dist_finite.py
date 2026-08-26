import json
import math
import subprocess
import textwrap
import unittest


POINTKAN_PYTHON = "/opt/conda/envs/pointkan/bin/python"


class NeuroSimDistFiniteTest(unittest.TestCase):
    def _run_json_script(self, script):
        proc = subprocess.run(
            [POINTKAN_PYTHON, "-c", textwrap.dedent(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return json.loads(stdout_lines[-1])

    def test_cross_and_self_terms_are_finite(self):
        result = self._run_json_script(
            """
            import json
            import sys

            sys.path.insert(0, "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM")
            import neurosim_dist

            common = dict(
                weight_bits=12,
                input_bits=12,
                array_rows=128,
                array_cols=128,
                clk_freq_hz=1e9,
                technode=22,
                adc_clk_freq_hz=2e8,
                sar_col_power_w=128e-6,
                r_on_ohm=240000.0,
                r_off_ohm=24000000.0,
                cell_bit=1,
                memcell_type=2,
                avg_activity=0.5,
                avg_conductance=1e-6,
                level_output=64,
                num_col_muxed=12,
                saradc=1,
                current_mode=1,
                adc_mode=0,
            )

            print(json.dumps({
                "cross": neurosim_dist.run_neurosim_fc(cin=3, cout=1024, **common),
                "self": neurosim_dist.run_neurosim_fc(cin=3, cout=1, **common),
            }))
            """
        )

        for term_name in ("cross", "self"):
            for key in (
                "layer_read_latency_ns",
                "layer_read_energy_pj",
                "layer_subarray_latency_ns",
                "layer_subarray_energy_pj",
                "layer_adc_latency_ns",
                "layer_adc_energy_pj",
            ):
                value = result[term_name][key]
                self.assertTrue(math.isfinite(value), f"{term_name}.{key}={value}")
                self.assertGreaterEqual(value, 0.0, f"{term_name}.{key}={value}")


if __name__ == "__main__":
    unittest.main()
