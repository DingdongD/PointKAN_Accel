import json
import subprocess
import textwrap
import unittest


POINTKAN_PYTHON = "/opt/conda/envs/pointkan/bin/python"


class NeuroSimDistAlignmentTest(unittest.TestCase):
    def _run_json_script(self, script):
        proc = subprocess.run(
            [POINTKAN_PYTHON, "-c", textwrap.dedent(script)],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return json.loads(stdout_lines[-1])

    def test_neurosim_dist_implementation_has_no_correction_terms(self):
        with open(
            "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/NeuroSimPybind.cpp",
            "r",
            encoding="utf-8",
        ) as handle:
            source = handle.read()

        self.assertNotIn("latency_scale", source)
        self.assertNotIn("energy_correction", source)
        self.assertNotIn("param->SetTechnode(technode);", source)

    def test_neurosim_dist_tracks_micsim_for_rram_distance_primitives(self):
        micsim = self._run_json_script(
            """
            import json
            import sys

            repo_root = "/workspace/PointKAN-pytorch"
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)

            sys.path.insert(0, "/workspace/PointKAN-pytorch/hw_sim/cycle_simulator/destiny_3d_cache")
            from matrix_power_estimator import _run_micsim_single_fc

            common = dict(
                weight_bits=12,
                input_bits=12,
                array_rows=128,
                array_cols=128,
                micsim_root="/workspace/MICSim_V1.0",
                avg_activity=0.5,
                avg_conductance=1e-6,
                clk_freq_hz=1e9,
                adc_clk_freq_hz=2e8,
                sar_column_power_w=128e-6,
                debug_trace_adc=False,
                num_col_muxed=12,
                saradc=1,
                current_mode=1,
                level_output=64,
                adc_mode=0,
                static_mem_type="rram",
                static_ron_ohm=240000.0,
                static_roff_ohm=24000000.0,
                static_cell_bit=1,
            )

            print(json.dumps({
                "cross": _run_micsim_single_fc(cin=3, cout=1024, **common),
                "self": _run_micsim_single_fc(cin=3, cout=1, **common),
            }))
            """
        )

        neurosim = self._run_json_script(
            """
            import json
            import sys

            sys.path.insert(0, "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM")

            import neurosim_dist

            ns_common = dict(
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
                "cross": neurosim_dist.run_neurosim_fc(cin=3, cout=1024, **ns_common),
                "self": neurosim_dist.run_neurosim_fc(cin=3, cout=1, **ns_common),
            }))
            """
        )

        def rel_err(a, b):
            return abs(a - b) / max(abs(b), 1e-9)

        result = {
            "cross_latency_rel_err": rel_err(neurosim["cross"]["layer_read_latency_ns"], micsim["cross"]["layer_read_latency_ns"]),
            "cross_energy_rel_err": rel_err(neurosim["cross"]["layer_read_energy_pj"], micsim["cross"]["layer_read_energy_pj"]),
            "self_latency_rel_err": rel_err(neurosim["self"]["layer_read_latency_ns"], micsim["self"]["layer_read_latency_ns"]),
            "self_energy_rel_err": rel_err(neurosim["self"]["layer_read_energy_pj"], micsim["self"]["layer_read_energy_pj"]),
            "ns_cross": neurosim["cross"],
            "ns_self": neurosim["self"],
            "micsim_cross": micsim["cross"],
            "micsim_self": micsim["self"],
        }

        self.assertLess(result["cross_latency_rel_err"], 0.05, result)
        self.assertLess(result["cross_energy_rel_err"], 0.05, result)
        self.assertLess(result["self_latency_rel_err"], 0.05, result)
        self.assertLess(result["self_energy_rel_err"], 0.05, result)


if __name__ == "__main__":
    unittest.main()
