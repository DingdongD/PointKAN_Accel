import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


_SCRIPT_PATH = Path("/workspace/PointKAN-pytorch/scripts/plot_neurosim_layer_breakdown_batch.py")
_SPEC = importlib.util.spec_from_file_location("plot_neurosim_layer_breakdown_batch", _SCRIPT_PATH)


class NeuroSimLayerBreakdownBatchPlotTest(unittest.TestCase):
    def _load_module(self):
        if _SPEC is None or _SPEC.loader is None:
            self.fail("Batch plotting script is not importable")
        module = importlib.util.module_from_spec(_SPEC)
        _SPEC.loader.exec_module(module)
        return module

    def test_discover_inputs_finds_result_directories(self):
        module = self._load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = root / "sim_results_a"
            run_a.mkdir()
            (run_a / "neurosim_layer_breakdown.csv").write_text(
                "layer_name,core_latency_adc_ns\nlayer0,1\n",
                encoding="utf-8",
            )
            run_b = root / "nested" / "sim_results_b"
            run_b.mkdir(parents=True)
            (run_b / "neurosim_layer_breakdown.json").write_text("[]", encoding="utf-8")

            found = module.discover_breakdown_inputs(root)

        self.assertEqual(
            set(found),
            {
                run_a / "neurosim_layer_breakdown.csv",
                run_b / "neurosim_layer_breakdown.json",
            },
        )

    def test_run_batch_writes_manifest_and_per_run_outputs(self):
        module = self._load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_a = root / "sim_results_alpha"
            run_b = root / "sim_results_beta"
            run_a.mkdir()
            run_b.mkdir()
            csv_text = (
                "layer_name,layer_type,core_latency_adc_ns,core_latency_accum_ns,"
                "core_latency_other_ns,core_energy_adc_nj,core_energy_accum_nj,"
                "core_energy_other_nj,adc_subarray_energy_nj,adc_precharger_energy_nj,"
                "adc_senseamp_energy_nj,adc_mlsa_energy_nj,adc_mlencoder_energy_nj,"
                "adc_sar_energy_nj\n"
                "layer0,conv,1,2,3,4,5,6,0.1,0.2,0.3,0.4,0,0\n"
            )
            (run_a / "neurosim_layer_breakdown.csv").write_text(csv_text, encoding="utf-8")
            (run_b / "neurosim_layer_breakdown.csv").write_text(csv_text, encoding="utf-8")

            out_dir = root / "batch_plots"
            results = module.run_batch(root, out_dir)
            manifest = out_dir / "neurosim_layer_breakdown_batch_manifest.csv"

            self.assertEqual(len(results), 2)
            self.assertTrue(manifest.is_file())
            self.assertTrue((out_dir / "sim_results_alpha" / "neurosim_layer_breakdown_latency_long.csv").is_file())
            self.assertTrue((out_dir / "sim_results_beta" / "neurosim_layer_breakdown_energy_long.csv").is_file())

            with open(manifest, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_name"], "sim_results_alpha")
        self.assertIn("neurosim_layer_breakdown.csv", rows[0]["input_path"])


if __name__ == "__main__":
    unittest.main()
