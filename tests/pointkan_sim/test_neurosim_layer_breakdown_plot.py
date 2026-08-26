import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_SCRIPT_PATH = Path("/workspace/PointKAN-pytorch/scripts/plot_neurosim_layer_breakdown.py")
_SPEC = importlib.util.spec_from_file_location("plot_neurosim_layer_breakdown", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class NeuroSimLayerBreakdownPlotTest(unittest.TestCase):
    def test_find_breakdown_input_prefers_csv_then_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "neurosim_layer_breakdown.csv"
            json_path = root / "neurosim_layer_breakdown.json"
            csv_path.write_text("layer_name,latency_ns\nfc1,1.0\n", encoding="utf-8")
            json_path.write_text("[]", encoding="utf-8")

            found = _MODULE.find_breakdown_input(root)

        self.assertEqual(found, csv_path)

    def test_load_breakdown_rows_from_csv_and_build_series(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "neurosim_layer_breakdown.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "layer_name",
                        "layer_type",
                        "latency_ns",
                        "energy_nj",
                        "core_latency_adc_ns",
                        "core_latency_accum_ns",
                        "core_latency_other_ns",
                        "core_energy_adc_nj",
                        "core_energy_accum_nj",
                        "core_energy_other_nj",
                        "adc_subarray_energy_nj",
                        "adc_precharger_energy_nj",
                        "adc_senseamp_energy_nj",
                        "adc_mlsa_energy_nj",
                        "adc_mlencoder_energy_nj",
                        "adc_sar_energy_nj",
                    ],
                )
                w.writeheader()
                w.writerow(
                    {
                        "layer_name": "SA1_feat",
                        "layer_type": "conv",
                        "latency_ns": "10",
                        "energy_nj": "20",
                        "core_latency_adc_ns": "3",
                        "core_latency_accum_ns": "2",
                        "core_latency_other_ns": "5",
                        "core_energy_adc_nj": "11",
                        "core_energy_accum_nj": "4",
                        "core_energy_other_nj": "5",
                        "adc_subarray_energy_nj": "1",
                        "adc_precharger_energy_nj": "2",
                        "adc_senseamp_energy_nj": "3",
                        "adc_mlsa_energy_nj": "1.5",
                        "adc_mlencoder_energy_nj": "1.0",
                        "adc_sar_energy_nj": "2.5",
                    }
                )

            rows = _MODULE.load_breakdown_rows(csv_path)
            latency_series = _MODULE.build_latency_series(rows)
            energy_series = _MODULE.build_energy_series(rows)
            adc_series = _MODULE.build_adc_energy_series(rows)

        self.assertEqual(rows[0]["layer_name"], "SA1_feat")
        self.assertEqual(latency_series["labels"], ["SA1_feat"])
        self.assertEqual(latency_series["components"]["ADC"], [3.0])
        self.assertEqual(latency_series["components"]["Accum"], [2.0])
        self.assertEqual(latency_series["components"]["Other"], [5.0])
        self.assertEqual(energy_series["components"]["ADC"], [11.0])
        self.assertEqual(energy_series["components"]["Accum"], [4.0])
        self.assertEqual(energy_series["components"]["Other"], [5.0])
        self.assertEqual(adc_series["components"]["Array"], [1.0])
        self.assertEqual(adc_series["components"]["Precharger"], [2.0])
        self.assertEqual(adc_series["components"]["SAR ADC"], [2.5])

    def test_write_plot_tables_emits_long_csv_files(self):
        rows = [
            {
                "layer_name": "fc1",
                "layer_type": "fc",
                "core_latency_adc_ns": 1.0,
                "core_latency_accum_ns": 2.0,
                "core_latency_other_ns": 3.0,
                "core_energy_adc_nj": 4.0,
                "core_energy_accum_nj": 5.0,
                "core_energy_other_nj": 6.0,
                "adc_subarray_energy_nj": 0.5,
                "adc_precharger_energy_nj": 0.6,
                "adc_senseamp_energy_nj": 0.7,
                "adc_mlsa_energy_nj": 0.8,
                "adc_mlencoder_energy_nj": 0.9,
                "adc_sar_energy_nj": 1.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            written = _MODULE.write_plot_tables(rows, out_dir)

            latency_csv = out_dir / "neurosim_layer_breakdown_latency_long.csv"
            energy_csv = out_dir / "neurosim_layer_breakdown_energy_long.csv"
            adc_csv = out_dir / "neurosim_layer_breakdown_adc_energy_long.csv"

            self.assertIn(latency_csv, written)
            self.assertIn(energy_csv, written)
            self.assertIn(adc_csv, written)

            with open(adc_csv, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

        self.assertGreaterEqual(len(lines), 2)
        self.assertIn("component,value_nj", lines[0])

    def test_load_breakdown_rows_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "neurosim_layer_breakdown.json"
            json_path.write_text(
                json.dumps(
                    [
                        {
                            "layer_name": "classifier",
                            "layer_type": "fc",
                            "core_latency_adc_ns": 1.25,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            rows = _MODULE.load_breakdown_rows(json_path)

        self.assertEqual(rows[0]["layer_name"], "classifier")
        self.assertEqual(rows[0]["core_latency_adc_ns"], 1.25)


if __name__ == "__main__":
    unittest.main()
