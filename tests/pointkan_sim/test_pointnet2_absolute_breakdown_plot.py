import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


_SCRIPT_PATH = Path("/workspace/PointKAN-pytorch/scripts/plot_pointnet2_w8a8_vs_w4a4_absolute_breakdown.py")
_SPEC = importlib.util.spec_from_file_location("plot_pointnet2_w8a8_vs_w4a4_absolute_breakdown", _SCRIPT_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


class PointNet2AbsoluteBreakdownPlotTest(unittest.TestCase):
    def _write_csv(self, path: Path, rows):
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_build_compare_rows_aligns_layers_and_components(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rows_w8 = [
                {
                    "model_key": "cls",
                    "model_title": "PointNet++ Cls-SSG",
                    "layer_index": "0",
                    "layer_name": "sa1.mlp_convs.0",
                    "layer_label": "L0: sa1.0",
                    "total_energy": "100",
                    "ADC_energy": "60",
                    "Buffer_energy": "30",
                    "Others_energy": "10",
                    "ADC_pct": "60",
                    "Buffer_pct": "30",
                    "Others_pct": "10",
                }
            ]
            rows_w4 = [
                {
                    "model_key": "cls",
                    "model_title": "PointNet++ Cls-SSG",
                    "layer_index": "0",
                    "layer_name": "sa1.mlp_convs.0",
                    "layer_label": "L0: sa1.0",
                    "total_energy": "25",
                    "ADC_energy": "15",
                    "Buffer_energy": "8",
                    "Others_energy": "2",
                    "ADC_pct": "60",
                    "Buffer_pct": "32",
                    "Others_pct": "8",
                }
            ]
            p8 = tmp / "w8.csv"
            p4 = tmp / "w4.csv"
            self._write_csv(p8, rows_w8)
            self._write_csv(p4, rows_w4)

            rows8 = _MODULE._read_merged_csv(p8)
            rows4 = _MODULE._read_merged_csv(p4)
            compare_rows, component_cols, total_col = _MODULE._build_compare_rows(rows8, rows4, "cls")

        self.assertEqual(component_cols, ["ADC_energy", "Buffer_energy", "Others_energy"])
        self.assertEqual(total_col, "total_energy")
        self.assertEqual(len(compare_rows), 1)
        self.assertEqual(compare_rows[0]["layer_label"], "L0")
        self.assertEqual(compare_rows[0]["w8_total"], 100.0)
        self.assertEqual(compare_rows[0]["w4_total"], 25.0)
        self.assertEqual(compare_rows[0]["w8_values"]["ADC_energy"], 60.0)
        self.assertEqual(compare_rows[0]["w4_values"]["Buffer_energy"], 8.0)


if __name__ == "__main__":
    unittest.main()
