import types
import unittest

from pointkan_sim.run_sim_with_real_eval import (
    _build_conv_linear_summary,
    _build_fallback_conv_linear_summary,
)


class ConvLinearSummaryTest(unittest.TestCase):
    def test_build_conv_linear_summary_contains_both_latency_views(self):
        neurosim_result = types.SimpleNamespace(
            layer_latency_ns=[100.0, 200.0, 50.0],
            layer_energy_nj=[1.5, 2.0, 0.5],
            chip_pipeline_latency_ns=80.0,
            clock_period_ns=10.0,
            workload_scale_factors=[1.0, 1.0, 1.0],
        )

        summary = _build_conv_linear_summary(neurosim_result, system_e2e_latency_ns=500.0)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["latency_layer_sum_ns"], 350.0)
        self.assertEqual(summary["latency_chip_pipeline_ns"], 80.0)
        self.assertEqual(summary["latency_chip_pipeline_raw_ns"], 80.0)
        self.assertTrue(summary["chip_pipeline_exact_for_workload"])
        self.assertEqual(summary["latency_chip_pipeline_plus_non_cim_ns"], 230.0)
        self.assertEqual(summary["energy_layer_sum_nj"], 4.0)
        self.assertEqual(summary["clock_period_ns"], 10.0)
        self.assertIn("latency_layer_sum_*", summary["modeling_note"])
        self.assertIn("latency_chip_pipeline_*", summary["modeling_note"])

    def test_build_conv_linear_summary_hides_unscaled_chip_pipeline_when_workload_capped(self):
        neurosim_result = types.SimpleNamespace(
            layer_latency_ns=[400.0, 600.0],
            layer_energy_nj=[4.0, 6.0],
            chip_pipeline_latency_ns=120.0,
            clock_period_ns=10.0,
            workload_scale_factors=[2.0, 1.0],
        )

        summary = _build_conv_linear_summary(neurosim_result, system_e2e_latency_ns=2000.0)

        self.assertIsNotNone(summary)
        self.assertEqual(summary["latency_layer_sum_ns"], 1000.0)
        self.assertIsNone(summary["latency_chip_pipeline_ns"])
        self.assertEqual(summary["latency_chip_pipeline_raw_ns"], 120.0)
        self.assertFalse(summary["chip_pipeline_exact_for_workload"])
        self.assertIsNone(summary["latency_chip_pipeline_plus_non_cim_ns"])

    def test_build_fallback_conv_linear_summary_reports_layer_sum_only(self):
        summary = _build_fallback_conv_linear_summary(
            {
                "Conv2d": {"latency_ns": 120.0, "energy_nj": 3.0},
                "Linear": {"latency_ns": 30.0, "energy_nj": 1.0},
                "ReLU": {"latency_ns": 999.0, "energy_nj": 5.0},
            }
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary["latency_layer_sum_ns"], 150.0)
        self.assertEqual(summary["energy_layer_sum_nj"], 4.0)
        self.assertIsNone(summary["latency_chip_pipeline_ns"])
        self.assertIn("fallback", summary["modeling_note"])


if __name__ == "__main__":
    unittest.main()
