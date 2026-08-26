import unittest
from types import SimpleNamespace

from pointkan_sim.integration.ppa_report import build_ppa_report


class PpaReportLatencyAccountingTest(unittest.TestCase):
    def test_report_exposes_busy_time_and_unattributed_gap(self):
        eval_result = SimpleNamespace(
            total_cycles=100,
            total_energy_nj=12.5,
            num_samples=1,
            accuracy=0.0,
            balanced_accuracy=0.0,
            test_loss=0.0,
        )
        full_operator_breakdown = {
            "Conv2d": {"cycles": 40, "latency_ns": 400.0, "energy_nj": 4.0, "area_um2": 10.0},
            "FPS": {"cycles": 20, "latency_ns": 200.0, "energy_nj": 8.5, "area_um2": 0.0},
        }

        report = build_ppa_report(
            eval_result,
            clock_ns=10.0,
            full_operator_breakdown=full_operator_breakdown,
            latency_per_sample_ns=1000.0,
            critical_path_latency_per_sample_ns=1000.0,
        )

        end_to_end = report["end_to_end"]
        self.assertEqual(end_to_end["latency_critical_path_ns"], 1000.0)
        self.assertEqual(end_to_end["busy_time_sum_ns"], 600.0)
        self.assertEqual(end_to_end["busy_time_sum_ms"], 0.0006)
        self.assertEqual(end_to_end["unattributed_gap_stall_ns"], 400.0)
        self.assertEqual(end_to_end["unattributed_gap_stall_ms"], 0.0004)

        totals = report["full_operator_totals"]
        self.assertEqual(totals["total_busy_time_ns"], 600.0)
        self.assertEqual(totals["critical_path_latency_ns"], 1000.0)
        self.assertEqual(totals["unattributed_gap_stall_ns"], 400.0)


if __name__ == "__main__":
    unittest.main()
