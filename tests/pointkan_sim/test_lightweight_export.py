import argparse
import unittest

from pointkan_sim.run_sim_with_real_eval import _build_export_options


class LightweightExportOptionsTest(unittest.TestCase):
    def test_default_export_keeps_heavy_artifacts_enabled(self):
        args = argparse.Namespace(
            lightweight_export=False,
            export_memory_transactions=True,
        )

        options = _build_export_options(args)

        self.assertFalse(options["lightweight_export"])
        self.assertTrue(options["embed_operator_sequence_in_report"])
        self.assertTrue(options["export_timelines"])
        self.assertTrue(options["export_operator_sequence_files"])
        self.assertTrue(options["export_runtime_memory_events"])
        self.assertTrue(options["export_memory_transactions"])

    def test_lightweight_export_disables_heavy_artifacts_but_keeps_report_breakdown_path(self):
        args = argparse.Namespace(
            lightweight_export=True,
            export_memory_transactions=True,
        )

        options = _build_export_options(args)

        self.assertTrue(options["lightweight_export"])
        self.assertFalse(options["embed_operator_sequence_in_report"])
        self.assertFalse(options["export_timelines"])
        self.assertFalse(options["export_operator_sequence_files"])
        self.assertFalse(options["export_runtime_memory_events"])
        self.assertFalse(options["export_memory_transactions"])


if __name__ == "__main__":
    unittest.main()
