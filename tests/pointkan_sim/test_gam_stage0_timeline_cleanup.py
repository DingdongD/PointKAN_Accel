import tempfile
import unittest
from pathlib import Path

from pointkan_sim.run_sim_with_real_eval import (
    _cleanup_optional_gam_stage0_timeline_artifacts,
    _log_gam_stage0_timeline_status,
)


class GamStage0TimelineCleanupTest(unittest.TestCase):
    def test_cleanup_removes_stale_optional_files_and_logs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = [
                root / "cycle_timeline_gam_stage0_postpredict.txt",
                root / "block_intervals_gam_stage0_postpredict.json",
                root / "cycle_timeline_gam_stage0_switch.txt",
            ]
            keep = root / "cycle_timeline_gam_stage0.txt"
            for path in stale + [keep]:
                path.write_text("old", encoding="utf-8")

            logs = []
            _cleanup_optional_gam_stage0_timeline_artifacts(root, logs.append)

            for path in stale:
                self.assertFalse(path.exists(), f"{path.name} should be removed")
            self.assertTrue(keep.exists(), "main GAM-stage0 timeline should remain")
            self.assertEqual(len(logs), 1)
            self.assertIn("旧文件已清理", logs[0])
            self.assertIn("cycle_timeline_gam_stage0_postpredict.txt", logs[0])

    def test_status_logger_reports_only_missing_timelines(self):
        logs = []

        _log_gam_stage0_timeline_status(logs.append, "warmup", False)
        _log_gam_stage0_timeline_status(logs.append, "post-predict", True)
        _log_gam_stage0_timeline_status(logs.append, "switch", False)

        self.assertEqual(
            logs,
            [
                "GAM-stage0 warmup timeline 本次未生成（无对应事件）",
                "GAM-stage0 switch timeline 本次未生成（无对应事件）",
            ],
        )


if __name__ == "__main__":
    unittest.main()
