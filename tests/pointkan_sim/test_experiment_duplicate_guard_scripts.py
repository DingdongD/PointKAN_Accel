from pathlib import Path
import unittest

ROOT = Path('/workspace/PointKAN-pytorch')


class ExperimentDuplicateGuardScriptsTest(unittest.TestCase):
    def _read(self, rel: str) -> str:
        return (ROOT / rel).read_text(encoding='utf-8')

    def test_experiment4_and_5_define_duplicate_guard(self):
        for rel in [
            'experiments_scripts/04_three_modes_scanobj_point_sweep.sh',
            'experiments_scripts/05_three_modes_pointnet2_all_models.sh',
        ]:
            text = self._read(rel)
            self.assertIn('ALLOW_DUPLICATE_RUNS="${ALLOW_DUPLICATE_RUNS:-0}"', text)
            self.assertIn('is_sim_results_dir_running()', text)
            self.assertIn('检测到同一 results_dir 已有运行中任务', text)
            self.assertIn('if [ "$ALLOW_DUPLICATE_RUNS" != "1" ] && is_sim_results_dir_running "$results_dir"; then', text)


if __name__ == '__main__':
    unittest.main()
