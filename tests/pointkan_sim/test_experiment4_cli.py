import subprocess
import unittest
from pathlib import Path

ROOT = Path('/workspace/PointKAN-pytorch')
SCRIPT = ROOT / 'experiments_scripts' / '04_three_modes_scanobj_point_sweep.sh'


class Experiment4CliTest(unittest.TestCase):
    def test_help_lists_mode_points_and_parallel_flags(self):
        out = subprocess.run(['bash', str(SCRIPT), '--help'], text=True, capture_output=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn('--mode MODE', out.stdout)
        self.assertIn('--points "1024 2048"', out.stdout)
        self.assertIn('--parallel N', out.stdout)


if __name__ == '__main__':
    unittest.main()
