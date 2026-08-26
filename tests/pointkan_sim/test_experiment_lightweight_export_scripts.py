from pathlib import Path
import unittest

ROOT = Path('/workspace/PointKAN-pytorch')


class ExperimentLightweightExportScriptsTest(unittest.TestCase):
    def _read(self, rel: str) -> str:
        return (ROOT / rel).read_text(encoding='utf-8')

    def test_experiment4_and_5_expose_lightweight_export_and_pass_it_to_all_modes(self):
        for rel in [
            'experiments_scripts/04_three_modes_scanobj_point_sweep.sh',
            'experiments_scripts/05_three_modes_pointnet2_all_models.sh',
        ]:
            text = self._read(rel)
            self.assertIn('LIGHTWEIGHT_EXPORT="${LIGHTWEIGHT_EXPORT:-1}"', text)
            self.assertGreaterEqual(text.count('LIGHTWEIGHT_EXPORT="$LIGHTWEIGHT_EXPORT"'), 3)


if __name__ == '__main__':
    unittest.main()
