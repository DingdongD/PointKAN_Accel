import tempfile
import textwrap
import unittest
from pathlib import Path

from pointkan_sim.integration import rram_matrix_estimator as rme


class RramDistModuleCacheTest(unittest.TestCase):
    def test_get_neurosim_dist_reuses_loaded_module_for_same_wrapper_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wrapper = Path(tmpdir) / "neurosim_dist.py"
            wrapper.write_text(
                textwrap.dedent(
                    """
                    import uuid
                    TOKEN = uuid.uuid4().hex
                    """
                ),
                encoding="utf-8",
            )

            rme._NEUROSIM_DIST_MODULE_CACHE.clear()
            mod1 = rme._get_neurosim_dist(tmpdir)
            mod2 = rme._get_neurosim_dist(tmpdir)

            self.assertIs(mod1, mod2)
            self.assertEqual(mod1.TOKEN, mod2.TOKEN)


if __name__ == "__main__":
    unittest.main()
