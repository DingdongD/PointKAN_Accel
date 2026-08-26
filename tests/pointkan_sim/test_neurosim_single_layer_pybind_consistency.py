import os
import unittest

from pointkan_sim.operators.neurosim_adapter import (
    estimate_conv1d_neurosim,
    estimate_linear_neurosim,
)


class NeuroSimSingleLayerPybindConsistencyTest(unittest.TestCase):
    ROOT = "/workspace/DNN_NeuroSim_V1.4/Inference_pytorch"

    @classmethod
    def setUpClass(cls):
        main_exe = os.path.join(cls.ROOT, "NeuroSIM", "main")
        if not os.path.isfile(main_exe):
            raise unittest.SkipTest(f"NeuroSim main not found: {main_exe}")
        try:
            import sys

            sys.path.insert(0, os.path.join(cls.ROOT, "NeuroSIM"))
            import neurosim_core  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest(f"neurosim_core unavailable: {exc}")

    def assert_estimates_close(self, ref, cand, latency_tol=0.05, energy_tol=0.05):
        self.assertGreater(ref.latency_ns, 0.0)
        self.assertGreater(ref.energy_nj, 0.0)
        self.assertGreater(cand.latency_ns, 0.0)
        self.assertGreater(cand.energy_nj, 0.0)
        self.assertLessEqual(abs(ref.latency_ns - cand.latency_ns), ref.latency_ns * latency_tol)
        self.assertLessEqual(abs(ref.energy_nj - cand.energy_nj), ref.energy_nj * energy_tol)

    def test_linear_pybind_matches_subprocess(self):
        kwargs = dict(
            in_features=32,
            out_features=64,
            tokens=8,
            clock_period_ns=10.0,
            neurosim_root=self.ROOT,
            clk_freq_hz=100_000_000.0,
            technode=32,
        )
        ref = estimate_linear_neurosim(**kwargs, mode="subprocess")
        cand = estimate_linear_neurosim(**kwargs, mode="pybind")
        self.assert_estimates_close(ref, cand)

    def test_conv1d_pybind_matches_subprocess(self):
        kwargs = dict(
            in_channels=16,
            out_channels=32,
            kernel_size=1,
            num_points=32,
            clock_period_ns=10.0,
            neurosim_root=self.ROOT,
            clk_freq_hz=100_000_000.0,
            technode=32,
        )
        ref = estimate_conv1d_neurosim(**kwargs, mode="subprocess")
        cand = estimate_conv1d_neurosim(**kwargs, mode="pybind")
        self.assert_estimates_close(ref, cand, latency_tol=0.05, energy_tol=0.05)


if __name__ == "__main__":
    unittest.main()
