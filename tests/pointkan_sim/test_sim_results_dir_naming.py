import argparse
import unittest

from pointkan_sim.run_sim_with_real_eval import get_sim_mode_results_dir


class SimResultsDirNamingTest(unittest.TestCase):
    def test_baseline_dir_name_includes_network_accelerator_and_points(self):
        args = argparse.Namespace(
            network="pointkan-elite",
            accelerator="baseline",
            use_scalesim_systolic=True,
            neurosim_full=False,
            neurosim_pybind=False,
            cim_backend=None,
            qd_en=0,
            knn_conv_pipeline_mode="strict",
            pointcim_sa_conv_pipeline=False,
            num_points=1024,
        )

        self.assertEqual(
            get_sim_mode_results_dir(args),
            "sim_results_pointkan-elite_baseline_scalesim_knnconv_strict_p1024",
        )

    def test_pointcim_dir_name_includes_network_backend_pipeline_and_points(self):
        args = argparse.Namespace(
            network="pointnet2-cls-ssg",
            accelerator="pointcim",
            use_scalesim_systolic=False,
            neurosim_full=True,
            neurosim_pybind=False,
            cim_backend=None,
            qd_en=0,
            knn_conv_pipeline_mode="strict",
            pointcim_sa_conv_pipeline=True,
            num_points=2048,
        )

        self.assertEqual(
            get_sim_mode_results_dir(args),
            "sim_results_pointnet2-cls-ssg_pointcim_neurosim_full_knnconv_strict_saconv_p2048",
        )


if __name__ == "__main__":
    unittest.main()
