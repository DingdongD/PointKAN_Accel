import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from pointkan_sim import main as main_module
from pointkan_sim import run_sim_with_real_eval as real_eval_module
from pointkan_sim.integration.neurosim_full import _parse_neurosim_layer_breakdown, run_neurosim_full
from pointkan_sim.integration.real_data_eval import ConvLinearLayerInfo
from pointkan_sim.operators.common_ops import (
    gen_conv1d_rram_cim_events,
    gen_linear_rram_cim_events,
)
from pointkan_sim.operators.neurosim_adapter import NeuroSimEstimate
from pointkan_sim.operators.neurosim_adapter import (
    _estimate_rram_dist_vmm_neurosim_REMOVED,
    estimate_conv1d_neurosim,
    estimate_linear_neurosim,
)


class NeuroSimAdcCliTest(unittest.TestCase):
    def test_main_parse_args_accepts_neurosim_adc_options(self):
        argv = [
            "pointkan_sim.main",
            "--network",
            "pointkan",
            "--accelerator",
            "acc",
            "--neurosim_adc_clk_freq",
            "250000000",
            "--neurosim_sar_column_power_uw",
            "12.5",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main_module.parse_args()

        self.assertEqual(args.neurosim_adc_clk_freq, 250000000.0)
        self.assertIsNone(args.neurosim_sar_column_power_w)
        self.assertEqual(args.neurosim_sar_column_power_uw, 12.5)

    def test_real_eval_parse_args_accepts_neurosim_adc_options(self):
        argv = [
            "pointkan_sim.run_sim_with_real_eval",
            "--accelerator",
            "acc",
            "--neurosim_full",
            "--neurosim_adc_clk_freq",
            "400000000",
            "--neurosim_sar_column_power_w",
            "0.000021",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = real_eval_module._parse_args()

        self.assertEqual(args.neurosim_adc_clk_freq, 400000000.0)
        self.assertEqual(args.neurosim_sar_column_power_w, 0.000021)
        self.assertIsNone(args.neurosim_sar_column_power_uw)

    def test_conv1d_neurosim_path_forwards_adc_options(self):
        hw_config = SimpleNamespace(
            cim_backend="neurosim",
            clock_period=1.0,
            neurosim_root="/tmp/neurosim",
            neurosim_technode=22,
            neurosim_adc_clk_freq_hz=333000000.0,
            neurosim_sar_column_power_w=8.5e-6,
        )

        with mock.patch("pointkan_sim.operators.common_ops.estimate_conv1d_neurosim") as estimate_mock:
            estimate_mock.return_value = NeuroSimEstimate(
                cycles=2,
                latency_ns=2.0,
                energy_nj=0.1,
                avg_power_mw=1.0,
                source="neurosim",
            )
            gen_conv1d_rram_cim_events(
                {
                    "in_channels": 8,
                    "out_channels": 16,
                    "kernel_size": 1,
                    "num_points": 32,
                },
                hw_config,
            )

        kwargs = estimate_mock.call_args.kwargs
        self.assertEqual(kwargs["clk_freq_hz"], 1e9)
        self.assertEqual(kwargs["technode"], 22)
        self.assertEqual(kwargs["adc_clk_freq_hz"], 333000000.0)
        self.assertEqual(kwargs["sar_column_power_w"], 8.5e-6)

    def test_linear_neurosim_path_forwards_adc_options(self):
        hw_config = SimpleNamespace(
            cim_backend="neurosim",
            clock_period=2.0,
            neurosim_root="/tmp/neurosim",
            neurosim_technode=14,
            neurosim_adc_clk_freq_hz=444000000.0,
            neurosim_sar_column_power_w=9.5e-6,
        )

        with mock.patch("pointkan_sim.operators.common_ops.estimate_linear_neurosim") as estimate_mock:
            estimate_mock.return_value = NeuroSimEstimate(
                cycles=2,
                latency_ns=4.0,
                energy_nj=0.1,
                avg_power_mw=1.0,
                source="neurosim",
            )
            gen_linear_rram_cim_events(
                {
                    "in_features": 64,
                    "out_features": 32,
                    "tokens": 3,
                },
                hw_config,
            )

        kwargs = estimate_mock.call_args.kwargs
        self.assertEqual(kwargs["clk_freq_hz"], 5e8)
        self.assertEqual(kwargs["technode"], 14)
        self.assertEqual(kwargs["adc_clk_freq_hz"], 444000000.0)
        self.assertEqual(kwargs["sar_column_power_w"], 9.5e-6)

    def test_run_neurosim_full_forwards_adc_options(self):
        captured = {}

        def _fake_run(*args, **kwargs):
            captured.update(kwargs)
            return True, (
                "layer1's readLatency is: 1 ns\n"
                "layer1's readDynamicEnergy is: 2 pJ\n"
                "layer1's buffer latency is: 3 ns\n"
                "layer1's buffer readDynamicEnergy is: 4 pJ\n"
            ), ""

        with tempfile.TemporaryDirectory() as tmpdir:
            neurosim_root = f"{tmpdir}/Inference_pytorch"
            neurosim_main_dir = f"{neurosim_root}/NeuroSIM"
            import os
            os.makedirs(neurosim_main_dir, exist_ok=True)

            with mock.patch("pointkan_sim.integration.neurosim_full._run_neurosim_main", side_effect=_fake_run):
                result = run_neurosim_full(
                    [
                        ConvLinearLayerInfo(
                            name="fc1",
                            layer_type="fc",
                            in_features=8,
                            out_features=4,
                            tokens=1,
                        )
                    ],
                    neurosim_root=neurosim_root,
                    clk_freq_hz=100000000.0,
                    technode=32,
                    adc_clk_freq_hz=250000000.0,
                    sar_column_power_w=7.5e-6,
                )

        self.assertTrue(result.success)
        self.assertEqual(captured["clk_freq_hz"], 100000000.0)
        self.assertEqual(captured["technode"], 32)
        self.assertEqual(captured["adc_clk_freq_hz"], 250000000.0)
        self.assertEqual(captured["sar_column_power_w"], 7.5e-6)

    def test_run_neurosim_full_does_not_double_count_buffer_latency_or_energy(self):
        def _fake_run(*args, **kwargs):
            return True, (
                "layer1's readLatency is: 1 ns\n"
                "layer1's readDynamicEnergy is: 2 pJ\n"
                "layer1's buffer latency is: 3 ns\n"
                "layer1's buffer readDynamicEnergy is: 4 pJ\n"
            ), ""

        with tempfile.TemporaryDirectory() as tmpdir:
            neurosim_root = f"{tmpdir}/Inference_pytorch"
            neurosim_main_dir = f"{neurosim_root}/NeuroSIM"
            import os
            os.makedirs(neurosim_main_dir, exist_ok=True)

            with mock.patch("pointkan_sim.integration.neurosim_full._run_neurosim_main", side_effect=_fake_run):
                result = run_neurosim_full(
                    [
                        ConvLinearLayerInfo(
                            name="fc1",
                            layer_type="fc",
                            in_features=8,
                            out_features=4,
                            tokens=1,
                        )
                    ],
                    neurosim_root=neurosim_root,
                    clk_freq_hz=100000000.0,
                    technode=32,
                )

        self.assertTrue(result.success)
        self.assertEqual(result.layer_latency_ns, [1.0])
        self.assertEqual(result.layer_energy_nj, [0.002])

    def test_parse_neurosim_layer_breakdown_extracts_per_layer_components(self):
        stdout = (
            "-------------------- Estimation of Layer 1 ----------------------\n"
            "layer1's readLatency is: 10 ns\n"
            "layer1's readDynamicEnergy is: 20 pJ\n"
            "layer1's buffer latency is: 3 ns\n"
            "layer1's buffer readDynamicEnergy is: 4 pJ\n"
            "layer1's ic latency is: 5 ns\n"
            "layer1's ic readDynamicEnergy is: 6 pJ\n"
            "----------- ADC (or S/As and precharger for SRAM) readLatency is : 7 ns\n"
            "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : 8 ns\n"
            "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : 9 ns\n"
            "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : 11 pJ\n"
            "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : 12 pJ\n"
            "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : 13 pJ\n"
            "----------- ADC sub-breakdown readDynamicEnergyArray is : 1 pJ\n"
            "----------- ADC sub-breakdown precharger is : 2 pJ\n"
            "----------- ADC sub-breakdown senseAmp (SRAM) is : 3 pJ\n"
            "----------- ADC sub-breakdown multilevelSenseAmp is : 4 pJ\n"
            "----------- ADC sub-breakdown multilevelSAEncoder is : 5 pJ\n"
            "----------- ADC sub-breakdown sarADC is : 6 pJ\n"
        )

        rows = _parse_neurosim_layer_breakdown(stdout, 1)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["read_latency_ns"], 10.0)
        self.assertAlmostEqual(row["read_energy_nj"], 0.02)
        self.assertEqual(row["buffer_latency_ns"], 3.0)
        self.assertAlmostEqual(row["buffer_energy_nj"], 0.004)
        self.assertEqual(row["ic_latency_ns"], 5.0)
        self.assertAlmostEqual(row["ic_energy_nj"], 0.006)
        self.assertEqual(row["core_latency_adc_ns"], 7.0)
        self.assertEqual(row["core_latency_accum_ns"], 8.0)
        self.assertEqual(row["core_latency_other_ns"], 9.0)
        self.assertAlmostEqual(row["core_energy_adc_nj"], 0.011)
        self.assertAlmostEqual(row["core_energy_accum_nj"], 0.012)
        self.assertAlmostEqual(row["core_energy_other_nj"], 0.013)
        self.assertAlmostEqual(row["adc_subarray_energy_nj"], 0.001)
        self.assertAlmostEqual(row["adc_precharger_energy_nj"], 0.002)
        self.assertAlmostEqual(row["adc_senseamp_energy_nj"], 0.003)
        self.assertAlmostEqual(row["adc_mlsa_energy_nj"], 0.004)
        self.assertAlmostEqual(row["adc_mlencoder_energy_nj"], 0.005)
        self.assertAlmostEqual(row["adc_sar_energy_nj"], 0.006)

    def test_run_neurosim_full_exposes_layer_breakdown(self):
        def _fake_run(*args, **kwargs):
            return True, (
                "-------------------- Estimation of Layer 1 ----------------------\n"
                "layer1's readLatency is: 1 ns\n"
                "layer1's readDynamicEnergy is: 2 pJ\n"
                "layer1's buffer latency is: 3 ns\n"
                "layer1's buffer readDynamicEnergy is: 4 pJ\n"
                "layer1's ic latency is: 5 ns\n"
                "layer1's ic readDynamicEnergy is: 6 pJ\n"
                "----------- ADC (or S/As and precharger for SRAM) readLatency is : 7 ns\n"
                "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : 8 ns\n"
                "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : 9 ns\n"
                "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : 10 pJ\n"
                "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : 11 pJ\n"
                "----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : 12 pJ\n"
                "----------- ADC sub-breakdown readDynamicEnergyArray is : 1 pJ\n"
            ), ""

        with tempfile.TemporaryDirectory() as tmpdir:
            neurosim_root = f"{tmpdir}/Inference_pytorch"
            neurosim_main_dir = f"{neurosim_root}/NeuroSIM"
            import os
            os.makedirs(neurosim_main_dir, exist_ok=True)
            with mock.patch("pointkan_sim.integration.neurosim_full._run_neurosim_main", side_effect=_fake_run):
                result = run_neurosim_full(
                    [
                        ConvLinearLayerInfo(
                            name="fc1",
                            layer_type="fc",
                            in_features=8,
                            out_features=4,
                            tokens=1,
                        )
                    ],
                    neurosim_root=neurosim_root,
                    clk_freq_hz=100000000.0,
                    technode=32,
                )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.layer_breakdown)
        self.assertEqual(len(result.layer_breakdown), 1)
        self.assertEqual(result.layer_breakdown[0]["core_latency_adc_ns"], 7.0)
        self.assertAlmostEqual(result.layer_breakdown[0]["core_energy_adc_nj"], 0.01)
        self.assertAlmostEqual(result.layer_breakdown[0]["adc_subarray_energy_nj"], 0.001)

    def test_estimate_conv1d_neurosim_does_not_double_count_buffer_latency_or_energy(self):
        def _fake_run(*args, **kwargs):
            return True, (
                "layer1's readLatency is: 10 ns\n"
                "layer1's readDynamicEnergy is: 20 pJ\n"
                "layer1's buffer latency is: 30 ns\n"
                "layer1's buffer readDynamicEnergy is: 40 pJ\n"
            ), ""

        with mock.patch("pointkan_sim.operators.neurosim_adapter._run_neurosim_main", side_effect=_fake_run):
            result = estimate_conv1d_neurosim(
                in_channels=8,
                out_channels=16,
                kernel_size=1,
                num_points=32,
                clock_period_ns=5.0,
            )

        self.assertEqual(result.latency_ns, 10.0)
        self.assertEqual(result.energy_nj, 0.02)
        self.assertEqual(result.cycles, 2)

    def test_estimate_linear_neurosim_does_not_double_count_buffer_latency_or_energy(self):
        def _fake_run(*args, **kwargs):
            return True, (
                "layer1's readLatency is: 10 ns\n"
                "layer1's readDynamicEnergy is: 20 pJ\n"
                "layer1's buffer latency is: 30 ns\n"
                "layer1's buffer readDynamicEnergy is: 40 pJ\n"
            ), ""

        with mock.patch("pointkan_sim.operators.neurosim_adapter._run_neurosim_main", side_effect=_fake_run):
            result = estimate_linear_neurosim(
                in_features=64,
                out_features=32,
                tokens=1,
                clock_period_ns=5.0,
            )

        self.assertEqual(result.latency_ns, 10.0)
        self.assertEqual(result.energy_nj, 0.02)
        self.assertEqual(result.cycles, 2)

    def test_estimate_rram_vmm_neurosim_does_not_double_count_buffer_latency_or_energy(self):
        def _fake_run(*args, **kwargs):
            return True, (
                "layer1's readLatency is: 10 ns\n"
                "layer1's readDynamicEnergy is: 20 pJ\n"
                "layer1's buffer latency is: 30 ns\n"
                "layer1's buffer readDynamicEnergy is: 40 pJ\n"
            ), ""

        with mock.patch("pointkan_sim.operators.neurosim_adapter._run_neurosim_main", side_effect=_fake_run):
            result = _estimate_rram_dist_vmm_neurosim_REMOVED(
                num_points=32,
                coord_dims=3,
                coord_bits=12,
                input_bits=12,
                clock_period_ns=5.0,
            )

        self.assertEqual(result.latency_ns, 10.0)
        self.assertEqual(result.energy_nj, 0.02)
        self.assertEqual(result.cycles, 2)


if __name__ == "__main__":
    unittest.main()
