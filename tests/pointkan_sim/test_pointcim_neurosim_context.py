import argparse
import types
import unittest
from unittest import mock

from pointkan_sim.configs.hw_config import get_accelerator_config
from pointkan_sim.integration.cim_full_context import (
    CIM_CLOCK_PERIOD_NS,
    CIM_FULL_BACKEND,
    CIM_FULL_NETWORK_ACTIVE,
    CIM_LAYER_ENERGY_NJ,
    CIM_LAYER_LATENCY_NS,
)
from pointkan_sim.main import run_network_simulation, _enforce_pointcim_neurosim_backend
from pointkan_sim.network.pointkan_net import LayerDef, OperatorRef
from pointkan_sim.operators.common_ops import gen_conv1d_rram_cim_events
from pointkan_sim.operators.neurosim_adapter import NeuroSimEstimate
from pointkan_sim.operators.dispatcher import _acc_predictor_events


class PointCimNeuroSimContextTest(unittest.TestCase):
    def test_pointcim_defaults_to_neurosim_backend(self):
        hw = get_accelerator_config("pointcim")
        self.assertEqual(hw.cim_backend, "neurosim")

    def test_pointcim_backend_is_forced_back_to_neurosim_after_cli_override(self):
        hw = get_accelerator_config("pointcim")
        hw.cim_backend = "fallback"
        _enforce_pointcim_neurosim_backend(hw)
        self.assertEqual(hw.cim_backend, "neurosim")

    @mock.patch("pointkan_sim.operators.common_ops.estimate_conv1d_neurosim")
    def test_pointcim_single_layer_neurosim_fallback_uses_neurosim_clock(self, estimate_mock):
        hw = get_accelerator_config("pointcim")
        hw.clock_period = 1.0
        hw.neurosim_clk_freq_hz = 100_000_000.0
        estimate_mock.return_value = NeuroSimEstimate(
            cycles=3,
            latency_ns=30.0,
            energy_nj=1.2,
            avg_power_mw=40.0,
            source="neurosim",
        )

        events = gen_conv1d_rram_cim_events(
            {
                "in_channels": 3,
                "out_channels": 32,
                "kernel_size": 1,
                "num_points": 1024,
                "layer_name": "embedding",
            },
            hw,
        )

        estimate_mock.assert_called_once()
        kwargs = estimate_mock.call_args.kwargs
        self.assertEqual(kwargs["clk_freq_hz"], 100_000_000.0)
        self.assertEqual(kwargs["clock_period_ns"], 10.0)
        self.assertEqual(kwargs["mode"], "subprocess")
        self.assertFalse(kwargs["allow_subarray_retry"])
        self.assertEqual(events[0].clock_period_ns, 1.0)
        self.assertEqual(events[0].duration, 30)
        self.assertEqual(events[0].metadata.get("latency_ns"), 30.0)
        self.assertEqual(events[0].metadata.get("energy_nj"), 1.2)
        self.assertEqual(events[0].metadata.get("native_clock_period_ns"), 10.0)
        self.assertEqual(events[0].metadata.get("native_cycles"), 3)

    @mock.patch("pointkan_sim.operators.common_ops.estimate_conv1d_neurosim")
    def test_pointcim_single_layer_neurosim_fallback_can_use_pybind_mode(self, estimate_mock):
        hw = get_accelerator_config("pointcim")
        hw.clock_period = 1.0
        hw.neurosim_clk_freq_hz = 100_000_000.0
        hw.neurosim_pybind = True
        estimate_mock.return_value = NeuroSimEstimate(
            cycles=3,
            latency_ns=30.0,
            energy_nj=1.2,
            avg_power_mw=40.0,
            source="neurosim_pybind",
        )

        gen_conv1d_rram_cim_events(
            {
                "in_channels": 3,
                "out_channels": 32,
                "kernel_size": 1,
                "num_points": 1024,
                "layer_name": "embedding",
            },
            hw,
        )

        kwargs = estimate_mock.call_args.kwargs
        self.assertEqual(kwargs["mode"], "pybind")
        self.assertFalse(kwargs["allow_subarray_retry"])

    @mock.patch("pointkan_sim.operators.dispatcher.estimate_linear_neurosim")
    def test_pointcim_predictor_uses_neurosim_linear_estimate(self, estimate_mock):
        hw = get_accelerator_config("pointcim")
        hw.neurosim_clk_freq_hz = 100_000_000.0
        hw.neurosim_pybind = True
        estimate_mock.return_value = NeuroSimEstimate(
            cycles=3,
            latency_ns=30.0,
            energy_nj=1.2,
            avg_power_mw=40.0,
            source="neurosim_pybind",
        )

        _acc_predictor_events(
            start_event_id=0,
            base_cycle=0,
            hw_config=hw,
            context={},
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
        )

        self.assertEqual(estimate_mock.call_count, 2)
        for call in estimate_mock.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["clk_freq_hz"], 100_000_000.0)
            self.assertEqual(kwargs["mode"], "pybind")
            self.assertFalse(kwargs["allow_subarray_retry"])
        self.assertFalse(kwargs["allow_subarray_retry"])

    @mock.patch("pointkan_sim.main.print_simulation_summary")
    @mock.patch("pointkan_sim.main._attach_breakdown_and_ppa")
    @mock.patch("pointkan_sim.main.simulate_network")
    @mock.patch("pointkan_sim.integration.neurosim_full.run_neurosim_full")
    @mock.patch("pointkan_sim.main.compute_activation_shapes")
    @mock.patch("pointkan_sim.main.load_network")
    def test_run_network_simulation_passes_neurosim_full_context_for_pointcim(
        self,
        load_network_mock,
        compute_shapes_mock,
        run_neurosim_full_mock,
        simulate_network_mock,
        _attach_breakdown_mock,
        _print_summary_mock,
    ):
        load_network_mock.return_value = [
            LayerDef(
                name="embedding",
                layer_type="embedding",
                operators=[OperatorRef(op_type="conv1d", params={"in_channels": 3, "out_channels": 32, "num_points": 1024})],
            )
        ]
        compute_shapes_mock.return_value = []
        run_neurosim_full_mock.return_value = types.SimpleNamespace(
            success=True,
            layer_latency_ns=[100.0],
            layer_energy_nj=[2.0],
            clock_period_ns=10.0,
            layer_area_um2=[123.0],
            conv_linear_order=[],
            total_area_um2=123.0,
        )
        simulate_network_mock.return_value = {
            "sim_result": None,
            "layer_results": [],
            "total_cycles": 0,
            "total_energy_nJ": 0.0,
            "avg_power_mW": 0.0,
            "area_mm2": 0.0,
        }

        args = argparse.Namespace(
            network="pointkan-elite",
            accelerator="pointcim",
            input_points=1024,
            num_classes=40,
            input_channels=None,
            distance_matrix_reuse=False,
            logic_clk_freq=None,
            ns_simulation_mode="event",
            enable_w8a8_quant_dequant=False,
            bn_en=1,
            cim_backend=None,
            use_scalesim_systolic=False,
            scalesim_root=None,
            scalesim_config=None,
            knn_conv_pipeline_mode="disabled",
            fc_num_slices=1,
            fc_cores_per_slice=16,
            fc_muls_per_core=1024,
            fc_coalescing_factor=0.0,
            logic_par_quant_dequant=None,
            logic_par_layernorm=None,
            logic_par_relu=None,
            logic_par_spool=None,
            show_layers=False,
            no_layer_buffer=False,
            show_shapes=False,
            show_events=False,
            real_conv_linear_list=[types.SimpleNamespace(name="embedding")],
            neurosim_full=True,
            neurosim_timeout_s=None,
            neurosim_clk_freq=100_000_000.0,
            neurosim_technode=32,
            neurosim_adc_clk_freq=None,
            neurosim_sar_column_power_w=None,
            neurosim_sar_column_power_uw=None,
        )

        run_network_simulation(args)

        self.assertTrue(simulate_network_mock.called)
        kwargs = simulate_network_mock.call_args.kwargs
        ctx = kwargs.get("cim_full_context")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx[CIM_FULL_BACKEND], "neurosim")
        self.assertEqual(ctx[CIM_LAYER_LATENCY_NS], [100.0])
        self.assertEqual(ctx[CIM_LAYER_ENERGY_NJ], [2.0])
        self.assertEqual(ctx[CIM_CLOCK_PERIOD_NS], 10.0)
        self.assertTrue(ctx[CIM_FULL_NETWORK_ACTIVE])


if __name__ == "__main__":
    unittest.main()
