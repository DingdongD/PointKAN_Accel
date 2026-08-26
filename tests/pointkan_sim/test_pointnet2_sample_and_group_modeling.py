import unittest
from types import SimpleNamespace
from unittest import mock

from pointkan_sim.main import simulate_network
from pointkan_sim.network.activation_shape import ActivationShapeInfo
from pointkan_sim.network.pointkan_net import LayerDef, OperatorRef
from pointkan_sim.operators.common_ops import gen_sample_and_group_events
from pointkan_sim.operators.dispatcher import _gen_fps, _pipeline_conv_after_knn, dispatch_layer
from pointkan_sim.simulator.event import Event


class PointNet2SampleAndGroupModelingTest(unittest.TestCase):
    def _make_pointcim_hw(self):
        return SimpleNamespace(
            clock_period=1.0,
            is_acc=True,
            use_rram_dist=True,
            pointcim_sa_conv_pipeline=True,
            knn_conv_pipeline_mode="strict",
            bn_fuse_with_conv_linear=False,
            enable_w8a8_quant_dequant=False,
            feature_computing=None,
            inter_layer_buffer=SimpleNamespace(
                size_bytes=1 << 20,
                capacity_bytes=1 << 20,
                spill_to_dram=True,
                pipeline_mode="without_pipeline",
                double_buffering=False,
            ),
            dram=SimpleNamespace(
                dramsim3_bin="fake",
                dramsim3_config_ini="fake",
                block_size_bytes=64,
            ),
        )

    def _make_two_sa_layers(self):
        ops = [
            OperatorRef("fps", {"output_points": 2}),
            OperatorRef("ballquery", {"k": 4}),
            OperatorRef("conv2d", {"kernel_size": 1, "in_channels": 6, "num_points": 8}),
        ]
        return [
            LayerDef(name="SA1", layer_type="SA", operators=ops),
            LayerDef(name="SA2", layer_type="SA", operators=ops),
        ]

    def test_pointcim_fps_does_not_reuse_first_layer_indices(self):
        hw = SimpleNamespace(
            is_acc=True,
            use_rram_dist=True,
            distance_in_phase2=False,
            clock_period=1.0,
        )
        fps_ref = SimpleNamespace(op_type="fps", params={"input_points": 32, "output_points": 8})

        sentinel = [Event(0, 0, "FPS", operation="fps_compute", duration=7, clock_period_ns=1.0)]
        with mock.patch(
            "pointkan_sim.operators.fps.FPSOperator.generate_events",
            return_value=sentinel,
        ) as mocked_generate:
            events = _gen_fps(
                fps_ref,
                hw,
                context={"first_layer_with_fps": 0, "layer_index": 1},
            )

        self.assertEqual(events, sentinel)
        mocked_generate.assert_called_once()
        self.assertNotEqual(events[0].operation, "fps_reuse_first_layer")

    def test_dispatch_layer_pointcim_sa_conv_pipeline_does_not_emit_standalone_sample_and_group(self):
        hw = SimpleNamespace(
            use_rram_dist=True,
            is_acc=True,
            pointcim_sa_conv_pipeline=True,
            knn_conv_pipeline_mode="disabled",
            bn_fuse_with_conv_linear=False,
            enable_w8a8_quant_dequant=False,
            feature_computing=None,
        )
        ops = [
            SimpleNamespace(op_type="fps", params={"output_points": 2}),
            SimpleNamespace(op_type="ballquery", params={"k": 4}),
            SimpleNamespace(op_type="conv2d", params={"kernel_size": 1, "in_channels": 6, "num_points": 8}),
        ]
        fused_events = [
            Event(0, 0, "FPS", operation="bq_done_q0", duration=1, clock_period_ns=1.0, metadata={"query": 0}),
            Event(1, 2, "FPS", operation="bq_done_q1", duration=1, clock_period_ns=1.0, metadata={"query": 1}),
        ]

        with mock.patch(
            "pointkan_sim.operators.dispatcher._gen_pointcim_fused_fps_bq",
            return_value=fused_events,
        ), mock.patch(
            "pointkan_sim.operators.common_ops.gen_sample_and_group_events",
            side_effect=AssertionError("standalone sample_and_group should not be emitted for pointcim_sa_conv_pipeline"),
        ), mock.patch(
            "pointkan_sim.operators.dispatcher._pipeline_conv_after_knn",
            return_value=fused_events,
        ) as mocked_pipeline:
            events = dispatch_layer(ops, hw, context={"layer_index": 0, "layer_name": "SA1"})

        self.assertEqual(events, fused_events)
        mocked_pipeline.assert_called_once()

    def test_sample_and_group_feature_gather_uses_dram_when_layer_requires_spill(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            inter_layer_buffer=SimpleNamespace(capacity_bytes=1 << 20),
            xyz_sram_buffer=SimpleNamespace(size_bytes=1 << 20, spill_to_dram=False),
            feature_residency_plan_by_layer={
                "SA2": SimpleNamespace(input_dram_bytes=4096, input_sram_bytes=1024),
            },
        )

        with mock.patch(
            "pointkan_sim.operators.common_ops.query_sram_cycles_energy",
            return_value=(2, 0.2),
        ), mock.patch(
            "pointkan_sim.operators.common_ops.query_dram_cycles_energy",
            return_value=(7, 0.7),
        ):
            events, _eid, _cycles, _tag = gen_sample_and_group_events(
                sampled_points=8,
                k=4,
                xyz_channels=3,
                feature_channels=16,
                hw_config=hw,
                current_layer_name="SA2",
            )

        feature_event = next(ev for ev in events if (ev.metadata or {}).get("sub_op") == "feature_gather")
        self.assertEqual(feature_event.circuit_module, "dram")
        self.assertEqual(feature_event.metadata.get("source_tier"), "dram")

    def test_sample_and_group_xyz_gather_uses_dram_when_xyz_sram_cannot_hold_points(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            inter_layer_buffer=SimpleNamespace(capacity_bytes=1 << 20),
            xyz_sram_buffer=SimpleNamespace(size_bytes=64, spill_to_dram=True),
            feature_residency_plan_by_layer={},
        )

        with mock.patch(
            "pointkan_sim.operators.common_ops.query_sram_cycles_energy",
            return_value=(2, 0.2),
        ), mock.patch(
            "pointkan_sim.operators.common_ops.query_dram_cycles_energy",
            return_value=(9, 0.9),
        ):
            events, _eid, _cycles, _tag = gen_sample_and_group_events(
                sampled_points=8,
                k=4,
                xyz_channels=3,
                feature_channels=0,
                hw_config=hw,
            )

        xyz_event = next(ev for ev in events if (ev.metadata or {}).get("sub_op") == "xyz_gather")
        self.assertEqual(xyz_event.circuit_module, "dram")
        self.assertEqual(xyz_event.metadata.get("source_tier"), "dram")

    def test_query_level_sample_and_group_tags_are_unique_per_layer(self):
        hw = SimpleNamespace(
            knn_conv_pipeline_mode="strict",
            enable_w8a8_quant_dequant=False,
            feature_computing=None,
            clock_period=1.0,
            inter_layer_buffer=SimpleNamespace(capacity_bytes=1 << 20),
            xyz_sram_buffer=SimpleNamespace(size_bytes=64, spill_to_dram=True),
            feature_residency_plan_by_layer={
                "SA2_xyz": SimpleNamespace(input_dram_bytes=4096, input_sram_bytes=1024),
            },
        )
        conv_ref = SimpleNamespace(
            op_type="conv2d",
            params={"kernel_size": 1, "in_channels": 6, "out_channels": 32, "num_points": 4},
        )
        preceding = [
            Event(0, 10, "BallQuery", operation="bq_done_q0", duration=1, clock_period_ns=1.0, metadata={"query": 0}),
        ]

        with mock.patch(
            "pointkan_sim.operators.common_ops.query_dram_cycles_energy",
            return_value=(9, 0.9),
        ), mock.patch(
            "pointkan_sim.operators.common_ops.query_sram_cycles_energy",
            return_value=(2, 0.2),
        ):
            sa1_events = _pipeline_conv_after_knn(
                preceding,
                conv_ref,
                num_queries=1,
                hw_config=hw,
                context={"layer_name": "SA1_xyz"},
                group_size_hint=4,
                knn_in_channels=3,
                force_strict=True,
                use_sample_and_group=True,
            )
            sa2_events = _pipeline_conv_after_knn(
                preceding,
                conv_ref,
                num_queries=1,
                hw_config=hw,
                context={"layer_name": "SA2_xyz"},
                group_size_hint=4,
                knn_in_channels=6,
                force_strict=True,
                use_sample_and_group=True,
            )

        sa1_concat = next(ev for ev in sa1_events if (ev.metadata or {}).get("sub_op") == "concat")
        sa2_concat = next(ev for ev in sa2_events if (ev.metadata or {}).get("sub_op") == "concat")
        self.assertNotEqual(
            sa1_concat.metadata.get("dependency_tag"),
            sa2_concat.metadata.get("dependency_tag"),
        )
        self.assertIn("SA1_xyz", sa1_concat.metadata.get("dependency_tag", ""))
        self.assertIn("SA2_xyz", sa2_concat.metadata.get("dependency_tag", ""))

        sa1_conv = next(ev for ev in sa1_events if ev.operation == "conv2d_rram_cim")
        sa2_conv = next(ev for ev in sa2_events if ev.operation == "conv2d_rram_cim")
        self.assertEqual(
            sa1_conv.metadata.get("depends_on_tags"),
            [sa1_concat.metadata.get("dependency_tag")],
        )
        self.assertEqual(
            sa2_conv.metadata.get("depends_on_tags"),
            [sa2_concat.metadata.get("dependency_tag")],
        )

    def test_sample_and_group_gather_depends_on_query_ready_tag(self):
        hw = SimpleNamespace(
            knn_conv_pipeline_mode="strict",
            enable_w8a8_quant_dequant=False,
            feature_computing=None,
            clock_period=1.0,
            inter_layer_buffer=SimpleNamespace(capacity_bytes=1 << 20),
            xyz_sram_buffer=SimpleNamespace(size_bytes=1 << 20, spill_to_dram=False),
            feature_residency_plan_by_layer={
                "SA2_xyz": SimpleNamespace(input_dram_bytes=4096, input_sram_bytes=1024),
            },
        )
        conv_ref = SimpleNamespace(
            op_type="conv2d",
            params={"kernel_size": 1, "in_channels": 6, "out_channels": 32, "num_points": 4},
        )
        preceding = [
            Event(
                0,
                10,
                "BallQuery",
                operation="bq_done_q0",
                duration=1,
                clock_period_ns=1.0,
                metadata={"query": 0, "dependency_tag": "bq_q0_done"},
            ),
        ]

        with mock.patch(
            "pointkan_sim.operators.common_ops.query_dram_cycles_energy",
            return_value=(9, 0.9),
        ), mock.patch(
            "pointkan_sim.operators.common_ops.query_sram_cycles_energy",
            return_value=(2, 0.2),
        ):
            events = _pipeline_conv_after_knn(
                preceding,
                conv_ref,
                num_queries=1,
                hw_config=hw,
                context={"layer_name": "SA2_xyz"},
                group_size_hint=4,
                knn_in_channels=6,
                force_strict=True,
                use_sample_and_group=True,
            )

        xyz_event = next(ev for ev in events if (ev.metadata or {}).get("sub_op") == "xyz_gather")
        feat_event = next(ev for ev in events if (ev.metadata or {}).get("sub_op") == "feature_gather")
        self.assertIn("bq_q0_done", xyz_event.metadata.get("depends_on_tags", []))
        self.assertIn("bq_q0_done", feat_event.metadata.get("depends_on_tags", []))

    @mock.patch("pointkan_sim.integration.memory_model_adapter.fill_inter_layer_buffer_from_cacti", return_value=True)
    @mock.patch("pointkan_sim.integration.memory_runtime.create_memory_runtime_context", return_value=None)
    @mock.patch("pointkan_sim.memory.feature_residency.build_feature_residency_plan", return_value=[])
    @mock.patch("pointkan_sim.memory.feature_residency.plan_by_layer_name", return_value={})
    @mock.patch("pointkan_sim.memory.feature_residency.get_feature_buffer_slots", return_value=1)
    @mock.patch("pointkan_sim.integration.memory_model_adapter.run_dram_trace_for_layer_buffer_dram", return_value=[(1, 0.0)])
    @mock.patch("pointkan_sim.main.gen_layer_buffer_write_events", return_value=[])
    @mock.patch("pointkan_sim.main.gen_first_layer_input_load_events", return_value=[])
    def test_simulate_network_skips_monolithic_input_read_for_pointcim_sa_pipeline(
        self,
        _load_mock,
        _write_mock,
        _dram_trace_mock,
        _slots_mock,
        _plan_map_mock,
        _plan_mock,
        _memctx_mock,
        _fill_mock,
    ):
        hw = self._make_pointcim_hw()
        layers = self._make_two_sa_layers()
        shapes = [
            ActivationShapeInfo("SA1", "SA", (8, 3), (2, 32), 24, 64, 96, 256),
            ActivationShapeInfo("SA2", "SA", (2, 32), (1, 64), 64, 64, 256, 256),
        ]

        with mock.patch(
            "pointkan_sim.main.dispatch_layer",
            return_value=[Event(0, 0, "FPS", operation="fps_compute", duration=1, clock_period_ns=1.0)],
        ), mock.patch(
            "pointkan_sim.main.gen_layer_buffer_read_events",
            side_effect=AssertionError("monolithic SA input read should be skipped for pointcim SA pipeline"),
        ):
            simulate_network(
                layers,
                hw,
                activation_shapes=shapes,
                enable_layer_buffer=True,
                repo_root=None,
            )

    @mock.patch("pointkan_sim.integration.memory_model_adapter.fill_inter_layer_buffer_from_cacti", return_value=True)
    @mock.patch("pointkan_sim.integration.memory_runtime.create_memory_runtime_context", return_value=None)
    @mock.patch("pointkan_sim.memory.feature_residency.build_feature_residency_plan", return_value=[])
    @mock.patch("pointkan_sim.memory.feature_residency.plan_by_layer_name", return_value={})
    @mock.patch("pointkan_sim.memory.feature_residency.get_feature_buffer_slots", return_value=1)
    @mock.patch("pointkan_sim.integration.memory_model_adapter.run_dram_trace_for_layer_buffer_dram", return_value=[(1, 0.0)])
    @mock.patch("pointkan_sim.main.gen_first_layer_input_load_events", return_value=[])
    @mock.patch("pointkan_sim.main.gen_layer_buffer_read_events", return_value=[])
    def test_sa2_xyz_can_start_from_prev_xyz_end_even_if_prev_feature_write_runs_later(
        self,
        _read_mock,
        _load_mock,
        _dram_trace_mock,
        _slots_mock,
        _plan_map_mock,
        _plan_mock,
        _memctx_mock,
        _fill_mock,
    ):
        hw = self._make_pointcim_hw()
        layers = self._make_two_sa_layers()
        shapes = [
            ActivationShapeInfo("SA1", "SA", (8, 3), (2, 32), 24, 64, 96, 256),
            ActivationShapeInfo("SA2", "SA", (2, 32), (1, 64), 64, 64, 256, 256),
        ]

        def fake_dispatch(ops, _hw, context):
            layer = context["layer_name"]
            return [
                Event(0, 0, "FPS", operation=f"{layer}_fps", duration=10, clock_period_ns=1.0),
                Event(1, 20, "Conv2d", operation=f"{layer}_conv", duration=10, clock_period_ns=1.0),
            ]

        write_events = [
            Event(
                0,
                0,
                "DRAM",
                operation="inter_layer_spill_write",
                duration=100,
                clock_period_ns=1.0,
            )
        ]

        with mock.patch("pointkan_sim.main.dispatch_layer", side_effect=fake_dispatch), mock.patch(
            "pointkan_sim.main.gen_layer_buffer_write_events",
            return_value=write_events,
        ):
            results = simulate_network(
                layers,
                hw,
                activation_shapes=shapes,
                enable_layer_buffer=True,
                repo_root=None,
            )

        rows = [row for row in results["operator_sequence"] if row.get("layer_name") in {"SA1_xyz", "SA2_xyz"}]
        sa1_end = max(int(row["end_cycle"]) for row in rows if row["layer_name"] == "SA1_xyz" and row["operator"] == "FPS")
        sa2_start = min(int(row["cycle"]) for row in rows if row["layer_name"] == "SA2_xyz" and row["operator"] == "FPS")
        self.assertEqual(sa1_end + 1, sa2_start)


if __name__ == "__main__":
    unittest.main()
