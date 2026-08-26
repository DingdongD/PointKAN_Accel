#!/usr/bin/env python3
"""
End-to-end test for base_v3 distance matrix reuse + cycle-accurate NS engine.

Tests:
  1. base_v1 (baseline):   no distance_reuse, no streaming DRAM write
  2. base_v3 (event mode): distance_reuse enabled, streaming DRAM write, event-driven NS
  3. base_v3 (cycle_accurate): same as 2 but uses NSEngineCycleAccurate for KNN
  4. Verify PointKAN-Elite and PointNet++ network definitions
"""

import sys
sys.path.insert(0, "/workspace/PointKAN-pytorch")

from pointkan_sim.configs.hw_config import get_accelerator_config, list_accelerator_configs
from pointkan_sim.network.pointkan_net import OperatorRef
from pointkan_sim.network.pointkan_elite_net import create_pointkan_elite_partseg
from pointkan_sim.network.pointnet2_net import create_pointnet2_partseg_ssg, create_pointnet2_sem_ssg
from pointkan_sim.operators.dispatcher import dispatch_operator, dispatch_layer

def test_config_registry():
    print("=== Test 1: Config registry ===")
    configs = list_accelerator_configs()
    print(f"Available configs: {configs}")
    assert "base_v3" in configs, "base_v3 not in registry"

    v3 = get_accelerator_config("base_v3")
    assert v3.distance_matrix_reuse is True, "base_v3 should have distance_matrix_reuse=True"
    assert v3.is_base_v3 is True, "is_base_v3 should be True"
    assert v3.ns_simulation_mode == "event", "default ns_simulation_mode should be 'event'"

    v1 = get_accelerator_config("baseline")
    assert v1.distance_matrix_reuse is False, "baseline should NOT have distance_matrix_reuse"
    assert v1.is_base_v3 is False
    print("  PASS: config registry\n")

def test_dram_config_row_buffer():
    print("=== Test 2: DRAMConfig row buffer params ===")
    v3 = get_accelerator_config("base_v3")
    dram = v3.dram
    assert dram is not None
    assert hasattr(dram, "row_size_bytes")
    assert hasattr(dram, "tRCD_cycles")
    assert hasattr(dram, "tRP_cycles")
    assert hasattr(dram, "tCAS_cycles")
    assert hasattr(dram, "burst_cycles")
    print(f"  row_size={dram.row_size_bytes}, tRCD={dram.tRCD_cycles}, tRP={dram.tRP_cycles}, "
          f"tCAS={dram.tCAS_cycles}, burst={dram.burst_cycles}")
    print("  PASS: DRAM config\n")

def test_strided_dram_model():
    print("=== Test 3: Strided DRAM access model ===")
    from pointkan_sim.integration.memory_model_adapter import query_dram_strided_cycles_energy
    v3 = get_accelerator_config("base_v3")

    # Contiguous (stride <= block_size): should be fast
    cy_c, nj_c = query_dram_strided_cycles_energy(
        num_elements=512, element_bytes=4, stride_bytes=4,
        is_read=True, hw_config=v3,
    )
    # Strided (stride >> row_size): every access is a row miss
    cy_s, nj_s = query_dram_strided_cycles_energy(
        num_elements=512, element_bytes=4, stride_bytes=2048 * 4,
        is_read=True, hw_config=v3,
    )
    print(f"  Contiguous: {cy_c} cycles, {nj_c:.2f} nJ")
    print(f"  Strided:    {cy_s} cycles, {nj_s:.2f} nJ")
    assert cy_s > cy_c, f"Strided should be slower: {cy_s} vs {cy_c}"
    print("  PASS: strided model\n")

def test_knn_event_mode():
    print("=== Test 4: KNN event-driven (base_v3) ===")
    v3 = get_accelerator_config("base_v3")

    # SA-0 KNN with dist_matrix_dram_write
    knn_op_ref = OperatorRef(
        op_type="knn",
        params={
            "input_points": 2048,
            "query_points": 512,
            "k": 24,
            "dist_matrix_dram_write": True,
        },
    )
    events = dispatch_operator(knn_op_ref, v3)
    stages = {e.stage for e in events}
    print(f"  SA-0 KNN stages: {stages}")
    assert "dist_matrix_dram_write" in stages, "Should have dist_matrix_dram_write stage"
    wb_events = [e for e in events if e.stage == "dist_matrix_dram_write"]
    assert len(wb_events) == 512, f"Streaming should produce per-query events, got {len(wb_events)}"
    assert any(getattr(e, "overlapped", False) for e in wb_events), "Streaming events should be overlapped"
    print(f"  Streaming writeback events: {len(wb_events)} (one per query)")

    # FP KNN with distance_reuse (transpose read)
    fp_knn_ref = OperatorRef(
        op_type="knn",
        params={
            "input_points": 512,
            "query_points": 2048,
            "k": 3,
            "distance_reuse": True,
        },
    )
    fp_events = dispatch_operator(fp_knn_ref, v3)
    fp_stages = {e.stage for e in fp_events}
    print(f"  FP KNN stages: {fp_stages}")
    assert "dist_reuse_dram_read" in fp_stages, "Should have dist_reuse_dram_read stage"
    read_evts = [e for e in fp_events if e.stage == "dist_reuse_dram_read"]
    if read_evts:
        patterns = {e.metadata.get("access_pattern") for e in read_evts}
        print(f"  FP read access patterns: {patterns}")
        assert "strided_transpose" in patterns, "FP should use strided_transpose"
    print("  PASS: KNN event mode\n")

def test_knn_base_v1_no_reuse():
    print("=== Test 5: KNN base_v1 (no reuse) ===")
    v1 = get_accelerator_config("baseline")

    knn_op_ref = OperatorRef(
        op_type="knn",
        params={
            "input_points": 2048,
            "query_points": 512,
            "k": 24,
            "dist_matrix_dram_write": False,
        },
    )
    events = dispatch_operator(knn_op_ref, v1)
    stages = {e.stage for e in events}
    assert "dist_matrix_dram_write" not in stages, "base_v1 should NOT have dist_matrix_dram_write"
    print(f"  base_v1 KNN stages: {stages}")
    print("  PASS: base_v1 no reuse\n")

def test_cycle_accurate_ns():
    print("=== Test 6: Cycle-accurate NS engine ===")
    v3 = get_accelerator_config("base_v3")
    v3.ns_simulation_mode = "cycle_accurate"

    # SA KNN (compute mode)
    knn_ref = OperatorRef(
        op_type="knn",
        params={
            "input_points": 64,
            "query_points": 16,
            "k": 8,
        },
    )
    events = dispatch_operator(knn_ref, v3)
    stages = {e.stage for e in events}
    print(f"  Cycle-accurate SA stages: {stages}")
    assert any("ns_cycle" in s for s in stages), f"Should have ns_cycle_* stages: {stages}"
    total_cy = max((e.cycle + e.duration for e in events), default=0)
    print(f"  Total cycles: {total_cy}, events: {len(events)}")

    # FP KNN with distance_reuse (reuse mode)
    fp_ref = OperatorRef(
        op_type="knn",
        params={
            "input_points": 16,
            "query_points": 64,
            "k": 3,
            "distance_reuse": True,
        },
    )
    fp_events = dispatch_operator(fp_ref, v3)
    fp_stages = {e.stage for e in fp_events}
    print(f"  Cycle-accurate FP stages: {fp_stages}")
    assert any("ns_cycle" in s for s in fp_stages), f"Should have ns_cycle_* stages: {fp_stages}"
    print("  PASS: cycle-accurate NS\n")

def test_network_defs_v3():
    print("=== Test 7: Network definitions with enable_distance_reuse ===")

    # PointKAN-Elite
    layers_pke = create_pointkan_elite_partseg(enable_distance_reuse=True)
    gam_layers = [l for l in layers_pke if l.layer_type == "GAM"]
    gam0_ops = gam_layers[0].operators  # GAM-stage0
    gam0_knn = [o for o in gam0_ops if o.op_type == "knn"][0]
    assert gam0_knn.params.get("dist_matrix_dram_write") is True, "GAM-0 should have dist_matrix_dram_write"
    assert gam0_knn.params.get("distance_reuse", False) is False, "GAM-0 should NOT have distance_reuse"

    if len(gam_layers) > 1:
        gam1_knn = [o for o in gam_layers[1].operators if o.op_type == "knn"][0]
        assert gam1_knn.params.get("distance_reuse") is True, "GAM-1 should have distance_reuse"
        assert gam1_knn.params.get("dist_matrix_dram_write", False) is False

    fp_layers = [l for l in layers_pke if l.layer_type == "FP"]
    for fp in fp_layers:
        fp_knn = [o for o in fp.operators if o.op_type == "knn"]
        if fp_knn:
            assert fp_knn[0].params.get("distance_reuse") is True, f"{fp.name} should have distance_reuse"
    print(f"  PointKAN-Elite: {len(layers_pke)} layers, GAM layers: {len(gam_layers)}, FP layers: {len(fp_layers)}")

    # Same without reuse
    layers_pke_v1 = create_pointkan_elite_partseg(enable_distance_reuse=False)
    gam0_v1 = [l for l in layers_pke_v1 if l.layer_type == "GAM"][0]
    gam0_knn_v1 = [o for o in gam0_v1.operators if o.op_type == "knn"][0]
    assert gam0_knn_v1.params.get("dist_matrix_dram_write", False) is False, "Without reuse, no dram_write"

    # PointNet++ partseg
    layers_pn2 = create_pointnet2_partseg_ssg(enable_distance_reuse=True)
    sa1_ops = [l for l in layers_pn2 if l.name == "SA1"][0].operators
    sa1_knn = [o for o in sa1_ops if o.op_type == "knn"][0]
    assert sa1_knn.params.get("dist_matrix_dram_write") is True, "SA1 should have dist_matrix_dram_write"

    sa2_ops = [l for l in layers_pn2 if l.name == "SA2"][0].operators
    sa2_knn = [o for o in sa2_ops if o.op_type == "knn"][0]
    assert sa2_knn.params.get("distance_reuse") is True, "SA2 should have distance_reuse"

    fp2_ops = [l for l in layers_pn2 if l.name == "FP2"][0].operators
    fp2_knn = [o for o in fp2_ops if o.op_type == "knn"][0]
    assert fp2_knn.params.get("distance_reuse") is True, "FP2 should have distance_reuse"
    print(f"  PointNet++ partseg: {len(layers_pn2)} layers")

    # PointNet++ sem
    layers_sem = create_pointnet2_sem_ssg(enable_distance_reuse=True)
    sa_layers_sem = [l for l in layers_sem if l.layer_type == "SA"]
    for i, sal in enumerate(sa_layers_sem):
        knn_ops = [o for o in sal.operators if o.op_type == "knn"]
        if knn_ops:
            if i == 0:
                assert knn_ops[0].params.get("dist_matrix_dram_write") is True
            else:
                assert knn_ops[0].params.get("distance_reuse") is True
    print(f"  PointNet++ sem: {len(layers_sem)} layers")
    print("  PASS: network definitions\n")

def test_dispatch_knn_interp_fp():
    print("=== Test 8: FP layer dispatch (KNN + InterpN3 + Conv1d fallthrough) ===")
    v3 = get_accelerator_config("base_v3")

    fp_ops = [
        OperatorRef(
            op_type="knn",
            params={"input_points": 512, "query_points": 2048, "k": 3, "distance_reuse": True},
        ),
        OperatorRef(
            op_type="interp_n3",
            params={"num_output_points": 2048, "num_source_points": 512, "src_channels": 128, "k": 3},
        ),
        OperatorRef(
            op_type="conv1d",
            params={"in_channels": 256, "out_channels": 128, "num_points": 2048},
        ),
    ]
    try:
        events = dispatch_layer(fp_ops, v3)
        total_cy = max((e.cycle + e.duration for e in events), default=0)
        stages = {e.stage for e in events}
        print(f"  FP dispatch: {len(events)} events, {total_cy} cycles")
        print(f"  Stages: {stages}")
    except RuntimeError as e:
        if "SCALE-Sim" in str(e):
            print(f"  Conv1d skipped (SCALE-Sim required): {e}")
            # Re-test without conv1d
            events = dispatch_layer(fp_ops[:2], v3)
            total_cy = max((e.cycle + e.duration for e in events), default=0)
            print(f"  KNN+InterpN3 only: {len(events)} events, {total_cy} cycles")
        else:
            raise
    print("  PASS: FP dispatch\n")


if __name__ == "__main__":
    test_config_registry()
    test_dram_config_row_buffer()
    test_strided_dram_model()
    test_knn_event_mode()
    test_knn_base_v1_no_reuse()
    test_cycle_accurate_ns()
    test_network_defs_v3()
    test_dispatch_knn_interp_fp()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
