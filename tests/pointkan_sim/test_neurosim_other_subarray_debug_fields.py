from pathlib import Path
import re


EXPECTED_FIELDS = [
    "other_sub_tile_pe_wl_decoder_latency_ns",
    "other_sub_tile_pe_wl_driver_latency_ns",
    "other_sub_tile_pe_wl_switch_latency_ns",
    "other_sub_tile_pe_sl_switch_latency_ns",
    "other_sub_tile_pe_mux_latency_ns",
    "other_sub_tile_pe_mux_decoder_latency_ns",
    "other_sub_tile_pe_level_shifter_latency_ns",
    "other_sub_tile_pe_wl_path_latency_ns",
    "other_sub_tile_pe_mux_path_latency_ns",
    "other_sub_tile_pe_selected_path_latency_ns",
    "other_sub_tile_pe_other_multiplier",
    "other_sub_tile_pe_num_col_muxed",
    "other_sub_tile_pe_num_add",
    "other_sub_tile_pe_num_row_activity",
]

EXPECTED_CORE_BREAKDOWN_FIELDS = [
    "array_latency_ns",
    "array_energy_nj",
    "adc_latency_precharger_ns",
    "adc_latency_col_delay_ns",
    "adc_latency_senseamp_ns",
    "adc_latency_mlsa_ns",
    "adc_latency_mlencoder_ns",
    "adc_latency_sar_ns",
    "adc_energy_precharger_nj",
    "adc_energy_senseamp_nj",
    "adc_energy_mlsa_nj",
    "adc_energy_mlencoder_nj",
    "adc_energy_sar_nj",
    "accum_latency_adder_ns",
    "accum_latency_dff_ns",
    "accum_latency_shiftadd_weight_ns",
    "accum_latency_shiftadd_input_ns",
    "accum_latency_adder_tree_ns",
    "accum_latency_tile_ns",
    "accum_energy_adder_nj",
    "accum_energy_dff_nj",
    "accum_energy_shiftadd_weight_nj",
    "accum_energy_shiftadd_input_nj",
    "accum_energy_adder_tree_nj",
    "accum_energy_tile_nj",
]

EXPECTED_VISIBLE_HIDDEN_FIELDS = [
    "core_latency_other_visible_ns",
    "core_latency_other_hidden_ns",
    "core_latency_other_visible_share_pct",
    "core_latency_other_hidden_share_pct",
    "core_energy_other_visible_nj",
    "core_energy_other_hidden_nj",
]


ROOT = Path("/workspace")


def test_neurosim_full_exports_subarray_other_debug_fields():
    src = (ROOT / "PointKAN-pytorch/pointkan_sim/integration/neurosim_full.py").read_text()
    for field in EXPECTED_FIELDS:
        assert field in src


def test_sim_runner_writes_subarray_other_debug_fields_to_breakdown_csv():
    src = (ROOT / "PointKAN-pytorch/pointkan_sim/run_sim_with_real_eval.py").read_text()
    for field in EXPECTED_FIELDS:
        assert field in src


def test_pybind_exposes_subarray_other_debug_fields():
    src = (ROOT / "DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/NeuroSimPybind.cpp").read_text()
    for field in EXPECTED_FIELDS:
        assert field in src


def test_new_core_breakdown_fields_are_exported_end_to_end():
    pybind_src = (ROOT / "DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/NeuroSimPybind.cpp").read_text()
    runner_src = (ROOT / "PointKAN-pytorch/pointkan_sim/run_sim_with_real_eval.py").read_text()
    export_src = (ROOT / "PointKAN-pytorch/analysis_scripts/export_exp8_neurosim_layer_breakdown.py").read_text()
    for field in EXPECTED_CORE_BREAKDOWN_FIELDS:
        assert field in pybind_src
        assert field in runner_src or field in export_src


def test_breakdown_csv_ordered_keys_include_visible_hidden_fields():
    runner_src = (ROOT / "PointKAN-pytorch/pointkan_sim/run_sim_with_real_eval.py").read_text()
    m = re.search(r"ordered_keys = \[(.*?)\]\n\s+enriched_rows = \[\]", runner_src, re.S)
    assert m, "ordered_keys block for neurosim breakdown csv not found"
    ordered_keys_block = m.group(1)
    for field in EXPECTED_VISIBLE_HIDDEN_FIELDS:
        assert field in ordered_keys_block
