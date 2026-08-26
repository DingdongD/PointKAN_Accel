from pathlib import Path


ROOT = Path("/workspace")


def test_run_script_forwards_neurosim_subarray_env():
    src = (ROOT / "PointKAN-pytorch/run_sim_with_real_eval.sh").read_text()
    assert "NEUROSIM_SUBARRAY" in src
    assert "--neurosim_subarray" in src
    assert "NEUROSIM_SUBARRAY_ROWS" in src
    assert "NEUROSIM_SUBARRAY_COLS" in src
    assert "--neurosim_subarray_rows" in src
    assert "--neurosim_subarray_cols" in src


def test_python_runner_accepts_and_applies_neurosim_subarray():
    src = (ROOT / "PointKAN-pytorch/pointkan_sim/run_sim_with_real_eval.py").read_text()
    assert "--neurosim_subarray" in src
    assert "hw_config.neurosim_subarray" in src
    assert "--neurosim_subarray_rows" in src
    assert "--neurosim_subarray_cols" in src
    assert "hw_config.neurosim_subarray_rows" in src
    assert "hw_config.neurosim_subarray_cols" in src
    assert "subarray_cols=" in src
    assert "parallel_read=int(getattr(hw_config, \"neurosim_subarray_rows\"" in src


def test_subarray_sweep_script_exists_and_mentions_tilebuffer_summary():
    src = (ROOT / "PointKAN-pytorch/scripts/sweep_pointnet2_pointcim_subarray_auto.py").read_text()
    assert "--subarrays" in src
    assert "--subarray-rows" in src
    assert "--subarray-cols" in src
    assert "sum_tile_buffer_htree_latency_ns" in src
    assert "tile_buffer_share_of_other_pct" in src


def test_neurosim_main_and_processing_unit_wire_num_col_subarray():
    main_src = (ROOT / "DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/main.cpp").read_text()
    pu_src = (ROOT / "DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/ProcessingUnit.cpp").read_text()
    assert "param->numColSubArray" in main_src
    assert "subArray->numCol = param->numColSubArray;" in pu_src
