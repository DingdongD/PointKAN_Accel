from pathlib import Path


ROOT = Path("/workspace")


def test_run_script_forwards_neurosim_tilebuffer_env():
    src = (ROOT / "PointKAN-pytorch/run_sim_with_real_eval.sh").read_text()
    assert "NEUROSIM_TILE_BUFFER_CORE_SIZE_ROW" in src
    assert "NEUROSIM_TILE_BUFFER_CORE_SIZE_COL" in src
    assert "--neurosim_tile_buffer_core_size_row" in src
    assert "--neurosim_tile_buffer_core_size_col" in src


def test_python_runner_accepts_and_applies_neurosim_tilebuffer():
    src = (ROOT / "PointKAN-pytorch/pointkan_sim/run_sim_with_real_eval.py").read_text()
    assert "--neurosim_tile_buffer_core_size_row" in src
    assert "--neurosim_tile_buffer_core_size_col" in src
    assert "hw_config.neurosim_tile_buffer_core_size_row" in src
    assert "hw_config.neurosim_tile_buffer_core_size_col" in src
    assert "tile_buffer_core_size_row=getattr(hw_config, \"neurosim_tile_buffer_core_size_row\", None)" in src
    assert "tile_buffer_core_size_col=getattr(hw_config, \"neurosim_tile_buffer_core_size_col\", None)" in src


def test_adapter_and_main_wire_tilebuffer_optional_args():
    adapter_src = (ROOT / "PointKAN-pytorch/pointkan_sim/operators/neurosim_adapter.py").read_text()
    main_src = (ROOT / "DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/main.cpp").read_text()
    assert "tile_buffer_core_size_row" in adapter_src
    assert "tile_buffer_core_size_col" in adapter_src
    assert "tileBufferCoreSizeRow" in main_src
    assert "tileBufferCoreSizeCol" in main_src


def test_tilebuffer_sweep_script_exists_and_mentions_latency_outputs():
    src = (ROOT / "PointKAN-pytorch/scripts/sweep_pointnet2_pointcim_tilebuffer_auto.py").read_text()
    assert "--tilebuffer-rows" in src
    assert "--tilebuffer-cols" in src
    assert "sum_tile_buffer_htree_latency_ns" in src
    assert "tile_buffer_share_of_other_pct" in src
    assert "clk_period_ns" in src
