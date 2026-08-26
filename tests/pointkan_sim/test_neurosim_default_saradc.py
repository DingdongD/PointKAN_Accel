from pathlib import Path


def test_neurosim_param_default_saradc_is_true():
    param_cpp = Path("/workspace/DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/Param.cpp")
    text = param_cpp.read_text(encoding="utf-8")
    assert "SARADC = true;" in text
