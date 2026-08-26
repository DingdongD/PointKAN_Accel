import unittest
from types import SimpleNamespace

from pointkan_sim.integration import scalesim_adapter
from pointkan_sim.operators.dispatcher import _merge_real_conv_linear_params


class ScalesimProgressLoggingTest(unittest.TestCase):
    def test_merge_real_conv_linear_params_includes_progress_fields(self):
        op_ref = SimpleNamespace(
            op_type="conv1d",
            params={"layer_name": "fallback_name"},
        )
        context = {
            "real_conv_linear_list": [
                SimpleNamespace(
                    layer_type="conv",
                    name="sa1.conv1",
                    input_shape=(1, 64, 1024),
                    output_shape=(1, 128, 1024),
                    length=1024,
                    in_channels=64,
                    out_channels=128,
                    kernel_size=1,
                ),
                SimpleNamespace(layer_type="fc", name="head.fc"),
            ],
            "real_conv_linear_index": 0,
        }

        params = _merge_real_conv_linear_params(op_ref, context)

        self.assertEqual(params["real_module_name"], "sa1.conv1")
        self.assertEqual(params["progress_index"], 1)
        self.assertEqual(params["progress_total"], 2)

    def test_progress_prefix_formats_layer_position(self):
        prefix = scalesim_adapter._build_scalesim_progress_prefix(  # type: ignore[attr-defined]
            "conv1d",
            "sa1.conv1",
            progress_index=3,
            progress_total=12,
        )

        self.assertEqual(prefix, "SCALE-Sim progress [conv1d 3/12 sa1.conv1]")


if __name__ == "__main__":
    unittest.main()
