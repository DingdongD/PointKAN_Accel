import unittest

from pointkan_sim.configs.hw_config import BQLeafEngineConfig, BQMergeEngineConfig, get_accelerator_config
from pointkan_sim.operators import dispatcher as dispatcher_module


class BQEngineAreaTest(unittest.TestCase):
    def test_bq_engine_events_carry_leaf_and_merge_area_metadata(self):
        hw = get_accelerator_config("pointcim")
        hw.clock_period = 1.0
        hw.sorting_circuit.bq_engine.leaf = BQLeafEngineConfig(
            key="bq64_top32",
            block_n=64,
            output_k=32,
            latency_cycles=22,
            power_mw=133.0,
            area_um2=12345.0,
        )
        hw.sorting_circuit.bq_engine.merge = BQMergeEngineConfig(
            key="merge32",
            k=32,
            latency_cycles=2,
            power_mw=87.0,
            area_um2=6789.0,
        )

        events, _, _ = dispatcher_module._gen_bq_engine_events(
            num_points=128,
            k=32,
            hw_config=hw,
            start_cycle=0,
            start_event_id=0,
            operator_name="BallQuery",
            query_index=0,
            block_ready_cycles=None,
            tag_scope="",
        )

        leaf_events = [e for e in events if str(getattr(e, "sub_stage", "") or "") == "bq_leaf_process"]
        merge_events = [e for e in events if str(getattr(e, "sub_stage", "") or "") == "bq_merge"]

        self.assertGreaterEqual(len(leaf_events), 1)
        self.assertGreaterEqual(len(merge_events), 1)
        self.assertTrue(all(float((e.metadata or {}).get("area_um2", 0.0) or 0.0) == 12345.0 for e in leaf_events))
        self.assertTrue(all(float((e.metadata or {}).get("area_um2", 0.0) or 0.0) == 6789.0 for e in merge_events))


if __name__ == "__main__":
    unittest.main()
