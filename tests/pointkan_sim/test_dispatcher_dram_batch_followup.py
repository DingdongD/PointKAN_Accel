import unittest
from types import SimpleNamespace
from unittest import mock

from pointkan_sim.operators.dispatcher import (
    _acc_precompute_banked_startup_dram_results,
    _gen_baseline_phase2_distance_events,
    _phase2_distance_cycles_per_query,
)


class DispatcherDramBatchFollowupTest(unittest.TestCase):
    def test_banked_startup_dram_refill_uses_batch_replay(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            xyz_sram_buffer=SimpleNamespace(size_bytes=4096, spill_to_dram=False, block_size_bytes=64),
            dram=SimpleNamespace(),
        )
        bank_specs = [
            {"bank_idx": 0, "bank_keys": 64, "xyz_dram_bytes": 192, "xyz_sram_bytes": 192},
            {"bank_idx": 1, "bank_keys": 64, "xyz_dram_bytes": 192, "xyz_sram_bytes": 192},
        ]

        with mock.patch(
            "pointkan_sim.operators.dispatcher.query_dram_batch_cycles_energy",
            side_effect=[
                [(5, 0.5), (6, 0.6)],
                [(5, 0.5), (6, 0.6)],
            ],
        ) as mocked_batch, mock.patch(
            "pointkan_sim.operators.dispatcher._acc_append_buffer_fill_events",
            side_effect=[(1, 3, 0), (1, 3, 0)],
        ), mock.patch(
            "pointkan_sim.operators.dispatcher._acc_append_buffer_consume_events",
            side_effect=[(1, 2), (1, 2)],
        ), mock.patch(
            "pointkan_sim.operators.dispatcher.query_dram_cycles_energy",
            side_effect=AssertionError("single DRAM query should not be used for banked startup refill"),
        ):
            starts, results, setup_overheads, dram_cycles = _acc_precompute_banked_startup_dram_results(
                bank_specs=bank_specs,
                preload_cursor=10,
                operator_name="KNN",
                query_index=0,
                num_banks=2,
                hw_config=hw,
                clk=1.0,
                q12_cycles=4,
                memory_context=None,
            )

        self.assertEqual(starts, [10, 24])
        self.assertEqual([res["cycles"] for res in results], [5, 6])
        self.assertEqual(setup_overheads, [9, 9])
        self.assertEqual(dram_cycles, [5, 6])
        self.assertEqual(mocked_batch.call_count, 2)

    def test_baseline_phase2_distance_uses_batch_query_for_query_and_key_reads(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            is_baseline=True,
            baseline_dist2_engine=SimpleNamespace(pe_count=64, cycles=7, power=1.0),
            xyz_sram_buffer=SimpleNamespace(size_bytes=4096, spill_to_dram=False, block_size_bytes=64),
            query_xyz_sram_buffer=SimpleNamespace(size_bytes=64, spill_to_dram=False, block_size_bytes=16),
            dist_sram_buffer=SimpleNamespace(size_bytes=4096, spill_to_dram=False, block_size_bytes=64),
            dram=SimpleNamespace(),
        )

        with mock.patch(
            "pointkan_sim.operators.dispatcher.query_dram_batch_cycles_energy",
            side_effect=[
                [(5, 0.5)],
                [(11, 1.1), (12, 1.2)],
            ],
        ) as mocked_batch, mock.patch(
            "pointkan_sim.operators.dispatcher.query_dram_cycles_energy",
            side_effect=AssertionError("single DRAM query should not be used in baseline phase2 batching"),
        ), mock.patch(
            "pointkan_sim.operators.dispatcher.query_sram_cycles_energy",
            return_value=(1, 0.1),
        ):
            events, _eid, _cycle = _gen_baseline_phase2_distance_events(
                num_keys=96,
                hw_config=hw,
                operator_name="KNN",
                include_query_dram_read=True,
            )

        dram_events = [ev for ev in events if ev.circuit_module == "dram"]
        self.assertEqual([ev.duration for ev in dram_events], [5, 11, 12])
        self.assertEqual(mocked_batch.call_count, 2)

    def test_phase2_distance_cycles_per_query_uses_batch_query_for_baseline_batches(self):
        hw = SimpleNamespace(
            clock_period=1.0,
            is_baseline=True,
            distance_include_query_dram_read=True,
            baseline_dist2_engine=SimpleNamespace(pe_count=64, cycles=7, power=1.0),
            xyz_sram_buffer=SimpleNamespace(),
            sram_ii=None,
            dram=SimpleNamespace(),
        )

        with mock.patch(
            "pointkan_sim.operators.dispatcher.query_dram_batch_cycles_energy",
            side_effect=[
                [(10, 1.0), (14, 1.4)],
                [(3, 0.3)],
            ],
        ) as mocked_batch, mock.patch(
            "pointkan_sim.operators.dispatcher.query_dram_cycles_energy",
            side_effect=AssertionError("single DRAM query should not be used in baseline phase2 cycle estimate"),
        ), mock.patch(
            "pointkan_sim.operators.dispatcher.query_sram_cycles_energy",
            side_effect=[(2, 0.2), (2, 0.2), (2, 0.2), (2, 0.2)],
        ):
            cyc = _phase2_distance_cycles_per_query(96, hw)

        self.assertEqual(cyc, (10 + 2 + 2 + 7) + (14 + 2 + 2 + 7) + 3)
        self.assertEqual(mocked_batch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
