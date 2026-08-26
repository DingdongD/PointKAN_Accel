import pytest

from pointkan_sim.integration.neurosim_full import _parse_neurosim_layer_breakdown


def test_parser_keeps_core_adc_separate_from_fine_grained_adc_lines():
    stdout = """
-------------------- Estimation of Layer 1 ----------------------
----------- ADC (or S/As and precharger for SRAM) readLatency is : 123ns
----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : 456ns
----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readLatency is : 789ns
----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : 100pJ
----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : 200pJ
----------- Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, IC, pooling and activation units) readDynamicEnergy is : 300pJ
----------- ADC fine-grained precharger readLatency is : 1ns
----------- ADC fine-grained colDelay readLatency is : 2ns
----------- ADC fine-grained senseAmp readLatency is : 3ns
----------- ADC fine-grained multilevelSenseAmp readLatency is : 4ns
----------- ADC fine-grained multilevelSAEncoder readLatency is : 5ns
----------- ADC fine-grained sarADC readLatency is : 6ns
----------- ADC fine-grained precharger readDynamicEnergy is : 7pJ
----------- ADC fine-grained senseAmp readDynamicEnergy is : 8pJ
----------- ADC fine-grained multilevelSenseAmp readDynamicEnergy is : 9pJ
----------- ADC fine-grained multilevelSAEncoder readDynamicEnergy is : 10pJ
----------- ADC fine-grained sarADC readDynamicEnergy is : 11pJ
----------- Accum fine-grained adder readLatency is : 12ns
----------- Accum fine-grained adder readDynamicEnergy is : 13pJ
""".strip()

    rows = _parse_neurosim_layer_breakdown(stdout, 1)
    row = rows[0]

    assert row["core_latency_adc_ns"] == 123.0
    assert row["core_latency_accum_ns"] == 456.0
    assert row["core_latency_other_ns"] == 789.0
    assert row["core_energy_adc_nj"] == 0.1
    assert row["core_energy_accum_nj"] == 0.2
    assert row["core_energy_other_nj"] == 0.3

    assert row["adc_latency_precharger_ns"] == 1.0
    assert row["adc_latency_col_delay_ns"] == 2.0
    assert row["adc_latency_senseamp_ns"] == 3.0
    assert row["adc_latency_mlsa_ns"] == 4.0
    assert row["adc_latency_mlencoder_ns"] == 5.0
    assert row["adc_latency_sar_ns"] == 6.0
    assert row["accum_latency_adder_ns"] == 12.0
    assert row["accum_energy_adder_nj"] == pytest.approx(0.013)
