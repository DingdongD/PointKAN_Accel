# Conv1d 与 Linear 的 Cycle 构成与能耗计算

## 1. 模式约束

- **base_v1 / base_v2（baseline）**：Conv1d/Linear **仅**通过 SCALE-Sim 脉动阵列仿真，FXP16 计算，**不考虑量化/反量化**。若未启用 `use_scalesim_for_systolic` 或 SCALE-Sim 不可用/失败则 **raise**，不回退。
- **acc**：Conv1d/Linear 使用 MNSIM/NeuroSim 全网络或单层 RRAM-CIM；**仅在此模式下**可选 W8A8 量化/反量化。
- **其他配置**（如 baseline 但无 systolic_array）：**raise not supported**。

## 2. 通用能耗公式（所有事件）

单条事件的能耗 (nJ) 在 `pointkan_sim.simulator.event.Event.energy` 中计算：

```
energy_nj = power_mW × duration_cycles × clock_period_ns / 1000
```

- `power`：该电路模块功耗 (mW)，来自 hw_config（如 systolic_array.mac_power_mw、im2col.power、logic_circuit.*.power）。
- `duration`：该事件持续周期数 (cycles)。
- `clock_period_ns`：数字逻辑时钟周期 (ns)，默认 1 ns = 1 GHz。

算子级/层级能耗 = 该算子/层下所有事件的 `energy` 之和；总能耗 = 所有事件 `energy` 之和（见 `analysis/breakdown.py`）。

---

## 3. Conv1d（baseline，SCALE-Sim 脉动路径，FXP16）

### 3.1 Cycle 构成

| 阶段 | 含义 | 周期来源 |
|------|------|----------|
| **im2col** | 输入重排为矩阵 | `im2col_cycles = L × im2col_cpo`，其中 L = num_points，im2col_cpo = hw_config.im2col.cycles_per_output（默认 1） |
| **systolic** | N×N 脉动阵列矩阵乘 | 仅 SCALE-Sim：`run_scalesim_conv1d_layer` 的 total_cycles；未启用或失败时 **raise**，不回退 |

baseline 脉动路径为 **FXP16**，无 quant/dequant 阶段。

### 3.2 能耗构成

- **im2col**：`energy = im2col_power × im2col_cycles × clock_period_ns / 1000`（im2col_power 来自 hw_config.im2col.power，默认 15 mW）。
- **systolic**：能耗与 hw_config 联动。  
  - **MAC**：`SystolicArrayConfig.mac_power_mw`（默认 7.144 mW），1 GHz 下 1 cycle = 7.144 pJ/MAC；通用为 `macs × mac_power_mw × clock_period_ns / 1000` (nJ)。  
  - **SRAM**：`InterLayerBufferConfig.read_energy_nj_per_access` / `write_energy_nj_per_access`（CACTI 回填）× 访问次数；未回填时 SRAM 贡献为 0。  
  - **DRAM**：`DRAMConfig.read_energy_nj_per_access` / `write_energy_nj_per_access` 或 trace→DRAMSim3 结果；未回填时 DRAM 贡献为 0。

---

## 4. Linear（baseline，SCALE-Sim 脉动路径，FXP16）

### 4.1 Cycle 构成

| 阶段 | 含义 | 周期来源 |
|------|------|----------|
| **systolic** | N×N 脉动阵列矩阵乘（无 im2col） | 仅 SCALE-Sim：`run_scalesim_linear_layer` 的 total_cycles；未启用或失败时 **raise**，不回退 |

baseline 脉动路径为 **FXP16**，无 quant/dequant 阶段。

### 4.2 能耗构成

- **systolic**：与 Conv1d 相同，由 hw_config 的 MAC/SRAM/DRAM 按访问计能耗（见上节）。

---

## 5. Conv1d / Linear（acc，RRAM-CIM / MNSIM 路径）

当加速器为 **acc** 时，Conv1d/Linear 走 RRAM-CIM，不再使用 im2col + 脉动阵列。

### 5.1 有 MNSIM / NeuroSim 全网络时

- **Cycle**：由 mnsim_context 的 `mnsim_latency_ns` 与时钟周期换算，即 `cycles = latency_ns / clock_period_ns`（若存在 mnsim_clock_period_ns 则用该时钟）。
- **能耗**：直接使用 mnsim_context 的 `mnsim_energy_nj`，不再用 power×time。

### 5.2 无全网络（单层估算）

- **Cycle**：MNSIM 或 NeuroSim 单层接口（如 `estimate_rram_cim_op`、`estimate_conv1d_neurosim`、`estimate_linear_neurosim`）返回的 cycles；NeuroSim 失败时用 fallback 公式。
- **能耗**：由 MNSIM/NeuroSim 返回的 energy_nj；fallback 时为 `fallback_power_mW × latency_ns / 1000`。

**仅 acc 模式**下可启用 `enable_w8a8_quant_dequant`，在 Conv1d/Linear 前后插入 quant/dequant 事件。

---

## 6. 汇总表

| 模式 | 算子 | Cycle 组成 | 能耗计算方式 |
|------|------|------------|--------------|
| **baseline** | Conv1d | im2col_cycles + systolic_cycles（FXP16，无量化） | 脉动段由 SCALE-Sim + hw_config |
| **baseline** | Linear | systolic_cycles（FXP16，无量化） | 同上 |
| **acc** | Conv1d/Linear | MNSIM 全网络或 RRAM-CIM 单层（可含量化/反量化） | mnsim_energy_nj 或 CIM 估算 |

日志中的「每个算子 cycle 的详细构成」和 `operator_cycle_detail.txt` 按 (算子, 层名) 聚合上述各事件的 duration 与 energy，便于查看逐层 cycle 与能耗构成。
