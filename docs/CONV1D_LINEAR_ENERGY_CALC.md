# Conv1d / Linear 能耗计算说明

## 1. 终端 breakdown 中的 “Conv1d” / “Linear” 是什么

日志里按算子汇总的 **cycles / energy_nj / latency_ns / area** 来自 `main.py` 的 `operator_breakdown`：

- 对所有仿真事件按 **算子名**（`e.operator`）分组；
- **Conv1d** = 所有 `operator=="Conv1d"` 的事件的能耗之和；
- **Linear** = 所有 `operator=="Linear"` 的事件的能耗之和。

每个 Conv1d 层会生成多类事件：

- **im2col**：数据重排，`circuit_module="im2col"`，功耗来自 `hw_config.im2col.power`；
- **systolic**：脉动阵列矩阵乘，`circuit_module="systolic_array"`，能耗即 **MAC 能耗**；
- **LayerBuffer / DRAM**：该层若启用 SCALE-Sim 内存建模，还有 SRAM/DRAM 访问事件（归属到 Conv1d/Linear 的 owner）。

因此：

- **Conv1d 总能耗** = 所有 Conv1d 层的 (im2col + systolic + 该层 SRAM/DRAM) 之和。
- **Linear 总能耗** = 所有 Linear 层的 (systolic + 该层 SRAM/DRAM) 之和（无 im2col）。

终端里看到的 “Conv1d 899,698.48 nJ”“Linear 901,805.52 nJ” 等，是上述 **整算子总能耗**（含内存与 im2col），不是“仅脉动阵列”或“仅 MAC”。仅脉动（MAC）部分通常远小于该值（例如同一结果中 Conv1d 约 10,651 nJ、Linear 约 12,831 nJ）。

---

## 2. 脉动阵列（MAC）部分能耗公式

仅 **systolic_array** 部分的能耗由本工程公式计算（与 SCALE-Sim 的周期、维度一致，能耗不用 SCALE-Sim）：

```text
mac_energy_nj = MACs × mac_power_mw × clock_period_ns / 1000
```

- **MACs**：该层矩阵乘的 MAC 数，M×K×N（由 SCALE-Sim 的 ifmap/filter 维度得到）；
- **mac_power_mw**：单 MAC 等效功耗 (mW)，来自 `hw_config.systolic_array.mac_power_mw`，**默认 7.144**；
- **clock_period_ns**：数字逻辑时钟周期 (ns)，默认 1.0（1 GHz）。

即：**每 MAC 能耗 = 7.144 mW × 1 ns = 7.144 pJ**（1 GHz 下），总 MAC 能耗 = MACs × 7.144 / 1000 (nJ)。

实现位置：

- `pointkan_sim/integration/scalesim_adapter.py`：`_energy_breakdown_from_hw_config()` 中 `mac_nj = macs * mac_power_mw * clock_ns / 1000.0`；
- `pointkan_sim/operators/common_ops.py`：`gen_conv1d_systolic_events` / `gen_linear_systolic_events` 用该 MAC 能耗生成脉动事件（或由 `scalesim_mac_energy_nj` 反推功率再算能耗）。

---

## 3. 如何用脚本验证 “MAC 数 × 7.144” 与报告一致

- **scalesim_conv_linear_layers.csv** 中每一行对应一层 Conv1d 或 Linear 的 SCALE-Sim 结果，列包括：
  - `operator`：Conv1d / Linear；
  - `macs`：该层 MAC 数；
  - `mac_energy_nj`：该层脉动阵列（MAC）能耗 (nJ)。

验证步骤：

1. 按算子类型（Conv1d / Linear）分别对 **macs** 和 **mac_energy_nj** 求和。
2. 用公式计算期望 MAC 能耗：  
   `expected_mac_energy_nj = sum(macs) * 7.144 * 1.0 / 1000`  
   （若时钟不是 1 ns，则替换 `clock_period_ns`）。
3. 比较 **sum(mac_energy_nj)** 与 **expected_mac_energy_nj**，应基本一致（浮点误差内）。

脚本见：**`scripts/verify_conv_linear_mac_energy.py`**，可指定某次仿真的 `sim_results` 目录（含 `scalesim_conv_linear_layers.csv` 或 `report.json`），输出上述汇总与对比结果。

---

## 4. 小结

| 项目 | 含义 |
|------|------|
| 终端 “Conv1d” / “Linear” | 该算子**全部事件**能耗之和（im2col + systolic + SRAM/DRAM 等） |
| 脉动阵列（MAC）能耗 | 仅 systolic 事件：**MACs × mac_power_mw × clock_ns / 1000**，默认 **MACs × 7.144 / 1000** (nJ) |
| 数据来源 | 每层 MAC 数与 mac_energy_nj 见 `scalesim_conv_linear_layers.csv`；总能耗见 `report.json` 的 operator_sequence / full_operator_breakdown |
