# Embedding 层 SCALE-Sim 周期与能耗说明

针对 `scalesim_conv_linear_layers.csv` 中 **embedding** 行：`total_cycles=1213`，`energy_nj≈702.28`。

## 1. 数据来源与 Conv1d 映射

- **实现位置**：`pointkan_sim/integration/scalesim_adapter.py`
  - 周期：`run_scalesim_conv1d_layer()` → SCALE-Sim → `COMPUTE_REPORT.csv` 的 **Total Cycles**
  - 能耗：同次运行得到 SRAM/DRAM 访问次数后，用 `_energy_nj_from_hw_config()` 在本工程内算 nJ

- **Embedding 层参数**（与 CSV 一致）  
  - `num_points=1024`，`in_channels=3`，`out_channels=32`，`kernel_size=1`

- **Conv1d 在 SCALE-Sim 中的拓扑**（`run_scalesim_conv1d_layer` 内）：
  - IFMAP 空间尺寸：`ifmap_h=1`, `ifmap_w=1024`
  - Filter 空间尺寸：`filter_h=1`, `filter_w=1`
  - 通道：`channels=3`，`num_filter=32`  
  即 SCALE-Sim 看到的是 **1×1024 的 ifmap**、**1×1 的 filter**、3 输入通道、32 个 filter。

---

## 2. 1213 cycle 怎么来的

- **1213** 不是在本工程里用公式手算的，而是 **SCALE-Sim 跑完该层拓扑后的直接输出**。
- 流程：
  1. 用上述拓扑写出 `topology.csv`，用 `hw_config.systolic_array.rows/cols`（默认 64×64）生成临时 config。
  2. 调用 `scalesim.run_scale(top_path=out_dir)`。
  3. 读 `COMPUTE_REPORT.csv` 第一行的 **Total Cycles**，得到 1213。
- 因此：**1213 = 在该 1×1024、1×1、3 通道、32 filter、64×64 脉动阵列配置下，SCALE-Sim 的周期仿真结果**。  
  若要看更细的 stall、利用率等，可看同目录下的 `Stall Cycles`、`Overall Util %`、`Mapping Efficiency %`、`Compute Util %` 等列（与 CSV 中 `overall_util_pct` 等对应）。

---

## 3. Energy 702.28 nJ 怎么算的

- **公式**（`_energy_nj_from_hw_config`）：  
  **总能耗 = MAC 能耗 + SRAM 访问能耗 + DRAM 访问能耗**

### 3.1 MAC 能耗（当前 embedding 的主要部分）

- 公式：  
  `mac_nj = macs × mac_power_mw × clock_period_ns / 1000`
- Embedding 的 MAC 数（与 adapter 中一致）：  
  `macs = (ifmap_w - filter_w + 1) × num_filter × (filter_h × filter_w × channels)`  
  `= (1024 - 1 + 1) × 32 × (1×1×3) = 1024 × 32 × 3 = 98,304`
- 默认配置（baseline）：  
  - `SystolicArrayConfig.mac_power_mw = 7.144`  
  - `HardwareConfig.clock_period = 1.0` ns  
  因此：  
  `mac_nj = 98304 × 7.144 × 1.0 / 1000 ≈ 702.28 nJ`  
  这与 CSV 中的 `energy_nj` 一致，说明 **当前 baseline 下 embedding 的能耗主要就是 MAC 部分**。

### 3.2 SRAM / DRAM 部分

- SRAM：  
  `sram_nj = (SRAM IFMAP reads + SRAM Filter reads) × read_energy_nj_per_access + SRAM OFMAP writes × write_energy_nj_per_access`  
  其中读/写次数来自 SCALE-Sim 的 **DETAILED_ACCESS_REPORT.csv**；  
  单次能耗来自 `InterLayerBufferConfig.read_energy_nj_per_access` / `write_energy_nj_per_access`（CACTI 回填）。  
  **默认 baseline 未设这两项，因此 sram_nj = 0。**

- DRAM：  
  - 若配置了 `DRAMConfig.dramsim3_bin` 且用 trace 跑了 DRAMSim3，会用返回的 `dram_nj_override` 作为 DRAM 能耗；  
  - 否则：  
    `dram_nj = (DRAM IFMAP reads + DRAM Filter reads) × read_energy_nj_per_access + DRAM OFMAP writes × write_energy_nj_per_access`  
    **默认 baseline 未设 DRAM 的 per_access，因此 dram_nj = 0。**

所以在 **未配置 SRAM/DRAM 单次访问能耗** 的情况下，embedding 行的 **702.28 nJ ≈ 纯 MAC 能耗**；配置 CACTI/DRAMSim3 回填后，同一公式会再加上 SRAM/DRAM 部分。

---

## 4. 小结

| 项目        | 来源 / 计算方式 |
|-------------|------------------|
| **1213 cycle** | SCALE-Sim 对该层拓扑 + 64×64 阵列的 **Total Cycles** 输出 |
| **702.28 nJ**  | 本工程 `_energy_nj_from_hw_config`：**MAC**（98304×7.144×1e-3 nJ）+ SRAM（默认 0）+ DRAM（默认 0） |

要看 SCALE-Sim 原始报告，可在临时目录（运行时的 `tempfile.TemporaryDirectory`）的 `out/<run>/COMPUTE_REPORT.csv` 与 `DETAILED_ACCESS_REPORT.csv` 中查看；若要保留这些文件，需在 adapter 里关掉 `save_disk_space` 或把输出拷到固定目录。
