# NeuroSim 接入 pointkan_sim 可行性分析

基于对 `/workspace/DNN_NeuroSim_V1.4` 及 `Inference_pytorch/VMM_CIM.ipynb` 的阅读，结论与接入方案如下。

---

## 一、NeuroSim 是否支持 Conv1d

**结论：支持。** NeuroSim 的层配置是 **Conv2d 形式的 8 元组**，无单独 “Conv1d” 类型：

- 配置 CSV 每行 8 列：`in_h, in_w, in_c, k_h, k_w, out_c, pool, stride`
- 与 `main.cpp` 中 `getNetStructure()` 及 `ChipCalculatePerformance` 使用的 `netStructure[i][0..7]` 一致
- 矩阵维度由该 8 元组推导（如 `weightMatrixRow = in_c * k_h * k_w * numRowPerSynapse`）

**Conv1d 的映射方式：**

- Conv1d 输入 `(C_in, L)`，核 `(K)`，输出 `(C_out, L)`，可视为 2D：`(1, L)` 为空间维
- 对应 NeuroSim 一行：**in_h=1, in_w=L, in_c=C_in, k_h=1, k_w=K, out_c=C_out, pool=0, stride=1**

**Linear 的映射方式：**

- 全连接即 “1×1 conv”：**in_h=1, in_w=1, in_c=in_features, k_h=1, k_w=1, out_c=out_features, pool=0, stride=1**

VMM_CIM.ipynb 中 DGCNN 已用大量 `1,1,in_c,1,1,out_c,0,1` 的层，与 Conv1d/Linear 的上述映射一致，因此 **Conv1d 与 Linear 均可通过同一套 8 元组接口支持**。

---

## 二、每层仿真是否包含 buffer 交互能耗

**结论：包含。** NeuroSim 对 **每一层** 都单独给出：

- `layer i's readLatency`（核心计算延迟，ns）
- `layer i's readDynamicEnergy`（核心计算动态能耗，pJ）
- **`layer i's buffer latency`**（该层 buffer 延迟，ns）
- **`layer i's buffer readDynamicEnergy`**（该层 buffer 动态能耗，pJ）
- `layer i's ic latency` / `layer i's ic readDynamicEnergy`（互连等）

`main.cpp` 中（约 335–341 行、433–439 行）逐层打印上述项；`Chip.cpp` 中 `ChipCalculatePerformance` 通过 `bufferLatency`、`bufferDynamicEnergy` 等指针累加 globalBuffer 的读写延迟与能耗。因此 **按层统计中已包含 buffer 的交互延迟与能耗**，接入时可直接使用 “readLatency + buffer latency” 和 “readDynamicEnergy + buffer readDynamicEnergy” 作为该层总延迟与总能耗（或按需拆开）。

---

## 三、NeuroSim 输出为 ns、无 cycle 的对接方式

**结论：在 pointkan_sim 内用 “ns → cycle” 即可，无需 NeuroSim 改输出。**

- NeuroSim 输出：**readLatency / buffer latency 单位为 ns**，**readDynamicEnergy / buffer readDynamicEnergy 单位为 pJ**。
- 仿真末尾会打印 **`clkPeriod: X`（秒）**，即芯片时钟周期；部分流程下为 `Initialization Clk Period: 1e-09` 等。
- pointkan_sim 侧已有 **`hw_config.clock_period`（ns）**，与 MNSIM 等一致。

**推荐换算：**

- **周期数**：`cycles = ceil(latency_ns / hw_config.clock_period)`，其中 `latency_ns` 取该层 `readLatency (ns) + buffer latency (ns)`（若希望把 buffer 算进同一段执行时间）；或仅用 `readLatency`，buffer 单独建事件，由你方策略决定。
- **能耗**：`energy_nj = (readDynamicEnergy_pJ + buffer_readDynamicEnergy_pJ) / 1000`，得到 nJ，与现有 `energy_nj` 一致。

即：**NeuroSim 只给 ns/pJ，我们在适配层里用 `clock_period_ns` 转 cycle、用 1000 转 nJ**，不要求 NeuroSim 增加 cycle 输出。

---

## 四、接入 pointkan_sim 的可行性总结

| 问题 | 结论 |
|------|------|
| Conv1d 是否支持？ | 支持，映射为 in_h=1, in_w=L, in_c, k_h=1, k_w=K, out_c, 0, 1 |
| Linear 是否支持？ | 支持，映射为 1,1,in_f,1,1,out_f,0,1 |
| 每层是否含 buffer 延迟/能耗？ | 是，有 buffer latency 与 buffer readDynamicEnergy 按层输出 |
| 只有 ns、没有 cycle？ | 在适配层用 `latency_ns / clock_period_ns` 得到 cycle 即可 |

**结论：可以接入。** 实现一个 “NeuroSim 适配器” 即可：为每一 Conv1d/Linear 层生成 NeuroSim 所需 config CSV + 每层 weight/input CSV，调用已编译的 `NeuroSIM/main`，解析 stdout 得到每层 readLatency、readDynamicEnergy、buffer latency、buffer readDynamicEnergy（ns/pJ），再在 pointkan_sim 内转为 cycle 与 nJ，与现有 MNSIM/fallback 路径并列或按配置二选一。

---

## 五、实现要点（供后续接 NeuroSim 接口时参考）

1. **调用方式**  
   - `main` 参数：`argv[1]=config.csv`, `argv[2]=wl_weight`, `argv[3]=wl_activation`, `argv[4]=subarray`, `argv[5]=parallelRead`, 随后每层两个文件：`argv[6],argv[7]` 为 layer0 的 weight/input，`argv[8],argv[9]` 为 layer1，依此类推。  
   - 与 VMM_CIM.ipynb 中 `generate_trace_command_script` 一致。

2. **Config CSV**  
   - 每行 8 个数：in_h, in_w, in_c, k_h, k_w, out_c, pool, stride；Conv1d/Linear 按上文映射即可。

3. **Weight / Input CSV**  
   - `Chip.cpp` 会按 `netStructure` 推断 weight 行/列、input 规模并调用 `LoadInWeightData`、`LoadInInputData`，因此生成的文件 **行/列数需与 netStructure 一致**；内容可为随机或占位（与 notebook 一致）。

4. **解析 stdout**  
   - 按行匹配例如：`layer i's readLatency is: X ns`、`layer i's readDynamicEnergy is: Y pJ`、`layer i's buffer latency is: ...`、`layer i's buffer readDynamicEnergy is: ...`，以及最后的 `clkPeriod: Z`（若要用 NeuroSim 的时钟可解析，否则统一用 `hw_config.clock_period`）。

5. **与现有 MNSIM/fallback 的关系**  
   - 可在 `pointkan_sim` 中增加选项（如 `--cim_backend mnsim | neurosim | fallback`），当选择 `neurosim` 时对 Conv1d/Linear 走 NeuroSim 适配器（生成 CSV → 调 main → 解析 → 转 cycle/nJ），其它算子不变；或按层/按网络在配置中指定用哪一端。

6. **依赖与路径**  
   - 需已编译 `DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/main`（或同名可执行文件）；适配器需指定 NeuroSim 工作目录（如 `/workspace/DNN_NeuroSim_V1.4/Inference_pytorch`），用于写 config/weight/input 和调用 main。

按上述方式即可在保留 “仅 ns、无 cycle” 的前提下，将 NeuroSim 的 Conv1d/Linear 与 buffer 结果接入 pointkan_sim 的 cycle 与 energy 管线。

---

## 六、已接入说明（当前实现）

- **单层估算**：`pointkan_sim/operators/neurosim_adapter.py`
  - `estimate_conv1d_neurosim(...)` / `estimate_linear_neurosim(...)`：生成单层 config + weight/input CSV，调用 `NeuroSIM/main`，解析 stdout，返回 `NeuroSimEstimate(cycles, latency_ns, energy_nj, avg_power_mw, source)`。
  - 由 `common_ops.gen_conv1d_rram_cim_events` / `gen_linear_rram_cim_events` 在 `hw_config.cim_backend == "neurosim"` 时调用。
- **全网络仿真**：`pointkan_sim/integration/neurosim_full.py`
  - `run_neurosim_full(conv_linear_list, neurosim_root, ...)`：按层生成 config 与 weight/input CSV，一次调用 main，解析每层 readLatency + buffer latency、readDynamicEnergy + buffer energy，返回 `NeuroSimFullResult(layer_latency_ns, layer_energy_nj)`。
  - 在 `run_sim_with_real_eval.py` 中通过 `--neurosim_full` 使用，结果注入 `mnsim_context`，与 MNSIM 全 NetStruct 共用同一 dispatcher 接口。
- **配置**：`hw_config.cim_backend`（`"mnsim"` | `"neurosim"` | `"fallback"`）、`hw_config.neurosim_root`（Inference_pytorch 路径）。
- **命令行**：`--cim_backend neurosim`、`--neurosim_root <path>`、`--neurosim_full`（真实数据评估时用 NeuroSim 全网络）。
