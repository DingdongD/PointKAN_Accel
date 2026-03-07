# NeuroSim 电路时钟与 Buffer 说明

## 一、如何改变实际电路时钟频率（NeuroSim）

NeuroSim 的工作频率由**实际电路路径延迟**决定：`synchronous = true` 时，`clkFreq` 会被 sensing/ADC 等路径算出的最小周期限制。要改变电路时钟，需调节以下**底层参数**（均在 `DNN_NeuroSim_V1.4/Inference_pytorch/NeuroSIM/` 下）。

### 1. 直接设定期望频率（上限）

**文件**: `Param.cpp`  
**参数**: `clkFreq`（单位：Hz）

```cpp
clkFreq = 1e9;   // 1 GHz（默认）
// 改为 500 MHz: clkFreq = 500e6;
// 改为 2 GHz:   clkFreq = 2e9;
```

- 实际频率 = **min(clkFreq, 1/clkPeriod)**。  
- `clkPeriod` 由各层电路路径（SubArray 内 sensing、ADC、row/mux 等）计算得到；若某层路径更慢，实际会用 `1/clkPeriod` 压低频率。

### 2. 影响 clkPeriod 的电路参数（改“实际电路时钟”必须动这些）

这些在 **SubArray.cpp** 里被用来算 `readLatency`，进而通过 `CalculateclkFreq=true` 得到 `layerclkPeriod`，最终 `clkPeriod = max(layerclkPeriod)`。

| 参数 | 文件 | 含义 | 典型改法 |
|------|------|------|----------|
| **beta** | Param.cpp | sensing 周期延迟因子（默认 1.4） | 减小可缩短周期、提高频率 |
| **technode** | Param.cpp | 工艺节点 (nm)，如 22 | 改小节点会缩短器件延迟 |
| **ADClatency** | 由 SubArray 计算后写回 Param | ADC/SenseAmp 延迟 | 受 SARADC、currentMode、levelOutput 等影响 |
| **rowdelay** | 同上 | 行方向 (WL switch matrix + buffer) 延迟 | SubArray 内 bufferlatency、wlSwitchMatrix 等 |
| **muxdelay** | 同上 | Mux + MuxDecoder 延迟 | 受 numColMuxed、sizingfactor_MUX 等影响 |
| **sizingfactor_MUX** | Param.cpp | Mux 驱动尺寸因子 | 增大可减小延迟、提高频率 |
| **sizingfactor_WLdecoder** | Param.cpp | WL 译码器驱动尺寸因子 | 增大可减小 row 路径延迟 |
| **buffernumber, buffersizeratio** | Param.cpp | SubArray 内插入的 buffer 数量与尺寸比 | 影响 rowdelay 中的 bufferlatency |

**推荐步骤**（在 NeuroSIM 源码中）：

1. **提高频率**：在 `Param.cpp` 中把 `clkFreq` 调大（如 `2e9`）；若实际仍被路径限制，适当减小 `beta`（如 1.2），或增大 `sizingfactor_MUX` / `sizingfactor_WLdecoder`。
2. **降低频率**：见下文「3. 如何降低实际频率」。
3. 修改后需**重新编译** NeuroSIM 的 `main`，再跑本仓库脚本查看 `clock_period_ns` / 频率是否变化。

### 3. 如何降低实际频率

若要**降低**电路时钟频率（增大周期、更保守设计），可任选或组合使用：

| 做法 | 文件 | 操作 | 说明 |
|------|------|------|------|
| 直接设上限 | Param.cpp | 将 `clkFreq` 改小，如 `clkFreq = 500e6;`（500 MHz） | 实际频率 = min(clkFreq, 1/clkPeriod)，直接封顶 |
| 增大 sensing 周期 | Param.cpp | 在 `validated` 分支内将 `beta` 改大，如 `beta = 1.8;` 或 `2.0;`（默认 1.4） | 增大后 SubArray 的 readLatency 变长，clkPeriod 变大、频率降低 |
| 减小 Mux/WL 驱动 | Param.cpp | 在 `switch(technode){...}` **之后** 增加：`sizingfactor_MUX = 80;`、`sizingfactor_WLdecoder = 0.8;`（或更小） | 注意 22 nm 时 switch 会设 sizingfactor_MUX=110，需在 switch 后覆盖为更小值以增大延迟 |
| 换更大工艺节点 | Param.cpp | 将 `technode = 22;` 改为 `technode = 32;` 或 `45;` | 器件与走线延迟变大，整体频率下降；同时会改变 Metal/readVoltage 等一整套工艺参数 |

**当前仿真工艺节点**：NeuroSim 默认 **`technode = 22`**（22 nm），在 `Param.cpp` 第 127 行附近。若未自行修改，则当前仿真即 22 nm。

### 4. 通过 Python 传入时钟频率（与 subarray 一样从命令行传入）

本仓库已支持从 Python **命令行**把时钟频率上限传给 NeuroSim 的 `main`，无需改 Param.cpp 再重新编译即可换频率跑多次：

- **脚本参数**：`--clk_freq <Hz>`，例如 `--clk_freq 500e6`（500 MHz）、`--clk_freq 1e9`（1 GHz）。
- **底层约定**：`main` 若在参数列表末尾多收到一个数，会把它当作 `clkFreq`（Hz）覆盖 Param 的默认值（需 NeuroSim 的 `main.cpp` 已按本仓库约定加入“可选最后一参为 clkFreq”的逻辑并重新编译）。
- **实际频率**：仍为 **min(传入的 clk_freq, 1/clkPeriod)**；若电路路径更慢，实际频率会被 clkPeriod 限制。

用法示例（与 `--subarray`、`--parallel_read` 类似）：

```bash
python test_neurosim_full_standalone.py --pointkan_elite_full --output_dir sim_results --clk_freq 500e6
```

### 5. 28 nm 工艺节点

NeuroSim 原版未提供 28 nm；本仓库在 **Param.cpp** 中为 **28 nm** 增加了分支，与 32/22 nm 并列：

- 在 `Param.cpp` 中将 **`technode = 22;`** 改为 **`technode = 28;`**（约第 127 行），保存后**重新编译** NeuroSIM 的 `main`，即可用 28 nm 工艺参数跑仿真。
- 28 nm 使用的参数：`sizingfactor_MUX=100`，Metal/wire 宽度 48 nm，`readVoltage=0.52`，介于 32 nm 与 22 nm 之间；SRAM 仍走 `technode>14` 分支（与 22/32 nm 一致）。

工艺节点可通过 **Python 命令行**传入（与 `--clk_freq` 相同方式）：使用 **`--technode <nm>`**（如 `--technode 28`、`--technode 22`），main 会将最后一参或倒数第二参（与 clk_freq 一起传时）当作 technode 并调用 `Param::SetTechnode`。需 NeuroSim main 已支持可选参数并重编。

---

## 二、NeuroSim 的 Buffer：类型与规模

### 0. SRAM buffer 是否考虑「第一次加载的写入」？

**是的。** NeuroSim 在计算每层 global buffer 的延迟与能耗时，同时包含**读**和**写**：

- **Chip.cpp** 中每层调用 `globalBuffer->CalculateLatency(..., numBitToLoadOut/..., numBitToLoadIn/...)` 与 `CalculatePower(...)`，其中 `numBitToLoadOut` 对应写入 buffer 的数据量（上一层的输出/本层输入的加载），`numBitToLoadIn` 对应从 buffer 读出的数据量。
- 每层累加到总 buffer 的量为：`bufferLatency += globalBuffer->readLatency + globalBuffer->writeLatency`，`bufferDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy`（见 Chip.cpp 约 1108、1121 行）。
- **Buffer.cpp** 的 `CalculateLatency(numAccessBitRead, numRead, numAccessBitWrite, numWrite)` 会分别计算读延迟/写延迟（SRAM 下含 WL decoder、precharger、sramWriteDriver、senseAmp 等），因此**第一次（或每次）把数据写入 SRAM buffer 的写入延迟与能耗都已计入**。

### 1. sync_data_transfer 是什么、用于干嘛？

**sync_data_transfer** 是 NeuroSim 里的一种**数据传递/时序建模模式**（Param.cpp 中 `sync_data_transfer=1` 表示开启，源码注释为 “230920 update”）：

- **作用**：  
  - **开启（=1）**：认为层间/片级数据传递与计算**按时钟同步**进行，数据以「分片」（fraction）方式在全局 buffer 与 tile/PE 之间传输；global buffer 的延迟按 `chip_bufferclk * ceil(totalInput/fraction)` 等方式折算，**IC（互连）延迟不再单独加在 readLatency 上**（`icLatency=0`），而是被包含在这种同步传输模型中。  
  - **关闭（=0）**：采用传统方式，把 buffer 读/写延迟和互连（H-Tree 等）延迟**分别**算出来，再加到每层的 readLatency / bufferLatency / icLatency 上。

- **对结构的要求**（main.cpp 中的检查）：  
  当 `sync_data_transfer=1` 时，必须同时满足：`globalBusType==true`、`chipActivation==true`、`novelMapping==true`、`pipeline==true`、`synchronous==true`，且 **`globalBufferType==false`**（片级 global buffer 必须是**寄存器**，不能是 SRAM）。因此当前 Param 默认若开 `sync_data_transfer=1`，需要把 **globalBufferType** 设为 **false**，否则会报 “sync_data_transfer mode not supported for the current parameter combination”。

- **对容量/面积的影响**：  
  开启时 Chip 层 global buffer 的 `bufferSize` 会按 `numBitInput*maxLayerInput*2` 计算（约为关闭时的 2 倍），且 H-Tree、Tile/PE 内 input/output buffer 的初始化与延迟/能耗计算方式与关闭时不同（见 Chip.cpp、Tile.cpp、ProcessingUnit.cpp 中 `if (param->sync_data_transfer)` 分支）。

**总结**：sync_data_transfer 用于在**同步、流水、novel mapping** 的配置下，用「与时钟同步的分片传输」来建模层间数据传递，从而不单独显式加 IC 延迟；开启时要求 global buffer 为寄存器（globalBufferType=false）。

---

### 2. Buffer 类型（SRAM vs 寄存器）

在 **Param.cpp** 中：

- **globalBufferType = true** → 片级 **global buffer 使用 SRAM**（与 sync_data_transfer=1 不兼容，需设为 false）。  
- **globalBufferType = false** → 片级 global buffer 使用 **register file**；当 **sync_data_transfer=1** 时必须为 false。  
- **tileBufferType = false** → Tile 级 buffer 使用 **register file**（默认）。  
- **peBufferType = false** → PE 级 buffer 使用 **register file**（默认）。

我们解析到的 “buffer latency / buffer readDynamicEnergy” 来自 **Chip.cpp**：  
每层会累加 **global buffer** 的 read+write 延迟/能耗，以及 Tile 内 buffer 的贡献。因此**层间/片级 buffer 默认是 SRAM**；Tile/PE 级为寄存器。

### 3. Global Buffer 容量（总规模）

**Chip.cpp** 中：

- `bufferSize = param->numBitInput * maxLayerInput`（或 sync 模式下 ×2）。  
- `numBufferCore = ceil(bufferSize / (globalBufferCoreSizeRow * globalBufferCoreSizeCol))`。  
- 每个 core 大小为 **globalBufferCoreSizeRow × globalBufferCoreSizeCol**（Param 默认 128×128 **bit**）。

因此：  
**总 buffer 容量（bit）** = `numBufferCore * globalBufferCoreSizeRow * globalBufferCoreSizeCol`  
是按**全网络最大的一层输入**算的，**不是按层分配**，而是共享、按最大层尺寸设计。

### 4. 每层“占用”的 buffer 数据量（size）

NeuroSim **不会**在 stdout 里打印每层的 buffer 容量；它只给每层的 buffer **latency/energy**。  
每层在 buffer 上通过的数据量可由**本仓库的层配置**推算：

- **Conv**: 输入 bit 数 ≈ `numBitInput * in_c * in_h * in_w`（即该层输入激活的 bit 数）。  
- **Linear**: 输入 bit 数 ≈ `numBitInput * tokens * in_f`。

在 `test_neurosim_full_standalone.py` 或 `pointkan_sim` 中，已知每层 `in_c, in_w, in_f, tokens` 等，可据此算每层的 **buffer_data_bits**（或转成 bytes）并写入 CSV，作为“每层使用的 buffer 数据 size”的近似。

### 5. sync_data_transfer 开启时：层消耗的 buffer 容量如何算？

开启 **sync_data_transfer** 时，global buffer 仍是**整片共享**，总容量为 **bufferSize = numBitInput × maxLayerInput × 2**（bit），其中 `maxLayerInput` 在 Chip 初始化时由 `netStructure` 所有层的 **in_h×in_w×in_c** 取最大（或 pipeline 下的累加）得到。NeuroSim **不会**在 stdout 里按层打印“该层占用的 buffer 容量”，因为 buffer 是共享的，没有按层划分。

**每层“消耗”的 buffer 数据量**（即该层输入/输出经 buffer 的数据规模）可按**该层输入维度**算：

- 与 config 对应：`netStructure[l][0]*netStructure[l][1]*netStructure[l][2]` = 该层输入激活元素数（in_h × in_w × in_c）。
- **每层经 buffer 的数据量（bit）** = **numBitInput × (in_h × in_w × in_c)**；若需字节则 ÷8。

本仓库在 `test_neurosim_full_standalone.py` 生成 **neurosim_layer_detail.csv** 时，已按层写入 **buffer_data_bits**、**buffer_data_bytes**（由层规格推导），即上述“每层经 buffer 的数据量”，可直接用作 sync_data_transfer 开启时**每层消耗的 buffer 数据规模**的近似。

### 6. 阵列、ADC、shift-adder 等“每次转换”的 latency 如何确定？

NeuroSim 在 **synchronous** 模式下会先对**每一层**调用 `ChipCalculatePerformance(..., CalculateclkFreq=true, &layerclkPeriod)`，在 **SubArray.cpp** 里计算**单次 sensing 周期**的延迟（一次“转换”的路径：WL/行译码 → 预充 → 列延迟 → SenseAmp/ADC → 若多 bit 再经 shift-add 等），并写回全局 **Param**：

- **param->rowdelay**：行方向延迟（WL switch matrix + buffer 等）。
- **param->muxdelay**：Mux + MuxDecoder 延迟。
- **param->ADClatency**：SenseAmp/ADC 转换延迟。

**单次 sensing 周期**的 readLatency（秒）在 SubArray 内由这些路径**串起来**，再乘 **beta**（默认 1.4）得到；**shift-adder / accumulation** 在同步模式下可与 ADC 流水，因此**决定时钟周期的**是**一条 sensing 链**（rowdelay + precharger + colDelay + SenseAmp/ADC + 若有多级再加 multilevelSAEncoder、sarADC 等），而不是把 shift-add 再单独加在周期上。

**main.cpp** 里：对每层算完 `layerclkPeriod` 后取 **clkPeriod = max(layerclkPeriod)**，再令 **param->clkFreq = min(param->clkFreq, 1/clkPeriod)**。因此：

- **“每次转换”对应的 latency 时间** = **clkPeriod**（秒），即 **1 / 实际工作频率**。
- 该 clkPeriod 由**最慢的那一层的 SubArray 单次 sensing 路径**决定；若要看**具体哪一段最慢**，需在 NeuroSim 源码中打印或记录 **param->rowdelay**、**param->muxdelay**、**param->ADClatency**（在 SubArray 里 CalculateclkFreq=true 时已写入）。

**总结**：阵列+ADC+shift-adder 等的“每次转换”latency = 实际 **clock period**（clkPeriod）；其数值由 SubArray 内 rowdelay、muxdelay、ADClatency、precharger、colDelay 等组成的一条 sensing 链 × beta 决定，main 输出中的 `clkPeriod` 即为该值（秒），取的是所有层中的最大值。

---

## 三、本仓库中与 MNSIM 一起用的“层间 buffer”

在 **pointkan_sim** 里（如 `run_sim_with_real_eval.py`、`main.run_network_simulation`）：

- **层间 buffer** 由 **hw_config.inter_layer_buffer**（`InterLayerBufferConfig`）描述，包括：
  - `size_bytes`：容量（字节）
  - `bandwidth_bytes_per_cycle`：每周期读/写字节数  
  用于生成**层间 buffer 读/写事件**和（可选）DRAM spill，**不区分 SRAM 还是寄存器**，是抽象配置。

- **MNSIM** 的 `total_buffer_latency` 等来自 MNSIM 自身模型（`mnsim_netstruct` 中使用的接口），和 NeuroSim 的 global/tile buffer 不是同一套实现；本仓库侧只使用其延迟/能耗结果，buffer 类型由 MNSIM 内部定义。

若需要“每层用了多少 size 的 buffer”，可：

- 对 **NeuroSim**：本仓库在运行 `test_neurosim_full_standalone.py` 并指定 `--output_dir` 时，会在 `neurosim_layer_detail.csv` 中写入每层的 **buffer_data_bits**、**buffer_data_bytes**（由该层输入激活维度×位宽计算，近似该层经层间 buffer 的数据量）。  
- 对 **pointkan_sim 的层间 buffer**：每层读写量由 `activation_shapes`（input_bytes / output_bytes）决定，可直接用现有 `ActivationShapeInfo` 的 input/output 字节数作为该层占用 buffer 的 size 近似。

---

## 四、小结

| 问题 | 结论 |
|------|------|
| 如何改变 NeuroSim 实际电路时钟？ | 改 `Param.cpp` 的 `clkFreq`（上限）；若要提高实际频率，需改 `beta`、`sizingfactor_*`、`technode` 等影响 SubArray 延迟的参数并重编 main。 |
| NeuroSim 层间 buffer 是 SRAM 还是寄存器？ | 片级 global buffer 默认 **SRAM**（globalBufferType=true）；Tile/PE 默认 **register file**。 |
| 每层用了多少 size 的 buffer？ | NeuroSim 不输出每层 buffer 容量；可按层用 `numBitInput*层输入维度` 算 buffer 数据量，并在本仓库 CSV 中增加一列；层间 buffer 的 size 可用 activation_shapes 的 input/output 字节数近似。 |
| SRAM buffer 是否考虑第一次加载的写入？ | 是。每层 buffer 延迟/能耗 = read + write，Buffer 模块同时计算读与写的延迟和能耗。 |
| 如何降低实际频率？ | 减小 `clkFreq`；或增大 `beta`；或在 technode 的 switch 之后减小 `sizingfactor_MUX`/`sizingfactor_WLdecoder`；或增大 `technode`（如改为 32/45 nm）。 |
| 当前仿真工艺节点？ | 默认 **22 nm**（`Param.cpp` 中 `technode = 22`）。支持 **28 nm**：改为 `technode = 28` 后重编即可。 |
| 能否用 Python 改时钟频率？ | 可以。脚本加 `--clk_freq <Hz>`，main 将对应可选参当作 clkFreq（需已合入本仓库对 main.cpp 的修改并重编）。 |
| 能否用 Python 改工艺节点？ | 可以。脚本加 `--technode <nm>`（如 22、28、32），main 调用 `Param::SetTechnode`（需已重编）。 |
| sync_data_transfer 开启时每层 buffer 容量？ | 总 buffer = numBitInput×maxLayerInput×2（bit）；每层“消耗”数据量 = numBitInput×(in_h×in_w×in_c)，见 CSV 的 buffer_data_bits/bytes。 |
| 阵列/ADC/shift-adder 每次转换 latency？ | 由 SubArray 单次 sensing 链（rowdelay+muxdelay+ADClatency+…）× beta 决定；实际 = clkPeriod（main 输出的 clock period，取各层最大）。 |
