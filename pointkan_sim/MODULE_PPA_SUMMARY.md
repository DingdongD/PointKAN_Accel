# 各模块 Cycle / 功耗 / 面积 汇总（当前代码 + 需求文档对照）

> 检查范围：pointkan_sim（hw_config + common_ops + fps + knn），以及 SYSTEM_LEVEL_UPDATE_REQUIREMENTS.md、hw_sim/rtl/resampler。
> 周期单位：cycle；功耗单位：mW；面积单位：um²（未配置或 TODO 标为 0 或 —）。

---

## 1. 逻辑电路类（Logic Circuit）

| 模块 | 周期 (cycles) | 功耗 (mW) | 面积 (um²) | 说明 |
|------|----------------|-----------|------------|------|
| **ReLU** | `ceil(numel / par)` | 0.45 × par（默认 par=1） | 22.0 | hw_config: ReLUConfig；common_ops.gen_relu_events |
| **SPool** | `56 + (N-1)×47`，N=B×C×P | 26.82（代码写死，未用 config 17.0） | 34375.0 | hw_config: SPoolConfig；common_ops.gen_spool_events |
| **LayerNorm** | `ceil(base/par)`，base=(6F+32)+n_vec×10×F | 180.0 × par | 177025.0 | hw_config: LayerNormConfig；common_ops.gen_layernorm_events |
| **AdaptiveMaxPool1d** | `ceil(base_cycles/par)`，base=B×C×L | 13.2 × par | 同 SPool 34375 | 复用 logic_circuit.spool；common_ops.gen_adaptive_max_pool_events |
| **Mul** | `ceil(num_elements / vec)`，vec 默认 cim_parallelism | params.power_mw 默认 **1.1** | **未配置（0）** | common_ops.gen_mul_events；main 无 logic_area_map 条目 |
| **Add** | `ceil(num_elements / vec)`；若 offchip 另有 DRAM 读/写周期 | params.power_mw 默认 **0.8** | **未配置（0）** | common_ops.gen_add_events；残差可带 dram 50mW |
| **Div** | **未实现** | — | — | 需求文档：cycle=10，power=2.59，area=6307，并行度 P |
| **Batchnorm** | `batch×(1 + ceil(channels×s_dim/par))` | 3.1718 × par（config 默认 0，代码用 3.1718） | **0（TODO）** | hw_config: BatchNorm1dConfig power=0, cycles=0；common_ops 有公式 |
| **AffineNorm** | `2×(ceil(l_work/32)+1)+46`，l_work=S×K×D | 212.7465（复用 layernorm 配置） | **0** | main 中 logic_area_map["AffineNorm"]=0；需求文档 304976 |
| **量化 (Quant)** | `ceil(numel/par)+6`（当 config.cycles=0）或 ceil(cycles_cfg/par) | 2.59 × par | 6307.0 | hw_config: QuantDequantConfig cycles=1 但公式用 numel 时 +6 |
| **反量化 (Dequant)** | 同量化 | 2.59 × par | 6307.0 | common_ops._quant_dequant_power_cycles |

---

## 2. 排序与 FPS/KNN 相关

| 模块 | 周期 (cycles) | 功耗 (mW) | 面积 (um²) | 说明 |
|------|----------------|-----------|------------|------|
| **Bitonic (FPS)** | 每迭代 **10**（FPSParams.bitonic_sort_cycles，MAX 模式） | hw_config **0**（TODO） | **0** | comp_tree_1024_max；fps.generate_single_bitonic_event |
| **Bitonic (KNN)** | 每查询 **19**（KNNParams.bitonic_sort_cycles，MIN 模式） | hw_config **0**（TODO） | **0** | comp_bitonic_1024_min32；需求文档：2097.4mW，2157561um² |
| **Coarse filter (acc)** | 4×rounds+1（rounds 可配置） | 38.0（代码默认；config 为 0） | hw_config **0** | knn.generate_single_fuse_events；two_stage_reconf_comparator；需求 644686um² |
| **Fine filter (acc)** | 2 | 30.0（代码写死） | hw_config **0** | 需求 53261um² |

---

## 3. SRAM-CIM 与距离计算

| 模块 | 周期 (cycles) | 功耗 (mW) | 面积 (um²) | 说明 |
|------|----------------|-----------|------------|------|
| **SRAM CIM array** | 每批距离：`points_per_axis_quant_width/bit_slice`（如 28/2=14） | **0**（TODO） | 无单独 area 字段 | hw_config: SramCimArrayConfig write_power=0, cim_power=0；FPS/KNN stage1/2 |
| **ShiftAdd** | 每批 1（accumulate）等 | **0**（TODO） | 0 | hw_config: ShiftAddConfig |
| **Min-distance table / dist2 update** | 每批 1（table update） | **0**（TODO） | 0 | hw_config: MinDistanceTableConfig；FPS stage4 |
| **dist2_merge_adder** | 需求：1 cycle/批 | 需求 1.3×128 mW | 需求 2179×128 | pointkan_sim **未单独配置**，需求文档 3.2 |
| **Buffer (SRAM 写回)** | 1（KNN stage4 等） | **0**（TODO） | 0 | hw_config: BufferConfig |

---

## 4. RRAM-LUT（KAN / GRF）

| 模块 | 周期 (cycles) | 功耗 (mW) | 面积 (um²) | 说明 |
|------|----------------|-----------|------------|------|
| **GRF engine (rram_lut)** | lut_cycles=ceil(work/lut_parallel)，work=num_points×in_features×lut_factor | config **0**，代码默认 **5.0** | **0**（TODO） | common_ops.gen_kan_rram_lut_events；需求 base 数字 GRF 19 cycle/数据，23.7833mW，47877um² |
| **KAN buffer 读写** | buffer_cycles=ceil(num_points×(in+out)/lut_parallel) | config buffer **0**，代码默认 **0.5** | — | 同上 |

---

## 5. Resampler / 插值（acc）

| 模块 | 周期 (cycles) | 功耗 (mW) | 面积 (um²) | 说明 |
|------|----------------|-----------|------------|------|
| **Resampler (interpolation)** | hw_config **0**（TODO） | **0**（TODO） | 0 | hw_config: InterpolationConfig（仅 acc）；需求 T=N+2，3.95mW，4719；下/上采样 LUT 需求 13485+85331 |
| **RTL fp32_resampler** | 全插值 1+17×N_OUT；全整数 1+2×N_OUT | — | — | hw_sim/rtl/resampler/fp32_resampler.v 注释 |

---

## 6. 当前 hw_config 默认值速查（configs/hw_config.py）

| 配置项 | power | cycles | area |
|--------|-------|--------|------|
| BatchNorm1dConfig | 0 | 0 | — |
| LayerNormConfig | 180.0 | 0(公式) | 177025.0 |
| ReLUConfig | 0.45 | 0(公式) | 22.0 |
| SPoolConfig | 17.0 | 0(公式) | 34375.0 |
| QuantDequantConfig | 2.59 | 1 | 6307.0 |
| InterpolationConfig | 0 | 0 | — |
| SramCimArrayConfig | write 0, cim 0 | — | — |
| ShiftAddConfig | 0 | 0 | — |
| MinDistanceTableConfig | 0 | 0 | — |
| BufferConfig | 0 | 0 | — |
| RramLutConfig | 0 | 0 | 0 |
| CompTreeMaxConfig (FPS bitonic) | 0 | 0 | 0 |
| CompBitonicMinConfig (KNN bitonic) | 0 | 0 | 0 |
| TwoStageReconfComparatorConfig | 0 | 0 | 0 |

---

## 7. 小结与建议

- **已按公式或默认给出 PPA 的**：ReLU、SPool、LayerNorm、AdaptiveMaxPool、AffineNorm、量化/反量化、Add/Mul（周期与功耗来自 params/公式；面积 Add/Mul 未配置）。
- **周期/功耗由公式计算、但 config 多为 0 的**：Batchnorm、SRAM-CIM/ShiftAdd/MinDistanceTable/Buffer、Bitonic、Coarse/Fine、GRF（代码内部分有 fallback 默认功率）。
- **未实现或未单独建模的**：Div；dist2_merge_adder；resampler 在 pointkan_sim 仅占位 0。
- **建议**：若需与 SYSTEM_LEVEL_UPDATE_REQUIREMENTS.md 对齐，需在 `hw_config.py` 为上述 TODO 项填入标称值，并在 main.py 的 logic_area_map 中为 Add/Mul/AffineNorm 等补面积（或明确标注“不占面积”）。
