# 数字模块仿真器性能评估表

## 1. 核心模块性能指标 (Core Module Metrics)

### 1.1 已实现 RTL 算子

| 模块名称 (Module) | 功能描述 | 周期数公式 (Cycles) | 频率 (Freq) | 功耗 (Power,mw) | 面积 (Area,um2) | 备注/优化点 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **op_relu** | FP32 ReLU | `T = N_elements` | 1 GHz | `0.45` | `222` | 可并行 M 个 op_relu 电路，周期缩减为 `N_elements / M`，功耗和面积对应倍增 M |
| **op_spool** | FP32 S-Pool | `T = 56 + (N - 1) × 47` | 1 GHz | `26.82` | `54926` | N = B × N_points × N_channels |
| **op_layer_norm** | FP32 LayerNorm | `T = N x (3D/8 + 17)` | 1 GHz | `56` | `139582` | 移除内部寄存器组，由外部片上缓存输入数据，并且将原本寄存器的面积替换为更多的ALU单元，增加了模块内并行度，周期缩减 |
| **op_adaptive_max_pool1d** | FP32 全并行MaxPool (output_size=1) | `T = N_fibers` | 1 GHz | `13.2` | `18406` | MAX_DIM=64；N_fibers=池化输出元素总数（池化维≤64）。举例：K近邻池化 [N_points, D, K]→[N_points, D] 时 N_fibers=N_points×D；全局池化 [D, N_points]→[D] 时 N_fibers=D |
| **fp32_mul** | FP32 乘法器 | `1` (单次延迟) | 1 GHz | `2.5` | `2024` | 寄存器1拍输出 |
| **fp32_add** | FP32 加法器 | `1` (单次延迟) | 1 GHz | `1.16` | `1273` | 寄存器1拍输出 |
| **fp32_div** | FP32 除法器 (Newton-Raphson) | `~10` (单次延迟) | 1 GHz | `6307` | `2.59` | 3轮NR迭代 |
| **baseline_bitonic_sort_1024_max_32min** | 双模式双调排序网络 (1024输入取max或32min) | max模式: `10`, 32min模式: `19`（1024输入，其余输入见备注） | 1 GHz | `fps: 1570; knn: 1540` | `2157561` | 512-32：17cycles；256-32：15cycles；128-32：13cycles |
| **op_batch_norm_1d** | FP32 BatchNorm1d (推理模式, y=γx+β) | `T = N_elem + 1` | 1 GHz | `3.1718` | `3228` | 2级流水(fp32_mul→fp32_add)，延迟2cycles，吞吐1elem/cycle |
| **affine_norm** | FP32 仿射归一化 (anchor模式, 双遍扫描) | `T = 2×(L/P + 1) + 46`, L=S×K×D, P=32 | 1 GHz | `212.7465` | `304976` | L=SxKxD=393216(对于网络模型所有该算子的调用处，均恒定)；P=32路并行 |
| **quantize_pipeline (8bit)** | FP32→INT8 量化流水线 | `T = N + 6` | 1 GHz | `2.234` | `2460` | N=待量化元素数，6cycles流水线填充延迟 |
| **quantize_pipeline (12bit)** | FP32→INT12 量化流水线 | `T = N + 6` | 1 GHz | `2.28` | `2525` | N=待量化元素数，6cycles流水线填充延迟 |
| **dequantize_pipeline** | INT8→FP32 反量化流水线 | `T = N + 4` | 1 GHz | `2.30` | `2078` | N=待反量化元素数，4cycles流水线填充延迟 |
| **grf** | GRF  | `19` (固定延迟) | 1 GHz | `23.7833` | `47887` | 无法流水，每次调用固定19cycles |
| **fp32_resampler(下采样，51->32，连续流水计算)** | FP32 下采样 | `T = N + 2` | 1 GHz | `3.95` | `4719` | N=输出元素数，3cycles延迟，流水吞吐1elem/cycle；下采样：51点插值至32点 |
| **fp32_resampler(上采样，每次单点插值)** | FP32 上采样 | `T = 3` | 1 GHz | `3.95` | `4719` | N=输出元素数，3cycles延迟；上采样：64点插值到512点, 根据讨论每次进行单点插值，和距离计算并行（淹没时间）| 
| **fp32_resampler_up_mem(下采样lut)** | 下采样ram | `T = 1` | 1 GHz | `0.094` | `13485` | 用于保存下采样所需的数据，需要51+32个周期（无阶段依赖，上电即可执行，故只需统计能耗） | 
| **fp32_resampler_up_mem(上采样lut)** | 上采样ram | `T = 1` | 1 GHz | `0.094` | `85331` | 用于保存上采样所需的数据，需要64+512个周期（无阶段依赖，上电即可执行，故只需统计能耗） | 
| **Coarse_filter_fps128in_knn512in** | FPS/KNN可重构粗筛电路 | `T_fps_mode1 = 1; T_fps_mode0 = 3; T_knn_mode = 4`| 1GHz | fps_mode0: 190; fps_mode1: 210; knn_mode: 200 | 644686 | fps_mode1对于前10%的采样点，每阶段针对sram-cim array输出的128个距离值进行max排序，每次开销1个cycle，持续工作8个阶段，最终得到1024个距离值中的max值，无需使用细筛电路；fps_mode0对于后90%的采样点，每个阶段对于128个距离值进行基于预测距离的粗筛，每次开销3个cycles(被隐藏在sram-cim array距离计算的6cycles周期中)，等8个粗筛周期后输出最多32个max候选距离值至细筛电路；knn_mode，在8组128个距离计算完，且最后一组FPS排序完成后，复用1024个距离值进行KNN粗筛，由于根据动态阈值进行调整，最多6次迭代周期，KNN粗筛开销4个周期+1个output周期，示例：KNN第1次阈值即满足要求，开销4+1=5cycles；第6次阈值满足要求，开销4*6+1=25cycles(也是最大开销)。最终将N个min值输出至细筛电路(32≥N≥24) |
| **fine_filter** | FPS/KNN可重构细筛电路 | `T = 2` | 1GHz | 38 | 53261 | FPS和KNN细筛均2cycles |
| **dist2_update_table** | 最小距离更新表 | `T = 1` | 1GHz | 30 | 26610 | FPS每阶段采样后更新最小距离表 |
| **SRAM_CIM_array** | sram-cim_array 距离计算宏 | `T = (num_input_points / num_array_parrallelism) * (num_xyz_quant_bit / num_bit_slice)` | 1GHz | 6+1.3*128 | / | num_bit_slice=2; num_array_parralleism = 128; 每次计算128个距离值，开销(num_xyz_quant_bit / num_bit_slice)个周期；1.3为Local Computing Unit中的移位加法器，128个LCU，用于每个bit_slice计算时进行移位累加 |
| **dist2_merge_adder** | 将距离值的交叉项和自平方项聚合 | `T = 1` | 1GHz | 1.3 | 2179 | 在sram_cim_array距离计算后，执行1个周期；共128个 |
| **dist2_engine** | 脉动距离阵列 | `T = 131(64个距离计算)` | 1GHz | 94 | 89884 | 131cycles完成64个距离计算；1024点需16批 |
| **sort_64in_batch32min** | 每次处理64个距离的冒泡排序+merge | `T = 22` | 1GHz | 312 | 180441 | 配合dist2_engine，每次输出64个距离，将64个距离处理为当前batch数据（64个距离）的32min，采用思想冒泡排序+merge |
| **final_merge** | merge16个batch的32min，得到最终的32min | `T = 24` | 1GHz | 698 | 418299 | 配合dist2_engine，将16个batch的32min进行merge，得到最终32min，采用思想merge |
| **Coarse_filter_fpsknn128in** | FPS/KNN可重构粗筛电路 | `T_fps_mode1 = 1; T_fps_mode0 = 1; T_knn_mode = 1`| 1GHz | 81.12 | 94204 | fps_mode1对于前10%的采样点，每阶段针对sram-cim array输出的128个距离值进行max排序，每次开销1个cycle，持续工作8个阶段，最终得到1024个距离值中的max值，无需使用细筛电路；fps_mode0对于后90%的采样点，每个阶段对于128个距离值进行基于预测距离的粗筛，每次开销1个cycles(被隐藏在sram-cim array距离计算的6cycles周期中)，等8个粗筛周期后输出最多32个max候选距离值至细筛电路；knn_mode，在每组128个距离计算完成且FPS粗筛完成后，复用128个距离值进行KNN粗筛，将每组距离根据当前阈值粗筛后的点保存在buffer中，如果8组距离粗筛完成后，为满足点数收敛条件，则在已筛选的子集中进一步采用更新的阈值，再次进行新的一轮粗筛，粗筛周期需根据子集点数和粗筛输入端口数目(128)进行计算。最终将N个min值输出至细筛电路(目前约束，32≥N≥24) |

### 备注：sort_64in_batch32min / final_merge
由于上游dist2_engine每次只能输出64个数据给排序电路，一共1024个数据，需要分16个阶段输出。
sort_64in_batch32min模块，每次处理64个输入输出局部32min，分时复用16个阶段，第2个阶段仍然用sort_64in_batch32min进行处理。反复16个阶段后，得到16个32min，接下来采用final_merge模块，得到最终topk32。思想即冒泡排序+merge

### 更新日志：3-14
1、coarse_filter：不同模式功耗更新
2、baseline_bitonic_sort_1024_max_32min：不同模式功耗更新
3、新增dist2_engine对应的topk电路

### 更新日志：3-15
1、op_layer_norm：更新电路设计，去除buffer，将原本buffer面积酌情替换为增加内部alu资源，以内部计算并行度；功耗和周期公式均需更新

### 更新日志：3-18
1、新增'Coarse_filter_fpsknn128in'：粗筛接口为128，KNN粗筛逻辑更改，第一个阈值周期，会和距离计算以及FPS粗筛同时流水进行，后续如果未满足点数约束，需要更新阈值并在已筛选的子集中进一步完成粗筛过滤，反复直至实现约束。