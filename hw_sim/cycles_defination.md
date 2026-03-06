# 数字模块仿真器性能评估表

## 1. 核心模块性能指标 (Core Module Metrics)

### 1.1 已实现 RTL 算子

| 模块名称 (Module) | 功能描述 | 周期数公式 (Cycles) | 频率 (Freq) | 功耗 (Power,mw) | 面积 (Area,um2) | 备注/优化点 |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **op_relu** | FP32 ReLU 逐元素激活 | `T = N_elements` | 1 GHz | `0.45` | `222` | 可并行 M 个 op_relu 电路，周期缩减为 `N_elements / M`，功耗和面积对应倍增 M |
| **op_spool** | FP32 S-Pool (Softmax加权池化) | `T = N_points × D × ((K/4)×23 + 14)` | 1 GHz | `17` | `34375` | K=近邻数，D=通道数，N_points=输出点数 |
| **op_layer_norm** | FP32 LayerNorm 双FSM流水线 | `T = (6D+32) + N_vec × 10D` | 1 GHz | `180` | `177025` | D=normalized_shape (需 D≥9)，N_vec=待归一化向量总数 |
| **op_adaptive_max_pool1d** | FP32 全并行MaxPool (output_size=1) | `T = N_fibers` | 1 GHz | `13.2` | `18406` | MAX_DIM=64；N_fibers=池化输出元素总数（池化维≤64）。举例：K近邻池化 [N_points, D, K]→[N_points, D] 时 N_fibers=N_points×D；全局池化 [D, N_points]→[D] 时 N_fibers=D |
| **fp32_mul** | FP32 乘法器 | `1` (单次延迟) | 1 GHz | `2.5` | `2024` | 寄存器1拍输出 |
| **fp32_add** | FP32 加法器 | `1` (单次延迟) | 1 GHz | `1.16` | `1273` | 寄存器1拍输出 |
| **fp32_div** | FP32 除法器 (Newton-Raphson) | `~10` (单次延迟) | 1 GHz | `6307` | `2.59` | 3轮NR迭代 |


### 1.2 待实现模块

| 模块名称 (Module) | 功能描述 | 周期数 (Cycles) | 频率 (Freq) | 功耗 (Power) | 面积 (Area) | 备注/优化点 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **De-Quantizer** | 反量化 (INT8->FP32) | `` | `` | `` | `` |  |
| **Quant Engine** | 量化引擎 | `` | `` | `` | `` |  |
| **Resampler** | 重采样单元 | `` | `` | `` | `` | 硬件插值 |
| **FPS-KNN** | FPS-KNN排序规约 | `` | `` | `` | `` | *K=不同数值场景？ |


