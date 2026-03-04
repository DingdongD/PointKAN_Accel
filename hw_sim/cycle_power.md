# 数字模块仿真器性能评估表

## 1. 核心模块性能指标 (Core Module Metrics)

| 模块名称 (Module) | 功能描述 | 周期数 (Cycles) | 频率 (Freq MHz) | 功耗 (Power mW) | 面积 (Area mm²) | 备注/优化点 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Spool** | pointkan非线性算子 | `` | `` | `` | `` |  |
| **BatchNorm** | 批归一化加速 | `` | `` | `` | `` |  |
| **De-Quantizer** | 反量化 (INT8->FP32) | `` | `` | `` | `` |  |
| **Quant Engine** | 量化引擎 | `` | `` | `` | `` |  |
| **Resampler** | 重采样单元 | `` | `` | `` | `` | 硬件插值 |
| **Multiplier** | 乘法器 | `` | `` | `` | `` |  |
| **Adder** | 加法器 | `` | `` | `` | `` | CSA架构 |
| **FPS-KNN** | FPS-KNN排序规约 | `` | `` | `` | `` | *K=不同数值场景？ |

> **单位说明**: Cycles为单次标准操作周期; Power为动态功耗; Area为综合后面积.



## 2. 关键技术亮点 (Key Highlights)

