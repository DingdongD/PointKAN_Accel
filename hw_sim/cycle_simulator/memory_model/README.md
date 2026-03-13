# memory_model

内存建模模块：集成 CACTI（SRAM/缓存）与 DRAMsim3（DRAM）仿真，支持点云流水线 trace 生成与统一内存系统接口。

## 目录结构

```
memory_model/
├── README.md
├── configs/
│   ├── cacti/           # CACTI 配置与生成配置
│   └── dramsim3/        # DRAMsim3 配置与生成配置
├── traces/              # 访问 trace 与生成 trace
├── outputs/             # CACTI / DRAMsim3 仿真输出
├── memory_model/        # Python 包
└── demo/                # 演示脚本
```

## 依赖

- CACTI（SRAM/缓存面积与功耗）
- DRAMsim3（DRAM 周期与功耗）
- Python 3.8+

## 使用

见 `demo/` 下示例：`demo_cacti.py`、`demo_dramsim3.py`、`demo_pointcloud_pipeline.py`。
