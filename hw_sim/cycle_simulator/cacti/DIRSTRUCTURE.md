# CACTI 目录结构说明

本目录在 CACTI 上游源码基础上增加了脚本与统一输出目录，便于维护与复现。

## 目录一览

```
cacti/
├── README                 # CACTI 使用说明（上游）
├── DIRSTRUCTURE.md        # 本说明文件
├── makefile, cacti.mk     # 构建
├── *.cc, *.h              # CACTI 源码（cache/DRAM/IO 建模）
├── *.cfg                  # 根目录配置：cache.cfg, ddr3.cfg, dram.cfg, lpddr.cfg 等
├── tech_params/           # 工艺参数（180nm～16nm）
├── sample_config_files/   # 示例配置（上游）
│
├── scripts/               # 本仓 Python 脚本（扫描、绘图、接口）
│   ├── scan_ddr_cfg_size.py   # DDR3 容量扫描，生成 outputs 下 cfg/log/csv 与 figures
│   ├── run_capacity_scan.py   # Cache 容量扫描，同上
│   ├── cacti_dram_interface.py # CACTI DDR IO 校准与 DRAM 能耗/延迟接口
│   └── pointcloud_dram_sim.py # Point cloud 访存仿真与图表
│
└── outputs/               # 所有生成结果（可删后由脚本重新生成）
    ├── generated_cfgs/        # run_capacity_scan 生成的 cache 配置与 .cfg.out
    ├── generated_ddr3_cfgs/   # scan_ddr_cfg_size 生成的 ddr3 配置与 .cfg.out
    ├── logs/                  # run_capacity_scan 的 CACTI 运行日志
    ├── ddr3_logs/             # scan_ddr_cfg_size 的 CACTI 运行日志
    ├── dram_io_calib_runs/    # cacti_dram_interface 校准产生的 cfg/log/json
    ├── *.csv                  # 扫描结果汇总（capacity_scan_results.csv 等）
    └── figures/               # 所有 PNG 图表
```

## 使用方式

- **编译 CACTI**：在 cacti 根目录执行 `make`，得到可执行文件 `cacti`。
- **运行脚本**：建议在 cacti 根目录下执行，以便脚本通过 `Path(__file__).parent.parent` 定位根目录：
  - `python scripts/scan_ddr_cfg_size.py` — DDR3 容量扫描与绘图
  - `python scripts/run_capacity_scan.py` — Cache 容量扫描与绘图
  - `python scripts/cacti_dram_interface.py [选项]` — IO 校准或单次能耗/延迟查询
  - `python scripts/pointcloud_dram_sim.py` — Point cloud DRAM 仿真与绘图
- **输出位置**：所有生成的 cfg、log、csv、png 均落在 `outputs/` 及 `outputs/figures/`，根目录不再产生散落文件。

## 与上游的差异

- 新增 `scripts/`、`outputs/` 及本说明文件；其余为 CACTI 上游或少量示例配置，未改源码逻辑。
