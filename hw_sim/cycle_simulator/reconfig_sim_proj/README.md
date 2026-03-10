# reconfig_sim_proj

A modular Python simulator for a reconfigurable systolic array supporting OS/WS dataflows.

## What is implemented

- GEMM tiler / dispatcher / cycle-energy model
- Conv2D via im2col-to-GEMM
- Analytic buffer model
- Analytic NoC model
- Modular controllers at the simulator level (dispatcher and mode assignment)

## Why not BookSim by default?

This project uses an analytic NoC model first because it is fast and easy to sweep across many design points.
BookSim is useful when you need packet-level contention and detailed router/link timing for a concrete traffic pattern.
A practical flow is:

1. Use the analytic model for architecture exploration and OS/WS scheduling sweeps.
2. Export a few representative traffic points.
3. Calibrate the analytic parameters against BookSim for those points.

## Directory structure

- `reconfig_sim/configs`: configuration dataclasses
- `reconfig_sim/models`: tile, buffer, noc, cycle-energy models
- `reconfig_sim/schedulers`: tiler and dispatcher
- `reconfig_sim/workloads`: GEMM / Conv2D workloads
- `reconfig_sim/utils`: im2col and report helpers
- `reconfig_sim/examples`: runnable demo

## Run

```bash
cd reconfig_sim_proj
PYTHONPATH=. python reconfig_sim/examples/run_demo.py
```

## Notes on buffer and NoC modeling

### Buffers
Current model is analytic and counts bytes for A/B/C movement per tile. It reports:
- A/B reads
- C read/write bytes
- energy using per-byte coefficients

This is enough for first-order architecture studies. Later, you can refine it with:
- double buffering
- spill / refill policy
- SRAM banking conflicts
- DMA burst startup costs

### NoC
Current model is analytic, not BookSim-backed. It estimates:
- bytes moved
- average hop cost
- cycles from link bandwidth
- energy from byte-hop and flit-router terms

This is a good fit for early-stage systolic array exploration because traffic is structured and repeated.
If you later need detailed contention analysis, add a BookSim adapter that consumes exported traffic traces.

---

## 中文说明

### 工程概述

本工程是一个**可重构脉动阵列的周期级仿真器**，支持 **OS（Output Stationary）** 与 **WS（Weight Stationary）** 两种数据流模式，用于架构探索与调度策略评估。

### 已实现功能

- **GEMM 分块与调度**：按阵列规模（arr_r × arr_c）将矩阵乘切分为多个 tile；调度器可为每块指定 OS/WS 或按启发式自动选择（AUTO）。
- **Conv2D**：通过 im2col 将卷积转为 GEMM，再复用上述分块与周期/能耗模型。
- **周期-能耗模型**：对每个 tile 按 OS/WS 公式估算计算周期（含 clear/preload、compute、drain/gap），并叠加缓冲读写与 NoC 传输的周期与能耗。
- **缓冲模型**：解析模型，按 A/B/C 读写字节数与每字节能耗系数估算缓冲能耗（当前不建模 stall）。
- **NoC 模型**：解析模型，按字节数、平均跳数、链路带宽与 flit 能耗估算传输周期与能耗；默认未集成 BookSim，适合先做快速架构扫描，再对少数流量点用 BookSim 标定。

### 目录与模块对应

| 目录/文件 | 作用 |
|----------|------|
| `reconfig_sim/configs/defaults.py` | 阵列、缓冲、NoC、时序、能耗等配置数据类 |
| `reconfig_sim/models/tile.py` | 单块描述 TileDesc（op_type、mode、m0/k0/n0、m_tile/k_tile/n_tile） |
| `reconfig_sim/models/buffer_model.py` | 缓冲访问与能耗估计 |
| `reconfig_sim/models/noc_model.py` | NoC 传输周期与能耗估计 |
| `reconfig_sim/models/cycle_energy.py` | 单块周期-能耗汇总（计算 + 缓冲 + NoC + 调度/控制） |
| `reconfig_sim/schedulers/tiler.py` | GEMM / Conv2D(im2col) 分块器 |
| `reconfig_sim/schedulers/dispatcher.py` | 为 tile 分配 OS/WS 及模式切换惩罚 |
| `reconfig_sim/workloads/gemm.py` | GEMM 工作负载与 golden 校验和 |
| `reconfig_sim/workloads/conv2d.py` | Conv2D 工作负载（NCHW） |
| `reconfig_sim/utils/im2col.py` | NCHW im2col 与 conv2d_to_gemm |
| `reconfig_sim/utils/report.py` | JSON 报告写入 |
| `reconfig_sim/simulator.py` | 仿真器入口：run_gemm / run_conv2d，内部调用分块、调度与周期-能耗模型 |
| `reconfig_sim/examples/run_demo.py` | 示例：一次 GEMM + 一次 Conv2D，输出周期/能耗并写 JSON |

### 运行方式

在工程根目录 `reconfig_sim_proj` 下执行：

```bash
PYTHONPATH=. python reconfig_sim/examples/run_demo.py
```

会生成 `reconfig_sim_demo_report.json`，其中包含 GEMM/Conv2D 的周期、能耗、模式分解、配置与 trace。冒烟测试：

```bash
PYTHONPATH=. python tests/smoke_test.py
```

### 缓冲与 NoC 建模说明

- **缓冲**：当前仅按每块 A/B/C 读写字节数 × 每字节能耗计算，不区分双缓冲、bank 冲突等；适合一阶架构对比，后续可扩展。
- **NoC**：解析模型基于总字节、平均跳数、链路带宽与竞争因子算周期，基于跳数与 flit 数算能耗；需要更细粒度拥塞与延迟时，可对接 BookSim 并用量化流量做标定。
