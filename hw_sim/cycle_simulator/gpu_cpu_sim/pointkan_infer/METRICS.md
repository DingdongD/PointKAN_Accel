# 指标说明：仅 model(x) 推理时间与能耗

本仓**只关心 model(x) 一次前向**的推理时间与能耗，不统计进程级指标。下面说明各字段含义及测量方式。

---

## 一、本仓实际使用的指标（仅推理）

| 字段 | 含义 | 单位 |
|------|------|------|
| **inference_only_per_forward_s** | **单次** model(x) 的推理时间（多 batch 时 = inference_only_s / num_batches） | 秒 |
| **inference_only_s** | 本 run 内**所有** model(x) 的推理时间之和（如 20 次前向则约为单次的 20 倍） | 秒 |
| **gpu_energy_j** | GPU 在本 run 期间的能耗（NVML/dmon 采样积分） | 焦耳 |
| **cpu_energy_inference_pkg_j** / **cpu_energy_inference_total_j** | **仅推理阶段** CPU 能耗（pyRAPL 只包住 model 前向循环） | 焦耳 |

绘图与 `--print-inference-only` 终端输出**只使用以上字段**。

---

## 二、进程级字段（本仓不用于图表，可忽略）

以下字段由 base_version 的 profiler 记录，**与整条命令/进程相关**，本仓不画图、不参与对比：

| 字段 | 含义 | 说明 |
|------|------|------|
| **latency_s** / **wall_time_s** | 进程从启动到退出的墙上时间 | 含 import、建模型、数据准备、所有 batch、进程退出等 |
| **cpu_energy_pkg_j** / **cpu_energy_total_j** | 整进程的 CPU 能耗（perf/sysfs RAPL） | 本仓用 pyRAPL 仅包推理循环，不用这些 |
| **cpu_avg_percent** / **rss_peak_mb** | 进程 CPU 利用率、内存峰值 | 与 model(x) 推理无直接对应 |
| **perf_stats** / **rapl_source** | perf 或 sysfs 的整进程统计 | 本仓默认不启用 |

`summary.json` 和 `run_*.json` 里仍会写入这些字段（供调试或其它用途），但**本仓脚本与图表一律只看上面的「仅推理」指标**。

---

## 三、model(x) 推理时间是怎么测的

### CPU

- **方式**：在 **model(x) 前后**用 `time.perf_counter()` 打点，只算中间一段。
- **代码位置**：`rapl_inference.time_forward_seconds(device, forward_fn)`，当 `device.type == "cpu"` 时：
  - `t0 = time.perf_counter()`
  - 执行 `forward_fn()`（即一次 `net(x)`）
  - `return time.perf_counter() - t0`
- 即：**统计的就是「model(x) 前一刻到 model(x) 后一刻」的 CPU 时间**，不含数据准备、不含进程启动/退出。

### GPU

- **方式**：用 **torch.cuda.Event** 在 GPU 流上记录 model(x) 前后的时间戳，再算差值。
- **代码位置**：同上 `time_forward_seconds`，当 `device.type == "cuda"` 时：
  - `torch.cuda.synchronize()` 保证前面算完
  - `start_ev.record()` → 执行 `forward_fn()`（即一次 `net(x)`）→ `end_ev.record()`
  - `torch.cuda.synchronize()` 后 `start_ev.elapsed_time(end_ev)` 得到毫秒
- 即：**统计的是 GPU 执行 model(x) kernel 的时间**，不含 CPU 端、不含整进程。

---

## 四、CPU 能耗是怎么测的

- **方式**：在推理脚本里用 **pyRAPL**，`Measurement.begin()` 与 `Measurement.end()` **只包住「推理循环」**（即多次 model(x) 的那段），不包 import、不包建模型、不包进程结束。
- **写入**：推理脚本把 `cpu_energy_inference_*` 写入 `*_inference_only.json`，profiler 读入后汇总到 `summary.json` 的 aggregate。
- 若未安装 pyRAPL 或运行异常（如 returncode≠0），这些字段为 null，图中 CPU 能耗显示为 0。

---

## 五、为何 CPU 的 inference_only 有时是 null

若某次 run 的 **returncode 不为 0**（例如脚本在写 `*_inference_only.json` 之前就异常退出），则 profiler 读不到该次的推理时间/能耗，aggregate 里 `inference_only_*`、`cpu_energy_inference_*` 会为 null。  
此时应查看该 run 的 `run_*.stderr.txt` 或 stdout 定位错误；修复后重跑即可得到正常的推理指标。
