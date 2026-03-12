# nn_infer_profiler

统一分析神经网络在 GPU / CPU 推理时的 latency 和 energy。

## 功能
- Python 统一入口：`profiler.py`
- GPU 功耗采样：优先 `pynvml`
- GPU energy：对功率时间序列做梯形积分
- CPU 进程采样：`psutil`
- 外部工具脚本：`perf` / `ftrace` / `nsys` / `nvidia-smi dmon` / `bpftrace`

### GPU 推理分析
```Bash
CMD='python infer.py --batch 1' bash scripts/profile_gpu_infer.sh
```


### CPU 推理分析
```Bash
CMD='python infer_cpu.py --batch 1' bash scripts/profile_cpu_infer.sh
```

输出

summary.json：聚合结果
run_*.json：单次运行摘要
*_gpu_samples.csv：GPU 功耗 / 利用率采样
*_cpu_samples.csv：CPU / RSS 采样

perf, trace-cmd, nsys, bpftrace 需要系统已安装，且通常需要更高权限

NVML 读数与驱动 / GPU 架构相关，论文中建议注明采样周期和采样源
```bash
CMD='python infer.py --batch 1' \
GPU_INDEX=0 \
WARMUP=2 \
REPEAT=5 \
ENABLE_PERF=1 \
ENABLE_NSYS=1 \
bash scripts/profile_gpu_infer.sh
```

```bash
CMD='python infer_cpu.py --batch 1' \
WARMUP=2 \
REPEAT=5 \
ENABLE_PERF=1 \
bash scripts/profile_cpu_infer.sh
```