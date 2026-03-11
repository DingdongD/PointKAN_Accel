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

```Bash
CMD='python infer_cpu.py --batch 1' bash scripts/profile_cpu_infer.sh