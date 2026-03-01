# FPS / FPS-KNN 评估（ModelNet40）

基于 `main.py` 使用的 ModelNet40 数据集，完成 todo_1.md 中的 FPS 校准与 FPS-KNN 对比需求。

## 环境

与训练脚本相同：在 `classification_ModelNet40` 目录下需已安装 `torch`、`pointnet2_ops`、`h5py` 等（与运行 `main.py` 一致）。

## 1. FPS 校准与 radii 曲线

- **脚本**: `eval_fps_radii.py`
- **作用**:
  - 按类别层次采样构建校准集（默认每类 50 个样本）
  - 对每个样本做 FPS，记录每步的「最小距离」radii
  - 统计校准集上 radii 的分布，绘制曲线：横轴 = 采样迭代次数，纵轴 = radii（mean ± std）
- **运行示例**（在 `classification_ModelNet40` 下）:

```bash
# 完整 1024 步 FPS，并出图
python opt_fps/eval_fps_radii.py --partition train --num_points 1024 --fps_steps 1024 --samples_per_class 50 --out_dir opt_fps/figures

# 快速测试（少样本、少步数、不绘图）
python opt_fps/eval_fps_radii.py --samples_per_class 10 --fps_steps 256 --batch_size 16 --no_plot
```

- **输出**: 在 `--out_dir` 下生成 `fps_radii.png` / `fps_radii.pdf`（满足 todo_1.md 的绘图风格：DPI 600、PNG/PDF、`bbox_inches='tight'`、`pad_inches=0.02`、透明背景、Arial 等）。

## 2. FPS-KNN 对比（vanilla vs adaptive）

- **脚本**: `eval_fps_knn_compare.py`
- **作用**:
  - **Vanilla FPS** 使用**朴素实现**（与 `fps_with_knn_adaptive` 相同的种子策略 `--fps_seed first/random/max_norm`），便于与优化版对齐对比。
  - **KNN** 仍为 `models.pointkan.knn_point`。与 **adaptive** 实现 `fps_with_knn_adaptive` 在相同点云上对比。
  - 默认 **batch_size=1**，便于得到每样本 IoU 并做统计分析。
  - 可保存 **FPS/KNN 索引** 与 **每样本 IoU** 到 JSON。
- **运行示例**（在 `classification_ModelNet40` 下）:

```bash
# 默认：batch_size=1，50 样本，结果写 txt + JSON，前 3 个样本保存索引
python opt_fps/eval_fps_knn_compare.py --partition test --num_batches 50 --K 512 --k_neighbors 24 \
  --out_txt opt_fps/figures/fps_knn_compare.txt \
  --out_json opt_fps/figures/fps_knn_compare.json \
  --save_indices_n 3
```

- **输出**: 控制台与 `--out_txt` 的摘要；`--out_json` 含 `per_sample_fps_iou`、`per_sample_knn_iou`、`summary`、可选 `saved_indices`（前 N 个样本的 FPS/KNN 索引与 IoU）。

### 动态阈值 KNN 参数网格搜索（tol / max_iter / init_threshold）

- **默认**：sweep 在指定 **partition**（默认 `test`）上按 shuffle 取前 **num_batches** 个样本，**不是**校准集。
- **在校准集上 sweep**：加 **`--use_calibration`**，并可选 **`--samples_per_class 50`**。校准集为按类别分层采样（与 `eval_fps_radii` 一致），在 `partition`（建议 `train`）上构建。
- 同时指定 **`--sweep_tol`** 与 **`--sweep_max_iter`** 时，对每组 `(tol, max_iter)` 或 `(tol, max_iter, init_threshold)` 跑一次对比，将各组的 **FPS/KNN IoU**、**iters_used**（迭代次数）、**thresholds**（最终阈值）统计写入 **JSON**，并自动生成 **图**：
  - **热力图**：KNN IoU、FPS IoU、**iters_used 均值**、**最终阈值均值**（tol × max_iter；若扫 init_threshold 则按 init_threshold 分片出图，文件名带 `_init_threshold_<值>`）
  - **曲线图**：KNN IoU vs tol、KNN IoU vs max_iter
- 可选 **`--sweep_init_threshold`**：逗号分隔的 init_threshold 列表，加入网格后会对每个 init_threshold 分别出图。
- **运行示例**:

```bash
# tol × max_iter × init_threshold，结果与图保存到 opt_fps/figures（默认在 test 上取 50 batch）
python opt_fps/eval_fps_knn_compare.py \
  --sweep_tol "0.01,0.05,0.1" \
  --sweep_max_iter "5,10,20" \
  --sweep_init_threshold "0.05,0.1,0.2" \
  --num_batches 50 \
  --out_json opt_fps/figures/sweep_tol_max_iter.json \
  --sweep_out_dir opt_fps/figures

# 在校准集上 sweep（与 eval_fps_radii 同一套分层采样）
python opt_fps/eval_fps_knn_compare.py \
  --partition train --use_calibration --samples_per_class 50 \
  --sweep_tol "0.01,0.05,0.1" --sweep_max_iter "5,10,20" \
  --num_batches 200 --out_json opt_fps/figures/sweep_calibration.json
```

- 或直接执行脚本：`bash opt_fps/run_sweep_tol_max_iter.sh`
- **JSON 结构**：`config`（含 `tol_list`、`max_iter_list`、`init_threshold_list` 等）、`grid`（每项含 `tol`、`max_iter`、可选 `init_threshold`、`fps_iou_mean`、`knn_iou_mean`、**`iters_used_mean`**、**`iters_used_std`**、**`thresholds_mean`**、**`thresholds_std`**、`num_samples`）。

---

## 3. PointKAN 推理 Latency 与模块 Breakdown

- **脚本**: `benchmark_latency.py`（与 `main.py` 同目录）
- **作用**: 在 ModelNet40 上测 PointKAN 推理时延，并给出各模块耗时（embedding、各 stage 的 GAM/LFP/GFP、pool、classifier）。
- **运行示例**（在 `classification_ModelNet40` 下）:

```bash
# 使用随机权重，warmup 20，repeat 100，结果写 JSON
python benchmark_latency.py --model pointKAN --num_points 1024 --warmup 20 --repeat 100 \
  --out_json opt_fps/figures/latency_breakdown.json

# 指定 checkpoint
python benchmark_latency.py --checkpoint checkpoints/pointKAN-xxx/last_checkpoint.pth \
  --out_json opt_fps/figures/latency_breakdown.json
```

- **输出**: 控制台打印 full forward 均值±标准差及各模块 breakdown（ms）；`--out_json` 保存完整数值供分析。
- **Shell 脚本**: `run_benchmark_latency.sh`（在项目根目录执行即可）。
