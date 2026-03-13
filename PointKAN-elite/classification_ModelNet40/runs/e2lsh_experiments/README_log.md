# 如何理解 E2LSH 实验日志

## 日志结构

- **Part 1（参数扫）**：对每个 (L, w, candidate_mode) 跑一遍 E2LSH + int_vanilla，汇报：
  - **cand_ratio**：候选集大小 / ref 点数（越小越省算力，但可能伤 recall）
  - **shortage**：候选数 < k 的 query 占比（越高说明很多 query 根本凑不齐 k 近邻）
  - **recall@k**：相对 float_vanilla GT 的 Recall@k（越高越接近“全 ref”的 KNN）

- **Part 2（backend 对比）**：固定一组 E2LSH 参数，跑三种 backend（int_vanilla / safe_strict / fastnn），看 IoU、Recall、cand_ratio、shortage。

## 为什么 Part 2 会“这么差”？（例如 recall≈0.06, shortage=100%）

你这份日志里 Part 2 用的是 **candidate_mode=intersection**，且索引是 **two-level E2LSH**：

1. **Two-level 索引**：每张表用 (H1, H2) 两个键，只有 (slot, fp) 完全一致才算同桶。
2. **Intersection**：要求一个 ref 点在 **L 张表里都和 query 同桶**（同一 (slot, fp)）才会进入候选集。
3. 两者叠加 → 候选集极严：几乎只有和 query 在 L×K 维哈希空间里都撞到同一格的点才会留下，所以：
   - **cand_ratio 极低**（如 0.1%），
   - **shortage=100%**（几乎每个 query 的候选数都 < k），
   - **recall@k 崩盘**（连 k 个候选都凑不齐，自然无法复现 GT 的 k-NN）。

所以“差”主要来自：**two-level + intersection 过于严格**，适合论文复现“最严格”设定，不适合作为默认的实用配置。

## 如何得到有意义的 Part 2 结果？

- 把 Part 2 的 **candidate_mode 改为 union**（或 min_hits），并选 Part 1 里 recall 较高、shortage 较低的 (L, w)，例如：
  - `PICK_CANDIDATE_MODE=union PICK_L=8 PICK_W=0.5 ./run_e2lsh_experiments.sh --skip_sweep`
- 你当前 Part 1 里：L=8, w=0.5, union → recall≈0.997, shortage≈0%，cand_ratio≈26%，这类配置下 Part 2 的 IoU/Recall 才会接近“E2LSH 候选 + backend”的真实水平。

## Part 1 里“小 w”为什么也差？

w 小 → 桶变细 → 每张表里同桶的点很少；再取 union 仍然候选偏少，shortage 高、recall 低。只有 w 调大（如 0.3、0.5）或减少表数 L，候选才够多，recall 才上来。这是 E2LSH 的经典 trade-off：桶越细越准但候选越少，桶越粗候选越多但越糙。
