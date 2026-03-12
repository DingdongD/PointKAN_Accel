"""
spec_kan_fps_opt.py：基于 segment 阈值过滤的 FPS 与 vanilla FPS 的 IOU 评估

流程：
  1. 先跑一次 exact FPS，得到 radii_ref（同一批点云）
  2. 用 seg_filter 模式模拟 segment 阈值过滤的 FPS
  3. 在 ModelNet40 上评估 seg_filter FPS 与 vanilla (exact) FPS 的索引集合 IOU
  4. 固定初始点（seed="first"），保证两种方式起点一致
"""
import os
import sys
import json
import argparse
import numpy as np
from typing import Tuple, Optional, Dict, List

# 与 spec_kan_fps 一致：当前任务目录与项目根目录加入 path，以便 from data import ModelNet40
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =========================================================
# 1. FPS：exact / seg_filter / block_filter（内部使用平方距离）
# =========================================================

@torch.no_grad()
def fps_adaptive_argmax(
    X: torch.Tensor,
    K: int,
    seed: str = "first",
    # --- selection mode ---
    mode: str = "exact",                 # "exact" | "seg_filter" | "block_filter"
    radii_ref: Optional[torch.Tensor] = None,  # (B,K) from an exact FPS run (same sample)
    segment_len: int = 32,               # e.g., 32 => 512/32=16 segments
    # block filter (hardware-like)
    block_len: int = 64,                 # points per block for block_filter
    # numerical
    empty_fallback: str = "global",       # "global" (recommended)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    带可选 segment 阈值过滤的 FPS，用于模拟硬件/分段选择。

    参数:
        X: 点云 [B, N, D]
        K: 采样点数
        seed: 初始点方式，"first" 表示固定为第 0 点（固定初始点）
        mode: "exact" | "seg_filter" | "block_filter"
        radii_ref: (B,K)，exact 一次跑完后得到的半径曲线，seg_filter 时用作阈值参考
        segment_len: 每段长度，如 32 表示 512 点分 16 段，每段用 seg_end 处的半径做 tau
        block_len: block_filter 时每块点数
        empty_fallback: 候选为空时的回退策略，"global" 表示退化为全局 argmax

    返回:
        idx: (B,K) 采样索引
        radii: (B,K) 覆盖半径曲线（平方距离，非欧氏）
        distances_matrix: (B,K,N) 各选中点到全部点的距离（平方）
        stats: candidate_ratio (B,K)、used_fallback (B,K) 等诊断信息

    注意：内部全程使用平方欧氏距离（无 sqrt）；若需欧氏半径请对 radii 做 sqrt。
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    K = min(K, N)

    # Precompute norms
    x2 = (X * X).sum(dim=-1)  # (B,N)

    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=dtype, device=device)  # squared radii
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    distances_matrix = torch.zeros((B, K, N), dtype=dtype, device=device)  # squared distances

    # Stats
    candidate_ratio = torch.ones((B, K), dtype=dtype, device=device)
    used_fallback = torch.zeros((B, K), dtype=torch.bool, device=device)

    # 固定初始点：first 表示每 batch 都取第 0 点，保证与 vanilla 一致
    if seed == "first":
        first = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        first = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    # Helper: compute squared distance from all X to C (B,D)
    def sqdist_to_center(C: torch.Tensor) -> torch.Tensor:
        # C: (B,D)
        c2 = (C * C).sum(dim=-1, keepdim=True)           # (B,1)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)          # (B,N)
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0)       # (B,N) squared
        return d2

    # init
    idx[:, 0] = first
    C0 = X[torch.arange(B), first]
    d2 = sqdist_to_center(C0)
    distances_matrix[:, 0, :] = d2
    min_dist.copy_(d2)
    radii[:, 0], _ = torch.max(min_dist, dim=1)
    min_dist[torch.arange(B), first] = 0.0

    # seg_filter / block_filter 需要 radii_ref，且与当前 (B,K) 一致（同批点云的 exact 结果）
    if mode in ("seg_filter", "block_filter"):
        if radii_ref is None:
            raise ValueError("mode 为 seg_filter/block_filter 时必须传入 radii_ref（同一样本的 exact FPS 半径曲线）。")
        if radii_ref.shape[:2] != (B, K):
            raise ValueError(f"radii_ref 形状应为 (B,K) = {(B,K)}，当前为 {tuple(radii_ref.shape)}")
        # 若外部传入的是欧氏半径，则平方以与内部平方距离一致
        if torch.median(radii_ref) > 0:
            # compare scale: squared distances roughly ~ median(x2) magnitude; euclidean radii roughly ~ sqrt
            # If radii_ref^2 is closer to median(x2) than radii_ref, treat radii_ref as euclidean.
            med_x2 = torch.median(x2)
            med_r = torch.median(radii_ref)
            if (med_r * med_r - med_x2).abs() < (med_r - med_x2).abs():
                radii_ref = radii_ref * radii_ref

    # Candidate selection helpers
    def select_farthest_exact(md: torch.Tensor) -> torch.Tensor:
        return torch.argmax(md, dim=1)

    def select_farthest_seg_filter(md: torch.Tensor, t: int) -> torch.Tensor:
        # 当前步 t 所属 segment：用该段末尾的半径作为阈值 tau，模拟 segment 阈值过滤
        L = max(1, int(segment_len))
        j = t // L
        seg_end = min((j + 1) * L - 1, K - 1)
        tau = radii_ref[:, seg_end]  # (B,) 平方距离阈值

        mask = md >= tau.unsqueeze(1)
        # stats
        cand_cnt = mask.sum(dim=1).to(dtype)
        candidate_ratio[:, t] = cand_cnt / float(N)

        empty = ~mask.any(dim=1)
        if empty.any():
            used_fallback[:, t] = True
            if empty_fallback == "global":
                # just ignore mask for those batches
                # implement by setting mask[empty] = True
                mask[empty] = True
            else:
                mask[empty] = True  # default to global anyway

        cand = md.clone()
        cand[~mask] = -float("inf")
        return torch.argmax(cand, dim=1)

    def select_farthest_block_filter(md: torch.Tensor, t: int) -> torch.Tensor:
        # same tau as seg_filter, but skip blocks that can't pass tau
        L = max(1, int(segment_len))
        j = t // L
        seg_end = min((j + 1) * L - 1, K - 1)
        tau = radii_ref[:, seg_end]  # (B,)

        BL = max(1, int(block_len))
        nb = (N + BL - 1) // BL

        # pad md to nb*BL
        padN = nb * BL
        if padN != N:
            pad = torch.full((B, padN - N), -float("inf"), dtype=md.dtype, device=md.device)
            md_pad = torch.cat([md, pad], dim=1)  # (B,padN)
        else:
            md_pad = md

        md_blk = md_pad.view(B, nb, BL)  # (B,nb,BL)

        # block flag: any >= tau
        flag = (md_blk >= tau.view(B, 1, 1)).any(dim=2)  # (B,nb)

        # stats: candidate points approximated by flagged blocks * BL (upper bound)
        candidate_ratio[:, t] = (flag.sum(dim=1).to(dtype) * BL) / float(N)

        empty = ~flag.any(dim=1)
        if empty.any():
            used_fallback[:, t] = True
            # fallback to global argmax
            flag[empty] = True

        # For unflagged blocks, set to -inf
        md_blk_masked = md_blk.clone()
        md_blk_masked[~flag.unsqueeze(-1)] = -float("inf")

        # block-local max and argmax
        blk_vals, blk_pos = md_blk_masked.max(dim=2)  # (B,nb), (B,nb) pos in block
        best_blk = blk_vals.argmax(dim=1)             # (B,)
        best_pos = blk_pos[torch.arange(B), best_blk] # (B,)

        farthest = best_blk * BL + best_pos
        farthest = farthest.clamp_max(N - 1)
        return farthest

    # FPS loop
    for t in range(1, K):
        if mode == "exact":
            farthest = select_farthest_exact(min_dist)
        elif mode == "seg_filter":
            farthest = select_farthest_seg_filter(min_dist, t)
        elif mode == "block_filter":
            farthest = select_farthest_block_filter(min_dist, t)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        idx[:, t] = farthest

        C = X[torch.arange(B), farthest]
        d2 = sqdist_to_center(C)
        distances_matrix[:, t, :] = d2

        torch.minimum(min_dist, d2, out=min_dist)
        radii[:, t], _ = torch.max(min_dist, dim=1)

        # prevent reselection
        min_dist[torch.arange(B), farthest] = 0.0

    stats = {
        "candidate_ratio": candidate_ratio,
        "used_fallback": used_fallback,
    }
    return idx, radii, distances_matrix, stats


# =========================================================
# 2. 数据集：ModelNet40（与 spec_kan_fps 一致）
# =========================================================

def _try_import_modelnet40():
    """优先使用项目 data.ModelNet40，否则尝试 PyG ModelNet。"""
    try:
        from data import ModelNet40
        return ModelNet40
    except Exception:
        pass
    try:
        from modelnet40 import ModelNet40
        return ModelNet40
    except Exception:
        return None


class ModelNet40Wrapper(Dataset):
    """统一输出 points [N,3]、label；与 spec_kan_fps 一致。"""
    def __init__(self, partition: str, num_points: int, root: str = "./data"):
        super().__init__()
        self.partition = partition
        self.num_points = num_points
        UserModelNet40 = _try_import_modelnet40()
        if UserModelNet40 is not None:
            self.backend = "user"
            self.ds = UserModelNet40(partition=partition, num_points=num_points)
            self.length = len(self.ds)
            return
        try:
            from torch_geometric.datasets import ModelNet
        except Exception as e:
            raise RuntimeError(
                "未找到 ModelNet40（已尝试 data.ModelNet40、modelnet40、PyG ModelNet）。"
                f" 原始错误: {repr(e)}"
            )
        self.backend = "pyg"
        split = "train" if partition == "train" else "test"
        self.ds = ModelNet(root=root, name="40", train=(split == "train"))
        self.length = len(self.ds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.backend == "user":
            pts, label = self.ds[idx]
            if isinstance(pts, np.ndarray):
                pts = torch.from_numpy(pts)
            pts = pts[:, :3].float()
            return pts, int(label)
        data = self.ds[idx]
        pts = data.pos.float()
        y = int(data.y.item()) if hasattr(data, "y") else 0
        N = pts.shape[0]
        if N >= self.num_points:
            choice = torch.randperm(N)[: self.num_points]
            pts = pts[choice]
        else:
            pad_idx = torch.randint(0, N, (self.num_points - N,))
            pts = torch.cat([pts, pts[pad_idx]], dim=0)
        return pts, y


def collate_points(batch):
    pts = torch.stack([b[0] for b in batch], dim=0)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pts, y


# =========================================================
# 3. IOU：vanilla (exact) FPS 与 seg_filter FPS 的索引集合交并比
# =========================================================

def compute_iou_batch(
    idx_vanilla: torch.Tensor,
    idx_seg: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """
    计算每个样本上两套 FPS 索引集合的 IOU，并返回均值与标准差。
    idx_vanilla / idx_seg: (B, K)，同一批点云、同一 K、固定初始点下的两套索引。
    返回: iou_per_sample (B,), mean_iou, std_iou
    """
    B, K = idx_vanilla.shape
    device = idx_vanilla.device
    iou_list: List[float] = []
    for b in range(B):
        set_a = set(idx_vanilla[b].cpu().numpy().tolist())
        set_b = set(idx_seg[b].cpu().numpy().tolist())
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        iou_b = (inter / union) if union > 0 else 1.0
        iou_list.append(iou_b)
    iou_per_sample = torch.tensor(iou_list, dtype=torch.float32, device=device)
    mean_iou = float(np.mean(iou_list))
    std_iou = float(np.std(iou_list))
    return iou_per_sample, mean_iou, std_iou


# =========================================================
# 4. 单样本可视化：候选点数与阈值随迭代变化（一图双纵轴，英文）
# =========================================================

def plot_one_sample_seg_filter(
    pts_one: torch.Tensor,
    K: int,
    segment_len: int,
    seed: str,
    device: torch.device,
    save_path: str,
) -> None:
    """
    以单样本 (1, N, 3) 为例：先 exact 得 radii_ref，再 seg_filter 得每步候选比例；
    在同一图内绘制「候选点数 vs 迭代」与「阈值 tau vs 迭代」，字体英文。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts_one = pts_one.to(device, dtype=torch.float32)
    N = pts_one.shape[1]

    # exact -> radii_ref
    _, radii_ref, _, _ = fps_adaptive_argmax(pts_one, K, seed=seed, mode="exact")
    # seg_filter -> candidate_ratio per step
    _, _, _, stats = fps_adaptive_argmax(
        pts_one, K, seed=seed, mode="seg_filter",
        radii_ref=radii_ref, segment_len=segment_len,
    )
    candidate_ratio = stats["candidate_ratio"][0].cpu().numpy()  # (K,)
    radii_ref_np = radii_ref[0].cpu().numpy()  # (K,)

    iterations = np.arange(K)
    candidate_count = candidate_ratio * N

    # 每步使用的阈值 tau：与 seg_filter 内一致，tau[t] = radii_ref[:, seg_end]
    tau = np.zeros(K, dtype=np.float64)
    L = max(1, segment_len)
    for t in range(K):
        j = t // L
        seg_end = min((j + 1) * L - 1, K - 1)
        tau[t] = float(radii_ref_np[seg_end])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.plot(iterations, candidate_count, color="C0", label="Candidate count", linewidth=1.2)
    ax1.set_ylabel("Candidate count", color="C0", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper right", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(iterations, tau, color="C1", label="Threshold (squared dist)", linewidth=1.2)
    ax2.set_ylabel("Threshold (squared dist)", color="C1", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", fontsize=10)

    plt.title("Segment-filter FPS: candidate count and threshold vs iteration (one sample)", fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================
# 5. 主流程：exact 取 radii_ref → seg_filter → 评估 IOU
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="seg_filter FPS 与 vanilla FPS 的 IOU 评估（ModelNet40）")
    p.add_argument("--root", type=str, default="./data", help="ModelNet40 数据根目录")
    p.add_argument("--partition", type=str, default="test", choices=("train", "test"),
                   help="评估集：train 或 test")
    p.add_argument("--num_points", type=int, default=1024, help="每样本点数 N")
    p.add_argument("--K", type=int, default=512, help="FPS 采样点数 K")
    p.add_argument("--segment_len", type=int, default=32,
                   help="seg_filter 每段长度，如 32 表示 512/32=16 段")
    p.add_argument("--batch_size", type=int, default=16, help="批大小")
    p.add_argument("--seed", type=str, default="first",
                   help="初始点方式，固定用 first 以保证与 vanilla 一致")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./runs/fps_seg_filter_iou",
                   help="结果与 log 保存目录")
    p.add_argument("--vis_one_sample", action="store_true",
                   help="绘制一个样本的候选点数与阈值随迭代变化图（英文标注）")
    return p.parse_args()


@torch.no_grad()
def run_evaluation(
    loader: DataLoader,
    K: int,
    segment_len: int,
    seed: str,
    device: torch.device,
    num_points: int,
) -> Tuple[float, float, Dict]:
    """
    对 DataLoader 逐批：先 exact 得到 radii_ref，再 seg_filter 得到 idx_seg；
    与 exact 的 idx 计算 IOU，并统计每步过滤点数及相对 vanilla 全扫描的节省。
    返回 mean_iou、std_iou 以及汇总统计（含 per_step 过滤与节省分析）。
    """
    all_ious: List[float] = []
    all_candidate_ratio: List[torch.Tensor] = []
    all_used_fallback: List[torch.Tensor] = []

    for pts, _ in tqdm(loader, desc="IOU 评估", ncols=100):
        pts = pts.to(device, dtype=torch.float32)  # (B, N, 3)
        B = pts.size(0)

        # 1) 先跑 exact，得到 radii_ref 和 vanilla 索引（固定初始点）
        idx_exact, radii_ref, _, _ = fps_adaptive_argmax(
            pts, K, seed=seed, mode="exact"
        )

        # 2) 用 radii_ref 跑 seg_filter，模拟 segment 阈值过滤
        idx_seg, _, _, stats = fps_adaptive_argmax(
            pts, K, seed=seed, mode="seg_filter",
            radii_ref=radii_ref, segment_len=segment_len,
        )

        # 3) 计算本批 IOU（每个样本的 exact 与 seg_filter 索引集合的交并比）
        iou_per_sample, _, _ = compute_iou_batch(idx_exact, idx_seg)
        all_ious.extend(iou_per_sample.cpu().numpy().tolist())
        all_candidate_ratio.append(stats["candidate_ratio"])
        all_used_fallback.append(stats["used_fallback"])

    mean_iou = float(np.mean(all_ious))
    std_iou = float(np.std(all_ious))

    # 汇总统计
    candidate_ratio_cat = torch.cat([t.flatten() for t in all_candidate_ratio])
    used_fallback_cat = torch.cat([t.flatten() for t in all_used_fallback])

    # ---------- 每步过滤分析：segment_filter 每次迭代能过滤掉多少点 ----------
    # 形状: 每个 (B,K)，stack 成 (total_samples, K)，再对 step 维度做统计
    cr_stack = torch.cat(all_candidate_ratio, dim=0)  # (total_samples, K)
    per_step_candidate_ratio_mean = cr_stack.mean(dim=0).cpu().numpy()  # (K,)
    per_step_candidate_ratio_std = cr_stack.std(dim=0).cpu().numpy()  # (K,)
    # 每步平均保留的候选点数、平均被过滤掉的点数
    per_step_candidates_mean = (num_points * per_step_candidate_ratio_mean).tolist()
    per_step_filtered_out_mean = (num_points * (1.0 - per_step_candidate_ratio_mean)).tolist()
    per_step_candidate_ratio_mean_list = per_step_candidate_ratio_mean.tolist()
    per_step_candidate_ratio_std_list = per_step_candidate_ratio_std.tolist()

    # ---------- 相对 vanilla 的节省（已选点不参与后续扫描） ----------
    # Vanilla：第 t 步时已选 t 个点，只需在剩余 (N-t) 个点上做 argmax，故总扫描量 = sum_{t=1}^{K}(N-t) = K*N - K*(K+1)/2
    # Seg_filter：第 t 步在剩余 (N-t) 点中只有 candidate_ratio_t 比例通过 tau，故每步扫描量 ~ candidate_ratio_t * (N-t)，总扫描量按 per_step 求和
    overall_candidate_ratio_mean = candidate_ratio_cat.mean().item()
    overall_candidate_ratio_std = candidate_ratio_cat.std().item()
    search_space_reduction_ratio = 1.0 - overall_candidate_ratio_mean  # 每步平均“少看”的比例（相对当步剩余点数）
    # Vanilla 总扫描量：排除已选点后每步 (N-1)+(N-2)+...+(N-K) = K*N - K*(K+1)/2
    total_scan_vanilla = K * num_points - K * (K + 1) // 2
    # Seg_filter 总扫描量：第 t 步（1-based）剩余点数 = N-t，对应 per_step 下标 t-1；对 t=1..K 求和
    remaining_per_step = np.maximum(0, num_points - 1 - np.arange(K))  # (K,) 下标 i 对应第 i+1 次选择，剩余 N-1-i
    total_scan_seg_filter = float(np.sum(per_step_candidate_ratio_mean * remaining_per_step))
    scan_saved_ratio = 1.0 - (total_scan_seg_filter / total_scan_vanilla) if total_scan_vanilla > 0 else 0.0

    # 按 segment 聚合（便于看每段的过滤效果）
    num_segments = (K + segment_len - 1) // segment_len
    per_segment_candidate_ratio_mean: List[float] = []
    per_segment_filtered_out_mean: List[float] = []
    for seg_j in range(num_segments):
        start_t = seg_j * segment_len
        end_t = min((seg_j + 1) * segment_len, K)
        seg_slice = per_step_candidate_ratio_mean[start_t:end_t]
        seg_mean = float(np.mean(seg_slice))
        per_segment_candidate_ratio_mean.append(seg_mean)
        per_segment_filtered_out_mean.append(float(num_points * (1.0 - seg_mean)))

    summary = {
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "num_samples": len(all_ious),
        "candidate_ratio_mean": overall_candidate_ratio_mean,
        "candidate_ratio_std": overall_candidate_ratio_std,
        "used_fallback_any_ratio": used_fallback_cat.float().mean().item(),
        # 每步过滤与节省
        "num_points": num_points,
        "per_step_candidate_ratio_mean": per_step_candidate_ratio_mean_list,
        "per_step_candidate_ratio_std": per_step_candidate_ratio_std_list,
        "per_step_candidates_mean": per_step_candidates_mean,
        "per_step_filtered_out_mean": per_step_filtered_out_mean,
        "search_space_reduction_ratio": search_space_reduction_ratio,
        "scan_saved_ratio": scan_saved_ratio,
        "total_scan_vanilla": total_scan_vanilla,
        "total_scan_seg_filter": float(total_scan_seg_filter),
        "per_segment_candidate_ratio_mean": per_segment_candidate_ratio_mean,
        "per_segment_filtered_out_mean": per_segment_filtered_out_mean,
    }
    return mean_iou, std_iou, summary


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 固定初始点
    seed = args.seed if args.seed else "first"

    ds = ModelNet40Wrapper(partition=args.partition, num_points=args.num_points, root=args.root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )

    mean_iou, std_iou, summary = run_evaluation(
        loader, K=args.K, segment_len=args.segment_len, seed=seed, device=device,
        num_points=args.num_points,
    )

    # 单样本可视化：候选点数与阈值随迭代变化（一图，英文）
    if args.vis_one_sample:
        try:
            first_batch = next(iter(loader))
            pts_one = first_batch[0][:1].to(device)  # (1, N, 3) 取第一个样本
            vis_path = os.path.join(args.save_dir, "seg_filter_one_sample.png")
            plot_one_sample_seg_filter(
                pts_one, K=args.K, segment_len=args.segment_len, seed=seed,
                device=device, save_path=vis_path,
            )
            print(f"  [Vis] 单样本图已保存: {vis_path}")
        except ImportError as e:
            print(f"  [Vis] 跳过绘图（需安装 matplotlib）: {e}")

    # 写入结果（含每步过滤与节省分析）
    results = {
        "partition": args.partition,
        "num_points": args.num_points,
        "K": args.K,
        "segment_len": args.segment_len,
        "seed": seed,
        "mean_iou": mean_iou,
        "std_iou": std_iou,
        "num_samples": summary["num_samples"],
        "candidate_ratio_mean": summary["candidate_ratio_mean"],
        "candidate_ratio_std": summary["candidate_ratio_std"],
        "used_fallback_any_ratio": summary["used_fallback_any_ratio"],
        # segment_filter 过滤与节省
        "search_space_reduction_ratio": summary["search_space_reduction_ratio"],
        "scan_saved_ratio": summary["scan_saved_ratio"],
        "total_scan_vanilla": summary["total_scan_vanilla"],
        "total_scan_seg_filter": summary["total_scan_seg_filter"],
        "per_step_candidate_ratio_mean": summary["per_step_candidate_ratio_mean"],
        "per_step_candidate_ratio_std": summary["per_step_candidate_ratio_std"],
        "per_step_candidates_mean": summary["per_step_candidates_mean"],
        "per_step_filtered_out_mean": summary["per_step_filtered_out_mean"],
        "per_segment_candidate_ratio_mean": summary["per_segment_candidate_ratio_mean"],
        "per_segment_filtered_out_mean": summary["per_segment_filtered_out_mean"],
    }
    results_path = os.path.join(args.save_dir, "iou_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 控制台输出：IOU + 每步过滤与相对 vanilla 的节省
    print("=" * 60)
    print("seg_filter FPS 与 vanilla (exact) FPS IOU 评估结果 (ModelNet40)")
    print("=" * 60)
    print(f"  分区: {args.partition}  点数: N={args.num_points}  K={args.K}  segment_len: {args.segment_len}")
    print(f"  固定初始点: seed={seed}")
    print(f"  mean_iou: {mean_iou:.6f}  std_iou: {std_iou:.6f}")
    print(f"  样本数: {summary['num_samples']}")
    print("-" * 60)
    print("segment_filter 每次迭代过滤与相对 vanilla 全扫描的节省:")
    print(f"  每步平均候选比例 candidate_ratio: {summary['candidate_ratio_mean']:.4f} ± {summary['candidate_ratio_std']:.4f}")
    print(f"  每步平均保留候选点数: {args.num_points * summary['candidate_ratio_mean']:.1f} / {args.num_points}")
    print(f"  每步平均过滤掉点数: {args.num_points * (1 - summary['candidate_ratio_mean']):.1f} / {args.num_points}")
    print(f"  搜索空间节省比例 (相对 vanilla 全扫描): {summary['search_space_reduction_ratio']:.4f} ({summary['search_space_reduction_ratio']*100:.2f}%)")
    print(f"  总扫描量 vanilla（已选点不重复扫描）: {summary['total_scan_vanilla']}  seg_filter 等效: {summary['total_scan_seg_filter']:.0f}  节省: {summary['scan_saved_ratio']*100:.2f}%")
    print("-" * 60)
    print("按 segment 汇总 (每段内步取平均):")
    num_seg = len(summary["per_segment_candidate_ratio_mean"])
    for j in range(num_seg):
        cr = summary["per_segment_candidate_ratio_mean"][j]
        fo = summary["per_segment_filtered_out_mean"][j]
        start_t = j * args.segment_len
        end_t = min((j + 1) * args.segment_len, args.K)
        print(f"  segment {j:2d} (步 {start_t:3d}~{end_t-1:3d}): 候选比例={cr:.4f}  平均过滤点数={fo:.1f}")
    print("=" * 60)
    print(f"  结果已保存: {results_path}")


if __name__ == "__main__":
    main()