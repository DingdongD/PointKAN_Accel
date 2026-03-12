"""
混合 FPS 与 Vanilla FPS 的 IoU 评估（ModelNet40）

算法（见图）：FlashFPS 风格
- 前 (1-p)*M 个点：vanilla FPS
- 后 p*M 个点：从 P \\ Pfps 中随机采样
- 输出 P1 = Pfps ∪ Prand，与 vanilla FPS 的采样集合做 IoU 比较。
"""
import os
import sys
import argparse
import random
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from spec_kan_fps import (
    fps_with_knn_adaptive,
    ModelNet40Wrapper,
    collate_points,
)


# ---------------------------------------------------------------------------
# 1) Vanilla FPS：只返回采样索引 [B, K]
# ---------------------------------------------------------------------------

@torch.no_grad()
def vanilla_fps_indices(
    X: torch.Tensor,
    K: int,
    seed: str = "first",
) -> torch.Tensor:
    """
    X: [B, N, D]，返回 FPS 采样索引 idx [B, K]。
    """
    idx, *_ = fps_with_knn_adaptive(
        X, K, k_neighbors=1, init_threshold=0.1,
        max_iter=1, tol=0.05, seed=seed,
    )
    return idx


# ---------------------------------------------------------------------------
# 2) 混合 FPS（图示算法）：前 (1-p)*K 为 FPS，后 p*K 为随机（P \\ Pfps）
# ---------------------------------------------------------------------------

@torch.no_grad()
def hybrid_fps_indices(
    X: torch.Tensor,
    K: int,
    p: float,
    seed: str = "first",
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    前 (1-p)*K 点用 vanilla FPS，后 p*K 点从剩余点中随机采样。
    p ∈ [0, 1)，p=0 即全 FPS，p 越大随机比例越高。
    X: [B, N, D]，返回 idx [B, K]。
    """
    assert 0 <= p < 1, "p 应在 [0, 1)"
    B, N, _ = X.shape
    device = X.device
    K = min(K, N)

    k_fps = max(0, int(round((1 - p) * K)))
    k_rand = K - k_fps

    if k_fps == 0:
        # 全随机
        idx = torch.zeros(B, K, dtype=torch.long, device=device)
        for b in range(B):
            idx[b] = torch.randperm(N, device=device, generator=rng)[:K]
        return idx

    if k_rand == 0:
        return vanilla_fps_indices(X, K, seed=seed)

    # Pfps: [B, k_fps]
    idx_fps = vanilla_fps_indices(X, k_fps, seed=seed)
    idx_rand = torch.zeros(B, k_rand, dtype=torch.long, device=device)

    for b in range(B):
        selected = torch.zeros(N, dtype=torch.bool, device=device)
        selected[idx_fps[b]] = True
        remaining = torch.where(~selected)[0]
        perm = torch.randperm(remaining.size(0), device=device, generator=rng)[:k_rand]
        idx_rand[b] = remaining[perm]

    idx = torch.cat([idx_fps, idx_rand], dim=1)
    return idx


# ---------------------------------------------------------------------------
# 2b) 混合 FPS（带候选剪枝）：P'=UniformSample(P,(1-p)N)，再在 P' 上做 FPS
# ---------------------------------------------------------------------------

@torch.no_grad()
def hybrid_fps_indices_with_pruning(
    X: torch.Tensor,
    K: int,
    p: float,
    seed: str = "first",
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    带候选剪枝的 FlashFPS 风格：
    1) P' ← UniformSample(P, (1-p)·N)
    2) k ← (1-p)·K，Pfps ← FPS(P', k)，映射回原空间
    3) Prand ← RandomSample(P \\ Pfps, p·K)
    4) P1 = Pfps ∪ Prand
    p ∈ [0, 1)，X: [B, N, D]，返回 idx [B, K]。
    """
    assert 0 <= p < 1, "p 应在 [0, 1)"
    B, N, D = X.shape
    device = X.device
    K = min(K, N)

    # 候选剪枝：P' 大小 (1-p)*N
    N_prime = max(1, int(round((1 - p) * N)))
    k_fps = max(0, int(round((1 - p) * K)))
    k_fps = min(k_fps, N_prime)  # FPS 不能超过 P' 大小
    k_rand = K - k_fps

    if k_fps == 0:
        idx = torch.zeros(B, K, dtype=torch.long, device=device)
        for b in range(B):
            idx[b] = torch.randperm(N, device=device, generator=rng)[:K]
        return idx

    # 1) 对每个 batch 均匀采样得到 P' 的索引 [B, N_prime]
    uniform_indices = torch.zeros(B, N_prime, dtype=torch.long, device=device)
    for b in range(B):
        uniform_indices[b] = torch.randperm(N, device=device, generator=rng)[:N_prime]

    # 2) X' = X[:, uniform_indices, :] -> [B, N_prime, D]
    X_prime = torch.gather(X, 1, uniform_indices.unsqueeze(-1).expand(B, N_prime, D))

    # 3) Pfps 在 P' 上的索引 [B, k_fps]，再映射回原 P 的索引
    idx_in_P_prime = vanilla_fps_indices(X_prime, k_fps, seed=seed)
    idx_fps = torch.gather(uniform_indices, 1, idx_in_P_prime)

    if k_rand == 0:
        return idx_fps

    # 4) Prand: 从 P \ Pfps 中随机采 k_rand 个
    idx_rand = torch.zeros(B, k_rand, dtype=torch.long, device=device)
    for b in range(B):
        selected = torch.zeros(N, dtype=torch.bool, device=device)
        selected[idx_fps[b]] = True
        remaining = torch.where(~selected)[0]
        perm = torch.randperm(remaining.size(0), device=device, generator=rng)[:k_rand]
        idx_rand[b] = remaining[perm]

    idx = torch.cat([idx_fps, idx_rand], dim=1)
    return idx


# ---------------------------------------------------------------------------
# 3) 采样集合 IoU：两组采样索引（集合）的交并比
# ---------------------------------------------------------------------------

def sampling_set_iou(
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
) -> torch.Tensor:
    """
    idx_a, idx_b: [B, K]，每行为一组采样的点索引。
    返回每个样本的 IoU [B]，IoU = |A ∩ B| / |A ∪ B|。
    """
    B, K = idx_a.shape
    device = idx_a.device
    ious = []
    for b in range(B):
        set_a = set(idx_a[b].cpu().tolist())
        set_b = set(idx_b[b].cpu().tolist())
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        ious.append(inter / union if union > 0 else 1.0)
    return torch.tensor(ious, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4) ModelNet40 上评估：Vanilla vs Hybrid 的 IoU
# ---------------------------------------------------------------------------

def evaluate_iou(
    partition: str = "test",
    root: str = "./data",
    num_points: int = 1024,
    K: int = 512,
    p_values: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5),
    batch_size: int = 32,
    max_samples: int = 0,
    seed: int = 42,
    device: str = "cuda",
    use_pruning: bool = False,
):
    """
    在 ModelNet40 上对多组 p 做 Hybrid vs Vanilla FPS 的 IoU 评估。
    use_pruning=True 时使用带候选剪枝的混合 FPS（P'=UniformSample(P,(1-p)N) 再 FPS(P',k)）。
    max_samples=0 表示用全部数据。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = torch.Generator(device=device if device == "cuda" else "cpu").manual_seed(seed)

    dataset = ModelNet40Wrapper(partition=partition, num_points=num_points, root=root)
    total = len(dataset)
    if max_samples > 0:
        total = min(total, max_samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )

    # 每个 p 存该 p 下所有 batch 的 IoU
    all_ious = {p: [] for p in p_values}
    n_done = 0

    for pts, _ in tqdm(loader, desc="IoU eval"):
        if max_samples > 0 and n_done >= max_samples:
            break
        pts = pts.to(device, dtype=torch.float32)
        B = pts.size(0)
        if max_samples > 0 and n_done + B > max_samples:
            pts = pts[: max_samples - n_done]
            B = pts.size(0)
        n_done += B

        idx_vanilla = vanilla_fps_indices(pts, K, seed="first")

        hybrid_fn = hybrid_fps_indices_with_pruning if use_pruning else hybrid_fps_indices
        for p in p_values:
            idx_hybrid = hybrid_fn(pts, K, p=p, seed="first", rng=rng)
            iou = sampling_set_iou(idx_vanilla, idx_hybrid)
            all_ious[p].append(iou.cpu().numpy())

    # 汇总
    results = {}
    for p in p_values:
        arr = np.concatenate(all_ious[p], axis=0)
        results[p] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": len(arr)}
    return results, total


def main():
    parser = argparse.ArgumentParser(description="Hybrid FPS vs Vanilla FPS IoU on ModelNet40")
    parser.add_argument("--root", type=str, default="./data", help="ModelNet40 数据根目录")
    parser.add_argument("--partition", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--num_points", type=int, default=1024, help="每样本点数 N")
    parser.add_argument("--K", type=int, default=512, help="FPS 采样数 K")
    parser.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5],
                        help="混合比例 p（随机占比），可多个")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0, help="0=全部")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pruning", action="store_true", help="使用候选剪枝：P'=UniformSample(P,(1-p)N)，再在 P' 上做 FPS")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    p_values = tuple(args.p)

    results, total = evaluate_iou(
        partition=args.partition,
        root=args.root,
        num_points=args.num_points,
        K=args.K,
        p_values=p_values,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        device=device,
        use_pruning=args.pruning,
    )

    mode = "带候选剪枝" if args.pruning else "无候选剪枝"
    print("\n========== Hybrid FPS vs Vanilla FPS 采样 IoU (ModelNet40) ==========")
    print(f"mode={mode}  partition={args.partition}  num_points={args.num_points}  K={args.K}  n_samples={total}")
    print("-" * 60)
    print(f"{'p (random ratio)':<20} {'IoU mean':<12} {'IoU std':<10} {'n':<8}")
    print("-" * 60)
    for p in p_values:
        r = results[p]
        print(f"{p:<20.2f} {r['mean']:<12.6f} {r['std']:<10.6f} {r['n']:<8}")
    print("-" * 60)
    if args.pruning:
        print("带候选剪枝：P'=UniformSample(P,(1-p)N)，FPS 在 P' 上做，故与 Vanilla FPS(全 P) 的 IoU 会偏低。")
    else:
        print("p=0 为全 FPS（Hybrid 与 Vanilla 一致，IoU=1.0）。p 越大随机越多，与 Vanilla 重叠越少，IoU 越低。")
    print("=" * 60)


if __name__ == "__main__":
    main()
