import os
import sys
import math
import json
import random
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List

import numpy as np

# 与 inference_and_kan_analysis.py 一致：当前任务目录与项目根目录加入 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
for d in (SCRIPT_DIR, ROOT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =========================================================
# 0. 你的 FPS + adaptive KNN 代码（基本原样保留）
# =========================================================

@torch.no_grad()
def fps_with_knn_adaptive(
    X: torch.Tensor,
    K: int,
    k_neighbors: int = 16,
    init_threshold: float = 0.1,
    max_iter: int = 10,
    tol: float = 0.05,
    seed: str = "first",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FPS + 动态阈值KNN
    返回：
        - idx: FPS采样索引 (B,K)
        - radii: FPS半径曲线 (B,K) 这里 radii[t] = max(min_dist) after selecting t-th point
        - knn_idx: 动态阈值下的KNN索引 (B,K,k_neighbors)
        - distances_matrix: 距离矩阵 (B,K,N)
        - knn_radii: 第k近邻半径 (B,K)
        - thresholds: 最终阈值 (B,K)
        - iters_used: 每个采样点的迭代次数 (B,K)
    """
    assert X.ndim == 3
    B, N, D = X.shape
    device, dtype = X.device, X.dtype
    K = min(K, N)

    x2 = (X * X).sum(dim=-1)  # (B, N)
    idx = torch.zeros((B, K), dtype=torch.long, device=device)
    radii = torch.zeros((B, K), dtype=dtype, device=device)
    min_dist = torch.full((B, N), float("inf"), dtype=dtype, device=device)
    distances_matrix = torch.zeros((B, K, N), dtype=dtype, device=device)
    thresholds = torch.zeros((B, K), dtype=dtype, device=device)
    iters_used = torch.zeros((B, K), dtype=torch.int32, device=device)

    # 初始化种子点
    if seed == "first":
        first = torch.zeros(B, dtype=torch.long, device=device)
    elif seed == "random":
        first = torch.randint(0, N, (B,), device=device)
    elif seed == "max_norm":
        first = torch.argmax(x2, dim=1)
    else:
        raise ValueError(f"Unknown seed: {seed}")

    idx[:, 0] = first
    C = X[torch.arange(B), first]
    c2 = (C * C).sum(dim=-1, keepdim=True)
    dot = (X @ C.unsqueeze(-1)).squeeze(-1)
    d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()
    distances_matrix[:, 0, :] = d2
    min_dist.copy_(d2)
    radii[:, 0], _ = torch.max(min_dist, dim=1)
    min_dist[torch.arange(B), first] = 0.0

    # FPS主循环
    for t in range(1, K):
        farthest = torch.argmax(min_dist, dim=1)
        idx[:, t] = farthest

        C = X[torch.arange(B), farthest]
        c2 = (C * C).sum(dim=-1, keepdim=True)
        dot = (X @ C.unsqueeze(-1)).squeeze(-1)
        d2 = (x2 + c2 - 2.0 * dot).clamp_min_(0.0).sqrt()

        distances_matrix[:, t, :] = d2
        torch.minimum(min_dist, d2, out=min_dist)
        radii[:, t], _ = torch.max(min_dist, dim=1)
        min_dist[torch.arange(B), farthest] = 0.0

    # ===== 动态阈值KNN选择 =====
    knn_idx = torch.zeros((B, K, k_neighbors), dtype=torch.long, device=device)
    knn_radii = torch.zeros((B, K), dtype=dtype, device=device)

    for t in range(K):
        d = distances_matrix[:, t, :]  # (B, N)
        threshold = torch.full((B,), init_threshold, dtype=dtype, device=device)

        for i in range(max_iter):
            mask = d < threshold.unsqueeze(-1)
            count = mask.sum(dim=1).float()

            done = (count >= k_neighbors * (1 - tol)) & (count <= k_neighbors * (1 + tol))
            if done.all():
                iters_used[:, t] = i + 1
                break

            too_many = (count > (k_neighbors * (1 + tol))).float()
            too_few  = (count < (k_neighbors * (1 - tol))).float()
            adjust = torch.where(too_many.bool(), -0.1, torch.where(too_few.bool(), 0.1, 0.0))
            threshold = threshold * (1 + adjust)

            if i == max_iter - 1:
                iters_used[:, t] = max_iter

        thresholds[:, t] = threshold

        mask = d < threshold.unsqueeze(-1)
        masked_d = d.clone()
        masked_d[~mask] = float("inf")
        knn_vals, knn_idx_t = torch.topk(masked_d, k=min(k_neighbors, N), dim=1, largest=False)
        knn_idx[:, t, :] = knn_idx_t
        knn_radii[:, t] = knn_vals[:, -1]

    return idx, radii, knn_idx, distances_matrix, knn_radii, thresholds, iters_used


@torch.no_grad()
def compute_fps_radii_batch(X: torch.Tensor, K: int, seed: str = "first") -> torch.Tensor:
    """
    X: [B, N, 3], 返回 radii [B, K]
    """
    _, radii, *_ = fps_with_knn_adaptive(
        X, K, k_neighbors=16, init_threshold=0.1,
        max_iter=10, tol=0.05, seed=seed
    )
    return radii


# =========================================================
# 1) 数据集适配（优先用你的 ModelNet40；否则尝试 PyG）
# =========================================================

def try_import_user_modelnet40():
    """
    优先使用项目 data 包中的 ModelNet40（与 inference_and_kan_analysis.py / main.py 一致）；
    若无则尝试用户自定义 modelnet40 模块。
    """
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
    """
    统一输出:
      - points: torch.FloatTensor [N,3]
      - label: int
    """
    def __init__(self, partition: str, num_points: int, root: str = "./data"):
        super().__init__()
        self.partition = partition
        self.num_points = num_points

        UserModelNet40 = try_import_user_modelnet40()
        if UserModelNet40 is not None:
            self.backend = "user"
            self.ds = UserModelNet40(partition=partition, num_points=num_points)
            self.length = len(self.ds)
            return

        # fallback: torch_geometric
        try:
            from torch_geometric.datasets import ModelNet
        except Exception as e:
            raise RuntimeError(
                "找不到 ModelNet40 实现（已尝试 data.ModelNet40、modelnet40.ModelNet40），也未安装 torch_geometric.datasets.ModelNet。\n"
                "解决方式：在 classification_ModelNet40 目录下应存在 data.py 并提供 ModelNet40(partition=..., num_points=...)；\n"
                "或安装 PyG 并使用 torch_geometric.datasets.ModelNet。\n"
                f"原始错误: {repr(e)}"
            )

        self.backend = "pyg"
        split = "train" if partition == "train" else "test"
        self.ds = ModelNet(root=root, name="40", train=(split == "train"))
        self.length = len(self.ds)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.backend == "user":
            pts, label = self.ds[idx]  # 你的实现一般就是 (points,label)
            if isinstance(pts, np.ndarray):
                pts = torch.from_numpy(pts)
            pts = pts[:, :3].float()
            return pts, int(label)

        # pyg backend
        data = self.ds[idx]
        pts = data.pos.float()  # [N,3]
        y = int(data.y.item()) if hasattr(data, "y") else 0

        # 采样/裁剪到 num_points
        N = pts.shape[0]
        if N >= self.num_points:
            choice = torch.randperm(N)[: self.num_points]
            pts = pts[choice]
        else:
            # pad by repeat
            repeat = self.num_points - N
            pad_idx = torch.randint(0, N, (repeat,))
            pts = torch.cat([pts, pts[pad_idx]], dim=0)

        return pts, y


def collate_points(batch):
    pts = torch.stack([b[0] for b in batch], dim=0)  # [B,N,3]
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return pts, y


# =========================================================
# 2) radii 曲线 -> (input32, target64) 的构造
# =========================================================

def lin_resample_1d(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """
    x: [B, L] or [L]
    返回同 batch 的线性插值重采样到 out_len
    """
    if x.ndim == 1:
        x = x[None, :]
    B, L = x.shape
    if L == out_len:
        return x

    # grid in [0, L-1]
    src_pos = torch.linspace(0, L - 1, steps=out_len, device=x.device, dtype=x.dtype)  # [out_len]
    left = torch.floor(src_pos).long().clamp(0, L - 1)
    right = (left + 1).clamp(0, L - 1)
    w = (src_pos - left.to(x.dtype))[None, :]  # [1,out_len]

    x_left = x[:, left]   # [B,out_len]
    x_right = x[:, right] # [B,out_len]
    y = (1 - w) * x_left + w * x_right
    return y


def build_prefix_suffix_io(
    radii: torch.Tensor,
    prefix_ratio: float,
    in_dim: int = 32,
    out_dim: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    radii: [B, K]
    取前 p% -> resample 到 in_dim
    取后 (1-p)% -> resample 到 out_dim
    """
    assert radii.ndim == 2
    B, K = radii.shape
    prefix_len = max(2, int(round(K * prefix_ratio)))
    prefix_len = min(prefix_len, K - 2)

    prefix = radii[:, :prefix_len]         # [B, prefix_len]
    suffix = radii[:, prefix_len:]         # [B, K-prefix_len]

    x = lin_resample_1d(prefix, in_dim)    # [B, in_dim]
    y = lin_resample_1d(suffix, out_dim)   # [B, out_dim]
    return x, y


# =========================================================
# 3) 拟合网络：MLP 或 Efficient_KAN (32->64)
# =========================================================

class RadiiMLP(nn.Module):
    """3 层 MLP：in_dim -> hidden -> hidden -> out_dim"""
    def __init__(self, in_dim=32, hidden=256, out_dim=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_model(cfg, device) -> nn.Module:
    """
    根据 cfg.model_type 构建模型：mlp -> RadiiMLP；kan -> Efficient_KAN (pointkanElite)。
    输入维度 in_dim，输出维度 out_dim，隐藏维度 hidden。
    """
    if getattr(cfg, "model_type", "mlp") == "kan":
        from models.pointkanElite import Efficient_KAN
        model = Efficient_KAN(
            in_features=cfg.in_dim,
            hidden_features=cfg.hidden,
            out_features=cfg.out_dim,
            drop=cfg.dropout,
            act_init="gelu",
        )
    else:
        model = RadiiMLP(
            in_dim=cfg.in_dim,
            hidden=cfg.hidden,
            out_dim=cfg.out_dim,
            dropout=cfg.dropout,
        )
    return model.to(device)


# =========================================================
# 4) 端到端：提取 radii + 训练 + 测试
# =========================================================

@dataclass
class Config:
    root: str = "./data"
    num_points: int = 1024
    fps_steps: int = 512

    prefix_ratio: float = 0.10
    in_dim: int = 32
    out_dim: int = 64

    batch_size_fps: int = 16   # FPS 本身 O(B*K*N)，别太大
    batch_size_train: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 256
    dropout: float = 0.0

    seed: int = 42
    device: str = "cuda"
    save_dir: str = "./runs/fps_radii_mlp"
    model_type: str = "mlp"   # "mlp" | "kan"，kan 使用 Efficient_KAN (pointkanElite)
    radii_dir: str = ""      # 为空则用 save_dir；多组 prefix_ratio 可共用同一 radii 缓存目录


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_radii_for_split(cfg: Config, partition: str) -> np.ndarray:
    """
    返回 all_radii: [N_samples, K]
    """
    ds = ModelNet40Wrapper(partition=partition, num_points=cfg.num_points, root=cfg.root)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size_fps,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_points,
    )

    all_radii = []
    pbar = tqdm(loader, desc=f"Extract radii [{partition}]", ncols=110)
    for pts, _ in pbar:
        pts = pts.to(cfg.device, dtype=torch.float32)   # [B,N,3]
        radii = compute_fps_radii_batch(pts, cfg.fps_steps, seed="first")  # [B,K]
        all_radii.append(radii.cpu().numpy())

    all_radii = np.concatenate(all_radii, axis=0)  # [N,K]
    return all_radii


class RadiiIOTensorDataset(Dataset):
    """
    存 radii 曲线数组，按 prefix/suffix 规则动态产生 (x,y)
    """
    def __init__(self, radii_np: np.ndarray, prefix_ratio: float, in_dim: int, out_dim: int, device: str = "cpu"):
        super().__init__()
        self.radii = torch.from_numpy(radii_np).float()  # [N,K]
        self.prefix_ratio = prefix_ratio
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

    def __len__(self):
        return self.radii.shape[0]

    def __getitem__(self, i):
        r = self.radii[i]  # [K]
        x, y = build_prefix_suffix_io(
            r[None, :], self.prefix_ratio, self.in_dim, self.out_dim
        )
        return x.squeeze(0), y.squeeze(0)


def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


@torch.no_grad()
def eval_mse(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(1, n)


def _radii_paths(save_dir: str, fps_steps: int) -> Tuple[str, str]:
    """训练集/测试集 radii 缓存路径，名称含 radii 长度 K（如 512/256）。"""
    return (
        os.path.join(save_dir, f"radii_train_K{fps_steps}.npy"),
        os.path.join(save_dir, f"radii_test_K{fps_steps}.npy"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="FPS radii MLP 训练（支持断点续训）")
    parser.add_argument("--save_dir", type=str, default="./runs/fps_radii_mlp", help="保存目录（含 radii 缓存、checkpoint、log）")
    parser.add_argument("--resume", type=str, default="", help="断点续训：指向 last.pt 或包含 last.pt 的目录")
    parser.add_argument("--fps_steps", type=int, default=512, help="FPS 采样点数 K，即 radii 长度（用于缓存文件名）")
    parser.add_argument("--prefix_ratio", type=float, default=0.10, help="前缀比例，用于 prefix/suffix 划分与实验对比")
    parser.add_argument("--num_points", type=int, default=1024, help="点云点数")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size_train", type=int, default=256)
    parser.add_argument("--batch_size_fps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_type", type=str, default="mlp", choices=("mlp", "kan"),
                        help="mlp=RadiiMLP, kan=Efficient_KAN (pointkanElite)")
    parser.add_argument("--radii_dir", type=str, default="",
                        help="radii 缓存目录；为空则用 save_dir。多组 prefix_ratio 可共用一个 radii_dir，只构建一次")
    return parser.parse_args()


def setup_logger(save_dir: str) -> logging.Logger:
    """配置 logger：同时输出到控制台和 save_dir/train.log。"""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    logger = logging.getLogger("fps_radii")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    args = parse_args()
    cfg = Config()
    cfg.save_dir = args.save_dir
    cfg.fps_steps = args.fps_steps
    cfg.prefix_ratio = args.prefix_ratio
    cfg.num_points = args.num_points
    cfg.epochs = args.epochs
    cfg.batch_size_train = args.batch_size_train
    cfg.batch_size_fps = args.batch_size_fps
    cfg.lr = args.lr
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.model_type = args.model_type
    radii_dir = (args.radii_dir or "").strip() or cfg.save_dir
    cfg.radii_dir = radii_dir if radii_dir != cfg.save_dir else ""

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(radii_dir, exist_ok=True)
    logger = setup_logger(cfg.save_dir)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cfg.device = str(device)

    logger.info("save_dir=%s radii_dir=%s fps_steps=%s prefix_ratio=%s model_type=%s",
                cfg.save_dir, radii_dir, cfg.fps_steps, cfg.prefix_ratio, cfg.model_type)

    train_radii_path, test_radii_path = _radii_paths(radii_dir, cfg.fps_steps)

    # ---------- 1) 训练/测试集 radii：有则从 radii_dir 加载，无则构建一次并保存到 radii_dir ----------
    if os.path.exists(train_radii_path):
        train_radii = np.load(train_radii_path)
        logger.info("[Load] %s  shape=%s", train_radii_path, train_radii.shape)
    else:
        train_radii = extract_radii_for_split(cfg, "train")
        np.save(train_radii_path, train_radii)
        logger.info("[Save] %s  shape=%s", train_radii_path, train_radii.shape)

    if os.path.exists(test_radii_path):
        test_radii = np.load(test_radii_path)
        logger.info("[Load] %s  shape=%s", test_radii_path, test_radii.shape)
    else:
        test_radii = extract_radii_for_split(cfg, "test")
        np.save(test_radii_path, test_radii)
        logger.info("[Save] %s  shape=%s", test_radii_path, test_radii.shape)

    ds_train = RadiiIOTensorDataset(train_radii, cfg.prefix_ratio, cfg.in_dim, cfg.out_dim)
    ds_test  = RadiiIOTensorDataset(test_radii,  cfg.prefix_ratio, cfg.in_dim, cfg.out_dim)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size_train, shuffle=True, drop_last=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=cfg.batch_size_train, shuffle=False, drop_last=False, num_workers=0)

    model = build_model(cfg, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------- 模型结构写入 log ----------
    logger.info("========== Model structure ==========")
    for line in str(model).split("\n"):
        logger.info("%s", line)
    logger.info("=====================================")

    best_test_mse = float("inf")
    best_path = os.path.join(cfg.save_dir, "best.pt")
    last_path = os.path.join(cfg.save_dir, "last.pt")

    log = {
        "config": cfg.__dict__,
        "train_mse": [],
        "test_mse": [],
    }
    start_epoch = 1

    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "last.pt")
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"断点续训需要 last.pt，未找到: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_test_mse = ckpt.get("best_test_mse", float("inf"))
        if "log" in ckpt:
            log = ckpt["log"]
        logger.info("[Resume] from %s -> epoch %s, best_test_mse=%.6e", resume_path, start_epoch, best_test_mse)

    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, device)
        te = eval_mse(model, test_loader, device)

        log["train_mse"].append(tr)
        log["test_mse"].append(te)

        if te < best_test_mse:
            best_test_mse = te
            torch.save({
                "model": model.state_dict(),
                "cfg": cfg.__dict__,
                "best_test_mse": best_test_mse,
                "epoch": epoch,
            }, best_path)
            logger.info("  -> new best, saved to %s", best_path)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
            "best_test_mse": best_test_mse,
            "cfg": cfg.__dict__,
            "log": log,
        }, last_path)

        logger.info("Epoch %03d | train_mse=%.6e | test_mse=%.6e | best=%.6e", epoch, tr, te, best_test_mse)

    with open(os.path.join(cfg.save_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    # ---------- 单次运行结果（供 prefix_ratio 扫描汇总用） ----------
    results = {
        "prefix_ratio": cfg.prefix_ratio,
        "fps_steps": cfg.fps_steps,
        "best_test_mse": float(best_test_mse),
        "epochs": cfg.epochs,
    }
    results_path = os.path.join(cfg.save_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("[Done] best checkpoint: %s (best_test_mse=%.6e)", best_path, best_test_mse)
    logger.info("[Done] last checkpoint: %s", last_path)
    logger.info("[Done] log: %s", os.path.join(cfg.save_dir, "log.json"))
    logger.info("[Done] results: %s", results_path)


if __name__ == "__main__":
    main()