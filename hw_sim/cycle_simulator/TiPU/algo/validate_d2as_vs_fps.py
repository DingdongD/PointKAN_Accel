import argparse
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def square_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [..., 3]
    b: [..., 3]
    return: [...]
    """
    return ((a - b) ** 2).sum(dim=-1)


def pairwise_square_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [M, 3]
    y: [N, 3]
    return: [M, N]
    """
    # torch.cdist returns euclidean distance; square it
    return torch.cdist(x, y, p=2) ** 2


# ============================================================
# Metrics
# ============================================================

def chamfer_distance_sqr(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Symmetric Chamfer distance (squared) between two point sets.
    x: [Mx, 3]
    y: [My, 3]
    """
    dmat = pairwise_square_dist(x, y)
    cd = dmat.min(dim=1).values.mean() + dmat.min(dim=0).values.mean()
    return float(cd.item())


def directed_cover_stats(points: torch.Tensor, samples: torch.Tensor) -> Tuple[float, float]:
    """
    Measure how well samples cover original points.
    points:   [N, 3]
    samples:  [M, 3]

    return:
      max_cover (sqrt distance),
      avg_cover (sqrt distance)
    """
    dmat = torch.cdist(points, samples, p=2)  # [N, M]
    nearest = dmat.min(dim=1).values
    return float(nearest.max().item()), float(nearest.mean().item())


def index_overlap_ratio(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """
    idx_a: [Ma]
    idx_b: [Mb]
    overlap = |intersection| / |union|
    """
    set_a = set(idx_a.detach().cpu().tolist())
    set_b = set(idx_b.detach().cpu().tolist())
    union = len(set_a | set_b)
    inter = len(set_a & set_b)
    return float(inter / union) if union > 0 else 1.0


# ============================================================
# Result containers
# ============================================================

@dataclass
class SampleResult:
    indices: torch.Tensor          # [M]
    points: torch.Tensor           # [M, 3]
    dist_ops: int                  # estimated #distance evaluations
    runtime_ms: float              # wall clock time
    extra: Optional[Dict] = None


@dataclass
class CompareResult:
    n_input: int
    m_fps: int
    m_d2as: int
    overlap_ratio: float
    chamfer_sqr: float
    fps_max_cover: float
    fps_avg_cover: float
    d2as_max_cover: float
    d2as_avg_cover: float
    fps_dist_ops: int
    d2as_dist_ops: int
    fps_runtime_ms: float
    d2as_runtime_ms: float


# ============================================================
# Vanilla FPS
# ============================================================

@torch.no_grad()
def vanilla_fps_single(
    points: torch.Tensor,          # [N, 3]
    m: int,
    start_idx: int = 0,
) -> SampleResult:
    """
    Standard FPS with fixed sample count m.
    """
    assert points.dim() == 2 and points.size(1) == 3
    device = points.device
    n = points.size(0)
    assert 1 <= m <= n

    t0 = time.perf_counter()

    selected = torch.empty(m, dtype=torch.long, device=device)
    min_dist = torch.full((n,), float("inf"), device=device)

    cur = start_idx
    dist_ops = 0

    for i in range(m):
        selected[i] = cur

        cur_point = points[cur].unsqueeze(0)      # [1, 3]
        d = pairwise_square_dist(points, cur_point).squeeze(1)  # [N]
        dist_ops += n

        min_dist = torch.minimum(min_dist, d)
        min_dist[cur] = -1.0

        if i != m - 1:
            cur = int(torch.argmax(min_dist).item())

    t1 = time.perf_counter()

    sampled = points[selected]
    return SampleResult(
        indices=selected,
        points=sampled,
        dist_ops=dist_ops,
        runtime_ms=(t1 - t0) * 1000.0,
        extra={"start_idx": start_idx},
    )


# ============================================================
# Tile partition
# ============================================================

def build_xy_tiles(
    points: torch.Tensor,          # [N, 3]
    tile_size: float,
    xy_min: Optional[torch.Tensor] = None,
) -> Tuple[Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
    """
    Partition points by XY plane into square tiles.

    return:
      tiles: {(tx, ty): idx_tensor}
      xy_origin: [2]
    """
    xy = points[:, :2]
    if xy_min is None:
        xy_min = xy.min(dim=0).values

    tile_coord = torch.floor((xy - xy_min) / tile_size).long()  # [N, 2]

    tiles: Dict[Tuple[int, int], List[int]] = {}
    for i in range(points.size(0)):
        key = (int(tile_coord[i, 0].item()), int(tile_coord[i, 1].item()))
        if key not in tiles:
            tiles[key] = []
        tiles[key].append(i)

    tile_tensors = {
        k: torch.tensor(v, dtype=torch.long, device=points.device)
        for k, v in tiles.items()
    }
    return tile_tensors, xy_min


# ============================================================
# D2AS-FPS (tile-local, radius-stop)
# ============================================================

@torch.no_grad()
def d2as_fps_in_tile(
    tile_points: torch.Tensor,     # [Nt, 3]
    tile_global_idx: torch.Tensor, # [Nt]
    radius: float,
    start_local_idx: int = 0,
) -> Tuple[torch.Tensor, int]:
    """
    Perform radius-stop FPS inside one tile.

    return:
      selected_global_idx: [Mt]
      dist_ops
    """
    nt = tile_points.size(0)
    if nt == 0:
        return torch.empty(0, dtype=torch.long, device=tile_points.device), 0

    radius_sqr = radius * radius

    selected_local: List[int] = []
    active = torch.ones(nt, dtype=torch.bool, device=tile_points.device)
    min_dist = torch.full((nt,), float("inf"), device=tile_points.device)

    cur = start_local_idx
    dist_ops = 0

    while True:
        selected_local.append(cur)
        active[cur] = False

        cur_point = tile_points[cur].unsqueeze(0)                  # [1, 3]
        d = pairwise_square_dist(tile_points, cur_point).squeeze(1)
        dist_ops += nt

        min_dist = torch.minimum(min_dist, d)
        min_dist[~active] = -1.0

        if active.sum() == 0:
            break

        dmax, nxt = torch.max(min_dist, dim=0)
        if float(dmax.item()) <= radius_sqr:
            break

        cur = int(nxt.item())

    selected_local_tensor = torch.tensor(
        selected_local, dtype=torch.long, device=tile_points.device
    )
    selected_global_idx = tile_global_idx[selected_local_tensor]
    return selected_global_idx, dist_ops


@torch.no_grad()
def tile_based_d2as_fps_single(
    points: torch.Tensor,          # [N, 3]
    radius: float,
    tile_size: float,
    start_mode: str = "first",
) -> SampleResult:
    """
    Tile-based D2AS-FPS:
      1) partition by XY tile
      2) run radius-stop FPS independently inside each tile
      3) concatenate all selected indices
    """
    assert points.dim() == 2 and points.size(1) == 3
    device = points.device

    t0 = time.perf_counter()

    tiles, xy_origin = build_xy_tiles(points, tile_size=tile_size)

    all_selected: List[torch.Tensor] = []
    total_dist_ops = 0
    per_tile_counts = {}

    for key in sorted(tiles.keys()):
        idx = tiles[key]  # global indices
        tile_pts = points[idx]

        if start_mode == "first":
            start_local = 0
        elif start_mode == "random":
            start_local = int(torch.randint(0, tile_pts.size(0), (1,), device=device).item())
        else:
            raise ValueError(f"Unsupported start_mode: {start_mode}")

        selected_global_idx, dist_ops = d2as_fps_in_tile(
            tile_points=tile_pts,
            tile_global_idx=idx,
            radius=radius,
            start_local_idx=start_local,
        )

        all_selected.append(selected_global_idx)
        total_dist_ops += dist_ops
        per_tile_counts[str(key)] = int(selected_global_idx.numel())

    if len(all_selected) == 0:
        final_idx = torch.empty(0, dtype=torch.long, device=device)
    else:
        final_idx = torch.cat(all_selected, dim=0)

    sampled = points[final_idx]
    t1 = time.perf_counter()

    return SampleResult(
        indices=final_idx,
        points=sampled,
        dist_ops=total_dist_ops,
        runtime_ms=(t1 - t0) * 1000.0,
        extra={
            "radius": radius,
            "tile_size": tile_size,
            "xy_origin": xy_origin.detach().cpu().tolist(),
            "per_tile_counts": per_tile_counts,
            "num_tiles": len(tiles),
        },
    )


# ============================================================
# Search radius to match target sample count
# ============================================================

@torch.no_grad()
def calibrate_radius_to_target_count(
    points: torch.Tensor,
    target_m: int,
    tile_size: float,
    radius_lo: float,
    radius_hi: float,
    max_steps: int = 20,
    tol_ratio: float = 0.05,
) -> Tuple[float, SampleResult]:
    """
    Binary search radius so that D2AS sample count approaches target_m.

    Note:
      larger radius -> fewer samples
      smaller radius -> more samples
    """
    best_res = None
    best_r = None
    best_gap = float("inf")

    lo = radius_lo
    hi = radius_hi

    for _ in range(max_steps):
        mid = 0.5 * (lo + hi)
        res = tile_based_d2as_fps_single(
            points=points,
            radius=mid,
            tile_size=tile_size,
            start_mode="first",
        )
        m_mid = res.indices.numel()
        gap = abs(m_mid - target_m)

        if gap < best_gap:
            best_gap = gap
            best_r = mid
            best_res = res

        if abs(m_mid - target_m) / max(target_m, 1) <= tol_ratio:
            return mid, res

        if m_mid > target_m:
            # too many samples -> increase radius
            lo = mid
        else:
            # too few samples -> decrease radius
            hi = mid

    assert best_res is not None
    return best_r, best_res


# ============================================================
# Comparison
# ============================================================

@torch.no_grad()
def compare_one_pointcloud(
    points: torch.Tensor,          # [N, 3]
    fps_m: int,
    tile_size: float,
    radius_mode: str = "calibrate",
    radius: Optional[float] = None,
    radius_lo: float = 1e-4,
    radius_hi: float = 1.0,
    start_idx: int = 0,
) -> Tuple[CompareResult, SampleResult, SampleResult]:
    """
    Compare vanilla FPS vs tile D2AS-FPS on one point cloud.
    """
    fps_res = vanilla_fps_single(points=points, m=fps_m, start_idx=start_idx)

    if radius_mode == "fixed":
        assert radius is not None
        d2as_res = tile_based_d2as_fps_single(
            points=points,
            radius=radius,
            tile_size=tile_size,
            start_mode="first",
        )
    elif radius_mode == "calibrate":
        _, d2as_res = calibrate_radius_to_target_count(
            points=points,
            target_m=fps_m,
            tile_size=tile_size,
            radius_lo=radius_lo,
            radius_hi=radius_hi,
        )
    else:
        raise ValueError(f"Unsupported radius_mode: {radius_mode}")

    overlap = index_overlap_ratio(fps_res.indices, d2as_res.indices)
    cd = chamfer_distance_sqr(fps_res.points, d2as_res.points)

    fps_max_cover, fps_avg_cover = directed_cover_stats(points, fps_res.points)
    d2as_max_cover, d2as_avg_cover = directed_cover_stats(points, d2as_res.points)

    comp = CompareResult(
        n_input=int(points.size(0)),
        m_fps=int(fps_res.indices.numel()),
        m_d2as=int(d2as_res.indices.numel()),
        overlap_ratio=overlap,
        chamfer_sqr=cd,
        fps_max_cover=fps_max_cover,
        fps_avg_cover=fps_avg_cover,
        d2as_max_cover=d2as_max_cover,
        d2as_avg_cover=d2as_avg_cover,
        fps_dist_ops=int(fps_res.dist_ops),
        d2as_dist_ops=int(d2as_res.dist_ops),
        fps_runtime_ms=float(fps_res.runtime_ms),
        d2as_runtime_ms=float(d2as_res.runtime_ms),
    )
    return comp, fps_res, d2as_res


# ============================================================
# Demo data
# ============================================================

def make_demo_pointcloud(
    n: int,
    mode: str = "uniform",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a demo point cloud for quick validation.
    """
    if mode == "uniform":
        pts = torch.rand(n, 3, device=device)
    elif mode == "cluster":
        num_centers = 8
        centers = torch.rand(num_centers, 3, device=device)
        assign = torch.randint(0, num_centers, (n,), device=device)
        pts = centers[assign] + 0.05 * torch.randn(n, 3, device=device)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return pts


# ============================================================
# Batch runner
# ============================================================

@torch.no_grad()
def run_batch(
    batch_points: torch.Tensor,    # [B, N, 3]
    fps_m: int,
    tile_size: float,
    radius_mode: str,
    radius: Optional[float],
    radius_lo: float,
    radius_hi: float,
) -> List[Dict]:
    results = []
    b = batch_points.size(0)
    for i in range(b):
        comp, fps_res, d2as_res = compare_one_pointcloud(
            points=batch_points[i],
            fps_m=fps_m,
            tile_size=tile_size,
            radius_mode=radius_mode,
            radius=radius,
            radius_lo=radius_lo,
            radius_hi=radius_hi,
            start_idx=0,
        )
        item = asdict(comp)
        item["fps_extra"] = fps_res.extra
        item["d2as_extra"] = d2as_res.extra
        results.append(item)
    return results


def summarize(results: List[Dict]) -> Dict:
    keys_mean = [
        "m_fps",
        "m_d2as",
        "overlap_ratio",
        "chamfer_sqr",
        "fps_max_cover",
        "fps_avg_cover",
        "d2as_max_cover",
        "d2as_avg_cover",
        "fps_dist_ops",
        "d2as_dist_ops",
        "fps_runtime_ms",
        "d2as_runtime_ms",
    ]
    summary = {}
    for k in keys_mean:
        vals = [float(r[k]) for r in results]
        summary[k + "_mean"] = sum(vals) / len(vals)

    summary["dist_ops_ratio_mean"] = (
        summary["d2as_dist_ops_mean"] / max(summary["fps_dist_ops_mean"], 1e-12)
    )
    summary["runtime_ratio_mean"] = (
        summary["d2as_runtime_ms_mean"] / max(summary["fps_runtime_ms_mean"], 1e-12)
    )
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_points", type=int, default=1024)
    parser.add_argument("--fps_m", type=int, default=256)

    parser.add_argument("--demo_mode", type=str, default="cluster", choices=["uniform", "cluster"])

    parser.add_argument("--tile_size", type=float, default=0.25)

    parser.add_argument("--radius_mode", type=str, default="calibrate", choices=["fixed", "calibrate"])
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument("--radius_lo", type=float, default=1e-4)
    parser.add_argument("--radius_hi", type=float, default=1.0)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    set_seed(args.seed)

    batch = torch.stack(
        [
            make_demo_pointcloud(
                n=args.num_points,
                mode=args.demo_mode,
                device=args.device,
            )
            for _ in range(args.batch_size)
        ],
        dim=0,
    )  # [B, N, 3]

    results = run_batch(
        batch_points=batch,
        fps_m=args.fps_m,
        tile_size=args.tile_size,
        radius_mode=args.radius_mode,
        radius=args.radius if args.radius_mode == "fixed" else None,
        radius_lo=args.radius_lo,
        radius_hi=args.radius_hi,
    )

    summary = summarize(results)

    print("\n================ Per-sample Results ================")
    for i, r in enumerate(results):
        print(f"[Sample {i}]")
        print(
            f"  N={r['n_input']}, "
            f"M_fps={r['m_fps']}, M_d2as={r['m_d2as']}, "
            f"overlap={r['overlap_ratio']:.4f}, "
            f"CD^2={r['chamfer_sqr']:.6f}"
        )
        print(
            f"  cover_fps(max/avg)=({r['fps_max_cover']:.6f}, {r['fps_avg_cover']:.6f}), "
            f"cover_d2as(max/avg)=({r['d2as_max_cover']:.6f}, {r['d2as_avg_cover']:.6f})"
        )
        print(
            f"  dist_ops_fps={r['fps_dist_ops']}, dist_ops_d2as={r['d2as_dist_ops']}, "
            f"time_fps={r['fps_runtime_ms']:.3f}ms, time_d2as={r['d2as_runtime_ms']:.3f}ms"
        )

    print("\n================ Summary ================")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()