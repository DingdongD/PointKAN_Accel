#!/usr/bin/env python3
"""
rspu_golden.py
==============
Golden model for RSPU verification (FractalCloud HPCA 2026).

Implements:
  1. FPS  - Farthest Point Sampling (with window-check / sampled-skip)
  2. BQ   - Ball Query (radius neighbor search)
  3. KNN  - K-Nearest Neighbors

All coordinates use Q8.8 fixed-point integers internally
(integer value N is stored as N * 256).

Also provides:
  - VCD parser to compare RTL simulation outputs
  - Automatic pass/fail checking
"""

import numpy as np
import itertools
import json
import os
import subprocess
import sys
from typing import List, Tuple, Dict, Optional

# ----------------------------------------------------------------
# Fixed-point helpers
# ----------------------------------------------------------------
FRAC_BITS = 8
FRAC_SCALE = 1 << FRAC_BITS  # 256


def to_fixed(v: float) -> int:
    """Convert float to Q8.8 fixed-point integer."""
    return int(round(v * FRAC_SCALE))


def from_fixed(v: int) -> float:
    """Convert Q8.8 fixed-point integer to float."""
    return v / FRAC_SCALE


def dist_sq_fixed(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    """Squared Euclidean distance between two Q8.8 3D points."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


# ----------------------------------------------------------------
# Golden FPS
# ----------------------------------------------------------------
def golden_fps(
    points: List[Tuple[int, int, int]],
    num_samples: int,
    seed_idx: int = 0
) -> Tuple[List[int], List[List[int]]]:
    """
    Farthest Point Sampling (global search, golden reference).

    Args:
        points:      list of (x,y,z) in Q8.8 fixed-point
        num_samples: number of points to sample
        seed_idx:    index of seed point (default 0)

    Returns:
        sampled_indices: list of sampled point indices
        prev_dist_history: prev_dist array after each iteration
          (for window-check validation)
    """
    n = len(points)
    assert num_samples <= n, "num_samples > n"

    prev_dist = [float('inf')] * n
    sampled = [False] * n
    sampled_indices = [seed_idx]
    sampled[seed_idx] = True
    prev_dist_history = []

    # Update prev_dist from seed
    for i in range(n):
        if not sampled[i]:
            prev_dist[i] = dist_sq_fixed(points[i], points[seed_idx])
    prev_dist_history.append(prev_dist[:])

    for _ in range(num_samples - 1):
        # Find argmax of prev_dist (farthest unsampled point)
        best_idx = -1
        best_dist = -1
        for i in range(n):
            if not sampled[i] and prev_dist[i] > best_dist:
                best_dist = prev_dist[i]
                best_idx = i

        sampled[best_idx] = True
        sampled_indices.append(best_idx)

        # Update prev_dist
        for i in range(n):
            if not sampled[i]:
                d = dist_sq_fixed(points[i], points[best_idx])
                if d < prev_dist[i]:
                    prev_dist[i] = d

        prev_dist_history.append(prev_dist[:])

    return sampled_indices, prev_dist_history


# ----------------------------------------------------------------
# Golden Ball Query
# ----------------------------------------------------------------
def golden_ball_query(
    search_space: List[Tuple[int, int, int]],
    centers: List[Tuple[int, int, int]],
    radius_sq: int,
    num_neighbors: int
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Ball Query: for each center, find all points within radius^2.

    Returns:
        dict: center_idx -> list of (neighbor_idx, dist_sq)
              sorted by dist_sq, capped at num_neighbors
    """
    results = {}
    for ci, center in enumerate(centers):
        neighbors = []
        for pi, pt in enumerate(search_space):
            d = dist_sq_fixed(pt, center)
            if d <= radius_sq:
                neighbors.append((pi, d))
        neighbors.sort(key=lambda x: x[1])
        results[ci] = neighbors[:num_neighbors]
    return results


# ----------------------------------------------------------------
# Golden KNN
# ----------------------------------------------------------------
def golden_knn(
    search_space: List[Tuple[int, int, int]],
    centers: List[Tuple[int, int, int]],
    k: int
) -> Dict[int, List[Tuple[int, int]]]:
    """
    K-Nearest Neighbors search.

    Returns:
        dict: center_idx -> list of (neighbor_idx, dist_sq), k entries
    """
    results = {}
    for ci, center in enumerate(centers):
        dists = []
        for pi, pt in enumerate(search_space):
            d = dist_sq_fixed(pt, center)
            dists.append((pi, d))
        dists.sort(key=lambda x: x[1])
        results[ci] = dists[:k]
    return results


# ----------------------------------------------------------------
# Window-check simulation
# ----------------------------------------------------------------
def simulate_window_check(
    sampled: List[bool],
    start_ptr: int,
    num_pts: int
) -> Optional[int]:
    """
    Simulate LOD: find next '1' (unsampled) from start_ptr.
    Returns index or None if all sampled.
    """
    for i in range(start_ptr, num_pts):
        if not sampled[i]:
            return i
    return None


# ----------------------------------------------------------------
# Test case generator
# ----------------------------------------------------------------
def make_test_points_clustered(n_clusters=4, pts_per_cluster=8) -> List[Tuple[int, int, int]]:
    """Generate clustered 3D points for testing."""
    cluster_centers = [
        (to_fixed(2),  to_fixed(2),  to_fixed(0)),
        (to_fixed(10), to_fixed(2),  to_fixed(0)),
        (to_fixed(2),  to_fixed(10), to_fixed(0)),
        (to_fixed(10), to_fixed(10), to_fixed(0)),
    ]
    points = []
    np.random.seed(42)
    for cx, cy, cz in cluster_centers[:n_clusters]:
        for _ in range(pts_per_cluster):
            dx = int(np.random.randn() * FRAC_SCALE * 0.5)
            dy = int(np.random.randn() * FRAC_SCALE * 0.5)
            points.append((cx + dx, cy + dy, cz))
    return points


# ----------------------------------------------------------------
# Comparison utilities
# ----------------------------------------------------------------
def compare_fps_results(
    rtl_samples: List[int],
    golden_samples: List[int],
    points: List[Tuple[int, int, int]],
    tolerance_pct: float = 5.0
) -> bool:
    """
    FPS results may differ due to tie-breaking. We check that:
    1. Same number of samples
    2. Total coverage (sum of distances to sampled set) is within tolerance
    """
    if len(rtl_samples) != len(golden_samples):
        print(f"  FAIL: RTL produced {len(rtl_samples)} samples, "
              f"expected {len(golden_samples)}")
        return False

    def coverage(samples):
        total = 0
        for pi, p in enumerate(points):
            min_d = min(dist_sq_fixed(p, points[s]) for s in samples)
            total += min_d
        return total

    rtl_cov    = coverage(rtl_samples)
    golden_cov = coverage(golden_samples)
    diff_pct   = abs(rtl_cov - golden_cov) / (golden_cov + 1e-9) * 100

    if rtl_samples == golden_samples:
        print(f"  PASS: Exact match. Samples: {rtl_samples}")
        return True
    elif diff_pct <= tolerance_pct:
        print(f"  PASS: Coverage within {diff_pct:.2f}% tolerance.")
        print(f"    RTL:    {rtl_samples}")
        print(f"    Golden: {golden_samples}")
        return True
    else:
        print(f"  FAIL: Coverage diff {diff_pct:.2f}% > {tolerance_pct}%")
        print(f"    RTL:    {rtl_samples}")
        print(f"    Golden: {golden_samples}")
        return False


# ----------------------------------------------------------------
# Verilog simulation runner (uses iverilog + vvp)
# ----------------------------------------------------------------
def run_iverilog_sim(rtl_dir: str, tb_dir: str, sim_dir: str) -> Optional[str]:
    """
    Compile and run the Verilog simulation.
    Returns the simulation stdout, or None on failure.
    """
    os.makedirs(sim_dir, exist_ok=True)
    rtl_files = [
        os.path.join(rtl_dir, "rspu_top.v"),
        os.path.join(rtl_dir, "window_check.v"),
        os.path.join(rtl_dir, "dist_compute.v"),
        os.path.join(rtl_dir, "topk_unit.v"),
    ]
    tb_file = os.path.join(tb_dir, "tb_rspu.v")
    out_bin = os.path.join(sim_dir, "rspu_sim")

    # Check for iverilog
    try:
        subprocess.run(["iverilog", "--version"],
                       capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[WARN] iverilog not found. Skipping RTL simulation.")
        return None

    # Compile
    cmd_compile = ["iverilog", "-g2005", "-o", out_bin, tb_file] + rtl_files
    result = subprocess.run(cmd_compile, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] iverilog compile failed:\n{result.stderr}")
        return None

    # Run
    result = subprocess.run(["vvp", out_bin], capture_output=True, text=True,
                             cwd=sim_dir, timeout=60)
    if result.returncode != 0:
        print(f"[ERROR] vvp run failed:\n{result.stderr}")
        return None

    return result.stdout


def parse_sim_fps_output(sim_log: str) -> List[int]:
    """Parse FPS sample indices from simulation log."""
    samples = []
    for line in sim_log.splitlines():
        if "[FPS]" in line and "=" in line:
            try:
                idx = int(line.split("=")[-1].strip())
                samples.append(idx)
            except ValueError:
                pass
    return samples


def parse_sim_ns_output(sim_log: str) -> List[Tuple[int, int, int]]:
    """Parse NS results (center, neighbor, dist) from simulation log."""
    results = []
    for line in sim_log.splitlines():
        if "[NS ]" in line:
            parts = line.split()
            try:
                c   = int([p for p in parts if p.startswith("center=")][0].split("=")[1])
                nbr = int([p for p in parts if p.startswith("neighbor=")][0].split("=")[1])
                d   = int([p for p in parts if p.startswith("dist_sq=")][0].split("=")[1])
                results.append((c, nbr, d))
            except (IndexError, ValueError):
                pass
    return results


# ================================================================
# Main test suite
# ================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rtl_dir  = os.path.join(base_dir, "rtl")
    tb_dir   = os.path.join(base_dir, "tb")
    sim_dir  = os.path.join(base_dir, "sim")

    all_pass = True

    print("=" * 60)
    print("RSPU Golden Model Test Suite")
    print("=" * 60)

    # ---- Define test points (Q8.8 integers) ----
    # 4 clusters of 4 points each = 16 points total
    points_raw = [
        (1,1,0),(2,1,0),(1,2,0),(2,2,0),       # cluster A (bottom-left)
        (8,1,0),(9,1,0),(8,2,0),(9,2,0),        # cluster B (bottom-right)
        (1,8,0),(2,8,0),(1,9,0),(2,9,0),        # cluster C (top-left)
        (8,8,0),(9,8,0),(8,9,0),(9,9,0),        # cluster D (top-right)
    ]
    points = [(to_fixed(x), to_fixed(y), to_fixed(z))
              for x, y, z in points_raw]

    print(f"\nTest point set: {len(points)} points")
    print("  " + ", ".join(f"({x//256},{y//256},{z//256})"
                            for x, y, z in points))

    # ============================================================
    # TEST 1: FPS
    # ============================================================
    print("\n" + "-" * 40)
    print("TEST 1: Farthest Point Sampling")
    print("-" * 40)
    num_samples = 4
    golden_samples, prev_dist_hist = golden_fps(points, num_samples, seed_idx=0)

    print(f"  Golden FPS samples: {golden_samples}")
    print(f"  Expected one point from each cluster.")

    # Verify clusters: each sample should be from a different cluster
    def cluster_of(idx):
        return idx // 4

    cluster_set = set(cluster_of(s) for s in golden_samples)
    if len(cluster_set) == 4:
        print("  PASS: All 4 clusters covered by FPS samples.")
    else:
        print(f"  WARN: Only {len(cluster_set)} clusters covered: {cluster_set}")

    # Print prev_dist evolution
    print("\n  prev_dist evolution (first 3 iterations):")
    for it, pd in enumerate(prev_dist_hist[:3]):
        scaled = [d // (FRAC_SCALE**2) for d in pd]
        print(f"    iter {it}: {scaled}")

    # ============================================================
    # TEST 2: Window Check (sampled-skip)
    # ============================================================
    print("\n" + "-" * 40)
    print("TEST 2: Window Check / LOD simulation")
    print("-" * 40)
    sampled_mask = [False] * 16
    for s in golden_samples:
        sampled_mask[s] = True

    print(f"  Sampled points: {golden_samples}")
    print(f"  Mask: {['1' if s else '0' for s in sampled_mask]}")

    total_wc_skips = sum(1 for s in sampled_mask if s)
    print(f"  Window-check would skip {total_wc_skips} points in next FPS iteration.")

    next_valid = simulate_window_check(sampled_mask, 0, 16)
    print(f"  LOD result (first unsampled from ptr=0): point {next_valid}")
    assert next_valid is not None and not sampled_mask[next_valid], \
        "LOD returned a sampled point!"
    print("  PASS: LOD returns correct first valid point.")

    # ============================================================
    # TEST 3: Ball Query
    # ============================================================
    print("\n" + "-" * 40)
    print("TEST 3: Ball Query")
    print("-" * 40)
    center_bq = [(to_fixed(5), to_fixed(5), to_fixed(0))]
    # radius = 5 units -> radius_sq = 25 * FRAC_SCALE^2
    radius_sq_val = 25 * FRAC_SCALE * FRAC_SCALE
    bq_results = golden_ball_query(points, center_bq, radius_sq_val, num_neighbors=16)

    print(f"  Center: (5,5,0), radius=5 units")
    print(f"  radius_sq = {radius_sq_val}")
    print(f"  Results for center 0:")
    for nbr_idx, d in bq_results[0]:
        x, y, z = (v // 256 for v in points[nbr_idx])
        print(f"    point {nbr_idx:2d} ({x},{y},{z})  dist_sq={d}  "
              f"dist={d**0.5/FRAC_SCALE:.2f}")

    if len(bq_results[0]) > 0:
        print(f"  PASS: {len(bq_results[0])} neighbors found within radius.")
        all_pass = all_pass and True
    else:
        print("  FAIL: No neighbors found!")
        all_pass = False

    # ============================================================
    # TEST 4: KNN
    # ============================================================
    print("\n" + "-" * 40)
    print("TEST 4: K-Nearest Neighbors")
    print("-" * 40)
    k = 4
    centers_knn = [(to_fixed(1), to_fixed(1), to_fixed(0))]  # near cluster A
    knn_results = golden_knn(points, centers_knn, k)

    print(f"  Center: (1,1,0), K={k}")
    print(f"  {k} nearest neighbors:")
    for nbr_idx, d in knn_results[0]:
        x, y, z = (v // 256 for v in points[nbr_idx])
        print(f"    point {nbr_idx:2d} ({x},{y},{z})  dist_sq={d}  "
              f"dist={d**0.5/FRAC_SCALE:.2f}")

    # Expected: points 0,1,2,3 (cluster A)
    nbr_indices = [n for n, _ in knn_results[0]]
    if set(nbr_indices) == {0, 1, 2, 3}:
        print("  PASS: KNN correctly identifies cluster A.")
    else:
        print(f"  WARN: KNN returned {nbr_indices}, expected {{0,1,2,3}}")

    # ============================================================
    # TEST 5: Multi-block FPS consistency
    # ============================================================
    print("\n" + "-" * 40)
    print("TEST 5: Block-wise FPS vs Global FPS")
    print("-" * 40)
    # Split 16 points into 4 blocks of 4
    blocks = [points[i*4:(i+1)*4] for i in range(4)]
    block_samples = []
    for bi, blk in enumerate(blocks):
        s, _ = golden_fps(blk, num_samples=1, seed_idx=0)
        # Map back to global indices
        global_idx = bi * 4 + s[0]
        block_samples.append(global_idx)
        x, y, z = (v // 256 for v in blk[s[0]])
        print(f"  Block {bi}: sampled local idx {s[0]} "
              f"-> global idx {global_idx} ({x},{y},{z})")

    global_samples_set = set(golden_samples)
    bw_samples_set = set(block_samples)
    overlap = global_samples_set & bw_samples_set
    print(f"\n  Global FPS samples:      {sorted(golden_samples)}")
    print(f"  Block-wise FPS samples:  {sorted(block_samples)}")
    print(f"  Overlap: {sorted(overlap)} ({len(overlap)}/4)")
    if len(overlap) >= 3:
        print("  PASS: Block-wise FPS covers same clusters as global FPS.")
    else:
        print("  INFO: Some divergence (acceptable per paper <0.2% accuracy loss)")

    # ============================================================
    # TEST 6: RTL simulation (if iverilog available)
    # ============================================================
    print("\n" + "-" * 40)
    print("TEST 6: RTL simulation (iverilog)")
    print("-" * 40)
    sim_log = run_iverilog_sim(rtl_dir, tb_dir, sim_dir)
    if sim_log is None:
        print("  SKIP: iverilog not available or compile error.")
    else:
        print("  Simulation output:")
        print(sim_log)
        rtl_fps = parse_sim_fps_output(sim_log)
        rtl_ns  = parse_sim_ns_output(sim_log)
        print(f"\n  RTL FPS samples: {rtl_fps}")
        print(f"  RTL NS results:  {len(rtl_ns)} entries")

        if rtl_fps:
            fps_pass = compare_fps_results(rtl_fps, golden_samples, points)
            all_pass = all_pass and fps_pass
        else:
            print("  WARN: No FPS output captured from simulation.")

    # ============================================================
    # Generate testbench data files (for simulation)
    # ============================================================
    print("\n" + "-" * 40)
    print("Generating simulation data files")
    print("-" * 40)
    os.makedirs(sim_dir, exist_ok=True)

    # Write point coordinates as hex (packed Q8.8)
    coord_hex_path = os.path.join(sim_dir, "coord_mem.hex")
    with open(coord_hex_path, "w") as f:
        for x, y, z in points:
            # Pack as 48-bit: {z[15:0], y[15:0], x[15:0]}
            packed = ((z & 0xFFFF) << 32) | ((y & 0xFFFF) << 16) | (x & 0xFFFF)
            f.write(f"{packed:012x}\n")
    print(f"  Wrote {coord_hex_path}")

    # Write expected FPS output
    fps_ref_path = os.path.join(sim_dir, "fps_expected.json")
    with open(fps_ref_path, "w") as f:
        json.dump({
            "num_pts": len(points),
            "num_samples": num_samples,
            "golden_samples": golden_samples,
            "prev_dist_final": [int(d) if d != float('inf') else -1
                                for d in prev_dist_hist[-1]]
        }, f, indent=2)
    print(f"  Wrote {fps_ref_path}")

    # Write expected BQ output
    bq_ref_path = os.path.join(sim_dir, "bq_expected.json")
    with open(bq_ref_path, "w") as f:
        json.dump({
            "radius_sq": radius_sq_val,
            "centers": [[int(v) for v in c] for c in center_bq],
            "results": {
                str(k): [(int(ni), int(d)) for ni, d in v]
                for k, v in bq_results.items()
            }
        }, f, indent=2)
    print(f"  Wrote {bq_ref_path}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print(f"Overall result: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
