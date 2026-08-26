# -*- coding: utf-8 -*-
"""
benchmark_merkle_fps_knn.py

Benchmark for Merkle-style FPS-KNN prototype.

Assumes the following objects are defined in merkle_fps.py:
    - exact_fps
    - tree_fps
    - exact_knn
    - compare_fps
    - summarize_fps_stats

Usage examples:
    python benchmark_merkle_fps_knn.py
    python benchmark_merkle_fps_knn.py --num-trials 50 --N 1024 --K 32 --k 16
"""

import os
import math
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from merkle_fps import (
    exact_fps,
    tree_fps,
    exact_knn,
    compare_fps,
    summarize_fps_stats,
)


# ============================================================
# Data generation
# ============================================================

def gen_clustered_points(
    N: int,
    D: int = 3,
    num_clusters: int = 8,
    cluster_std: float = 0.35,
    center_scale: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts_per_cluster = N // num_clusters
    remain = N - pts_per_cluster * num_clusters

    centers = rng.standard_normal((num_clusters, D), dtype=np.float32) * center_scale

    X_list = []
    for i in range(num_clusters):
        cur_n = pts_per_cluster + (1 if i < remain else 0)
        pts = centers[i] + cluster_std * rng.standard_normal((cur_n, D), dtype=np.float32)
        X_list.append(pts)

    X = np.concatenate(X_list, axis=0)
    perm = rng.permutation(len(X))
    X = X[perm]
    return X.astype(np.float32)


def gen_uniform_cube_points(
    N: int,
    D: int = 3,
    cube_half_extent: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.uniform(
        low=-cube_half_extent,
        high=cube_half_extent,
        size=(N, D)
    ).astype(np.float32)
    return X


def gen_sphere_surface_points(
    N: int,
    D: int = 3,
    radius: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    assert D == 3, "sphere surface generator currently supports D=3 only"
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, 3), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = radius * X / norms
    return X.astype(np.float32)


def make_dataset(
    dataset_type: str,
    N: int,
    D: int,
    seed: int,
    num_clusters: int = 8,
    cluster_std: float = 0.35,
) -> np.ndarray:
    if dataset_type == "clustered":
        return gen_clustered_points(
            N=N,
            D=D,
            num_clusters=num_clusters,
            cluster_std=cluster_std,
            seed=seed,
        )
    elif dataset_type == "uniform_cube":
        return gen_uniform_cube_points(
            N=N,
            D=D,
            seed=seed,
        )
    elif dataset_type == "sphere_surface":
        return gen_sphere_surface_points(
            N=N,
            D=D,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ============================================================
# Metrics
# ============================================================

def safe_mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))


def evaluate_single_trial(
    X: np.ndarray,
    K: int,
    k: int,
    leaf_size: int,
    seed_idx: int = 0,
    query_mode: str = "random_point",
    trial_seed: int = 0,
):
    N = X.shape[0]
    rng = np.random.default_rng(trial_seed)

    # ---------- FPS ----------
    sel_exact, _ = exact_fps(X, K=K, seed_idx=seed_idx)
    sel_tree, _, fps_stats_list, tree = tree_fps(X, K=K, leaf_size=leaf_size, seed_idx=seed_idx)

    fps_cmp = compare_fps(sel_exact, sel_tree)
    fps_stats = summarize_fps_stats(fps_stats_list)

    fps_set_recall = fps_cmp["set_recall_vs_exact"]
    fps_update_ratio = fps_stats["total_updated_points"] / max(1, (K - 1) * N)
    fps_eval_ratio = fps_stats["total_distance_evals"] / max(1, (K - 1) * N)

    num_leaves = math.ceil(N / leaf_size)
    fps_touched_leaf_ratio = (
        fps_stats["avg_touched_leaves_per_iter"] / max(1, num_leaves)
    )

    # ---------- KNN ----------
    if query_mode == "random_point":
        q_idx = int(rng.integers(0, N))
        q = X[q_idx]
    elif query_mode == "random_perturbed_point":
        q_idx = int(rng.integers(0, N))
        q = X[q_idx] + 0.01 * rng.standard_normal(X.shape[1], dtype=np.float32)
    else:
        raise ValueError(f"Unknown query_mode: {query_mode}")

    idx_exact, _ = exact_knn(X, q, k)
    idx_tree, _, knn_stats = tree.tree_knn(q, k)

    knn_overlap = len(set(idx_exact.tolist()) & set(idx_tree.tolist())) / max(1, k)
    knn_eval_ratio = knn_stats["distance_evals"] / max(1, N)
    knn_leaf_ratio = knn_stats["touched_leaves"] / max(1, num_leaves)

    return {
    "fps_set_recall": fps_set_recall,
    "fps_update_ratio": fps_update_ratio,
    "fps_eval_ratio": fps_eval_ratio,
    "fps_touched_leaf_ratio": fps_touched_leaf_ratio,
    "fps_avg_visited_nodes": fps_stats["avg_visited_nodes_per_iter"],
    "fps_avg_touched_leaves": fps_stats["avg_touched_leaves_per_iter"],
    "fps_avg_distance_evals": fps_stats["avg_distance_evals_per_iter"],

    "knn_overlap": knn_overlap,
    "knn_eval_ratio": knn_eval_ratio,
    "knn_leaf_ratio": knn_leaf_ratio,
    "knn_visited_nodes": knn_stats["visited_nodes"],
    "knn_touched_leaves": knn_stats["touched_leaves"],
    "knn_distance_evals": knn_stats["distance_evals"],
}


def aggregate_trial_results(results):
    keys = results[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in results]
        mean, std = safe_mean_std(vals)
        out[k + "_mean"] = mean
        out[k + "_std"] = std
    return out


# ============================================================
# Benchmark runners
# ============================================================

def run_trials_for_config(
    dataset_type: str,
    num_trials: int,
    N: int,
    D: int,
    K: int,
    k: int,
    leaf_size: int,
    num_clusters: int,
    cluster_std: float,
):
    results = []
    for seed in range(num_trials):
        X = make_dataset(
            dataset_type=dataset_type,
            N=N,
            D=D,
            seed=seed,
            num_clusters=num_clusters,
            cluster_std=cluster_std,
        )

        res = evaluate_single_trial(
            X=X,
            K=K,
            k=k,
            leaf_size=leaf_size,
            seed_idx=0,
            query_mode="random_perturbed_point",
            trial_seed=seed,
        )
        results.append(res)

    return aggregate_trial_results(results)


def sweep_leaf_sizes(
    out_dir: str,
    dataset_type: str,
    leaf_sizes,
    num_trials: int,
    N: int,
    D: int,
    K: int,
    k: int,
    num_clusters: int,
    cluster_std: float,
):
    rows = []
    for leaf_size in leaf_sizes:
        agg = run_trials_for_config(
            dataset_type=dataset_type,
            num_trials=num_trials,
            N=N,
            D=D,
            K=K,
            k=k,
            leaf_size=leaf_size,
            num_clusters=num_clusters,
            cluster_std=cluster_std,
        )
        agg["leaf_size"] = leaf_size
        rows.append(agg)
        print(f"[leaf sweep] dataset={dataset_type} leaf={leaf_size} done")

    save_leaf_sweep_plots(rows, out_dir, dataset_type)
    save_rows_txt(rows, os.path.join(out_dir, f"leaf_sweep_{dataset_type}.txt"))
    return rows


def sweep_cluster_std(
    out_dir: str,
    cluster_stds,
    num_trials: int,
    N: int,
    D: int,
    K: int,
    k: int,
    leaf_size: int,
    num_clusters: int,
):
    rows = []
    for cluster_std in cluster_stds:
        agg = run_trials_for_config(
            dataset_type="clustered",
            num_trials=num_trials,
            N=N,
            D=D,
            K=K,
            k=k,
            leaf_size=leaf_size,
            num_clusters=num_clusters,
            cluster_std=cluster_std,
        )
        agg["cluster_std"] = cluster_std
        rows.append(agg)
        print(f"[cluster_std sweep] std={cluster_std} done")

    save_cluster_std_sweep_plots(rows, out_dir)
    save_rows_txt(rows, os.path.join(out_dir, "cluster_std_sweep.txt"))
    return rows


def compare_distributions(
    out_dir: str,
    dataset_types,
    num_trials: int,
    N: int,
    D: int,
    K: int,
    k: int,
    leaf_size: int,
    num_clusters: int,
    cluster_std: float,
):
    rows = []
    for dataset_type in dataset_types:
        agg = run_trials_for_config(
            dataset_type=dataset_type,
            num_trials=num_trials,
            N=N,
            D=D,
            K=K,
            k=k,
            leaf_size=leaf_size,
            num_clusters=num_clusters,
            cluster_std=cluster_std,
        )
        agg["dataset_type"] = dataset_type
        rows.append(agg)
        print(f"[dist compare] dataset={dataset_type} done")

    save_distribution_bar_plots(rows, out_dir)
    save_rows_txt(rows, os.path.join(out_dir, "distribution_compare.txt"))
    return rows


# ============================================================
# Plot helpers
# ============================================================

def plot_line_with_errorbar(x, y, yerr, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(6.2, 4.6))
    plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_bar_with_errorbar(labels, y, yerr, ylabel, title, save_path):
    plt.figure(figsize=(7.0, 4.8))
    x = np.arange(len(labels))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_leaf_sweep_plots(rows, out_dir, dataset_type):
    xs = [r["leaf_size"] for r in rows]

    plot_line_with_errorbar(
        x=xs,
        y=[r["fps_set_recall_mean"] for r in rows],
        yerr=[r["fps_set_recall_std"] for r in rows],
        xlabel="Leaf Size",
        ylabel="FPS Set Recall vs Exact",
        title=f"Leaf Sweep ({dataset_type}) - FPS Recall",
        save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_fps_recall.png"),
    )
    plot_line_with_errorbar(
    x=xs,
    y=[r["fps_eval_ratio_mean"] for r in rows],
    yerr=[r["fps_eval_ratio_std"] for r in rows],
    xlabel="Leaf Size",
    ylabel="FPS Distance Eval Ratio",
    title=f"Leaf Sweep ({dataset_type}) - FPS Eval Ratio",
    save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_fps_eval_ratio.png"),
)

    plot_line_with_errorbar(
        x=xs,
        y=[r["fps_update_ratio_mean"] for r in rows],
        yerr=[r["fps_update_ratio_std"] for r in rows],
        xlabel="Leaf Size",
        ylabel="FPS Update Ratio",
        title=f"Leaf Sweep ({dataset_type}) - FPS Update Ratio",
        save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_fps_update_ratio.png"),
    )
    
    plot_line_with_errorbar(
        x=xs,
        y=[r["fps_eval_ratio_mean"] for r in rows],
        yerr=[r["fps_eval_ratio_std"] for r in rows],
        xlabel="Leaf Size",
        ylabel="FPS Distance Eval Ratio",
        title=f"Leaf Sweep ({dataset_type}) - FPS Eval Ratio",
        save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_fps_eval_ratio.png"),
    )
    

    plot_line_with_errorbar(
        x=xs,
        y=[r["fps_touched_leaf_ratio_mean"] for r in rows],
        yerr=[r["fps_touched_leaf_ratio_std"] for r in rows],
        xlabel="Leaf Size",
        ylabel="FPS Touched Leaf Ratio",
        title=f"Leaf Sweep ({dataset_type}) - FPS Touched Leaf Ratio",
        save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_fps_leaf_ratio.png"),
    )

    plot_line_with_errorbar(
        x=xs,
        y=[r["knn_overlap_mean"] for r in rows],
        yerr=[r["knn_overlap_std"] for r in rows],
        xlabel="Leaf Size",
        ylabel="KNN Overlap@k",
        title=f"Leaf Sweep ({dataset_type}) - KNN Overlap",
        save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_knn_overlap.png"),
    )

    plot_line_with_errorbar(
        x=xs,
        y=[r["knn_eval_ratio_mean"] for r in rows],
        yerr=[r["knn_eval_ratio_std"] for r in rows],
        xlabel="Leaf Size",
        ylabel="KNN Distance Eval Ratio",
        title=f"Leaf Sweep ({dataset_type}) - KNN Eval Ratio",
        save_path=os.path.join(out_dir, f"leaf_sweep_{dataset_type}_knn_eval_ratio.png"),
    )


def save_cluster_std_sweep_plots(rows, out_dir):
    xs = [r["cluster_std"] for r in rows]

    plot_line_with_errorbar(
        x=xs,
        y=[r["fps_set_recall_mean"] for r in rows],
        yerr=[r["fps_set_recall_std"] for r in rows],
        xlabel="Cluster Std",
        ylabel="FPS Set Recall vs Exact",
        title="Cluster Std Sweep - FPS Recall",
        save_path=os.path.join(out_dir, "cluster_std_fps_recall.png"),
    )

    plot_line_with_errorbar(
        x=xs,
        y=[r["fps_update_ratio_mean"] for r in rows],
        yerr=[r["fps_update_ratio_std"] for r in rows],
        xlabel="Cluster Std",
        ylabel="FPS Update Ratio",
        title="Cluster Std Sweep - FPS Update Ratio",
        save_path=os.path.join(out_dir, "cluster_std_fps_update_ratio.png"),
    )
    plot_line_with_errorbar(
    x=xs,
    y=[r["fps_eval_ratio_mean"] for r in rows],
    yerr=[r["fps_eval_ratio_std"] for r in rows],
    xlabel="Cluster Std",
    ylabel="FPS Distance Eval Ratio",
    title="Cluster Std Sweep - FPS Eval Ratio",
    save_path=os.path.join(out_dir, "cluster_std_fps_eval_ratio.png"),
)
    plot_line_with_errorbar(
        x=xs,
        y=[r["knn_overlap_mean"] for r in rows],
        yerr=[r["knn_overlap_std"] for r in rows],
        xlabel="Cluster Std",
        ylabel="KNN Overlap@k",
        title="Cluster Std Sweep - KNN Overlap",
        save_path=os.path.join(out_dir, "cluster_std_knn_overlap.png"),
    )

    plot_line_with_errorbar(
        x=xs,
        y=[r["knn_eval_ratio_mean"] for r in rows],
        yerr=[r["knn_eval_ratio_std"] for r in rows],
        xlabel="Cluster Std",
        ylabel="KNN Distance Eval Ratio",
        title="Cluster Std Sweep - KNN Eval Ratio",
        save_path=os.path.join(out_dir, "cluster_std_knn_eval_ratio.png"),
    )


def save_distribution_bar_plots(rows, out_dir):
    labels = [r["dataset_type"] for r in rows]

    plot_bar_with_errorbar(
        labels=labels,
        y=[r["fps_set_recall_mean"] for r in rows],
        yerr=[r["fps_set_recall_std"] for r in rows],
        ylabel="FPS Set Recall vs Exact",
        title="Distribution Compare - FPS Recall",
        save_path=os.path.join(out_dir, "dist_compare_fps_recall.png"),
    )

    plot_bar_with_errorbar(
        labels=labels,
        y=[r["fps_update_ratio_mean"] for r in rows],
        yerr=[r["fps_update_ratio_std"] for r in rows],
        ylabel="FPS Update Ratio",
        title="Distribution Compare - FPS Update Ratio",
        save_path=os.path.join(out_dir, "dist_compare_fps_update_ratio.png"),
    )
    
    plot_bar_with_errorbar(
        labels=labels,
        y=[r["fps_eval_ratio_mean"] for r in rows],
        yerr=[r["fps_eval_ratio_std"] for r in rows],
        ylabel="FPS Distance Eval Ratio",
        title="Distribution Compare - FPS Eval Ratio",
        save_path=os.path.join(out_dir, "dist_compare_fps_eval_ratio.png"),
    )
    plot_bar_with_errorbar(
        labels=labels,
        y=[r["knn_overlap_mean"] for r in rows],
        yerr=[r["knn_overlap_std"] for r in rows],
        ylabel="KNN Overlap@k",
        title="Distribution Compare - KNN Overlap",
        save_path=os.path.join(out_dir, "dist_compare_knn_overlap.png"),
    )

    plot_bar_with_errorbar(
        labels=labels,
        y=[r["knn_eval_ratio_mean"] for r in rows],
        yerr=[r["knn_eval_ratio_std"] for r in rows],
        ylabel="KNN Distance Eval Ratio",
        title="Distribution Compare - KNN Eval Ratio",
        save_path=os.path.join(out_dir, "dist_compare_knn_eval_ratio.png"),
    )


# ============================================================
# Text output
# ============================================================

def format_metric(mean_key, std_key, row):
    return f"{row[mean_key]:.6f} ± {row[std_key]:.6f}"


def save_rows_txt(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(f"===== row {i} =====\n")
            for k, v in row.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")


def print_leaf_table(rows, dataset_type):
    print(f"\n===== Leaf Sweep Summary ({dataset_type}) =====")

    header = (
    f"{'leaf':>6} | {'fps_recall':>20} | {'fps_update_ratio':>20} | "
    f"{'fps_eval_ratio':>20} | {'fps_leaf_ratio':>20} | "
    f"{'knn_overlap':>20} | {'knn_eval_ratio':>20}"
)
    
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
        f"{row['leaf_size']:6d} | "
        f"{format_metric('fps_set_recall_mean', 'fps_set_recall_std', row):>20} | "
        f"{format_metric('fps_update_ratio_mean', 'fps_update_ratio_std', row):>20} | "
        f"{format_metric('fps_eval_ratio_mean', 'fps_eval_ratio_std', row):>20} | "
        f"{format_metric('fps_touched_leaf_ratio_mean', 'fps_touched_leaf_ratio_std', row):>20} | "
        f"{format_metric('knn_overlap_mean', 'knn_overlap_std', row):>20} | "
        f"{format_metric('knn_eval_ratio_mean', 'knn_eval_ratio_std', row):>20}"
    )


def print_cluster_std_table(rows):
    print("\n===== Cluster Std Sweep Summary =====")
    header = (
    f"{'std':>8} | {'fps_recall':>20} | {'fps_update_ratio':>20} | "
    f"{'fps_eval_ratio':>20} | {'knn_overlap':>20} | {'knn_eval_ratio':>20}"
)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
    f"{row['cluster_std']:8.3f} | "
    f"{format_metric('fps_set_recall_mean', 'fps_set_recall_std', row):>20} | "
    f"{format_metric('fps_update_ratio_mean', 'fps_update_ratio_std', row):>20} | "
    f"{format_metric('fps_eval_ratio_mean', 'fps_eval_ratio_std', row):>20} | "
    f"{format_metric('knn_overlap_mean', 'knn_overlap_std', row):>20} | "
    f"{format_metric('knn_eval_ratio_mean', 'knn_eval_ratio_std', row):>20}"
)


def print_distribution_table(rows):
    print("\n===== Distribution Compare Summary =====")
    header = (
    f"{'dataset':>16} | {'fps_recall':>20} | {'fps_update_ratio':>20} | "
    f"{'fps_eval_ratio':>20} | {'knn_overlap':>20} | {'knn_eval_ratio':>20}"
)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
    f"{row['dataset_type']:>16} | "
    f"{format_metric('fps_set_recall_mean', 'fps_set_recall_std', row):>20} | "
    f"{format_metric('fps_update_ratio_mean', 'fps_update_ratio_std', row):>20} | "
    f"{format_metric('fps_eval_ratio_mean', 'fps_eval_ratio_std', row):>20} | "
    f"{format_metric('knn_overlap_mean', 'knn_overlap_std', row):>20} | "
    f"{format_metric('knn_eval_ratio_mean', 'knn_eval_ratio_std', row):>20}"
)


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="benchmark_out")
    parser.add_argument("--num-trials", type=int, default=30)

    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)

    parser.add_argument("--leaf-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--cluster-stds", type=float, nargs="+", default=[0.10, 0.20, 0.35, 0.50, 0.80, 1.20])

    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--default-cluster-std", type=float, default=0.35)
    parser.add_argument("--default-leaf-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) leaf sweep on clustered
    leaf_rows_clustered = sweep_leaf_sizes(
        out_dir=args.out_dir,
        dataset_type="clustered",
        leaf_sizes=args.leaf_sizes,
        num_trials=args.num_trials,
        N=args.N,
        D=args.D,
        K=args.K,
        k=args.k,
        num_clusters=args.num_clusters,
        cluster_std=args.default_cluster_std,
    )
    print_leaf_table(leaf_rows_clustered, dataset_type="clustered")

    # 2) cluster std sweep
    std_rows = sweep_cluster_std(
        out_dir=args.out_dir,
        cluster_stds=args.cluster_stds,
        num_trials=args.num_trials,
        N=args.N,
        D=args.D,
        K=args.K,
        k=args.k,
        leaf_size=args.default_leaf_size,
        num_clusters=args.num_clusters,
    )
    print_cluster_std_table(std_rows)

    # 3) distribution compare
    dist_rows = compare_distributions(
        out_dir=args.out_dir,
        dataset_types=["clustered", "uniform_cube", "sphere_surface"],
        num_trials=args.num_trials,
        N=args.N,
        D=args.D,
        K=args.K,
        k=args.k,
        leaf_size=args.default_leaf_size,
        num_clusters=args.num_clusters,
        cluster_std=args.default_cluster_std,
    )
    print_distribution_table(dist_rows)

    print(f"\nAll results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()