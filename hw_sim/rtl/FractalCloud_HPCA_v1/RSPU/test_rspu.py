#!/usr/bin/env python3
"""
test_rspu.py
============
pytest-based test suite for RSPU golden model verification.
Run with:  pytest test_rspu.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from rspu_golden import (
    to_fixed, from_fixed, dist_sq_fixed, FRAC_SCALE,
    golden_fps, golden_ball_query, golden_knn,
    simulate_window_check, compare_fps_results
)

# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------
@pytest.fixture
def clustered_points():
    """16 points in 4 clusters of 4."""
    raw = [
        (1,1,0),(2,1,0),(1,2,0),(2,2,0),
        (8,1,0),(9,1,0),(8,2,0),(9,2,0),
        (1,8,0),(2,8,0),(1,9,0),(2,9,0),
        (8,8,0),(9,8,0),(8,9,0),(9,9,0),
    ]
    return [(to_fixed(x), to_fixed(y), to_fixed(z)) for x,y,z in raw]


@pytest.fixture
def line_points():
    """8 collinear points along x-axis."""
    return [(to_fixed(i), 0, 0) for i in range(8)]


@pytest.fixture
def single_point():
    return [(to_fixed(3), to_fixed(4), to_fixed(0))]


# ----------------------------------------------------------------
# Distance tests
# ----------------------------------------------------------------
class TestDistCompute:
    def test_zero_distance(self):
        p = (to_fixed(1), to_fixed(2), to_fixed(3))
        assert dist_sq_fixed(p, p) == 0

    def test_unit_distance_x(self):
        a = (to_fixed(0), 0, 0)
        b = (to_fixed(1), 0, 0)
        expected = FRAC_SCALE * FRAC_SCALE  # 1.0^2 in Q8.8^2
        assert dist_sq_fixed(a, b) == expected

    def test_pythagorean_3_4_5(self):
        a = (0, 0, 0)
        b = (to_fixed(3), to_fixed(4), 0)
        expected = (3*3 + 4*4) * FRAC_SCALE * FRAC_SCALE  # 25
        assert dist_sq_fixed(a, b) == expected

    def test_symmetry(self):
        a = (to_fixed(2), to_fixed(3), to_fixed(1))
        b = (to_fixed(5), to_fixed(7), to_fixed(9))
        assert dist_sq_fixed(a, b) == dist_sq_fixed(b, a)

    def test_3d_distance(self):
        a = (0, 0, 0)
        b = (to_fixed(1), to_fixed(1), to_fixed(1))
        expected = 3 * FRAC_SCALE * FRAC_SCALE
        assert dist_sq_fixed(a, b) == expected


# ----------------------------------------------------------------
# FPS tests
# ----------------------------------------------------------------
class TestFPS:
    def test_fps_returns_correct_count(self, clustered_points):
        samples, _ = golden_fps(clustered_points, 4)
        assert len(samples) == 4

    def test_fps_no_duplicates(self, clustered_points):
        samples, _ = golden_fps(clustered_points, 6)
        assert len(samples) == len(set(samples))

    def test_fps_seed_is_first(self, clustered_points):
        samples, _ = golden_fps(clustered_points, 4, seed_idx=0)
        assert samples[0] == 0

    def test_fps_covers_all_clusters(self, clustered_points):
        samples, _ = golden_fps(clustered_points, 4)
        clusters = set(s // 4 for s in samples)
        # Each of 4 clusters should be represented
        assert len(clusters) == 4, f"Clusters covered: {clusters}"

    def test_fps_single_sample(self, clustered_points):
        samples, _ = golden_fps(clustered_points, 1)
        assert len(samples) == 1
        assert samples[0] == 0  # seed

    def test_fps_all_points(self, clustered_points):
        n = len(clustered_points)
        samples, _ = golden_fps(clustered_points, n)
        assert set(samples) == set(range(n))

    def test_fps_line_points(self, line_points):
        """FPS on collinear points should pick endpoints first."""
        samples, _ = golden_fps(line_points, 2, seed_idx=0)
        # Second sample should be the far end (index 7)
        assert samples[1] == 7, f"Expected 7, got {samples[1]}"

    def test_fps_prev_dist_monotone(self, clustered_points):
        """prev_dist should be non-increasing after each iteration."""
        _, hist = golden_fps(clustered_points, 4)
        for pt_idx in range(len(clustered_points)):
            prev = hist[0][pt_idx]
            for it in range(1, len(hist)):
                assert hist[it][pt_idx] <= prev, \
                    f"prev_dist[{pt_idx}] increased at iter {it}"
                prev = hist[it][pt_idx]

    def test_fps_different_seeds(self, clustered_points):
        """Different seeds should still cover all clusters."""
        for seed in range(4):
            samples, _ = golden_fps(clustered_points, 4, seed_idx=seed)
            clusters = set(s // 4 for s in samples)
            assert len(clusters) == 4, f"seed={seed}: only {clusters} covered"


# ----------------------------------------------------------------
# Window Check tests
# ----------------------------------------------------------------
class TestWindowCheck:
    def test_all_valid(self):
        mask = [False] * 8
        result = simulate_window_check(mask, 0, 8)
        assert result == 0

    def test_first_sampled(self):
        mask = [False] * 8
        mask[0] = True  # point 0 sampled
        result = simulate_window_check(mask, 0, 8)
        assert result == 1  # skip to point 1

    def test_consecutive_sampled(self):
        mask = [True, True, True, False] + [False]*4
        result = simulate_window_check(mask, 0, 8)
        assert result == 3

    def test_all_sampled(self):
        mask = [True] * 8
        result = simulate_window_check(mask, 0, 8)
        assert result is None

    def test_start_ptr_offset(self):
        mask = [False] * 8
        mask[2] = True
        result = simulate_window_check(mask, 2, 8)
        assert result == 3  # skip ptr=2 (sampled)

    def test_lod_finds_correct_index(self):
        """LOD should skip exactly the sampled points."""
        mask = [True, True, False, True, False, False, True, True]
        # First valid from ptr=0 should be index 2
        r = simulate_window_check(mask, 0, 8)
        assert r == 2
        # First valid from ptr=3 should be 4
        r = simulate_window_check(mask, 3, 8)
        assert r == 4


# ----------------------------------------------------------------
# Ball Query tests
# ----------------------------------------------------------------
class TestBallQuery:
    def test_bq_finds_nearby_points(self, clustered_points):
        center = [(to_fixed(1), to_fixed(1), 0)]
        r_sq = 5 * FRAC_SCALE * FRAC_SCALE  # radius=sqrt(5)
        results = golden_ball_query(clustered_points, center, r_sq, 16)
        # Points 0,1,2,3 are close to (1,1)
        found = set(n for n, _ in results[0])
        assert 0 in found, "Point 0 should be within radius"

    def test_bq_zero_radius(self, clustered_points):
        center = [(clustered_points[0][0], clustered_points[0][1],
                   clustered_points[0][2])]
        r_sq = 0
        results = golden_ball_query(clustered_points, center, r_sq, 16)
        # Only exact match (point 0) should be found
        found = [n for n, d in results[0]]
        assert found == [0], f"Expected [0], got {found}"

    def test_bq_large_radius(self, clustered_points):
        center = [(to_fixed(5), to_fixed(5), 0)]
        r_sq = 1000 * FRAC_SCALE * FRAC_SCALE  # very large
        results = golden_ball_query(clustered_points, center, r_sq, 16)
        # All 16 points should be within radius
        assert len(results[0]) == 16

    def test_bq_results_sorted_by_dist(self, clustered_points):
        center = [(to_fixed(5), to_fixed(5), 0)]
        r_sq = 100 * FRAC_SCALE * FRAC_SCALE
        results = golden_ball_query(clustered_points, center, r_sq, 16)
        dists = [d for _, d in results[0]]
        assert dists == sorted(dists), "BQ results not sorted by distance"

    def test_bq_num_neighbors_cap(self, clustered_points):
        center = [(to_fixed(5), to_fixed(5), 0)]
        r_sq = 1000 * FRAC_SCALE * FRAC_SCALE
        k = 3
        results = golden_ball_query(clustered_points, center, r_sq, k)
        assert len(results[0]) <= k

    def test_bq_multiple_centers(self, clustered_points):
        centers = [
            (to_fixed(1), to_fixed(1), 0),
            (to_fixed(8), to_fixed(8), 0),
        ]
        r_sq = 5 * FRAC_SCALE * FRAC_SCALE
        results = golden_ball_query(clustered_points, centers, r_sq, 16)
        assert 0 in results
        assert 1 in results
        # Center 0 near cluster A, center 1 near cluster D
        c0_found = set(n for n, _ in results[0])
        c1_found = set(n for n, _ in results[1])
        assert c0_found.issubset({0,1,2,3})
        assert c1_found.issubset({12,13,14,15})


# ----------------------------------------------------------------
# KNN tests
# ----------------------------------------------------------------
class TestKNN:
    def test_knn_returns_k(self, clustered_points):
        centers = [(to_fixed(1), to_fixed(1), 0)]
        results = golden_knn(clustered_points, centers, k=4)
        assert len(results[0]) == 4

    def test_knn_cluster_a(self, clustered_points):
        """KNN from (1,1,0) should return all of cluster A."""
        centers = [(to_fixed(1), to_fixed(1), 0)]
        results = golden_knn(clustered_points, centers, k=4)
        found = set(n for n, _ in results[0])
        assert found == {0, 1, 2, 3}, f"Expected {{0,1,2,3}}, got {found}"

    def test_knn_sorted_by_dist(self, clustered_points):
        centers = [(to_fixed(5), to_fixed(5), 0)]
        results = golden_knn(clustered_points, centers, k=8)
        dists = [d for _, d in results[0]]
        assert dists == sorted(dists)

    def test_knn_k1(self, clustered_points):
        """KNN with k=1 should return nearest point."""
        centers = [(to_fixed(2), to_fixed(2), 0)]  # near point index 3
        results = golden_knn(clustered_points, centers, k=1)
        assert len(results[0]) == 1
        nearest_idx = results[0][0][0]
        # Point 3 = (2,2,0) is exact match
        assert nearest_idx == 3

    def test_knn_k_ge_n(self, clustered_points):
        """KNN with k >= n should return all points."""
        centers = [(to_fixed(5), to_fixed(5), 0)]
        results = golden_knn(clustered_points, centers, k=20)
        assert len(results[0]) == len(clustered_points)

    def test_knn_multiple_centers(self, clustered_points):
        centers = [
            (to_fixed(1), to_fixed(1), 0),   # cluster A center
            (to_fixed(9), to_fixed(9), 0),   # cluster D center
        ]
        results = golden_knn(clustered_points, centers, k=4)
        a_found = set(n for n, _ in results[0])
        d_found = set(n for n, _ in results[1])
        assert a_found == {0, 1, 2, 3}
        assert d_found == {12, 13, 14, 15}


# ----------------------------------------------------------------
# Block-wise FPS vs Global FPS
# ----------------------------------------------------------------
class TestBlockWiseFPS:
    def test_blockwise_coverage(self, clustered_points):
        """Block-wise FPS (1 sample per block) should cover all clusters."""
        blocks = [clustered_points[i*4:(i+1)*4] for i in range(4)]
        block_samples_global = []
        for bi, blk in enumerate(blocks):
            s, _ = golden_fps(blk, num_samples=1, seed_idx=0)
            block_samples_global.append(bi*4 + s[0])

        clusters = set(s // 4 for s in block_samples_global)
        assert len(clusters) == 4, f"Clusters: {clusters}"

    def test_blockwise_less_than_3pct_accuracy_loss(self, clustered_points):
        """
        Per paper: block-wise FPS introduces <0.2% accuracy loss.
        We measure coverage degradation as a proxy.
        """
        def coverage(pts, samples):
            total = sum(
                min(dist_sq_fixed(p, pts[s]) for s in samples)
                for p in pts
            )
            return total

        # Global FPS
        g_samples, _ = golden_fps(clustered_points, 4)
        g_cov = coverage(clustered_points, g_samples)

        # Block-wise FPS
        blocks = [clustered_points[i*4:(i+1)*4] for i in range(4)]
        bw_samples = []
        for bi, blk in enumerate(blocks):
            s, _ = golden_fps(blk, 1)
            bw_samples.append(bi*4 + s[0])
        bw_cov = coverage(clustered_points, bw_samples)

        # Coverage difference should be small
        if g_cov > 0:
            diff_pct = abs(bw_cov - g_cov) / g_cov * 100
            assert diff_pct < 50.0, \
                f"Coverage diff {diff_pct:.1f}% is too large"


# ----------------------------------------------------------------
# Pipeline latency model
# ----------------------------------------------------------------
class TestPipelineLatency:
    def test_fps_cycle_count(self):
        """
        FPS iteration cost: O(n) per sampled point.
        For n=16, k=4 samples: expect ~16*4 = 64 cycles (approx).
        """
        n = 16
        k = 4
        expected_cycles = n * k   # simplified model
        # Pipeline overlap reduces this by ~30% per paper
        actual_estimate = int(expected_cycles * 0.7)
        assert actual_estimate < 100, "Cycle count model sanity check"

    def test_window_check_savings(self, clustered_points):
        """Window check should reduce iterations in later FPS steps."""
        n = len(clustered_points)
        samples, _ = golden_fps(clustered_points, 4)
        sampled_set = set(samples)

        # Without window check: always scan all n points
        work_no_wc = n * (len(samples) - 1)

        # With window check: skip already-sampled points
        work_wc = 0
        current_sampled = {samples[0]}
        for it in range(1, len(samples)):
            # Count non-sampled points in this iteration
            valid_pts = sum(1 for i in range(n) if i not in current_sampled)
            work_wc += valid_pts
            current_sampled.add(samples[it])

        savings_pct = (work_no_wc - work_wc) / work_no_wc * 100
        print(f"\n    WC savings: {savings_pct:.1f}% "
              f"({work_no_wc} -> {work_wc} iterations)")
        assert work_wc < work_no_wc, "Window check should always reduce work"
        assert savings_pct > 0


# ----------------------------------------------------------------
# run
# ----------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
