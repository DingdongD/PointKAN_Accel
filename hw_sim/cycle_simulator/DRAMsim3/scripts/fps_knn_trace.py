#!/usr/bin/env python3
"""
Generate DRAMsim3 trace for a simplified FPS + KNN workload.

This script models *memory access cost* (not compute) by emitting DRAM transactions
with timestamps (cycles) in DRAMsim3 trace format:

  0xADDR READ  <cycle>
  0xADDR WRITE <cycle>

Two access patterns are supported:
- ordered: sequential/stride-friendly accesses (high locality)
- random : randomized/gather-like accesses (low locality)

The model is intentionally simple but captures the main difference:
- FPS: repeated full scans over points + updates to a min-dist array
- KNN: for each query/centroid, gather K neighbor points (and optionally write outputs)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Layout:
    base_points: int
    base_min_dist: int
    base_centroids: int
    base_knn_idx: int
    base_knn_out: int

    point_stride: int
    dist_stride: int
    centroid_stride: int
    knn_idx_stride: int
    knn_out_stride: int


def _hex(addr: int) -> str:
    return hex(addr & ((1 << 64) - 1))


def _emit(
    out,
    addr: int,
    is_write: bool,
    cycle: int,
) -> None:
    out.write(f"{_hex(addr)} {'WRITE' if is_write else 'READ'} {cycle}\n")


def _iter_points_ordered(n: int) -> Iterable[int]:
    return range(n)


def _iter_points_random(n: int, rng: random.Random) -> Iterable[int]:
    # Faster than shuffling a huge list for very large n: use permutation if feasible.
    # Here we keep it simple; for large n, consider streaming RNG indices.
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def _choose_neighbors_ordered(center: int, n: int, k: int) -> List[int]:
    # Contiguous window around center (wrap-around) to approximate locality.
    half = k // 2
    return [((center - half + i) % n) for i in range(k)]


def _choose_neighbors_random(n: int, k: int, rng: random.Random) -> List[int]:
    # Unique neighbors (approx) to avoid pathological repeats.
    if k >= n:
        return list(range(n))
    return rng.sample(range(n), k)

def _make_perm(n: int, mode: str, rng: random.Random) -> List[int]:
    """
    Return a permutation list perm[logical] -> physical_slot.
    - linear  : identity
    - shuffled: random permutation
    """
    if mode == "linear":
        return list(range(n))
    if mode == "shuffled":
        perm = list(range(n))
        rng.shuffle(perm)
        return perm
    raise ValueError(f"unknown perm mode: {mode}")


def _invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for logical, phys in enumerate(perm):
        inv[phys] = logical
    return inv


def _point_addr(layout: Layout, phys_slot: int) -> int:
    return layout.base_points + phys_slot * layout.point_stride


def _dist_addr(layout: Layout, phys_slot: int) -> int:
    return layout.base_min_dist + phys_slot * layout.dist_stride


def build_default_layout(
    point_bytes: int,
    dist_bytes: int,
    centroid_bytes: int,
    knn_idx_bytes: int,
    knn_out_bytes: int,
    base: int,
) -> Layout:
    # Keep arrays separated to avoid overlapping addresses.
    # Alignment is a best-effort; DRAMsim3 only sees addresses.
    align = 4096

    def a(x: int) -> int:
        return (x + (align - 1)) // align * align

    base_points = a(base)
    base_min_dist = a(base_points + (1 << 30))      # +1GB
    base_centroids = a(base_min_dist + (1 << 30))   # +1GB
    base_knn_idx = a(base_centroids + (1 << 30))    # +1GB
    base_knn_out = a(base_knn_idx + (1 << 30))      # +1GB

    return Layout(
        base_points=base_points,
        base_min_dist=base_min_dist,
        base_centroids=base_centroids,
        base_knn_idx=base_knn_idx,
        base_knn_out=base_knn_out,
        point_stride=point_bytes,
        dist_stride=dist_bytes,
        centroid_stride=centroid_bytes,
        knn_idx_stride=knn_idx_bytes,
        knn_out_stride=knn_out_bytes,
    )


def gen_fps_trace(
    *,
    out,
    n_points: int,
    n_centroids: int,
    layout: Layout,
    mode: str,
    cycle0: int,
    interarrival: int,
    rng: random.Random,
    write_min_dist: bool,
    write_centroids: bool,
    points_perm: Optional[List[int]],
) -> int:
    """
    FPS model (simplified):
    For each centroid selection step:
    - scan all points: read point coords + read current min_dist
    - update min_dist (write) (optional)
    - write chosen centroid index/result (optional)
    """
    cycle = cycle0

    iter_points = (
        _iter_points_ordered
        if mode == "ordered"
        else (lambda n: _iter_points_random(n, rng))
    )

    for c in range(n_centroids):
        # Full scan over all points
        for i in iter_points(n_points):
            phys = points_perm[i] if points_perm is not None else i
            p_addr = _point_addr(layout, phys)
            d_addr = _dist_addr(layout, phys)
            _emit(out, p_addr, False, cycle)
            cycle += interarrival
            _emit(out, d_addr, False, cycle)
            cycle += interarrival
            if write_min_dist:
                _emit(out, d_addr, True, cycle)
                cycle += interarrival

        if write_centroids:
            cent_addr = layout.base_centroids + c * layout.centroid_stride
            _emit(out, cent_addr, True, cycle)
            cycle += interarrival

    return cycle


def gen_knn_trace(
    *,
    out,
    n_points: int,
    n_queries: int,
    k: int,
    layout: Layout,
    mode: str,
    cycle0: int,
    interarrival: int,
    rng: random.Random,
    write_knn_idx: bool,
    write_knn_out: bool,
    points_perm: Optional[List[int]],
    remap_knn_indices: bool,
) -> int:
    """
    KNN model (simplified):
    For each query:
    - read query point/centroid (read centroid array)
    - gather k neighbor points (read points)
    - optionally write neighbor indices and/or output features
    """
    cycle = cycle0

    for q in range(n_queries):
        q_addr = layout.base_centroids + q * layout.centroid_stride
        _emit(out, q_addr, False, cycle)
        cycle += interarrival

        # Pick a "center" id to correlate ordered neighbors.
        center = q % n_points
        if mode == "ordered":
            neigh = _choose_neighbors_ordered(center, n_points, k)
        else:
            neigh = _choose_neighbors_random(n_points, k, rng)

        for j, ni in enumerate(neigh):
            # If points are physically permuted, KNN gather will still be random
            # unless indices are remapped to the physical order after re-layout.
            if points_perm is None:
                phys = ni
            else:
                if remap_knn_indices:
                    # Treat ni as "physical order index" (i.e., algorithm updated indices
                    # to match new in-DRAM layout).
                    phys = ni
                else:
                    # Treat ni as original logical index into an array that is physically permuted.
                    phys = points_perm[ni]
            p_addr = _point_addr(layout, phys)
            _emit(out, p_addr, False, cycle)
            cycle += interarrival

            if write_knn_idx:
                idx_addr = layout.base_knn_idx + (q * k + j) * layout.knn_idx_stride
                _emit(out, idx_addr, True, cycle)
                cycle += interarrival

            if write_knn_out:
                out_addr = layout.base_knn_out + (q * k + j) * layout.knn_out_stride
                _emit(out, out_addr, True, cycle)
                cycle += interarrival

    return cycle


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate DRAMsim3 trace for FPS + KNN memory accesses"
    )
    ap.add_argument("-o", "--out", required=True, help="Output trace path")
    ap.add_argument("--mode", choices=["ordered", "random"], default="ordered")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--points-phys",
        choices=["linear", "shuffled"],
        default="linear",
        help="Physical layout of points/min_dist in DRAM (permutes addresses)",
    )
    ap.add_argument(
        "--remap-knn-indices",
        action="store_true",
        default=False,
        help="Assume KNN neighbor indices were remapped after physically reordering points",
    )

    # Problem sizes
    ap.add_argument("--n-points", type=int, default=131072)
    ap.add_argument("--n-centroids", type=int, default=1024, help="FPS samples")
    ap.add_argument("--k", type=int, default=32, help="KNN neighbors")
    ap.add_argument("--knn-queries", type=int, default=None, help="Defaults to n-centroids")

    # Timing
    ap.add_argument("--cycle0", type=int, default=0)
    ap.add_argument(
        "--interarrival",
        type=int,
        default=10,
        help="Cycles between consecutive transactions",
    )

    # Memory layout (bytes)
    ap.add_argument("--point-bytes", type=int, default=16, help="Bytes per point record")
    ap.add_argument("--dist-bytes", type=int, default=4, help="Bytes per min-dist entry")
    ap.add_argument("--centroid-bytes", type=int, default=4, help="Bytes per centroid entry")
    ap.add_argument("--knn-idx-bytes", type=int, default=4, help="Bytes per neighbor index")
    ap.add_argument("--knn-out-bytes", type=int, default=16, help="Bytes per neighbor output record")
    ap.add_argument("--base-addr", type=lambda x: int(x, 0), default=0x100000000)

    # Read/write knobs
    ap.add_argument("--fps-write-min-dist", action="store_true", default=True)
    ap.add_argument("--fps-no-write-min-dist", dest="fps_write_min_dist", action="store_false")
    ap.add_argument("--fps-write-centroids", action="store_true", default=True)
    ap.add_argument("--fps-no-write-centroids", dest="fps_write_centroids", action="store_false")
    ap.add_argument("--knn-write-idx", action="store_true", default=True)
    ap.add_argument("--knn-no-write-idx", dest="knn_write_idx", action="store_false")
    ap.add_argument("--knn-write-out", action="store_true", default=False)
    ap.add_argument("--knn-no-write-out", dest="knn_write_out", action="store_false")

    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_queries = args.knn_queries if args.knn_queries is not None else args.n_centroids

    layout = build_default_layout(
        point_bytes=args.point_bytes,
        dist_bytes=args.dist_bytes,
        centroid_bytes=args.centroid_bytes,
        knn_idx_bytes=args.knn_idx_bytes,
        knn_out_bytes=args.knn_out_bytes,
        base=args.base_addr,
    )

    points_perm = None
    if args.points_phys != "linear":
        points_perm = _make_perm(args.n_points, args.points_phys, rng)

    with open(args.out, "w", encoding="utf-8") as f:
        cycle = gen_fps_trace(
            out=f,
            n_points=args.n_points,
            n_centroids=args.n_centroids,
            layout=layout,
            mode=args.mode,
            cycle0=args.cycle0,
            interarrival=args.interarrival,
            rng=rng,
            write_min_dist=args.fps_write_min_dist,
            write_centroids=args.fps_write_centroids,
            points_perm=points_perm,
        )
        _emit(f, 0, False, cycle)  # separator marker (addr=0 read) for visibility
        cycle += args.interarrival
        _ = gen_knn_trace(
            out=f,
            n_points=args.n_points,
            n_queries=n_queries,
            k=args.k,
            layout=layout,
            mode=args.mode,
            cycle0=cycle,
            interarrival=args.interarrival,
            rng=rng,
            write_knn_idx=args.knn_write_idx,
            write_knn_out=args.knn_write_out,
            points_perm=points_perm,
            remap_knn_indices=args.remap_knn_indices,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

