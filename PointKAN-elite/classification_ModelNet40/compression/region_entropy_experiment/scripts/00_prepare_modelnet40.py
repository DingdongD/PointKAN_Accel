#!/usr/bin/env python3
"""阶段 0：导出固定顺序的 ModelNet40 测试集为 NPZ（无随机打乱）。"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.utils.io import save_id_map  # noqa: E402


def load_all_test(
    input_dir: str, num_points: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = sorted(glob.glob(os.path.join(input_dir, "ply_data_test*.h5")))
    if not paths:
        raise FileNotFoundError(f"未找到 {input_dir}/ply_data_test*.h5")
    all_data = []
    all_label = []
    id_strings: list[str] = []
    for h5_path in paths:
        base = os.path.basename(h5_path)
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:, :num_points, :].astype(np.float32)
            label = f["label"][:].astype(np.int64).reshape(-1)
        for i in range(data.shape[0]):
            id_strings.append(f"{base}#{i}")
        all_data.append(data)
        all_label.append(label)
    pts = np.concatenate(all_data, axis=0)
    labs = np.concatenate(all_label, axis=0)
    return pts, labs, id_strings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default="/workspace/VoxelNet/pc_data/modelnet40_ply_hdf5_2048",
        help="ModelNet40 HDF5 目录",
    )
    ap.add_argument("--num-points", type=int, default=1024)
    ap.add_argument(
        "--output",
        type=str,
        default=os.path.join(_ROOT, "data", "modelnet40_test.npz"),
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    points, labels, id_strings = load_all_test(args.input, args.num_points)
    n = points.shape[0]
    sample_ids = np.arange(n, dtype=np.int64)

    np.savez_compressed(
        args.output,
        points=points,
        labels=labels,
        sample_ids=sample_ids,
    )
    save_id_map(args.output, id_strings)

    print("样本数:", n)
    print("每样本点数:", points.shape[1])
    print("坐标 min:", float(points.min()), "max:", float(points.max()))
    print("NaN:", bool(np.isnan(points).any()), "Inf:", bool(np.isinf(points).any()))
    print("已写入:", args.output)
    print("id 映射:", args.output.replace(".npz", "_id_map.json"))


if __name__ == "__main__":
    main()
