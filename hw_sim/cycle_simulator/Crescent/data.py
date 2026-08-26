from __future__ import annotations
import numpy as np


def generate_point_cloud(num_points: int, feat_dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    xyz = rng.random((num_points, 3), dtype=np.float32)
    # induce some locality in features so BCE impact is visible but not catastrophic
    base = np.concatenate([xyz, xyz ** 2, np.sin(2 * np.pi * xyz)], axis=1)
    reps = (feat_dim + base.shape[1] - 1) // base.shape[1]
    feat = np.tile(base, (1, reps))[:, :feat_dim]
    feat += 0.01 * rng.standard_normal(feat.shape)
    feat = feat.astype(np.float32)
    return xyz, feat


def sample_queries(xyz: np.ndarray, num_queries: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    qids = rng.choice(np.arange(len(xyz)), size=num_queries, replace=False)
    return qids, xyz[qids]
