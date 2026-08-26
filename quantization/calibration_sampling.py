from __future__ import absolute_import

from collections import defaultdict
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def _label_to_int(label):
    arr = np.asarray(label).reshape(-1)
    return int(arr[0])


def dataset_labels(dataset):
    if hasattr(dataset, "label"):
        return [_label_to_int(x) for x in np.asarray(dataset.label)]
    if hasattr(dataset, "labels"):
        return [_label_to_int(x) for x in np.asarray(dataset.labels)]
    return [_label_to_int(dataset[i][1]) for i in range(len(dataset))]


def class_balanced_indices(labels, calibration_size, seed=42):
    labels = [_label_to_int(x) for x in labels]
    by_class = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    rng = random.Random(int(seed))
    classes = sorted(by_class)
    if not classes or calibration_size <= 0:
        return []

    base = int(calibration_size) // len(classes)
    remainder = int(calibration_size) % len(classes)
    selected = []
    for pos, cls in enumerate(classes):
        quota = base + (1 if pos < remainder else 0)
        pool = list(by_class[cls])
        rng.shuffle(pool)
        if quota <= len(pool):
            selected.extend(pool[:quota])
        else:
            selected.extend(pool)
            selected.extend(rng.choice(pool) for _ in range(quota - len(pool)))
    rng.shuffle(selected)
    return selected[: int(calibration_size)]


def build_calibration_loader(
    dataset,
    batch_size,
    workers,
    calibration_size,
    sampling="random",
    seed=42,
    drop_last=True,
):
    if sampling == "random":
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=drop_last,
            generator=generator,
        )
    if sampling == "class_balanced":
        labels = dataset_labels(dataset)
        indices = class_balanced_indices(labels, calibration_size, seed=seed)
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            drop_last=drop_last,
        )
    raise ValueError("Unknown calibration sampling mode: %s" % sampling)
