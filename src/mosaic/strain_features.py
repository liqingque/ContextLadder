"""Fold and distance helpers for external natural-isolate baselines."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd


def make_group_folds(metadata: pd.DataFrame, n_splits: int = 5, seed: int = 20260810) -> pd.DataFrame:
    """Assign whole clades to balanced folds without splitting a clade."""
    if "isolate_id" not in metadata or "fold_group" not in metadata:
        raise ValueError("metadata requires isolate_id and fold_group")
    counts = metadata.groupby("fold_group").size().sort_values(ascending=False)
    rng = np.random.RandomState(seed)
    groups = list(counts.index)
    # Randomize equal-size groups while preserving the large-first balancing.
    jitter = {group: float(rng.uniform()) for group in groups}
    groups.sort(key=lambda group: (-int(counts[group]), jitter[group]))
    fold_sizes = [0] * n_splits
    assignment = {}
    for group in groups:
        fold = int(np.argmin(fold_sizes))
        assignment[group] = fold
        fold_sizes[fold] += int(counts[group])
    result = metadata.copy()
    result["holdout_fold"] = result["fold_group"].map(assignment).astype(int)
    return result


def knn_predict(train_values: np.ndarray, distances: np.ndarray, k: int) -> np.ndarray:
    """Missing-aware unweighted genomic-neighbour prediction."""
    if distances.ndim != 2 or distances.shape[1] != train_values.shape[0]:
        raise ValueError("distance/train dimensions do not align")
    k = min(int(k), train_values.shape[0])
    neighbour_order = np.argsort(distances, axis=1)[:, :k]
    predictions = np.full((distances.shape[0], train_values.shape[1]), np.nan, dtype=np.float64)
    for row, neighbours in enumerate(neighbour_order):
        block = train_values[neighbours]
        counts = np.isfinite(block).sum(axis=0)
        sums = np.nansum(block, axis=0)
        predictions[row] = np.divide(sums, counts, out=np.full(block.shape[1], np.nan), where=counts > 0)
    return predictions


def pooled_rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    mask = np.isfinite(prediction) & np.isfinite(truth)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(prediction[mask] - truth[mask]))))

