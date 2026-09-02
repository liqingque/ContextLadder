"""Metrics aligned with the updated GOAI virtual-cell explanation."""

from typing import Dict

import numpy as np

from .evaluator import _rowwise_pcc, _rowwise_r2


def _finite_protein_scores(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    pcc = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    r2 = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for j in range(y_true.shape[1]):
        mask = np.isfinite(y_pred[:, j]) & np.isfinite(y_true[:, j])
        if mask.sum() < 2:
            continue
        yp = y_pred[mask, j]
        yt = y_true[mask, j]
        yp0 = yp - yp.mean()
        yt0 = yt - yt.mean()
        denom = np.sqrt(np.dot(yp0, yp0) * np.dot(yt0, yt0))
        if denom > 0:
            pcc[j] = np.dot(yp0, yt0) / denom
        sst = np.sum((yt - yt.mean()) ** 2)
        if sst > 0:
            r2[j] = 1.0 - np.sum((yp - yt) ** 2) / sst
    return pcc, r2


def official_absolute_metrics(y_pred, y_true) -> Dict:
    """Return sample-axis and protein-axis metrics with NA masks preserved."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise ValueError("Prediction and truth must be 2-D arrays with equal shape")
    sample_pcc = _rowwise_pcc(y_pred, y_true)
    sample_r2 = _rowwise_r2(y_pred, y_true)
    protein_pcc, protein_r2 = _finite_protein_scores(y_pred, y_true)
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    diff = (y_pred - y_true)[finite]
    yt = y_true[finite]
    global_r2 = None
    if yt.size and np.sum((yt - yt.mean()) ** 2) > 0:
        global_r2 = float(1.0 - np.sum((diff) ** 2) / np.sum((yt - yt.mean()) ** 2))
    return {
        "n_samples": int(y_true.shape[0]),
        "n_proteins": int(y_true.shape[1]),
        "observed_entries": int(finite.sum()),
        "sample_pcc_mean": float(np.nanmean(sample_pcc)) if np.isfinite(sample_pcc).any() else None,
        "sample_pcc_median": float(np.nanmedian(sample_pcc)) if np.isfinite(sample_pcc).any() else None,
        "sample_r2_mean": float(np.nanmean(sample_r2)) if np.isfinite(sample_r2).any() else None,
        "sample_r2_median": float(np.nanmedian(sample_r2)) if np.isfinite(sample_r2).any() else None,
        "protein_pcc_mean": float(np.nanmean(protein_pcc)) if np.isfinite(protein_pcc).any() else None,
        "protein_pcc_median": float(np.nanmedian(protein_pcc)) if np.isfinite(protein_pcc).any() else None,
        "protein_r2_mean": float(np.nanmean(protein_r2)) if np.isfinite(protein_r2).any() else None,
        "protein_r2_median": float(np.nanmedian(protein_r2)) if np.isfinite(protein_r2).any() else None,
        "global_r2": global_r2,
        "log2_rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size else None,
        "log2_mae": float(np.mean(np.abs(diff))) if diff.size else None,
    }


def high_effect_metrics(delta_pred, delta_true, threshold=1.0) -> Dict:
    """Direction and detection metrics for |true delta| > threshold."""
    delta_pred = np.asarray(delta_pred, dtype=np.float64)
    delta_true = np.asarray(delta_true, dtype=np.float64)
    mask = np.isfinite(delta_pred) & np.isfinite(delta_true)
    high = mask & (np.abs(delta_true) > threshold)
    if not high.any():
        return {"threshold": threshold, "high_effect_entries": 0, "direction_accuracy": None, "high_effect_pcc": None, "precision": None, "recall": None, "f1": None}
    signs_ok = np.sign(delta_pred[high]) == np.sign(delta_true[high])
    hp = delta_pred[high]; ht = delta_true[high]
    hp0 = hp - hp.mean(); ht0 = ht - ht.mean()
    denom = np.sqrt(np.dot(hp0, hp0) * np.dot(ht0, ht0))
    pcc = float(np.dot(hp0, ht0) / denom) if denom > 0 else None
    pred_high = mask & (np.abs(delta_pred) > threshold)
    tp = float(np.logical_and(pred_high, high).sum())
    fp = float(np.logical_and(pred_high, mask & ~high).sum())
    fn = float(np.logical_and(~pred_high, high).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": threshold,
        "high_effect_entries": int(high.sum()),
        "direction_accuracy": float(np.mean(signs_ok)),
        "high_effect_pcc": pcc,
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }

