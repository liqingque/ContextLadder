"""Common, split-aware metrics for the GOAI virtual-cell task."""

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _clean_number(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def _rowwise_pcc(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    out = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() < 2:
            continue
        pred = pred[mask]
        true = true[mask]
        pred_centered = pred - pred.mean()
        true_centered = true - true.mean()
        denom = np.sqrt(np.dot(pred_centered, pred_centered) * np.dot(true_centered, true_centered))
        if denom > 0:
            out[i] = np.dot(pred_centered, true_centered) / denom
    return out


def _rowwise_r2(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    out = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    for i, (pred, true) in enumerate(zip(y_pred, y_true)):
        mask = np.isfinite(pred) & np.isfinite(true)
        if mask.sum() < 2:
            continue
        pred = pred[mask]
        true = true[mask]
        denom = np.sum((true - true.mean()) ** 2)
        if denom > 0:
            out[i] = 1.0 - np.sum((pred - true) ** 2) / denom
    return out


def basic_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape or y_pred.ndim != 2:
        raise ValueError("Prediction and truth must be 2-D arrays with the same shape")
    finite = np.isfinite(y_pred) & np.isfinite(y_true)
    diffs = (y_pred - y_true)[finite]
    pcc = _rowwise_pcc(y_pred, y_true)
    r2 = _rowwise_r2(y_pred, y_true)
    valid_pcc = pcc[np.isfinite(pcc)]
    valid_r2 = r2[np.isfinite(r2)]
    return {
        "n_samples": int(y_true.shape[0]),
        "n_proteins": int(y_true.shape[1]),
        "valid_pcc_samples": int(valid_pcc.size),
        "valid_r2_samples": int(valid_r2.size),
        "abs_pcc": _clean_number(np.nanmean(pcc)),
        "abs_pcc_median": _clean_number(np.nanmedian(pcc)),
        "abs_pcc_p25": _clean_number(np.nanpercentile(valid_pcc, 25)) if valid_pcc.size else None,
        "abs_pcc_p75": _clean_number(np.nanpercentile(valid_pcc, 75)) if valid_pcc.size else None,
        "abs_r2": _clean_number(np.nanmean(r2)),
        "rmse": _clean_number(np.sqrt(np.nanmean(diffs ** 2))) if diffs.size else None,
        "mae": _clean_number(np.nanmean(np.abs(diffs))) if diffs.size else None,
    }


def sample_metrics(y_pred: np.ndarray, y_true: np.ndarray, sample_ids: Optional[Iterable] = None) -> pd.DataFrame:
    pcc = _rowwise_pcc(np.asarray(y_pred), np.asarray(y_true))
    r2 = _rowwise_r2(np.asarray(y_pred), np.asarray(y_true))
    errors = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    rmse = np.sqrt(np.nanmean(errors ** 2, axis=1))
    mae = np.nanmean(np.abs(errors), axis=1)
    data = {"pcc": pcc, "r2": r2, "rmse": rmse, "mae": mae}
    if sample_ids is not None:
        data = {"sample_ID": list(sample_ids), **data}
    return pd.DataFrame(data)


def split_metrics(y_pred: np.ndarray, y_true: np.ndarray, metadata: pd.DataFrame, split_col: str = "split_final") -> pd.DataFrame:
    if split_col not in metadata.columns:
        raise KeyError("Missing split column: %s" % split_col)
    rows = []
    split_values = metadata[split_col].fillna("UNKNOWN").astype(str).to_numpy()
    for label in pd.unique(split_values):
        mask = split_values == label
        row = {"split": label}
        row.update(basic_metrics(y_pred[mask], y_true[mask]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def evaluate_basic_and_splits(y_pred, y_true, metadata: pd.DataFrame, split_col: str = "split_final", sample_ids: Optional[Iterable] = None) -> Dict:
    return {
        "overall": basic_metrics(y_pred, y_true),
        "by_split": split_metrics(y_pred, y_true, metadata, split_col=split_col),
        "samples": sample_metrics(y_pred, y_true, sample_ids=sample_ids),
    }
