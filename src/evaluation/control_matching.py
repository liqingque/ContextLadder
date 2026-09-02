"""Post-hoc matched-control utilities; controls are never used to fit models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluator import basic_metrics


def match_controls(metadata: pd.DataFrame, eval_mask: np.ndarray, field_mapping: Dict, control_labels: Optional[List[str]] = None, control_pool_mask: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Match treatment rows to same-context controls in the supplied table."""
    compound_col = field_mapping["compound"]
    controls = {str(x).strip().lower() for x in (control_labels or ["DMSO", "Water"])}
    context_keys = []
    for key in ("source", "strain", "medium", "temperature", "time", "time_unit", "instrument", "plate"):
        col = field_mapping.get(key)
        if col and col in metadata.columns:
            context_keys.append(col)
    sample_col = field_mapping["sample_id"]
    work = metadata.reset_index(drop=True).copy()
    work["__row"] = np.arange(len(work))
    work["__compound_norm"] = work[compound_col].fillna("").astype(str).str.strip().str.lower()
    dataset_col = field_mapping.get("source")
    if dataset_col and dataset_col in work.columns:
        work["__dataset"] = work[dataset_col].fillna("<NA>").astype(str)
    else:
        work["__dataset"] = "all"
    key_cols = ["__dataset"] + context_keys
    for col in key_cols:
        work[col] = work[col].fillna("<NA>").astype(str)
    if control_pool_mask is None:
        control_pool_mask = np.asarray(eval_mask, dtype=bool)
    control_rows = work[np.asarray(control_pool_mask, dtype=bool) & work["__compound_norm"].isin(controls)]
    control_groups = {}
    for key, group in control_rows.groupby(key_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        by_label = {}
        for label, label_group in group.groupby("__compound_norm", sort=False):
            by_label[label] = label_group["__row"].tolist()
        control_groups[key] = by_label
    rows = []
    for row_idx in np.flatnonzero(np.asarray(eval_mask)):
        row = work.iloc[row_idx]
        if row["__compound_norm"] in controls or row["__compound_norm"] == "quality control":
            continue
        key = tuple(row[c] for c in key_cols)
        candidates = control_groups.get(key, {})
        if not candidates:
            rows.append({"treat_row": int(row_idx), "sample_ID": str(row[sample_col]), "matched": False, "n_controls": 0})
            continue
        preferred = [x for x in ("dmso", "water") if x in candidates]
        label = preferred[0] if preferred else sorted(candidates)[0]
        controls_for_row = candidates[label]
        rows.append({
            "treat_row": int(row_idx), "sample_ID": str(row[sample_col]), "matched": True,
            "control_type": label, "n_controls": len(controls_for_row),
            "control_rows": ",".join(str(x) for x in controls_for_row),
            "match_key": "|".join(str(row[c]) for c in key_cols),
        })
    return pd.DataFrame(rows)


def matched_fc(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    metadata: pd.DataFrame,
    eval_mask: np.ndarray,
    field_mapping: Dict,
    control_pool_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """Evaluate matched-control FC with an explicit, leakage-safe pool.

    Callers evaluating a held-out split should pass ``control_pool_mask`` for
    the legal training controls (or train+evaluation controls when reproducing
    the official post-hoc metric).  The old behavior remains the default for
    backwards compatibility with split-local diagnostics.
    """
    matches = match_controls(
        metadata,
        eval_mask,
        field_mapping,
        control_pool_mask=control_pool_mask,
    )
    if matches.empty:
        return {"n_treatments": 0, "n_matched": 0, "coverage": None, "fc_pcc": None}, matches
    delta_pred, delta_true = [], []
    for _, row in matches[matches["matched"]].iterrows():
        treat_idx = int(row["treat_row"])
        control_indices = [int(x) for x in str(row["control_rows"]).split(",") if x]
        control_true = np.nanmean(y_true[control_indices], axis=0)
        delta_pred.append(y_pred[treat_idx] - control_true)
        delta_true.append(y_true[treat_idx] - control_true)
    if not delta_pred:
        metrics = {"n_treatments": int(len(matches)), "n_matched": 0, "coverage": 0.0, "fc_pcc": None}
        return metrics, matches
    metrics = basic_metrics(np.asarray(delta_pred), np.asarray(delta_true))
    return {
        "n_treatments": int(len(matches)), "n_matched": int(len(delta_pred)),
        "coverage": float(len(delta_pred) / len(matches)), "fc_pcc": metrics["abs_pcc"],
        "fc_pcc_median": metrics["abs_pcc_median"],
    }, matches
