#!/usr/bin/env python
"""E2: shared utilities for the train-internal four-axis grouped pseudo-OOD protocol.

This module is READ-ONLY with respect to scripts/run_hcce.py and
scripts/a2b_train_variants.py: it imports from them but never edits them, so
the frozen runs/final/ results and any other agent's concurrent work on those
two files stay reproducible.

Everything here operates strictly on split_final == "train" rows (5,920 of
them). Every fold recomputes categorical vocabularies and normalisation
statistics from the fold-train rows only (this happens naturally because we
call the untouched HCCEMetaEncoder.fit()/fit_model_variant() with fold-local
fit_indices); the held-out entity's rows never contribute a category id other
than the reserved <NA>/unknown token (index 0) once they are mapped through a
fold-train-only vocabulary.

Four axes:
  plate    - Yeast_cell_plate,   144 entities -> grouped into ~12 folds (LPT
             bin-packing balances the very different plate sizes; a full
             144-fold LOPO sweep would take well over an hour per variant).
  strain   - Strains,              4 entities -> complete leave-one-strain-out
             (4 folds); flagged everywhere as a low-power axis.
  compound - perturbation_no_concentration, 40 entities -> grouped into 10
             folds (LPT bin-packing).
  time     - pert_time,            6 entities -> complete leave-one-time-out
             (6 folds); each held-out block is a single sorted time value, so
             it is trivially a contiguous block by construction.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2b_train_variants import fit_model_variant, load_run_hcce  # noqa: E402
from src.evaluation.control_matching import matched_fc  # noqa: E402
from src.evaluation.evaluator import basic_metrics  # noqa: E402

MANIFEST_SEED = 20260902  # deterministic; only used for bootstrap RNGs, never for the split itself

AXES = {
    "plate": {"mapping_key": "plate", "n_folds": 12, "mode": "grouped"},
    "strain": {"mapping_key": "strain", "n_folds": 4, "mode": "complete"},
    "compound": {"mapping_key": "compound", "n_folds": 10, "mode": "grouped"},
    "time": {"mapping_key": "time", "n_folds": 6, "mode": "complete"},
}


# --------------------------------------------------------------------------- #
# universe loading (train-only rows, official preprocessing, byte-identical
# to the pipeline in scripts/a2b_train_variants.py::main)
# --------------------------------------------------------------------------- #
def load_universe():
    rh = load_run_hcce()
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    meta = rh.load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = rh.load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, all_proteins = rh.align_metadata_proteome(meta, prot)
    raw = rh.finite_float_matrix(prot, all_proteins)
    proteins, _, keep_mask = rh.apply_official_protein_filter(meta, raw, all_proteins, mapping, threshold=0.80)
    raw = raw[:, keep_mask]
    y = rh.to_log2_proteome(raw).astype(np.float64)
    train_mask = meta[mapping["split"]].astype(str).eq("train").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    if len(train_indices) != 5920:
        raise AssertionError(f"expected 5920 split_final=='train' rows, found {len(train_indices)}")
    meta_train = meta.iloc[train_indices].reset_index(drop=True)
    y_train = y[train_indices]
    return rh, meta_train, y_train, proteins, mapping


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def lpt_partition(counts: Dict[str, int], k: int) -> List[List[str]]:
    """Longest-processing-time greedy bin packing: balances total row count
    per bin. Deterministic: entities sorted by (descending count, name);
    ties in bin total broken by lowest bin index."""
    if k > len(counts):
        raise ValueError(f"cannot partition {len(counts)} entities into {k} non-empty bins")
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    bins: List[List[str]] = [[] for _ in range(k)]
    totals = [0] * k
    for name, cnt in items:
        i = min(range(k), key=lambda b: (totals[b], b))
        bins[i].append(name)
        totals[i] += cnt
    return bins


def build_axis_groups(meta_train: pd.DataFrame, axis: str, mapping: dict) -> Tuple[str, List[List[str]]]:
    spec = AXES[axis]
    field_col = mapping[spec["mapping_key"]]
    values = meta_train[field_col].astype(str)
    counts = values.value_counts().to_dict()
    n_entities = len(counts)
    n_folds = spec["n_folds"]
    if spec["mode"] == "complete":
        if n_entities != n_folds:
            raise AssertionError(
                f"axis={axis}: expected exactly {n_folds} entities for a complete "
                f"leave-one-out sweep, found {n_entities} ({sorted(counts)})"
            )
        # one entity per fold; sort for a stable, reportable fold order
        if axis == "time":
            order = sorted(counts, key=lambda v: float(v))
        else:
            order = sorted(counts)
        groups = [[e] for e in order]
    else:
        groups = lpt_partition(counts, n_folds)
        groups = [sorted(g) for g in groups]
    return field_col, groups


def fold_row_indices(meta_train: pd.DataFrame, field_col: str, group_entities: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    values = meta_train[field_col].astype(str).to_numpy()
    test_mask = np.isin(values, group_entities)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    return train_idx, test_idx


def assert_no_leakage(meta_train: pd.DataFrame, field_col: str, train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Hard leakage gate: raises AssertionError (aborts the caller) rather
    than warning. Returns the assertion record on success."""
    n_total = len(meta_train)
    values = meta_train[field_col].astype(str).to_numpy()
    train_entities = set(values[train_idx])
    test_entities = set(values[test_idx])
    entity_overlap = sorted(train_entities & test_entities)
    idx_overlap = np.intersect1d(train_idx, test_idx)
    union_ok = len(train_idx) + len(test_idx) == n_total
    disjoint_entities = len(entity_overlap) == 0
    disjoint_rows = len(idx_overlap) == 0
    non_empty = len(train_idx) > 0 and len(test_idx) > 0
    record = {
        "field": field_col,
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "entity_intersection_empty": bool(disjoint_entities),
        "entity_intersection": entity_overlap,
        "row_intersection_empty": bool(disjoint_rows),
        "train_test_union_covers_all_rows": bool(union_ok),
        "both_sides_non_empty": bool(non_empty),
        "pass": bool(disjoint_entities and disjoint_rows and union_ok and non_empty),
    }
    if not record["pass"]:
        raise AssertionError(f"LEAKAGE ASSERTION FAILED for field={field_col}: {record}")
    return record


# --------------------------------------------------------------------------- #
# metrics: entity-macro + cluster bootstrap
# --------------------------------------------------------------------------- #
def per_entity_metrics(pred: np.ndarray, true: np.ndarray, entity_values: np.ndarray, sample_ids=None) -> pd.DataFrame:
    rows = []
    for entity in sorted(set(entity_values)):
        mask = entity_values == entity
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue
        m = basic_metrics(pred[mask], true[mask])
        rows.append({
            "entity": entity, "n_rows": n_rows,
            "sample_pcc": m["abs_pcc"], "sample_pcc_median": m["abs_pcc_median"],
            "log2_rmse": m["rmse"], "log2_mae": m["mae"], "r2": m["abs_r2"],
        })
    return pd.DataFrame(rows)


def entity_macro(per_entity_df: pd.DataFrame, metric_col: str) -> float:
    vals = per_entity_df[metric_col].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def cluster_bootstrap_ci(per_entity_df: pd.DataFrame, metric_col: str, n_boot: int = 2000, seed: int = MANIFEST_SEED) -> dict:
    """Entity-cluster bootstrap: resample entities (not rows) with
    replacement, macro-average the metric each draw."""
    vals = per_entity_df[metric_col].dropna().to_numpy(dtype=float)
    n = len(vals)
    if n == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n_entities": 0, "n_boot": n_boot}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        boot_means[b] = float(np.mean(vals[draw]))
    return {
        "mean": float(np.mean(vals)),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "n_entities": int(n),
        "n_boot": int(n_boot),
    }


def matched_fc_for_fold(y_pred_test, y_true_full, meta_train, mapping, train_idx, test_idx):
    """Matched-control FC using only fold-train rows as the legal control pool.

    y_pred_test are predictions for test_idx rows only; the rest of the
    full-length prediction array is filled with NaN and excluded from
    evaluation by eval_mask so it can never leak into a metric.
    """
    n = len(meta_train)
    n_proteins = y_true_full.shape[1]
    y_pred_full = np.full((n, n_proteins), np.nan, dtype=np.float64)
    y_pred_full[test_idx] = y_pred_test
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[test_idx] = True
    control_pool_mask = np.zeros(n, dtype=bool)
    control_pool_mask[train_idx] = True
    fc, matches = matched_fc(y_pred_full, y_true_full, meta_train, eval_mask, mapping, control_pool_mask=control_pool_mask)
    return fc


# --------------------------------------------------------------------------- #
# reusable candidate adapters: HCCE variants via scripts/a2b_train_variants.py
# --------------------------------------------------------------------------- #
def build_hcce_variant_functions(rh, mapping, variant: str, embedding_dim: int = 64,
                                  mask_p: float = 0.25, strain_mask_p=None, denoise_tau: float = 1.0):
    """Factory: returns (train_fn, predict_fn) closing over a fixed HCCE
    variant name, matching the reusable evaluate_four_axis contract in
    scripts/evaluate_grouped_ood.py.

    train_fn(meta, y, fit_idx, mapping, seed, epochs, device) -> state
    predict_fn(state, meta_subset, device) -> np.ndarray predictions (mean50
      of the legacy/FiLM experts, matching the primary metric reported by
      scripts/a2b_train_variants.py::main).
    """
    def train_fn(meta, y, fit_idx, mapping_, seed, epochs, device):
        model, encoder, target_mean, target_std, history = fit_model_variant(
            rh, meta, y, fit_idx, mapping_, seed, epochs, device, embedding_dim, variant,
            mask_p=mask_p, strain_mask_p=strain_mask_p, denoise_tau=denoise_tau,
        )
        return {"model": model, "encoder": encoder, "target_mean": target_mean,
                "target_std": target_std, "history": history}

    def predict_fn(state, meta_subset, device):
        concat, film, legacy = rh.predict_model(
            state["model"], state["encoder"], state["target_mean"], state["target_std"], meta_subset, device,
        )
        return 0.5 * legacy + 0.5 * film

    return train_fn, predict_fn
