"""Train-only protein precision weights for RAP-Proteome.

The estimator follows the duplicate-condition construction used by the
metric-structure diagnosis: rows are grouped by source and biological
condition, and pairwise differences estimate single-observation noise.  No
validation or test rows are touched when ``fit_indices`` is supplied.
"""

from itertools import combinations

import numpy as np


def _norm(value):
    return "<NA>" if value is None else str(value)


def estimate_precision_weights(metadata, y, fit_indices, mapping, tau=0.1, clip=(0.25, 20.0)):
    """Estimate normalized per-protein precision weights from fit rows.

    ``sigma2`` is the estimated variance of one observation, obtained as half
    the mean squared pairwise difference within exact duplicate conditions.
    The returned weights have mean one over proteins with finite estimates.
    """
    fit_indices = np.asarray(fit_indices, dtype=int)
    fit_set = set(int(x) for x in fit_indices)
    source = mapping["source"]
    cond_cols = [
        mapping["strain"], mapping["compound"], mapping["medium"],
        mapping["temperature"], mapping["time"],
    ]
    keys = []
    for idx in fit_indices:
        row = metadata.iloc[int(idx)]
        key = tuple([_norm(row[source])] + [_norm(row[col]) for col in cond_cols])
        keys.append((key, int(idx)))
    groups = {}
    for key, idx in keys:
        groups.setdefault(key, []).append(idx)

    y = np.asarray(y, dtype=np.float64)
    n_proteins = y.shape[1]
    sq_sum = np.zeros(n_proteins, dtype=np.float64)
    pair_count = np.zeros(n_proteins, dtype=np.float64)
    duplicate_groups = 0
    duplicate_pairs = 0
    for rows in groups.values():
        rows = [x for x in rows if x in fit_set]
        if len(rows) < 2:
            continue
        duplicate_groups += 1
        for a, b in combinations(rows, 2):
            diff = y[a] - y[b]
            valid = np.isfinite(diff)
            sq_sum[valid] += diff[valid] ** 2
            pair_count[valid] += 1.0
            duplicate_pairs += 1

    sigma2 = np.full(n_proteins, np.nan, dtype=np.float64)
    valid = pair_count > 0
    sigma2[valid] = sq_sum[valid] / (2.0 * pair_count[valid])
    fallback = np.nanmedian(sigma2[valid]) if np.any(valid) else np.nan
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = float(np.nanvar(y[fit_indices])) if len(fit_indices) else 1.0
    fallback = max(fallback, 1e-6)
    sigma2 = np.where(np.isfinite(sigma2) & (sigma2 > 0), sigma2, fallback)
    raw = 1.0 / (sigma2 + float(tau) ** 2)
    finite_raw = np.isfinite(raw) & (raw > 0)
    center = float(np.nanmean(raw[finite_raw])) if np.any(finite_raw) else 1.0
    weights = raw / max(center, 1e-12)
    weights = np.clip(weights, float(clip[0]), float(clip[1])).astype(np.float32)
    summary = {
        "tau": float(tau),
        "duplicate_groups": int(duplicate_groups),
        "duplicate_pairs": int(duplicate_pairs),
        "proteins_with_pair_estimates": int(valid.sum()),
        "sigma2_median": float(np.nanmedian(sigma2)),
        "sigma2_p25": float(np.nanpercentile(sigma2, 25)),
        "sigma2_p75": float(np.nanpercentile(sigma2, 75)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "fit_rows": int(len(fit_indices)),
        "no_validation_or_test_truth": True,
    }
    return weights, summary
