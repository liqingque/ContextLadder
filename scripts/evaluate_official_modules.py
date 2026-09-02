#!/usr/bin/env python
"""M0: official-caliber module evaluator for frozen-validation predictions.

Fixes three caliber problems in the previous evaluation path:
  1. control pool is `train | val` (official controls ship with the eval set),
     not train-only -- the train-only pool silently drops every unseen-strain
     and both-unseen matched row;
  2. scenarios are stratified by "is the entity in the train split", giving
     seen / S1 (chem_only) / S2 (strain_only) / S3 (both);
  3. the context mean for the context-residual module is reported at several
     granularities. The handbook fixes the key as "same strain/medium/
     temperature/time/frozen-batch"; "frozen-batch" is the one word it does not
     bind to a column, so official_plate and official_source bracket it, with
     the historical strict/bio pair kept for continuity.

Reports the six official modules per stratum, permutation nulls, and paired
cluster-bootstrap CIs against a baseline candidate. Test truth is never read.
"""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_hcce import apply_official_protein_filter
from src.data.io import (align_metadata_proteome, finite_float_matrix,
                         load_metadata, load_proteome, to_log2_proteome)
from src.evaluation.control_matching import match_controls

# Context-key granularities for mu_ctx. The handbook defines the module as
# removing the response shared within one 菌株/培养基/温度/时间/冻结批次 block --
# note it lists neither 来源 nor 仪器 here, unlike the 25% FC control-matching
# key, which does. Only 冻结批次 is unbound to a column, so the two "official_*"
# readings bracket it; "strict"/"bio" are the historical pair, kept so earlier
# numbers stay comparable.
CTX_KEYS = {
    "official_plate": ["strain", "medium", "temperature", "time", "time_unit", "plate"],
    "official_source": ["strain", "medium", "temperature", "time", "time_unit", "source"],
    "strict": ["source", "strain", "medium", "temperature", "time", "time_unit", "instrument", "plate"],
    "bio": ["strain", "medium", "temperature", "time", "time_unit"],
}

_SHARED = {}


def masked_rowwise_pcc(a, b, mask):
    """Per-row Pearson correlation over the observed entries of each row."""
    m = mask.astype(np.float64)
    # zero the unobserved entries before summing: NaN * 0 would poison the sums
    a = np.where(mask, a, 0.0)
    b = np.where(mask, b, 0.0)
    n = m.sum(1)
    sx = a.sum(1)
    sy = b.sum(1)
    sxx = (a * a).sum(1)
    syy = (b * b).sum(1)
    sxy = (a * b).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        r = cov / np.sqrt(vx * vy)
    r[(n < 3) | (vx <= 0) | (vy <= 0)] = np.nan
    return r


def build_index(meta, y, mapping):
    """Matched-control deltas for val (pool = train|val) and train (pool = train)."""
    split = meta[mapping["split"]].astype(str)
    val_mask = ~split.eq("train").to_numpy()
    train_mask = ~val_mask
    scol, ccol = mapping["strain"], mapping["compound"]
    train_strains = set(meta.loc[train_mask, scol].astype(str))
    train_compounds = set(meta.loc[train_mask, ccol].astype(str))

    def rows_for(eval_mask, pool_mask):
        matches = match_controls(meta, eval_mask, mapping, control_pool_mask=pool_mask)
        out = []
        if matches.empty:
            return out
        for _, r in matches[matches["matched"]].iterrows():
            t = int(r["treat_row"])
            controls = [int(x) for x in str(r["control_rows"]).split(",") if x]
            out.append({
                "row": t,
                "sample_ID": str(r["sample_ID"]),
                "compound": str(meta.iloc[t][ccol]),
                "strain": str(meta.iloc[t][scol]),
                "control": np.nanmean(y[controls], axis=0),
                "ctx": {name: tuple(str(meta.iloc[t][mapping[k]]) for k in ks if mapping.get(k) in meta.columns)
                        for name, ks in CTX_KEYS.items()},
            })
        return out

    val_rows = rows_for(val_mask, train_mask | val_mask)
    train_rows = rows_for(train_mask, train_mask)
    for r in val_rows:
        us = r["strain"] not in train_strains
        uc = r["compound"] not in train_compounds
        r["scenario"] = "S3_both" if (us and uc) else ("S2_strain_only" if us else ("S1_chem_only" if uc else "seen"))
    return val_rows, train_rows, val_mask


def module_scores(dp, dt, obs, rows, ctx_mean, drug_mean, rng, n_perm=20):
    """Six official modules + permutation nulls, given per-row delta matrices."""
    res = {}
    scen = np.array([r["scenario"] for r in rows])

    def residualize(sel, mu):
        """Subtract a context/drug mean and re-derive the observation mask.

        mu can itself be NaN at proteins that no train row in the group
        observed; those entries must leave the mask rather than poison the row.
        """
        rp = dp[sel] - mu
        rt = dt[sel] - mu
        ro = obs[sel] & np.isfinite(rp) & np.isfinite(rt)
        return np.nan_to_num(rp), np.nan_to_num(rt), ro

    # --- 25% matched-control FC, overall and per scenario ---
    fc_rows = masked_rowwise_pcc(dp, dt, obs & np.isfinite(dp))
    res["fc"] = {"ALL": {"pcc": float(np.nanmean(fc_rows)), "n": int(np.isfinite(fc_rows).sum())}}
    for s in ("seen", "S1_chem_only", "S2_strain_only", "S3_both"):
        sel = scen == s
        if sel.sum():
            res["fc"][s] = {"pcc": float(np.nanmean(fc_rows[sel])), "n": int(sel.sum())}

    # --- 20% context-mean residual (S1), both granularities ---
    res["context_residual"] = {}
    for gran in CTX_KEYS:
        sel = np.array([r["scenario"] == "S1_chem_only" and r["ctx"][gran] in ctx_mean[gran] for r in rows])
        if not sel.any():
            res["context_residual"][gran] = {"n": 0}
            continue
        mu = np.stack([ctx_mean[gran][r["ctx"][gran]] for r, k in zip(rows, sel) if k])
        rp, rt, ro = residualize(sel, mu)
        per_row = masked_rowwise_pcc(rp, rt, ro)
        # compound-permutation null: swap the model's prediction with another
        # unseen compound measured in the same context, keeping this row's
        # control anchor and target.
        idx = np.flatnonzero(sel)
        by_ctx = defaultdict(list)
        for j, i in enumerate(idx):
            by_ctx[rows[i]["ctx"][gran]].append(j)
        nulls = []
        for _ in range(n_perm):
            src = np.arange(len(idx))
            for _ctx, members in by_ctx.items():
                if len(members) < 2:
                    continue
                comps = np.array([rows[idx[j]]["compound"] for j in members])
                for pos, j in enumerate(members):
                    alt = [m for m, c in zip(members, comps) if c != comps[pos]]
                    if alt:
                        src[j] = rng.choice(alt)
            nulls.append(np.nanmean(masked_rowwise_pcc(rp[src], rt, ro)))
        res["context_residual"][gran] = {
            "pcc": float(np.nanmean(per_row)), "n": int(sel.sum()),
            "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
            "gain_over_null": float(np.nanmean(per_row) - np.mean(nulls)),
            "sigma_ratio": float(np.nanstd(rp[ro]) / np.nanstd(rt[ro])),
            "per_row": per_row, "row_index": idx,
        }

    # --- 20% drug-mean residual (S2) ---
    sel = np.array([r["scenario"] == "S2_strain_only" and r["compound"] in drug_mean for r in rows])
    if sel.any():
        mu = np.stack([drug_mean[r["compound"]] for r, k in zip(rows, sel) if k])
        rp, rt, ro = residualize(sel, mu)
        per_row = masked_rowwise_pcc(rp, rt, ro)
        clusters = sorted({rows[i]["strain"] for i in np.flatnonzero(sel)})
        res["drug_residual"] = {
            "pcc": float(np.nanmean(per_row)), "n": int(sel.sum()),
            "n_clusters": len(clusters), "clusters": clusters,
            "undecidable_single_cluster": len(clusters) < 2,
            "sigma_ratio": float(np.nanstd(rp[ro]) / np.nanstd(rt[ro])),
            "per_row": per_row, "row_index": np.flatnonzero(sel),
        }
    else:
        res["drug_residual"] = {"n": 0}

    # --- 5% high-effect / DEP, on |delta_true| > 1 ---
    he = obs & np.isfinite(dp) & (np.abs(dt) > 1.0)
    if he.sum():
        res["dep"] = {
            "n_entries": int(he.sum()),
            "direction_accuracy": float((np.sign(dp[he]) == np.sign(dt[he])).mean()),
            "high_effect_pcc": float(np.corrcoef(dp[he], dt[he])[0, 1]),
        }
    return res


def evaluate_candidate(item):
    name, path = item
    sh = _SHARED
    meta, mapping, keep = sh["meta"], sh["mapping"], sh["keep"]
    rows, dt, obs, val_mask = sh["rows"], sh["dt"], sh["obs"], sh["val_mask"]
    y = sh["y"]

    frame = pd.read_parquet(path)
    idc = mapping["sample_id"]
    if idc in frame.columns:
        frame = frame.set_index(idc)
    missing = [c for c in keep if c not in frame.columns]
    if missing:
        raise ValueError(f"{name}: prediction is missing {len(missing)} contract proteins")
    val_ids = meta.loc[val_mask, idc].astype(str).values
    pred_val = frame.reindex(index=val_ids)[keep].to_numpy(dtype=np.float64)
    if not np.isfinite(pred_val).all():
        raise ValueError(f"{name}: prediction contains non-finite values")

    full = np.full(y.shape, np.nan, dtype=np.float64)
    full[val_mask] = pred_val
    dp = np.stack([full[r["row"]] - r["control"] for r in rows])

    # 20% absolute fidelity, on all val rows
    y_val = y[val_mask]
    ov = np.isfinite(y_val)
    abs_pcc = masked_rowwise_pcc(pred_val, y_val, ov)
    err = np.where(ov, pred_val - y_val, np.nan)
    absolute = {
        "sample_pcc_mean": float(np.nanmean(abs_pcc)),
        "log2_rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "n_samples": int(len(y_val)),
    }

    rng = np.random.default_rng(20260816)
    mod = module_scores(dp, dt, obs, rows, sh["ctx_mean"], sh["drug_mean"], rng)
    per_row_keep = {f"s1_{g}": (mod["context_residual"][g].pop("per_row", None),
                                mod["context_residual"][g].pop("row_index", None))
                    for g in CTX_KEYS}
    per_row_keep["s2"] = (mod["drug_residual"].pop("per_row", None),
                          mod["drug_residual"].pop("row_index", None))
    return name, {"prediction": str(path), "absolute": absolute, **mod}, per_row_keep


def init_shared(**kwargs):
    _SHARED.update(kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="JSON file: {name: path_to_prediction_val.parquet}. Preregister and freeze.")
    ap.add_argument("--baseline", required=True, help="candidate name used as the paired-bootstrap reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(Path(args.candidates).read_text())
    if args.baseline not in candidates:
        raise SystemExit(f"baseline {args.baseline} not in candidate list")

    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text())
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text())
    meta = load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, all_proteins = align_metadata_proteome(meta, prot)
    raw = finite_float_matrix(prot, all_proteins)
    keep, _, keep_mask = apply_official_protein_filter(meta, raw, all_proteins, mapping, threshold=0.80)
    y = to_log2_proteome(raw[:, keep_mask]).astype(np.float64)

    rows, train_rows, val_mask = build_index(meta, y, mapping)
    dt = np.stack([y[r["row"]] - r["control"] for r in rows])
    obs = np.isfinite(dt)

    ctx_mean = {}
    for gran in CTX_KEYS:
        g = defaultdict(list)
        for r in train_rows:
            g[r["ctx"][gran]].append(y[r["row"]] - r["control"])
        ctx_mean[gran] = {k: np.nanmean(np.stack(v), axis=0) for k, v in g.items()}
    gd = defaultdict(list)
    for r in train_rows:
        gd[r["compound"]].append(y[r["row"]] - r["control"])
    drug_mean = {k: np.nanmean(np.stack(v), axis=0) for k, v in gd.items()}

    shared = dict(meta=meta, mapping=mapping, keep=keep, rows=rows, dt=dt, obs=obs,
                  val_mask=val_mask, y=y, ctx_mean=ctx_mean, drug_mean=drug_mean)
    init_shared(**shared)

    results, per_row = {}, {}
    items = list(candidates.items())
    if args.workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(items))) as ex:
            for name, res, pr in ex.map(evaluate_candidate, items):
                results[name], per_row[name] = res, pr
    else:
        for it in items:
            name, res, pr = evaluate_candidate(it)
            results[name], per_row[name] = res, pr

    # --- paired cluster bootstrap of the module difference vs the baseline ---
    rng = np.random.default_rng(20260816)
    clusters_by_gran, draws_by_gran = {}, {}
    for gran in CTX_KEYS:
        cl = defaultdict(list)
        for j, i in enumerate(per_row[args.baseline][f"s1_{gran}"][1]):
            cl[rows[i]["compound"]].append(j)
        clusters_by_gran[gran] = cl
        nm = sorted(cl)
        draws_by_gran[gran] = (nm, [rng.choice(len(nm), len(nm), replace=True)
                                    for _ in range(args.bootstrap)])
    clusters_s1 = clusters_by_gran["strict"]
    names_s1 = sorted(clusters_s1)

    paired = {}
    for name in candidates:
        if name == args.baseline:
            continue
        entry = {}
        for gran in CTX_KEYS:
            key, label = f"s1_{gran}", f"S1_context_residual_{gran}"
            cl = clusters_by_gran[gran]
            nm, draws = draws_by_gran[gran]
            a, ia = per_row[args.baseline][key]
            b, ib = per_row[name][key]
            if a is None or b is None or not np.array_equal(ia, ib):
                entry[label] = {"error": "row sets differ; not comparable"}
                continue
            diff = b - a
            boots = []
            for pick in draws:
                idx = np.concatenate([cl[nm[j]] for j in pick])
                boots.append(np.nanmean(diff[idx]))
            entry[label] = {
                "delta": float(np.nanmean(diff)),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "sd": float(np.std(boots)),
                "ci_excludes_zero": bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0),
            }
        a, ia = per_row[args.baseline]["s2"]
        b, ib = per_row[name]["s2"]
        if a is not None and b is not None and np.array_equal(ia, ib):
            entry["S2_drug_residual"] = {
                "delta": float(np.nanmean(b - a)),
                "note": "single unseen-strain cluster in validation; not decidable by cluster bootstrap",
            }
        paired[name] = entry

    # per-cluster (per unseen compound) S1 table
    per_cluster = {}
    for name in candidates:
        arr, idx = per_row[name]["s1_strict"]
        per_cluster[name] = {c: float(np.nanmean(arr[js])) for c, js in clusters_s1.items()}

    summary = {
        "control_pool": "train|val (official caliber)",
        "n_val_matched": len(rows),
        "scenario_counts": {s: int(sum(1 for r in rows if r["scenario"] == s))
                            for s in ("seen", "S1_chem_only", "S2_strain_only", "S3_both")},
        "s1_clusters": names_s1,
        "baseline": args.baseline,
        "candidates": results,
        "paired_cluster_bootstrap_vs_baseline": paired,
        "s1_per_compound": per_cluster,
        "no_test_truth": True,
    }
    (out / "module_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates"}, ensure_ascii=False,
                     indent=2, default=float))
    for name, res in results.items():
        print(f"\n== {name} ==")
        print(f"  abs sample-PCC {res['absolute']['sample_pcc_mean']:.6f}  RMSE {res['absolute']['log2_rmse']:.6f}")
        print(f"  FC  ALL {res['fc']['ALL']['pcc']:.6f}  " +
              "  ".join(f"{s.split('_')[0]} {res['fc'][s]['pcc']:.4f}" for s in res["fc"] if s != "ALL"))
        for gran in CTX_KEYS:
            c = res["context_residual"][gran]
            if not c.get("n"):
                continue
            print(f"  S1 ctx-resid {gran:15s} {c['pcc']:.6f} "
                  f"(n {c['n']}, null {c['null_mean']:.6f}, gain {c['gain_over_null']:+.6f})")
        dr = res["drug_residual"]
        if dr.get("n"):
            print(f"  S2 drug-resid {dr['pcc']:.6f} (clusters {dr['n_clusters']})")


if __name__ == "__main__":
    main()
