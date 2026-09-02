#!/usr/bin/env python
"""Compound-invariance identity check for unseen-compound scenarios.

Replaces the zero-power compound-permutation test. On every candidate whose
unseen compounds are routed to one shared token, permuting compound labels
within a context changes nothing, so the permutation null has zero variance
and its p-value is 1.0 by construction -- it is an identity, not a test.

This script measures the identity directly: inside each context block, take
every pair of rows carrying *different* unseen compounds and report how far
apart their predicted proteome vectors are. A model that emits one vector per
context gives max|delta| = 0 on every pair; a model that carries genuine
compound-specific information does not.

Reads validation metadata and candidate predictions only. No proteome matrix,
no test metadata, no test truth.
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Same granularity definitions as scripts/evaluate_official_modules.py.
# The handbook fixes the 20% module's key as strain/medium/temperature/time/
# frozen-batch; official_plate and official_source bracket the one word it
# does not bind to a column.
CTX_KEYS = {
    "official_plate": ["strain", "medium", "temperature", "time", "time_unit", "plate"],
    "official_source": ["strain", "medium", "temperature", "time", "time_unit", "source"],
    "strict": ["source", "strain", "medium", "temperature", "time", "time_unit", "instrument", "plate"],
    "bio": ["strain", "medium", "temperature", "time", "time_unit"],
}


def load_meta():
    paths = yaml.safe_load((ROOT / "configs" / "data_paths.yaml").read_text())
    mapping = yaml.safe_load((ROOT / "configs" / "field_mapping.yaml").read_text())
    meta = pd.read_csv(ROOT / paths["metadata_train_val"], low_memory=False)
    return meta, mapping


def scenario_rows(meta, mapping, granularity):
    """Validation rows split into S1 (unseen compound, seen strain) and S3 (both unseen)."""
    split = meta[mapping["split"]].astype(str)
    train = split == "train"
    comp = meta[mapping["compound"]].astype(str)
    strain = meta[mapping["strain"]].astype(str)
    train_comp = set(comp[train])
    train_strain = set(strain[train])

    controls = {str(x).strip().lower() for x in mapping["control_labels"]}
    qc = str(mapping["qc_label"]).strip().lower()
    is_treat = ~comp.str.strip().str.lower().isin(controls | {qc})

    key_cols = [mapping[k] for k in CTX_KEYS[granularity]]
    ctx = meta[key_cols].astype(str).agg("|".join, axis=1)

    out = {}
    for name, sel in (
        ("S1_chem_only", (~train) & is_treat & (~comp.isin(train_comp)) & strain.isin(train_strain)),
        ("S3_both", (~train) & is_treat & (~comp.isin(train_comp)) & (~strain.isin(train_strain))),
    ):
        idx = np.flatnonzero(sel.to_numpy())
        out[name] = pd.DataFrame({
            "sample_ID": meta.loc[sel, mapping["sample_id"]].astype(str).to_numpy(),
            "compound": comp[sel].to_numpy(),
            "ctx": ctx[sel].to_numpy(),
        }, index=idx)
    return out


def analyse(frame, rows):
    """Cross-compound pair deviations inside each context block."""
    pred = frame.reindex(index=rows["sample_ID"].to_numpy())
    if pred.isna().any().any():
        raise ValueError("candidate is missing validation sample IDs")
    mat = pred.to_numpy(dtype=np.float64)
    pos = {sid: i for i, sid in enumerate(rows["sample_ID"])}

    by_ctx = defaultdict(list)
    for sid, cpd, ctx in zip(rows["sample_ID"], rows["compound"], rows["ctx"]):
        by_ctx[ctx].append((sid, cpd))

    devs, n_ctx_multi = [], 0
    for members in by_ctx.values():
        if len({c for _, c in members}) < 2:
            continue
        n_ctx_multi += 1
        for (sa, ca), (sb, cb) in combinations(members, 2):
            if ca == cb:
                continue
            devs.append(float(np.max(np.abs(mat[pos[sa]] - mat[pos[sb]]))))
    devs = np.asarray(devs, dtype=np.float64)
    identical = int((devs == 0.0).sum())
    return {
        "n_rows": int(len(rows)),
        "n_contexts_with_multiple_compounds": n_ctx_multi,
        "n_cross_compound_pairs": int(devs.size),
        "n_pairs_bitwise_identical": identical,
        "fraction_identical": float(identical / devs.size) if devs.size else None,
        "max_deviation": float(devs.max()) if devs.size else None,
        "median_deviation": float(np.median(devs)) if devs.size else None,
        "compound_invariant": bool(devs.size and devs.max() == 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="JSON {name: prediction_val parquet}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", default="official_plate", choices=sorted(CTX_KEYS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta, mapping = load_meta()
    scen = scenario_rows(meta, mapping, args.granularity)
    candidates = json.loads(Path(args.candidates).read_text())

    results = {}
    for name, path in candidates.items():
        frame = pd.read_parquet(ROOT / path if not Path(path).is_absolute() else path)
        frame = frame.set_index(mapping["sample_id"])
        results[name] = {k: analyse(frame, rows) for k, rows in scen.items()}

    summary = {
        "granularity": args.granularity,
        "what_this_measures": (
            "Maximum absolute difference between the predicted proteome vectors of two rows that "
            "sit in the same context block but carry different unseen compounds. Zero on every pair "
            "means the model emits one vector per context and carries no compound-specific signal. "
            "This is the identity that makes a compound-permutation test degenerate: such a test has "
            "zero null variance and p=1.0 by construction, so it is reported here as an invariance "
            "check and NOT as a statistical test."
        ),
        "candidates": results,
        "no_test_truth": True,
    }
    (out / "compound_invariance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"granularity: {args.granularity}\n")
    for scen_name in ("S1_chem_only", "S3_both"):
        print(f"== {scen_name} ==")
        print(f"{'candidate':32s} {'rows':>6s} {'ctx':>5s} {'pairs':>7s} {'identical':>10s} {'max|d|':>10s}")
        for name, res in results.items():
            r = res[scen_name]
            md = "n/a" if r["max_deviation"] is None else f"{r['max_deviation']:.3e}"
            print(f"{name:32s} {r['n_rows']:6d} {r['n_contexts_with_multiple_compounds']:5d} "
                  f"{r['n_cross_compound_pairs']:7d} {r['n_pairs_bitwise_identical']:10d} {md:>10s}")
        print()
    print(f"wrote {out/'compound_invariance.json'}")


if __name__ == "__main__":
    main()
