#!/usr/bin/env python
"""R002 stage 1: build the shared DCB-40 cache (train-only).

Produces, for every matched train treatment row: the matched-control delta,
the strict and bio context ids, and the compound label.  Also produces per
(context, compound) NaN-aware sums and counts, so any leave-compound(s)-out
context mean is an O(1) subtraction later instead of a rescan.

Control pool is train-only.  Validation and test are never touched.
"""

import json
import sys
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

# strict key is the full matched-control key (identical to scripts/ssps_gate0_loco.py)
CTX_STRICT = ["source", "strain", "medium", "temperature", "time", "time_unit",
              "instrument", "plate"]
CTX_BIO = ["strain", "medium", "temperature", "time", "time_unit"]
OUT = ROOT / "outputs/dcb40/cache"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text())
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text())
    meta = load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, all_proteins = align_metadata_proteome(meta, prot)
    raw = finite_float_matrix(prot, all_proteins)
    keep, _, keep_mask = apply_official_protein_filter(meta, raw, all_proteins, mapping,
                                                       threshold=0.80)
    y = to_log2_proteome(raw[:, keep_mask]).astype(np.float64)

    split = meta[mapping["split"]].astype(str)
    train_mask = split.eq("train").to_numpy()
    ccol = mapping["compound"]

    matches = match_controls(meta, train_mask, mapping, control_pool_mask=train_mask)
    matched = matches[matches["matched"]]
    print(f"train rows {train_mask.sum()}  matched {len(matched)}")

    deltas, compounds, ctx_s, ctx_b = [], [], [], []
    for _, r in matched.iterrows():
        t = int(r["treat_row"])
        controls = [int(x) for x in str(r["control_rows"]).split(",") if x]
        with np.errstate(invalid="ignore"):
            cmean = np.nanmean(y[controls], axis=0)
        deltas.append(y[t] - cmean)
        compounds.append(str(meta.iloc[t][ccol]))
        ctx_s.append(tuple(str(meta.iloc[t][mapping[k]]) for k in CTX_STRICT
                           if mapping.get(k) in meta.columns))
        ctx_b.append(tuple(str(meta.iloc[t][mapping[k]]) for k in CTX_BIO
                           if mapping.get(k) in meta.columns))

    D = np.vstack(deltas)                       # (n_matched, 4422) with NaN
    comp = np.array(compounds)
    uc, comp_id = np.unique(comp, return_inverse=True)
    us, ctxs_id = np.unique(np.array([hash(c) for c in ctx_s]), return_inverse=True)
    ub, ctxb_id = np.unique(np.array([hash(c) for c in ctx_b]), return_inverse=True)

    print(f"deltas {D.shape}  compounds {len(uc)}  ctx_strict {len(us)}  ctx_bio {len(ub)}")
    print(f"observed fraction {np.isfinite(D).mean():.4f}")

    np.savez_compressed(
        OUT / "cache.npz",
        deltas=D.astype(np.float32),           # float32 halves the file; math stays f64
        compound_id=comp_id, ctx_strict_id=ctxs_id, ctx_bio_id=ctxb_id,
        compounds=uc, proteins=np.array(keep),
    )
    (OUT / "cache_manifest.json").write_text(json.dumps({
        "n_matched_rows": int(len(D)), "n_proteins": int(D.shape[1]),
        "n_compounds": int(len(uc)), "n_ctx_strict": int(len(us)),
        "n_ctx_bio": int(len(ub)),
        "observed_fraction": float(np.isfinite(D).mean()),
        "control_pool": "train only", "split_used": "split_final=train",
        "no_val_or_test_read": True,
        "rows_per_compound": {c: int((comp == c).sum()) for c in uc},
    }, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'cache.npz'}")


if __name__ == "__main__":
    main()
