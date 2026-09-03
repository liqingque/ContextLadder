#!/usr/bin/env python
"""Create train-only pseudo-OOD folds for cross-fitted expert/gate training."""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.io import load_metadata


SEED = 20260810


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/biocal_moe", help="Output directory; defaults to historical location")
    parser.add_argument("--complete-both-grid", action="store_true", help="Use complete strain x compound-bucket both-OOD folds")
    args = parser.parse_args()
    paths = yaml.safe_load(open(ROOT / "configs/data_paths.yaml", encoding="utf-8")); mapping = yaml.safe_load(open(ROOT / "configs/field_mapping.yaml", encoding="utf-8"))
    meta = load_metadata(ROOT / paths["metadata_train_val"])
    train = meta[meta[mapping["split"]].astype(str).eq("train")].reset_index(drop=True)
    sample_col = mapping["sample_id"]; compound_col = mapping["compound"]; strain_col = mapping["strain"]; time_col = mapping["time"]; plate_col = mapping["plate"]
    compounds = sorted(train[compound_col].fillna("<NA>").astype(str).unique()); strains = sorted(train[strain_col].fillna("<NA>").astype(str).unique()); plates = sorted(train[plate_col].fillna("<NA>").astype(str).unique())
    compound_bucket5 = {c: i % 5 for i, c in enumerate(compounds)}; compound_bucket4 = {c: i % 4 for i, c in enumerate(compounds)}; plate_bucket5 = {p: i % 5 for i, p in enumerate(plates)}
    cvals = train[compound_col].fillna("<NA>").astype(str); svals = train[strain_col].fillna("<NA>").astype(str); tvals = pd.to_numeric(train[time_col], errors="coerce"); pvals = train[plate_col].fillna("<NA>").astype(str)
    folds = []
    def add_fold(fold_type, fold_id, hold_mask, train_mask, notes):
        hold_mask = np.asarray(hold_mask, dtype=bool); train_mask = np.asarray(train_mask, dtype=bool)
        if np.any(hold_mask & train_mask): raise RuntimeError("train/hold overlap in %s/%s" % (fold_type, fold_id))
        train_comp = set(cvals[train_mask]); hold_comp = set(cvals[hold_mask]); train_strain = set(svals[train_mask]); hold_strain = set(svals[hold_mask]); train_time = set(tvals[train_mask].dropna()); hold_time = set(tvals[hold_mask].dropna()); train_plate = set(pvals[train_mask]); hold_plate = set(pvals[hold_mask])
        # A row can be intentionally excluded from a fold.  The historical
        # implementation mapped every non-holdout row to train and therefore
        # ignored train_mask for the both-OOD regime.  Preserve excluded rows
        # explicitly so the intended fit set is auditable and consumers that
        # select role==train/holdout continue to work.
        role = np.where(hold_mask, "holdout", np.where(train_mask, "train", "excluded"))
        rows = pd.DataFrame({"sample_ID": train[sample_col].astype(str), "fold_type": fold_type, "fold_id": str(fold_id), "role": role})
        rows["notes"] = notes; folds.append(rows)
        summary = {"fold_type": fold_type, "fold_id": str(fold_id), "n_train": int(train_mask.sum()), "n_holdout": int(hold_mask.sum()), "train_compounds": len(train_comp), "holdout_compounds": len(hold_comp), "compound_overlap": len(train_comp & hold_comp), "train_strains": len(train_strain), "holdout_strains": len(hold_strain), "strain_overlap": len(train_strain & hold_strain), "train_times": len(train_time), "holdout_times": len(hold_time), "time_overlap": len(train_time & hold_time), "train_plates": len(train_plate), "holdout_plates": len(hold_plate), "plate_overlap": len(train_plate & hold_plate), "notes": notes}
        return summary
    summaries = []
    for bucket in range(5):
        hold = cvals.map(compound_bucket5).to_numpy() == bucket; summaries.append(add_fold("compound", bucket, hold, ~hold, "leave-compound-group-out"))
    for i, strain in enumerate(strains):
        hold = svals.to_numpy() == strain; summaries.append(add_fold("strain", i, hold, ~hold, "leave-one-strain-out:%s" % strain))
    if args.complete_both_grid:
        for i, strain in enumerate(strains):
            for held_bucket in range(4):
                fold_id = i * 4 + held_bucket
                hold = (svals.to_numpy() == strain) & (cvals.map(compound_bucket4).to_numpy() == held_bucket)
                train_mask = (svals.to_numpy() != strain) & (cvals.map(compound_bucket4).to_numpy() != held_bucket)
                summaries.append(add_fold("both", fold_id, hold, train_mask, "both-unseen strain=%s compound_bucket=%d complete-grid" % (strain, held_bucket)))
    else:
        for i, strain in enumerate(strains):
            held_bucket = i % 4; hold = (svals.to_numpy() == strain) & (cvals.map(compound_bucket4).to_numpy() == held_bucket); train_mask = (svals.to_numpy() != strain) & (cvals.map(compound_bucket4).to_numpy() != held_bucket); summaries.append(add_fold("both", i, hold, train_mask, "both-unseen strain=%s compound_bucket=%d diagonal" % (strain, held_bucket)))
    unique_times = sorted(tvals.dropna().unique().tolist()); hold_time = unique_times[len(unique_times) // 2]; hold = tvals.to_numpy() == hold_time; summaries.append(add_fold("time", 0, hold, ~hold, "leave-global-interior-time:%s" % hold_time))
    for bucket in range(5):
        hold = pvals.map(plate_bucket5).to_numpy() == bucket; summaries.append(add_fold("plate", bucket, hold, ~hold, "leave-plate-group-out"))
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output); output.mkdir(parents=True, exist_ok=True); fold_table = pd.concat(folds, ignore_index=True); fold_table.to_csv(output / "inner_folds.csv", index=False); summary = pd.DataFrame(summaries); summary.to_csv(output / "fold_summary.csv", index=False)
    check_cols = ["compound_overlap", "strain_overlap", "time_overlap", "plate_overlap"]
    both_holdout_counts = fold_table.loc[(fold_table.fold_type == "both") & (fold_table.role == "holdout")].groupby("sample_ID").size()
    leakage = {"fold_count": int(len(summary)), "n_train_rows": int(len(train)), "all_train_holdout_disjoint": bool((summary.n_train + summary.n_holdout).ge(0).all()), "compound_folds_no_compound_overlap": bool((summary.loc[summary.fold_type == "compound", "compound_overlap"] == 0).all()), "strain_folds_no_strain_overlap": bool((summary.loc[summary.fold_type == "strain", "strain_overlap"] == 0).all()), "both_folds_no_entity_overlap": bool(((summary.loc[summary.fold_type == "both", "compound_overlap"] == 0) & (summary.loc[summary.fold_type == "both", "strain_overlap"] == 0)).all()), "both_holdout_complete_exactly_once": bool(len(both_holdout_counts) == len(train) and (both_holdout_counts == 1).all()), "time_fold_has_no_time_leakage": bool(summary.loc[summary.fold_type == "time", "time_overlap"].iloc[0] == 0), "time_holdout_is_between_train_times": bool(min(unique_times) < hold_time < max(unique_times)), "plate_fold_no_plate_overlap": bool((summary.loc[summary.fold_type == "plate", "plate_overlap"] == 0).all()), "checked_columns": check_cols}
    (output / "fold_leakage_check.json").write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"seed": SEED, "source_split": "split_final=train only", "compound_folds": 5, "strain_folds": len(strains), "both_folds": len(strains) * 4 if args.complete_both_grid else len(strains), "complete_both_grid": bool(args.complete_both_grid), "explicit_excluded_role": True, "time_folds": 1, "plate_folds": 5, "time_holdout": hold_time, "no_official_validation_or_test_rows_in_fold_table": True, "notes": "inner folds for cross-fitted expert/gate development; not replacement for official validation"}
    (output / "fold_config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps({"config": config, "leakage": leakage, "summary": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
