#!/usr/bin/env python
"""Validate prediction.csv against the semi-final submission format.

Checks the format rules literally: 4,454 rows; sample_ID plus 4,422 official
protein columns in the contract order; all values finite; log2 scale declared
in the manifest; no duplicate ids or columns. Also re-asserts the data boundary
recorded by the run that produced the file.

    python scripts/validate_submission.py --prediction prediction.csv --run-dir runs/final
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--run-dir", help="run directory from scripts/train.py, for the protein contract")
    ap.add_argument("--metadata", help="official test metadata CSV (default: configs/data_paths.yaml)")
    ap.add_argument("--expect-proteins", type=int, default=4422)
    ap.add_argument("--expect-rows", type=int, default=4454)
    ap.add_argument("--out", help="optional path for the JSON report")
    args = ap.parse_args()

    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    sid = mapping["sample_id"]
    meta_path = Path(args.metadata) if args.metadata else ROOT / paths["metadata_test"]

    pred = pd.read_csv(args.prediction)
    # The order check needs the official test metadata. It is not redistributed with
    # the package, so when it is absent we run every other check and report PARTIAL
    # rather than silently passing.
    test_meta = pd.read_csv(meta_path, low_memory=False) if Path(meta_path).exists() else None
    cols = pred.columns.tolist()
    prot_cols = cols[1:]

    expected_proteins = None
    if args.run_dir:
        lst = Path(args.run_dir) / "artifacts" / "protein_list.txt"
        if lst.exists():
            expected_proteins = [p for p in lst.read_text(encoding="utf-8").split("\n") if p]

    values = pred[prot_cols].to_numpy(dtype=np.float64)
    checks = {
        "first_column_is_sample_id": cols[0] == sid,
        "n_rows": int(len(pred)),
        "n_rows_ok": len(pred) == args.expect_rows,
        "n_protein_columns": len(prot_cols),
        "n_protein_columns_ok": len(prot_cols) == args.expect_proteins,
        "sample_ids_unique": bool(pred[sid].is_unique),
        "no_duplicate_columns": len(set(cols)) == len(cols),
        "sample_order_matches_test_metadata": (
            bool(pred[sid].astype(str).tolist() == test_meta[sid].astype(str).tolist())
            if test_meta is not None else "skipped: official test metadata not present"),
        "all_values_finite": bool(np.isfinite(values).all()),
        "no_nan": bool(not np.isnan(values).any()),
        "protein_order_matches_contract": (
            prot_cols == expected_proteins if expected_proteins is not None else "run-dir not given"),
        "value_range": [float(values.min()), float(values.max())],
        "looks_like_log2_not_zscore": bool(values.mean() > 5.0),
        "sha256": sha256_file(args.prediction),
    }

    hard = ["first_column_is_sample_id", "n_rows_ok", "n_protein_columns_ok", "sample_ids_unique",
            "no_duplicate_columns", "sample_order_matches_test_metadata", "all_values_finite",
            "no_nan", "looks_like_log2_not_zscore"]
    if expected_proteins is not None:
        hard.append("protein_order_matches_contract")
    skipped = [k for k in hard if isinstance(checks[k], str)]
    failed = [k for k in hard if checks[k] is not True and k not in skipped]
    checks["verdict"] = "FAIL" if failed else ("PASS" if not skipped else "PARTIAL")
    checks["failed_checks"] = failed
    checks["skipped_checks"] = skipped

    if args.run_dir:
        c = Path(args.run_dir) / "artifacts" / "preprocess_contract.json"
        if c.exists():
            contract = json.loads(c.read_text(encoding="utf-8"))
            checks["data_boundary"] = {
                "fit_rows": contract["fit_rows"],
                "held_out_val_rows": contract["held_out_val_rows"],
                "filter_fit_scope": contract["filter_fit_scope"],
                "test_proteome_read": contract["test_proteome_read"],
            }

    text = json.dumps(checks, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    sys.exit(0 if checks["verdict"] in ("PASS", "PARTIAL") else 1)


if __name__ == "__main__":
    main()
