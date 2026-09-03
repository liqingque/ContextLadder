#!/usr/bin/env python
"""E2 step 1: build the train-internal four-axis grouped pseudo-OOD split.

Reads only split_final == "train" rows (5,920) from the official metadata; no
test truth and no official-validation truth is ever touched. For each of the
four axes (plate / strain / compound / time) this constructs a leakage-free
grouped K-fold partition where every row of a given plate/strain/compound/
time-value sits entirely on one side of every fold split, and writes:

  outputs/grouped_ood/split_manifest.json   - full manifest (entities, sample
                                               IDs, row counts per fold)
  outputs/grouped_ood/leakage_assertions.json - one assertion record per fold

A leakage assertion failure raises immediately (process aborts, non-zero
exit); this script never downgrades a failed assertion to a warning.

scripts/run_hcce.py and scripts/a2b_train_variants.py are imported only
(never modified) via scripts/grouped_ood_common.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grouped_ood_common import (  # noqa: E402
    AXES, assert_no_leakage, build_axis_groups, fold_row_indices, load_universe,
)

OUT = ROOT / "outputs/grouped_ood"


def build_axis_manifest(meta_train, axis, mapping):
    field_col, groups = build_axis_groups(meta_train, axis, mapping)
    sample_col = mapping["sample_id"]
    sample_ids = meta_train[sample_col].astype(str).to_numpy()
    folds = []
    assertions = []
    seen_test_idx = np.zeros(len(meta_train), dtype=bool)
    for fold_id, entities in enumerate(groups):
        train_idx, test_idx = fold_row_indices(meta_train, field_col, entities)
        record = assert_no_leakage(meta_train, field_col, train_idx, test_idx)  # raises on failure
        record.update({"axis": axis, "fold_id": fold_id, "entities": entities})
        assertions.append(record)
        if seen_test_idx[test_idx].any():
            raise AssertionError(f"axis={axis} fold={fold_id}: a row was assigned test-side in >1 fold")
        seen_test_idx[test_idx] = True
        folds.append({
            "fold_id": fold_id,
            "entities": entities,
            "n_entities": len(entities),
            "n_train_rows": int(len(train_idx)),
            "n_test_rows": int(len(test_idx)),
            "train_sample_ids": sample_ids[train_idx].tolist(),
            "test_sample_ids": sample_ids[test_idx].tolist(),
        })
    if not seen_test_idx.all():
        missing = int((~seen_test_idx).sum())
        raise AssertionError(f"axis={axis}: {missing} train rows never appear in any fold's test side "
                              f"(the {len(groups)} groups do not partition all entities)")
    if seen_test_idx.sum() != len(meta_train):
        raise AssertionError(f"axis={axis}: test-side row count {int(seen_test_idx.sum())} != universe size {len(meta_train)}")
    axis_block = {
        "field": field_col,
        "mode": AXES[axis]["mode"],
        "n_entities": len(groups) if AXES[axis]["mode"] == "complete" else sum(len(g) for g in groups),
        "n_folds": len(groups),
        "entity_row_counts": meta_train[field_col].astype(str).value_counts().to_dict(),
        "folds": folds,
    }
    if axis == "time":
        latest = max(axis_block["entity_row_counts"], key=lambda v: float(v))
        for f in folds:
            f["is_latest_block"] = (f["entities"] == [latest])
        axis_block["note"] = (
            "6 distinct pert_time values in split_final=='train'; each fold holds out exactly one "
            "value (trivially a contiguous block since it is a single point). The fold whose held-out "
            f"value is the latest time point ({latest}) is flagged is_latest_block=true, matching the "
            "plan's 'hold out the latest time point' scenario; the other five folds sweep the remaining "
            "interior/early blocks so blocked-time coverage is complete rather than a single point estimate."
        )
    return axis_block, assertions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _, meta_train, y_train, proteins, mapping = load_universe()

    manifest = {
        "protocol": "E2 train-internal four-axis grouped pseudo-OOD",
        "data_boundary": "split_final == 'train' only; official validation/test truth never read",
        "n_train_rows": int(len(meta_train)),
        "n_proteins": int(len(proteins)),
        "axes": {},
    }
    all_assertions = []
    for axis in AXES:
        axis_block, assertions = build_axis_manifest(meta_train, axis, mapping)
        manifest["axes"][axis] = axis_block
        all_assertions.extend(assertions)
        n_folds = axis_block["n_folds"]
        n_ent = axis_block["n_entities"]
        print(f"axis={axis:9s} field={axis_block['field']:32s} n_entities={n_ent:4d} "
              f"n_folds={n_folds:3d} all_leakage_assertions_pass="
              f"{all(a['pass'] for a in assertions)}")

    manifest["leakage_summary"] = {
        "n_assertions": len(all_assertions),
        "n_pass": sum(1 for a in all_assertions if a["pass"]),
        "all_pass": all(a["pass"] for a in all_assertions),
    }

    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out / "leakage_assertions.json").write_text(json.dumps({
        "n_assertions": len(all_assertions),
        "all_pass": manifest["leakage_summary"]["all_pass"],
        "assertions": all_assertions,
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out/'split_manifest.json'}")
    print(f"wrote {out/'leakage_assertions.json'}")
    print(f"leakage: {manifest['leakage_summary']}")

    if not manifest["leakage_summary"]["all_pass"]:
        # Should be unreachable: assert_no_leakage already raises on the first
        # failure. Kept as a defense-in-depth hard exit, not a warning.
        sys.exit(1)


if __name__ == "__main__":
    main()
