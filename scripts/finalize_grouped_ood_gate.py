#!/usr/bin/env python
"""E2 step 3: assemble GATE.json from the manifest, leakage assertions and
aggregate metrics already written to outputs/grouped_ood/.

This gate is about the PROTOCOL, not about picking a winning model: it
records whether the four-axis split was built without leakage and whether
the requested axis/fold-count contract (plate~=12, strain=4, compound~=10,
time=6) was met, plus a flat summary of which candidates have been
evaluated. It does not decide whether any candidate should replace the
frozen competition submission -- that judgment is out of scope for E2 and is
left to the report's narrative and to whatever downstream gate (e.g. E3)
consumes these numbers.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grouped_ood_common import AXES  # noqa: E402

OUT = ROOT / "outputs/grouped_ood"

EXPECTED = {"plate": 12, "strain": 4, "compound": 10, "time": 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.dir)

    manifest = json.loads((out / "split_manifest.json").read_text())
    leakage = json.loads((out / "leakage_assertions.json").read_text())
    agg = json.loads((out / "aggregate_metrics.json").read_text()) if (out / "aggregate_metrics.json").exists() else {}

    fold_count_checks = {}
    for axis, expected_k in EXPECTED.items():
        actual_k = manifest["axes"][axis]["n_folds"]
        fold_count_checks[axis] = {"expected": expected_k, "actual": actual_k, "pass": actual_k == expected_k}

    low_power_axes = {
        "strain": {
            "n_entities": manifest["axes"]["strain"]["n_entities"],
            "warning": ("Only 4 strain entities in split_final=='train' (BAH, CEK, DHY210, CGD); "
                        "leave-one-strain-out therefore has 4 folds total. Any strain-axis macro "
                        "metric or bootstrap CI here reflects n=4 clusters and has very limited "
                        "statistical power -- it can rule out large regressions but cannot resolve "
                        "small deltas between candidates. This mirrors, but does partially relieve, "
                        "the single-unseen-strain limitation of the official validation split."),
        },
    }

    candidates_evaluated = list(agg.keys())
    evaluation_completed = len(candidates_evaluated) > 0 and all(
        set(agg[c]["axes_evaluated"]) == set(EXPECTED.keys()) for c in candidates_evaluated
    )

    gate = {
        "protocol": "E2 train-internal four-axis grouped pseudo-OOD",
        "data_boundary_ok": manifest["data_boundary"] == "split_final == 'train' only; official validation/test truth never read",
        "n_train_rows": manifest["n_train_rows"],
        "leakage_assertions": {
            "n_assertions": leakage["n_assertions"],
            "all_pass": leakage["all_pass"],
        },
        "fold_count_contract": fold_count_checks,
        "fold_count_contract_pass": all(v["pass"] for v in fold_count_checks.values()),
        "candidates_evaluated": candidates_evaluated,
        "evaluation_completed_all_axes": evaluation_completed,
        "low_power_axes": low_power_axes,
        "scope_note": ("This GATE reports whether the E2 protocol itself was constructed and executed "
                        "correctly (no leakage, correct fold contract, baseline evaluated on all four "
                        "axes). It is NOT a model-selection gate and does NOT authorize replacing any "
                        "frozen competition prediction; see E2_REPORT.md limitations section."),
        "overall_pass": bool(leakage["all_pass"]) and all(v["pass"] for v in fold_count_checks.values()) and evaluation_completed,
    }

    (out / "GATE.json").write_text(json.dumps(gate, indent=2, ensure_ascii=False))
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"\nwrote {out/'GATE.json'}")


if __name__ == "__main__":
    main()
