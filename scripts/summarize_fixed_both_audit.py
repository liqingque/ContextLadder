#!/usr/bin/env python
"""Summarize independently generated fixed-both residual audits."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/e0_hcce_residual_contract/fixed_both"
SEEDS = (20260810, 3407, 42)


def main():
    rows = []
    details = {}
    for seed in SEEDS:
        path = BASE / f"seed{seed}" / "audit.json"
        if not path.exists():
            details[str(seed)] = {"status": "pending", "audit": str(path.relative_to(ROOT))}
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        regime = audit["regimes"]["both"]
        detail = {
            "status": regime["integrity_status"],
            "consumer_ready": regime["consumer_ready"],
            "n_samples": regime["n_samples"],
            "coverage_fraction": regime["sample_coverage_fraction"],
            "exactly_once": regime["each_official_train_sample_exactly_once"],
            "entity_exclusivity": regime["entity_exclusivity"]["entity_exclusivity_pass"],
            "fallback": regime["fallback"],
            "metrics": regime["metrics"],
            "audit": str(path.relative_to(ROOT)),
        }
        details[str(seed)] = detail
        rows.append({"seed": seed, **{k: detail[k] for k in ("status", "consumer_ready", "n_samples", "coverage_fraction", "exactly_once", "entity_exclusivity")}, **detail["metrics"]})
    complete = len(rows) == len(SEEDS)
    passed = complete and all(details[str(seed)]["consumer_ready"] for seed in SEEDS)
    summary = {
        "audit_id": "E0_FIXED_BOTH_3SEED_20260813",
        "integrity_status": "pass" if passed else "pending" if not complete else "fail",
        "gate_decision": "PASS" if passed else "PENDING" if not complete else "FAIL",
        "fold_table": "outputs/e0_hcce_residual_contract/inner_folds_fixed.csv",
        "design": "complete 4-strain x 4-compound-bucket grid",
        "test_truth_read": False,
        "seeds": details,
    }
    out_json = ROOT / "reports/E0_FIXED_BOTH_AUDIT_20260813.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# E0 Fixed-Both Audit",
        "",
        f"- Status: **{summary['gate_decision']}**",
        "- Fold design: complete 4-strain × 4-compound-bucket grid (16 folds)",
        "- Test proteome/truth accessed: **no**",
        "",
        "| Seed | Status | Coverage | Exactly once | Entity exclusive | Fallback bitwise | RMSE | abs PCC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        item = details[str(seed)]
        if item["status"] == "pending":
            lines.append(f"| {seed} | pending | — | — | — | — | — | — |")
        else:
            lines.append(
                f"| {seed} | {item['status']} | {item['coverage_fraction']:.6f} | {item['exactly_once']} | "
                f"{item['entity_exclusivity']} | {item['fallback']['bitwise_equal']} | "
                f"{item['metrics']['rmse']:.6f} | {item['metrics']['abs_pcc']:.6f} |"
            )
    lines += [
        "",
        "These are newly trained fixed-fold predictions. They do not retroactively validate the historical contaminated both-fold artifacts, which remain documented in `reports/EXPERIMENT_AUDIT.json`.",
        "",
    ]
    (ROOT / "reports/E0_FIXED_BOTH_AUDIT_20260813.md").write_text("\n".join(lines), encoding="utf-8")
    if rows:
        pd.DataFrame(rows).to_csv(BASE / "three_seed_scorecard.csv", index=False)
    print(json.dumps({"status": summary["gate_decision"], "seeds": {k: v["status"] for k, v in details.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
