#!/usr/bin/env python
"""Gate-W summary table from an evaluate_official_modules.py metrics file.

Gate-W (frozen in configs/p05_gates.json) is the official-weighted score

    W = 0.20*absPCC + 0.25*FC_all + 0.20*S1 + 0.20*S2 + 0.10*T + 0.05*DEPacc

Honesty note carried into the output: the evaluator emits no separate
val_time module (val_time rows have both entities in train, so they are
stratified as `seen`), therefore the 10% term T is filled with the S3 FC
alone. This is recorded in the output rather than silently substituted, and
no time-FC number is invented.

W_noS2 drops the 20% drug-residual term and renormalises over the remaining
0.80, because validation carries a single unseen strain (BAI): any S2 delta
is a single-cluster measurement and is not generalisation evidence.

No confidence interval is attached to Delta-W. The per-module paired cluster
bootstrap already lives in the metrics file and is what the gate decides on;
manufacturing a CI for the weighted sum from module point estimates would be
a fabricated number.
"""

import argparse
import json
from pathlib import Path

WEIGHTS = {"abs": 0.20, "fc": 0.25, "s1": 0.20, "s2": 0.20, "t10": 0.10, "dep": 0.05}
ADOPTION_FLOOR = 0.010  # = frozen S1 effect floor 0.053 * that module's 0.20 official weight


def modules(entry, granularity):
    s3 = entry["fc"].get("S3_both", {}).get("pcc")
    return {
        "abs": entry["absolute"]["sample_pcc_mean"],
        "rmse": entry["absolute"]["log2_rmse"],
        "fc": entry["fc"]["ALL"]["pcc"],
        "s1": entry["context_residual"][granularity]["pcc"],
        "s2": entry["drug_residual"]["pcc"],
        "t10": s3,
        "dep": entry["dep"]["direction_accuracy"],
    }


def weighted(m):
    w = sum(WEIGHTS[k] * m[k] for k in ("abs", "fc", "s1", "s2", "t10", "dep"))
    keys_no_s2 = ("abs", "fc", "s1", "t10", "dep")
    w_no_s2 = sum(WEIGHTS[k] * m[k] for k in keys_no_s2) / sum(WEIGHTS[k] for k in keys_no_s2)
    return w, w_no_s2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="module_metrics.json from the official evaluator")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--granularity", default="official_plate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.metrics).read_text())
    if args.baseline not in d["candidates"]:
        raise SystemExit(f"baseline {args.baseline} not in metrics file")

    mods = {n: modules(e, args.granularity) for n, e in d["candidates"].items()}
    base_w, base_wn = weighted(mods[args.baseline])
    paired = d.get("paired_cluster_bootstrap_vs_baseline", {})

    rows = {}
    for name, m in mods.items():
        w, wn = weighted(m)
        s1b = paired.get(name, {}).get(f"S1_context_residual_{args.granularity}", {})
        rows[name] = {
            **m,
            "W": w, "W_noS2": wn,
            "delta_W": w - base_w, "delta_W_noS2": wn - base_wn,
            "s1_delta": s1b.get("delta"), "s1_ci95": s1b.get("ci95"),
            "s1_ci_excludes_zero": s1b.get("ci_excludes_zero"),
        }

    # Gate-W verdict, per the frozen rule.
    for name, r in rows.items():
        if name == args.baseline:
            r["verdict"] = "BASELINE"
            continue
        neg_sig = bool(r["s1_ci_excludes_zero"]) and (r["s1_delta"] or 0) < 0
        if neg_sig:
            r["verdict"] = "DOMINATED" if r["delta_W"] < 0 else "REJECT_negative_module"
        elif r["delta_W"] >= ADOPTION_FLOOR and r["delta_W_noS2"] >= ADOPTION_FLOOR:
            r["verdict"] = "ADOPT_candidate_pending_CI"
        else:
            r["verdict"] = "REJECT_below_floor"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "granularity": args.granularity,
        "baseline": args.baseline,
        "weights": WEIGHTS,
        "adoption_floor_delta_W": ADOPTION_FLOOR,
        "adoption_floor_derivation": "frozen S1 effect floor 0.053 x official module weight 0.20",
        "t10_term": "S3 FC only -- the evaluator emits no separate val_time module; no time-FC value invented",
        "delta_W_confidence_interval": "not computed; the frozen gate decides on per-module paired "
                                       "cluster bootstrap, and a CI for the weighted sum is not "
                                       "derivable from module point estimates",
        "candidates": rows,
        "no_test_truth": True,
    }
    (out / "gate_w_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = ("| candidate | abs PCC | RMSE | FC all | S1 | S2 | S3 FC | DEP | W | ΔW | ΔW_noS2 | verdict |\n"
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    lines = []
    for name, r in rows.items():
        dw = "—" if name == args.baseline else f"{r['delta_W']:+.4f}"
        dwn = "—" if name == args.baseline else f"{r['delta_W_noS2']:+.4f}"
        lines.append(f"| {name} | {r['abs']:.6f} | {r['rmse']:.6f} | {r['fc']:.6f} | {r['s1']:.6f} | "
                     f"{r['s2']:.6f} | {r['t10']:.6f} | {r['dep']:.6f} | {r['W']:.4f} | {dw} | {dwn} | {r['verdict']} |")
    table = hdr + "\n".join(lines) + "\n"
    (out / "gate_w_table.md").write_text(table, encoding="utf-8")
    print(table)
    print(f"adoption floor ΔW >= {ADOPTION_FLOOR:+.3f}  (0.053 frozen S1 floor x 0.20 weight)")
    print(f"wrote {out/'gate_w_summary.json'} and {out/'gate_w_table.md'}")


if __name__ == "__main__":
    main()
