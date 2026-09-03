#!/usr/bin/env python
"""Merge per-seed E2 grouped-OOD runs into the 3-seed summary the plan asks for.

`scripts/evaluate_grouped_ood.py` takes a single `--seed` and dedups written rows
on ``(candidate, axis, fold_id)`` -- seed is *not* part of that key, so re-running
the same variant with a second seed silently overwrites the first. Each seed is
therefore run into its own `--out` directory and merged here instead, which also
means this script never has to modify the E2 harness.

Plan reference: 复赛综合审查与提升实验计划.md §6.6.3 -- "三种子均值、标准差和每个
seed 方向".

    /home/lxm/anaconda3/envs/tl/bin/python scripts/merge_grouped_ood_seeds.py
"""
import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRS = {
    20260810: ROOT / "outputs/grouped_ood",
    3407: ROOT / "outputs/grouped_ood_seed3407",
    42: ROOT / "outputs/grouped_ood_seed42",
}
METRICS = ("sample_pcc", "log2_rmse", "log2_mae")
AXES = ("plate", "strain", "compound", "time")


def load(dirs, candidate):
    out = {}
    for seed, d in dirs.items():
        path = Path(d) / "aggregate_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run that seed first")
        blob = json.loads(path.read_text())
        if candidate not in blob:
            raise KeyError(f"{path} has no candidate {candidate!r}; has {list(blob)}")
        rec = blob[candidate]
        # Guard against a mislabelled directory: the file records the seed it ran.
        if int(rec.get("seed", seed)) != int(seed):
            raise ValueError(f"{path} reports seed {rec.get('seed')} but was filed under {seed}")
        out[seed] = rec["aggregate"]
    return out


def merge(per_seed):
    seeds = sorted(per_seed)
    summary = {}
    for axis in AXES:
        if not all(axis in per_seed[s] for s in seeds):
            continue
        a = {"n_entities": per_seed[seeds[0]][axis]["n_entities"],
             "n_folds": per_seed[seeds[0]][axis]["n_folds"],
             "seeds": seeds}
        for metric in METRICS:
            vals = [per_seed[s][axis][metric]["entity_macro"] for s in seeds]
            a[metric] = {
                "per_seed": {str(s): v for s, v in zip(seeds, vals)},
                "mean": st.fmean(vals),
                "std": st.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "spread": max(vals) - min(vals),
                # Single-seed CIs are entity-cluster bootstrap; the spread across
                # seeds is a *different* source of variation and is reported
                # separately rather than folded into one interval.
                "single_seed_ci_seed{}".format(seeds[0]): [
                    per_seed[seeds[0]][axis][metric]["ci_lo"],
                    per_seed[seeds[0]][axis][metric]["ci_hi"],
                ],
            }
        summary[axis] = a
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="mask_compound")
    ap.add_argument("--out", default=str(ROOT / "outputs/grouped_ood/aggregate_3seed.json"))
    args = ap.parse_args()

    per_seed = load(DEFAULT_DIRS, args.candidate)
    summary = merge(per_seed)
    blob = {
        "candidate": args.candidate,
        "seeds": sorted(per_seed),
        "source_dirs": {str(k): str(v) for k, v in DEFAULT_DIRS.items()},
        "aggregation": "entity-macro per seed, then mean/std across seeds",
        "note": ("Seed spread and the single-seed entity-cluster bootstrap CI measure different "
                 "things; both are reported, neither is combined into one interval."),
        "axes": summary,
    }
    Path(args.out).write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"candidate={args.candidate}  seeds={sorted(per_seed)}\n")
    hdr = f"{'axis':10s} {'ent':>4s} {'folds':>5s} " + " ".join(f"{m:>26s}" for m in METRICS)
    print(hdr)
    for axis, a in summary.items():
        cells = []
        for m in METRICS:
            v = a[m]
            cells.append(f"{v['mean']:.4f}±{v['std']:.4f} (Δ{v['spread']:.4f})".rjust(26))
        print(f"{axis:10s} {a['n_entities']:4d} {a['n_folds']:5d} " + " ".join(cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
