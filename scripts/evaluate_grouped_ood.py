#!/usr/bin/env python
"""E2 step 2: run a candidate through the four-axis grouped pseudo-OOD protocol.

Reusable core: ``evaluate_four_axis(train_fn, predict_fn, candidate_name, ...)``.
Any candidate -- the current HCCE mask_compound baseline, or later one of E1's
six variants -- plugs in by providing:

  train_fn(meta, y, fit_idx, mapping, seed, epochs, device) -> state
  predict_fn(state, meta_subset, device) -> np.ndarray predictions aligned to
      meta_subset's rows (log2 space, same protein columns as ``proteins``)

scripts.grouped_ood_common.build_hcce_variant_functions(...) builds this pair
for any of scripts/a2b_train_variants.py's HCCE variants without touching
that file. A brand-new architecture just needs to hand this module its own
(train_fn, predict_fn) pair -- nothing else in this script is HCCE-specific.

For every fold: fold-train categorical vocabularies/normalisation stats are
refit from scratch inside train_fn (this falls straight out of calling the
untouched HCCEMetaEncoder.fit()/fit_model_variant() with fold-local indices);
the held-out entity's rows are scored against a model that has never seen
that entity. Metrics are aggregated entity-macro (per-entity metric first,
then equal-weight mean across entities), with entity-cluster bootstrap CIs
(2000 resamples of entities, not rows).

No test truth and no official-validation truth is read anywhere in this file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grouped_ood_common import (  # noqa: E402
    assert_no_leakage, build_hcce_variant_functions, cluster_bootstrap_ci,
    entity_macro, load_universe, matched_fc_for_fold, per_entity_metrics,
)
from scripts.run_baselines import json_safe  # noqa: E402
from src.evaluation.evaluator import basic_metrics  # noqa: E402

OUT = ROOT / "outputs/grouped_ood"
METRIC_COLS = ["sample_pcc", "log2_rmse", "log2_mae"]


def evaluate_four_axis(train_fn, predict_fn, candidate_name, manifest, meta_train, y_train, mapping,
                        seed=20260810, epochs=40, device="cuda", axes=None, n_bootstrap=2000,
                        compute_in_sample_reference=True, verbose=True):
    """Core reusable harness. Returns a dict with per_fold/per_entity
    DataFrames and an aggregate dict, keyed by axis."""
    axes = list(axes) if axes else list(manifest["axes"].keys())
    device_t = torch.device(device)
    sample_id_col = mapping["sample_id"]
    sample_id_to_row = {sid: i for i, sid in enumerate(meta_train[sample_id_col].astype(str))}

    per_fold_rows, per_entity_frames = [], []
    for axis in axes:
        axis_block = manifest["axes"][axis]
        field_col = axis_block["field"]
        entity_values_all = meta_train[field_col].astype(str).to_numpy()
        for fold in axis_block["folds"]:
            fold_id = fold["fold_id"]
            train_idx = np.array([sample_id_to_row[s] for s in fold["train_sample_ids"]], dtype=int)
            test_idx = np.array([sample_id_to_row[s] for s in fold["test_sample_ids"]], dtype=int)
            assert_no_leakage(meta_train, field_col, train_idx, test_idx)  # re-verified at eval time, aborts on failure

            t0 = time.time()
            state = train_fn(meta_train, y_train, train_idx, mapping, seed, epochs, device_t)
            test_meta = meta_train.iloc[test_idx].reset_index(drop=True)
            pred_test = predict_fn(state, test_meta, device_t)
            true_test = y_train[test_idx]
            unknown_metrics = basic_metrics(pred_test, true_test)

            entity_df = per_entity_metrics(pred_test, true_test, entity_values_all[test_idx])
            entity_df["axis"] = axis
            entity_df["fold_id"] = fold_id
            entity_df["candidate"] = candidate_name
            per_entity_frames.append(entity_df)

            fc = matched_fc_for_fold(pred_test, y_train, meta_train, mapping, train_idx, test_idx)

            known_metrics = {}
            if compute_in_sample_reference:
                train_meta_subset = meta_train.iloc[train_idx].reset_index(drop=True)
                pred_train = predict_fn(state, train_meta_subset, device_t)
                known_metrics = basic_metrics(pred_train, y_train[train_idx])

            elapsed = time.time() - t0
            per_fold_rows.append({
                "candidate": candidate_name, "axis": axis, "fold_id": fold_id,
                "n_entities": fold["n_entities"], "entities": ";".join(fold["entities"]),
                "n_train_rows": fold["n_train_rows"], "n_test_rows": fold["n_test_rows"],
                "unknown_sample_pcc": unknown_metrics["abs_pcc"], "unknown_log2_rmse": unknown_metrics["rmse"],
                "unknown_log2_mae": unknown_metrics["mae"],
                "known_in_sample_sample_pcc": known_metrics.get("abs_pcc"),
                "known_in_sample_log2_rmse": known_metrics.get("rmse"),
                "known_in_sample_log2_mae": known_metrics.get("mae"),
                "fc_pcc": fc.get("fc_pcc"), "fc_coverage": fc.get("coverage"),
                "fc_n_matched": fc.get("n_matched"), "fc_n_treatments": fc.get("n_treatments"),
                "elapsed_sec": elapsed,
            })
            if verbose:
                print(f"  [{candidate_name}] axis={axis:9s} fold={fold_id:2d} "
                      f"n_test={fold['n_test_rows']:4d} unknown_pcc={unknown_metrics['abs_pcc']:+.4f} "
                      f"unknown_rmse={unknown_metrics['rmse']:.4f} fc_coverage={fc.get('coverage')} "
                      f"({elapsed:.1f}s)")

    per_fold_df = pd.DataFrame(per_fold_rows)
    per_entity_df = pd.concat(per_entity_frames, ignore_index=True) if per_entity_frames else pd.DataFrame()

    aggregate = {}
    for axis in axes:
        sub_entity = per_entity_df[per_entity_df.axis == axis]
        sub_fold = per_fold_df[per_fold_df.axis == axis]
        axis_agg = {"n_entities": int(sub_entity["entity"].nunique()),
                    "n_folds": int(sub_fold["fold_id"].nunique()),
                    "total_test_rows": int(sub_fold["n_test_rows"].sum())}
        for metric in METRIC_COLS:
            axis_agg[metric] = {
                "entity_macro": entity_macro(sub_entity, metric),
                **cluster_bootstrap_ci(sub_entity, metric, n_boot=n_bootstrap),
            }
        u_pcc = float(np.nanmean(sub_fold["unknown_sample_pcc"]))
        k_pcc = float(np.nanmean(sub_fold["known_in_sample_sample_pcc"])) if compute_in_sample_reference else None
        u_rmse = float(np.nanmean(sub_fold["unknown_log2_rmse"]))
        k_rmse = float(np.nanmean(sub_fold["known_in_sample_log2_rmse"])) if compute_in_sample_reference else None
        axis_agg["unknown_vs_known"] = {
            "definition": ("'known' here is the IN-SAMPLE fit metric of the same fold-trained model "
                            "evaluated on its own fold-train rows (entities it was fit on); it is an "
                            "optimistic upper bound on true known-entity generalization, not a held-out "
                            "known-entity estimate (that would need nested/double CV, out of scope for "
                            "this first pass). 'unknown' is the true held-out fold-test metric."),
            "unknown_macro_pcc": u_pcc, "known_in_sample_macro_pcc": k_pcc,
            "gap_pcc_known_minus_unknown": (k_pcc - u_pcc) if k_pcc is not None else None,
            "unknown_macro_rmse": u_rmse, "known_in_sample_macro_rmse": k_rmse,
            "gap_rmse_unknown_minus_known": (u_rmse - k_rmse) if k_rmse is not None else None,
        }
        fc_cov = sub_fold["fc_coverage"].dropna()
        axis_agg["matched_control_fc"] = {
            "mean_fc_pcc": float(sub_fold["fc_pcc"].dropna().mean()) if sub_fold["fc_pcc"].notna().any() else None,
            "mean_coverage": float(fc_cov.mean()) if len(fc_cov) else None,
            "note": ("coverage collapses toward 0 for the plate/strain/time axes because match_controls "
                     "requires an exact context match (including plate/strain/time), and that context is "
                     "exactly what the fold removes from the legal (fold-train) control pool; compound "
                     "is the axis where matched-control FC is actually informative."),
        }
        aggregate[axis] = axis_agg

    return {"candidate": candidate_name, "seed": seed, "epochs": epochs,
            "per_fold": per_fold_df, "per_entity": per_entity_df, "aggregate": aggregate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mask_compound",
                    choices=["baseline", "mask", "mask_compound", "denoise", "mask_denoise", "mask_compound_denoise"])
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--axes", nargs="+", default=None, choices=["plate", "strain", "compound", "time"])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--manifest", default=str(OUT / "split_manifest.json"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest["leakage_summary"]["all_pass"]:
        raise AssertionError("manifest leakage_summary.all_pass is False; refusing to evaluate on a leaking split")

    rh, meta_train, y_train, proteins, mapping = load_universe()
    train_fn, predict_fn = build_hcce_variant_functions(rh, mapping, args.variant, embedding_dim=args.embedding_dim)

    started = time.time()
    result = evaluate_four_axis(
        train_fn, predict_fn, args.variant, manifest, meta_train, y_train, mapping,
        seed=args.seed, epochs=args.epochs, device=device, axes=args.axes, n_bootstrap=args.n_bootstrap,
    )
    elapsed = time.time() - started

    per_fold_path = out / "per_fold_metrics.csv"
    per_entity_path = out / "per_entity_metrics.csv"
    if per_fold_path.exists():
        result["per_fold"] = pd.concat([pd.read_csv(per_fold_path), result["per_fold"]], ignore_index=True) \
            .drop_duplicates(subset=["candidate", "axis", "fold_id"], keep="last")
    if per_entity_path.exists():
        result["per_entity"] = pd.concat([pd.read_csv(per_entity_path), result["per_entity"]], ignore_index=True) \
            .drop_duplicates(subset=["candidate", "axis", "fold_id", "entity"], keep="last")
    result["per_fold"].to_csv(per_fold_path, index=False)
    result["per_entity"].to_csv(per_entity_path, index=False)

    agg_path = out / "aggregate_metrics.json"
    all_agg = json.loads(agg_path.read_text()) if agg_path.exists() else {}
    all_agg[args.variant] = {
        "seed": args.seed, "epochs": args.epochs, "embedding_dim": args.embedding_dim,
        "axes_evaluated": args.axes or list(manifest["axes"].keys()),
        "n_bootstrap": args.n_bootstrap, "elapsed_sec": elapsed, "aggregate": result["aggregate"],
    }
    agg_path.write_text(json.dumps(json_safe(all_agg), indent=2, ensure_ascii=False))

    print(f"\nwrote {per_fold_path}")
    print(f"wrote {per_entity_path}")
    print(f"wrote {agg_path}")
    print(f"[{args.variant}] total elapsed {elapsed:.1f}s")
    for axis, agg in result["aggregate"].items():
        print(f"  axis={axis:9s} n_entities={agg['n_entities']:3d} n_folds={agg['n_folds']:2d} "
              f"pcc_macro={agg['sample_pcc']['entity_macro']:.4f} "
              f"[{agg['sample_pcc']['ci_lo']:.4f},{agg['sample_pcc']['ci_hi']:.4f}] "
              f"rmse_macro={agg['log2_rmse']['entity_macro']:.4f}")


if __name__ == "__main__":
    main()
