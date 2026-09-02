#!/usr/bin/env python
"""3-seed ensemble submission: refit the A2 mask_compound variant on the
`train` split only, for three seeds, and average the test predictions.

Mirrors make_hcce_submission.py (single-seed) but ensembles three seeds, which
was the frozen-validation-selected combination (C result). Test proteome/test
truth are never read.

Official data constraint (participation manual, "数据使用约束与最终评测"):
training may only use the `train` split's proteome labels; the validation and
test splits must not enter training, nor be used to estimate any statistic
(including the retained-protein list and normalization parameters).  Every
fitted quantity below is therefore restricted to `split_final == "train"`:
the protein filter, the categorical/numeric encoders, the target mean/std and
the fallback means used for the filtered-out proteins.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2b_train_variants import fit_model_variant, load_run_hcce
from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome


SEEDS = [20260810, 3407, 42]
EPOCHS = 40
EMBEDDING_DIM = 64
MASK_P = 0.25


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def main():
    rh = load_run_hcce()
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    train_meta = load_metadata(ROOT / paths["metadata_train_val"])
    test_meta = load_metadata(ROOT / paths["metadata_test"])
    train_prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    train_meta, train_prot, _, all_proteins = align_metadata_proteome(train_meta, train_prot)
    raw_train_all = finite_float_matrix(train_prot, all_proteins)
    proteins, missing_rate, keep_mask = rh.apply_official_protein_filter(train_meta, raw_train_all, all_proteins, mapping, threshold=0.80)
    train_y = to_log2_proteome(raw_train_all[:, keep_mask]).astype(np.float64)
    combined = pd.concat([train_meta, test_meta], ignore_index=True)
    combined_y = np.full((len(combined), len(proteins)), np.nan, dtype=np.float64)
    combined_y[:len(train_meta)] = train_y
    # Train-only fit scope: validation rows are excluded from training and from
    # every statistic estimated downstream (encoders, target mean/std, fallbacks).
    train_split_mask = train_meta[mapping["split"]].astype(str).eq("train").to_numpy()
    fit_indices = np.flatnonzero(train_split_mask).astype(int)
    test_meta_reset = test_meta.reset_index(drop=True)

    legacy_parts, film_parts = [], []
    for seed in SEEDS:
        model, encoder, target_mean, target_std, history = fit_model_variant(
            rh, combined, combined_y, fit_indices, mapping, seed, EPOCHS, device,
            EMBEDDING_DIM, "mask_compound", mask_p=MASK_P,
        )
        _, film_test, legacy_test = rh.predict_model(model, encoder, target_mean, target_std, test_meta_reset, device)
        legacy_parts.append(legacy_test)
        film_parts.append(film_test)
        print(f"seed {seed} refit done", flush=True)
    filtered_prediction = (0.5 * np.mean(np.stack(legacy_parts), axis=0) + 0.5 * np.mean(np.stack(film_parts), axis=0)).astype(np.float32)
    # Fallback values for the proteins the filter dropped: per-protein mean over
    # the train split only.  Proteins with no finite train observation fall back
    # to the global train mean instead of 0.0 — 0.0 is log2 intensity 1, far
    # outside the observed log2 range, and would distort per-sample fidelity.
    log2_train_only = to_log2_proteome(raw_train_all[train_split_mask]).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        train_mean_all = np.nanmean(log2_train_only, axis=0)
        global_train_mean = float(np.nanmean(log2_train_only))
    n_global_fallback = int(np.sum(~np.isfinite(train_mean_all)))
    train_mean_all = np.where(np.isfinite(train_mean_all), train_mean_all, global_train_mean)
    prediction = np.tile(train_mean_all.astype(np.float32), (len(test_meta_reset), 1))
    prediction[:, keep_mask] = filtered_prediction
    output_dir = ROOT / "outputs/submission_mask_compound_3seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(prediction, columns=all_proteins)
    pred.insert(0, mapping["sample_id"], test_meta_reset[mapping["sample_id"]].astype(str).to_numpy())
    pred_path = output_dir / "prediction.csv"
    pred.to_csv(pred_path, index=False)
    contract = {
        "sample_id_column": mapping["sample_id"],
        "n_rows": int(len(pred)), "expected_rows": int(len(test_meta_reset)),
        "sample_ids_unique": bool(pred[mapping["sample_id"]].is_unique),
        "sample_ids_match_test_order": bool(pred[mapping["sample_id"]].tolist() == test_meta_reset[mapping["sample_id"]].astype(str).tolist()),
        "protein_columns_match_contract": bool(pred.columns.tolist()[1:] == all_proteins),
        "n_protein_columns": int(len(all_proteins)),
        "finite_all_values": bool(np.isfinite(prediction).all()),
        "target_space": "log2(raw)", "standardized_output": False,
    }
    contract["submission_contract"] = "PASS" if all([
        contract["n_rows"] == contract["expected_rows"], contract["sample_ids_unique"],
        contract["sample_ids_match_test_order"], contract["protein_columns_match_contract"],
        contract["finite_all_values"], contract["target_space"] == "log2(raw)", not contract["standardized_output"],
    ]) else "FAIL"
    config = {
        "run_id": "HCCE_mask_compound_3seed_submission_refit_train_only",
        "model": "HCCE-Proteome + A2 masked-entity training (compound-only), 3-seed ensemble",
        "expert_blend": "0.5 legacy FiLM + 0.5 HCCE FiLM; seeds averaged",
        "seeds": SEEDS, "mask_p": MASK_P, "epochs": EPOCHS, "embedding_dim": EMBEDDING_DIM,
        "fit_rows": int(len(fit_indices)), "train_val_rows": int(len(train_meta)),
        "fit_split": "split_final == 'train'",
        "held_out_val_rows": int(len(train_meta) - len(fit_indices)),
        "raw_protein_count": int(len(all_proteins)),
        "model_protein_count": int(len(proteins)), "removed_protein_count": int(len(all_proteins) - len(proteins)),
        "fit_source": (
            "train split labels only; validation rows excluded from training and from all "
            "estimated statistics (protein filter, encoders, target mean/std, fallback means); "
            "test metadata only for inference; test proteome/test truth not loaded"
        ),
        "filtered_protein_fill": "train-split per-protein log2 mean",
        "global_mean_fallback_value": global_train_mean,
        "global_mean_fallback_columns": n_global_fallback,
        "output": str(pred_path), "contract": contract, "elapsed_sec": time.time() - started,
    }
    (output_dir / "SUBMISSION_VALIDATION.json").write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")
    # This is the designated final entry, so it also refreshes the package-root
    # submission artifacts.  The alternative entries deliberately do not.
    pred.to_csv(ROOT / "prediction.csv", index=False)
    root_config = dict(config, output="prediction.csv")
    (ROOT / "SUBMISSION_VALIDATION.json").write_text(json.dumps(json_safe(root_config), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
