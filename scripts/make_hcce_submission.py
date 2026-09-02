#!/usr/bin/env python
"""Refit HCCE-Proteome on the `train` split only and predict test.

Historical single-seed baseline entry; the final submission is
make_mask_compound_3seed_submission.py. The candidate uses the train-only
validated 50/50 blend of the legacy FiLM expert and the HCCE FiLM expert.
Test proteome/test truth are never read.

Same official data constraint as the final entry: only `split_final == "train"`
rows may be used for training or for estimating any statistic.
"""

import json
import random
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

from scripts.run_hcce import apply_official_protein_filter, fit_model, predict_model
from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome


SEED = 20260810
EPOCHS = 40
EMBEDDING_DIM = 64


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
    started = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    train_meta = load_metadata(ROOT / paths["metadata_train_val"])
    test_meta = load_metadata(ROOT / paths["metadata_test"])
    train_prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    train_meta, train_prot, _, all_proteins = align_metadata_proteome(train_meta, train_prot)
    raw_train_all = finite_float_matrix(train_prot, all_proteins)
    proteins, missing_rate, keep_mask = apply_official_protein_filter(train_meta, raw_train_all, all_proteins, mapping, threshold=0.80)
    train_y = to_log2_proteome(raw_train_all[:, keep_mask]).astype(np.float64)
    combined = pd.concat([train_meta, test_meta], ignore_index=True)
    combined_y = np.full((len(combined), len(proteins)), np.nan, dtype=np.float64)
    combined_y[:len(train_meta)] = train_y
    # Train-only fit scope: validation rows never enter training or any statistic.
    train_split_mask = train_meta[mapping["split"]].astype(str).eq("train").to_numpy()
    fit_indices = np.flatnonzero(train_split_mask).astype(int)
    model, encoder, target_mean, target_std, history = fit_model(
        combined, combined_y, fit_indices, mapping, SEED, EPOCHS, device, EMBEDDING_DIM,
    )
    test_meta_reset = test_meta.reset_index(drop=True)
    _, film_test, legacy_test = predict_model(model, encoder, target_mean, target_std, test_meta_reset, device)
    filtered_prediction = (0.5 * legacy_test + 0.5 * film_test).astype(np.float32)
    log2_train_only = to_log2_proteome(raw_train_all[train_split_mask]).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        train_mean_all = np.nanmean(log2_train_only, axis=0)
        global_train_mean = float(np.nanmean(log2_train_only))
    train_mean_all = np.where(np.isfinite(train_mean_all), train_mean_all, global_train_mean)
    prediction = np.tile(train_mean_all.astype(np.float32), (len(test_meta_reset), 1))
    prediction[:, keep_mask] = filtered_prediction
    output_dir = ROOT / "outputs/submission_hcce"
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(prediction, columns=all_proteins)
    pred.insert(0, mapping["sample_id"], test_meta_reset[mapping["sample_id"]].astype(str).to_numpy())
    pred_path = output_dir / "prediction.csv"
    pred.to_csv(pred_path, index=False)
    filtered_pred = pd.DataFrame(filtered_prediction, columns=proteins)
    filtered_pred.insert(0, mapping["sample_id"], test_meta_reset[mapping["sample_id"]].astype(str).to_numpy())
    filtered_pred.to_csv(output_dir / f"prediction_filtered_{len(proteins)}.csv", index=False)
    contract = {
        "sample_id_column": mapping["sample_id"],
        "n_rows": int(len(pred)),
        "expected_rows": int(len(test_meta_reset)),
        "sample_ids_unique": bool(pred[mapping["sample_id"]].is_unique),
        "sample_ids_match_test_order": bool(pred[mapping["sample_id"]].tolist() == test_meta_reset[mapping["sample_id"]].astype(str).tolist()),
        "protein_columns_match_contract": bool(pred.columns.tolist()[1:] == all_proteins),
        "n_protein_columns": int(len(all_proteins)),
        "finite_all_values": bool(np.isfinite(prediction).all()),
        "target_space": "log2(raw)",
        "standardized_output": False,
    }
    contract["submission_contract"] = "PASS" if all([
        contract["n_rows"] == contract["expected_rows"], contract["sample_ids_unique"],
        contract["sample_ids_match_test_order"], contract["protein_columns_match_contract"],
        contract["finite_all_values"], contract["target_space"] == "log2(raw)",
        not contract["standardized_output"],
    ]) else "FAIL"
    config = {
        "run_id": "HCCE_submission_refit_train_only_seed20260810",
        "model": "HCCE-Proteome",
        "expert_blend": "0.5 legacy FiLM + 0.5 HCCE FiLM; selected by train-only three-seed/OOF evidence",
        "seed": SEED, "device": str(device), "epochs": EPOCHS, "embedding_dim": EMBEDDING_DIM,
        "fit_rows": int(len(fit_indices)), "fit_split": "split_final == 'train'", "raw_protein_count": int(len(all_proteins)), "model_protein_count": int(len(proteins)), "removed_protein_count": int(len(all_proteins) - len(proteins)), "missing_threshold": 0.80,
        "fit_source": "train split labels only; validation rows excluded from training and all estimated statistics; test metadata only for inference; test proteome/test truth not loaded",
        "feature_contract": "unknown-safe categorical maps; biological strain/medium/time/temperature; measurement source/instrument/plate; continuous time bases",
        "target": "absolute_log2_proteome", "scale": "log2(raw)",
        "output": str(pred_path), "contract": contract, "elapsed_sec": time.time() - started,
    }
    (output_dir / "SUBMISSION_VALIDATION.json").write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    torch.save({"model": model.state_dict(), "proteins": proteins, "all_proteins": all_proteins, "keep_mask": keep_mask, "target_mean": target_mean, "target_std": target_std, "epochs": EPOCHS, "embedding_dim": EMBEDDING_DIM}, output_dir / "model_final.pt")
    np.save(output_dir / "target_mean.npy", target_mean)
    np.save(output_dir / "target_std.npy", target_std)
    print(json.dumps(json_safe(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
