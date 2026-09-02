#!/usr/bin/env python
"""Run train-mean and metadata Ridge baselines on frozen validation."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome
from src.evaluation.control_matching import matched_fc
from src.evaluation.evaluator import evaluate_basic_and_splits


SEED = 20260810


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def metrics_for(pred, true, meta, sample_ids, mapping):
    evaluated = evaluate_basic_and_splits(pred, true, meta, split_col=mapping["split"], sample_ids=sample_ids)
    eval_mask = np.ones(len(meta), dtype=bool)
    fc, matches = matched_fc(pred, true, meta, eval_mask, mapping)
    evaluated["overall"]["fc_pcc"] = fc.get("fc_pcc")
    evaluated["overall"]["fc_coverage"] = fc.get("coverage")
    evaluated["fc"] = fc
    evaluated["fc_matches"] = matches
    return evaluated


def save_prediction(path, sample_ids, protein_columns, prediction):
    frame = pd.DataFrame(prediction, columns=protein_columns)
    frame.insert(0, "sample_ID", np.asarray(sample_ids, dtype=str))
    frame.to_parquet(path, index=False)


def write_metrics(folder, evaluated, config):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"overall": evaluated["overall"], "fc": evaluated.get("fc"), "config": config}), f, ensure_ascii=False, indent=2)
    evaluated["by_split"].to_csv(folder / "metrics_by_split.csv", index=False)
    evaluated["samples"].to_csv(folder / "sample_metrics.csv", index=False)
    evaluated.get("fc_matches", pd.DataFrame()).to_csv(folder / "control_match_coverage.csv", index=False)
    with open(folder / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(json_safe(config), f, allow_unicode=True, sort_keys=False)


def make_features(meta, mapping, group):
    groups = yaml.safe_load(open(ROOT / "configs/field_mapping.yaml", encoding="utf-8"))["feature_groups"]
    keys = groups[group]
    frame = pd.DataFrame(index=meta.index)
    categorical, numeric = [], []
    for key in keys:
        col = mapping.get(key)
        if not col or col not in meta.columns:
            continue
        if key in ("temperature", "time"):
            frame[key] = pd.to_numeric(meta[col], errors="coerce")
            numeric.append(key)
        else:
            frame[key] = meta[col].fillna("<NA>").astype(str)
            categorical.append(key)
    if numeric and frame[numeric].isna().any().any():
        raise ValueError("Missing numeric metadata in feature table; inspect data before fitting")
    return frame, categorical, numeric


def fit_ridge(meta, y, train_mask, val_mask, mapping, group, alpha):
    X, categorical, numeric = make_features(meta, mapping, group)
    transformers = []
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    y_train = y[train_mask].astype(np.float64)
    observed = np.isfinite(y_train)
    observed_count = observed.sum(axis=0)
    target_mean = np.nanmean(y_train, axis=0)
    target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0)
    target_std = np.nanstd(y_train, axis=0)
    target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0)
    # Explicit target-side handling: only train-observed means are used; no
    # validation/test value is imputed or used to fit the model.
    y_train_filled = np.where(observed, y_train, target_mean[None, :])
    y_train_z = ((y_train_filled - target_mean) / target_std).astype(np.float32)
    model = Pipeline([("preprocessor", preprocessor), ("ridge", Ridge(alpha=alpha, solver="lsqr"))])
    model.fit(X.loc[train_mask], y_train_z)
    pred_z = model.predict(X.loc[val_mask])
    pred = pred_z * target_std + target_mean
    info = {"categorical": categorical, "numeric": numeric, "n_input_features": int(X.shape[1]), "alpha": alpha, "train_target_observed_min": int(observed_count.min()), "train_target_observed_max": int(observed_count.max()), "train_target_missing_cells": int((~observed).sum()), "target_missing_policy": "train-only observed protein mean"}
    return model, pred.astype(np.float32), target_mean, target_std, info


def run_train_mean(meta, y, train_mask, val_mask, protein_columns, mapping):
    folder = ROOT / "outputs/p1_baselines/train_mean"
    started = time.time()
    train_mean = np.nanmean(y[train_mask], axis=0)
    train_mean = np.where(np.isfinite(train_mean), train_mean, 0.0).astype(np.float32)
    pred = np.repeat(train_mean[None, :], int(val_mask.sum()), axis=0)
    val_meta = meta.loc[val_mask].reset_index(drop=True)
    val_true = y[val_mask]
    evaluated = metrics_for(pred, val_true, val_meta, val_meta[mapping["sample_id"]].astype(str), mapping)
    config = {"run_id": "P1A_train_mean_seed20260810", "model": "TrainMean", "seed": SEED, "fit_split": "split_final=train", "eval_split": "validation", "target": "absolute_log2_proteome", "scale": "log2(raw)", "train_only_target_mean": True, "target_missing_policy": "train-only observed protein mean; all-missing train proteins set to 0 and reported"}
    write_metrics(folder, evaluated, config)
    save_prediction(folder / "prediction_val.parquet", val_meta[mapping["sample_id"]], protein_columns, pred)
    np.save(folder / "train_protein_mean.npy", train_mean)
    with open(folder / "run.log", "w", encoding="utf-8") as f:
        f.write("elapsed_sec=%.6f\n" % (time.time() - started))
    return {"model": "TrainMean", "folder": folder, "metrics": evaluated["overall"], "by_split": evaluated["by_split"], "config": config}


def run_ridge_group(meta, y, train_mask, val_mask, protein_columns, mapping, group):
    folder = ROOT / ("outputs/p1_baselines/ridge_%s" % group)
    folder.mkdir(parents=True, exist_ok=True)
    val_meta = meta.loc[val_mask].reset_index(drop=True)
    val_true = y[val_mask]
    rows = []
    best = None
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        started = time.time()
        model, pred, target_mean, target_std, feature_info = fit_ridge(meta, y, train_mask, val_mask, mapping, group, alpha)
        evaluated = metrics_for(pred, val_true, val_meta, val_meta[mapping["sample_id"]].astype(str), mapping)
        row = {"alpha": alpha, "runtime_sec": time.time() - started, **evaluated["overall"], "fc_pcc": evaluated["fc"].get("fc_pcc"), "fc_coverage": evaluated["fc"].get("coverage")}
        rows.append(row)
        current_score = row.get("abs_pcc")
        best_score = None if best is None else best["row"].get("abs_pcc")
        if current_score is not None and (best is None or best_score is None or current_score > best_score):
            best = {"row": row, "model": model, "pred": pred, "target_mean": target_mean, "target_std": target_std, "feature_info": feature_info, "evaluated": evaluated}
    pd.DataFrame(rows).to_csv(folder / "alpha_grid.csv", index=False)
    if best is None:
        raise RuntimeError("No Ridge alpha completed")
    config = {"run_id": "P1B_ridge_%s_seed20260810_alpha%s" % (group, str(best["row"]["alpha"]).replace(".", "p")), "model": "MetadataRidge", "feature_group": group, "seed": SEED, "fit_split": "split_final=train", "eval_split": "validation", "alpha_grid": [0.1, 1.0, 10.0, 100.0], "selected_alpha": best["row"]["alpha"], "target": "absolute_log2_proteome", "scale": "log2(raw)", "target_standardization": "train-only observed mean/std", "feature_info": best["feature_info"]}
    write_metrics(folder, best["evaluated"], config)
    save_prediction(folder / "prediction_val.parquet", val_meta[mapping["sample_id"]], protein_columns, best["pred"])
    joblib.dump(best["model"], folder / "model.joblib")
    np.save(folder / "target_mean.npy", best["target_mean"])
    np.save(folder / "target_std.npy", best["target_std"])
    with open(folder / "run.log", "w", encoding="utf-8") as f:
        f.write(json.dumps(json_safe({"config": config, "selected_metrics": best["row"]}), ensure_ascii=False, indent=2))
    return {"model": "Ridge_%s" % group, "folder": folder, "metrics": best["evaluated"]["overall"], "by_split": best["evaluated"]["by_split"], "config": config}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["all", "mean", "ridge"], default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = yaml.safe_load(open(ROOT / "configs/data_paths.yaml", encoding="utf-8"))
    mapping = yaml.safe_load(open(ROOT / "configs/field_mapping.yaml", encoding="utf-8"))
    meta = load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, protein_columns = align_metadata_proteome(meta, prot)
    y = to_log2_proteome(finite_float_matrix(prot, protein_columns))
    if not np.isfinite(y).all():
        logging.info("Proteome contains missing target cells; using the explicit train-only target policy recorded in each run config")
    train_mask = meta[mapping["split"]].astype(str).eq("train").to_numpy()
    val_mask = ~train_mask
    records = []
    if args.only in ("all", "mean"):
        records.append(run_train_mean(meta, y, train_mask, val_mask, protein_columns, mapping))
    if args.only in ("all", "ridge"):
        for group in ("biological", "measurement", "full"):
            records.append(run_ridge_group(meta, y, train_mask, val_mask, protein_columns, mapping, group))
    leaderboard = []
    for rec in records:
        m = rec["metrics"]
        leaderboard.append({"run_id": rec["config"]["run_id"], "model": rec["model"], "feature_set": rec["config"].get("feature_group", "train_mean"), "target_type": "absolute", "seed": SEED, "abs_pcc": m.get("abs_pcc"), "abs_r2": m.get("abs_r2"), "rmse": m.get("rmse"), "fc_pcc": m.get("fc_pcc"), "chem_residual_pcc": "NA", "strain_residual_pcc": "NA", "dep_auprc": "NA", "s1_score": "NA", "s2_score": "NA", "s3_score": "NA", "time_score": "NA", "runtime_sec": "NA", "notes": "validation-only first-round baseline"})
    pd.DataFrame(leaderboard).to_csv(ROOT / "outputs/LEADERBOARD_LOCAL.csv", index=False)
    print(json.dumps(json_safe(leaderboard), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
