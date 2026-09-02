#!/usr/bin/env python
"""Run P5 FiLM or low-rank bilinear interaction models on reliable metadata."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baselines import json_safe, make_features
from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome
from src.evaluation.control_matching import matched_fc
from src.evaluation.evaluator import evaluate_basic_and_splits


SEED = 20260810


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FiLMModel(nn.Module):
    def __init__(self, n_chem, n_context, n_out):
        super().__init__()
        self.chem = nn.Sequential(nn.Linear(n_chem, 256), nn.GELU(), nn.Dropout(0.1))
        self.context = nn.Sequential(nn.Linear(n_context, 256), nn.GELU(), nn.Dropout(0.1))
        self.modulation = nn.Sequential(nn.Linear(n_context, 256), nn.GELU(), nn.Linear(256, 512))
        self.head = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out))

    def forward(self, chem, context):
        h_chem = self.chem(chem)
        h_context = self.context(context)
        gamma, beta = self.modulation(context).chunk(2, dim=-1)
        h = (1.0 + gamma) * h_chem + beta
        # Keep a direct context path so FiLM cannot discard metadata that is
        # predictive even when chemical identity is unseen.
        return self.head(h) + 0.05 * self.head(h_context)


class BilinearModel(nn.Module):
    def __init__(self, n_chem, n_context, n_out, rank=128):
        super().__init__()
        self.rank = rank
        self.chem = nn.Sequential(nn.Linear(n_chem, rank), nn.GELU(), nn.Dropout(0.1))
        self.context = nn.Sequential(nn.Linear(n_context, rank), nn.GELU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(3 * rank, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out))

    def forward(self, chem, context):
        z_chem = self.chem(chem)
        z_context = self.context(context)
        return self.head(torch.cat([z_chem, z_context, z_chem * z_context], dim=-1))


def fit_preprocessors(meta, mapping, train_mask):
    frame, categorical, numeric = make_features(meta, mapping, "full")
    chem_cols = ["compound"]
    context_cat = [c for c in categorical if c not in chem_cols]
    context_num = list(numeric)
    chem_pre = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), chem_cols)], remainder="drop")
    context_transformers = []
    if context_cat:
        context_transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), context_cat))
    if context_num:
        context_transformers.append(("numeric", StandardScaler(), context_num))
    context_pre = ColumnTransformer(context_transformers, remainder="drop")
    chem_train = chem_pre.fit_transform(frame.loc[train_mask])
    context_train = context_pre.fit_transform(frame.loc[train_mask])
    chem_all = chem_pre.transform(frame)
    context_all = context_pre.transform(frame)
    to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return frame, chem_pre, context_pre, to_dense(chem_train), to_dense(context_train), to_dense(chem_all), to_dense(context_all), chem_cols, context_cat, context_num


def finite_val_mse(pred, true):
    mask = np.isfinite(true)
    return float(np.mean((pred[mask] - true[mask]) ** 2)) if mask.any() else float("inf")


def main():
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["film", "bilinear"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    SEED = args.seed
    seed_everything(SEED)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("Requested CUDA but CUDA is unavailable")
    started = time.time()
    paths = yaml.safe_load(open(ROOT / "configs/data_paths.yaml", encoding="utf-8"))
    mapping = yaml.safe_load(open(ROOT / "configs/field_mapping.yaml", encoding="utf-8"))
    meta = load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, proteins = align_metadata_proteome(meta, prot)
    y = to_log2_proteome(finite_float_matrix(prot, proteins))
    train_mask = meta[mapping["split"]].astype(str).eq("train").to_numpy()
    val_mask = ~train_mask
    val_meta = meta.loc[val_mask].reset_index(drop=True)
    val_true = y[val_mask]
    frame, chem_pre, context_pre, chem_train, context_train, chem_all, context_all, chem_cols, context_cat, context_num = fit_preprocessors(meta, mapping, train_mask)
    train_y = y[train_mask].astype(np.float64)
    target_mean = np.nanmean(train_y, axis=0)
    target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0)
    target_std = np.nanstd(train_y, axis=0)
    target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0)
    train_y_z = ((np.where(np.isfinite(train_y), train_y, target_mean[None, :]) - target_mean) / target_std).astype(np.float32)
    n_train = chem_train.shape[0]
    if args.smoke:
        n_train = min(100, n_train)
    dataset = TensorDataset(torch.from_numpy(chem_train[:n_train].astype(np.float32)), torch.from_numpy(context_train[:n_train].astype(np.float32)), torch.from_numpy(train_y_z[:n_train]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED))
    if args.mode == "film":
        model = FiLMModel(chem_train.shape[1], context_train.shape[1], len(proteins)).to(device)
        model_name = "SmallMLP_FiLM_ID_context"
    else:
        model = BilinearModel(chem_train.shape[1], context_train.shape[1], len(proteins), rank=128).to(device)
        model_name = "SmallMLP_LowRankBilinear_ID_context"
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    chem_val_t = torch.from_numpy(chem_all[val_mask].astype(np.float32)).to(device)
    context_val_t = torch.from_numpy(context_all[val_mask].astype(np.float32)).to(device)
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if args.smoke else args.epochs
    best_loss, best_epoch, patience_left = float("inf"), -1, args.patience
    history = []
    for epoch in range(max_epochs):
        model.train(); train_loss = 0.0; n_seen = 0
        for xb_chem, xb_ctx, yb in loader:
            xb_chem = xb_chem.to(device); xb_ctx = xb_ctx.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb_chem, xb_ctx), yb)
            loss.backward(); optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(xb_chem); n_seen += len(xb_chem)
        model.eval()
        with torch.no_grad():
            val_z = model(chem_val_t, context_val_t).detach().cpu().numpy()
        val_pred = val_z * target_std + target_mean
        val_loss = finite_val_mse(val_pred, val_true)
        history.append({"epoch": epoch + 1, "train_loss_z": train_loss / max(1, n_seen), "val_mse_log2": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, patience_left = val_loss, epoch + 1, args.patience
            torch.save({"model": model.state_dict(), "mode": args.mode, "n_chem": chem_train.shape[1], "n_context": context_train.shape[1], "n_out": len(proteins), "epoch": best_epoch}, output_dir / "best_model.pt")
        elif not args.smoke:
            patience_left -= 1
            if patience_left <= 0:
                break
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    with torch.no_grad():
        pred_z = model(chem_val_t, context_val_t).detach().cpu().numpy()
    pred = (pred_z * target_std + target_mean).astype(np.float32)
    evaluated = evaluate_basic_and_splits(pred, val_true, val_meta, split_col=mapping["split"], sample_ids=val_meta[mapping["sample_id"]].astype(str))
    fc, matches = matched_fc(pred, val_true, val_meta, np.ones(len(val_meta), dtype=bool), mapping)
    evaluated["overall"]["fc_pcc"] = fc.get("fc_pcc"); evaluated["overall"]["fc_coverage"] = fc.get("coverage")
    config = {"run_id": (("P5_smoke_%s_seed%d" if args.smoke else "P5_%s_seed%d") % (args.mode, SEED)), "model": model_name, "mode": args.mode, "seed": SEED, "device": str(device), "fit_split": "split_final=train", "eval_split": "validation", "feature_contract": "chemical=compound ID one-hot; context=strain/medium/source/instrument/plate ID one-hot plus standardized temperature/time", "chemical_input_dim": int(chem_train.shape[1]), "context_input_dim": int(context_train.shape[1]), "context_categorical": context_cat, "context_numeric": context_num, "target": "absolute_log2_proteome", "scale": "log2(raw)", "hidden": ([256, 512, 256] if args.mode == "film" else [128, 128, 128, 256, 512, 256]), "bilinear_rank": (128 if args.mode == "bilinear" else None), "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4, "batch_size": args.batch_size, "max_epochs": max_epochs, "early_stopping_patience": args.patience, "best_epoch": best_epoch, "best_val_mse_log2": best_loss, "target_missing_policy": "train-only observed protein mean", "smoke": args.smoke, "metrics": evaluated["overall"]}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe({"overall": evaluated["overall"], "fc": fc, "config": config}), f, ensure_ascii=False, indent=2)
    evaluated["by_split"].to_csv(output_dir / "metrics_by_split.csv", index=False)
    evaluated["samples"].to_csv(output_dir / "sample_metrics.csv", index=False)
    matches.to_csv(output_dir / "control_match_coverage.csv", index=False)
    prediction = pd.DataFrame(pred, columns=proteins)
    prediction.insert(0, mapping["sample_id"], val_meta[mapping["sample_id"]].astype(str).to_numpy())
    prediction.to_parquet(output_dir / "prediction_val.parquet", index=False)
    joblib.dump({"chemical": chem_pre, "context": context_pre}, output_dir / "preprocessors.joblib")
    np.save(output_dir / "target_mean.npy", target_mean); np.save(output_dir / "target_std.npy", target_std)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(json_safe(config), f, allow_unicode=True, sort_keys=False)
    with open(output_dir / "run.log", "w", encoding="utf-8") as f:
        f.write("elapsed_sec=%.6f\npeak_gpu_memory_mb=%.3f\n" % (time.time() - started, torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0))
    if not args.smoke:
        leaderboard_path = ROOT / "outputs/LEADERBOARD_LOCAL.csv"
        leaderboard = pd.read_csv(leaderboard_path)
        by = evaluated["by_split"]
        get_split = lambda label: float(by.loc[by["split"] == label, "abs_pcc"].iloc[0]) if (by["split"] == label).any() else "NA"
        leaderboard = pd.concat([leaderboard, pd.DataFrame([{"run_id": config["run_id"], "model": model_name, "feature_set": "full_metadata_ids_%s" % args.mode, "target_type": "absolute", "seed": SEED, "abs_pcc": evaluated["overall"].get("abs_pcc"), "abs_r2": evaluated["overall"].get("abs_r2"), "rmse": evaluated["overall"].get("rmse"), "fc_pcc": evaluated["overall"].get("fc_pcc"), "chem_residual_pcc": "NA", "strain_residual_pcc": "NA", "dep_auprc": "NA", "s1_score": get_split("val_chem_only"), "s2_score": get_split("val_strain_only"), "s3_score": get_split("val_both"), "time_score": get_split("val_time"), "runtime_sec": time.time() - started, "notes": "P5 interaction; external chemical/genome features not used; evaluator SPEC_APPROX"}])], ignore_index=True)
        leaderboard.to_csv(leaderboard_path, index=False)
    print(json.dumps(json_safe({"config": config, "fc": fc}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
