#!/usr/bin/env python
"""Train one fixed-budget OOF Concat/FiLM expert on an inner train-only fold."""

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
from scripts.run_p5_interactions import FiLMModel, fit_preprocessors
from scripts.run_p5_mlp import SmallMLP
from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome
from src.evaluation.evaluator import basic_metrics


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-type", required=True, choices=["compound", "strain", "both", "time", "plate"])
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--mode", required=True, choices=["concat", "film"])
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda": raise RuntimeError("Requested CUDA but CUDA unavailable")
    started = time.time()
    paths = yaml.safe_load(open(ROOT / "configs/data_paths.yaml", encoding="utf-8")); mapping = yaml.safe_load(open(ROOT / "configs/field_mapping.yaml", encoding="utf-8"))
    meta = load_metadata(ROOT / paths["metadata_train_val"]); prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"]); meta, prot, _, proteins = align_metadata_proteome(meta, prot); y = to_log2_proteome(finite_float_matrix(prot, proteins))
    fold_table = pd.read_csv(ROOT / "outputs/biocal_moe/inner_folds.csv", dtype={"sample_ID": str, "fold_type": str, "fold_id": str, "role": str})
    selected = fold_table[(fold_table["fold_type"] == args.fold_type) & (fold_table["fold_id"] == str(args.fold_id))].copy()
    role = selected.set_index("sample_ID")["role"].to_dict(); sample_ids = meta[mapping["sample_id"]].astype(str).to_numpy(); train_base = meta[mapping["split"]].astype(str).eq("train").to_numpy(); train_mask = train_base & np.array([role.get(s) == "train" for s in sample_ids]); hold_mask = train_base & np.array([role.get(s) == "holdout" for s in sample_ids])
    if not train_mask.any() or not hold_mask.any(): raise RuntimeError("Empty train/holdout in fold")
    val_meta = meta.loc[hold_mask].reset_index(drop=True); true = y[hold_mask]
    if args.mode == "concat":
        frame, categorical, numeric = make_features(meta, mapping, "full"); transformers = []
        if categorical: transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
        if numeric: transformers.append(("numeric", StandardScaler(), numeric))
        pre = ColumnTransformer(transformers=transformers, remainder="drop"); x_train_sp = pre.fit_transform(frame.loc[train_mask]); x_all_sp = pre.transform(frame); to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else np.asarray(x); x_train = to_dense(x_train_sp); x_hold = to_dense(x_all_sp[hold_mask]); model = SmallMLP(x_train.shape[1], len(proteins)).to(device); pre_save = pre; feature_info = {"input_dim": int(x_train.shape[1]), "categorical": categorical, "numeric": numeric}
        dataset_x = x_train
    else:
        _, chem_pre, context_pre, chem_train, context_train, chem_all, context_all, chem_cols, context_cat, context_num = fit_preprocessors(meta, mapping, train_mask); chem_hold = chem_all[hold_mask]; context_hold = context_all[hold_mask]; model = FiLMModel(chem_train.shape[1], context_train.shape[1], len(proteins)).to(device); pre_save = {"chemical": chem_pre, "context": context_pre}; feature_info = {"chemical_input_dim": int(chem_train.shape[1]), "context_input_dim": int(context_train.shape[1]), "context_categorical": context_cat, "context_numeric": context_num}; dataset_x = (chem_train, context_train)
    train_y = y[train_mask].astype(np.float64); target_mean = np.nanmean(train_y, axis=0); target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0); target_std = np.nanstd(train_y, axis=0); target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0); train_y_z = ((np.where(np.isfinite(train_y), train_y, target_mean[None, :]) - target_mean) / target_std).astype(np.float32)
    n_train = min(100, int(train_mask.sum())) if args.smoke else int(train_mask.sum()); max_epochs = 1 if args.smoke else args.epochs
    if args.mode == "concat": dataset = TensorDataset(torch.from_numpy(dataset_x[:n_train].astype(np.float32)), torch.from_numpy(train_y_z[:n_train]))
    else: dataset = TensorDataset(torch.from_numpy(dataset_x[0][:n_train].astype(np.float32)), torch.from_numpy(dataset_x[1][:n_train].astype(np.float32)), torch.from_numpy(train_y_z[:n_train]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed)); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); loss_fn = nn.MSELoss()
    for _ in range(max_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if args.mode == "concat": xb, yb = batch; pred = model(xb.to(device)); yb = yb.to(device)
            else: xb_chem, xb_ctx, yb = batch; pred = model(xb_chem.to(device), xb_ctx.to(device)); yb = yb.to(device)
            loss = loss_fn(pred, yb); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        if args.mode == "concat": pred_z = model(torch.from_numpy(x_hold.astype(np.float32)).to(device)).cpu().numpy()
        else: pred_z = model(torch.from_numpy(chem_hold.astype(np.float32)).to(device), torch.from_numpy(context_hold.astype(np.float32)).to(device)).cpu().numpy()
    pred = (pred_z * target_std + target_mean).astype(np.float32); metrics = basic_metrics(pred, true)
    output = ROOT / args.output; output.mkdir(parents=True, exist_ok=True); pred_frame = pd.DataFrame(pred, columns=proteins); pred_frame.insert(0, mapping["sample_id"], val_meta[mapping["sample_id"]].astype(str).to_numpy()); pred_frame.to_parquet(output / "prediction_oof.parquet", index=False)
    rows = pd.DataFrame({"sample_ID": val_meta[mapping["sample_id"]].astype(str), "fold_type": args.fold_type, "fold_id": str(args.fold_id), "mode": args.mode, "seed": args.seed, "n_train": int(train_mask.sum()), "n_holdout": int(hold_mask.sum())}); rows.to_csv(output / "oof_rows.csv", index=False)
    config = {"run_id": "OOF_%s_%s_%s_seed%d" % (args.fold_type, args.fold_id, args.mode, args.seed), "fold_type": args.fold_type, "fold_id": str(args.fold_id), "mode": args.mode, "seed": args.seed, "device": str(device), "fit_scope": "official train rows only; inner fold train role", "heldout_scope": "official train rows only; inner fold holdout role", "epochs": max_epochs, "batch_size": args.batch_size, "feature_info": feature_info, "target_policy": "fold-train-only observed protein mean/std", "metrics_oof": metrics, "n_train": int(train_mask.sum()), "n_holdout": int(hold_mask.sum()), "elapsed_sec": time.time() - started}
    (output / "metrics.json").write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8"); (output / "run_config.yaml").write_text(yaml.safe_dump(json_safe(config), allow_unicode=True, sort_keys=False), encoding="utf-8"); joblib.dump(pre_save, output / "preprocessor.joblib"); np.save(output / "target_mean.npy", target_mean); np.save(output / "target_std.npy", target_std); torch.save({"model": model.state_dict(), "mode": args.mode, "seed": args.seed}, output / "model.pt")
    (output / "run.log").write_text("elapsed_sec=%.6f\npeak_gpu_memory_mb=%.3f\n" % (time.time() - started, torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0), encoding="utf-8")
    print(json.dumps(json_safe(config), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
