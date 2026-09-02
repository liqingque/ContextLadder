#!/usr/bin/env python
"""Refit the validation-selected FiLM model on labeled train_val and predict test.

The test proteome is intentionally not loaded here. Test truth, if evaluated,
belongs in a separate post-hoc script.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_p5_interactions import FiLMModel, fit_preprocessors
from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome


SEED = 20260810
SELECTED_EPOCHS = 37


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return None if not np.isfinite(value) else float(value)
    return value


def main():
    started = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    paths = yaml.safe_load(open(ROOT / "configs/data_paths.yaml", encoding="utf-8")); mapping = yaml.safe_load(open(ROOT / "configs/field_mapping.yaml", encoding="utf-8"))
    train_meta = load_metadata(ROOT / paths["metadata_train_val"]); test_meta = load_metadata(ROOT / paths["metadata_test"])
    train_prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    train_meta, train_prot, _, proteins = align_metadata_proteome(train_meta, train_prot)
    y = to_log2_proteome(finite_float_matrix(train_prot, proteins)).astype(np.float64)
    combined = pd.concat([train_meta, test_meta], ignore_index=True)
    # Train-only fit scope: only `split_final == "train"` rows may be used for
    # training or for estimating any statistic (official data-use constraint).
    train_split_mask = train_meta[mapping["split"]].astype(str).eq("train").to_numpy()
    train_mask = np.concatenate([train_split_mask, np.zeros(len(test_meta), dtype=bool)])
    _, chem_pre, context_pre, chem_train, context_train, chem_all, context_all, chem_cols, context_cat, context_num = fit_preprocessors(combined, mapping, train_mask)
    chem_test = chem_all[len(train_meta):].astype(np.float32); context_test = context_all[len(train_meta):].astype(np.float32)
    train_y = y[train_split_mask]; target_mean = np.nanmean(train_y, axis=0); target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0); target_std = np.nanstd(train_y, axis=0); target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0)
    train_y_filled = np.where(np.isfinite(train_y), train_y, target_mean[None, :]); train_y_z = ((train_y_filled - target_mean) / target_std).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(chem_train.astype(np.float32)), torch.from_numpy(context_train.astype(np.float32)), torch.from_numpy(train_y_z)), batch_size=128, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(SEED))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FiLMModel(chem_train.shape[1], context_train.shape[1], len(proteins)).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); loss_fn = nn.MSELoss(); history = []
    for epoch in range(SELECTED_EPOCHS):
        model.train(); total = 0.0; seen = 0
        for xb_chem, xb_ctx, yb in loader:
            xb_chem = xb_chem.to(device); xb_ctx = xb_ctx.to(device); yb = yb.to(device); optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(xb_chem, xb_ctx), yb); loss.backward(); optimizer.step(); total += float(loss.detach().cpu()) * len(xb_chem); seen += len(xb_chem)
        history.append({"epoch": epoch + 1, "train_loss_z": total / max(1, seen)})
    model.eval(); test_chem_t = torch.from_numpy(chem_test).to(device); test_context_t = torch.from_numpy(context_test).to(device)
    with torch.no_grad(): test_z = model(test_chem_t, test_context_t).detach().cpu().numpy()
    prediction = (test_z * target_std + target_mean).astype(np.float32)
    output_dir = ROOT / "outputs/submission_film"; output_dir.mkdir(parents=True, exist_ok=True)
    pred_frame = pd.DataFrame(prediction, columns=proteins); pred_frame.insert(0, mapping["sample_id"], test_meta[mapping["sample_id"]].astype(str).to_numpy())
    pred_path = output_dir / "prediction.csv"; pred_frame.to_csv(pred_path, index=False)  # alternative entry: does not overwrite the frozen root prediction.csv
    contract = {"sample_id_column": mapping["sample_id"], "n_rows": int(len(pred_frame)), "expected_rows": int(len(test_meta)), "sample_ids_unique": bool(pred_frame[mapping["sample_id"]].is_unique), "sample_ids_match_test_order": bool(pred_frame[mapping["sample_id"]].tolist() == test_meta[mapping["sample_id"]].astype(str).tolist()), "protein_columns_match_contract": bool(pred_frame.columns.tolist()[1:] == proteins), "n_protein_columns": int(len(proteins)), "finite_all_values": bool(np.isfinite(prediction).all()), "target_space": "log2(raw)", "standardized_output": False, "submission_contract": "PASS" if len(pred_frame) == len(test_meta) and pred_frame[mapping["sample_id"]].is_unique and pred_frame[mapping["sample_id"]].tolist() == test_meta[mapping["sample_id"]].astype(str).tolist() and pred_frame.columns.tolist()[1:] == proteins and np.isfinite(prediction).all() else "FAIL"}
    config = {"run_id": "SUBMISSION_FiLM_refit_train_only_seed20260810", "model": "SmallMLP_FiLM_ID_context", "seed": SEED, "device": str(device), "fit_rows": int(train_split_mask.sum()), "fit_split": "split_final == 'train'", "fit_source": "train split labels only; validation rows excluded from training and all estimated statistics; no test proteome or test truth loaded", "selected_epoch_source": "P5 frozen-validation FiLM best_epoch=37", "epochs": SELECTED_EPOCHS, "feature_contract": "chemical=compound ID one-hot; context=strain/medium/source/instrument/plate ID one-hot plus standardized temperature/time; preprocessors fit on train split metadata only", "target": "absolute_log2_proteome", "scale": "log2(raw)", "target_missing_policy": "train split observed protein mean", "output": str(pred_path), "contract": contract, "elapsed_sec": time.time() - started}
    (output_dir / "SUBMISSION_VALIDATION.json").write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")  # alternative entry: does not overwrite the frozen root SUBMISSION_VALIDATION.json
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False); torch.save({"model": model.state_dict(), "proteins": proteins, "selected_epochs": SELECTED_EPOCHS}, output_dir / "model_final.pt"); pd.DataFrame({"protein": proteins, "target_mean": target_mean, "target_std": target_std}).to_csv(output_dir / "target_stats.csv", index=False)
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f: yaml.safe_dump(json_safe(config), f, allow_unicode=True, sort_keys=False)
    with open(output_dir / "run.log", "w", encoding="utf-8") as f: f.write("elapsed_sec=%.6f\npeak_gpu_memory_mb=%.3f\n" % (time.time() - started, torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0))
    print(json.dumps(json_safe(config), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
