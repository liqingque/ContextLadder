#!/usr/bin/env python
"""Train and evaluate HCCE-Proteome.

HCCE combines an unknown-safe hierarchical context encoder, continuous-time
features, complementary Concat/FiLM experts, an OOF shrinkage gate, and a
small source/instrument residual calibrator.  All fitting is restricted to
the supplied fit indices for each fold; validation/test protein truth is not
used by the model or the gate.
"""

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
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_baselines import json_safe
from src.data.io import align_metadata_proteome, finite_float_matrix, load_metadata, load_proteome, to_log2_proteome
from src.evaluation.control_matching import matched_fc
from src.evaluation.evaluator import basic_metrics, evaluate_basic_and_splits
from src.evaluation.official_metrics import official_absolute_metrics
from src.precision_weights import estimate_precision_weights


SEED_DEFAULT = 20260810
CAT_FIELDS = ["compound", "strain", "medium", "source", "instrument", "plate"]
BIO_FIELDS = ["strain", "medium"]
MEASURE_FIELDS = ["source", "instrument", "plate"]
GATE_FEATURES = [
    "compound_seen", "strain_seen", "time_seen", "plate_seen", "source_seen",
    "instrument_seen", "time_distance", "expert_disagreement",
]
OOF_TYPES = ["compound", "strain", "both", "time", "plate"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_official_protein_filter(metadata, raw_proteome, protein_columns, mapping, threshold=0.80):
    """Keep proteins with train-only missing rate strictly below 80%."""
    train_mask = metadata[mapping["split"]].astype(str).eq("train").to_numpy()
    raw_train = np.asarray(raw_proteome[train_mask], dtype=np.float32)
    # Official scale is log2 intensity; raw non-positive values therefore
    # become missing observations as well as explicit CSV NA values.
    log2_train = to_log2_proteome(raw_train)
    missing_rate = np.mean(~np.isfinite(log2_train), axis=0)
    keep = np.isfinite(missing_rate) & (missing_rate < threshold)
    keep_columns = [p for p, flag in zip(protein_columns, keep) if flag]
    return keep_columns, missing_rate, keep


class HCCEMetaEncoder:
    """Train-only categorical maps plus continuous time/temperature bases."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.maps = {}
        self.numeric_mean = None
        self.numeric_std = None
        self.rbf_centers = None
        self.rbf_width = None

    def fit(self, meta, fit_indices):
        fit = meta.iloc[np.asarray(fit_indices)]
        for field in CAT_FIELDS:
            col = self.mapping[field]
            values = fit[col].fillna("<NA>").astype(str).tolist()
            self.maps[field] = {v: i + 1 for i, v in enumerate(sorted(set(values)))}
        time_values = pd.to_numeric(fit[self.mapping["time"]], errors="coerce").to_numpy(dtype=float)
        temp_values = pd.to_numeric(fit[self.mapping["temperature"]], errors="coerce").to_numpy(dtype=float)
        log_time = np.log1p(np.maximum(time_values, 0.0))
        raw = np.column_stack([time_values, temp_values, log_time])
        self.numeric_mean = np.nanmean(raw, axis=0)
        self.numeric_std = np.nanstd(raw, axis=0)
        self.numeric_std = np.where(np.isfinite(self.numeric_std) & (self.numeric_std >= 1e-8), self.numeric_std, 1.0)
        t_min = float(np.nanmin(time_values)) if len(time_values) else 0.0
        t_max = float(np.nanmax(time_values)) if len(time_values) else 1.0
        self.rbf_centers = np.linspace(t_min, t_max, 5, dtype=float)
        self.rbf_width = max(float(np.nanstd(time_values)), 1.0)
        return self

    def transform(self, meta):
        ids = []
        for field in CAT_FIELDS:
            col = self.mapping[field]
            lookup = self.maps[field]
            ids.append(meta[col].fillna("<NA>").astype(str).map(lookup).fillna(0).to_numpy(dtype=np.int64))
        cat = np.column_stack(ids)
        time_values = pd.to_numeric(meta[self.mapping["time"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        temp_values = pd.to_numeric(meta[self.mapping["temperature"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        raw = np.column_stack([time_values, temp_values, np.log1p(np.maximum(time_values, 0.0))])
        z = (raw - self.numeric_mean[None, :]) / self.numeric_std[None, :]
        rbf = np.exp(-0.5 * ((time_values[:, None] - self.rbf_centers[None, :]) / self.rbf_width) ** 2)
        numeric = np.column_stack([z, rbf]).astype(np.float32)
        return cat, numeric

    def vocab_sizes(self):
        return {field: len(self.maps[field]) + 1 for field in CAT_FIELDS}


class HCCEModel(nn.Module):
    def __init__(self, vocab_sizes, n_numeric, n_out, embedding_dim=64):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.emb = nn.ModuleDict({field: nn.Embedding(vocab_sizes[field], self.embedding_dim) for field in CAT_FIELDS})
        self.chem = nn.Sequential(nn.Linear(self.embedding_dim, 256), nn.GELU(), nn.Dropout(0.10))
        self.bio = nn.Sequential(nn.Linear(self.embedding_dim * len(BIO_FIELDS) + n_numeric, 256), nn.GELU(), nn.Dropout(0.10))
        self.measure = nn.Sequential(nn.Linear(self.embedding_dim * len(MEASURE_FIELDS), 128), nn.GELU(), nn.Dropout(0.10))
        # Keep a high-capacity one-hot route from the proven baseline. The
        # hierarchical route augments it; it does not replace exact IDs.
        self.legacy_chem = nn.Sequential(nn.Linear(vocab_sizes["compound"], 256), nn.GELU(), nn.Dropout(0.10))
        self.legacy_bio = nn.Sequential(nn.Linear(sum(vocab_sizes[f] for f in BIO_FIELDS) + n_numeric, 256), nn.GELU(), nn.Dropout(0.10))
        self.legacy_measure = nn.Sequential(nn.Linear(sum(vocab_sizes[f] for f in MEASURE_FIELDS), 128), nn.GELU(), nn.Dropout(0.10))
        legacy_context_dim = sum(vocab_sizes[field] for field in CAT_FIELDS if field != "compound") + 2
        self.legacy_chem_net = nn.Sequential(nn.Linear(vocab_sizes["compound"], 256), nn.GELU(), nn.Dropout(0.10))
        self.legacy_context_net = nn.Sequential(nn.Linear(legacy_context_dim, 256), nn.GELU(), nn.Dropout(0.10))
        self.legacy_modulation = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 512))
        self.legacy_head = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.10), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out))
        self.legacy_direct = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, n_out))
        self.concat_head = nn.Sequential(
            nn.Linear(256 + 256 + 128, 512), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out),
        )
        self.modulation = nn.Sequential(nn.Linear(256 + 128, 256), nn.GELU(), nn.Linear(256, 512))
        self.film_head = nn.Sequential(
            nn.Linear(256, 512), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(512, 256), nn.GELU(), nn.Linear(256, n_out),
        )
        self.direct_context = nn.Sequential(nn.Linear(256 + 128, 256), nn.GELU(), nn.Linear(256, n_out))

    def encode(self, cat, numeric):
        e = {field: self.emb[field](cat[:, i]) for i, field in enumerate(CAT_FIELDS)}
        chem = self.chem(e["compound"]) + self.legacy_chem(F.one_hot(cat[:, 0], num_classes=self.emb["compound"].num_embeddings).float())
        bio_onehot = torch.cat([F.one_hot(cat[:, i], num_classes=self.emb[field].num_embeddings).float() for i, field in enumerate(CAT_FIELDS) if field in BIO_FIELDS] + [numeric], dim=-1)
        bio = self.bio(torch.cat([e[field] for field in BIO_FIELDS] + [numeric], dim=-1)) + self.legacy_bio(bio_onehot)
        # Hierarchical measurement effect: plate is a shrunk residual on top
        # of source/instrument, which gives unseen plates a useful fallback.
        measure_input = torch.cat([e["source"], e["instrument"], 0.25 * e["plate"]], dim=-1)
        measure_onehot = torch.cat([F.one_hot(cat[:, i], num_classes=self.emb[field].num_embeddings).float() for i, field in enumerate(CAT_FIELDS) if field in MEASURE_FIELDS], dim=-1)
        measure = self.measure(measure_input) + self.legacy_measure(measure_onehot)
        return chem, bio, measure

    def forward(self, cat, numeric):
        chem, bio, measure = self.encode(cat, numeric)
        context = torch.cat([bio, measure], dim=-1)
        concat = self.concat_head(torch.cat([chem, context], dim=-1))
        gamma, beta = self.modulation(context).chunk(2, dim=-1)
        film = self.film_head((1.0 + gamma) * chem + beta) + 0.05 * self.direct_context(context)
        legacy_chem = self.legacy_chem_net(F.one_hot(cat[:, 0], num_classes=self.emb["compound"].num_embeddings).float())
        legacy_ctx_input = torch.cat(
            [F.one_hot(cat[:, i], num_classes=self.emb[field].num_embeddings).float() for i, field in enumerate(CAT_FIELDS) if field != "compound"]
            + [numeric[:, :2]], dim=-1,
        )
        legacy_context = self.legacy_context_net(legacy_ctx_input)
        legacy_gamma, legacy_beta = self.legacy_modulation(legacy_context).chunk(2, dim=-1)
        legacy = self.legacy_head((1.0 + legacy_gamma) * legacy_chem + legacy_beta) + 0.05 * self.legacy_direct(legacy_context)
        return concat, film, legacy

    def regularization(self):
        # Stronger shrinkage for plate deviations; other embeddings remain
        # learnable for biological/context transfer.
        plate = self.emb["plate"].weight[1:]
        return 1e-4 * torch.mean(plate * plate)


class ShrinkGate(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x):
        return 0.5 + 0.25 * torch.tanh(self.net(x).squeeze(-1))


def fit_gate(frame, train_mask, seed):
    seed_everything(seed + 1701)
    x = frame[GATE_FEATURES].to_numpy(dtype=np.float64)
    mean = np.nanmean(x[train_mask], axis=0)
    std = np.nanstd(x[train_mask], axis=0)
    std = np.where(np.isfinite(std) & (std >= 1e-8), std, 1.0)
    x = ((np.nan_to_num(x, nan=0.0) - mean) / std).astype(np.float32)
    target = frame["gate_target"].to_numpy(dtype=np.float32)
    weight = frame["gate_weight"].to_numpy(dtype=np.float32)
    use = np.asarray(train_mask, dtype=bool)
    model = ShrinkGate(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-3)
    xt = torch.from_numpy(x[use]); yt = torch.from_numpy(target[use]); wt = torch.from_numpy(weight[use])
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        pred = model(xt)
        loss = (wt * (pred - yt) ** 2).mean() + 0.02 * (pred.mean() - 0.5) ** 2
        loss.backward()
        opt.step()
    return model.eval(), mean, std


def predict_gate(model, frame, mean, std):
    x = ((np.nan_to_num(frame[GATE_FEATURES].to_numpy(dtype=np.float64), nan=0.0) - mean) / std).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def feature_rows(meta, sample_ids, train_ids, mapping, disagreement):
    work = meta.set_index(mapping["sample_id"])
    train_meta = work.loc[list(train_ids)]
    sets = {}
    for key in ["compound", "strain", "time", "plate", "source", "instrument"]:
        col = mapping[key]
        if key == "time":
            values = pd.to_numeric(train_meta[col], errors="coerce").dropna().to_numpy(dtype=float)
        else:
            values = train_meta[col].fillna("<NA>").astype(str).tolist()
        sets[key] = set(values.tolist() if hasattr(values, "tolist") else values)
    train_times = np.asarray(sorted(sets["time"]), dtype=float)
    rows = []
    for sid, dis in zip(sample_ids, disagreement):
        row = work.loc[sid]
        values = {}
        for key in ["compound", "strain", "plate", "source", "instrument"]:
            v = str(row[mapping[key]]) if pd.notna(row[mapping[key]]) else "<NA>"
            values[key] = v
        t = float(row[mapping["time"]])
        rows.append({
            "sample_ID": str(sid),
            "compound_seen": float(values["compound"] in sets["compound"]),
            "strain_seen": float(values["strain"] in sets["strain"]),
            "time_seen": float(t in sets["time"]),
            "plate_seen": float(values["plate"] in sets["plate"]),
            "source_seen": float(values["source"] in sets["source"]),
            "instrument_seen": float(values["instrument"] in sets["instrument"]),
            "time_distance": float(np.min(np.abs(train_times - t))) if len(train_times) else 0.0,
            "expert_disagreement": float(dis),
        })
    return pd.DataFrame(rows)


def fit_model(meta, y, fit_indices, mapping, seed, epochs, device, embedding_dim, precision_tau=None):
    seed_everything(seed)
    encoder = HCCEMetaEncoder(mapping).fit(meta, fit_indices)
    cat, numeric = encoder.transform(meta)
    fit_indices = np.asarray(fit_indices, dtype=int)
    fit_cat = torch.from_numpy(cat[fit_indices]).long()
    fit_num = torch.from_numpy(numeric[fit_indices]).float()
    fit_y = y[fit_indices].astype(np.float64)
    target_mean = np.nanmean(fit_y, axis=0)
    target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0)
    target_std = np.nanstd(fit_y, axis=0)
    target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0)
    filled = np.where(np.isfinite(fit_y), fit_y, target_mean[None, :])
    fit_observed = np.isfinite(fit_y)
    fit_z = ((filled - target_mean[None, :]) / target_std[None, :]).astype(np.float32)
    precision_weights = None
    if precision_tau is not None:
        precision_weights, _ = estimate_precision_weights(meta, y, fit_indices, mapping, tau=precision_tau)
        precision_weights = torch.from_numpy(precision_weights.astype(np.float32))
    vocab = encoder.vocab_sizes()
    model = HCCEModel(vocab, numeric.shape[1], y.shape[1], embedding_dim=embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(fit_cat, fit_num, torch.from_numpy(fit_z), torch.from_numpy(fit_observed.astype(np.float32))),
        batch_size=128, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    loss_fn = nn.MSELoss()
    model.train()
    history = []
    for epoch in range(epochs):
        total = 0.0; seen = 0
        for xb_cat, xb_num, yb, ymask in loader:
            xb_cat = xb_cat.to(device, non_blocking=True)
            xb_num = xb_num.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ymask = ymask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred_concat, pred_film, pred_legacy = model(xb_cat, xb_num)
            if precision_weights is None:
                loss_mask = ymask
            else:
                loss_mask = ymask * precision_weights.to(ymask.device).unsqueeze(0)
            denom = loss_mask.sum().clamp_min(1.0)
            mse_concat = (((pred_concat - yb) ** 2) * loss_mask).sum() / denom
            mse_film = (((pred_film - yb) ** 2) * loss_mask).sum() / denom
            mse_legacy = (((pred_legacy - yb) ** 2) * loss_mask).sum() / denom
            loss = (mse_concat + mse_film + mse_legacy) / 3.0 + model.regularization()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(xb_cat)
            seen += len(xb_cat)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, seen)})
    return model.eval(), encoder, target_mean, target_std, history


def predict_model(model, encoder, target_mean, target_std, meta, device):
    cat, numeric = encoder.transform(meta)
    with torch.no_grad():
        pred_concat, pred_film, pred_legacy = model(torch.from_numpy(cat).long().to(device), torch.from_numpy(numeric).float().to(device))
    concat = pred_concat.detach().cpu().numpy() * target_std[None, :] + target_mean[None, :]
    film = pred_film.detach().cpu().numpy() * target_std[None, :] + target_mean[None, :]
    legacy = pred_legacy.detach().cpu().numpy() * target_std[None, :] + target_mean[None, :]
    return concat.astype(np.float32), film.astype(np.float32), legacy.astype(np.float32)


def fold_records(meta, y, proteins, mapping, fold_table, fold_types, seed, epochs, device, output, embedding_dim, precision_tau=None):
    sample_ids = meta[mapping["sample_id"]].astype(str).tolist()
    sample_to_row = {sid: i for i, sid in enumerate(sample_ids)}
    records = []
    fold_manifest = []
    for fold_type in fold_types:
        n_folds = int(fold_table.loc[fold_table.fold_type == fold_type, "fold_id"].astype(int).max()) + 1
        for fold_id in range(n_folds):
            selected = fold_table[(fold_table.fold_type == fold_type) & (fold_table.fold_id == str(fold_id))]
            hold_ids = selected.loc[selected.role == "holdout", "sample_ID"].astype(str).tolist()
            train_ids = selected.loc[selected.role == "train", "sample_ID"].astype(str).tolist()
            if not hold_ids or not train_ids:
                continue
            fit_indices = np.asarray([sample_to_row[sid] for sid in train_ids], dtype=int)
            hold_indices = np.asarray([sample_to_row[sid] for sid in hold_ids], dtype=int)
            model, encoder, target_mean, target_std, history = fit_model(meta, y, fit_indices, mapping, seed + fold_id + 100 * len(records), epochs, device, embedding_dim, precision_tau=precision_tau)
            hold_meta = meta.iloc[hold_indices].reset_index(drop=True)
            hcce_concat, film, legacy = predict_model(model, encoder, target_mean, target_std, hold_meta, device)
            # The proven legacy FiLM route is the base expert; the new HCCE
            # FiLM route is the context-aware expert used by the gate.
            concat = legacy
            true = y[hold_indices]
            disagreement = np.sqrt(np.nanmean((film - concat) ** 2, axis=1))
            feat = feature_rows(meta, hold_ids, set(train_ids), mapping, disagreement)
            diff = film.astype(np.float64) - concat.astype(np.float64)
            residual = true.astype(np.float64) - concat.astype(np.float64)
            finite = np.isfinite(diff) & np.isfinite(residual)
            numerator = np.nansum(np.where(finite, diff * residual, 0.0), axis=1)
            denominator = np.nansum(np.where(finite, diff * diff, 0.0), axis=1)
            target = np.where(denominator > 1e-8, np.clip(numerator / np.maximum(denominator, 1e-8), 0.0, 1.0), 0.5)
            positive = denominator[denominator > 0]
            med = np.nanmedian(positive) if len(positive) else 1.0
            weight = np.clip(denominator / max(med, 1e-8), 0.1, 10.0)
            feat["fold_type"] = fold_type
            feat["fold_id"] = str(fold_id)
            feat["gate_target"] = target
            feat["gate_weight"] = weight
            records.append({
                "fold_type": fold_type, "fold_id": str(fold_id), "sample_ids": hold_ids,
                "train_ids": train_ids, "concat": concat, "film": film, "true": true,
                "features": feat,
            })
            fold_manifest.append({"fold_type": fold_type, "fold_id": fold_id, "n_train": len(train_ids), "n_holdout": len(hold_ids), "epochs": len(history)})
            fold_dir = output / "oof" / f"{fold_type}_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            out = pd.DataFrame(concat, columns=proteins)
            out.insert(0, mapping["sample_id"], hold_ids)
            out.to_parquet(fold_dir / "prediction_concat.parquet", index=False)
            out2 = pd.DataFrame(film, columns=proteins)
            out2.insert(0, mapping["sample_id"], hold_ids)
            out2.to_parquet(fold_dir / "prediction_film.parquet", index=False)
            out3 = pd.DataFrame(hcce_concat, columns=proteins)
            out3.insert(0, mapping["sample_id"], hold_ids)
            out3.to_parquet(fold_dir / "prediction_hcce_concat.parquet", index=False)
    pd.DataFrame(fold_manifest).to_csv(output / "oof_manifest.csv", index=False)
    return records


def records_to_gate_frame(records):
    return pd.concat([r["features"] for r in records], ignore_index=True)


def mixed_metrics(records, gates):
    concat = np.concatenate([r["concat"] for r in records], axis=0)
    film = np.concatenate([r["film"] for r in records], axis=0)
    true = np.concatenate([r["true"] for r in records], axis=0)
    pred = (1.0 - gates[:, None]) * concat + gates[:, None] * film
    return pred, true, basic_metrics(pred, true)


def fit_calibrator(plate_records, mapping, target_std, gate_model=None, gate_mean=None, gate_std=None, alpha=10.0, cap_fraction=0.02):
    ids = [sid for r in plate_records for sid in r["sample_ids"]]
    base_parts = []
    for r in plate_records:
        if gate_model is None:
            gates = np.full(len(r["sample_ids"]), 0.5, dtype=float)
        else:
            gates = predict_gate(gate_model, r["features"], gate_mean, gate_std)
        r["gate"] = gates
        base_parts.append((1.0 - gates[:, None]) * r["concat"] + gates[:, None] * r["film"])
    base = np.concatenate(base_parts, axis=0).astype(np.float64)
    true = np.concatenate([r["true"] for r in plate_records], axis=0).astype(np.float64)
    meta_rows = []
    for r in plate_records:
        # feature_rows includes sample_ID; source/instrument are restored later
        meta_rows.extend(r["sample_meta"])
    frame = pd.DataFrame(meta_rows)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    x = enc.fit_transform(frame[["source", "instrument"]].astype(str))
    residual = true - base
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    coef = np.zeros((design.shape[1], residual.shape[1]), dtype=np.float64)
    for j in range(residual.shape[1]):
        valid = np.isfinite(residual[:, j])
        if not np.any(valid):
            continue
        d = design[valid]
        rr = residual[valid, j]
        gram = d.T @ d
        gram[np.diag_indices(gram.shape[0])] += alpha
        gram[0, 0] -= alpha
        coef[:, j] = np.linalg.solve(gram, d.T @ rr)
    raw = design @ coef
    valid_std = np.asarray(target_std)[np.isfinite(target_std) & (np.asarray(target_std) > 0)]
    limit = cap_fraction * float(np.linalg.norm(valid_std))
    norms = np.linalg.norm(raw, axis=1)
    scales = np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    correction = raw * scales[:, None]
    return {"encoder": enc, "coef": coef, "cap_limit": limit, "cap_fraction": cap_fraction, "ids": ids, "base": base, "true": true, "correction": correction, "scales": scales, "fit_frame": frame}


def apply_calibrator(cal, meta, target_std):
    frame = pd.DataFrame({
        "source": meta["data_source"].fillna("<NA>").astype(str).to_numpy(),
        "instrument": meta["instrument"].fillna("<NA>").astype(str).to_numpy(),
    })
    x = cal["encoder"].transform(frame[["source", "instrument"]].astype(str))
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    raw = design @ cal["coef"]
    norms = np.linalg.norm(raw, axis=1)
    scales = np.minimum(1.0, cal["cap_limit"] / np.maximum(norms, 1e-12))
    return raw * scales[:, None], scales


def metrics_with_fc(pred, true, meta, mapping):
    metrics = basic_metrics(pred, true)
    metrics["official_absolute"] = official_absolute_metrics(pred, true)
    fc, matches = matched_fc(pred, true, meta, np.ones(len(meta), dtype=bool), mapping)
    metrics["fc_pcc"] = fc.get("fc_pcc")
    metrics["fc_coverage"] = fc.get("coverage")
    return metrics, fc, matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-oof", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--official-preprocess", action="store_true", help="filter train-missing>=80%% proteins and use masked loss")
    parser.add_argument("--precision-weighted-loss", action="store_true", help="use train-only duplicate-condition precision weights")
    parser.add_argument("--precision-tau", type=float, default=0.10)
    parser.add_argument("--fold-table", default="outputs/biocal_moe/inner_folds.csv", help="Train-only OOF fold table; default preserves historical behavior")
    parser.add_argument("--oof-types", nargs="+", choices=OOF_TYPES, default=OOF_TYPES, help="OOF regimes to run; use e.g. --oof-types both for a targeted rerun")
    parser.add_argument("--oof-only", action="store_true", help="Stop after writing requested OOF fold predictions; useful for targeted fixed-fold reruns")
    args = parser.parse_args()
    started = time.time()
    seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("Requested CUDA but CUDA is unavailable")
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    meta = load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, proteins = align_metadata_proteome(meta, prot)
    all_proteins = list(proteins)
    raw = finite_float_matrix(prot, all_proteins)
    if args.official_preprocess:
        proteins, missing_rate, keep_mask = apply_official_protein_filter(meta, raw, all_proteins, mapping, threshold=0.80)
        raw = raw[:, keep_mask]
    y = to_log2_proteome(raw).astype(np.float64)
    train_mask = meta[mapping["split"]].astype(str).eq("train").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(~train_mask)
    val_meta = meta.iloc[val_indices].reset_index(drop=True)
    val_true = y[val_indices]

    if not args.skip_oof:
        fold_table_path = Path(args.fold_table)
        if not fold_table_path.is_absolute():
            fold_table_path = ROOT / fold_table_path
        fold_table = pd.read_csv(fold_table_path, dtype=str)
        precision_tau = args.precision_tau if args.precision_weighted_loss else None
        records = fold_records(meta, y, proteins, mapping, fold_table, args.oof_types, args.seed, args.epochs, device, output, args.embedding_dim, precision_tau=precision_tau)
        if args.oof_only:
            oof_result = {
                "run_id": f"HCCE_Proteome_OOF_only_seed{args.seed}",
                "seed": args.seed,
                "device": str(device),
                "epochs": args.epochs,
                "embedding_dim": args.embedding_dim,
                "fold_table": str(fold_table_path),
                "oof_types": list(args.oof_types),
                "n_fold_models": len(records),
                "n_holdout_predictions": int(sum(len(record["sample_ids"]) for record in records)),
                "n_proteins": len(proteins),
                "output_space": "log2(raw)",
                "no_test_truth": True,
                "elapsed_sec": time.time() - started,
            }
            (output / "oof_only_config.json").write_text(json.dumps(json_safe(oof_result), ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(json_safe(oof_result), ensure_ascii=False, indent=2))
            return
        frame = records_to_gate_frame(records)
        non_plate = [r for r in records if r["fold_type"] != "plate"]
        gate_train_mask = frame["fold_type"].isin(["compound", "strain"]).to_numpy()
        non_plate_mask = frame["fold_type"] != "plate"
        gate_model, gate_mean, gate_std = fit_gate(frame.loc[non_plate_mask].reset_index(drop=True), frame.loc[non_plate_mask, "fold_type"].isin(["compound", "strain"]).to_numpy(), args.seed)
        gate_frame_nonplate = frame.loc[non_plate_mask].reset_index(drop=True)
        gate_check = []
        for regime in ["both", "time"]:
            subset = gate_frame_nonplate[gate_frame_nonplate.fold_type == regime].reset_index(drop=True)
            gates = predict_gate(gate_model, subset, gate_mean, gate_std)
            subrecords = [r for r in non_plate if r["fold_type"] == regime]
            pred, true, mm = mixed_metrics(subrecords, gates)
            concat_m = basic_metrics(np.concatenate([r["concat"] for r in subrecords]), true)
            film_m = basic_metrics(np.concatenate([r["film"] for r in subrecords]), true)
            gate_check.append({"regime": regime, "n_samples": len(true), "gate_abs_pcc": mm["abs_pcc"], "concat_abs_pcc": concat_m["abs_pcc"], "film_abs_pcc": film_m["abs_pcc"], "gate_delta_vs_best": mm["abs_pcc"] - max(concat_m["abs_pcc"], film_m["abs_pcc"])})
        gate_model_final, gate_mean_final, gate_std_final = fit_gate(gate_frame_nonplate, np.ones(len(gate_frame_nonplate), dtype=bool), args.seed + 1)
        joblib.dump({"model": gate_model_final.state_dict(), "mean": gate_mean_final, "std": gate_std_final, "features": GATE_FEATURES}, output / "gate.joblib")
        pd.DataFrame(gate_check).to_csv(output / "gate_oof_generalization.csv", index=False)
        # Attach raw metadata for plate calibration.
        by_sid = meta.set_index(mapping["sample_id"])
        for r in records:
            r["sample_meta"] = [{"sample_ID": sid, "source": str(by_sid.loc[sid, mapping["source"]]), "instrument": str(by_sid.loc[sid, mapping["instrument"]])} for sid in r["sample_ids"]]
        plate_records = [r for r in records if r["fold_type"] == "plate"]
        if plate_records:
            train_target_std = np.nanstd(y[train_indices], axis=0)
            train_target_std = np.where(np.isfinite(train_target_std) & (train_target_std >= 1e-8), train_target_std, 1.0)
            calibrator = fit_calibrator(plate_records, mapping, train_target_std, gate_model_final, gate_mean_final, gate_std_final, alpha=10.0, cap_fraction=0.02)
            joblib.dump(calibrator, output / "measurement_calibrator.joblib")
            plate_base = calibrator["base"]
            plate_cal = plate_base + calibrator["correction"]
            plate_true = calibrator["true"]
            plate_metrics = {"base": basic_metrics(plate_base, plate_true), "calibrated": basic_metrics(plate_cal, plate_true), "cap_fraction": float(np.mean(calibrator["scales"] < 1.0)), "cap_limit": calibrator["cap_limit"]}
        else:
            calibrator = None
            plate_metrics = {}
    else:
        records = []
        gate_model_final = gate_mean_final = gate_std_final = calibrator = None
        gate_check = []
        plate_metrics = {}

    precision_tau = args.precision_tau if args.precision_weighted_loss else None
    model, encoder, target_mean, target_std, history = fit_model(meta, y, train_indices, mapping, args.seed + 9000, args.epochs, device, args.embedding_dim, precision_tau=precision_tau)
    hcce_concat_val, film_val, legacy_val = predict_model(model, encoder, target_mean, target_std, val_meta, device)
    concat_val = legacy_val
    mean50_val = 0.5 * concat_val + 0.5 * film_val
    val_ids = val_meta[mapping["sample_id"]].astype(str).tolist()
    official_train_ids = set(meta.iloc[train_indices][mapping["sample_id"]].astype(str))
    disagreement = np.sqrt(np.nanmean((film_val - concat_val) ** 2, axis=1))
    if gate_model_final is not None:
        gate_features = feature_rows(meta, val_ids, official_train_ids, mapping, disagreement)
        gate_val = predict_gate(gate_model_final, gate_features, gate_mean_final, gate_std_final)
        gate_pred = (1.0 - gate_val[:, None]) * concat_val + gate_val[:, None] * film_val
        if calibrator is not None:
            correction_val, correction_scales = apply_calibrator(calibrator, val_meta, target_std)
            calibrated_pred = gate_pred + correction_val
        else:
            calibrated_pred = gate_pred
            correction_scales = np.ones(len(val_meta), dtype=float)
    else:
        gate_val = np.full(len(val_meta), 0.5, dtype=float)
        gate_features = pd.DataFrame()
        gate_pred = mean50_val
        calibrated_pred = gate_pred
        correction_scales = np.ones(len(val_meta), dtype=float)

    predictions = {"legacy": concat_val, "hcce_concat": hcce_concat_val, "film": film_val, "mean50": mean50_val, "gate": gate_pred, "calibrated": calibrated_pred}
    metrics = {}
    split_metrics = {}
    for name, pred in predictions.items():
        m, fc, matches = metrics_with_fc(pred, val_true, val_meta, mapping)
        metrics[name] = m
        split_metrics[name] = evaluate_basic_and_splits(pred, val_true, val_meta, split_col=mapping["split"], sample_ids=val_ids)["by_split"]
        matches.to_csv(output / f"{name}_control_matches.csv", index=False)
        frame_pred = pd.DataFrame(pred, columns=proteins)
        frame_pred.insert(0, mapping["sample_id"], val_ids)
        frame_pred.to_parquet(output / f"prediction_val_{name}.parquet", index=False)
    if gate_features.shape[0]:
        gate_features.assign(gate=gate_val, correction_scale=correction_scales).to_csv(output / "official_val_gate_features.csv", index=False)
    for name, frame_split in split_metrics.items():
        frame_split.to_csv(output / f"metrics_by_split_{name}.csv", index=False)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    torch.save({"model": model.state_dict(), "proteins": proteins, "target_mean": target_mean, "target_std": target_std, "selected_epochs": args.epochs}, output / "model_final.pt")
    joblib.dump(encoder, output / "preprocessor.joblib")
    np.save(output / "target_mean.npy", target_mean)
    np.save(output / "target_std.npy", target_std)
    result = {
        "run_id": f"HCCE_Proteome_seed{args.seed}", "seed": args.seed, "device": str(device), "epochs": args.epochs, "embedding_dim": args.embedding_dim,
        "fit_rows": int(len(train_indices)), "validation_rows": int(len(val_indices)), "n_proteins": int(len(proteins)),
        "preprocessing": {"official_updated_rule": bool(args.official_preprocess), "raw_protein_count": int(len(all_proteins)), "model_protein_count": int(len(proteins)), "missing_threshold": 0.80 if args.official_preprocess else None, "removed_proteins": int(len(all_proteins) - len(proteins))},
        "model": "HCCE-Proteome", "feature_contract": {"unknown_safe": True, "biological": BIO_FIELDS, "measurement": MEASURE_FIELDS, "continuous_time": True, "output_space": "log2(raw)"},
        "precision_weighted_loss": bool(args.precision_weighted_loss), "precision_tau": float(args.precision_tau) if args.precision_weighted_loss else None,
        "metrics": metrics, "gate_oof_generalization": gate_check, "plate_calibration_oof": plate_metrics,
        "gate_mean": float(np.mean(gate_val)), "calibration_cap_fraction_validation": float(np.mean(correction_scales < 1.0)),
        "no_test_truth": True, "elapsed_sec": time.time() - started,
    }
    (output / "metrics.json").write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "run_config.yaml").write_text(yaml.safe_dump(json_safe(result), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
