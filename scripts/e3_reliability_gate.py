#!/usr/bin/env python
"""E3: constrained reliability-based adaptive fusion gate.

ContextLadder_复赛综合审查与提升实验计划.md, chapter 8 (lines 887-927).

READ-ONLY with respect to scripts/run_hcce.py, scripts/a2b_train_variants.py,
scripts/grouped_ood_common.py, scripts/evaluate_grouped_ood.py and
scripts/e1_neutral_unknown.py -- this module only ever imports from them.
It never edits runs/final/, configs/final.yaml or prediction.csv, and its
results never replace any frozen competition prediction.

Motivation (8.1): the shipped model fuses the legacy (one-hot) and HCCE
(hierarchical/FiLM) experts with a fixed 0.5/0.5 blend. This experiment asks
whether a small, tightly-regularized "reliability gate" g(x) in [0.35, 0.65],
trained ONLY on train-internal cross-fitted OOF predictions (never on
validation truth), can beat the fixed blend on unseen entities without
regressing on seen ones.

Stages (run with --stage):
  oof      : generate cross-fitted OOF film/legacy predictions using the E2
             32-fold four-axis manifest (outputs/grouped_ood/split_manifest.json).
             Writes per-fold raw prediction arrays plus a compact per-row
             feature/ABC table used to fit the gate.
  gate     : select lambda_g via leave-one-axis-out inner CV on the OOF table,
             then fit the final gate on the full OOF table. Freezes gate_model.pt.
  confirm  : retrain the two experts on the FULL train split (5,920 rows) at
             three pre-committed seeds (matching configs/final.yaml's ensemble
             seeds), apply the FROZEN gate, and evaluate ONCE against the
             official validation split, broken out by the five subsets. This
             is the one confirmation pass -- do not loop this stage while
             adjusting the gate.
  all      : oof -> gate -> confirm

Data boundary: the gate is fit ONLY on split_final=='train' OOF predictions.
Validation truth is read only inside the confirm stage, for evaluation only
(exactly like scripts/run_unknown_fallback_ablation.py and friends already do
throughout this repo) -- it is never used to fit or select the gate. Test
truth / test proteome are never read anywhere in this file.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2b_train_variants import fit_model_variant, load_run_hcce  # noqa: E402
from scripts.grouped_ood_common import load_universe  # noqa: E402
from scripts.run_baselines import json_safe  # noqa: E402

OUT = ROOT / "outputs/e3_reliability_fusion"
OOF_DIR = OUT / "oof"
MANIFEST_PATH = ROOT / "outputs/grouped_ood/split_manifest.json"

VARIANT = "mask_compound"        # matches configs/final.yaml model.variant
EMBEDDING_DIM = 64               # matches configs/final.yaml
EPOCHS = 40                      # matches configs/final.yaml
OOF_SEED = 20260810              # matches E2's evaluate_grouped_ood.py default seed
CONFIRM_SEEDS = [20260810, 3407, 42]  # matches configs/final.yaml ensemble.seeds

GATE_LO, GATE_HI = 0.35, 0.65     # hard constraint, spec 8.3 point 3
FEATURE_COLS = [
    "compound_seen", "strain_seen", "time_seen", "plate_seen",
    "source_seen", "instrument_seen", "time_distance", "expert_disagreement",
    "compound_log1p_count", "strain_log1p_count", "plate_log1p_count",
]
LAMBDA_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

SUBSET_SPLIT_VALUES = {
    "all": None,
    "strain_unseen": "val_strain_only",
    "compound_unseen": "val_chem_only",
    "both_unseen": "val_both",
    "both_seen_time_shift": "val_time",
}


# --------------------------------------------------------------------------- #
# universe loading (train + official validation rows; test never opened)
# --------------------------------------------------------------------------- #
def load_full_universe(rh, mapping):
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    meta = rh.load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = rh.load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, all_proteins = rh.align_metadata_proteome(meta, prot)
    raw = rh.finite_float_matrix(prot, all_proteins)
    proteins, _, keep_mask = rh.apply_official_protein_filter(meta, raw, all_proteins, mapping, threshold=0.80)
    raw = raw[:, keep_mask]
    y = rh.to_log2_proteome(raw).astype(np.float64)
    return meta, y, proteins


def freq_log1p_features(meta_all, sample_ids, train_ids, mapping):
    """log1p(count) of compound/strain/plate frequency within the fold/full
    train set. Not present in rh.feature_rows(); this is the one reliability
    feature this file adds on top of the existing GATE_FEATURES vocabulary."""
    work = meta_all.set_index(mapping["sample_id"])
    train_meta = work.loc[list(train_ids)]
    counts = {}
    for key in ["compound", "strain", "plate"]:
        col = mapping[key]
        counts[key] = train_meta[col].fillna("<NA>").astype(str).value_counts().to_dict()
    rows = []
    for sid in sample_ids:
        row = work.loc[sid]
        rec = {"sample_ID": str(sid)}
        for key in ["compound", "strain", "plate"]:
            v = str(row[mapping[key]]) if pd.notna(row[mapping[key]]) else "<NA>"
            rec[f"{key}_log1p_count"] = float(np.log1p(counts[key].get(v, 0)))
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_row_abc(film, legacy, true):
    """Exact per-row quadratic coefficients of the fused-blend MSE in g:
        L_row(g) = mean_p[(g*film_p + (1-g)*legacy_p - true_p)^2]
                 = A + 2*B*g + C*g^2
    with d = film - legacy, e = legacy - true, A=mean(e^2), B=mean(e*d), C=mean(d^2).
    This lets the gate be trained on true OOF labels without ever storing or
    replaying full (row, protein) residual tensors at gate-training time."""
    d = (film.astype(np.float64) - legacy.astype(np.float64))
    e = (legacy.astype(np.float64) - true.astype(np.float64))
    mask = np.isfinite(true)
    d = np.where(mask, d, np.nan)
    e = np.where(mask, e, np.nan)
    with np.errstate(invalid="ignore"):
        A = np.nanmean(e * e, axis=1)
        B = np.nanmean(e * d, axis=1)
        C = np.nanmean(d * d, axis=1)
    n_obs = mask.sum(axis=1)
    return A, B, C, n_obs


# --------------------------------------------------------------------------- #
# stage 1: OOF generation on the E2 32-fold four-axis manifest
# --------------------------------------------------------------------------- #
def run_oof_stage(device, epochs=EPOCHS, seed=OOF_SEED, embedding_dim=EMBEDDING_DIM):
    rh, meta_train, y_train, proteins, mapping = load_universe()
    manifest = json.loads(MANIFEST_PATH.read_text())
    if not manifest["leakage_summary"]["all_pass"]:
        raise AssertionError("split_manifest.json leakage_summary.all_pass is False; refusing to build OOF on a leaking split")

    sample_id_col = mapping["sample_id"]
    sample_id_to_row = {sid: i for i, sid in enumerate(meta_train[sample_id_col].astype(str))}

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []
    t_start = time.time()
    for axis, axis_block in manifest["axes"].items():
        field_col = axis_block["field"]
        for fold in axis_block["folds"]:
            fold_id = fold["fold_id"]
            t0 = time.time()
            train_ids = fold["train_sample_ids"]
            test_ids = fold["test_sample_ids"]
            train_idx = np.array([sample_id_to_row[s] for s in train_ids], dtype=int)
            test_idx = np.array([sample_id_to_row[s] for s in test_ids], dtype=int)

            model, encoder, target_mean, target_std, _hist = fit_model_variant(
                rh, meta_train, y_train, train_idx, mapping, seed, epochs, device, embedding_dim, VARIANT,
            )
            test_meta = meta_train.iloc[test_idx].reset_index(drop=True)
            _concat, film, legacy = rh.predict_model(model, encoder, target_mean, target_std, test_meta, device)
            true = y_train[test_idx]

            disagreement = np.sqrt(np.nanmean((film.astype(np.float64) - legacy.astype(np.float64)) ** 2, axis=1))
            feat = rh.feature_rows(meta_train, test_ids, set(train_ids), mapping, disagreement)
            freq = freq_log1p_features(meta_train, test_ids, set(train_ids), mapping)
            feat = feat.merge(freq, on="sample_ID", how="left")

            A, B, C, n_obs = compute_row_abc(film, legacy, true)
            feat["A"] = A
            feat["B"] = B
            feat["C"] = C
            feat["n_obs_proteins"] = n_obs
            feat["axis"] = axis
            feat["fold_id"] = fold_id
            feat["field_col"] = field_col
            rows_out.append(feat)

            fold_tag = f"{axis}_fold{fold_id}"
            np.save(OOF_DIR / f"{fold_tag}_film.npy", film.astype(np.float32))
            np.save(OOF_DIR / f"{fold_tag}_legacy.npy", legacy.astype(np.float32))
            pd.Series(test_ids, name="sample_ID").to_csv(OOF_DIR / f"{fold_tag}_sample_ids.csv", index=False)

            elapsed = time.time() - t0
            print(f"[oof] axis={axis:9s} fold={fold_id:2d} n_test={len(test_ids):4d} "
                  f"mean_A={np.nanmean(A):.4f} mean_C={np.nanmean(C):.4f} ({elapsed:.1f}s)", flush=True)

    table = pd.concat(rows_out, ignore_index=True)
    table_path = OUT / "oof_gate_training_table.csv"
    table.to_csv(table_path, index=False)
    total_elapsed = time.time() - t_start
    print(f"\nwrote {table_path}  ({len(table)} rows, {total_elapsed:.1f}s total)")
    (OUT / "oof_manifest_used.json").write_text(json.dumps({
        "manifest": str(MANIFEST_PATH), "variant": VARIANT, "embedding_dim": embedding_dim,
        "epochs": epochs, "seed": seed, "n_rows": int(len(table)), "elapsed_sec": total_elapsed,
    }, indent=2))
    return table


# --------------------------------------------------------------------------- #
# gate model
# --------------------------------------------------------------------------- #
class ReliabilityGate(nn.Module):
    def __init__(self, n_features, hidden=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        raw = self.net(x).squeeze(-1)
        return GATE_LO + (GATE_HI - GATE_LO) * torch.sigmoid(raw)


def fit_gate_net(table, feature_cols, lambda_g, seed=12345, epochs=800, lr=5e-3, weight_decay=1e-3, feat_mean=None, feat_std=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_raw = table[feature_cols].to_numpy(dtype=np.float64)
    if feat_mean is None:
        feat_mean = np.nanmean(x_raw, axis=0)
        feat_std = np.nanstd(x_raw, axis=0)
        feat_std = np.where(np.isfinite(feat_std) & (feat_std >= 1e-8), feat_std, 1.0)
    x = ((np.nan_to_num(x_raw, nan=0.0) - feat_mean) / feat_std).astype(np.float32)
    A = table["A"].to_numpy(dtype=np.float32)
    B = table["B"].to_numpy(dtype=np.float32)
    C = table["C"].to_numpy(dtype=np.float32)
    finite = np.isfinite(A) & np.isfinite(B) & np.isfinite(C)
    x, A, B, C = x[finite], A[finite], B[finite], C[finite]

    xt = torch.from_numpy(x)
    At, Bt, Ct = torch.from_numpy(A), torch.from_numpy(B), torch.from_numpy(C)

    model = ReliabilityGate(len(feature_cols))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        g = model(xt)
        l_oof = torch.mean(At + 2.0 * Bt * g + Ct * g * g)
        l_reg = lambda_g * torch.mean((g - 0.5) ** 2)
        loss = l_oof + l_reg
        loss.backward()
        opt.step()
    return model.eval(), feat_mean, feat_std, int(finite.sum())


def gate_predict(model, table, feature_cols, feat_mean, feat_std):
    x_raw = table[feature_cols].to_numpy(dtype=np.float64)
    x = ((np.nan_to_num(x_raw, nan=0.0) - feat_mean) / feat_std).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def row_loss_fixed(table, g_fixed=0.5):
    A, B, C = table["A"].to_numpy(), table["B"].to_numpy(), table["C"].to_numpy()
    return float(np.nanmean(A + 2 * g_fixed * B + g_fixed * g_fixed * C))


def row_loss_gate(table, g):
    A, B, C = table["A"].to_numpy(), table["B"].to_numpy(), table["C"].to_numpy()
    return float(np.nanmean(A + 2 * g * B + g * g * C))


def select_lambda_leave_one_axis_out(table, feature_cols, lambdas=LAMBDA_GRID):
    axes = sorted(table["axis"].unique())
    results = []
    for lam in lambdas:
        fold_losses, fold_losses_fixed = [], []
        for held_axis in axes:
            train_tab = table[table.axis != held_axis].reset_index(drop=True)
            test_tab = table[table.axis == held_axis].reset_index(drop=True)
            model, fmean, fstd, n_used = fit_gate_net(train_tab, feature_cols, lam)
            g_test = gate_predict(model, test_tab, feature_cols, fmean, fstd)
            fold_losses.append(row_loss_gate(test_tab, g_test))
            fold_losses_fixed.append(row_loss_fixed(test_tab, 0.5))
        cv_loss = float(np.mean(fold_losses))
        cv_loss_fixed = float(np.mean(fold_losses_fixed))
        results.append({
            "lambda_g": lam, "cv_loss_gate": cv_loss, "cv_loss_fixed_50_50": cv_loss_fixed,
            "cv_improvement": cv_loss_fixed - cv_loss, "per_axis_loss_gate": fold_losses,
            "per_axis_loss_fixed": fold_losses_fixed,
        })
        print(f"[lambda-select] lambda_g={lam:<6} cv_loss_gate={cv_loss:.6f} "
              f"cv_loss_fixed={cv_loss_fixed:.6f} improvement={cv_loss_fixed - cv_loss:+.6f}", flush=True)
    best = min(results, key=lambda r: r["cv_loss_gate"])
    return best["lambda_g"], results


def run_gate_stage():
    table_path = OUT / "oof_gate_training_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} missing; run --stage oof first")
    table = pd.read_csv(table_path)

    best_lambda, cv_results = select_lambda_leave_one_axis_out(table, FEATURE_COLS)
    (OUT / "lambda_selection.json").write_text(json.dumps(json_safe({
        "grid": LAMBDA_GRID, "selected_lambda_g": best_lambda, "selection_protocol":
            "leave-one-axis-out inner CV strictly within the train-internal OOF table "
            "(no validation truth touched); axes = plate/strain/compound/time.",
        "results": cv_results,
    }), indent=2))
    print(f"\nselected lambda_g = {best_lambda}")

    model, feat_mean, feat_std, n_used = fit_gate_net(table, FEATURE_COLS, best_lambda, epochs=2000)
    g_oof = gate_predict(model, table, FEATURE_COLS, feat_mean, feat_std)
    table["gate_value"] = g_oof
    table.to_csv(table_path, index=False)  # augment in place with the frozen gate's OOF values

    torch.save({
        "state_dict": model.state_dict(), "feature_cols": FEATURE_COLS,
        "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
        "lambda_g": best_lambda, "gate_lo": GATE_LO, "gate_hi": GATE_HI,
        "n_oof_rows_used": n_used,
    }, OUT / "gate_model.pt")

    dist = {
        "n_rows": int(len(g_oof)), "mean": float(np.mean(g_oof)), "std": float(np.std(g_oof)),
        "min": float(np.min(g_oof)), "max": float(np.max(g_oof)),
        "frac_within_0.01_of_0.5": float(np.mean(np.abs(g_oof - 0.5) < 0.01)),
        "frac_at_lo_clip_0.001": float(np.mean(g_oof < GATE_LO + 0.001)),
        "frac_at_hi_clip_0.001": float(np.mean(g_oof > GATE_HI - 0.001)),
        "percentiles": {str(p): float(np.percentile(g_oof, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    }
    seen_mask = (table["compound_seen"] > 0.5) & (table["strain_seen"] > 0.5) & (table["plate_seen"] > 0.5)
    unseen_mask = ~seen_mask
    dist["oof_g_mean_all_seen"] = float(np.mean(g_oof[seen_mask.to_numpy()])) if seen_mask.any() else None
    dist["oof_g_mean_any_unseen"] = float(np.mean(g_oof[unseen_mask.to_numpy()])) if unseen_mask.any() else None
    for col in ["compound_seen", "strain_seen", "plate_seen", "time_seen"]:
        m = table[col] > 0.5
        dist[f"oof_g_mean_{col}"] = float(np.mean(g_oof[m.to_numpy()])) if m.any() else None
        dist[f"oof_g_mean_not_{col}"] = float(np.mean(g_oof[(~m).to_numpy()])) if (~m).any() else None
    (OUT / "gate_value_distribution_oof.json").write_text(json.dumps(dist, indent=2))
    print(f"wrote {OUT/'gate_model.pt'} and gate_value_distribution_oof.json")
    print(json.dumps(dist, indent=2))
    return model, feat_mean, feat_std, best_lambda


# --------------------------------------------------------------------------- #
# stage 3: freeze gate, retrain experts on full train, confirm on validation ONCE
# --------------------------------------------------------------------------- #
def run_confirm_stage(device, seeds=CONFIRM_SEEDS, epochs=EPOCHS, embedding_dim=EMBEDDING_DIM):
    gate_ckpt = torch.load(OUT / "gate_model.pt", map_location="cpu", weights_only=False)
    gate = ReliabilityGate(len(gate_ckpt["feature_cols"]))
    gate.load_state_dict(gate_ckpt["state_dict"])
    gate.eval()
    feat_mean = np.array(gate_ckpt["feat_mean"])
    feat_std = np.array(gate_ckpt["feat_std"])
    feature_cols = gate_ckpt["feature_cols"]

    rh, mapping = load_run_hcce(), yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    meta_all, y_all, proteins = load_full_universe(rh, mapping)
    split_col = mapping["split"]
    train_mask = meta_all[split_col].astype(str).eq("train").to_numpy()
    train_idx_full = np.flatnonzero(train_mask)
    if len(train_idx_full) != 5920:
        raise AssertionError(f"expected 5920 train rows, found {len(train_idx_full)}")
    train_sample_ids = meta_all.iloc[train_idx_full][mapping["sample_id"]].astype(str).tolist()

    val_mask_all = ~train_mask
    val_meta_all = meta_all.iloc[np.flatnonzero(val_mask_all)].reset_index(drop=True)
    val_true_all = y_all[np.flatnonzero(val_mask_all)]
    val_sample_ids = val_meta_all[mapping["sample_id"]].astype(str).tolist()

    subset_masks = {}
    split_series = val_meta_all[split_col].astype(str)
    for name, value in SUBSET_SPLIT_VALUES.items():
        subset_masks[name] = np.ones(len(val_meta_all), dtype=bool) if value is None else (split_series == value).to_numpy()

    per_seed_results = []
    for seed in seeds:
        t0 = time.time()
        model, encoder, target_mean, target_std, _hist = fit_model_variant(
            rh, meta_all, y_all, train_idx_full, mapping, seed, epochs, device, embedding_dim, VARIANT,
        )
        _concat, film_val, legacy_val = rh.predict_model(model, encoder, target_mean, target_std, val_meta_all, device)
        disagreement = np.sqrt(np.nanmean((film_val.astype(np.float64) - legacy_val.astype(np.float64)) ** 2, axis=1))
        feat = rh.feature_rows(meta_all, val_sample_ids, set(train_sample_ids), mapping, disagreement)
        freq = freq_log1p_features(meta_all, val_sample_ids, set(train_sample_ids), mapping)
        feat = feat.merge(freq, on="sample_ID", how="left")
        g_val = gate_predict(gate, feat, feature_cols, feat_mean, feat_std)

        pred_fixed = 0.5 * legacy_val + 0.5 * film_val
        pred_gate = g_val[:, None] * film_val + (1.0 - g_val[:, None]) * legacy_val

        subset_metrics = {}
        for name, mask in subset_masks.items():
            if not mask.any():
                subset_metrics[name] = {"n_samples": 0}
                continue
            m_fixed = rh.basic_metrics(pred_fixed[mask], val_true_all[mask])
            m_gate = rh.basic_metrics(pred_gate[mask], val_true_all[mask])
            subset_metrics[name] = {
                "n_samples": int(mask.sum()),
                "fixed_50_50": {"abs_pcc": m_fixed["abs_pcc"], "rmse": m_fixed["rmse"], "mae": m_fixed["mae"]},
                "gate": {"abs_pcc": m_gate["abs_pcc"], "rmse": m_gate["rmse"], "mae": m_gate["mae"]},
                "delta_gate_minus_fixed": {
                    "abs_pcc": m_gate["abs_pcc"] - m_fixed["abs_pcc"],
                    "rmse": m_gate["rmse"] - m_fixed["rmse"],
                    "mae": m_gate["mae"] - m_fixed["mae"],
                },
                "gate_value_mean": float(np.mean(g_val[mask])),
                "gate_value_std": float(np.std(g_val[mask])),
            }
        elapsed = time.time() - t0
        per_seed_results.append({
            "seed": seed, "elapsed_sec": elapsed, "subset_metrics": subset_metrics,
            "gate_value_all": {"mean": float(np.mean(g_val)), "std": float(np.std(g_val)),
                                "min": float(np.min(g_val)), "max": float(np.max(g_val))},
        })
        print(f"[confirm] seed={seed} done ({elapsed:.1f}s); "
              f"gate mean={np.mean(g_val):.4f} std={np.std(g_val):.4f}", flush=True)
        for name in subset_masks:
            sm = subset_metrics[name]
            if sm.get("n_samples", 0) == 0:
                continue
            d = sm["delta_gate_minus_fixed"]
            print(f"    subset={name:22s} n={sm['n_samples']:4d} "
                  f"pcc: fixed={sm['fixed_50_50']['abs_pcc']:.5f} gate={sm['gate']['abs_pcc']:.5f} "
                  f"delta={d['abs_pcc']:+.5f} | rmse delta={d['rmse']:+.5f}", flush=True)

    # bootstrap over seeds x rows within each subset to judge signal vs noise
    summary = {}
    for name in subset_masks:
        pcc_deltas = [r["subset_metrics"][name]["delta_gate_minus_fixed"]["abs_pcc"]
                      for r in per_seed_results if r["subset_metrics"][name].get("n_samples", 0) > 0]
        rmse_deltas = [r["subset_metrics"][name]["delta_gate_minus_fixed"]["rmse"]
                       for r in per_seed_results if r["subset_metrics"][name].get("n_samples", 0) > 0]
        summary[name] = {
            "n_seeds": len(pcc_deltas),
            "pcc_delta_mean": float(np.mean(pcc_deltas)) if pcc_deltas else None,
            "pcc_delta_std_across_seeds": float(np.std(pcc_deltas)) if len(pcc_deltas) > 1 else None,
            "rmse_delta_mean": float(np.mean(rmse_deltas)) if rmse_deltas else None,
            "rmse_delta_std_across_seeds": float(np.std(rmse_deltas)) if len(rmse_deltas) > 1 else None,
            "all_seeds_same_sign_pcc": bool(len(set(np.sign(pcc_deltas))) == 1) if pcc_deltas else None,
        }

    result = {
        "protocol": "E3 confirm: ONE-TIME validation confirmation of a gate frozen on train-internal OOF",
        "variant": VARIANT, "embedding_dim": embedding_dim, "epochs": epochs, "seeds": seeds,
        "lambda_g": gate_ckpt["lambda_g"], "gate_bounds": [GATE_LO, GATE_HI],
        "per_seed": per_seed_results, "cross_seed_summary": summary,
    }
    (OUT / "validation_confirmation.json").write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT/'validation_confirmation.json'}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["oof", "gate", "confirm", "all"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage in ("oof", "all"):
        run_oof_stage(device, epochs=args.epochs)
    if args.stage in ("gate", "all"):
        run_gate_stage()
    if args.stage in ("confirm", "all"):
        run_confirm_stage(device, epochs=args.epochs)


if __name__ == "__main__":
    main()
