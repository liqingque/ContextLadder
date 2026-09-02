#!/usr/bin/env python
"""A2/B: masked-entity training (A2) and replicate-averaged denoised targets (B).

Variants (same HCCE architecture, 40 epochs, dim 64, official preprocessing):
  baseline      : original fit_model (final-model convention, seed+9000)
  mask          : A2 - random 25% of compound rows and 25% of strain rows are
                  mapped to the unknown token (index 0) during training with
                  their true targets; index 0 thereby learns the optimal
                  marginal (average-entity) embedding end-to-end.
  denoise       : B - train targets replaced by NaN-aware group means for
                  exact duplicate condition groups (strain/compound/medium/
                  temperature/time), with per-group weight n/(n+1).
  mask_denoise  : A2 + B combined.

All fitting restricted to split_final=train (5,920 rows); validation truth is
evaluation-only; test data is never loaded.
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_run_hcce():
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location("run_hcce", ROOT / "scripts" / "run_hcce.py")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["run_hcce"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_denoise_targets(meta, fit_indices, mapping, tau=1.0):
    """Return (denoised targets y, per-row weights) for train rows.

    Exact duplicate condition groups: strain, compound, medium, temperature,
    time.  NaN-aware group mean per protein; weight n_rep/(n_rep+tau) for
    groups with >=2 replicates, 1.0 otherwise."""
    keys = [mapping["strain"], mapping["compound"], mapping["medium"],
            mapping["temperature"], mapping["time"]]
    fit_meta = meta.iloc[np.asarray(fit_indices)].reset_index(drop=True)
    groups = fit_meta.groupby(keys, sort=False).indices
    weights = np.ones(len(fit_meta), dtype=np.float64)
    denoised = None
    return groups, weights


def fit_model_variant(rh, meta, y, fit_indices, mapping, seed, epochs, device,
                      embedding_dim, variant, mask_p=0.25, strain_mask_p=None, denoise_tau=1.0,
                      model_cls=None, esm_basis=None, swa_last_n=0, ema_decay=0.0):
    rh.seed_everything(seed)
    rng = np.random.default_rng(seed)
    encoder = rh.HCCEMetaEncoder(mapping).fit(meta, fit_indices)
    cat, numeric = encoder.transform(meta)
    fit_indices = np.asarray(fit_indices, dtype=int)
    fit_cat = cat[fit_indices].copy()
    fit_num = numeric[fit_indices]
    fit_y = y[fit_indices].astype(np.float64)

    row_weights = np.ones(len(fit_indices), dtype=np.float32)

    if "mask" in variant:
        # A2: map a random 25% of compound rows (and, for the both-field
        # variant, 25% of strain rows) to token 0; true targets unchanged.
        cmask = rng.random(len(fit_indices)) < mask_p
        # strain_mask_p is None => legacy behaviour (strain shares mask_p, and only the
        # both-field variants mask it at all). A positive rate enables strain masking on
        # any mask* variant at its own rate. The two rng.random calls stay in this order:
        # swapping or dropping one would break bit-identity with every existing run.
        do_strain = (variant in ("mask", "mask_denoise")
                     or (strain_mask_p is not None and strain_mask_p > 0.0))
        s_rate = mask_p if strain_mask_p is None else strain_mask_p
        smask = (rng.random(len(fit_indices)) < s_rate) if do_strain else np.zeros(len(fit_indices), dtype=bool)
        fit_cat[cmask, 0] = 0
        fit_cat[smask, 1] = 0

    if "denoise" in variant:
        # B: replicate-averaged targets + group weights.  Exact duplicates are
        # defined as same (source, strain, compound, medium, temperature, time)
        # — matching src/precision_weights.py; cross-source rows are NOT pooled.
        keys = [mapping["source"], mapping["strain"], mapping["compound"],
                mapping["medium"], mapping["temperature"], mapping["time"]]
        fit_meta = meta.iloc[fit_indices].reset_index(drop=True)
        groups = fit_meta.groupby(keys, sort=False).indices
        denoised = fit_y.copy()
        for idxs in groups.values():
            idxs = np.asarray(idxs, dtype=int)
            if len(idxs) >= 2:
                gmean = np.nanmean(fit_y[idxs], axis=0)
                denoised[idxs] = np.where(np.isfinite(gmean)[None, :], gmean[None, :], fit_y[idxs])
                w = len(idxs) / (len(idxs) + denoise_tau)
                row_weights[idxs] = w
        fit_y = denoised

    target_mean = np.nanmean(fit_y, axis=0)
    target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0)
    target_std = np.nanstd(fit_y, axis=0)
    target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0)
    filled = np.where(np.isfinite(fit_y), fit_y, target_mean[None, :])
    fit_observed = np.isfinite(fit_y)
    fit_z = ((filled - target_mean[None, :]) / target_std[None, :]).astype(np.float32)

    vocab = encoder.vocab_sizes()
    if model_cls is not None:
        model = model_cls(vocab, numeric.shape[1], y.shape[1], embedding_dim=embedding_dim, esm_basis=esm_basis).to(device)
    else:
        model = rh.HCCEModel(vocab, numeric.shape[1], y.shape[1], embedding_dim=embedding_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w_tensor = torch.from_numpy(row_weights).float().view(-1, 1)
    row_idx = torch.arange(len(fit_indices)).long()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_cat).long(), torch.from_numpy(fit_num).float(),
                      torch.from_numpy(fit_z), torch.from_numpy(fit_observed.astype(np.float32)),
                      row_idx),
        batch_size=128, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    history = []
    swa_state = None
    swa_count = 0
    swa_start = max(0, epochs - int(swa_last_n)) if swa_last_n > 0 else epochs
    if ema_decay > 0.0 and swa_last_n > 0:
        raise ValueError("SWA and EMA are mutually exclusive; set at most one of --swa-last-n / --ema-decay")
    ema_state = None
    if ema_decay > 0.0:
        # Shadow copy only: the EMA never feeds back into the optimizer, so the
        # training trajectory is bit-identical to the ema_decay=0 run.
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for xb_cat, xb_num, yb, ymask, xb_idx in loader:
            xb_cat = xb_cat.to(device, non_blocking=True)
            xb_num = xb_num.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ymask = ymask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred_concat, pred_film, pred_legacy = model(xb_cat, xb_num)
            loss_mask = ymask
            if "denoise" in variant:
                loss_mask = loss_mask * w_tensor[xb_idx].to(device)
            denom = loss_mask.sum().clamp_min(1.0)
            mse_concat = (((pred_concat - yb) ** 2) * loss_mask).sum() / denom
            mse_film = (((pred_film - yb) ** 2) * loss_mask).sum() / denom
            mse_legacy = (((pred_legacy - yb) ** 2) * loss_mask).sum() / denom
            loss = (mse_concat + mse_film + mse_legacy) / 3.0 + model.regularization()
            loss.backward()
            opt.step()
            if ema_state is not None:
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        if value.is_floating_point():
                            ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                        else:
                            ema_state[key].copy_(value.detach())
            total += float(loss.detach().cpu()) * len(xb_cat)
            seen += len(xb_cat)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, seen)})
        if swa_last_n > 0 and epoch + 1 > swa_start:
            current = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in current.items()}
                swa_count = 1
            else:
                swa_count += 1
                for key, value in current.items():
                    if value.is_floating_point():
                        swa_state[key].add_((value.detach() - swa_state[key]) / swa_count)
                    else:
                        swa_state[key].copy_(value.detach())
    if swa_state is not None:
        model.load_state_dict(swa_state)
    if ema_state is not None:
        model.load_state_dict(ema_state)
    return model.eval(), encoder, target_mean, target_std, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "mask", "mask_compound", "denoise",
                             "mask_denoise", "mask_compound_denoise"])
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-p", type=float, default=0.25)
    ap.add_argument("--strain-mask-p", type=float, default=None,
                    help="independent mask rate for the strain token; None (default) keeps the "
                         "legacy behaviour where strain shares --mask-p and only the both-field "
                         "variants mask it at all")
    ap.add_argument("--denoise-tau", type=float, default=1.0)
    ap.add_argument("--swa-last-n", type=int, default=0,
                    help="running-average state_dict over final N epochs; 0 disables SWA")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="exponential moving average of weights, updated every optimizer step; "
                         "0 disables EMA (default, bit-identical to the pre-P3 script)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rh = load_run_hcce()
    torch.set_num_threads(8)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    meta = rh.load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = rh.load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, all_proteins = rh.align_metadata_proteome(meta, prot)
    raw = rh.finite_float_matrix(prot, all_proteins)
    proteins, _, keep_mask = rh.apply_official_protein_filter(meta, raw, all_proteins, mapping, threshold=0.80)
    raw = raw[:, keep_mask]
    y = rh.to_log2_proteome(raw).astype(np.float64)
    train_mask = meta[mapping["split"]].astype(str).eq("train").to_numpy()
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(~train_mask)
    val_meta = meta.iloc[val_indices].reset_index(drop=True)
    val_true = y[val_indices]
    val_ids = val_meta[mapping["sample_id"]].astype(str).tolist()

    train_compounds = set(meta.iloc[train_indices][mapping["compound"]].astype(str))
    train_strains = set(meta.iloc[train_indices][mapping["strain"]].astype(str))
    comp_unseen = ~val_meta[mapping["compound"]].astype(str).isin(train_compounds).to_numpy()
    strain_unseen = ~val_meta[mapping["strain"]].astype(str).isin(train_strains).to_numpy()
    subset_masks = {
        "all": np.ones(len(val_meta), dtype=bool),
        "strain_unseen_BAI": strain_unseen,
        "compound_unseen_6": comp_unseen,
        "both_seen": (~strain_unseen) & (~comp_unseen),
    }

    fit_seed = args.seed + 9000  # match run_hcce.py final-model convention
    model, encoder, target_mean, target_std, history = fit_model_variant(
        rh, meta, y, train_indices, mapping, fit_seed, args.epochs, device,
        args.embedding_dim, args.variant, mask_p=args.mask_p,
        strain_mask_p=args.strain_mask_p, denoise_tau=args.denoise_tau,
        swa_last_n=args.swa_last_n, ema_decay=args.ema_decay)

    concat_val, film_val, legacy_val = rh.predict_model(model, encoder, target_mean, target_std, val_meta, device)
    mean50 = 0.5 * legacy_val + 0.5 * film_val

    metrics = {}
    for name, pred in [("mean50", mean50)]:
        m, fc, matches = rh.metrics_with_fc(pred, val_true, val_meta, mapping)
        metrics[name] = {"metrics": m, "fc_pcc": float(fc["fc_pcc"]) if fc and fc.get("fc_pcc") is not None else None}
        sub = {}
        for sub_name, smask in subset_masks.items():
            if smask.any():
                sub[sub_name] = {k: float(v) for k, v in rh.basic_metrics(pred[smask], val_true[smask]).items()
                                 if isinstance(v, (int, float, np.floating, np.integer))}
        metrics[name]["subsets"] = sub
        splits = rh.evaluate_basic_and_splits(pred, val_true, val_meta, split_col=mapping["split"], sample_ids=val_ids)
        metrics[name]["by_split"] = {str(k): {kk: float(vv) for kk, vv in v.items()
                                              if isinstance(vv, (int, float, np.floating, np.integer))}
                                     for k, v in splits["by_split"].items()}

    torch.save({"model": model.state_dict(), "proteins": proteins, "target_mean": target_mean,
                "target_std": target_std, "variant": args.variant, "fit_seed": fit_seed},
               out / "model_final.pt")
    rh.joblib.dump(encoder, out / "preprocessor.joblib")
    np.save(out / "target_mean.npy", target_mean)
    np.save(out / "target_std.npy", target_std)
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    result = {
        "run_id": f"HCCE_{args.variant}_seed{args.seed}",
        "variant": args.variant, "seed": args.seed, "fit_seed": fit_seed,
        "device": str(device), "epochs": args.epochs, "embedding_dim": args.embedding_dim,
        "mask_p": args.mask_p if "mask" in args.variant else None,
        "strain_mask_p": args.strain_mask_p,
        "denoise_tau": args.denoise_tau if "denoise" in args.variant else None,
        "swa_last_n": args.swa_last_n,
        "ema_decay": args.ema_decay,
        "fit_rows": int(len(train_indices)), "validation_rows": int(len(val_indices)),
        "n_proteins": int(len(proteins)),
        "subset_sizes": {k: int(v.sum()) for k, v in subset_masks.items()},
        "metrics": metrics,
        "no_test_truth": True, "elapsed_sec": time.time() - started,
    }
    (out / "metrics.json").write_text(json.dumps(rh.json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rh.json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
