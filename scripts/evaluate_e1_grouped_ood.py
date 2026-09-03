#!/usr/bin/env python
"""E1 x E2 bridge: run E1's neutral-unknown / strict-plate-residual variants (e1a-e1f,
scripts/e1_neutral_unknown.py) through E2's train-internal four-axis grouped pseudo-OOD
harness (scripts/evaluate_grouped_ood.py::evaluate_four_axis), so that e1e/e1f's plate
mechanism -- untestable on the frozen official validation split (0 unseen-plate rows) --
can finally be evaluated against a real leave-plate-out protocol (12 grouped folds,
train-internal, split_final=='train' only).

READ-ONLY with respect to:
    scripts/grouped_ood_common.py, scripts/evaluate_grouped_ood.py,
    scripts/e1_neutral_unknown.py, scripts/run_hcce.py, scripts/a2b_train_variants.py
Everything new lives in this file; those five are imported only.

Adapter contract (matches scripts/grouped_ood_common.py::build_hcce_variant_functions):
    train_fn(meta, y, fit_idx, mapping, seed, epochs, device) -> state
    predict_fn(state, meta_subset, device) -> np.ndarray  (mean50 = 0.5*legacy + 0.5*film,
        the same ensemble head E1's own report and E2's baseline both use)

Candidates:
    e1a, e1b, e1c, e1d, e1e, e1f   -- exactly scripts.e1_neutral_unknown.VARIANT_TABLE,
                                       unmodified, via fit_model_variant_e1.
    e1a_rng_matched_ef             -- a placebo control, NOT in E1's variant table. Its
                                       forward-pass MATH is bit-identical to e1a's (cfg=
                                       UnknownFallbackConfig(), no mechanism switches on
                                       at all), but it replays the *same extra CUDA-side
                                       nn.Dropout(0.10) call* that e1e/e1f's
                                       hierarchical_measure_context adds on top of the
                                       plain e1a measure path (see "RNG confound anatomy"
                                       in the module docstring below), so its dropout-mask
                                       RNG stream is shifted the same way e1e/e1f's is,
                                       without the plate-residual mechanism itself being
                                       engaged. Comparing e1a vs e1a_rng_matched_ef isolates
                                       "how much of e1e/e1f's delta from e1a is just this
                                       RNG-stream shift" from "how much is the mechanism."

RNG confound anatomy (empirically verified in this repo's environment -- see the probe
scripts run during this experiment, not re-included here to keep this file free of any
throwaway diagnostics):
    1. torch's CPU default generator and CUDA default generator are independent streams
       in this torch build (2.1.0). Extra CPU-side nn.Embedding/nn.Linear construction
       during NeutralUnknownHCCEModel.__init__ (the E1 report's own hypothesis for the
       confound) does NOT perturb the CUDA generator used by GPU-side nn.Dropout during
       training -- verified directly: a burn of 2e6 CPU-only random draws inserted right
       before building a second model, with the model's own state already fixed, produces
       BIT-IDENTICAL post-training weights to a no-burn control when both are trained on
       CUDA. So the E1 report's stated causal story for the confound is not what actually
       drives the observed effect. The document's underlying *concern* (e1a is not a clean
       RNG-matched control for e1e/e1f) is nonetheless correct -- the true mechanism is
       different from originally guessed:
    2. e1a's encode() calls exactly TWO CUDA nn.Dropout(0.10) ops on a (batch, 128) tensor
       while building `measure` (self.measure(...), then self.legacy_measure(...)). e1e/
       e1f's hierarchical_measure_context calls THREE: measure_base(...), plate_residual
       (...), then (back in encode()) self.legacy_measure(...) -- one extra (batch,128)
       Dropout call per forward pass, every training step, for the entire run. Because
       CUDA's default generator is a single continuously-advancing stream across the whole
       training loop (never reset between batches/epochs), this one extra call per step
       shifts every dropout mask computed afterwards in that same step, and that shift
       compounds through every later step -- confirmed directly: with an identical seed,
       identical initial weights, and an input batch built from its own explicit generator
       (so it never touches the shared default CUDA stream), e1a's forward pass in train()
       mode and e1e's diverge even on the "legacy" output head, which is built from a
       one-hot-only computation path that shares NO embedding weights with the
       zero-freeze/strict-residual mechanism and therefore *should* be bit-identical to
       e1a's if the only difference were embedding content -- it is not, which is the
       direct fingerprint of an RNG-stream shift, not a weight-content difference.
    3. This confound is specific to e1e/e1f (both set strict_plate_residual=True, hence
       the extra Dropout call). e1b/e1d (zero-freeze only, strict_plate_residual=False)
       keep e1a's exact two-call dropout structure -- verified: e1a vs e1b's legacy head
       is bit-identical (0.0 max abs diff) on a random probe batch, confirming e1b/e1d's
       measured difference from e1a is a clean embedding-content effect, NOT RNG-stream
       confounded. Only e1e/e1f need the placebo control below.
    4. The placebo (`RngMatchedPlaceboModel`) replays one extra (batch,128) Dropout(0.10)
       call right after computing `measure`, matching e1e/e1f's call COUNT and per-call
       shape. This closes most, but empirically not all, of the RNG-stream gap: on the
       same probe batch used in (2), the "legacy"-head diff between e1e and the placebo
       (both starting from identical seeds/weights) shrinks from 0.0080 (e1a vs e1e) to
       0.0018 (placebo vs e1e) -- roughly a 4-5x reduction, not a perfect replay (the
       within-call ordering of the 3 calls still differs slightly: e1e's extra call sits
       *between* the other two, the placebo's sits after both). Treat the placebo as a
       partial, not exact, RNG control; it bounds the confound, it does not eliminate it.

No official-validation or test truth is read anywhere in this file. All fitting and
evaluation happens strictly inside E2's split_final=='train' manifold via
scripts/evaluate_grouped_ood.py::evaluate_four_axis, which itself only ever indexes
meta_train / y_train (both already restricted to split_final=='train' by
scripts.grouped_ood_common.load_universe).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.e1_neutral_unknown import (  # noqa: E402
    NeutralUnknownHCCEModel, UnknownFallbackConfig, VARIANT_TABLE,
    enforce_neutral_unknown_contract,
)
from scripts.evaluate_grouped_ood import evaluate_four_axis  # noqa: E402
from scripts.grouped_ood_common import load_universe  # noqa: E402
from scripts.run_baselines import json_safe  # noqa: E402

OUT = ROOT / "outputs/e1_grouped_ood"
E1_CANDIDATES = ["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"]
PLACEBO_NAME = "e1a_rng_matched_ef"
ALL_CANDIDATES = E1_CANDIDATES + [PLACEBO_NAME]


# --------------------------------------------------------------------------------------- #
# Placebo model: e1a's exact forward math, plus one replayed, discarded CUDA Dropout(0.10)
# call on a (batch, 128) tensor -- see module docstring point 4 for what this does and does
# not control for.
# --------------------------------------------------------------------------------------- #
class RngMatchedPlaceboModel(NeutralUnknownHCCEModel):
    def __init__(self, vocab_sizes, n_numeric, n_out, embedding_dim: int = 64, dropout_seed: int = 0):
        super().__init__(vocab_sizes, n_numeric, n_out, embedding_dim=embedding_dim,
                          cfg=UnknownFallbackConfig(), dropout_seed=dropout_seed)

    def encode(self, cat: torch.Tensor, numeric: torch.Tensor):
        chem, bio, measure = super().encode(cat, numeric)
        if self.training:
            dummy = torch.zeros(cat.shape[0], 128, device=cat.device, dtype=measure.dtype)
            F.dropout(dummy, p=0.10, training=True)  # consumed and discarded; never touches `measure`
        return chem, bio, measure


def fit_placebo_variant(rh, meta, y, fit_indices, mapping, seed: int, epochs: int, device,
                         embedding_dim: int = 64, mask_p: float = 0.25):
    """Mirrors scripts/e1_neutral_unknown.py::fit_model_variant_e1 exactly (same seeding
    order, same compound-masking convention, same optimizer/loader setup) but builds
    RngMatchedPlaceboModel instead of NeutralUnknownHCCEModel. Not importable from
    e1_neutral_unknown.py (which must stay unmodified), so the loop is duplicated here
    rather than parameterizing that file's model class."""
    rh.seed_everything(seed)
    rng = np.random.default_rng(seed)
    encoder = rh.HCCEMetaEncoder(mapping).fit(meta, fit_indices)
    cat, numeric = encoder.transform(meta)
    fit_indices = np.asarray(fit_indices, dtype=int)
    fit_cat = cat[fit_indices].copy()
    fit_num = numeric[fit_indices]
    fit_y = y[fit_indices].astype(np.float64)

    cmask = rng.random(len(fit_indices)) < mask_p
    fit_cat[cmask, 0] = 0

    target_mean = np.nanmean(fit_y, axis=0)
    target_mean = np.where(np.isfinite(target_mean), target_mean, 0.0)
    target_std = np.nanstd(fit_y, axis=0)
    target_std = np.where(np.isfinite(target_std) & (target_std >= 1e-8), target_std, 1.0)
    filled = np.where(np.isfinite(fit_y), fit_y, target_mean[None, :])
    fit_observed = np.isfinite(fit_y)
    fit_z = ((filled - target_mean[None, :]) / target_std[None, :]).astype(np.float32)

    vocab = encoder.vocab_sizes()
    dropout_seed = int(seed) + 424242
    model = RngMatchedPlaceboModel(vocab, numeric.shape[1], y.shape[1], embedding_dim=embedding_dim,
                                    dropout_seed=dropout_seed).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(fit_cat).long(), torch.from_numpy(fit_num).float(),
                      torch.from_numpy(fit_z), torch.from_numpy(fit_observed.astype(np.float32))),
        batch_size=128, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    history = []
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for xb_cat, xb_num, yb, ymask in loader:
            xb_cat = xb_cat.to(device, non_blocking=True)
            xb_num = xb_num.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ymask = ymask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred_concat, pred_film, pred_legacy = model(xb_cat, xb_num)
            denom = ymask.sum().clamp_min(1.0)
            mse_concat = (((pred_concat - yb) ** 2) * ymask).sum() / denom
            mse_film = (((pred_film - yb) ** 2) * ymask).sum() / denom
            mse_legacy = (((pred_legacy - yb) ** 2) * ymask).sum() / denom
            loss = (mse_concat + mse_film + mse_legacy) / 3.0 + model.regularization()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu()) * len(xb_cat)
            seen += len(xb_cat)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, seen)})
    return model.eval(), encoder, target_mean, target_std, history


# --------------------------------------------------------------------------------------- #
# E2-contract adapter factory
# --------------------------------------------------------------------------------------- #
def build_e1_variant_functions(rh, mapping, candidate: str, embedding_dim: int = 64, mask_p: float = 0.25):
    """Returns (train_fn, predict_fn) for one of ALL_CANDIDATES, matching the exact
    contract scripts/evaluate_grouped_ood.py::evaluate_four_axis expects."""
    if candidate not in ALL_CANDIDATES:
        raise ValueError(f"unknown E1 candidate '{candidate}'; available: {ALL_CANDIDATES}")

    if candidate == PLACEBO_NAME:
        def train_fn(meta, y, fit_idx, mapping_, seed, epochs, device):
            model, encoder, target_mean, target_std, history = fit_placebo_variant(
                rh, meta, y, fit_idx, mapping_, seed, epochs, device, embedding_dim, mask_p=mask_p,
            )
            return {"model": model, "encoder": encoder, "target_mean": target_mean,
                    "target_std": target_std, "history": history}
    else:
        from scripts.e1_neutral_unknown import fit_model_variant_e1
        cfg = VARIANT_TABLE[candidate]

        def train_fn(meta, y, fit_idx, mapping_, seed, epochs, device):
            model, encoder, target_mean, target_std, history = fit_model_variant_e1(
                rh, meta, y, fit_idx, mapping_, seed, epochs, device, embedding_dim, cfg, mask_p=mask_p,
            )
            enforce_neutral_unknown_contract(model)  # idempotent; matches run_unknown_fallback_ablation.py
            return {"model": model, "encoder": encoder, "target_mean": target_mean,
                    "target_std": target_std, "history": history}

    def predict_fn(state, meta_subset, device):
        concat, film, legacy = rh.predict_model(
            state["model"], state["encoder"], state["target_mean"], state["target_std"], meta_subset, device,
        )
        return 0.5 * legacy + 0.5 * film  # mean50, same head E1's own report and E2's baseline use

    return train_fn, predict_fn


# --------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=ALL_CANDIDATES)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--mask-p", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--axes", nargs="+", default=["plate"], choices=["plate", "strain", "compound", "time"])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--manifest", default=str(ROOT / "outputs/grouped_ood/split_manifest.json"))
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
    train_fn, predict_fn = build_e1_variant_functions(
        rh, mapping, args.candidate, embedding_dim=args.embedding_dim, mask_p=args.mask_p)

    started = time.time()
    result = evaluate_four_axis(
        train_fn, predict_fn, args.candidate, manifest, meta_train, y_train, mapping,
        seed=args.seed, epochs=args.epochs, device=device, axes=args.axes, n_bootstrap=args.n_bootstrap,
    )
    elapsed = time.time() - started

    # candidate key in the accumulated CSVs/JSON encodes the seed so multi-seed runs of the
    # SAME candidate don't dedup-collide (evaluate_grouped_ood's own accumulation keys on
    # (candidate, axis, fold_id[, entity]) with keep="last", which would silently overwrite
    # seed 3407's rows with seed 42's rows for the SAME nominal candidate name otherwise).
    candidate_key = f"{args.candidate}__seed{args.seed}"
    result["per_fold"]["candidate"] = candidate_key
    result["per_entity"]["candidate"] = candidate_key

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
    all_agg[candidate_key] = {
        "candidate_family": args.candidate, "seed": args.seed, "epochs": args.epochs,
        "embedding_dim": args.embedding_dim, "mask_p": args.mask_p,
        "axes_evaluated": args.axes, "n_bootstrap": args.n_bootstrap, "elapsed_sec": elapsed,
        "aggregate": result["aggregate"],
    }
    agg_path.write_text(json.dumps(json_safe(all_agg), indent=2, ensure_ascii=False))

    print(f"\nwrote {per_fold_path}")
    print(f"wrote {per_entity_path}")
    print(f"wrote {agg_path}")
    print(f"[{candidate_key}] total elapsed {elapsed:.1f}s")
    for axis, agg in result["aggregate"].items():
        print(f"  axis={axis:9s} n_entities={agg['n_entities']:3d} n_folds={agg['n_folds']:2d} "
              f"pcc_macro={agg['sample_pcc']['entity_macro']:.4f} "
              f"[{agg['sample_pcc']['ci_lo']:.4f},{agg['sample_pcc']['ci_hi']:.4f}] "
              f"rmse_macro={agg['log2_rmse']['entity_macro']:.4f} "
              f"[{agg['log2_rmse']['ci_lo']:.4f},{agg['log2_rmse']['ci_hi']:.4f}]")


if __name__ == "__main__":
    main()
