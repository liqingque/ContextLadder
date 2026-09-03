#!/usr/bin/env python
"""E1: Neutral Unknown + Strict Hierarchical Residual (ContextLadder_复赛综合审查与提升实验计划.md,
chapter 6, section 6.4, Step 1-5).

This module does NOT modify scripts/run_hcce.py::HCCEModel or its forward logic in any way.
Every new behavior lives in a subclass (NeutralUnknownHCCEModel) that is only ever instantiated
by scripts/run_unknown_fallback_ablation.py, gated by an explicit UnknownFallbackConfig whose
default is the E1a baseline (all new behavior OFF, architecturally identical to HCCEModel).
The frozen runs/final/ checkpoints, scripts/train.py and scripts/predict.py are untouched and
must still reproduce runs/final/prediction.csv byte-for-byte (verified separately -- see
scripts/verify_frozen_prediction.py or the ad-hoc command in the E1 report).

Field-level unknown-index-0 policy (index 0 is reserved as "unseen"/"<NA>" by
HCCEMetaEncoder.transform, which maps any value absent from the train-only fitted vocabulary
to 0):

    compound    -> learned_masked        (unchanged: index 0 receives gradient via the existing
                                           25% compound masking in fit_model_variant / this
                                           module's fit_model_variant_e1)
    strain      -> neutral_zero          (index 0 embedding row frozen at exactly zero)
    medium      -> neutral_zero
    source      -> neutral_zero
    instrument  -> neutral_zero
    plate       -> strict_residual_zero  (neutral_zero embedding row AND the plate contribution
                                           to measure_context is gated to exactly zero whenever
                                           the plate id is 0, independent of what the frozen
                                           embedding row or one-hot slice would otherwise produce)

Do NOT reintroduce strain masking here: three prior strain-mask rates (0.05 / 0.10 / 0.25) all
regressed validation performance. This module only makes the *unseen* (id == 0) behavior of
already-unmasked fields mathematically neutral; it never manufactures new "seen -> unseen" noise
for strain.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]


class SchemaError(ValueError):
    """Raised when a config file (or a variant block within it) has an unrecognized key.

    T6 requires that unknown fields in the executable config fail loudly rather than being
    silently ignored, which is how a threshold could quietly drift after the gates are frozen.
    """


# ---------------------------------------------------------------------------
# Step 1: unknown policy is declared in configs/e1_unknown_fallback.yaml; this
# dataclass is the in-memory, validated form of one variant's flags.
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = {
    "zero_freeze_unknown_embedding",
    "mask_unknown_onehot",
    "strict_plate_residual",
    "hierarchical_dropout",
    "plate_scale",
    "plate_dropout_p",
    "instrument_dropout_p",
}


@dataclass
class UnknownFallbackConfig:
    """One E1 variant's mechanism switches. All default OFF == E1a (current model)."""

    zero_freeze_unknown_embedding: bool = False   # E1b: non-compound unknown embedding -> 0, frozen
    mask_unknown_onehot: bool = False              # E1c: non-compound unknown one-hot column -> 0
    strict_plate_residual: bool = False            # E1e: plate as a gated residual on source/instrument
    hierarchical_dropout: bool = False             # E1f: train-time instrument/plate field dropout
    plate_scale: float = 0.25
    plate_dropout_p: float = 0.15
    instrument_dropout_p: float = 0.05

    def __post_init__(self):
        unknown = {f.name for f in dataclass_fields(self)} - _CONFIG_FIELDS
        if unknown:
            raise SchemaError(f"UnknownFallbackConfig has fields not in the frozen schema: {sorted(unknown)}")
        if self.hierarchical_dropout and not self.strict_plate_residual:
            raise SchemaError("hierarchical_dropout requires strict_plate_residual (E1f = E1e + dropout)")
        if not (0.0 <= self.plate_dropout_p <= 1.0):
            raise SchemaError(f"plate_dropout_p out of [0,1]: {self.plate_dropout_p}")
        if not (0.0 <= self.instrument_dropout_p <= 1.0):
            raise SchemaError(f"instrument_dropout_p out of [0,1]: {self.instrument_dropout_p}")

    def is_baseline(self) -> bool:
        return not (self.zero_freeze_unknown_embedding or self.mask_unknown_onehot
                    or self.strict_plate_residual or self.hierarchical_dropout)


def _validate_keys(raw: dict, allowed: set, where: str):
    extra = set(raw.keys()) - allowed
    if extra:
        raise SchemaError(f"unrecognized key(s) in {where}: {sorted(extra)}")


def variant_config_from_dict(raw: dict, where: str = "variant block") -> UnknownFallbackConfig:
    _validate_keys(raw, _CONFIG_FIELDS, where)
    return UnknownFallbackConfig(**raw)


VARIANT_TABLE: Dict[str, UnknownFallbackConfig] = {
    "e1a": UnknownFallbackConfig(),
    "e1b": UnknownFallbackConfig(zero_freeze_unknown_embedding=True),
    "e1c": UnknownFallbackConfig(mask_unknown_onehot=True),
    "e1d": UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True),
    "e1e": UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                  strict_plate_residual=True),
    "e1f": UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                  strict_plate_residual=True, hierarchical_dropout=True),
}

TRAINABLE_UNKNOWN_FIELDS = {"compound"}


def load_e1_yaml_config(path) -> dict:
    """Load configs/e1_unknown_fallback.yaml with top-level schema validation (T6)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _validate_keys(raw, {"schema_version", "model", "variants", "training"}, "top level of e1 config")
    _validate_keys(raw["model"], {"base_variant", "embedding_dim", "epochs", "mask_p",
                                   "unknown_policy", "measurement_hierarchy"}, "model block")
    _validate_keys(raw["model"]["unknown_policy"],
                    {"compound", "strain", "medium", "source", "instrument", "plate"},
                    "model.unknown_policy")
    _validate_keys(raw["model"]["measurement_hierarchy"],
                    {"mode", "plate_residual_scale", "plate_dropout_p", "instrument_dropout_p"},
                    "model.measurement_hierarchy")
    for name, block in raw["variants"].items():
        _validate_keys(block, _CONFIG_FIELDS, f"variants.{name}")
    return raw


def variant_configs_from_yaml(path) -> Dict[str, UnknownFallbackConfig]:
    raw = load_e1_yaml_config(path)
    return {name: variant_config_from_dict(block, f"variants.{name}") for name, block in raw["variants"].items()}


# ---------------------------------------------------------------------------
# Step 2: per-field embedding construction.
# ---------------------------------------------------------------------------

def make_field_embedding(field: str, n: int, d: int, cfg: UnknownFallbackConfig) -> nn.Embedding:
    if field in TRAINABLE_UNKNOWN_FIELDS or not cfg.zero_freeze_unknown_embedding:
        # compound id=0 receives gradient through the existing masked-training contract;
        # if the E1b/E1d/E1e/E1f zero-freeze switch is off, every field stays as-is (E1a/E1c).
        return nn.Embedding(n, d)
    emb = nn.Embedding(n, d, padding_idx=0)
    with torch.no_grad():
        emb.weight[0].zero_()
    return emb


@torch.no_grad()
def enforce_neutral_unknown_contract(model: "NeutralUnknownHCCEModel") -> None:
    """Idempotent contract repair, safe to call after checkpoint load or an optimizer step.

    A no-op unless the model's config actually turned zero-freezing on -- it must never zero a
    field that was deliberately left learned (e.g. compound, or any field under E1a/E1c where
    zero_freeze_unknown_embedding is False and the row is expected to keep training normally).
    """
    cfg = getattr(model, "cfg", None)
    if cfg is None or not cfg.zero_freeze_unknown_embedding:
        return
    for field, emb in model.emb.items():
        if field not in TRAINABLE_UNKNOWN_FIELDS:
            emb.weight[0].zero_()


# ---------------------------------------------------------------------------
# Step 3: unknown one-hot masking.
# ---------------------------------------------------------------------------

def field_one_hot(ids: torch.Tensor, n: int, *, learned_unknown: bool) -> torch.Tensor:
    x = F.one_hot(ids, num_classes=n).float()
    if not learned_unknown:
        x = x * ids.ne(0).unsqueeze(-1).float()
    return x


# ---------------------------------------------------------------------------
# The model. Subclasses run_hcce.HCCEModel; never edits HCCEModel itself.
# ---------------------------------------------------------------------------

def _load_run_hcce():
    from scripts.a2b_train_variants import load_run_hcce
    return load_run_hcce()


class NeutralUnknownHCCEModel(nn.Module):
    """E1 candidate model. Wraps the frozen HCCEModel architecture and, only when the
    corresponding UnknownFallbackConfig switch is on, replaces the exact pieces named in
    ContextLadder_复赛综合审查与提升实验计划.md 6.4 Step 2-5. With cfg=UnknownFallbackConfig()
    (all flags False / E1a), this class is architecturally and numerically identical to
    run_hcce.HCCEModel: same submodules, same forward math, same output for the same input
    and weights.
    """

    def __init__(self, vocab_sizes, n_numeric, n_out, embedding_dim: int = 64,
                 cfg: Optional[UnknownFallbackConfig] = None, dropout_seed: int = 0):
        super().__init__()
        rh = _load_run_hcce()
        self.cfg = cfg if cfg is not None else UnknownFallbackConfig()
        self.CAT_FIELDS = list(rh.CAT_FIELDS)
        self.BIO_FIELDS = list(rh.BIO_FIELDS)
        self.MEASURE_FIELDS = list(rh.MEASURE_FIELDS)
        self._source_idx = self.CAT_FIELDS.index("source")
        self._instrument_idx = self.CAT_FIELDS.index("instrument")
        self._plate_idx = self.CAT_FIELDS.index("plate")
        self._compound_idx = self.CAT_FIELDS.index("compound")

        # Build the exact HCCEModel architecture first (frozen reference behavior), then
        # replace only what this variant's config asks for. This guarantees every submodule
        # that E1 does not touch (chem/legacy_chem/bio/legacy_bio/legacy_* heads/concat_head/
        # film_head/modulation/direct_context) is bit-for-bit the same construction as
        # run_hcce.HCCEModel.
        base = rh.HCCEModel(vocab_sizes, n_numeric, n_out, embedding_dim=embedding_dim)
        self.embedding_dim = base.embedding_dim
        for name, module in base.named_children():
            setattr(self, name, module)

        d = self.embedding_dim
        if self.cfg.zero_freeze_unknown_embedding:
            for field in self.CAT_FIELDS:
                if field in TRAINABLE_UNKNOWN_FIELDS:
                    continue
                new_emb = make_field_embedding(field, vocab_sizes[field], d, self.cfg)
                self.emb[field] = new_emb

        if self.cfg.strict_plate_residual:
            del self.measure  # replaced by the explicit base + gated-residual pair below
            self.measure_base = nn.Sequential(nn.Linear(d * 2, 128), nn.GELU(), nn.Dropout(0.10))
            self.plate_residual = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.10))

        # Independent RNG for hierarchical field dropout. Deliberately not the numpy `rng` used
        # for compound masking nor the global torch RNG seeded by rh.seed_everything / the
        # DataLoader generator -- inserting into either of those sequences would silently change
        # every other seeded draw downstream (see project note on harness rng-order sensitivity).
        # Always draws on CPU regardless of model device, then moves the tiny mask tensor over.
        self._dropout_generator = torch.Generator(device="cpu")
        self._dropout_generator.manual_seed(int(dropout_seed))

    # -- Step 4: strict hierarchical residual for plate -------------------------------------
    def hierarchical_measure_context(self, cat: torch.Tensor, numeric: torch.Tensor,
                                      *, force_disable_plate_residual: bool = False) -> torch.Tensor:
        if not self.cfg.strict_plate_residual:
            raise RuntimeError("hierarchical_measure_context requires cfg.strict_plate_residual=True")
        e_source = self.emb["source"](cat[:, self._source_idx])
        e_instrument = self.emb["instrument"](cat[:, self._instrument_idx])
        e_plate = self.emb["plate"](cat[:, self._plate_idx])

        plate_known = cat[:, self._plate_idx].ne(0).float().unsqueeze(-1)
        instrument_known = cat[:, self._instrument_idx].ne(0).float().unsqueeze(-1)

        # Step 5: hierarchical field dropout, training-time only, independent Generator.
        if self.training and self.cfg.hierarchical_dropout:
            n = plate_known.shape[0]
            drop_plate = (torch.rand((n, 1), generator=self._dropout_generator) < self.cfg.plate_dropout_p)
            drop_plate = drop_plate.to(plate_known.device).float()
            plate_known = plate_known * (1.0 - drop_plate)

            drop_inst = (torch.rand((n, 1), generator=self._dropout_generator) < self.cfg.instrument_dropout_p)
            drop_inst = drop_inst.to(plate_known.device).float()
            # Hierarchical consistency: masking instrument must also mask plate.
            instrument_known = instrument_known * (1.0 - drop_inst)
            plate_known = plate_known * (1.0 - drop_inst)

        if force_disable_plate_residual:
            plate_known = torch.zeros_like(plate_known)

        e_instrument_eff = e_instrument * instrument_known
        base_input = torch.cat([e_source, e_instrument_eff], dim=-1)
        measure_base_out = self.measure_base(base_input)
        plate_residual_out = self.plate_residual(e_plate)
        return measure_base_out + self.cfg.plate_scale * plate_known * plate_residual_out

    def _learned_flag(self, field: str) -> bool:
        return field in TRAINABLE_UNKNOWN_FIELDS or not self.cfg.mask_unknown_onehot

    def encode(self, cat: torch.Tensor, numeric: torch.Tensor):
        e = {field: self.emb[field](cat[:, i]) for i, field in enumerate(self.CAT_FIELDS)}
        chem = self.chem(e["compound"]) + self.legacy_chem(
            field_one_hot(cat[:, self._compound_idx], self.emb["compound"].num_embeddings, learned_unknown=True)
        )

        bio_onehot = torch.cat(
            [field_one_hot(cat[:, i], self.emb[field].num_embeddings, learned_unknown=self._learned_flag(field))
             for i, field in enumerate(self.CAT_FIELDS) if field in self.BIO_FIELDS] + [numeric], dim=-1)
        bio = self.bio(torch.cat([e[field] for field in self.BIO_FIELDS] + [numeric], dim=-1)) \
            + self.legacy_bio(bio_onehot)

        measure_onehot = torch.cat(
            [field_one_hot(cat[:, i], self.emb[field].num_embeddings, learned_unknown=self._learned_flag(field))
             for i, field in enumerate(self.CAT_FIELDS) if field in self.MEASURE_FIELDS], dim=-1)

        if self.cfg.strict_plate_residual:
            hierarchical = self.hierarchical_measure_context(cat, numeric)
            measure = hierarchical + self.legacy_measure(measure_onehot)
        else:
            measure_input = torch.cat([e["source"], e["instrument"], 0.25 * e["plate"]], dim=-1)
            measure = self.measure(measure_input) + self.legacy_measure(measure_onehot)

        return chem, bio, measure

    def forward(self, cat: torch.Tensor, numeric: torch.Tensor):
        chem, bio, measure = self.encode(cat, numeric)
        context = torch.cat([bio, measure], dim=-1)
        concat = self.concat_head(torch.cat([chem, context], dim=-1))
        gamma, beta = self.modulation(context).chunk(2, dim=-1)
        film = self.film_head((1.0 + gamma) * chem + beta) + 0.05 * self.direct_context(context)

        legacy_chem = self.legacy_chem_net(
            field_one_hot(cat[:, self._compound_idx], self.emb["compound"].num_embeddings, learned_unknown=True)
        )
        legacy_ctx_input = torch.cat(
            [field_one_hot(cat[:, i], self.emb[field].num_embeddings, learned_unknown=self._learned_flag(field))
             for i, field in enumerate(self.CAT_FIELDS) if field != "compound"]
            + [numeric[:, :2]], dim=-1,
        )
        legacy_context = self.legacy_context_net(legacy_ctx_input)
        legacy_gamma, legacy_beta = self.legacy_modulation(legacy_context).chunk(2, dim=-1)
        legacy = self.legacy_head((1.0 + legacy_gamma) * legacy_chem + legacy_beta) \
            + 0.05 * self.legacy_direct(legacy_context)
        return concat, film, legacy

    def regularization(self):
        plate = self.emb["plate"].weight[1:]
        return 1e-4 * torch.mean(plate * plate)


# ---------------------------------------------------------------------------
# Training loop (mirrors scripts/a2b_train_variants.py::fit_model_variant's compound-only
# masked-training path -- same single rng.random() draw for the compound mask, same
# seed_everything() call, same DataLoader(generator=...) convention -- but builds a
# NeutralUnknownHCCEModel instead of rh.HCCEModel and never edits the original file).
# ---------------------------------------------------------------------------

def fit_model_variant_e1(rh, meta, y, fit_indices, mapping, seed: int, epochs: int, device,
                          embedding_dim: int, cfg: UnknownFallbackConfig, mask_p: float = 0.25,
                          dropout_seed: Optional[int] = None):
    rh.seed_everything(seed)
    rng = np.random.default_rng(seed)  # same convention as a2b_train_variants.fit_model_variant
    encoder = rh.HCCEMetaEncoder(mapping).fit(meta, fit_indices)
    cat, numeric = encoder.transform(meta)
    fit_indices = np.asarray(fit_indices, dtype=int)
    fit_cat = cat[fit_indices].copy()
    fit_num = numeric[fit_indices]
    fit_y = y[fit_indices].astype(np.float64)

    # Compound-only masking (A2, unchanged semantics): 25% of compound tokens -> <UNK> (index 0)
    # with true targets kept. Strain is intentionally NEVER masked here -- see module docstring.
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
    if dropout_seed is None:
        # Deterministic, distinct from `seed` and from every other rng consumer in this run.
        dropout_seed = int(seed) + 424242
    model = NeutralUnknownHCCEModel(vocab, numeric.shape[1], y.shape[1], embedding_dim=embedding_dim,
                                     cfg=cfg, dropout_seed=dropout_seed).to(device)
    enforce_neutral_unknown_contract(model)

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
            enforce_neutral_unknown_contract(model)  # defensive; see T1 -- should already be a no-op
            total += float(loss.detach().cpu()) * len(xb_cat)
            seen += len(xb_cat)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, seen)})
    return model.eval(), encoder, target_mean, target_std, history
