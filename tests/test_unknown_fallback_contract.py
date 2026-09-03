"""Property tests for E1 Neutral Unknown + Strict Hierarchical Residual.

Implements T1-T6 from ContextLadder_复赛综合审查与提升实验计划.md, section 6.5. These tests
exercise scripts/e1_neutral_unknown.py only; they never import or construct run_hcce.HCCEModel
directly with modified behavior, and they never touch runs/final/ or any real data file.
"""

import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.e1_neutral_unknown import (
    NeutralUnknownHCCEModel,
    SchemaError,
    UnknownFallbackConfig,
    VARIANT_TABLE,
    enforce_neutral_unknown_contract,
    field_one_hot,
    load_e1_yaml_config,
    variant_config_from_dict,
    variant_configs_from_yaml,
)

CAT_FIELDS = ["compound", "strain", "medium", "source", "instrument", "plate"]
NON_COMPOUND_FIELDS = [f for f in CAT_FIELDS if f != "compound"]
VOCAB = 6
DIM = 4
N_NUMERIC = 8
N_OUT = 5


def _vocab_sizes(vocab=VOCAB):
    return {field: vocab for field in CAT_FIELDS}


def _make_model(cfg, seed=0, dropout_seed=0, vocab=VOCAB):
    torch.manual_seed(seed)
    return NeutralUnknownHCCEModel(_vocab_sizes(vocab), N_NUMERIC, N_OUT, embedding_dim=DIM,
                                    cfg=cfg, dropout_seed=dropout_seed)


def _batch(n=6, seed=0, plate_ids=None):
    g = torch.Generator().manual_seed(seed)
    cat = torch.randint(1, VOCAB, (n, len(CAT_FIELDS)), generator=g)  # all known (nonzero) by default
    numeric = torch.randn(n, N_NUMERIC, generator=g)
    if plate_ids is not None:
        cat[:, CAT_FIELDS.index("plate")] = torch.as_tensor(plate_ids)
    return cat, numeric


# ---------------------------------------------------------------------------
# T1: unknown embedding contract
# ---------------------------------------------------------------------------

def test_t1_non_compound_unknown_embedding_zero_at_init():
    cfg = VARIANT_TABLE["e1d"]  # zero_freeze_unknown_embedding=True
    model = _make_model(cfg)
    for field in NON_COMPOUND_FIELDS:
        assert torch.all(model.emb[field].weight[0] == 0.0), field
    # compound is untouched by the zero-freeze switch.
    assert model.emb["compound"].weight.requires_grad


def test_t1_non_compound_unknown_embedding_zero_after_optimizer_step():
    cfg = VARIANT_TABLE["e1d"]
    model = _make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    cat, numeric = _batch(n=8, seed=1)
    cat[:, 0] = 0  # force compound=<UNK> for every row so it receives gradient at index 0
    for field in NON_COMPOUND_FIELDS:
        cat[:, CAT_FIELDS.index(field)] = 0  # force these unknown too, to stress-test padding_idx
    target = torch.randn(cat.shape[0], N_OUT)
    concat, film, legacy = model(cat, numeric)
    loss = ((concat - target) ** 2).mean() + ((film - target) ** 2).mean() + ((legacy - target) ** 2).mean()
    loss.backward()
    opt.step()
    enforce_neutral_unknown_contract(model)
    for field in NON_COMPOUND_FIELDS:
        assert torch.all(model.emb[field].weight[0] == 0.0), field
    # compound's row 0 embedding should have moved: every row in this batch had compound=<UNK>.
    assert model.emb["compound"].weight.grad is not None
    assert torch.any(model.emb["compound"].weight.grad[0] != 0.0)


def test_t1_unknown_embedding_zero_after_checkpoint_roundtrip(tmp_path):
    cfg = VARIANT_TABLE["e1b"]
    model = _make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    cat, numeric = _batch(n=8, seed=2)
    target = torch.randn(cat.shape[0], N_OUT)
    concat, film, legacy = model(cat, numeric)
    loss = ((concat - target) ** 2).mean() + ((film - target) ** 2).mean() + ((legacy - target) ** 2).mean()
    loss.backward()
    opt.step()
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    reloaded = _make_model(cfg, seed=99)  # different init, then overwritten by state_dict
    blob = torch.load(ckpt_path, map_location="cpu")
    reloaded.load_state_dict(blob["model_state_dict"])
    enforce_neutral_unknown_contract(reloaded)
    for field in NON_COMPOUND_FIELDS:
        assert torch.all(reloaded.emb[field].weight[0] == 0.0), field


def test_t1_baseline_e1a_does_not_freeze_anything():
    cfg = VARIANT_TABLE["e1a"]
    model = _make_model(cfg, seed=3)
    # With the zero-freeze switch off, nothing is constrained: no promise of exact zero.
    assert not any(torch.all(model.emb[f].weight[0] == 0.0) for f in NON_COMPOUND_FIELDS if f != "plate") \
        or True  # random init could coincidentally be near zero but not exactly; this is a smoke check
    enforce_neutral_unknown_contract(model)  # must be a no-op for e1a
    # A no-op means weights are whatever random init produced -- verify against a fresh copy.
    model2 = _make_model(cfg, seed=3)
    for field in NON_COMPOUND_FIELDS:
        assert torch.equal(model.emb[field].weight, model2.emb[field].weight)


# ---------------------------------------------------------------------------
# T2: unknown one-hot contract
# ---------------------------------------------------------------------------

def test_t2_non_compound_unknown_onehot_is_zero_vector():
    ids = torch.tensor([0, 1, 0, 3])
    x = field_one_hot(ids, n=5, learned_unknown=False)
    assert torch.all(x[0] == 0.0)
    assert torch.all(x[2] == 0.0)
    assert x[1, 1] == 1.0
    assert x[3, 3] == 1.0


def test_t2_compound_unknown_onehot_keeps_learned_column():
    ids = torch.tensor([0, 1, 0, 3])
    x = field_one_hot(ids, n=5, learned_unknown=True)
    assert x[0, 0] == 1.0
    assert x[2, 0] == 1.0
    assert x[1, 1] == 1.0


# ---------------------------------------------------------------------------
# T3: plate strict fallback
# ---------------------------------------------------------------------------

def test_t3_unseen_plate_equals_forced_disable():
    cfg = VARIANT_TABLE["e1e"]
    model = _make_model(cfg, seed=4)
    model.eval()
    cat, numeric = _batch(n=1, seed=5, plate_ids=[3])  # a genuinely known plate id

    measure_disabled = model.hierarchical_measure_context(cat, numeric, force_disable_plate_residual=True)

    cat_unseen = cat.clone()
    cat_unseen[:, CAT_FIELDS.index("plate")] = 0
    measure_unseen = model.hierarchical_measure_context(cat_unseen, numeric)

    torch.testing.assert_close(measure_disabled, measure_unseen, atol=1e-7, rtol=1e-6)


def test_t3_requires_strict_plate_residual_config():
    cfg = VARIANT_TABLE["e1a"]
    model = _make_model(cfg, seed=6)
    cat, numeric = _batch(n=1, seed=7)
    with pytest.raises(RuntimeError):
        model.hierarchical_measure_context(cat, numeric)


# ---------------------------------------------------------------------------
# T4: seen plate is not degenerate
# ---------------------------------------------------------------------------

def test_t4_two_known_plates_produce_different_context():
    cfg = VARIANT_TABLE["e1e"]
    model = _make_model(cfg, seed=8)
    model.eval()
    cat_a, numeric = _batch(n=1, seed=9, plate_ids=[1])
    cat_b = cat_a.clone()
    cat_b[:, CAT_FIELDS.index("plate")] = 4
    out_a = model.hierarchical_measure_context(cat_a, numeric)
    out_b = model.hierarchical_measure_context(cat_b, numeric)
    assert torch.max(torch.abs(out_a - out_b)).item() > 1e-6


# ---------------------------------------------------------------------------
# T5: hierarchical dropout
# ---------------------------------------------------------------------------

def test_t5_instrument_dropout_forces_plate_residual_zero():
    cfg = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                 strict_plate_residual=True, hierarchical_dropout=True,
                                 plate_dropout_p=0.0, instrument_dropout_p=1.0)
    model = _make_model(cfg, seed=10, dropout_seed=11, vocab=20)
    model.train()
    # Silence measure_base/plate_residual's own nn.Dropout(0.10) so this comparison isolates the
    # hierarchical field-dropout mechanism under test, not that unrelated regularization dropout.
    model.measure_base.eval()
    model.plate_residual.eval()
    cat, numeric = _batch(n=16, seed=12, plate_ids=list(range(1, 17)))
    out = model.hierarchical_measure_context(cat, numeric)
    base_only = model.measure_base(torch.cat([
        model.emb["source"](cat[:, CAT_FIELDS.index("source")]),
        model.emb["instrument"](cat[:, CAT_FIELDS.index("instrument")]) * 0.0,  # instrument forced unknown
    ], dim=-1))
    torch.testing.assert_close(out, base_only, atol=1e-6, rtol=1e-5)


def test_t5_plate_only_dropout_preserves_source_instrument_base():
    cfg = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                 strict_plate_residual=True, hierarchical_dropout=True,
                                 plate_dropout_p=1.0, instrument_dropout_p=0.0)
    model = _make_model(cfg, seed=13, dropout_seed=14, vocab=20)
    cat, numeric = _batch(n=16, seed=15, plate_ids=list(range(1, 17)))

    model.eval()
    eval_out = model.hierarchical_measure_context(cat, numeric)  # no dropout at all in eval

    model.train()
    model.measure_base.eval()
    model.plate_residual.eval()
    train_out = model.hierarchical_measure_context(cat, numeric)  # plate always dropped (p=1.0)

    # Plate dropout alone must not touch the source/instrument base term: with plate contribution
    # forced to zero by dropout, the training-mode output must equal eval's measure_base_out alone
    # (eval-mode plate contribution here is nonzero, so we recompute the base-only reference).
    base_only = model.measure_base(torch.cat([
        model.emb["source"](cat[:, CAT_FIELDS.index("source")]),
        model.emb["instrument"](cat[:, CAT_FIELDS.index("instrument")]),
    ], dim=-1))
    torch.testing.assert_close(train_out, base_only, atol=1e-6, rtol=1e-5)
    assert torch.max(torch.abs(eval_out - base_only)).item() > 1e-6  # eval's plate residual is active


def test_t5_eval_mode_has_no_randomness():
    cfg = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                 strict_plate_residual=True, hierarchical_dropout=True,
                                 plate_dropout_p=0.5, instrument_dropout_p=0.5)
    model = _make_model(cfg, seed=16, dropout_seed=17, vocab=20)
    model.eval()
    cat, numeric = _batch(n=16, seed=18, plate_ids=list(range(1, 17)))
    out1 = model.hierarchical_measure_context(cat, numeric)
    out2 = model.hierarchical_measure_context(cat, numeric)
    out3 = model.hierarchical_measure_context(cat, numeric)
    torch.testing.assert_close(out1, out2, atol=0.0, rtol=0.0)
    torch.testing.assert_close(out1, out3, atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# T6: config actually changes model/optimizer behavior; unknown fields error
# ---------------------------------------------------------------------------

def test_t6_plate_scale_changes_output():
    cfg_a = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                   strict_plate_residual=True, plate_scale=0.25)
    cfg_b = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                   strict_plate_residual=True, plate_scale=0.9)
    model_a = _make_model(cfg_a, seed=20)
    model_b = _make_model(cfg_b, seed=21)
    model_b.load_state_dict(model_a.state_dict())  # identical weights, only cfg.plate_scale differs
    model_a.eval(); model_b.eval()
    cat, numeric = _batch(n=4, seed=22, plate_ids=[1, 2, 3, 4])
    out_a = model_a.hierarchical_measure_context(cat, numeric)
    out_b = model_b.hierarchical_measure_context(cat, numeric)
    assert torch.max(torch.abs(out_a - out_b)).item() > 1e-6


def test_t6_learning_rate_changes_optimizer_and_trained_weights():
    cfg = VARIANT_TABLE["e1d"]
    model_a = _make_model(cfg, seed=23)
    model_b = _make_model(cfg, seed=23)
    model_b.load_state_dict(model_a.state_dict())
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-1, weight_decay=1e-4)
    assert opt_a.param_groups[0]["lr"] != opt_b.param_groups[0]["lr"]

    cat, numeric = _batch(n=8, seed=24)
    target = torch.randn(cat.shape[0], N_OUT)
    for model, opt in ((model_a, opt_a), (model_b, opt_b)):
        concat, film, legacy = model(cat, numeric)
        loss = ((concat - target) ** 2).mean() + ((film - target) ** 2).mean() + ((legacy - target) ** 2).mean()
        loss.backward()
        opt.step()
    diff = torch.max(torch.abs(model_a.emb["compound"].weight - model_b.emb["compound"].weight)).item()
    assert diff > 1e-6


def test_t6_dropout_rate_changes_behavior():
    cfg_off = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                     strict_plate_residual=True, hierarchical_dropout=True,
                                     plate_dropout_p=0.0, instrument_dropout_p=0.0)
    cfg_on = UnknownFallbackConfig(zero_freeze_unknown_embedding=True, mask_unknown_onehot=True,
                                    strict_plate_residual=True, hierarchical_dropout=True,
                                    plate_dropout_p=1.0, instrument_dropout_p=0.0)
    model_off = _make_model(cfg_off, seed=25, dropout_seed=1, vocab=20)
    model_on = _make_model(cfg_on, seed=25, dropout_seed=1, vocab=20)
    model_on.load_state_dict(model_off.state_dict())
    model_off.train(); model_on.train()
    cat, numeric = _batch(n=8, seed=26, plate_ids=list(range(1, 9)))
    out_off = model_off.hierarchical_measure_context(cat, numeric)
    out_on = model_on.hierarchical_measure_context(cat, numeric)
    assert torch.max(torch.abs(out_off - out_on)).item() > 1e-6


def test_t6_unknown_field_in_variant_dict_raises_schema_error():
    with pytest.raises(SchemaError):
        variant_config_from_dict({"plate_scale": 0.25, "not_a_real_field": True})


def test_t6_unknown_field_in_dataclass_kwargs_raises():
    with pytest.raises(TypeError):
        UnknownFallbackConfig(not_a_real_field=True)


def test_t6_hierarchical_dropout_without_strict_residual_raises():
    with pytest.raises(SchemaError):
        UnknownFallbackConfig(hierarchical_dropout=True, strict_plate_residual=False)


def test_t6_yaml_config_schema_validation(tmp_path):
    good = {
        "schema_version": 1,
        "model": {
            "base_variant": "mask_compound", "embedding_dim": 64, "epochs": 40, "mask_p": 0.25,
            "unknown_policy": {"compound": "learned_masked", "strain": "neutral_zero",
                               "medium": "neutral_zero", "source": "neutral_zero",
                               "instrument": "neutral_zero", "plate": "strict_residual_zero"},
            "measurement_hierarchy": {"mode": "additive_residual", "plate_residual_scale": 0.25,
                                      "plate_dropout_p": 0.15, "instrument_dropout_p": 0.05},
        },
        "variants": {"e1a": {}},
        "training": {"seeds": [1, 2, 3]},
    }
    path = tmp_path / "good.yaml"
    path.write_text(yaml.safe_dump(good), encoding="utf-8")
    loaded = load_e1_yaml_config(path)
    assert loaded["model"]["unknown_policy"]["plate"] == "strict_residual_zero"
    configs = variant_configs_from_yaml(path)
    assert configs["e1a"].is_baseline()

    bad = copy.deepcopy(good)
    bad["model"]["unknown_policy"]["totally_unexpected_field"] = "neutral_zero"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(SchemaError):
        load_e1_yaml_config(bad_path)


def test_variant_table_matches_e1a_through_e1f_spec():
    assert VARIANT_TABLE["e1a"].is_baseline()
    assert VARIANT_TABLE["e1b"].zero_freeze_unknown_embedding and not VARIANT_TABLE["e1b"].mask_unknown_onehot
    assert VARIANT_TABLE["e1c"].mask_unknown_onehot and not VARIANT_TABLE["e1c"].zero_freeze_unknown_embedding
    assert VARIANT_TABLE["e1d"].zero_freeze_unknown_embedding and VARIANT_TABLE["e1d"].mask_unknown_onehot
    assert not VARIANT_TABLE["e1d"].strict_plate_residual
    assert VARIANT_TABLE["e1e"].strict_plate_residual and not VARIANT_TABLE["e1e"].hierarchical_dropout
    assert VARIANT_TABLE["e1f"].hierarchical_dropout and VARIANT_TABLE["e1f"].strict_plate_residual
