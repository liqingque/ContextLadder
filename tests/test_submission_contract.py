"""Contract tests that need no competition data.

Checks the things a reviewer would otherwise have to verify by reading: that
the frozen config declares the submission format the specification requires,
that the inference entry point cannot reach a proteome file, and that the
training entry point carries its data-boundary assertion.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_final_config_declares_submission_format():
    cfg = yaml.safe_load((ROOT / "configs/final.yaml").read_text(encoding="utf-8"))
    assert cfg["output"]["prediction_scale"] == "log2"
    assert cfg["output"]["protein_space"] == "contract_4422"
    assert cfg["output"]["n_rows"] == 4454
    assert cfg["preprocessing"]["protein_filter"]["expected_protein_count"] == 4422
    assert cfg["preprocessing"]["protein_filter"]["threshold"] == 0.80


def test_final_config_declares_train_only_boundary():
    cfg = yaml.safe_load((ROOT / "configs/final.yaml").read_text(encoding="utf-8"))
    assert "train" in cfg["data_boundary"]["training_labels"]
    assert cfg["external_features"]["used_by_final_model"] is False


def test_predict_never_opens_a_proteome():
    src = (ROOT / "scripts/predict.py").read_text(encoding="utf-8")
    assert "load_proteome" not in src
    assert "proteome_test" not in src
    assert "proteome_train_val" not in src


def test_train_asserts_the_data_boundary():
    src = (ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert 'split[fit_indices] == "train"' in src
    assert "metadata_test" not in src


def test_ensemble_members_are_frozen():
    cfg = yaml.safe_load((ROOT / "configs/final.yaml").read_text(encoding="utf-8"))
    assert cfg["ensemble"]["seeds"] == [20260810, 3407, 42]
    assert cfg["model"]["expert_blend"] == {"legacy_film": 0.5, "hcce_film": 0.5}
