#!/usr/bin/env python
"""Frozen-model inference: test metadata + a training run directory -> prediction.csv.

This script never opens a proteome file. It needs only the official *test*
metadata and the artifacts written by scripts/train.py (checkpoints, encoder,
target statistics, protein contract). The test proteome and any test-derived
cache are out of scope by construction, not by convention.

    python scripts/predict.py --metadata <test_metadata> --run-dir runs/final \
        --output prediction.csv
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2b_train_variants import load_run_hcce
from src.data.io import load_metadata


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", help="official test metadata CSV (default: configs/data_paths.yaml)")
    ap.add_argument("--run-dir", required=True, help="output-dir produced by scripts/train.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--protein-space", default="contract_4422", choices=["contract_4422", "full_5243"],
                    help="contract_4422 is the submission format; full_5243 is kept only for "
                         "traceability against the preliminary-round artifact and pads the "
                         "dropped proteins with the train-split per-protein mean.")
    args = ap.parse_args()

    started = time.time()
    run = Path(args.run_dir)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((run / "artifacts" / "preprocess_contract.json").read_text(encoding="utf-8"))
    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    meta_path = Path(args.metadata) if args.metadata else ROOT / paths["metadata_test"]

    rh = load_run_hcce()
    device = torch.device(args.device)
    test_meta = load_metadata(meta_path).reset_index(drop=True)

    legacy_parts, film_parts, used = [], [], []
    for member in manifest["members"]:
        ckpt_path = run / member["checkpoint"]
        digest = sha256_file(ckpt_path)
        if digest != member["sha256"]:
            raise SystemExit(f"checkpoint {ckpt_path} sha256 mismatch: manifest says {member['sha256']}")
        blob = torch.load(ckpt_path, map_location=device)
        encoder = blob["encoder"]
        model = rh.HCCEModel(encoder.vocab_sizes(), int(blob["n_numeric"]),
                             len(blob["proteins"]),
                             embedding_dim=blob["model_kwargs"]["embedding_dim"]).to(device)
        model.load_state_dict(blob["model_state_dict"])
        model.eval()
        _, film, legacy = rh.predict_model(model, encoder, blob["target_mean"], blob["target_std"],
                                           test_meta, device)
        film_parts.append(film)
        legacy_parts.append(legacy)
        used.append({"seed": member["seed"], "checkpoint_sha256": digest})
        print(f"seed {member['seed']} inferred", flush=True)

    blend = manifest["ensemble"]["expert_blend"]
    pred_kept = (float(blend["legacy_film"]) * np.mean(np.stack(legacy_parts), axis=0)
                 + float(blend["hcce_film"]) * np.mean(np.stack(film_parts), axis=0)).astype(np.float32)

    proteins = (run / "artifacts" / "protein_list.txt").read_text(encoding="utf-8").split("\n")
    proteins = [p for p in proteins if p]
    if args.protein_space == "contract_4422":
        columns, values = proteins, pred_kept
    else:
        full = json.loads((ROOT / "configs/protein_feature_contract.json").read_text(encoding="utf-8"))
        columns = full["protein_columns"]
        train_mean = np.load(run / "artifacts" / "train_protein_mean_kept.npy")
        pos = {p: i for i, p in enumerate(proteins)}
        fill = float(contract["global_train_mean"])
        values = np.tile(np.float32(fill), (len(test_meta), len(columns)))
        idx = [pos[c] for c in columns if c in pos]
        col_idx = [i for i, c in enumerate(columns) if c in pos]
        values[:, col_idx] = pred_kept[:, idx]
        del train_mean

    if not np.isfinite(values).all():
        raise SystemExit("prediction contains non-finite values")

    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, mapping["sample_id"], test_meta[mapping["sample_id"]].astype(str).to_numpy())
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outfile, index=False)

    pred_manifest = {
        "prediction_file": str(outfile),
        "prediction_sha256": sha256_file(outfile),
        "prediction_scale": "log2",
        "protein_space": args.protein_space,
        "n_rows": int(len(frame)),
        "n_protein_columns": int(len(columns)),
        "sample_id_column": mapping["sample_id"],
        "sample_ids_unique": bool(frame[mapping["sample_id"]].is_unique),
        "sample_order_matches_test_metadata": bool(
            frame[mapping["sample_id"]].tolist()
            == test_meta[mapping["sample_id"]].astype(str).tolist()),
        "all_values_finite": True,
        "run_id": manifest["run_id"],
        "config_sha256": manifest["config_sha256"],
        "protein_list_sha256": contract["protein_list_sha256"],
        "ensemble_members": used,
        "test_proteome_read": False,
        "test_truth_read": False,
        "elapsed_sec": round(time.time() - started, 2),
    }
    (outfile.parent / "prediction_manifest.json").write_text(
        json.dumps(pred_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pred_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
