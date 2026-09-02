#!/usr/bin/env python
"""Train the final ContextLadder / HCCE-Proteome ensemble from scratch.

Reads only the official *training* metadata and proteome. Test metadata is not
opened here at all, and the test proteome is never opened anywhere in this
package. Every fitted quantity -- the retained-protein list, the categorical
maps, the temperature/time normalisation, the target mean/std -- comes from
`split_final == "train"` rows only; validation rows are held out and a hard
assertion stops the run if any of them reaches the fit index.

Writes one checkpoint per ensemble member plus the preprocessing contract, so
that inference (scripts/predict.py) needs nothing but test metadata and this
run directory.

    python scripts/train.py --metadata <train_metadata> --proteome <train_proteome> \
        --config configs/final.yaml --output-dir runs/final
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2b_train_variants import fit_model_variant, load_run_hcce
from src.data.io import (align_metadata_proteome, finite_float_matrix,
                         load_metadata, load_proteome, to_log2_proteome)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", help="official train_val metadata CSV (default: configs/data_paths.yaml)")
    ap.add_argument("--proteome", help="official train_val proteome CSV (default: configs/data_paths.yaml)")
    ap.add_argument("--config", default=str(ROOT / "configs/final.yaml"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    started = time.time()
    cfg_text = Path(args.config).read_text(encoding="utf-8")
    cfg = yaml.safe_load(cfg_text)
    config_hash = sha256_text(cfg_text)

    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    meta_path = Path(args.metadata) if args.metadata else ROOT / paths["metadata_train_val"]
    prot_path = Path(args.proteome) if args.proteome else ROOT / paths["proteome_train_val"]

    out = Path(args.output_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)

    rh = load_run_hcce()
    device = torch.device(args.device)

    meta = load_metadata(meta_path)
    prot, _, _ = load_proteome(prot_path)
    meta, prot, _, all_proteins = align_metadata_proteome(meta, prot)
    raw_all = finite_float_matrix(prot, all_proteins)

    thr = float(cfg["preprocessing"]["protein_filter"]["threshold"])
    proteins, missing_rate, keep_mask = rh.apply_official_protein_filter(
        meta, raw_all, all_proteins, mapping, threshold=thr)
    y = to_log2_proteome(raw_all[:, keep_mask]).astype(np.float64)

    split = meta[mapping["split"]].astype(str).to_numpy()
    train_rows = split == "train"
    fit_indices = np.flatnonzero(train_rows).astype(int)
    # Hard data-boundary assertion: nothing outside the train split may be fitted on.
    assert fit_indices.size > 0 and (split[fit_indices] == "train").all(), \
        "fit index contains non-train rows"
    n_val = int((~train_rows).sum())

    expected = cfg["preprocessing"]["protein_filter"].get("expected_protein_count")
    if expected is not None and len(proteins) != int(expected):
        print(f"WARNING: retained {len(proteins)} proteins, config expects {expected}", flush=True)

    protein_list_path = out / "artifacts" / "protein_list.txt"
    protein_list_path.write_text("\n".join(proteins) + "\n", encoding="utf-8")

    # EMA is read from the config rather than defaulted. Before this passthrough existed,
    # setting ema_decay in final.yaml was a silent no-op: fit_model_variant would fall back
    # to 0.0 and train a model that was not the configured one, without erroring.
    ema_decay = float(cfg["model"].get("ema_decay", 0.0))

    members = []
    for seed in cfg["ensemble"]["seeds"]:
        t0 = time.time()
        model, encoder, target_mean, target_std, history = fit_model_variant(
            rh, meta, y, fit_indices, mapping, int(seed),
            int(cfg["model"]["epochs"]), device, int(cfg["model"]["embedding_dim"]),
            cfg["model"]["variant"], mask_p=float(cfg["model"]["mask_p"]),
            ema_decay=ema_decay,
        )
        ckpt = out / "checkpoints" / f"seed{seed}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_kwargs": {"embedding_dim": int(cfg["model"]["embedding_dim"])},
            "n_numeric": int(encoder.transform(meta.iloc[:1])[1].shape[1]),
            "encoder": encoder,
            "target_mean": target_mean,
            "target_std": target_std,
            "proteins": list(proteins),
            "seed": int(seed),
            "config_hash": config_hash,
            "variant": cfg["model"]["variant"],
            "epochs": int(cfg["model"]["epochs"]),
            "ema_decay": ema_decay,
        }, ckpt)
        members.append({
            "seed": int(seed), "checkpoint": str(ckpt.relative_to(out)),
            "epochs": int(cfg["model"]["epochs"]), "ema_decay": ema_decay,
            "sha256": sha256_file(ckpt), "train_seconds": round(time.time() - t0, 2),
            "final_train_loss": float(history[-1]["train_loss"]) if history else None,
        })
        print(f"seed {seed} trained -> {ckpt.name}", flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        log2_train_only = to_log2_proteome(raw_all[train_rows]).astype(np.float64)
        train_mean_kept = np.nanmean(log2_train_only[:, keep_mask], axis=0)
        global_train_mean = float(np.nanmean(log2_train_only))
    np.save(out / "artifacts" / "train_protein_mean_kept.npy", train_mean_kept)

    contract = {
        "protein_space": "contract_4422",
        "n_proteins": int(len(proteins)),
        "protein_list_file": "artifacts/protein_list.txt",
        "protein_list_sha256": sha256_file(protein_list_path),
        "filter_rule": cfg["preprocessing"]["protein_filter"]["rule"],
        "filter_threshold": thr,
        "filter_fit_scope": "split_final == 'train' only",
        "n_raw_protein_columns": int(len(all_proteins)),
        "n_dropped": int(len(all_proteins) - len(proteins)),
        "fit_rows": int(fit_indices.size),
        "held_out_val_rows": n_val,
        "total_train_val_rows": int(len(meta)),
        "target_space": "log2(raw)",
        "global_train_mean": global_train_mean,
        "test_proteome_read": False,
        "test_metadata_read": False,
    }
    (out / "artifacts" / "preprocess_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    def portable(path):
        """Record paths relative to the package root so a reviewer's manifest
        does not carry our development tree."""
        try:
            return str(Path(path).resolve().relative_to(ROOT))
        except ValueError:
            return str(path)

    manifest = {
        "run_id": cfg["run_id"],
        "paths_relative_to_package_root": True,
        "config_file": portable(args.config),
        "config_sha256": config_hash,
        "metadata_file": portable(meta_path), "metadata_sha256": sha256_file(meta_path),
        "proteome_file": portable(prot_path), "proteome_sha256": sha256_file(prot_path),
        "members": members,
        "ensemble": {"combine": cfg["ensemble"]["combine"],
                     "expert_blend": cfg["model"]["expert_blend"]},
        "preprocess_contract": "artifacts/preprocess_contract.json",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "device": str(device),
                        "cuda": torch.version.cuda, "platform": platform.platform()},
        "elapsed_sec": round(time.time() - started, 2),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "members"},
                     ensure_ascii=False, indent=2))
    print(f"members: {[m['seed'] for m in members]}; wrote {out/'run_manifest.json'}")


if __name__ == "__main__":
    main()
