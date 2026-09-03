#!/usr/bin/env python
"""E1 ablation runner: E1a-E1f x {20260810, 3407, 42} on the frozen official validation split.

ContextLadder_复赛综合审查与提升实验计划.md, chapter 6. Trains six mechanism variants (see
configs/e1_unknown_fallback.yaml / scripts/e1_neutral_unknown.py::VARIANT_TABLE) at three fixed
seeds each, using ONLY split_final=='train' rows (5,920) to fit anything -- categorical
vocabularies, normalization statistics, and model weights. Metrics are reported on the frozen
official validation rows (split_final != 'train', still inside the train_val file, never the
test file), broken out by the val_strain_only / val_chem_only / val_both / val_time categories.

The test proteome (WAYB_WAYC_proteome_raw_test.csv) and test metadata are never opened here --
only configs/data_paths.yaml's metadata_train_val / proteome_train_val, exactly like
scripts/train.py and scripts/a2b_train_variants.py.

Before training anything, this script hashes configs/e1_unknown_fallback_gates.yaml and writes
the digest to <output-dir>/gates_sha256.json so the gate thresholds cannot be edited after the
fact without it being visible.

Usage:
    python scripts/run_unknown_fallback_ablation.py \
        --config configs/e1_unknown_fallback.yaml \
        --gates-config configs/e1_unknown_fallback_gates.yaml \
        --variants e1a e1b e1c e1d e1e e1f \
        --seeds 20260810 3407 42 \
        --output-dir outputs/e1_unknown_fallback
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.a2b_train_variants import load_run_hcce
from scripts.e1_neutral_unknown import (
    enforce_neutral_unknown_contract,
    fit_model_variant_e1,
    load_e1_yaml_config,
    variant_configs_from_yaml,
)
from scripts.run_baselines import json_safe

SUBSET_SPLIT_VALUES = {
    "all": None,                    # every validation row
    "strain_unseen": "val_strain_only",
    "compound_unseen": "val_chem_only",
    "both_unseen": "val_both",
    "both_seen_time_shift": "val_time",   # bonus context, not a required subset
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_data(rh, mapping, paths, epoch_filter_threshold=0.80):
    meta = rh.load_metadata(ROOT / paths["metadata_train_val"])
    prot, _, _ = rh.load_proteome(ROOT / paths["proteome_train_val"])
    meta, prot, _, all_proteins = rh.align_metadata_proteome(meta, prot)
    raw = rh.finite_float_matrix(prot, all_proteins)
    proteins, missing_rate, keep_mask = rh.apply_official_protein_filter(
        meta, raw, all_proteins, mapping, threshold=epoch_filter_threshold)
    raw = raw[:, keep_mask]
    y = rh.to_log2_proteome(raw).astype(np.float64)
    return meta, y, proteins


def subset_masks(val_meta, mapping):
    split_col = val_meta[mapping["split"]].astype(str)
    masks = {}
    for name, value in SUBSET_SPLIT_VALUES.items():
        masks[name] = np.ones(len(val_meta), dtype=bool) if value is None else (split_col == value).to_numpy()
    return masks


def run_one(rh, meta, y, mapping, train_indices, val_indices, val_meta, val_true, masks,
            variant, cfg, seed, epochs, embedding_dim, mask_p, device, out_dir):
    started = time.time()
    fit_seed = int(seed) + 9000  # matches scripts/train.py's submission-path convention
    model, encoder, target_mean, target_std, history = fit_model_variant_e1(
        rh, meta, y, train_indices, mapping, fit_seed, epochs, device, embedding_dim,
        cfg, mask_p=mask_p,
    )
    enforce_neutral_unknown_contract(model)

    hcce_concat_val, film_val, legacy_val = rh.predict_model(model, encoder, target_mean, target_std, val_meta, device)
    mean50 = 0.5 * legacy_val + 0.5 * film_val

    metrics = {}
    for name, mask in masks.items():
        if not mask.any():
            metrics[name] = {"n_samples": 0}
            continue
        m = rh.basic_metrics(mean50[mask], val_true[mask])
        metrics[name] = {k: v for k, v in m.items()}

    run_dir = out_dir / variant / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "proteins": None,
        "target_mean": target_mean, "target_std": target_std,
        "variant": variant, "cfg": cfg.__dict__, "seed": seed, "fit_seed": fit_seed,
    }, run_dir / "model_final.pt")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    result = {
        "variant": variant, "seed": int(seed), "fit_seed": fit_seed,
        "cfg": cfg.__dict__, "epochs": epochs, "embedding_dim": embedding_dim, "mask_p": mask_p,
        "fit_rows": int(len(train_indices)), "validation_rows": int(len(val_indices)),
        "subset_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "metrics": metrics,
        "no_test_truth": True, "no_test_proteome": True,
        "elapsed_sec": time.time() - started,
    }
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/e1_unknown_fallback.yaml"))
    ap.add_argument("--gates-config", default=str(ROOT / "configs/e1_unknown_fallback_gates.yaml"))
    ap.add_argument("--variants", nargs="+", default=["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[20260810, 3407, 42])
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/e1_unknown_fallback"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None, help="default: model.epochs from --config")
    ap.add_argument("--embedding-dim", type=int, default=None, help="default: model.embedding_dim from --config")
    ap.add_argument("--mask-p", type=float, default=None, help="default: model.mask_p from --config")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hard constraint: gates must be frozen and their hash logged BEFORE any training happens.
    gates_path = Path(args.gates_config)
    gates_sha256 = sha256_file(gates_path)
    gates_record = {"gates_config": str(gates_path), "sha256": gates_sha256,
                     "recorded_at_unix": time.time(), "recorded_before_any_training": True}
    (out_dir / "gates_sha256.json").write_text(json.dumps(gates_record, indent=2), encoding="utf-8")
    print(f"[gate] {gates_path} sha256={gates_sha256} (recorded to {out_dir/'gates_sha256.json'} before training)", flush=True)

    e1_cfg_raw = load_e1_yaml_config(args.config)   # raises SchemaError on any unrecognized key
    variant_cfgs = variant_configs_from_yaml(args.config)
    for name in args.variants:
        if name not in variant_cfgs:
            raise SystemExit(f"unknown variant '{name}'; available: {sorted(variant_cfgs)}")

    epochs = args.epochs if args.epochs is not None else int(e1_cfg_raw["model"]["epochs"])
    embedding_dim = args.embedding_dim if args.embedding_dim is not None else int(e1_cfg_raw["model"]["embedding_dim"])
    mask_p = args.mask_p if args.mask_p is not None else float(e1_cfg_raw["model"]["mask_p"])

    rh = load_run_hcce()
    device = torch.device(args.device)
    if args.device.startswith("cuda") and device.type != "cuda":
        raise RuntimeError("CUDA requested but unavailable")

    paths = yaml.safe_load((ROOT / "configs/data_paths.yaml").read_text(encoding="utf-8"))
    mapping = yaml.safe_load((ROOT / "configs/field_mapping.yaml").read_text(encoding="utf-8"))
    assert "metadata_test" not in str(paths.get("metadata_train_val", "")), "sanity: wrong path key"

    meta, y, proteins = load_data(rh, mapping, paths)
    split = meta[mapping["split"]].astype(str).to_numpy()
    train_mask = split == "train"
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(~train_mask)
    assert (split[train_indices] == "train").all(), "fit index contains non-train rows"
    val_meta = meta.iloc[val_indices].reset_index(drop=True)
    val_true = y[val_indices]
    masks = subset_masks(val_meta, mapping)
    subset_sizes = {k: int(v.sum()) for k, v in masks.items()}
    print(f"[data] fit_rows={len(train_indices)} val_rows={len(val_indices)} subset_sizes={subset_sizes}", flush=True)

    all_results = []
    started_all = time.time()
    for variant in args.variants:
        cfg = variant_cfgs[variant]
        for seed in args.seeds:
            t0 = time.time()
            try:
                result = run_one(rh, meta, y, mapping, train_indices, val_indices, val_meta, val_true,
                                  masks, variant, cfg, seed, epochs, embedding_dim, mask_p, device, out_dir)
                status = "ok"
                error = None
            except Exception as exc:  # noqa: BLE001 -- must record failures, not skip them
                status = "FAILED"
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                result = {"variant": variant, "seed": int(seed), "cfg": cfg.__dict__, "metrics": {}}
                print(f"[FAIL] {variant} seed={seed}:\n{error}", file=sys.stderr, flush=True)
            result["status"] = status
            result["error"] = error
            all_results.append(result)
            print(f"[run] {variant} seed={seed} status={status} elapsed={time.time()-t0:.1f}s", flush=True)

    summary = {
        "gates_sha256": gates_sha256,
        "e1_config": str(args.config), "e1_config_sha256": sha256_file(args.config),
        "variants_run": list(args.variants), "seeds": list(args.seeds),
        "epochs": epochs, "embedding_dim": embedding_dim, "mask_p": mask_p,
        "fit_rows": int(len(train_indices)), "validation_rows": int(len(val_indices)),
        "subset_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "results": all_results,
        "no_test_truth": True, "no_test_proteome": True,
        "elapsed_sec_total": time.time() - started_all,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe({k: v for k, v in summary.items() if k != "results"}), ensure_ascii=False, indent=2))
    n_failed = sum(1 for r in all_results if r["status"] != "ok")
    print(f"wrote {out_dir/'summary.json'}; {len(all_results)} runs, {n_failed} failed")


if __name__ == "__main__":
    main()
