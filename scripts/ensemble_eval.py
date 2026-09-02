#!/usr/bin/env python
"""Ensemble evaluation: 3-seed average of a variant's val predictions.

Loads checkpoints from three seed dirs of a variant, averages the legacy and
FiLM experts across seeds, and reports mean50 + FC + entity subsets on the
frozen validation set. No test data loaded.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="three seed dirs of the variant")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dump-parquet", action="store_true",
                    help="also write prediction_val_mean50_<name>.parquet for module-level evaluation")
    args = ap.parse_args()

    rh = load_run_hcce()
    import __main__
    __main__.HCCEMetaEncoder = rh.HCCEMetaEncoder
    torch.set_num_threads(8)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

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
    val_meta = meta.loc[~train_mask].reset_index(drop=True)
    val_true = y[~train_mask]
    val_ids = val_meta[mapping["sample_id"]].astype(str).tolist()

    legacy_parts, film_parts = [], []
    for d in args.dirs:
        ck = torch.load(str(ROOT / d / "model_final.pt"), map_location="cpu")
        enc = rh.joblib.load(ROOT / d / "preprocessor.joblib")
        sd = ck["model"]
        tm, ts = ck["target_mean"], ck["target_std"]
        assert list(ck["proteins"]) == list(proteins)
        cat, num = enc.transform(val_meta)
        emb_key = next(k for k in sd if k.startswith("emb.") and k.endswith(".weight"))
        emb_dim = int(sd[emb_key].shape[1])  # infer from the checkpoint; P3 sweeps dim != 64
        model = rh.HCCEModel(enc.vocab_sizes(), num.shape[1], len(proteins), emb_dim).to(device)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            _, film, legacy = model(torch.from_numpy(cat).long().to(device), torch.from_numpy(num).float().to(device))
        legacy = legacy.cpu().numpy().astype(np.float32) * ts[None, :] + tm[None, :]
        film = film.cpu().numpy().astype(np.float32) * ts[None, :] + tm[None, :]
        legacy_parts.append(legacy)
        film_parts.append(film)
    legacy_ens = np.mean(np.stack(legacy_parts), axis=0)
    film_ens = np.mean(np.stack(film_parts), axis=0)
    mean50 = 0.5 * legacy_ens + 0.5 * film_ens

    if args.dump_parquet:
        frame = pd.DataFrame(mean50.astype(np.float32), columns=list(proteins))
        frame.insert(0, mapping["sample_id"], val_ids)
        frame.to_parquet(out / f"prediction_val_mean50_{args.name}.parquet", index=False)

    train_compounds = set(meta.loc[train_mask, mapping["compound"]].astype(str))
    train_strains = set(meta.loc[train_mask, mapping["strain"]].astype(str))
    comp_unseen = ~val_meta[mapping["compound"]].astype(str).isin(train_compounds).to_numpy()
    strain_unseen = ~val_meta[mapping["strain"]].astype(str).isin(train_strains).to_numpy()
    subset_masks = {
        "all": np.ones(len(val_meta), dtype=bool),
        "strain_unseen_BAI": strain_unseen,
        "compound_unseen_6": comp_unseen,
        "both_seen": (~strain_unseen) & (~comp_unseen),
    }

    m, fc, matches = rh.metrics_with_fc(mean50, val_true, val_meta, mapping)
    report = {
        "name": args.name,
        "overall": {k: float(v) for k, v in m.items() if isinstance(v, (int, float, np.floating, np.integer))},
        "fc_pcc": float(fc["fc_pcc"]) if fc and fc.get("fc_pcc") is not None else None,
        "fc_coverage": float(fc["coverage"]) if fc and fc.get("coverage") is not None else None,
        "subsets": {},
        "by_split": {},
    }
    for sn, sm in subset_masks.items():
        report["subsets"][sn] = {k: float(v) for k, v in rh.basic_metrics(mean50[sm], val_true[sm]).items()
                                 if isinstance(v, (int, float, np.floating, np.integer))}
    splits = rh.evaluate_basic_and_splits(mean50, val_true, val_meta, split_col=mapping["split"], sample_ids=val_ids)
    report["by_split"] = {str(k): {kk: float(vv) for kk, vv in v.items()
                                   if isinstance(vv, (int, float, np.floating, np.integer))}
                          for k, v in splits["by_split"].items()}
    (out / "ensemble_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
