#!/usr/bin/env python
"""Emit REPRODUCIBILITY_MANIFEST.json — the single file that ties the package together.

Cross-references the frozen config, every ensemble member checkpoint, the
preprocessing contract, the prediction file and the three reproduction
commands, each with a SHA256. Its purpose is the reviewer's version check:
"do the documentation, the code entry points, the configuration, the
checkpoints and the prediction all correspond to the same version?"

    python scripts/build_reproducibility_manifest.py --run-dir runs/final \
        --prediction prediction.csv --output REPRODUCIBILITY_MANIFEST.json
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--output", default=str(ROOT / "REPRODUCIBILITY_MANIFEST.json"))
    ap.add_argument("--work-id", default="TorchDragon_ContextLadder",
                    help="work identity: organizers did not issue a numeric work ID, "
                         "so the work is identified by team name + work name")
    args = ap.parse_args()

    run = Path(args.run_dir)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((run / "artifacts" / "preprocess_contract.json").read_text(encoding="utf-8"))

    payload = {
        "schema": "goai-vc-reproducibility-manifest/1",
        "work_id": args.work_id,
        "work_name": "上下文阶梯 ContextLadder",
        "team": "TorchDragon",
        "final_model": manifest["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "commands": {
            "1_build_external_features": {
                "command": "python scripts/build_embeddings.py --output artifacts/embeddings",
                "required": False,
                "note": "the final model uses no external features; this step can be skipped",
            },
            "2_train": {
                "command": ("python scripts/train.py --metadata <train_metadata> "
                            "--proteome <train_proteome> --config configs/final.yaml "
                            "--output-dir runs/final"),
                "required": True,
            },
            "3_predict": {
                "command": ("python scripts/predict.py --metadata <test_metadata> "
                            "--run-dir runs/final --output prediction.csv"),
                "required": True,
            },
            "4_validate": {
                "command": ("python scripts/validate_submission.py --prediction prediction.csv "
                            "--run-dir runs/final"),
                "required": False,
            },
        },

        "config": {"file": "configs/final.yaml", "sha256": sha256(ROOT / "configs/final.yaml")},
        "code_entry_points": {p: sha256(ROOT / "scripts" / p) for p in
                              ["train.py", "predict.py", "build_embeddings.py",
                               "validate_submission.py", "run_hcce.py", "a2b_train_variants.py"]},
        "preprocessing": {
            "contract_file": "runs/final/artifacts/preprocess_contract.json",
            "protein_space": contract["protein_space"],
            "n_proteins": contract["n_proteins"],
            "protein_list_sha256": contract["protein_list_sha256"],
            "filter_rule": contract["filter_rule"],
            "filter_fit_scope": contract["filter_fit_scope"],
        },
        "run_dir": str(run),
        "ensemble_members_root": f"{run}/  (checkpoint paths below are relative to this)",
        # The submission spec (IV.7) draws checkpoints/ and artifacts/ at the package root.
        # This package keeps them under runs/final/ because that is exactly what
        # `train.py --output-dir runs/final` produces; maintaining a second copy is how
        # path drift starts. The mapping is recorded here so a reviewer checking the
        # recommended layout can resolve each entry mechanically.
        "directory_layout": {
            "note": "semantically equivalent to the recommended layout in IV.7; "
                    "no duplicate copies are kept",
            "recommended -> actual": {
                "checkpoints/": f"{run}/checkpoints/",
                "artifacts/": f"{run}/artifacts/",
                "configs/final.yaml": "configs/final.yaml",
                "scripts/{build_embeddings,train,predict,validate_submission}.py":
                    "scripts/ (same names)",
                "external_data/{source_manifest.json,entity_mapping.csv}":
                    "external_data/ (same names)",
                "tests/": "tests/",
                "REPRODUCIBILITY_MANIFEST.json": "REPRODUCIBILITY_MANIFEST.json",
                "LICENSES/": "LICENSES/",
            },
        },
        "ensemble_members": [
            {"seed": m["seed"], "checkpoint": m["checkpoint"], "sha256": m["sha256"],
             "train_seconds": m["train_seconds"]} for m in manifest["members"]],
        "ensemble_rule": manifest["ensemble"],
        "prediction": {
            "file": str(Path(args.prediction).name),
            "sha256": sha256(args.prediction),
            "prediction_scale": "log2",
            "n_rows": contract.get("n_rows", 4454),
            "n_protein_columns": contract["n_proteins"],
        },
        "data_boundary": {
            "training_labels": "split_final == 'train' only",
            "fit_rows": contract["fit_rows"],
            "held_out_val_rows": contract["held_out_val_rows"],
            "validation_use": "model selection only",
            "test_proteome_read": False,
            "test_truth_read": False,
            "assertion": "scripts/train.py aborts if any non-train row enters the fit index",
        },
        "external_data": {
            "used_by_final_model": False,
            "manifest": "external_data/source_manifest.json",
            "entity_mapping": "external_data/entity_mapping.csv",
            "prose": "external_data/RAW_SOURCES.md",
        },
        "environment": manifest["environment"],
        "dependencies": {"file": "requirements.txt",
                         "sha256": sha256(ROOT / "requirements.txt")},
        "known_limitations": [
            "The semi-final submission specification states the standard modelling space is 4,422 "
            "proteins, and the stated rule (train-only missing rate < 0.80 over the 5,920 train "
            "rows) reproduces exactly that count, so the caliber is confirmed. What was not "
            "distributed is the feature contract FILE itself, so column ORDER is taken as the "
            "original train_val matrix order with dropped columns removed (first 1-Oct, last ZWF1); "
            "the list and its SHA256 ship in runs/final/artifacts/protein_list.txt. An earlier "
            "preliminary-round slide mentioned 4,232; that figure is superseded by the semi-final "
            "specification and is not reproducible from the released data under any variant of the "
            "stated rule that we tested.",
            "GPU reductions are not bit-deterministic across different hardware; seeds, cuDNN "
            "deterministic mode and benchmark=False are fixed, so a rerun on the same hardware "
            "reproduces the artifact, while different hardware may differ in the last digits.",
            "The final model contributes no compound-specific signal on unseen compounds; this "
            "is measured, not assumed (see 方案说明文档 §四 and §七).",
        ],
        "contact": "TorchDragon — 负责人 李晓蒙 (方法与实现) / 徐逸飞 (生物建模)",  # 公开仓库不含手机号
    }
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("work_id", "final_model", "config", "prediction",
                                              "data_boundary")}, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
