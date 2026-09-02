#!/usr/bin/env python
"""Emit external_data/source_manifest.json — machine-readable provenance.

Companion to external_data/RAW_SOURCES.md (prose). One entry per external
resource: source, version/revision, URL, licence, retrieval date, what it was
used for, and the SHA256 of the derived artifact that ships in this package.

Every entry has used_by_final_model = false. The final model reads official
sample metadata only; these resources exist because the falsification
experiments in 方案说明文档 §七 needed them.

    python scripts/build_source_manifest.py --output external_data/source_manifest.json
"""

import argparse
import hashlib
import json
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


SOURCES = [
    {
        "id": "yeast_1011_genomes",
        "name": "1011 Yeast Genomes public matrices",
        "citation": "Peter et al., Nature 2018",
        "url": "https://www.nature.com/articles/s41586-018-0030-5",
        "download_index": "http://1002genomes.u-strasbg.fr/files/",
        "version": "1011 public isolates release",
        "licence": "venue open-access terms; source-provider attribution retained",
        "retrieved": "2026-08-13",
        "entities_mapped": "5 competition strains matched exactly (BAH=SX3, BAI=BJ6, CEK=JCM_2985-4B, CGD=UCD_09-448, CRD=FIMA_3)",
        "used_for": "G1 / G1R genome-feature falsification (negative result)",
        "used_by_final_model": False,
        "shipped_artifacts": ["external_data/processed/strain_crosswalk_features.csv"],
    },
    {
        "id": "ncbi_r64_s288c",
        "name": "NCBI RefSeq S288C R64 assembly and annotation",
        "url": "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/",
        "version": "GCF_000146045.2 (R64)",
        "licence": "NCBI public domain / attribution",
        "retrieved": "2026-08-13",
        "entities_mapped": "protein gene names; DHY210 explicitly flagged proxy_flag=1, never treated as an exact 1011 row",
        "used_for": "protein sequence lookup for the ESM2 output-head experiment (negative result)",
        "used_by_final_model": False,
        "shipped_artifacts": [],
    },
    {
        "id": "pubchem",
        "name": "PubChem compound structures",
        "url": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial",
        "version": "PUG REST, retrieved 2026-08-10/11",
        "licence": "PubChem public data terms",
        "retrieved": "2026-08-11",
        "entities_mapped": "57 compound labels: 49 MAPPED, 3 AMBIGUOUS, 4 UNMAPPED, 1 QC_EXCLUDED; no ambiguous label is guessed into a structure",
        "used_for": "P3 / C1 / C1R chemical-feature falsification (negative result)",
        "used_by_final_model": False,
        "shipped_artifacts": ["external_data/processed/compound_mapping_features.csv"],
    },
    {
        "id": "pubmedbert_embeddings",
        "name": "NeuML/pubmedbert-base-embeddings",
        "url": "https://huggingface.co/NeuML/pubmedbert-base-embeddings",
        "version": "revision b79526d6ef3645e0df4530322e266f24c829f5ef",
        "licence": "open weights, per model card",
        "retrieved": "2026-08-16",
        "entities_mapped": "54 compounds x 768 dims; mean pooling over attention mask, no L2 normalisation, max_seq_length 512, fp32, CPU, one sequence at a time",
        "used_for": "DCB-40 primary content channel phi_TXT (Gate-0 not passed; negative result)",
        "used_by_final_model": False,
        "preregistered": "model id and revision locked before any rho was computed; not chosen by comparing downstream results",
        "abstention_rule": "compounds without substantive mechanism text get amplitude 0 (exact fallback); no mechanism is guessed",
        "shipped_artifacts": ["external_data/processed/compound_text_embeddings/embeddings.parquet",
                              "external_data/processed/compound_text_embeddings/manifest.json"],
    },
    {
        "id": "ssps_priors",
        "name": "Structured mechanism priors (LLM-authored, cached)",
        "version": "cached 2026-08-15",
        "licence": "generated text, released with this package",
        "retrieved": "2026-08-15",
        "entities_mapped": "per-compound mechanism summaries; no proteome truth of any split is read by the generation step",
        "used_for": "M3 / SSPS Gate-0 and the DCB-40 C2 ablation (both negative)",
        "used_by_final_model": False,
        "shipped_artifacts": [f"external_data/processed/ssps_priors/priors_R{i}.jsonl" for i in range(1, 6)],
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "external_data/source_manifest.json"))
    args = ap.parse_args()

    for s in SOURCES:
        s["shipped_artifact_sha256"] = {a: sha256(ROOT / a) for a in s["shipped_artifacts"]}

    payload = {
        "schema": "goai-vc-external-source-manifest/1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "final_model_external_inputs": [],
        "final_model_note": ("The final model consumes official sample metadata only "
                             "(compound, strain, medium, source, instrument, plate, temperature, time). "
                             "No entry below reaches inference."),
        "prose_provenance": "external_data/RAW_SOURCES.md",
        "entity_mapping": "external_data/entity_mapping.csv",
        "test_derived_content": ("none — no test proteome, matched-control truth or scoring "
                                 "result was used to generate, filter, normalise or calibrate "
                                 "any external feature"),
        "sources": SOURCES,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = [(s["id"], a) for s in SOURCES for a, h in s["shipped_artifact_sha256"].items() if h is None]
    for sid, a in missing:
        print(f"WARNING: {sid} declares {a} but it is not present")
    print(f"sources: {len(SOURCES)}; shipped artifacts hashed: "
          f"{sum(1 for s in SOURCES for h in s['shipped_artifact_sha256'].values() if h)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
