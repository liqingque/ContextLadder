#!/usr/bin/env python
"""Assemble external_data/entity_mapping.csv from the audit artifacts.

One row per competition entity that was ever mapped to a public database,
carrying the mapping status, the external identifier, the evidence, and --
importantly -- whether it reaches the final model. For this submission the
last column is False everywhere: the final model consumes official metadata
fields only. The mappings exist because the falsification experiments in
方案说明文档 §七 needed them, and they are shipped so those negative results
stay checkable.

Unmatched, ambiguous and proxy entities are kept as rows rather than dropped,
as the submission specification requires.

    python scripts/build_entity_mapping.py --output external_data/entity_mapping.csv
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["entity_type", "competition_id", "external_db", "external_id", "secondary_id",
           "mapping_status", "mapping_confidence", "proxy_flag", "evidence",
           "used_by_final_model", "used_in_experiments"]


def compounds():
    p = ROOT / "external_data/processed/compound_mapping_features.csv"
    d = pd.read_csv(p)
    rows = []
    for _, r in d.iterrows():
        cid = r.get("pubchem_cid")
        rows.append({
            "entity_type": "compound",
            "competition_id": r["query_label"],
            "external_db": "PubChem",
            "external_id": "" if pd.isna(cid) else f"CID:{int(cid)}",
            "secondary_id": "" if pd.isna(r.get("canonical_smiles")) else str(r["canonical_smiles"]),
            "mapping_status": r.get("mapping_status", ""),
            "mapping_confidence": r.get("mapping_confidence", ""),
            "proxy_flag": 0,
            "evidence": "PubChem PUG REST; see external_data/RAW_SOURCES.md",
            "used_by_final_model": False,
            "used_in_experiments": "P3 / C1 / C1R chemical-feature falsification (negative)",
        })
    return rows


def strains():
    cross = pd.read_csv(ROOT / "external_data/processed/strain_crosswalk_features.csv")
    audit = pd.read_csv(ROOT / "outputs/p4_strain/accession_mapping_audit.csv").set_index("strain_id")
    rows = []
    for _, r in cross.iterrows():
        cid = r["competition_id"]
        a = audit.loc[cid] if cid in audit.index else None
        acc = "" if a is None or pd.isna(a.get("assembly_accession")) else str(a["assembly_accession"])
        bios = "" if a is None or pd.isna(a.get("biosample_accession")) else str(a["biosample_accession"])
        rows.append({
            "entity_type": "strain",
            "competition_id": cid,
            "external_db": "1011 Yeast Genomes (Peter et al., Nature 2018)",
            "external_id": str(r["standard_name"]),
            "secondary_id": ";".join([x for x in (acc, bios) if x]),
            "mapping_status": str(r["mapping_status"]),
            "mapping_confidence": "" if a is None else str(a.get("mapping_confidence", "")),
            "proxy_flag": int(r["proxy_flag"]),
            "evidence": ("" if a is None else str(a.get("evidence", ""))[:180])
                        + " | see reports/ACCESSION_MAPPING_AUDIT.md",
            "used_by_final_model": False,
            "used_in_experiments": "G1 / G1R genome-feature falsification (negative)",
        })
    return rows


def proteins():
    embedded = (ROOT / "external_data/processed/esm2_embedded_protein_names.txt").read_text(
        encoding="utf-8").split("\n")
    embedded = [x for x in embedded if x.strip()]
    missing = (ROOT / "external_data/processed/esm2_missing_proteins.txt").read_text(
        encoding="utf-8").split("\n")
    missing = [x for x in missing if x.strip()]
    base = {
        "entity_type": "protein", "external_db": "NCBI RefSeq GCF_000146045.2 (R64) via SGD names",
        "proxy_flag": 0, "used_by_final_model": False,
        "used_in_experiments": "D / ESM2 output-head falsification (negative)",
    }
    rows = [dict(base, competition_id="<%d proteins, see esm2_embedded_protein_names.txt>" % len(embedded),
                 external_id="mapped to R64 sequence", secondary_id="",
                 mapping_status="MAPPED", mapping_confidence="exact gene-name match",
                 evidence="aggregate row; per-protein list shipped alongside")]
    for name in missing:
        rows.append(dict(base, competition_id=name, external_id="", secondary_id="",
                         mapping_status="UNMAPPED", mapping_confidence="none",
                         evidence="column name is not a yeast gene (spreadsheet date coercion); expected miss"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "external_data/entity_mapping.csv"))
    args = ap.parse_args()
    rows = compounds() + strains() + proteins()
    d = pd.DataFrame(rows, columns=COLUMNS)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(d.groupby(["entity_type", "mapping_status"]).size().to_string())
    print(f"\nrows: {len(d)}  used_by_final_model=True: {int(d['used_by_final_model'].sum())}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
