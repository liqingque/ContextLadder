# External data provenance

Retrieval date: 2026-08-13 (Asia/Shanghai).  Files below are kept under
`external_data/raw/`; the generated contract records byte counts and SHA256
hashes in `external_data/processed/external_feature_contract.json`.

## Yeast 1011 public matrices

- Scientific source: [Peter et al., Nature 2018](https://www.nature.com/articles/s41586-018-0030-5).
- Download index: `http://1002genomes.u-strasbg.fr/files/`.
- Files: `snp_distance.tab.gz`, `orf_distance.tab.gz`,
  `genes_presence_absence.tab.gz`, `genes_copy_number.tab.gz`,
  `genes_frameshift.tab.gz`, and `1011GWAS_matrix.{bed,bim,fam}`.
- The matrices contain 1,011 public isolates.  The competition crosswalk uses
  the official names SX3/BJ6/JCM_2985-4B/UCD_09-448/FIMA_3, which are
  represented in the matrix as BAH/BAI/CEK/CGD/CRD respectively.

## S288C proxy

- Sequence/annotation source: [NCBI RefSeq GCF_000146045.2 R64](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/).
- Files: `GCF_000146045.2_R64_feature_table.txt.gz` and
  `GCF_000146045.2_R64_genomic.gff.gz`.
- DHY210 is explicitly marked as a proxy (`proxy_flag=1`); it is not silently
  treated as an exact row in the 1011 matrices.

## Compound structures

- Existing audit source: PubChem PUG REST, with the mapping and provenance
  retained in `outputs/p3_chemical/compound_mapping_audit.csv` and copied to
  `external_data/processed/compound_mapping_features.csv`.
- PubChem service documentation: [PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial).
- RDKit Morgan fingerprints are generated locally from canonical SMILES.  No
  ambiguous or unmapped label is guessed into a structure.

## Reproducibility and licensing notes

- The exact downloaded bytes, URLs, and SHA256 values are the authoritative
  provenance record; rerunning `scripts/build_external_features.py` rebuilds
  the processed contract.
- Third-party terms remain those of the source providers.  The 1011 paper is
  published under the venue's open-access terms; PubChem and NCBI/SGD data
  retain their respective public-data and attribution requirements.  This
  repository does not redistribute a proprietary API response beyond the
  cached audit needed to reproduce the experiment.

## Compound mechanism-text embeddings (DCB-40 primary content channel, added 2026-08-16)

- **Model**: `NeuML/pubmedbert-base-embeddings`, revision pinned to
  `b79526d6ef3645e0df4530322e266f24c829f5ef`, obtained from the Hugging Face Hub.
  Open-weight, PubMedBERT-derived sentence encoder (768-dim).
- **Why this model, decided a priori**: the embedded text is pharmacological
  mechanism description, so a biomedical-domain encoder is the domain-matched
  choice.  The model was **preregistered before any rho was computed** — it was
  not selected by comparing downstream results across candidate encoders.
- **Input text**: the `mechanism` field of `external_data/processed/ssps_priors/*.jsonl`
  (LLM-authored mechanism summaries cached 2026-08-15).  No proteome truth of any
  split is read by the embedding step.
- **Pooling**: replicates the model's own `1_Pooling/config.json` exactly — mean over
  tokens weighted by the attention mask, no CLS pooling, **no L2 normalisation**
  (the model ships no Normalize module).  `max_seq_length` 512, fp32, CPU,
  one sequence at a time so batch padding cannot perturb the result.
- **Frozen artifact**: `external_data/processed/compound_text_embeddings/embeddings.parquet`
  (54 compounds x 768), sha256 `c373b32338226836f579a1a704e0ffc54b3bbb6d0f444e6cf5015de276816e8e`.
  Verified bit-identical across two independent runs.  The DCB pipeline reads only
  this parquet, so the reproduction review needs neither network access nor the
  model weights.
- **Builder**: `scripts/build_compound_text_embeddings.py`; provenance, per-compound
  text sha256, and the environment snapshot are in the sibling `manifest.json`.
- **Environment**: built with the existing `tl` environment (transformers 4.40.2,
  torch 2.1.0).  `sentence-transformers` was deliberately **not** installed — a
  dry run showed it would upgrade transformers to 4.46.3 and tokenizers to 0.20.3
  inside the frozen submission environment.
- **Coverage / abstention**: substantive mechanism text exists for 37/40 train
  compounds (the 3 without are the DMSO / Water / Quality-Control vehicles),
  5/6 val-unseen, 10/11 `test_chem_only` and 15/17 `test_both`.  `Abietic acid`
  and `Pentamidine isethionate` carry the literal text `unknown` and are handled
  by the preregistered abstention rule (a = 0, exact fallback to the base
  prediction), not by guessing a mechanism.
