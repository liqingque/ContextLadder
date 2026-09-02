#!/usr/bin/env python
"""Freeze the DCB-40 primary content channel: text embeddings of the SSPS
mechanism descriptions.

The channel is preregistered as a single model at a pinned revision.  This
script runs ONCE, writes a parquet plus a manifest, and the DCB pipeline
afterwards reads only the parquet -- so the reproduction review never needs
network access or the model weights.

Pooling replicates the model's own sentence-transformers config exactly
(1_Pooling/config.json: mean over tokens, no CLS, no normalisation module).
No L2 normalisation is applied here; the nested LOCO contract z-scores the q
PLS directions on each fold's fit set, which is the only scaling in the chain.

Reads:  external_data/processed/ssps_priors/*.jsonl  (mechanism text only)
Writes: external_data/processed/compound_text_embeddings/embeddings.parquet
        external_data/processed/compound_text_embeddings/manifest.json

No proteome truth of any split is read.
"""

import glob
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "NeuML/pubmedbert-base-embeddings"
MODEL_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
MAX_SEQ_LENGTH = 512          # sentence_bert_config.json
POOLING = "mean"              # 1_Pooling/config.json
L2_NORMALISE = False          # the model ships no Normalize module
OUT_DIR = ROOT / "external_data/processed/compound_text_embeddings"
PRIOR_GLOB = "external_data/processed/ssps_priors/*.jsonl"
CONTROL_TEXTS = {"control/vehicle"}   # DMSO / Water etc. carry no mechanism


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_mechanism_texts():
    """compound -> mechanism string, first occurrence wins (files are rounds)."""
    texts = {}
    for path in sorted(glob.glob(str(ROOT / PRIOR_GLOB))):
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                name = str(rec["compound"])
                if name in texts:
                    continue
                mech = (rec.get("mechanism") or "").strip()
                texts[name] = mech
    return texts


def mean_pool(last_hidden, attention_mask):
    """Attention-mask-weighted mean over tokens (sentence-transformers Pooling)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = load_mechanism_texts()
    usable = {c: t for c, t in texts.items() if t and t not in CONTROL_TEXTS}
    names = sorted(usable)
    print(f"compounds in cache: {len(texts)}  with mechanism text: {len(names)}")

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                      torch_dtype=torch.float32)
    model.eval()

    vecs = []
    with torch.no_grad():
        for name in names:                       # one at a time: no batch-padding drift
            enc = tok(usable[name], padding=False, truncation=True,
                      max_length=MAX_SEQ_LENGTH, return_tensors="pt")
            out = model(**enc).last_hidden_state
            vecs.append(mean_pool(out, enc["attention_mask"])[0].numpy().astype(np.float64))
    emb = np.vstack(vecs)
    if L2_NORMALISE:
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    print(f"embeddings: {emb.shape}  finite={np.isfinite(emb).all()}")

    frame = pd.DataFrame(emb, columns=[f"d{i:03d}" for i in range(emb.shape[1])])
    frame.insert(0, "compound", names)
    out_parquet = OUT_DIR / "embeddings.parquet"
    frame.to_parquet(out_parquet, index=False)

    manifest = {
        "purpose": "DCB-40 preregistered primary content channel phi_TXT",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pooling": POOLING,
        "l2_normalised": L2_NORMALISE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "n_compounds": len(names),
        "embedding_dim": int(emb.shape[1]),
        "source_texts": PRIOR_GLOB,
        "excluded_as_control": sorted(c for c, t in texts.items()
                                      if (not t) or t in CONTROL_TEXTS),
        "per_compound_text_sha256": {c: sha256_bytes(usable[c].encode()) for c in names},
        "output_sha256": sha256_bytes(out_parquet.read_bytes()),
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "no_proteome_truth_read": True,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {out_parquet}")
    print(f"output sha256 {manifest['output_sha256']}")
    print(f"excluded as control: {manifest['excluded_as_control']}")


if __name__ == "__main__":
    main()
