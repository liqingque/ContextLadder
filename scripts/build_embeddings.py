#!/usr/bin/env python
"""External feature / embedding construction — NOT REQUIRED for this submission.

The final ContextLadder model consumes only official sample metadata fields
(compound, strain, medium, source, instrument, plate, temperature, time). It
uses no strain genome, no compound structure, no protein sequence, no pathway
or PPI resource. There is therefore nothing to download or build before
training, and running this script is optional.

Running it anyway writes a manifest that states this explicitly, so that the
"external data" deliverable has a concrete, checkable artifact rather than an
absence.

External public resources DO appear elsewhere in the project: they were used in
the falsification experiments reported in 方案说明文档 §七 (mechanism-text
embeddings, structured mechanism priors, Tanimoto similarity, a random control).
All of them produced negative results and none enters any prediction path. Their
provenance is in external_data/RAW_SOURCES.md and external_data/source_manifest.json.

    python scripts/build_embeddings.py --output artifacts/embeddings
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", help="accepted for interface compatibility; not read")
    ap.add_argument("--output", default=str(ROOT / "artifacts" / "embeddings"))
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "external_features_used_by_final_model": False,
        "action_required": "none — this step can be skipped entirely",
        "final_model_inputs": [
            "compound (categorical)", "strain (categorical)", "medium (categorical)",
            "data_source (categorical)", "instrument (categorical)", "plate (categorical)",
            "temperature (numeric)", "perturbation time (numeric, + log-time + 5 RBF bases)",
        ],
        "external_resources_in_prediction_path": [],
        "external_resources_used_in_falsification_experiments_only": {
            "see": ["external_data/RAW_SOURCES.md", "external_data/source_manifest.json"],
            "note": "reported in 方案说明文档 §七; all negative; none enters inference",
        },
        "test_derived_inputs": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = out / "embedding_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("No external features are required by the final model; nothing was built.")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
