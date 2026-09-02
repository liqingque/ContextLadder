"""Train-only competition bundle and fold-local feature preparation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.evaluation.control_matching import match_controls
from src.mosaic.contracts import ContractViolation, assert_safe_input_path


CATEGORICAL_COLUMNS = (
    "Medium",
    "Temperature",
    "data_source",
    "instrument",
    "Yeast_cell_plate",
)


def require_preflight_pass(path: Path) -> Dict:
    safe = assert_safe_input_path(path, purpose="audit")
    with safe.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    required_false = (
        "test_proteome_read",
        "test_truth_read",
        "official_validation_used_for_selection",
    )
    if audit.get("integrity_status") != "pass" or audit.get("gate_decision") != "PASS":
        raise ContractViolation("MOSAIC P0 preflight has not passed: %s" % safe)
    if any(bool(audit.get(key, True)) for key in required_false):
        raise ContractViolation("P0 audit contains a leakage flag: %s" % safe)
    return audit


def build_control_pairs(
    metadata: pd.DataFrame,
    field_mapping: Mapping[str, str],
    allowed_sample_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Rematch controls inside the supplied legal sample universe.

    The output stores sample IDs instead of positional indices so later
    consumers cannot accidentally apply indices from another fold/table.
    """
    sample_col = field_mapping["sample_id"]
    work = metadata.copy()
    work[sample_col] = work[sample_col].astype(str)
    if allowed_sample_ids is not None:
        allowed = set(str(value) for value in allowed_sample_ids)
        work = work.loc[work[sample_col].isin(allowed)].copy()
        if set(work[sample_col]) != allowed:
            raise ContractViolation("Control-pair universe contains unknown sample IDs")
    work = work.reset_index(drop=True)
    legal_mask = np.ones(len(work), dtype=bool)
    matched = match_controls(
        work,
        legal_mask,
        dict(field_mapping),
        control_pool_mask=legal_mask,
    )
    rows = []
    for row in matched.itertuples(index=False):
        record = {
            "treatment_sample_ID": str(row.sample_ID),
            "matched": bool(row.matched),
            "control_type": str(getattr(row, "control_type", ""))
            if bool(row.matched)
            else "",
            "n_controls": int(row.n_controls),
            "control_sample_IDs": "",
            "match_key": str(getattr(row, "match_key", "")) if bool(row.matched) else "",
        }
        if bool(row.matched):
            positions = [int(value) for value in str(row.control_rows).split(",") if value]
            control_ids = work.iloc[positions][sample_col].astype(str).tolist()
            record["control_sample_IDs"] = ",".join(control_ids)
            if record["treatment_sample_ID"] in control_ids:
                raise ContractViolation("A treatment was matched to itself as control")
        rows.append(record)
    columns = [
        "treatment_sample_ID",
        "matched",
        "control_type",
        "n_controls",
        "control_sample_IDs",
        "match_key",
    ]
    return pd.DataFrame(rows, columns=columns)


@dataclass
class FoldVocabulary:
    categories: Dict[str, Dict[str, int]]
    time_scale: float

    @property
    def cardinalities(self) -> Dict[str, int]:
        return {column: len(values) + 1 for column, values in self.categories.items()}


def fit_fold_vocabulary(
    sample_index: pd.DataFrame,
    fit_sample_ids: Sequence[str],
    sample_column: str = "sample_ID",
) -> FoldVocabulary:
    by_id = sample_index.copy()
    by_id[sample_column] = by_id[sample_column].astype(str)
    by_id = by_id.set_index(sample_column, drop=False)
    fit = by_id.loc[[str(value) for value in fit_sample_ids]]
    categories: Dict[str, Dict[str, int]] = {}
    for column in CATEGORICAL_COLUMNS:
        values = sorted(fit[column].fillna("<NA>").astype(str).unique().tolist())
        categories[column] = {value: index + 1 for index, value in enumerate(values)}
    times = pd.to_numeric(fit["pert_time"], errors="coerce").to_numpy(dtype=np.float32)
    finite = times[np.isfinite(times)]
    time_scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    return FoldVocabulary(categories=categories, time_scale=max(time_scale, 1.0))


def _feature_columns(frame: pd.DataFrame, prefix: str) -> List[str]:
    return sorted([column for column in frame.columns if column.startswith(prefix)])


def prepare_model_inputs(
    sample_index: pd.DataFrame,
    sample_ids: Sequence[str],
    vocabulary: FoldVocabulary,
    strain_features: pd.DataFrame,
    compound_features: pd.DataFrame,
    external_mode: str = "gated",
) -> Dict[str, np.ndarray]:
    """Encode raw rows using a vocabulary fitted on the outer fit set only."""
    if external_mode not in ("gated", "zero", "open_entity"):
        raise ValueError("Unknown external mode: %s" % external_mode)
    sample_column = "sample_ID"
    index = sample_index.copy()
    index[sample_column] = index[sample_column].astype(str)
    index = index.set_index(sample_column, drop=False)
    rows = index.loc[[str(value) for value in sample_ids]]
    out: Dict[str, np.ndarray] = {}
    for column, mapping in vocabulary.categories.items():
        out[column] = (
            rows[column].fillna("<NA>").astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        )
    time = pd.to_numeric(rows["pert_time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    scaled_time = time / np.float32(vocabulary.time_scale)
    out["time_features"] = np.column_stack((scaled_time, scaled_time ** 2)).astype(np.float32)

    strain_cols = _feature_columns(strain_features, "strain_feature_")
    strain_by_id = strain_features.set_index("Strains", drop=False)
    strains = rows["Strains"].astype(str).tolist()
    strain_values = strain_by_id.reindex(strains)[strain_cols].fillna(0.0).to_numpy(dtype=np.float32)
    strain_mask = (
        strain_by_id.reindex(strains)["feature_available"].fillna(False).to_numpy(dtype=np.float32)
    )
    strain_allowed = (
        strain_by_id.reindex(strains)["training_allowed"].fillna(False).to_numpy(dtype=np.float32)
    )

    compound_cols = _feature_columns(compound_features, "compound_feature_")
    compound_by_id = compound_features.set_index("perturbation_no_concentration", drop=False)
    compounds = rows["perturbation_no_concentration"].astype(str).tolist()
    compound_values = (
        compound_by_id.reindex(compounds)[compound_cols].fillna(0.0).to_numpy(dtype=np.float32)
    )
    compound_mask = (
        compound_by_id.reindex(compounds)["feature_available"].fillna(False).to_numpy(dtype=np.float32)
    )
    compound_allowed = (
        compound_by_id.reindex(compounds)["training_allowed"].fillna(False).to_numpy(dtype=np.float32)
    )

    # Availability is a read/coverage property; ``training_allowed`` is the
    # branch gate.  Open-entity mode may expose approved public features, but
    # it must never bypass a failed data-feasibility/promotion gate.
    strain_mask *= strain_allowed
    compound_mask *= compound_allowed
    if external_mode == "zero":
        strain_mask[:] = 0.0
        compound_mask[:] = 0.0
    out["strain_external"] = strain_values
    out["strain_external_mask"] = strain_mask
    out["compound_external"] = compound_values
    out["compound_external_mask"] = compound_mask
    return out


def control_pair_indices(
    pairs: pd.DataFrame,
    sample_ids: Sequence[str],
) -> List[Tuple[int, List[int]]]:
    position = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
    out: List[Tuple[int, List[int]]] = []
    for row in pairs.loc[pairs.matched.astype(bool)].itertuples(index=False):
        treat = position.get(str(row.treatment_sample_ID))
        controls = [
            position.get(value)
            for value in str(row.control_sample_IDs).split(",")
            if value and position.get(value) is not None
        ]
        if treat is not None and controls and len(controls) == int(row.n_controls):
            out.append((treat, [int(value) for value in controls]))
    return out


def load_bundle(root: Path) -> Dict[str, pd.DataFrame]:
    root = assert_safe_input_path(root, purpose="training")
    contract_path = root / "data_contract.json"
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("integrity_status") != "pass":
        raise ContractViolation("Dataset bundle is not consumer-ready: %s" % root)
    if contract.get("test_proteome_read") or contract.get("official_validation_used_for_selection"):
        raise ContractViolation("Dataset bundle has a leakage flag")
    return {
        "contract": contract,
        "targets": pd.read_parquet(root / "targets_log2.parquet"),
        "mask": pd.read_parquet(root / "observed_mask.parquet"),
        "sample_index": pd.read_parquet(root / "sample_index.parquet"),
        "strain_features": pd.read_parquet(root / "strain_features.parquet"),
        "compound_features": pd.read_parquet(root / "compound_features.parquet"),
        "protein_features": pd.read_parquet(root / "protein_features.parquet"),
        "control_pairs": pd.read_parquet(root / "control_pairs.parquet"),
        "fold_roles": pd.read_parquet(root / "fold_roles.parquet"),
    }
