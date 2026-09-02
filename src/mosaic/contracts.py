"""Hard data, protein-order, and fold contracts for MOSAIC-VC.

The functions in this module are deliberately independent of model code.  All
future MOSAIC readers must pass paths through :func:`assert_safe_input_path`
before opening them, and all fitting metadata through
:func:`selection_metadata` before estimating any statistic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.data.io import align_metadata_proteome, load_metadata, load_proteome


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_FOLD_ROLES = frozenset(("train", "holdout", "excluded"))
FORBIDDEN_TEST_PROTEOME_BASENAMES = frozenset(("wayb_wayc_proteome_raw_test.csv",))
OFFICIAL_VALIDATION_SPLITS = frozenset(
    ("val_chem_only", "val_strain_only", "val_both", "val_time")
)


class ContractViolation(RuntimeError):
    """Raised when an input would violate a leakage or ordering contract."""


def resolve_project_path(path: Any, root: Path = ROOT) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def _normalise_name(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.name.lower()).strip("_")


def is_forbidden_test_target(path: Any, root: Path = ROOT) -> bool:
    """Return True for test proteome/truth targets, including symlink targets."""
    lexical = resolve_project_path(path, root=root)
    candidates = (Path(path), lexical)
    for candidate in candidates:
        basename = candidate.name.lower()
        normalised = _normalise_name(candidate)
        if basename in FORBIDDEN_TEST_PROTEOME_BASENAMES:
            return True
        tokens = set(normalised.split("_"))
        if "test" in tokens and ("proteome" in tokens or "truth" in tokens):
            return True
        if "test" in tokens and "response" in tokens:
            return True
    return False


def assert_safe_input_path(path: Any, purpose: str = "selection", root: Path = ROOT) -> Path:
    """Reject forbidden response targets *before* a caller opens the file.

    Test metadata may only be used for a final inference-only purpose.  It is
    not used by P0 and can never be admitted to a selection loader.
    """
    resolved = resolve_project_path(path, root=root)
    if is_forbidden_test_target(path, root=root):
        raise ContractViolation("Forbidden test proteome/test truth access: %s" % resolved)
    name_tokens = set(_normalise_name(resolved).split("_"))
    if "metadata" in name_tokens and "test" in name_tokens and purpose != "inference_only":
        raise ContractViolation("Test metadata is forbidden for purpose=%s: %s" % (purpose, resolved))
    return resolved


def guarded_read_csv(path: Any, purpose: str = "selection", **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(assert_safe_input_path(path, purpose=purpose), **kwargs)


def guarded_read_parquet(path: Any, purpose: str = "selection", **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(assert_safe_input_path(path, purpose=purpose), **kwargs)


def load_yaml(path: Any, root: Path = ROOT) -> Dict[str, Any]:
    safe = assert_safe_input_path(path, purpose="configuration", root=root)
    with safe.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ContractViolation("YAML contract must contain a mapping: %s" % safe)
    return value


def sha256_file(path: Any, root: Path = ROOT) -> str:
    safe = assert_safe_input_path(path, purpose="audit", root=root)
    digest = hashlib.sha256()
    with safe.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_yaml_hash(path: Any, length: int = 8, root: Path = ROOT) -> str:
    payload = load_yaml(path, root=root)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def selection_metadata(
    metadata: pd.DataFrame,
    split_column: str = "split_final",
    allowed_splits: Sequence[str] = ("train",),
    expected_samples: Optional[int] = 5920,
) -> pd.DataFrame:
    """Return the only rows permitted to estimate preprocessing/model state."""
    if split_column not in metadata.columns:
        raise ContractViolation("Missing split column: %s" % split_column)
    allowed = frozenset(str(value) for value in allowed_splits)
    if allowed != frozenset(("train",)):
        raise ContractViolation("MOSAIC selection may use split_final=train only")
    selected = metadata.loc[metadata[split_column].astype(str).isin(allowed)].copy()
    observed = frozenset(selected[split_column].dropna().astype(str).unique())
    if observed - allowed:
        raise ContractViolation("Non-train rows entered selection metadata: %s" % sorted(observed - allowed))
    if expected_samples is not None and len(selected) != int(expected_samples):
        raise ContractViolation(
            "Expected %d selection samples, found %d" % (int(expected_samples), len(selected))
        )
    return selected.reset_index(drop=True)


def read_submission_protein_contract(path: Any, root: Path = ROOT) -> Tuple[str, List[str]]:
    safe = assert_safe_input_path(path, purpose="contract", root=root)
    with safe.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    sample_id = str(contract.get("sample_id", ""))
    proteins = [str(value) for value in contract.get("protein_columns", [])]
    if not sample_id or int(contract.get("protein_count", -1)) != len(proteins):
        raise ContractViolation("Malformed protein submission contract: %s" % safe)
    if len(set(proteins)) != len(proteins):
        raise ContractViolation("Duplicate proteins in submission contract")
    return sample_id, proteins


def parquet_column_order(path: Any, root: Path = ROOT) -> List[str]:
    safe = assert_safe_input_path(path, purpose="contract", root=root)
    return list(pq.read_schema(safe).names)


def recompute_retained_proteins(
    train_proteome: pd.DataFrame,
    protein_columns: Sequence[str],
    threshold: float = 0.80,
) -> Tuple[List[str], np.ndarray]:
    """Apply the established train-only, raw-finite missingness rule."""
    missing_rate = train_proteome[list(protein_columns)].isna().mean(axis=0).to_numpy(dtype=float)
    keep = np.isfinite(missing_rate) & (missing_rate < float(threshold))
    return [protein for protein, flag in zip(protein_columns, keep) if bool(flag)], missing_rate


def validate_protein_contract(
    metadata_path: Any,
    proteome_path: Any,
    submission_contract_path: Any,
    retained_schema_reference: Any,
    split_column: str = "split_final",
    threshold: float = 0.80,
    expected_fit_samples: int = 5920,
    root: Path = ROOT,
) -> Dict[str, Any]:
    metadata_safe = assert_safe_input_path(metadata_path, purpose="preflight", root=root)
    proteome_safe = assert_safe_input_path(proteome_path, purpose="preflight", root=root)
    metadata = load_metadata(metadata_safe)
    proteome, proteome_sample_id, _ = load_proteome(proteome_safe)
    metadata, proteome, aligned_sample_id, raw_proteins = align_metadata_proteome(
        metadata, proteome, proteome_sample_id=proteome_sample_id
    )
    selected = selection_metadata(
        metadata,
        split_column=split_column,
        allowed_splits=("train",),
        expected_samples=expected_fit_samples,
    )
    selection_ids = selected[aligned_sample_id].astype(str).tolist()
    train_by_id = proteome.set_index(aligned_sample_id, drop=False).loc[selection_ids]

    contract_sample_id, submission_proteins = read_submission_protein_contract(
        submission_contract_path, root=root
    )
    if contract_sample_id != aligned_sample_id:
        raise ContractViolation(
            "Sample ID contract mismatch: %s != %s" % (contract_sample_id, aligned_sample_id)
        )
    submission_order_match = list(raw_proteins) == submission_proteins
    retained, missing_rate = recompute_retained_proteins(
        train_by_id, raw_proteins, threshold=threshold
    )
    reference_columns = parquet_column_order(retained_schema_reference, root=root)
    if not reference_columns or reference_columns[0] != aligned_sample_id:
        raise ContractViolation("Retained schema reference lacks leading sample_ID")
    retained_order_match = retained == reference_columns[1:]
    if not submission_order_match:
        raise ContractViolation("Raw proteome columns do not match the 5,243-protein submission order")
    if not retained_order_match:
        raise ContractViolation("Recomputed retained proteins do not match the 4,422-protein schema")
    return {
        "metadata_rows": int(len(metadata)),
        "proteome_rows": int(len(proteome)),
        "metadata_proteome_bijection": True,
        "fit_sample_count": int(len(selected)),
        "official_validation_sample_count": int(len(metadata) - len(selected)),
        "selection_splits": sorted(selected[split_column].astype(str).unique().tolist()),
        "sample_id_column": aligned_sample_id,
        "submission_protein_count": int(len(submission_proteins)),
        "retained_protein_count": int(len(retained)),
        "filtered_protein_count": int(len(submission_proteins) - len(retained)),
        "submission_order_match": bool(submission_order_match),
        "retained_order_match": bool(retained_order_match),
        "missing_rate_threshold": float(threshold),
        "threshold_comparison": "strictly_less_than",
        "max_retained_missing_rate": float(missing_rate[missing_rate < threshold].max()),
        "min_filtered_missing_rate": float(missing_rate[missing_rate >= threshold].min()),
        "retained_order_sha256": hashlib.sha256("\n".join(retained).encode("utf-8")).hexdigest(),
        "submission_order_sha256": hashlib.sha256(
            "\n".join(submission_proteins).encode("utf-8")
        ).hexdigest(),
    }


def _entity_values(
    metadata_by_id: pd.DataFrame,
    sample_ids: Sequence[str],
    column: str,
) -> set:
    if not sample_ids:
        return set()
    return set(metadata_by_id.loc[list(sample_ids), column].fillna("<NA>").astype(str))


def validate_fold_contract(
    fold_table: pd.DataFrame,
    train_metadata: pd.DataFrame,
    sample_column: str = "sample_ID",
    strain_column: str = "Strains",
    compound_column: str = "perturbation_no_concentration",
    time_column: str = "pert_time",
    plate_column: str = "Yeast_cell_plate",
    allowed_roles: Iterable[str] = ALLOWED_FOLD_ROLES,
    expected_fold_counts: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    required = {sample_column, "fold_type", "fold_id", "role"}
    missing_columns = sorted(required - set(fold_table.columns))
    if missing_columns:
        raise ContractViolation("Fold table missing columns: %s" % missing_columns)
    allowed = frozenset(str(role) for role in allowed_roles)
    observed_roles = frozenset(fold_table.role.dropna().astype(str).unique())
    if observed_roles - allowed:
        raise ContractViolation("Unknown fold roles: %s" % sorted(observed_roles - allowed))
    if not fold_table[[sample_column, "fold_type", "fold_id"]].duplicated().sum() == 0:
        raise ContractViolation("Duplicate sample assignment within a fold")
    if not train_metadata[sample_column].astype(str).is_unique:
        raise ContractViolation("Training metadata sample IDs are not unique")

    official_ids = set(train_metadata[sample_column].astype(str))
    table_ids = set(fold_table[sample_column].astype(str))
    if table_ids != official_ids:
        raise ContractViolation(
            "Fold sample universe mismatch: missing=%d extra=%d"
            % (len(official_ids - table_ids), len(table_ids - official_ids))
        )
    metadata_by_id = train_metadata.copy()
    metadata_by_id.index = metadata_by_id[sample_column].astype(str)
    fold_rows: List[Dict[str, Any]] = []
    fold_counts: Dict[str, int] = {}
    for (fold_type, fold_id), group in fold_table.groupby(["fold_type", "fold_id"], sort=False):
        if len(group) != len(official_ids) or set(group[sample_column].astype(str)) != official_ids:
            raise ContractViolation("Fold %s/%s does not assign every train sample" % (fold_type, fold_id))
        fit_ids = group.loc[group.role.eq("train"), sample_column].astype(str).tolist()
        holdout_ids = group.loc[group.role.eq("holdout"), sample_column].astype(str).tolist()
        excluded_ids = group.loc[group.role.eq("excluded"), sample_column].astype(str).tolist()
        if not fit_ids or not holdout_ids:
            raise ContractViolation("Fold %s/%s has an empty fit or holdout set" % (fold_type, fold_id))
        if set(fit_ids) & set(holdout_ids):
            raise ContractViolation("Fold %s/%s has sample leakage" % (fold_type, fold_id))

        strain_overlap = _entity_values(metadata_by_id, fit_ids, strain_column) & _entity_values(
            metadata_by_id, holdout_ids, strain_column
        )
        compound_overlap = _entity_values(
            metadata_by_id, fit_ids, compound_column
        ) & _entity_values(metadata_by_id, holdout_ids, compound_column)
        time_overlap = _entity_values(metadata_by_id, fit_ids, time_column) & _entity_values(
            metadata_by_id, holdout_ids, time_column
        )
        plate_overlap = _entity_values(metadata_by_id, fit_ids, plate_column) & _entity_values(
            metadata_by_id, holdout_ids, plate_column
        )
        if fold_type == "compound" and compound_overlap:
            raise ContractViolation("Compound leakage in fold %s" % fold_id)
        if fold_type == "strain" and strain_overlap:
            raise ContractViolation("Strain leakage in fold %s" % fold_id)
        if fold_type == "both" and (strain_overlap or compound_overlap):
            raise ContractViolation("Both-OOD entity leakage in fold %s" % fold_id)
        if fold_type == "time" and time_overlap:
            raise ContractViolation("Time leakage in fold %s" % fold_id)
        if fold_type == "plate" and plate_overlap:
            raise ContractViolation("Plate leakage in fold %s" % fold_id)
        fold_counts[str(fold_type)] = fold_counts.get(str(fold_type), 0) + 1
        fold_rows.append(
            {
                "fold_type": str(fold_type),
                "fold_id": str(fold_id),
                "n_train": len(fit_ids),
                "n_holdout": len(holdout_ids),
                "n_excluded": len(excluded_ids),
                "strain_overlap_count": len(strain_overlap),
                "compound_overlap_count": len(compound_overlap),
                "time_overlap_count": len(time_overlap),
                "plate_overlap_count": len(plate_overlap),
            }
        )

    if expected_fold_counts is not None:
        expected = {str(key): int(value) for key, value in expected_fold_counts.items()}
        if fold_counts != expected:
            raise ContractViolation("Fold counts mismatch: observed=%s expected=%s" % (fold_counts, expected))
    both = fold_table.loc[fold_table.fold_type.eq("both")]
    both_holdout_counts = both.loc[both.role.eq("holdout")].groupby(sample_column).size()
    both_exactly_once = bool(
        len(both_holdout_counts) == len(official_ids) and (both_holdout_counts == 1).all()
    )
    if not both_exactly_once:
        raise ContractViolation("Complete both grid does not hold out every train sample exactly once")
    return {
        "allowed_roles": sorted(allowed),
        "observed_roles": sorted(observed_roles),
        "fold_counts": fold_counts,
        "fold_count_total": int(sum(fold_counts.values())),
        "both_holdout_complete_exactly_once": both_exactly_once,
        "entity_exclusivity_pass": True,
        "sample_assignment_pass": True,
        "folds": fold_rows,
    }


def audit_fold_paths(
    fold_path: Any,
    metadata_path: Any,
    split_column: str,
    expected_fit_samples: int,
    expected_fold_counts: Mapping[str, int],
    root: Path = ROOT,
) -> Dict[str, Any]:
    metadata = load_metadata(assert_safe_input_path(metadata_path, purpose="preflight", root=root))
    train_metadata = selection_metadata(
        metadata,
        split_column=split_column,
        allowed_splits=("train",),
        expected_samples=expected_fit_samples,
    )
    folds = guarded_read_csv(
        resolve_project_path(fold_path, root=root), purpose="preflight", dtype=str, low_memory=False
    )
    return validate_fold_contract(
        folds,
        train_metadata,
        expected_fold_counts=expected_fold_counts,
    )

