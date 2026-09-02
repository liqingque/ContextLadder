"""Leakage-aware loaders for the GOAI virtual-cell data contract."""

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _find_sample_id(columns: Iterable[str]) -> str:
    columns = list(columns)
    for candidate in ("sample_ID", "sample_id", "Sample_ID", "sample"):
        if candidate in columns:
            return candidate
    raise ValueError("Could not identify sample ID column")


def load_metadata(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load metadata while preserving textual identifiers and missing values."""
    path = str(Path(path))
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=True, low_memory=False, nrows=nrows)
    sample_id = _find_sample_id(df.columns)
    df[sample_id] = df[sample_id].astype("string")
    return df


def load_proteome(path: str, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, str, list]:
    """Load a proteome table and return (frame, sample_id_column, protein_columns)."""
    path = str(Path(path))
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        # Read the header once, then parse all protein columns in one vectorized
        # CSV pass. Per-column conversion is prohibitively slow for 5,243 columns.
        header = pd.read_csv(path, nrows=0)
        sample_id = _find_sample_id(header.columns)
        dtype = {c: "float32" for c in header.columns if c != sample_id}
        dtype[sample_id] = str
        df = pd.read_csv(path, dtype=dtype, low_memory=False, nrows=nrows)
    sample_id = _find_sample_id(df.columns)
    df[sample_id] = df[sample_id].astype("string")
    protein_columns = [c for c in df.columns if c != sample_id]
    # CSV parsing already assigned float32 to protein columns. This also preserves
    # missing values for the explicit audit rather than silently imputing them.
    return df, sample_id, protein_columns


def validate_unique_sample_id(df: pd.DataFrame, sample_id: str) -> None:
    if df[sample_id].isna().any():
        raise ValueError("Sample ID contains missing values")
    if not df[sample_id].is_unique:
        dup = df.loc[df[sample_id].duplicated(keep=False), sample_id].astype(str).unique()[:10]
        raise ValueError("Sample ID is not unique: %s" % list(dup))


def align_metadata_proteome(
    metadata: pd.DataFrame, proteome: pd.DataFrame, metadata_sample_id: Optional[str] = None,
    proteome_sample_id: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, list]:
    """Align proteome rows to metadata order by sample ID without changing columns."""
    metadata_sample_id = metadata_sample_id or _find_sample_id(metadata.columns)
    proteome_sample_id = proteome_sample_id or _find_sample_id(proteome.columns)
    validate_unique_sample_id(metadata, metadata_sample_id)
    validate_unique_sample_id(proteome, proteome_sample_id)
    meta_ids = set(metadata[metadata_sample_id].astype(str))
    prot_ids = set(proteome[proteome_sample_id].astype(str))
    if meta_ids != prot_ids:
        missing_in_proteome = sorted(meta_ids - prot_ids)[:10]
        missing_in_metadata = sorted(prot_ids - meta_ids)[:10]
        raise ValueError(
            "Metadata/proteome sample IDs differ; missing_in_proteome=%s missing_in_metadata=%s"
            % (missing_in_proteome, missing_in_metadata)
        )
    index = pd.Index(metadata[metadata_sample_id].astype(str), name=proteome_sample_id)
    aligned_proteome = proteome.copy()
    aligned_proteome[proteome_sample_id] = aligned_proteome[proteome_sample_id].astype(str)
    aligned_proteome = aligned_proteome.set_index(proteome_sample_id).loc[index].reset_index()
    aligned_proteome[proteome_sample_id] = aligned_proteome[proteome_sample_id].astype("string")
    return metadata.reset_index(drop=True), aligned_proteome.reset_index(drop=True), proteome_sample_id, [
        c for c in aligned_proteome.columns if c != proteome_sample_id
    ]


def finite_float_matrix(proteome: pd.DataFrame, protein_columns: list) -> np.ndarray:
    """Return the matrix as float32; missing/inf values are preserved for audit."""
    return proteome[protein_columns].to_numpy(dtype=np.float32, copy=True)


def to_log2_proteome(raw: np.ndarray) -> np.ndarray:
    """Convert positive raw abundance values to the official log2 space.

    Non-finite and non-positive values remain NaN and are reported by the audit;
    no imputation is performed here.
    """
    raw = np.asarray(raw, dtype=np.float32)
    out = np.full(raw.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(raw) & (raw > 0)
    out[valid] = np.log2(raw[valid]).astype(np.float32)
    return out
