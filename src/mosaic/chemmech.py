"""Deterministic chemical identity and functional-feature utilities."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

import numpy as np


SALT_SUFFIXES = (
    " hydrochloride", " dihydrochloride", " citrate", " isethionate",
    " hyclate", " dihydrate", " monohydrate",
)


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("α", "alpha").replace("β", "beta")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalized_parent_name(value: object) -> str:
    text = normalize_name(value)
    for suffix in SALT_SUFFIXES:
        normalized = normalize_name(suffix)
        if text.endswith(" " + normalized):
            return text[: -(len(normalized) + 1)]
    return text


def strip_concentration(value: object) -> str:
    text = str(value or "")
    # Only strip a trailing parenthesized dose. Multi-agent strings separated
    # by | remain ineligible for exact single-compound mapping.
    return re.sub(r"\s*\([^)]*(?:m|u|µ|μ|n)?g?/?m?[lL]?[^)]*\)\s*$", "", text).strip()


def robust_zscore(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (values - median) / scale


def allowed_external(mode: str, exact_entity_profile: bool) -> bool:
    if mode not in {"open_entity", "strict_zero_shot"}:
        raise ValueError(mode)
    return bool(exact_entity_profile) if mode == "open_entity" else False

