"""Utilities for the MOSAIC natural-isolate proteome contract.

The module deliberately uses only the Python standard library to read XLSX
files.  The project training environment does not depend on ``openpyxl`` and
the official supplementary workbook is simple enough to stream safely.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np


XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
LOCUS_RE = re.compile(r"(Y[A-P][LR]\d{3}[CW](?:-[A-Z])?)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    value = 0
    for character in letters.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


class StreamingXLSX:
    """Small streaming reader for values in an OOXML workbook."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path)
        self.shared_strings = self._load_shared_strings()
        self.sheet_paths = self._load_sheet_paths()

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "StreamingXLSX":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _load_shared_strings(self) -> List[str]:
        if "xl/sharedStrings.xml" not in self._zip.namelist():
            return []
        root = ET.parse(self._zip.open("xl/sharedStrings.xml")).getroot()
        values = []
        for item in root.findall(XLSX_NS + "si"):
            values.append("".join(node.text or "" for node in item.iter(XLSX_NS + "t")))
        return values

    def _load_sheet_paths(self) -> Dict[str, str]:
        workbook = ET.parse(self._zip.open("xl/workbook.xml")).getroot()
        relationships = ET.parse(self._zip.open("xl/_rels/workbook.xml.rels")).getroot()
        target_by_id = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships.findall(PKG_REL_NS + "Relationship")
        }
        output: Dict[str, str] = {}
        sheets = workbook.find(XLSX_NS + "sheets")
        if sheets is None:
            return output
        for sheet in sheets.findall(XLSX_NS + "sheet"):
            relation_id = sheet.attrib[REL_NS + "id"]
            target = target_by_id[relation_id].lstrip("/")
            output[sheet.attrib["name"]] = target if target.startswith("xl/") else "xl/" + target
        return output

    @property
    def sheet_names(self) -> List[str]:
        return list(self.sheet_paths)

    def _cell_value(self, cell: ET.Element):
        cell_type = cell.attrib.get("t")
        value_node = cell.find(XLSX_NS + "v")
        if cell_type == "inlineStr":
            inline = cell.find(XLSX_NS + "is")
            return "" if inline is None else "".join(node.text or "" for node in inline.iter(XLSX_NS + "t"))
        if value_node is None or value_node.text is None:
            return None
        raw = value_node.text
        if cell_type == "s":
            return self.shared_strings[int(raw)]
        if cell_type in {"str", "e"}:
            return raw
        if cell_type == "b":
            return raw == "1"
        try:
            number = float(raw)
            return int(number) if number.is_integer() else number
        except ValueError:
            return raw

    def iter_rows(self, sheet_name: str) -> Iterator[Tuple[int, List[object]]]:
        if sheet_name not in self.sheet_paths:
            raise KeyError("Unknown sheet %r; available=%s" % (sheet_name, self.sheet_names))
        with self._zip.open(self.sheet_paths[sheet_name]) as handle:
            for _, row in ET.iterparse(handle, events=("end",)):
                if row.tag != XLSX_NS + "row":
                    continue
                values: Dict[int, object] = {}
                for cell in row.findall(XLSX_NS + "c"):
                    values[_column_index(cell.attrib.get("r", "A1"))] = self._cell_value(cell)
                width = max(values.keys()) + 1 if values else 0
                yield int(row.attrib.get("r", "0")), [values.get(index) for index in range(width)]
                row.clear()


def read_sheet_table(path: Path, sheet_name: str, header_row: int) -> Tuple[List[str], List[List[object]]]:
    header: Optional[List[str]] = None
    rows: List[List[object]] = []
    with StreamingXLSX(path) as workbook:
        for row_number, values in workbook.iter_rows(sheet_name):
            if row_number == header_row:
                header = ["" if value is None else str(value) for value in values]
            elif row_number > header_row and header is not None and any(value is not None for value in values):
                rows.append(values + [None] * (len(header) - len(values)))
    if header is None:
        raise ValueError("Header row %d not found in %s" % (header_row, sheet_name))
    return header, rows


def load_sgd_locus_map(path: Path) -> Dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("SGD feature table has no header")
        reader.fieldnames = [field.lstrip("# ") for field in reader.fieldnames]
        return {
            row["locus_tag"].upper(): row["symbol"].strip()
            for row in reader
            if row.get("feature") == "gene" and row.get("locus_tag") and row.get("symbol", "").strip()
        }


def parse_locus(value: object) -> Optional[str]:
    if value is None:
        return None
    match = LOCUS_RE.search(str(value))
    return match.group(1).upper() if match else None


def resolve_protein(source: object, locus_to_symbol: Mapping[str, str], retained: Sequence[str]) -> Tuple[Optional[str], Optional[str], str]:
    retained_set = retained if isinstance(retained, set) else set(retained)
    source_text = "" if source is None else str(source).strip()
    locus = parse_locus(source_text)
    symbol = locus_to_symbol.get(locus) if locus else None
    for candidate in (symbol, locus, source_text):
        if candidate and candidate in retained_set:
            return candidate, symbol, "mapped_retained"
    if locus is None:
        return None, symbol, "invalid_or_noncanonical_identifier"
    if symbol is None:
        return None, symbol, "locus_without_symbol_not_retained"
    return None, symbol, "symbol_not_retained"


def detect_log_transform(values: np.ndarray) -> Dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Natural proteome contains no finite values")
    quantiles = np.quantile(finite, [0.01, 0.50, 0.95, 0.99])
    raw_positive = bool(np.min(finite) > 0 and quantiles[2] > 64)
    return {
        "input_scale": "positive_linear_abundance" if raw_positive else "already_log_or_standardized",
        "transform": "log2" if raw_positive else "identity",
        "min": float(np.min(finite)),
        "q01": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
    }


def apply_detected_transform(values: np.ndarray, detection: Mapping[str, object]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    output[~np.isfinite(output)] = np.nan
    if detection["transform"] == "log2":
        output[output <= 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            output = np.log2(output)
    return output


def parse_aneuploidies(value: object) -> Tuple[int, Dict[int, int]]:
    text = "" if value is None else str(value)
    changes = {chromosome: 0 for chromosome in range(1, 17)}
    if text.lower() == "euploid":
        return 0, changes
    for sign, copies, chromosome in re.findall(r"([+-])(\d+)\*(\d+)", text):
        changes[int(chromosome)] += (1 if sign == "+" else -1) * int(copies)
    return int(sum(abs(value) for value in changes.values())), changes


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

