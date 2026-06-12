"""
labels.py — measured/derived provenance labels + a CSV writer that ENFORCES them.

Absolute v2 rule: every data column in every output CSV/JSON must declare whether
it is `measured` or `derived`. This module makes that mechanical — a column with
no label cannot be written (the writer raises). Provenance travels with the data
via a ``{column}__{label}`` header suffix, e.g. ``latency_ms__measured``,
``waves__derived``.

  measured : CUDA-event latency, BandwidthEstimator achieved BW, NVML readings.
  derived  : n_blocks, waves, util%, saturation-point estimates, share/ratio calc.
  metadata : inputs / identifiers (model, batch, sm_count, status). Not a claim
             about a result — distinguished so they are never mislabeled as
             measured/derived. (The v1 sweep_spec CSV convention already uses
             this measured/derived/metadata split.)

This module is dependency-light (stdlib only) so GPU-free experiments (E0) can
import it without torch.
"""

from __future__ import annotations

import csv
import json
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Union

__all__ = ["Label", "write_labeled_csv", "read_labeled_csv", "write_labeled_json"]


class Label(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    METADATA = "metadata"


LabelLike = Union[Label, str]
_SEP = "__"  # header suffix separator: "{col}__{label}"


def _coerce(label: LabelLike, col: str) -> Label:
    if isinstance(label, Label):
        return label
    try:
        return Label(str(label).strip().lower())
    except ValueError as e:
        raise ValueError(
            f"column {col!r}: invalid label {label!r}; "
            f"must be one of {[l.value for l in Label]}"
        ) from e


def _validate_schema(
    columns: Iterable[str], schema: Mapping[str, LabelLike]
) -> "dict[str, Label]":
    """Every column must have a label; every label must be valid. Else raise."""
    cols = list(columns)
    coerced = {c: _coerce(schema[c], c) for c in schema}

    missing = [c for c in cols if c not in coerced]
    if missing:
        raise ValueError(
            f"unlabeled column(s) {missing} — refusing to write. Add a "
            f"Label.MEASURED / DERIVED / METADATA entry for each in `schema`."
        )
    extra = [c for c in coerced if c not in cols]
    if extra:
        raise ValueError(
            f"schema declares column(s) {extra} not present in the rows."
        )
    return coerced


def write_labeled_csv(
    path: Union[str, Path],
    rows: "list[dict]",
    schema: Mapping[str, LabelLike],
) -> Path:
    """Write *rows* to *path*, suffixing each header with its provenance label.

    Args:
        path:   output CSV path (parent dirs created).
        rows:   list of dicts. All dicts must use exactly the columns in *schema*.
        schema: {column_name: Label}. EVERY column must appear here or the write
                is refused. Use Label.METADATA for inputs/identifiers.

    Raises:
        ValueError: a column is unlabeled, a label is invalid, or a row's keys
                    do not match the schema.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # still enforce that a schema was supplied; write header-only file.
        coerced = {c: _coerce(schema[c], c) for c in schema}
        header = [f"{c}{_SEP}{coerced[c].value}" for c in coerced]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([_value_kind_comment(coerced)])
            w.writerow(header)
        return path

    columns = list(rows[0].keys())
    coerced = _validate_schema(columns, schema)

    # Reject ragged rows so no column silently slips through unlabeled.
    colset = set(columns)
    for i, r in enumerate(rows):
        if set(r.keys()) != colset:
            raise ValueError(
                f"row {i} keys {sorted(r.keys())} != schema columns "
                f"{sorted(colset)} — every row must carry every labeled column."
            )

    header = [f"{c}{_SEP}{coerced[c].value}" for c in columns]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([_value_kind_comment(coerced)])  # human-readable provenance line
        w.writerow(header)
        for r in rows:
            w.writerow([r[c] for c in columns])
    return path


def _value_kind_comment(coerced: "dict[str, Label]") -> str:
    by_kind: "dict[str, list[str]]" = {l.value: [] for l in Label}
    for c, lab in coerced.items():
        by_kind[lab.value].append(c)
    parts = [f"{k}={','.join(v)}" for k, v in by_kind.items() if v]
    return "# value_kind: " + "; ".join(parts)


def read_labeled_csv(path: Union[str, Path]) -> "tuple[list[dict], dict[str, Label]]":
    """Inverse of :func:`write_labeled_csv`.

    Strips the leading ``# value_kind`` comment and the ``__label`` header
    suffixes, returning (rows, schema) with plain column names. Used by
    gates/adjudicate.py so the gate logic reads provenance, not bare numbers.
    """
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        if first and first[0].startswith("# value_kind"):
            header = next(reader)
        else:
            header = first
        schema: "dict[str, Label]" = {}
        plain: "list[str]" = []
        for h in header:
            if _SEP in h:
                col, lab = h.rsplit(_SEP, 1)
                schema[col] = Label(lab)
            else:
                col = h
                schema[col] = Label.METADATA
            plain.append(col)
        rows = [dict(zip(plain, rec)) for rec in reader]
    return rows, schema


def write_labeled_json(
    path: Union[str, Path],
    payload: dict,
    schema: Mapping[str, LabelLike],
) -> Path:
    """Write a JSON object alongside a ``_value_kind`` provenance map.

    The provenance map records, for each top-level key, whether it is
    measured/derived/metadata — the JSON analogue of the CSV header suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    coerced = {c: _coerce(schema[c], c) for c in schema}
    missing = [k for k in payload if k not in coerced]
    if missing:
        raise ValueError(
            f"unlabeled JSON key(s) {missing} — add them to `schema`."
        )
    out = dict(payload)
    out["_value_kind"] = {k: coerced[k].value for k in coerced}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path
