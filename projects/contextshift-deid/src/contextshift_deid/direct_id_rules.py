from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from .constants import DEFAULT_ACTION_LABELS

DIRECT_ID_ENTITY_TYPES = frozenset(
    {
        "EMAIL_ADDRESS",
        "URL",
        "IP_ADDRESS",
        "US_BANK_NUMBER",
        "US_PASSPORT",
        "US_SSN",
        "US_DRIVER_LICENSE",
    }
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_URL_RE = re.compile(
    r"(?:(?:https?://|www\.)[^\s]+|\b(?:[A-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|us|me|ai|app|dev|gg|tv|ly)\b(?:/[^\s]*)?)",
    re.IGNORECASE,
)
_IP_ADDRESS_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DOB_HINT_RE = re.compile(r"\b(?:dob|d\.o\.b\.|date of birth|birthdate|born)\b", re.IGNORECASE)
_DRIVER_LICENSE_RE = re.compile(r"\bDLN?[A-Z0-9-]{6,}\b", re.IGNORECASE)

_FORCED_REDACT_PROBABILITIES = {
    label: (1.0 if label == "REDACT" else 0.0)
    for label in DEFAULT_ACTION_LABELS
}


def detect_direct_id(span_text: str, *, entity_type: str | None = None) -> dict[str, str] | None:
    normalized_entity_type = (entity_type or "").strip().upper()
    if normalized_entity_type in DIRECT_ID_ENTITY_TYPES:
        return {
            "source": "entity_type",
            "reason": normalized_entity_type,
        }

    text = span_text.strip()
    if not text:
        return None

    if _EMAIL_RE.search(text):
        return {"source": "pattern", "reason": "EMAIL_ADDRESS"}
    if _URL_RE.search(text):
        return {"source": "pattern", "reason": "URL"}
    if _IP_ADDRESS_RE.search(text):
        return {"source": "pattern", "reason": "IP_ADDRESS"}
    if _SSN_RE.search(text):
        return {"source": "pattern", "reason": "US_SSN"}
    if _DOB_HINT_RE.search(text):
        return {"source": "pattern", "reason": "DATE_OF_BIRTH"}
    if _DRIVER_LICENSE_RE.search(text):
        return {"source": "pattern", "reason": "US_DRIVER_LICENSE"}
    return None


def override_prediction_for_direct_id(
    gold_row: Mapping[str, Any],
    prediction_row: Mapping[str, Any],
) -> dict[str, Any]:
    override = detect_direct_id(
        str(gold_row.get("span_text") or ""),
        entity_type=gold_row.get("entity_type"),
    )
    if override is None:
        return dict(prediction_row)
    patched = dict(prediction_row)
    patched["predicted_action"] = "REDACT"
    patched["confidence"] = 1.0
    patched["probabilities"] = dict(_FORCED_REDACT_PROBABILITIES)
    patched["direct_id_override"] = override
    return patched


def apply_direct_id_overrides(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    patched_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    overridden_ids: list[str] = []

    for gold_row in gold_rows:
        row_id = str(gold_row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row_id}")
        patched = override_prediction_for_direct_id(gold_row, prediction)
        override = patched.get("direct_id_override")
        if isinstance(override, Mapping):
            overridden_ids.append(row_id)
            reason_counts[str(override["reason"])] += 1
            source_counts[str(override["source"])] += 1
        patched_rows.append(patched)

    summary = {
        "override_count": len(overridden_ids),
        "overridden_ids": overridden_ids,
        "by_reason": dict(reason_counts),
        "by_source": dict(source_counts),
    }
    return patched_rows, summary
