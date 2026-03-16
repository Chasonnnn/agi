from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import ACTION_DIR, ANNOTATION_DIR, ARTIFACTS_DIR, CANDIDATE_DIR, DATA_DIR, DOCS_DIR, INTERIM_DIR, PREDICTIONS_DIR, RAW_DIR, RUNS_DIR
from .schemas import ActionRecord, CandidateRecord


def ensure_repo_layout() -> dict[str, Path]:
    paths = {
        "data": DATA_DIR,
        "raw": RAW_DIR,
        "interim": INTERIM_DIR,
        "candidate": CANDIDATE_DIR,
        "action": ACTION_DIR,
        "artifacts": ARTIFACTS_DIR,
        "annotation": ANNOTATION_DIR,
        "predictions": PREDICTIONS_DIR,
        "runs": RUNS_DIR,
        "docs": DOCS_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def _require_fields(record: dict[str, Any], required: tuple[str, ...], *, path: Path, index: int) -> None:
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"{path}:{index}: missing required fields: {', '.join(missing)}")


def validate_candidate_records(path: Path) -> list[CandidateRecord]:
    validated: list[CandidateRecord] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        _require_fields(record, ("id", "subject", "tokens", "labels"), path=path, index=index)
        tokens = record["tokens"]
        labels = record["labels"]
        if not isinstance(tokens, list) or not isinstance(labels, list):
            raise ValueError(f"{path}:{index}: tokens and labels must both be lists")
        if len(tokens) != len(labels):
            raise ValueError(f"{path}:{index}: tokens/labels length mismatch ({len(tokens)} != {len(labels)})")
        validated.append(
            CandidateRecord(
                id=str(record["id"]),
                subject=str(record["subject"]),
                tokens=[str(token) for token in tokens],
                labels=[str(label) for label in labels],
                anchor_text=record.get("anchor_text"),
                dialogue_id=record.get("dialogue_id"),
                speaker_role=record.get("speaker_role"),
                context_text=record.get("context_text"),
                metadata=record.get("metadata", {}) or {},
            )
        )
    return validated


def validate_action_records(path: Path) -> list[ActionRecord]:
    validated: list[ActionRecord] = []
    for index, record in enumerate(load_jsonl(path), start=1):
        _require_fields(record, ("id", "subject", "span_text", "context_text", "action_label"), path=path, index=index)
        validated.append(
            ActionRecord(
                id=str(record["id"]),
                subject=str(record["subject"]),
                span_text=str(record["span_text"]),
                context_text=str(record["context_text"]),
                action_label=str(record["action_label"]),
                eval_slice=record.get("eval_slice"),
                anchor_text=record.get("anchor_text"),
                speaker_role=record.get("speaker_role"),
                entity_type=record.get("entity_type"),
                semantic_role=record.get("semantic_role"),
                intent_label=record.get("intent_label"),
                dialogue_id=record.get("dialogue_id"),
                cost=record.get("cost"),
                latency_ms=record.get("latency_ms"),
                metadata=record.get("metadata", {}) or {},
            )
        )
    return validated
