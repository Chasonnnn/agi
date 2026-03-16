from __future__ import annotations

import argparse
import json
from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.annotation import (
    bio_spans,
    find_subsequence,
    guess_entity_type,
    is_contact_like,
    is_math_like,
    is_name_like,
    math_like_spans,
    preview_tokens,
    span_text,
    spans_overlap,
)
from contextshift_deid.constants import ANNOTATION_DIR, CANDIDATE_DIR
from contextshift_deid.data import ensure_repo_layout, load_jsonl, validate_action_records, validate_candidate_records
from contextshift_deid.tokenization import tokenize_text

TITLE_STOPWORDS = {"hi", "hello", "hey", "thanks", "thank", "ok", "okay", "yes", "no", "i"}
SPAN_SUFFIX_RE = re.compile(r"-span-\d+$")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _infer_split_name(path: Path) -> str:
    return path.stem


def _source_row_id(action_id: str) -> str:
    return SPAN_SUFFIX_RE.sub("", action_id)


def _load_prediction_map(path: Path, candidates_by_id: dict[str, Any]) -> dict[str, list[str]]:
    prediction_map: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        record_id = str(row["id"])
        if record_id not in candidates_by_id:
            raise ValueError(f"{path}: prediction id not found in candidate split: {record_id}")
        predicted_labels = [str(label) for label in row["predicted_labels"]]
        expected_length = len(candidates_by_id[record_id].labels)
        if len(predicted_labels) != expected_length:
            raise ValueError(
                f"{path}: prediction length mismatch for {record_id} "
                f"({len(predicted_labels)} != {expected_length})"
            )
        prediction_map[record_id] = predicted_labels
    return prediction_map


def _titlecase_probes(tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index].strip()
        normalized = token.strip(".,!?()[]{}:;\"'").casefold()
        if normalized in TITLE_STOPWORDS:
            index += 1
            continue
        if normalized in {"mr", "mrs", "ms", "dr", "prof", "teacher", "tutor"} and index + 1 < len(tokens):
            next_token = tokens[index + 1]
            if next_token[:1].isupper():
                spans.append((index, index + 2))
                index += 2
                continue
        if token[:1].isupper() and any(character.isalpha() for character in token):
            end = index + 1
            while end < len(tokens) and end - index < 3:
                next_token = tokens[end]
                if not next_token[:1].isupper():
                    break
                end += 1
            spans.append((index, end))
            index = end
            continue
        index += 1
    return spans


def _build_item(
    *,
    item_id: str,
    split_name: str,
    source_row_id: str,
    subject: str,
    span_value: str,
    context_text: str,
    anchor_text: str | None,
    dialogue_id: str | None,
    speaker_role: str | None,
    entity_type: str | None,
    semantic_role: str | None,
    token_span: tuple[int, int] | None,
    token_preview: str | None,
    pool_source: str,
    suggested_action: str,
    suggested_reason: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "split": split_name,
        "source_row_id": source_row_id,
        "subject": subject,
        "span_text": span_value,
        "context_text": context_text,
        "pool_source": pool_source,
        "suggested_action": suggested_action,
        "suggested_reason": suggested_reason,
        "metadata": metadata,
    }
    if anchor_text:
        item["anchor_text"] = anchor_text
    if dialogue_id:
        item["dialogue_id"] = dialogue_id
    if speaker_role:
        item["speaker_role"] = speaker_role
    if entity_type:
        item["entity_type"] = entity_type
    if semantic_role:
        item["semantic_role"] = semantic_role
    if token_span is not None:
        item["token_start"] = token_span[0]
        item["token_end"] = token_span[1]
    if token_preview:
        item["token_preview"] = token_preview
    return item


def _stable_rank(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _apply_source_caps(
    rows: list[dict[str, Any]],
    *,
    max_prediction_probes_total: int,
    max_titlecase_probes_total: int,
    max_math_probes_total: int,
) -> list[dict[str, Any]]:
    source_limits = {
        "candidate_prediction": max_prediction_probes_total,
        "titlecase_probe": max_titlecase_probes_total,
        "math_probe": max_math_probes_total,
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["pool_source"]), []).append(row)

    filtered: list[dict[str, Any]] = []
    for source, source_rows in grouped.items():
        limit = source_limits.get(source)
        if limit is None or limit < 0 or len(source_rows) <= limit:
            filtered.extend(source_rows)
            continue
        ranked_rows = sorted(source_rows, key=lambda row: (_stable_rank(str(row["id"])), str(row["id"])))
        filtered.extend(ranked_rows[:limit])
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an action-label annotation pool from candidate rows and optional predictions.")
    parser.add_argument("--candidate-file", type=Path, default=CANDIDATE_DIR / "dev.jsonl")
    parser.add_argument(
        "--provisional-action-file",
        type=Path,
        help="Optional positive-only action split to seed clear REDACT examples.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        help="Optional candidate prediction file. Predicted-only spans become manual-review probes.",
    )
    parser.add_argument("--split-name", help="Split label stored in the pool rows. Defaults to the candidate file stem.")
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Defaults to artifacts/annotation/action_pool_<split>.jsonl",
    )
    parser.add_argument(
        "--max-prediction-probes-per-row",
        type=int,
        default=2,
        help="Cap predicted-only probes from the candidate model per row.",
    )
    parser.add_argument(
        "--max-prediction-probes-total",
        type=int,
        default=400,
        help="Cap predicted-only probes across the entire exported pool. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-titlecase-probes-per-row",
        type=int,
        default=1,
        help="Cap extra titlecase/name-like probes per row when predictions are insufficient.",
    )
    parser.add_argument(
        "--max-math-probes-per-row",
        type=int,
        default=1,
        help="Cap extra math-like probes per row so likely KEEP cases are available during labeling.",
    )
    parser.add_argument(
        "--max-titlecase-probes-total",
        type=int,
        default=250,
        help="Cap titlecase/name-like probes across the entire exported pool. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-math-probes-total",
        type=int,
        default=250,
        help="Cap math-like probes across the entire exported pool. Use -1 to disable.",
    )
    args = parser.parse_args()

    ensure_repo_layout()
    split_name = args.split_name or _infer_split_name(args.candidate_file)
    output_file = args.output_file or (ANNOTATION_DIR / f"action_pool_{split_name}.jsonl")

    candidate_records = validate_candidate_records(args.candidate_file)
    candidates_by_id = {record.id: record for record in candidate_records}
    gold_spans_by_id = {record.id: bio_spans(record.labels) for record in candidate_records}

    prediction_map: dict[str, list[str]] = {}
    if args.prediction_file:
        prediction_map = _load_prediction_map(args.prediction_file, candidates_by_id)

    pool_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int | str, int | str]] = set()
    source_counts: Counter[str] = Counter()
    suggestion_counts: Counter[str] = Counter()

    def add_row(
        *,
        item_id: str,
        source_row_id: str,
        subject: str,
        span_value: str,
        context_text: str,
        anchor_text: str | None,
        dialogue_id: str | None,
        speaker_role: str | None,
        entity_type: str | None,
        semantic_role: str | None,
        token_span: tuple[int, int] | None,
        pool_source: str,
        suggested_action: str,
        suggested_reason: str,
        metadata: dict[str, Any],
    ) -> None:
        if token_span is not None:
            key: tuple[str, int | str, int | str] = (source_row_id, token_span[0], token_span[1])
        else:
            key = (source_row_id, pool_source, span_value.casefold())
        if key in seen_keys:
            return
        seen_keys.add(key)
        pool_rows.append(
            _build_item(
                item_id=item_id,
                split_name=split_name,
                source_row_id=source_row_id,
                subject=subject,
                span_value=span_value,
                context_text=context_text,
                anchor_text=anchor_text,
                dialogue_id=dialogue_id,
                speaker_role=speaker_role,
                entity_type=entity_type,
                semantic_role=semantic_role,
                token_span=token_span,
                token_preview=preview_tokens(candidates_by_id[source_row_id].tokens, token_span) if token_span is not None else None,
                pool_source=pool_source,
                suggested_action=suggested_action,
                suggested_reason=suggested_reason,
                metadata=metadata,
            )
        )
    if args.provisional_action_file is not None and args.provisional_action_file.exists():
        for record in validate_action_records(args.provisional_action_file):
            source_row_id = _source_row_id(record.id)
            candidate = candidates_by_id.get(source_row_id)
            token_span = None
            if candidate is not None:
                token_span = find_subsequence(candidate.tokens, tokenize_text(record.span_text))
            add_row(
                item_id=record.id,
                source_row_id=source_row_id,
                subject=record.subject,
                span_value=record.span_text,
                context_text=record.context_text,
                anchor_text=record.anchor_text,
                dialogue_id=record.dialogue_id,
                speaker_role=record.speaker_role,
                entity_type=record.entity_type,
                semantic_role=record.semantic_role,
                token_span=token_span,
                pool_source="legacy_positive",
                suggested_action="REDACT",
                suggested_reason="Imported from the current positive-only action split.",
                metadata={
                    "source": "legacy_positive",
                    "legacy_action_id": record.id,
                    "legacy_metadata": record.metadata,
                },
            )

    for record in candidate_records:
        occupied_spans = gold_spans_by_id[record.id][:]
        for span in gold_spans_by_id[record.id]:
            span_tokens = record.tokens[span[0] : span[1]]
            span_value = span_text(record.tokens, span)
            if is_contact_like(span_value):
                suggested_action = "REDACT"
                suggested_reason = "Gold candidate span looks like direct contact or account information."
            elif is_math_like(span_tokens):
                suggested_action = "KEEP"
                suggested_reason = "Gold candidate span looks curricular or mathematical and should be checked as a likely KEEP."
            else:
                suggested_action = "REVIEW"
                suggested_reason = "Gold candidate span from manual candidate labeling; confirm the final action."
            add_row(
                item_id=f"{record.id}-gold-{span[0]}-{span[1]}",
                source_row_id=record.id,
                subject=record.subject,
                span_value=span_value,
                context_text=record.context_text or " ".join(record.tokens),
                anchor_text=record.anchor_text,
                dialogue_id=record.dialogue_id,
                speaker_role=record.speaker_role,
                entity_type=guess_entity_type(span_value, span_tokens),
                semantic_role=None,
                token_span=span,
                pool_source="candidate_gold",
                suggested_action=suggested_action,
                suggested_reason=suggested_reason,
                metadata={
                    "source": "candidate_gold",
                    "candidate_label_source": (record.metadata or {}).get("label_source"),
                },
            )
        if record.id in prediction_map:
            prediction_spans = bio_spans(prediction_map[record.id])
            probe_count = 0
            for span in prediction_spans:
                if probe_count >= args.max_prediction_probes_per_row:
                    break
                if any(spans_overlap(span, existing) for existing in occupied_spans):
                    continue
                span_value = span_text(record.tokens, span)
                if is_contact_like(span_value):
                    suggested_action = "REDACT"
                    suggested_reason = "Candidate-only span looks like contact info and should be checked as a likely REDACT."
                elif is_math_like(record.tokens[span[0] : span[1]]):
                    suggested_action = "KEEP"
                    suggested_reason = "Candidate-only span looks curricular or mathematical and should be checked as a likely KEEP."
                else:
                    suggested_action = "REVIEW"
                    suggested_reason = "Candidate model proposed this span, but it does not overlap a legacy positive annotation."
                add_row(
                    item_id=f"{record.id}-pred-{span[0]}-{span[1]}",
                    source_row_id=record.id,
                    subject=record.subject,
                    span_value=span_value,
                    context_text=record.context_text or " ".join(record.tokens),
                    anchor_text=record.anchor_text,
                    dialogue_id=record.dialogue_id,
                    speaker_role=record.speaker_role,
                    entity_type=guess_entity_type(span_value, record.tokens[span[0] : span[1]]),
                    semantic_role=None,
                    token_span=span,
                    pool_source="candidate_prediction",
                    suggested_action=suggested_action,
                    suggested_reason=suggested_reason,
                    metadata={
                        "source": "candidate_prediction",
                        "prediction_file": str(args.prediction_file),
                    },
                )
                occupied_spans.append(span)
                probe_count += 1

        titlecase_count = 0
        for span in _titlecase_probes(record.tokens):
            if titlecase_count >= args.max_titlecase_probes_per_row:
                break
            if any(spans_overlap(span, existing) for existing in occupied_spans):
                continue
            span_tokens = record.tokens[span[0] : span[1]]
            if not is_name_like(span_tokens):
                continue
            span_value = span_text(record.tokens, span)
            if span_value.casefold() in TITLE_STOPWORDS:
                continue
            add_row(
                item_id=f"{record.id}-title-{span[0]}-{span[1]}",
                source_row_id=record.id,
                subject=record.subject,
                span_value=span_value,
                context_text=record.context_text or " ".join(record.tokens),
                anchor_text=record.anchor_text,
                dialogue_id=record.dialogue_id,
                speaker_role=record.speaker_role,
                entity_type=guess_entity_type(span_value, span_tokens),
                semantic_role=None,
                token_span=span,
                pool_source="titlecase_probe",
                suggested_action="REVIEW",
                suggested_reason="Name-like probe added so KEEP/REVIEW cases are available during manual labeling.",
                metadata={"source": "titlecase_probe"},
            )
            occupied_spans.append(span)
            titlecase_count += 1

        math_count = 0
        for span in math_like_spans(record.tokens):
            if math_count >= args.max_math_probes_per_row:
                break
            if any(spans_overlap(span, existing) for existing in occupied_spans):
                continue
            span_value = span_text(record.tokens, span)
            add_row(
                item_id=f"{record.id}-math-{span[0]}-{span[1]}",
                source_row_id=record.id,
                subject=record.subject,
                span_value=span_value,
                context_text=record.context_text or " ".join(record.tokens),
                anchor_text=record.anchor_text,
                dialogue_id=record.dialogue_id,
                speaker_role=record.speaker_role,
                entity_type=guess_entity_type(span_value, record.tokens[span[0] : span[1]]),
                semantic_role=None,
                token_span=span,
                pool_source="math_probe",
                suggested_action="KEEP",
                suggested_reason="Math-like probe added so likely curricular KEEP cases are present during manual labeling.",
                metadata={"source": "math_probe"},
            )
            occupied_spans.append(span)
            math_count += 1

    pool_rows = _apply_source_caps(
        pool_rows,
        max_prediction_probes_total=args.max_prediction_probes_total,
        max_titlecase_probes_total=args.max_titlecase_probes_total,
        max_math_probes_total=args.max_math_probes_total,
    )
    pool_rows.sort(key=lambda row: (row["source_row_id"], row["pool_source"], row["id"]))
    source_counts.clear()
    suggestion_counts.clear()
    for row in pool_rows:
        source_counts[str(row["pool_source"])] += 1
        suggestion_counts[str(row["suggested_action"])] += 1
    _write_jsonl(output_file, pool_rows)
    print(
        json.dumps(
            {
                "candidate_file": str(args.candidate_file),
                "provisional_action_file": (
                    str(args.provisional_action_file)
                    if args.provisional_action_file is not None and args.provisional_action_file.exists()
                    else None
                ),
                "prediction_file": str(args.prediction_file) if args.prediction_file else None,
                "output_file": str(output_file),
                "split_name": split_name,
                "row_count": len(pool_rows),
                "by_source": source_counts,
                "by_suggested_action": suggestion_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
