from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha1
from typing import Any, Iterable, Mapping, Sequence

from .annotation import bio_spans, spans_overlap
from .data import validate_action_records
from .tokenization import tokenize_with_offsets


def stable_hash(value: str, *, seed: int) -> str:
    return sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stable_sort_key(value: str, *, seed: int) -> tuple[str, str]:
    return (stable_hash(value, seed=seed), value)


def char_span_to_token_span(
    token_offsets: Sequence[tuple[int, int]],
    start: int,
    end: int,
) -> tuple[int, int] | None:
    overlapping = [
        index
        for index, (token_start, token_end) in enumerate(token_offsets)
        if token_start < end and token_end > start
    ]
    if not overlapping:
        return None
    return (overlapping[0], overlapping[-1] + 1)


def labels_from_token_spans(token_count: int, spans: Sequence[tuple[int, int]]) -> list[str]:
    labels = ["O"] * token_count
    for start, end in sorted(spans):
        if start < 0 or end > token_count or end <= start:
            continue
        labels[start] = "B-SUSPECT"
        for index in range(start + 1, end):
            labels[index] = "I-SUSPECT"
    return labels


def _turn_identifier(record_id: str, dialogue_id: str | None, turn_index: int | None) -> str:
    if dialogue_id is not None and turn_index is not None:
        return f"{dialogue_id}-turn-{turn_index}"
    if "-tag-" in record_id:
        return record_id.split("-tag-", 1)[0]
    return record_id


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _merge_text_field(current: str | None, candidate: str | None) -> str | None:
    if not candidate:
        return current
    if not current:
        return str(candidate)
    return str(candidate) if len(str(candidate)) > len(str(current)) else current


def build_candidate_proxy_rows_from_action(
    action_file,
    *,
    proxy_label_source: str = "action_proxy_seed_v1",
) -> list[dict[str, Any]]:
    action_records = validate_action_records(action_file)
    turns: dict[str, dict[str, Any]] = {}
    seed_char_spans_by_turn: dict[str, set[tuple[int, int]]] = defaultdict(set)

    for record in action_records:
        metadata = dict(record.metadata or {})
        turn_index = _safe_int(metadata.get("turn_index"))
        turn_id = _turn_identifier(record.id, record.dialogue_id, turn_index)
        turn_text = str(metadata.get("turn_text") or "")
        if not turn_text:
            continue
        turn_text_original = str(metadata.get("turn_text_original") or turn_text)
        tokenized = tokenize_with_offsets(turn_text)
        tokens = [token for token, _ in tokenized]
        token_offsets = [span for _, span in tokenized]
        if not tokens:
            continue

        row = turns.get(turn_id)
        if row is None:
            row = {
                "id": turn_id,
                "subject": record.subject,
                "tokens": tokens,
                "labels": ["O"] * len(tokens),
                "anchor_text": record.anchor_text,
                "dialogue_id": record.dialogue_id,
                "speaker_role": record.speaker_role,
                "context_text": record.context_text,
                "metadata": {
                    "source_action_file": str(action_file),
                    "proxy_label_source": proxy_label_source,
                    "candidate_proxy": True,
                    "turn_index": turn_index,
                    "turn_text": turn_text,
                    "turn_text_original": turn_text_original,
                    "action_seed_spans": [],
                    "action_row_count": 0,
                    "annotation_completed": False,
                },
            }
            turns[turn_id] = row
        else:
            row["anchor_text"] = _merge_text_field(row.get("anchor_text"), record.anchor_text)
            row["context_text"] = _merge_text_field(row.get("context_text"), record.context_text)
            if not row.get("speaker_role") and record.speaker_role:
                row["speaker_role"] = record.speaker_role
            if not row.get("dialogue_id") and record.dialogue_id:
                row["dialogue_id"] = record.dialogue_id

        row["metadata"]["action_row_count"] = int(row["metadata"]["action_row_count"]) + 1

        char_start = _safe_int(metadata.get("tag_start"))
        char_end = _safe_int(metadata.get("tag_end"))
        if char_start is None or char_end is None or char_end <= char_start:
            continue
        if (char_start, char_end) in seed_char_spans_by_turn[turn_id]:
            continue
        token_span = char_span_to_token_span(token_offsets, char_start, char_end)
        if token_span is None:
            continue

        seed_char_spans_by_turn[turn_id].add((char_start, char_end))
        span_text = turn_text[char_start:char_end]
        row["metadata"]["action_seed_spans"].append(
            {
                "char_start": char_start,
                "char_end": char_end,
                "token_start": token_span[0],
                "token_end": token_span[1],
                "span_text": span_text,
                "action_label": record.action_label,
                "entity_type": record.entity_type,
                "semantic_role": record.semantic_role,
                "eval_slice": record.eval_slice,
            }
        )

    proxy_rows: list[dict[str, Any]] = []
    for row in sorted(turns.values(), key=lambda item: str(item["id"])):
        seed_spans = row["metadata"]["action_seed_spans"]
        token_spans = [
            (int(span["token_start"]), int(span["token_end"]))
            for span in seed_spans
        ]
        row["labels"] = labels_from_token_spans(len(row["tokens"]), token_spans)
        row["metadata"]["action_seed_span_count"] = len(seed_spans)
        row["metadata"]["protected_seed_span_count"] = sum(
            1 for span in seed_spans if str(span.get("action_label")) == "REDACT"
        )
        row["metadata"]["has_multispan_seed"] = len(seed_spans) >= 2
        row["metadata"]["has_redact_seed"] = row["metadata"]["protected_seed_span_count"] > 0
        proxy_rows.append(row)
    return proxy_rows


def baseline_coverages_by_id(
    rows: Sequence[Mapping[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            continue
        predicted_spans = bio_spans([str(label) for label in prediction.get("predicted_labels", [])])
        seed_spans = list((row.get("metadata") or {}).get("action_seed_spans") or [])
        covered_count = 0
        covered_redact = 0
        misses: list[dict[str, Any]] = []
        for span in seed_spans:
            gold_span = (int(span["token_start"]), int(span["token_end"]))
            covered = any(spans_overlap(gold_span, predicted_span) for predicted_span in predicted_spans)
            if covered:
                covered_count += 1
                if str(span.get("action_label")) == "REDACT":
                    covered_redact += 1
            else:
                misses.append(span)
        coverage[row_id] = {
            "candidate_positive_row": bool(predicted_spans),
            "covered_seed_span_count": covered_count,
            "missed_seed_span_count": len(seed_spans) - covered_count,
            "covered_redact_seed_span_count": covered_redact,
            "missed_redact_seed_span_count": sum(
                1 for span in misses if str(span.get("action_label")) == "REDACT"
            ),
            "missed_seed_spans": misses,
        }
    return coverage


def annotate_baseline_misses(
    rows: Sequence[dict[str, Any]],
    predictions_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    coverages = baseline_coverages_by_id(rows, predictions_by_id)
    annotated: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row["id"])
        row_copy = dict(row)
        metadata = dict(row.get("metadata") or {})
        coverage = coverages.get(row_id)
        if coverage is not None:
            metadata.update(
                {
                    "baseline_candidate_positive_row": bool(coverage["candidate_positive_row"]),
                    "baseline_candidate_covered_seed_span_count": int(coverage["covered_seed_span_count"]),
                    "baseline_candidate_missed_seed_span_count": int(coverage["missed_seed_span_count"]),
                    "baseline_candidate_covered_redact_seed_span_count": int(
                        coverage["covered_redact_seed_span_count"]
                    ),
                    "baseline_candidate_missed_redact_seed_span_count": int(
                        coverage["missed_redact_seed_span_count"]
                    ),
                    "baseline_candidate_missed_action_seed": int(coverage["missed_seed_span_count"]) > 0,
                }
            )
        row_copy["metadata"] = metadata
        annotated.append(row_copy)
    return annotated


def sample_balanced_proxy_splits(
    rows: Sequence[dict[str, Any]],
    *,
    counts_by_split: Mapping[str, int],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    subjects = sorted({str(row.get("subject", "unknown")) for row in rows})
    if not subjects:
        raise ValueError("No candidate proxy rows were available to sample.")

    split_targets_by_subject: dict[str, dict[str, int]] = {
        split: _allocate_evenly(total, subjects)
        for split, total in counts_by_split.items()
    }
    total_targets_by_subject = Counter()
    for split_targets in split_targets_by_subject.values():
        total_targets_by_subject.update(split_targets)

    rows_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_subject[str(row.get("subject", "unknown"))].append(row)

    selected_by_subject: dict[str, list[dict[str, Any]]] = {}
    for subject in subjects:
        subject_rows = rows_by_subject.get(subject, [])
        required_total = int(total_targets_by_subject[subject])
        ordered_rows = sorted(
            subject_rows,
            key=lambda row: (
                -int(bool((row.get("metadata") or {}).get("baseline_candidate_missed_action_seed"))),
                -int(bool((row.get("metadata") or {}).get("has_multispan_seed"))),
                -int(bool((row.get("metadata") or {}).get("has_redact_seed"))),
                stable_sort_key(str(row["id"]), seed=seed),
            ),
        )
        if len(ordered_rows) < required_total:
            raise ValueError(
                f"Subject {subject} has only {len(ordered_rows)} rows but {required_total} were requested."
            )
        selected_by_subject[subject] = ordered_rows[:required_total]

    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in counts_by_split}
    for subject in subjects:
        subject_rows = selected_by_subject[subject]
        assigned_counts: Counter[str] = Counter()
        subject_targets = {
            split: split_targets_by_subject[split][subject]
            for split in counts_by_split
        }
        for row in subject_rows:
            open_splits = [
                split
                for split in counts_by_split
                if assigned_counts[split] < subject_targets[split]
            ]
            if not open_splits:
                break
            split = min(
                open_splits,
                key=lambda value: (
                    assigned_counts[value] / subject_targets[value],
                    -subject_targets[value],
                    value,
                ),
            )
            assigned_counts[split] += 1
            row_copy = dict(row)
            metadata = dict(row.get("metadata") or {})
            metadata["proxy_split"] = split
            row_copy["metadata"] = metadata
            splits[split].append(row_copy)
    return {
        split: sorted(
            split_rows,
            key=lambda row: stable_sort_key(str(row["id"]), seed=seed + index + 1),
        )
        for index, (split, split_rows) in enumerate(splits.items())
    }


def summarize_proxy_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    subject_counts = Counter(str(row.get("subject", "unknown")) for row in row_list)
    speaker_counts = Counter(str(row.get("speaker_role", "unknown")) for row in row_list)
    seed_counts = Counter()
    redact_turns = 0
    multispan_turns = 0
    missed_turns = 0
    for row in row_list:
        metadata = row.get("metadata") or {}
        seed_counts.update({"seed_spans": int(metadata.get("action_seed_span_count", 0))})
        if metadata.get("has_redact_seed"):
            redact_turns += 1
        if metadata.get("has_multispan_seed"):
            multispan_turns += 1
        if metadata.get("baseline_candidate_missed_action_seed"):
            missed_turns += 1
    return {
        "row_count": len(row_list),
        "counts_by_subject": dict(subject_counts),
        "counts_by_speaker_role": dict(speaker_counts),
        "total_seed_spans": int(seed_counts["seed_spans"]),
        "redact_turn_count": redact_turns,
        "multispan_turn_count": multispan_turns,
        "baseline_miss_turn_count": missed_turns,
    }


def _allocate_evenly(total: int, values: Sequence[str]) -> dict[str, int]:
    base = total // len(values)
    remainder = total % len(values)
    allocation: dict[str, int] = {}
    for index, value in enumerate(values):
        allocation[value] = base + (1 if index < remainder else 0)
    return allocation
