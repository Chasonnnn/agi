from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_adaptation import char_span_to_token_span, labels_from_token_spans, stable_sort_key
from .tokenization import tokenize_with_offsets

UPCHIEVE_SUBJECT = "upchieve_math_ground_truth"
SAGA_SUBJECT = "saga27_math_ground_truth"
CANONICAL_RARE_TYPES = (
    "AGE",
    "COURSE",
    "DATE",
    "EMAIL_ADDRESS",
    "GRADE_LEVEL",
    "IP_ADDRESS",
    "LOCATION",
    "MISC_ID",
    "NRP",
    "PHONE_NUMBER",
    "SCHOOL",
    "SOCIAL_HANDLE",
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "US_SSN",
)
PII_TYPE_ALIASES = {
    "NAME": "PERSON",
    "PERSON_NAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "E_MAIL": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUM": "PHONE_NUMBER",
    "HANDLE": "SOCIAL_HANDLE",
    "SCREEN_NAME": "SOCIAL_HANDLE",
    "USERNAME": "SOCIAL_HANDLE",
    "SSN": "US_SSN",
    "SOCIAL_SECURITY_NUMBER": "US_SSN",
    "PASSPORT": "US_PASSPORT",
    "PASSPORT_NUMBER": "US_PASSPORT",
    "BANK_ACCOUNT": "US_BANK_NUMBER",
    "BANK_NUMBER": "US_BANK_NUMBER",
    "ACCOUNT_NUMBER": "US_BANK_NUMBER",
    "DRIVER_LICENSE": "US_DRIVER_LICENSE",
    "DRIVERS_LICENSE": "US_DRIVER_LICENSE",
    "LICENSE_NUMBER": "US_DRIVER_LICENSE",
}


def load_upchieve_ground_truth_dialogues(path: Path) -> list[dict[str, Any]]:
    dialogues: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            dialogues.append(payload)
    return dialogues


def build_upchieve_turn_candidate_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dialogues = load_upchieve_ground_truth_dialogues(path)
    rows: list[dict[str, Any]] = []
    dialogue_summaries: list[dict[str, Any]] = []

    for dialogue_index, dialogue in enumerate(dialogues):
        transcript = list(dialogue.get("transcript") or [])
        if not transcript:
            continue
        dialogue_id = _upchieve_dialogue_id(transcript, dialogue_index)
        lead_role = str(dialogue.get("leadRole") or "unknown")
        positive_turn_count = 0
        pii_counts = Counter()

        for turn_index, turn in enumerate(transcript):
            turn_text = str(turn.get("content") or "")
            if not turn_text.strip():
                continue
            tokenized = tokenize_with_offsets(turn_text)
            if not tokenized:
                continue
            tokens = [token for token, _ in tokenized]
            token_offsets = [span for _, span in tokenized]
            raw_annotations = list(turn.get("annotations") or [])
            gold_spans: list[dict[str, Any]] = []

            for annotation in raw_annotations:
                span_start = int(annotation["start"])
                span_end = int(annotation["end"])
                token_span = char_span_to_token_span(token_offsets, span_start, span_end)
                if token_span is None:
                    raise ValueError(
                        f"{path}: could not project UpChieve char span {span_start}:{span_end} "
                        f"onto tokens for dialogue={dialogue_id} turn={turn_index}"
                    )
                span_label = _canonicalize_pii_type(annotation.get("pii_type") or "SUSPECT")
                pii_counts[span_label] += 1
                gold_spans.append(
                    {
                        "char_start": span_start,
                        "char_end": span_end,
                        "token_start": token_span[0],
                        "token_end": token_span[1],
                        "text": turn_text[span_start:span_end],
                        "label": span_label,
                        "surrogate": annotation.get("surrogate"),
                    }
                )

            if gold_spans:
                positive_turn_count += 1

            labels = labels_from_token_spans(
                len(tokens),
                [(int(span["token_start"]), int(span["token_end"])) for span in gold_spans],
            )
            context_text = _upchieve_neighbor_context(transcript, turn_index)
            pii_types = sorted({str(span["label"]) for span in gold_spans})
            row_id = f"{dialogue_id}-turn-{_safe_int(turn.get('sequence_id'), default=turn_index)}"
            rows.append(
                {
                    "id": row_id,
                    "subject": UPCHIEVE_SUBJECT,
                    "tokens": tokens,
                    "labels": labels,
                    "dialogue_id": dialogue_id,
                    "speaker_role": str(turn.get("role") or "unknown"),
                    "context_text": context_text,
                    "metadata": {
                        "source": "upchieve_math_ground_truth_jsonl",
                        "source_file": path.name,
                        "dialogue_index": dialogue_index,
                        "lead_role": lead_role,
                        "turn_index": turn_index,
                        "turn_sequence_id": _safe_int(turn.get("sequence_id"), default=turn_index),
                        "turn_id": turn.get("_id"),
                        "raw_text": turn_text,
                        "gold_spans": gold_spans,
                        "pii_types": pii_types,
                        "has_positive_label": any(label != "O" for label in labels),
                    },
                }
            )

        dialogue_summaries.append(
            {
                "dialogue_id": dialogue_id,
                "lead_role": lead_role,
                "turn_count": len(transcript),
                "positive_turn_count": positive_turn_count,
                "pii_type_counts": dict(sorted(pii_counts.items())),
                "rare_type_set": sorted(label for label in pii_counts if label in CANONICAL_RARE_TYPES),
            }
        )

    summary = {
        "source_file": str(path),
        "dialogue_count": len(dialogue_summaries),
        "row_count": len(rows),
        "positive_row_count": sum(1 for row in rows if any(label != "O" for label in row["labels"])),
        "positive_dialogue_count": sum(1 for item in dialogue_summaries if item["positive_turn_count"] > 0),
        "speaker_role_counts": dict(sorted(Counter(str(row.get("speaker_role") or "unknown") for row in rows).items())),
        "pii_type_counts": dict(sorted(_aggregate_pii_type_counts(rows).items())),
        "dialogue_summaries": dialogue_summaries,
    }
    return sorted(rows, key=lambda row: str(row["id"])), summary


def split_upchieve_dialogues(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_dialogues: int,
    dev_dialogues: int,
    test_dialogues: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    target_dialogue_counts = {
        "train": int(train_dialogues),
        "dev": int(dev_dialogues),
        "test": int(test_dialogues),
    }
    if sum(target_dialogue_counts.values()) <= 0:
        raise ValueError("Dialogue split sizes must sum to a positive value.")

    rows_by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dialogue_id = row.get("dialogue_id")
        if dialogue_id is None:
            raise ValueError(f"Missing dialogue_id for row {row.get('id')}")
        rows_by_dialogue[str(dialogue_id)].append(dict(row))

    if len(rows_by_dialogue) != sum(target_dialogue_counts.values()):
        raise ValueError(
            f"Expected {sum(target_dialogue_counts.values())} dialogues but found {len(rows_by_dialogue)}."
        )

    dialogue_infos: list[dict[str, Any]] = []
    positive_dialogue_count = 0
    rare_dialogue_counts = Counter()
    for dialogue_id, dialogue_rows in rows_by_dialogue.items():
        pii_types = {
            str(span.get("label") or "unknown")
            for row in dialogue_rows
            for span in list((row.get("metadata") or {}).get("gold_spans") or [])
        }
        rare_types = {label for label in pii_types if label in CANONICAL_RARE_TYPES}
        has_positive = any(any(label != "O" for label in row["labels"]) for row in dialogue_rows)
        positive_dialogue_count += int(has_positive)
        rare_dialogue_counts.update(rare_types)
        dialogue_infos.append(
            {
                "dialogue_id": dialogue_id,
                "rows": dialogue_rows,
                "has_positive": has_positive,
                "rare_types": rare_types,
                "pii_types": pii_types,
            }
        )

    target_positive_counts = _allocate_counts(positive_dialogue_count, target_dialogue_counts)
    target_rare_counts = {
        label: _allocate_counts(count, target_dialogue_counts)
        for label, count in rare_dialogue_counts.items()
    }

    ordered_infos = sorted(
        dialogue_infos,
        key=lambda info: (
            -len(info["rare_types"]),
            -int(info["has_positive"]),
            stable_sort_key(str(info["dialogue_id"]), seed=seed),
        ),
    )

    split_infos: dict[str, list[dict[str, Any]]] = {split: [] for split in target_dialogue_counts}
    split_dialogue_counts = Counter()
    split_positive_counts = Counter()
    split_rare_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for info in ordered_infos:
        split = min(
            (
                split_name
                for split_name in target_dialogue_counts
                if split_dialogue_counts[split_name] < target_dialogue_counts[split_name]
            ),
            key=lambda split_name: _split_score(
                split_name=split_name,
                info=info,
                split_dialogue_counts=split_dialogue_counts,
                split_positive_counts=split_positive_counts,
                split_rare_counts=split_rare_counts,
                target_dialogue_counts=target_dialogue_counts,
                target_positive_counts=target_positive_counts,
                target_rare_counts=target_rare_counts,
            ),
        )
        split_infos[split].append(info)
        split_dialogue_counts[split] += 1
        split_positive_counts[split] += int(info["has_positive"])
        split_rare_counts[split].update(info["rare_types"])

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_summary: dict[str, Any] = {"dialogue_counts": dict(split_dialogue_counts)}
    for split_name, infos in split_infos.items():
        rows_for_split: list[dict[str, Any]] = []
        for info in infos:
            for row in info["rows"]:
                row_copy = dict(row)
                metadata = dict(row.get("metadata") or {})
                metadata["benchmark_split"] = split_name
                row_copy["metadata"] = metadata
                rows_for_split.append(row_copy)
        split_rows[split_name] = sorted(rows_for_split, key=lambda row: stable_sort_key(str(row["id"]), seed=seed + 1))
        split_summary[split_name] = {
            "dialogue_count": len(infos),
            "row_count": len(rows_for_split),
            "positive_dialogue_count": sum(1 for info in infos if info["has_positive"]),
            "rare_type_dialogue_counts": dict(sorted(split_rare_counts[split_name].items())),
            "pii_type_counts": dict(sorted(_aggregate_pii_type_counts(rows_for_split).items())),
        }
    return split_rows, split_summary


def build_saga_segment_candidate_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_files = sorted(path.glob("*.ground_truth.json"))
    if not input_files:
        raise ValueError(f"No .ground_truth.json files found in {path}")

    rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []
    for file_path in input_files:
        file_rows, file_summary = _project_saga_file(file_path)
        rows.extend(file_rows)
        file_summaries.append(file_summary)

    summary = {
        "source_dir": str(path),
        "file_count": len(input_files),
        "row_count": len(rows),
        "positive_row_count": sum(1 for row in rows if any(label != "O" for label in row["labels"])),
        "pii_type_counts": dict(sorted(_aggregate_pii_type_counts(rows).items())),
        "file_summaries": file_summaries,
    }
    return sorted(rows, key=lambda row: str(row["id"])), summary


def summarize_candidate_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows_list = [dict(row) for row in rows]
    return {
        "row_count": len(rows_list),
        "dialogue_count": len({str(row.get("dialogue_id")) for row in rows_list}),
        "positive_row_count": sum(1 for row in rows_list if any(label != "O" for label in row["labels"])),
        "token_count": sum(len(row["tokens"]) for row in rows_list),
        "speaker_role_counts": dict(
            sorted(Counter(str(row.get("speaker_role") or "unknown") for row in rows_list).items())
        ),
        "pii_type_counts": dict(sorted(_aggregate_pii_type_counts(rows_list).items())),
    }


def _aggregate_pii_type_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for span in list((row.get("metadata") or {}).get("gold_spans") or []):
            counts[str(span.get("label") or "unknown")] += 1
    return counts


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _upchieve_dialogue_id(transcript: Sequence[Mapping[str, Any]], dialogue_index: int) -> str:
    for turn in transcript:
        session_id = turn.get("session_id")
        if session_id is not None:
            return str(session_id)
    return f"upchieve-dialogue-{dialogue_index:04d}"


def _format_turn(role: str | None, text: str) -> str:
    normalized_role = str(role or "unknown")
    return f"{normalized_role}: {text}"


def _upchieve_neighbor_context(transcript: Sequence[Mapping[str, Any]], turn_index: int) -> str:
    snippets: list[str] = []
    previous_index = turn_index - 1
    while previous_index >= 0:
        previous_text = str(transcript[previous_index].get("content") or "")
        if previous_text.strip():
            snippets.append(_format_turn(transcript[previous_index].get("role"), previous_text))
            break
        previous_index -= 1

    current_text = str(transcript[turn_index].get("content") or "")
    snippets.append(_format_turn(transcript[turn_index].get("role"), current_text))

    next_index = turn_index + 1
    while next_index < len(transcript):
        next_text = str(transcript[next_index].get("content") or "")
        if next_text.strip():
            snippets.append(_format_turn(transcript[next_index].get("role"), next_text))
            break
        next_index += 1
    return "\n".join(snippets)


def _allocate_counts(total: int, split_counts: Mapping[str, int]) -> dict[str, int]:
    total_weight = sum(int(value) for value in split_counts.values())
    if total_weight <= 0:
        raise ValueError("Split weights must sum to a positive value.")
    base = {
        split: (total * int(weight)) // total_weight
        for split, weight in split_counts.items()
    }
    remainder = total - sum(base.values())
    if remainder <= 0:
        return base
    ranked_splits = sorted(split_counts, key=lambda split: (-int(split_counts[split]), split))
    for index in range(remainder):
        base[ranked_splits[index % len(ranked_splits)]] += 1
    return base


def _split_score(
    *,
    split_name: str,
    info: Mapping[str, Any],
    split_dialogue_counts: Mapping[str, int],
    split_positive_counts: Mapping[str, int],
    split_rare_counts: Mapping[str, Counter[str]],
    target_dialogue_counts: Mapping[str, int],
    target_positive_counts: Mapping[str, int],
    target_rare_counts: Mapping[str, Mapping[str, int]],
) -> tuple[float, float, float, int, str]:
    dialogue_fill = split_dialogue_counts[split_name] / target_dialogue_counts[split_name]
    rare_deficit = sum(
        max(target_rare_counts[label][split_name] - split_rare_counts[split_name][label], 0)
        for label in info["rare_types"]
    )
    positive_deficit = 0
    if info["has_positive"]:
        positive_deficit = max(target_positive_counts[split_name] - split_positive_counts[split_name], 0)
    return (
        dialogue_fill,
        -float(rare_deficit),
        -float(positive_deficit),
        split_dialogue_counts[split_name],
        split_name,
    )


def _project_saga_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transcript = str(payload["transcript"])
    dialogue_id = str(payload["id"])
    spans = _saga_spans(payload)
    lines = _transcript_lines(transcript)
    segment_ranges = _segment_ranges_for_spans(lines, spans)
    segments: list[tuple[int, int, int, str]] = []
    for start_line, end_line in segment_ranges:
        start_offset = lines[start_line][0]
        final_offset, final_text = lines[end_line]
        end_offset = final_offset + len(final_text)
        segments.append((start_line, end_line, start_offset, transcript[start_offset:end_offset]))

    rows: list[dict[str, Any]] = []
    assigned_span_count = 0
    positive_row_count = 0
    for segment_index, (start_line, end_line, segment_start, segment_text) in enumerate(segments):
        if not segment_text.strip():
            continue
        tokenized = tokenize_with_offsets(segment_text)
        if not tokenized:
            continue
        tokens = [token for token, _ in tokenized]
        token_offsets = [span for _, span in tokenized]
        segment_end = segment_start + len(segment_text)
        local_gold_spans: list[dict[str, Any]] = []
        for span in spans:
            span_start = int(span["start"])
            span_end = int(span["end"])
            if not (segment_start <= span_start and span_end <= segment_end):
                continue
            token_span = char_span_to_token_span(token_offsets, span_start - segment_start, span_end - segment_start)
            if token_span is None:
                raise ValueError(
                    f"{path}: could not project Saga char span {span_start}:{span_end} "
                    f"onto segment {segment_index}"
                )
            local_gold_spans.append(
                {
                    "char_start": span_start - segment_start,
                    "char_end": span_end - segment_start,
                    "token_start": token_span[0],
                    "token_end": token_span[1],
                    "text": span["text"],
                    "label": _canonicalize_pii_type(span["label"]),
                }
            )

        assigned_span_count += len(local_gold_spans)
        if local_gold_spans:
            positive_row_count += 1
        labels = labels_from_token_spans(
            len(tokens),
            [(int(span["token_start"]), int(span["token_end"])) for span in local_gold_spans],
        )
        rows.append(
            {
                "id": f"{dialogue_id}-segment-{segment_index}",
                "subject": SAGA_SUBJECT,
                "tokens": tokens,
                "labels": labels,
                "dialogue_id": dialogue_id,
                "speaker_role": None,
                "context_text": _neighboring_nonempty_segments(segments, segment_index),
                "metadata": {
                    "source": "saga27_ground_truth_json_dir",
                    "source_file": path.name,
                    "ground_truth_source": payload.get("ground_truth_source"),
                    "raw_text": segment_text,
                    "segment_index": segment_index,
                    "start_line_index": start_line,
                    "end_line_index": end_line,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "gold_spans": local_gold_spans,
                    "pii_types": sorted({str(span["label"]) for span in local_gold_spans}),
                    "has_positive_label": any(label != "O" for label in labels),
                },
            }
        )

    if assigned_span_count != len(spans):
        raise ValueError(f"{path}: projected {assigned_span_count} spans onto rows, expected {len(spans)}")

    summary = {
        "dialogue_id": dialogue_id,
        "source_file": path.name,
        "segment_count": len(rows),
        "positive_segment_count": positive_row_count,
        "span_count": len(spans),
        "transcript_char_count": len(transcript),
    }
    return rows, summary


def _saga_spans(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("pii_occurrences"):
        return [
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "text": str(item["text"]),
                "label": _canonicalize_pii_type(item.get("pii_type") or "SUSPECT"),
            }
            for item in payload["pii_occurrences"]
        ]
    return [
        {
            "start": int(item["start"]),
            "end": int(item["end"]),
            "text": str(item["text"]),
            "label": _canonicalize_pii_type(item.get("label") or "SUSPECT"),
        }
        for item in payload.get("spans", [])
    ]


def _canonicalize_pii_type(value: Any) -> str:
    raw_value = str(value or "SUSPECT").strip()
    if not raw_value:
        return "SUSPECT"
    normalized = raw_value.upper().replace("-", "_").replace(" ", "_")
    return PII_TYPE_ALIASES.get(normalized, normalized)


def _transcript_lines(transcript: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    offset = 0
    for raw_line in transcript.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        lines.append((offset, content))
        offset += len(raw_line)
    if not transcript:
        lines.append((0, ""))
    return lines


def _line_index_for_offset(lines: Sequence[tuple[int, str]], offset: int) -> int:
    for index, (line_start, line_text) in enumerate(lines):
        line_end = line_start + len(line_text)
        if line_start <= offset < line_end:
            return index
        if offset == line_end and index == len(lines) - 1:
            return index
    raise ValueError(f"Could not map char offset {offset} onto transcript lines")


def _segment_ranges_for_spans(
    lines: Sequence[tuple[int, str]],
    spans: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int]]:
    merged_ranges: list[list[int]] = []
    for span in sorted(spans, key=lambda item: (int(item["start"]), int(item["end"]))):
        start = int(span["start"])
        end = int(span["end"])
        if end <= start:
            raise ValueError(f"Invalid span with end <= start: {span}")
        start_line = _line_index_for_offset(lines, start)
        end_line = _line_index_for_offset(lines, end - 1)
        if not merged_ranges or start_line > merged_ranges[-1][1] + 1:
            merged_ranges.append([start_line, end_line])
        else:
            merged_ranges[-1][1] = max(merged_ranges[-1][1], end_line)

    segment_ranges: list[tuple[int, int]] = []
    cursor = 0
    for start_line, end_line in merged_ranges:
        while cursor < start_line:
            segment_ranges.append((cursor, cursor))
            cursor += 1
        segment_ranges.append((start_line, end_line))
        cursor = end_line + 1
    while cursor < len(lines):
        segment_ranges.append((cursor, cursor))
        cursor += 1
    return segment_ranges


def _neighboring_nonempty_segments(
    segments: Sequence[tuple[int, int, int, str]],
    index: int,
) -> str:
    snippets: list[str] = []
    previous_index = index - 1
    while previous_index >= 0:
        previous_text = segments[previous_index][3]
        if previous_text.strip():
            snippets.append(previous_text)
            break
        previous_index -= 1

    snippets.append(segments[index][3])

    next_index = index + 1
    while next_index < len(segments):
        next_text = segments[next_index][3]
        if next_text.strip():
            snippets.append(next_text)
            break
        next_index += 1
    return "\n".join(snippets)
