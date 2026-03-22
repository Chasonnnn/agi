from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .candidate_adaptation import char_span_to_token_span, labels_from_token_spans, stable_sort_key
from .tokenization import tokenize_with_offsets

UPCHIEVE_SUBJECT = "upchieve_math_ground_truth"
SAGA_SUBJECT = "saga27_math_ground_truth"

MATH_BENCHMARK_CANONICAL_TYPES = (
    "NAME",
    "ADDRESS",
    "URL",
    "SCHOOL",
    "AGE",
    "IDENTIFYING_NUMBER",
    "TUTOR_PROVIDER",
    "DATE",
    "PHONE_NUMBER",
    "IP_ADDRESS",
)
CANONICAL_RARE_TYPES = tuple(label for label in MATH_BENCHMARK_CANONICAL_TYPES if label != "NAME")

PII_TYPE_ALIASES = {
    "NAME": "NAME",
    "PERSON": "NAME",
    "PERSON_NAME": "NAME",
    "ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "OTHER_LOCATIONS_IDENTIFIED": "OTHER_LOCATIONS_IDENTIFIED",
    "DATE": "DATE",
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "FAX_NUMBER": "FAX_NUMBER",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "E_MAIL": "EMAIL",
    "SSN": "SSN",
    "US_SSN": "SSN",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "BANK_ACCOUNT": "ACCOUNT_NUMBER",
    "BANK_NUMBER": "ACCOUNT_NUMBER",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "DEVICE_IDENTIFIER": "DEVICE_IDENTIFIER",
    "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS",
    "BIOMETRIC_IDENTIFIER": "BIOMETRIC_IDENTIFIER",
    "IMAGE": "IMAGE",
    "IDENTIFYING_NUMBER": "IDENTIFYING_NUMBER",
    "MISC_ID": "IDENTIFYING_NUMBER",
    "SOCIAL_HANDLE": "IDENTIFYING_NUMBER",
    "US_DRIVER_LICENSE": "IDENTIFYING_NUMBER",
    "US_PASSPORT": "IDENTIFYING_NUMBER",
    "AGE": "AGE",
    "SCHOOL": "SCHOOL",
    "TUTOR_PROVIDER": "TUTOR_PROVIDER",
    "CUSTOMIZED_FIELD": "CUSTOMIZED_FIELD",
}

UPCHIEVE_SPEAKER_RE = re.compile(r"^(student|volunteer):\s*(.*)$", re.IGNORECASE)
OFFSET_SEARCH_RADIUS = 128


def load_ground_truth_payloads(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not path.is_dir():
        raise ValueError(f"Expected a directory of .ground_truth.json files, found {path}")

    input_files = sorted(path.glob("*.ground_truth.json"))
    if not input_files:
        raise ValueError(f"No .ground_truth.json files found in {path}")

    payloads: list[tuple[Path, dict[str, Any]]] = []
    for file_path in input_files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        transcript = payload.get("transcript")
        if not isinstance(transcript, str):
            raise ValueError(f"{file_path}: expected transcript to be a string")
        payloads.append((file_path, payload))
    return payloads


def canonicalize_pii_type_counts(counts: Mapping[str, Any]) -> dict[str, int]:
    canonical_counts: Counter[str] = Counter()
    for label, count in counts.items():
        canonical_counts[_canonicalize_pii_type(label)] += int(count)
    return dict(sorted(canonical_counts.items()))


def canonicalize_pii_type(value: Any) -> str:
    return _canonicalize_pii_type(value)


def build_upchieve_turn_candidate_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payloads = load_ground_truth_payloads(path)
    rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    for file_path, payload in payloads:
        file_rows, file_summary = _project_upchieve_payload(file_path, payload)
        rows.extend(file_rows)
        file_summaries.append(file_summary)

    summary = {
        "source_dir": str(path),
        "file_count": len(payloads),
        "dialogue_count": len(file_summaries),
        "row_count": len(rows),
        "positive_row_count": sum(1 for row in rows if any(label != "O" for label in row["labels"])),
        "positive_dialogue_count": sum(1 for item in file_summaries if item["positive_turn_count"] > 0),
        "speaker_role_counts": dict(sorted(Counter(str(row.get("speaker_role") or "unknown") for row in rows).items())),
        "pii_type_counts": dict(sorted(_aggregate_pii_type_counts(rows).items())),
        "raw_pii_type_counts": dict(sorted(_aggregate_raw_pii_type_counts(rows).items())),
        "offset_resolution_counts": dict(sorted(_aggregate_offset_resolution_counts(rows).items())),
        "file_summaries": file_summaries,
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
            str(span.get("canonical_pii_type") or span.get("label") or "unknown")
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
    payloads = load_ground_truth_payloads(path)
    rows: list[dict[str, Any]] = []
    file_summaries: list[dict[str, Any]] = []

    for file_path, payload in payloads:
        file_rows, file_summary = _project_saga_payload(file_path, payload)
        rows.extend(file_rows)
        file_summaries.append(file_summary)

    summary = {
        "source_dir": str(path),
        "file_count": len(payloads),
        "row_count": len(rows),
        "positive_row_count": sum(1 for row in rows if any(label != "O" for label in row["labels"])),
        "pii_type_counts": dict(sorted(_aggregate_pii_type_counts(rows).items())),
        "raw_pii_type_counts": dict(sorted(_aggregate_raw_pii_type_counts(rows).items())),
        "offset_resolution_counts": dict(sorted(_aggregate_offset_resolution_counts(rows).items())),
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
        "raw_pii_type_counts": dict(sorted(_aggregate_raw_pii_type_counts(rows_list).items())),
        "offset_resolution_counts": dict(sorted(_aggregate_offset_resolution_counts(rows_list).items())),
    }


def _project_upchieve_payload(path: Path, payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcript = str(payload["transcript"])
    dialogue_id = _payload_dialogue_id(payload, path)
    turns = _parse_upchieve_turns(transcript)
    spans, span_summary = _payload_spans(payload, transcript=transcript, source_file=path)

    spans_by_turn_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        turn_index = _turn_index_for_span(turns, span)
        if turn_index is None:
            raise ValueError(
                f"{path}: could not assign span {span['source_annotation_start']}:{span['source_annotation_end']} "
                f"({span['source_annotation_text']!r}) to a parsed turn"
            )
        spans_by_turn_index[turn_index].append(span)

    rows: list[dict[str, Any]] = []
    positive_turn_count = 0
    for turn_index, turn in enumerate(turns):
        turn_text = str(turn["text"])
        if not turn_text.strip():
            continue

        tokenized = tokenize_with_offsets(turn_text)
        if not tokenized:
            continue
        tokens = [token for token, _ in tokenized]
        token_offsets = [span for _, span in tokenized]

        gold_spans: list[dict[str, Any]] = []
        for span in spans_by_turn_index.get(turn_index, []):
            local_start = int(span["start"]) - int(turn["start"])
            local_end = int(span["end"]) - int(turn["start"])
            token_span = char_span_to_token_span(token_offsets, local_start, local_end)
            if token_span is None:
                raise ValueError(
                    f"{path}: could not project UpChieve char span {span['start']}:{span['end']} "
                    f"onto turn {turn_index} for dialogue={dialogue_id}"
                )
            gold_spans.append(
                {
                    "char_start": local_start,
                    "char_end": local_end,
                    "token_start": token_span[0],
                    "token_end": token_span[1],
                    "text": span["text"],
                    "label": span["label"],
                    "raw_pii_type": span["raw_pii_type"],
                    "canonical_pii_type": span["label"],
                    "offset_resolution": span["offset_resolution"],
                    "source_annotation_start": span["source_annotation_start"],
                    "source_annotation_end": span["source_annotation_end"],
                    "source_annotation_text": span["source_annotation_text"],
                    "transcript_char_start": span["start"],
                    "transcript_char_end": span["end"],
                }
            )

        if gold_spans:
            positive_turn_count += 1

        neighbor_context = _upchieve_neighbor_turns(turns, turn_index)
        labels = labels_from_token_spans(
            len(tokens),
            [(int(span["token_start"]), int(span["token_end"])) for span in gold_spans],
        )
        row_id = f"{dialogue_id}-turn-{turn_index}"
        pii_types = sorted({str(span["canonical_pii_type"]) for span in gold_spans})
        raw_pii_types = sorted({str(span["raw_pii_type"]) for span in gold_spans})
        rows.append(
            {
                "id": row_id,
                "subject": UPCHIEVE_SUBJECT,
                "tokens": tokens,
                "labels": labels,
                "dialogue_id": dialogue_id,
                "speaker_role": str(turn["role"]),
                "context_text": _render_upchieve_neighbor_context(neighbor_context),
                "metadata": {
                    "source": "upchieve_math_ground_truth_json_dir",
                    "source_file": path.name,
                    "ground_truth_source": payload.get("ground_truth_source"),
                    "raw_filename": payload.get("filename"),
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "turn_start": int(turn["start"]),
                    "turn_end": int(turn["end"]),
                    "raw_text": turn_text,
                    "current_raw_text": turn_text,
                    "prev_turn_text": neighbor_context["previous"]["text"],
                    "next_turn_text": neighbor_context["next"]["text"],
                    "prev_speaker_role": neighbor_context["previous"]["role"],
                    "next_speaker_role": neighbor_context["next"]["role"],
                    "gold_spans": gold_spans,
                    "pii_types": pii_types,
                    "raw_pii_types": raw_pii_types,
                    "has_positive_label": any(label != "O" for label in labels),
                },
            }
        )

    summary = {
        "dialogue_id": dialogue_id,
        "source_file": path.name,
        "ground_truth_source": payload.get("ground_truth_source"),
        "turn_count": len(turns),
        "positive_turn_count": positive_turn_count,
        "pii_type_counts": dict(sorted(span_summary["canonical_counts"].items())),
        "raw_pii_type_counts": dict(sorted(span_summary["raw_counts"].items())),
        "offset_resolution_counts": dict(sorted(span_summary["offset_counts"].items())),
        "transcript_char_count": len(transcript),
    }
    return rows, summary


def _project_saga_payload(path: Path, payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcript = str(payload["transcript"])
    dialogue_id = _payload_dialogue_id(payload, path)
    spans, span_summary = _payload_spans(payload, transcript=transcript, source_file=path)
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
                    "label": span["label"],
                    "raw_pii_type": span["raw_pii_type"],
                    "canonical_pii_type": span["label"],
                    "offset_resolution": span["offset_resolution"],
                    "source_annotation_start": span["source_annotation_start"],
                    "source_annotation_end": span["source_annotation_end"],
                    "source_annotation_text": span["source_annotation_text"],
                    "transcript_char_start": span_start,
                    "transcript_char_end": span_end,
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
                    "raw_filename": payload.get("filename"),
                    "raw_text": segment_text,
                    "current_raw_text": segment_text,
                    "prev_turn_text": _neighboring_segment_text(segments, segment_index, direction="previous"),
                    "next_turn_text": _neighboring_segment_text(segments, segment_index, direction="next"),
                    "prev_speaker_role": None,
                    "next_speaker_role": None,
                    "segment_index": segment_index,
                    "start_line_index": start_line,
                    "end_line_index": end_line,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "gold_spans": local_gold_spans,
                    "pii_types": sorted({str(span["canonical_pii_type"]) for span in local_gold_spans}),
                    "raw_pii_types": sorted({str(span["raw_pii_type"]) for span in local_gold_spans}),
                    "has_positive_label": any(label != "O" for label in labels),
                },
            }
        )

    if assigned_span_count != len(spans):
        raise ValueError(f"{path}: projected {assigned_span_count} spans onto rows, expected {len(spans)}")

    summary = {
        "dialogue_id": dialogue_id,
        "source_file": path.name,
        "ground_truth_source": payload.get("ground_truth_source"),
        "segment_count": len(rows),
        "positive_segment_count": positive_row_count,
        "span_count": len(spans),
        "transcript_char_count": len(transcript),
        "pii_type_counts": dict(sorted(span_summary["canonical_counts"].items())),
        "raw_pii_type_counts": dict(sorted(span_summary["raw_counts"].items())),
        "offset_resolution_counts": dict(sorted(span_summary["offset_counts"].items())),
    }
    return rows, summary


def _payload_dialogue_id(payload: Mapping[str, Any], path: Path) -> str:
    return str(payload.get("id") or path.stem)


def _payload_spans(
    payload: Mapping[str, Any],
    *,
    transcript: str,
    source_file: Path,
) -> tuple[list[dict[str, Any]], dict[str, Counter[str]]]:
    raw_occurrences = list(payload.get("pii_occurrences") or payload.get("spans") or [])
    spans: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    offset_counts: Counter[str] = Counter()

    for item in raw_occurrences:
        source_annotation_start = int(item["start"])
        source_annotation_end = int(item["end"])
        source_annotation_text = str(item["text"])
        raw_pii_type = str(item.get("pii_type") or item.get("label") or "SUSPECT")
        canonical_pii_type = _canonicalize_pii_type(raw_pii_type)
        resolved_start, resolved_end, offset_resolution = _resolve_annotation_span(
            transcript=transcript,
            source_file=source_file,
            source_annotation_start=source_annotation_start,
            source_annotation_end=source_annotation_end,
            source_annotation_text=source_annotation_text,
        )
        raw_counts[raw_pii_type] += 1
        canonical_counts[canonical_pii_type] += 1
        offset_counts[offset_resolution] += 1
        spans.append(
            {
                "start": resolved_start,
                "end": resolved_end,
                "text": transcript[resolved_start:resolved_end],
                "label": canonical_pii_type,
                "raw_pii_type": raw_pii_type,
                "offset_resolution": offset_resolution,
                "source_annotation_start": source_annotation_start,
                "source_annotation_end": source_annotation_end,
                "source_annotation_text": source_annotation_text,
            }
        )

    spans.sort(key=lambda span: (int(span["start"]), int(span["end"]), str(span["text"])))
    return spans, {
        "raw_counts": raw_counts,
        "canonical_counts": canonical_counts,
        "offset_counts": offset_counts,
    }


def _resolve_annotation_span(
    *,
    transcript: str,
    source_file: Path,
    source_annotation_start: int,
    source_annotation_end: int,
    source_annotation_text: str,
) -> tuple[int, int, str]:
    if source_annotation_end < source_annotation_start:
        raise ValueError(
            f"{source_file}: invalid annotation range {source_annotation_start}:{source_annotation_end} "
            f"for {source_annotation_text!r}"
        )

    exact_slice = transcript[source_annotation_start:source_annotation_end]
    if exact_slice == source_annotation_text:
        return source_annotation_start, source_annotation_end, "exact"

    inclusive_end = min(source_annotation_end + 1, len(transcript))
    if transcript[source_annotation_start:inclusive_end] == source_annotation_text:
        return source_annotation_start, inclusive_end, "inclusive_end"

    trimmed_exact = _trimmed_match(
        transcript,
        start=source_annotation_start,
        end=source_annotation_end,
        target=source_annotation_text,
    )
    if trimmed_exact is not None:
        return trimmed_exact[0], trimmed_exact[1], "trimmed"

    trimmed_inclusive = _trimmed_match(
        transcript,
        start=source_annotation_start,
        end=inclusive_end,
        target=source_annotation_text,
    )
    if trimmed_inclusive is not None:
        return trimmed_inclusive[0], trimmed_inclusive[1], "trimmed_inclusive_end"

    if source_annotation_start + 1 <= len(transcript):
        shifted_end = min(source_annotation_end + 1, len(transcript))
        if transcript[source_annotation_start + 1 : shifted_end] == source_annotation_text:
            return source_annotation_start + 1, shifted_end, "shift_right"

    local_match = _local_exact_search(
        transcript,
        start=source_annotation_start,
        end=source_annotation_end,
        target=source_annotation_text,
        radius=OFFSET_SEARCH_RADIUS,
    )
    if local_match is not None:
        return local_match[0], local_match[1], "local_search"

    raise ValueError(
        f"{source_file}: could not resolve annotation range {source_annotation_start}:{source_annotation_end} "
        f"for {source_annotation_text!r}"
    )


def _trimmed_match(transcript: str, *, start: int, end: int, target: str) -> tuple[int, int] | None:
    fragment = transcript[start:end]
    if fragment.strip() != target:
        return None
    local_start = 0
    local_end = len(fragment)
    while local_start < local_end and fragment[local_start].isspace():
        local_start += 1
    while local_end > local_start and fragment[local_end - 1].isspace():
        local_end -= 1
    resolved_start = start + local_start
    resolved_end = start + local_end
    if transcript[resolved_start:resolved_end] == target:
        return resolved_start, resolved_end
    return None


def _local_exact_search(
    transcript: str,
    *,
    start: int,
    end: int,
    target: str,
    radius: int,
) -> tuple[int, int] | None:
    if not target:
        return None
    search_start = max(0, start - radius)
    search_end = min(len(transcript), max(end, start) + radius + len(target))
    window = transcript[search_start:search_end]
    candidates: list[int] = []
    cursor = window.find(target)
    while cursor != -1:
        candidates.append(search_start + cursor)
        cursor = window.find(target, cursor + 1)
    if not candidates:
        return None
    best_start = min(candidates, key=lambda candidate_start: (abs(candidate_start - start), candidate_start))
    return best_start, best_start + len(target)


def _parse_upchieve_turns(transcript: str) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line_start, content in _transcript_lines(transcript):
        line_end = line_start + len(content)
        match = UPCHIEVE_SPEAKER_RE.match(content)
        if match:
            if current is not None:
                turns.append(current)
            role = match.group(1).lower()
            turn_text = match.group(2)
            text_start = line_start + match.start(2)
            current = {
                "role": role,
                "start": text_start,
                "end": line_end,
                "text": turn_text,
            }
            continue

        if current is None:
            current = {
                "role": "unknown",
                "start": line_start,
                "end": line_end,
                "text": content,
            }
            continue

        current["text"] = f"{current['text']}\n{content}"
        current["end"] = line_end

    if current is not None:
        turns.append(current)
    return turns


def _turn_index_for_span(turns: Sequence[Mapping[str, Any]], span: Mapping[str, Any]) -> int | None:
    span_start = int(span["start"])
    span_end = int(span["end"])
    for turn_index, turn in enumerate(turns):
        if int(turn["start"]) <= span_start and span_end <= int(turn["end"]):
            return turn_index
    return None


def _aggregate_pii_type_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for span in list((row.get("metadata") or {}).get("gold_spans") or []):
            counts[str(span.get("canonical_pii_type") or span.get("label") or "unknown")] += 1
    return counts


def _aggregate_raw_pii_type_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for span in list((row.get("metadata") or {}).get("gold_spans") or []):
            counts[str(span.get("raw_pii_type") or "unknown")] += 1
    return counts


def _aggregate_offset_resolution_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for span in list((row.get("metadata") or {}).get("gold_spans") or []):
            counts[str(span.get("offset_resolution") or "unknown")] += 1
    return counts


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_turn(role: str | None, text: str) -> str:
    normalized_role = str(role or "unknown")
    return f"{normalized_role}: {text}"


def _upchieve_neighbor_turns(
    turns: Sequence[Mapping[str, Any]],
    turn_index: int,
) -> dict[str, dict[str, str | None]]:
    previous = {"text": None, "role": None}
    previous_index = turn_index - 1
    while previous_index >= 0:
        previous_text = str(turns[previous_index].get("text") or "")
        if previous_text.strip():
            previous = {
                "text": previous_text,
                "role": str(turns[previous_index].get("role") or "unknown"),
            }
            break
        previous_index -= 1

    current_text = str(turns[turn_index].get("text") or "")
    current = {
        "text": current_text,
        "role": str(turns[turn_index].get("role") or "unknown"),
    }

    next_turn = {"text": None, "role": None}
    next_index = turn_index + 1
    while next_index < len(turns):
        next_text = str(turns[next_index].get("text") or "")
        if next_text.strip():
            next_turn = {
                "text": next_text,
                "role": str(turns[next_index].get("role") or "unknown"),
            }
            break
        next_index += 1

    return {
        "previous": previous,
        "current": current,
        "next": next_turn,
    }


def _render_upchieve_neighbor_context(neighbor_context: Mapping[str, Mapping[str, str | None]]) -> str:
    snippets: list[str] = []
    previous_text = neighbor_context["previous"].get("text")
    if previous_text:
        snippets.append(_format_turn(neighbor_context["previous"].get("role"), previous_text))

    current_text = str(neighbor_context["current"].get("text") or "")
    snippets.append(_format_turn(neighbor_context["current"].get("role"), current_text))

    next_text = neighbor_context["next"].get("text")
    if next_text:
        snippets.append(_format_turn(neighbor_context["next"].get("role"), next_text))
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
    previous_text = _neighboring_segment_text(segments, index, direction="previous")
    if previous_text:
        snippets.append(previous_text)

    snippets.append(segments[index][3])

    next_text = _neighboring_segment_text(segments, index, direction="next")
    if next_text:
        snippets.append(next_text)
    return "\n".join(snippets)


def _neighboring_segment_text(
    segments: Sequence[tuple[int, int, int, str]],
    index: int,
    *,
    direction: str,
) -> str | None:
    step = -1 if direction == "previous" else 1
    cursor = index + step
    while 0 <= cursor < len(segments):
        text = segments[cursor][3]
        if text.strip():
            return text
        cursor += step
    return None
