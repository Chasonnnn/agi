from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schemas import CandidateRecord
from .tokenization import tokenize_text

INPUT_FORMATS = (
    "turn_only_v1",
    "pair_context_v1",
    "marked_window_v1",
    "speaker_marked_window_v1",
)
LEGACY_CONTEXT_MODE_TO_INPUT_FORMAT = {
    "none": "turn_only_v1",
    "pair": "pair_context_v1",
}
IGNORE_WORD_LABEL = "__IGNORE__"
CURRENT_TURN_START = "<CUR>"
CURRENT_TURN_END = "</CUR>"


def resolve_candidate_input_format(
    *,
    input_format: str | None,
    context_mode: str | None = None,
) -> str:
    if input_format is not None:
        if input_format not in INPUT_FORMATS:
            raise ValueError(f"Unsupported candidate input format: {input_format}")
        return input_format
    normalized_context_mode = str(context_mode or "none")
    try:
        return LEGACY_CONTEXT_MODE_TO_INPUT_FORMAT[normalized_context_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported candidate context mode: {normalized_context_mode}") from exc


def build_candidate_model_inputs(
    records: Sequence[CandidateRecord],
    *,
    input_format: str,
) -> list[dict[str, Any]]:
    normalized_format = resolve_candidate_input_format(input_format=input_format)
    return [build_candidate_model_input(record, input_format=normalized_format) for record in records]


def build_candidate_model_input(
    record: CandidateRecord,
    *,
    input_format: str,
) -> dict[str, Any]:
    normalized_format = resolve_candidate_input_format(input_format=input_format)
    current_tokens = list(record.tokens)
    current_labels = list(record.labels)
    current_token_count = len(current_tokens)
    if current_token_count != len(current_labels):
        raise ValueError(f"Candidate row {record.id} has mismatched token/label lengths")

    if normalized_format == "turn_only_v1":
        return {
            "id": record.id,
            "tokens": current_tokens,
            "labels": current_labels,
            "subject": record.subject,
            "context_text": record.context_text or "",
            "speaker_role": record.speaker_role,
            "dialogue_id": record.dialogue_id,
            "metadata": dict(record.metadata),
            "model_input_tokens": current_tokens,
            "model_input_pair_tokens": [],
            "model_word_labels": current_labels,
            "decode_map": list(range(current_token_count)),
            "input_format": normalized_format,
        }

    if normalized_format == "pair_context_v1":
        return {
            "id": record.id,
            "tokens": current_tokens,
            "labels": current_labels,
            "subject": record.subject,
            "context_text": record.context_text or "",
            "speaker_role": record.speaker_role,
            "dialogue_id": record.dialogue_id,
            "metadata": dict(record.metadata),
            "model_input_tokens": current_tokens,
            "model_input_pair_tokens": tokenize_text(record.context_text or ""),
            "model_word_labels": current_labels,
            "decode_map": list(range(current_token_count)),
            "input_format": normalized_format,
        }

    window_segments = _window_segments_for(record, include_speaker_prefix=normalized_format == "speaker_marked_window_v1")
    model_tokens: list[str] = []
    model_word_labels: list[str] = []
    decode_map: list[int] = []

    for segment in window_segments:
        segment_kind = str(segment["kind"])
        if segment_kind == "current":
            _append_segment(
                model_tokens,
                model_word_labels,
                decode_map,
                [CURRENT_TURN_START],
            )
            _append_segment(
                model_tokens,
                model_word_labels,
                decode_map,
                list(segment.get("prefix_tokens") or []),
            )
            _append_segment(
                model_tokens,
                model_word_labels,
                decode_map,
                list(segment["tokens"]),
                labels=current_labels,
            )
            _append_segment(
                model_tokens,
                model_word_labels,
                decode_map,
                [CURRENT_TURN_END],
            )
        else:
            _append_segment(
                model_tokens,
                model_word_labels,
                decode_map,
                list(segment["tokens"]),
            )

    return {
        "id": record.id,
        "tokens": current_tokens,
        "labels": current_labels,
        "subject": record.subject,
        "context_text": record.context_text or "",
        "speaker_role": record.speaker_role,
        "dialogue_id": record.dialogue_id,
        "metadata": dict(record.metadata),
        "model_input_tokens": model_tokens,
        "model_input_pair_tokens": [],
        "model_word_labels": model_word_labels,
        "decode_map": decode_map,
        "input_format": normalized_format,
    }


def tokenize_candidate_batch(
    tokenizer,
    batch: Mapping[str, Sequence[Sequence[str]]],
    *,
    max_length: int,
    input_format: str,
    padding: bool | str = False,
    return_tensors: str | None = None,
):
    normalized_format = resolve_candidate_input_format(input_format=input_format)
    if normalized_format == "pair_context_v1":
        return tokenizer(
            batch["model_input_tokens"],
            batch["model_input_pair_tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=padding,
            return_tensors=return_tensors,
        )
    return tokenizer(
        batch["model_input_tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


def decode_candidate_word_level_predictions(
    encoding,
    batch_index: int,
    *,
    original_token_count: int,
    predicted_ids,
    id_to_label: Mapping[int, str],
    decode_map: Sequence[int],
) -> list[str]:
    word_ids = encoding.word_ids(batch_index=batch_index)
    sequence_ids = encoding.sequence_ids(batch_index=batch_index)
    predictions = ["O"] * original_token_count
    previous_word_idx = None
    for token_idx, (word_idx, sequence_id) in enumerate(zip(word_ids, sequence_ids)):
        if sequence_id != 0 or word_idx is None or word_idx == previous_word_idx or word_idx >= len(decode_map):
            previous_word_idx = word_idx
            continue
        original_index = int(decode_map[word_idx])
        if original_index >= 0 and original_index < original_token_count:
            predictions[original_index] = str(id_to_label[int(predicted_ids[token_idx])])
        previous_word_idx = word_idx
    return predictions


def _window_segments_for(
    record: CandidateRecord,
    *,
    include_speaker_prefix: bool,
) -> list[dict[str, Any]]:
    metadata = dict(record.metadata or {})
    current_tokens = list(record.tokens)
    current_role = str(record.speaker_role or "unknown")
    prev_text = _nonempty(metadata.get("prev_turn_text"))
    next_text = _nonempty(metadata.get("next_turn_text"))
    prev_role = str(metadata.get("prev_speaker_role") or "unknown")
    next_role = str(metadata.get("next_speaker_role") or "unknown")

    segments: list[dict[str, Any]] = []
    if prev_text is not None:
        segments.append(
            {
                "kind": "previous",
                "tokens": _segment_tokens(prev_text, role=prev_role, include_speaker_prefix=include_speaker_prefix),
            }
        )
    segments.append(
        {
            "kind": "current",
            "prefix_tokens": [f"{current_role}:"] if include_speaker_prefix else [],
            "tokens": current_tokens,
        }
    )
    if next_text is not None:
        segments.append(
            {
                "kind": "next",
                "tokens": _segment_tokens(next_text, role=next_role, include_speaker_prefix=include_speaker_prefix),
            }
        )
    return segments


def _segment_tokens(text: str, *, role: str, include_speaker_prefix: bool) -> list[str]:
    tokens = tokenize_text(text)
    return _segment_tokens_from_tokens(tokens, role=role, include_speaker_prefix=include_speaker_prefix)


def _segment_tokens_from_tokens(tokens: Sequence[str], *, role: str, include_speaker_prefix: bool) -> list[str]:
    segment_tokens = list(tokens)
    if include_speaker_prefix:
        return [f"{role}:"] + segment_tokens
    return segment_tokens


def _append_segment(
    model_tokens: list[str],
    model_word_labels: list[str],
    decode_map: list[int],
    segment_tokens: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> None:
    if labels is not None and len(segment_tokens) != len(labels):
        raise ValueError("Current-turn segment tokens and labels must be the same length")
    start_index = sum(1 for item in decode_map if item >= 0)
    for offset, token in enumerate(segment_tokens):
        model_tokens.append(str(token))
        if labels is None:
            model_word_labels.append(IGNORE_WORD_LABEL)
            decode_map.append(-1)
        else:
            model_word_labels.append(str(labels[offset]))
            decode_map.append(start_index + offset)


def _nonempty(value: Any) -> str | None:
    text = str(value or "")
    return text if text.strip() else None
