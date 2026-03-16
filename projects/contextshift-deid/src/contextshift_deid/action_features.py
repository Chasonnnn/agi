from __future__ import annotations

from .schemas import ActionRecord

DEFAULT_ACTION_INPUT_FORMAT = "marked_turn_v1"
ACTION_INPUT_FORMAT_CHOICES = ("marked_turn_v1", "flat_v1")


def _flat_action_prompt(record: ActionRecord) -> str:
    parts = [
        f"subject: {record.subject}",
        f"context: {record.context_text}",
        f"span: {record.span_text}",
    ]
    if record.anchor_text:
        parts.append(f"anchor: {record.anchor_text}")
    if record.speaker_role:
        parts.append(f"speaker_role: {record.speaker_role}")
    if record.entity_type:
        parts.append(f"entity_type: {record.entity_type}")
    if record.intent_label:
        parts.append(f"intent: {record.intent_label}")
    return "\n".join(parts)


def _metadata_int(record: ActionRecord, key: str) -> int | None:
    value = record.metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _marked_target_turn(record: ActionRecord) -> tuple[str, int | None] | None:
    turn_text = record.metadata.get("turn_text")
    if not isinstance(turn_text, str) or not turn_text:
        return None
    tag_start = _metadata_int(record, "tag_start")
    tag_end = _metadata_int(record, "tag_end")
    if tag_start is None or tag_end is None:
        return None
    if tag_start < 0 or tag_end <= tag_start or tag_end > len(turn_text):
        return None
    highlighted = f"{turn_text[:tag_start]}[TGT]{turn_text[tag_start:tag_end]}[/TGT]{turn_text[tag_end:]}"
    return highlighted, _metadata_int(record, "tag_occurrence")


def build_action_prompt(record: ActionRecord, *, input_format: str = DEFAULT_ACTION_INPUT_FORMAT) -> str:
    if input_format not in ACTION_INPUT_FORMAT_CHOICES:
        raise ValueError(f"Unsupported action input format: {input_format}")
    if input_format == "flat_v1":
        return _flat_action_prompt(record)

    marked_turn = _marked_target_turn(record)
    if marked_turn is None:
        return _flat_action_prompt(record)

    target_turn, tag_occurrence = marked_turn
    parts = [
        f"subject: {record.subject}",
        f"speaker_role: {record.speaker_role or 'unknown'}",
        f"entity_type: {record.entity_type or 'unknown'}",
        f"target_turn: {target_turn}",
        f"span_text: {record.span_text}",
        f"local_context: {record.context_text}",
    ]
    if record.anchor_text:
        parts.append(f"anchor_text: {record.anchor_text}")
    if tag_occurrence is not None:
        parts.append(f"tag_occurrence: {tag_occurrence}")
    return "\n".join(parts)
