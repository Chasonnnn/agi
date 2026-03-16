from __future__ import annotations

from .schemas import ActionRecord


def build_action_prompt(record: ActionRecord) -> str:
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
