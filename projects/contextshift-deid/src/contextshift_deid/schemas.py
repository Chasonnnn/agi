from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateRecord:
    id: str
    subject: str
    tokens: list[str]
    labels: list[str]
    anchor_text: str | None = None
    dialogue_id: str | None = None
    speaker_role: str | None = None
    context_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionRecord:
    id: str
    subject: str
    span_text: str
    context_text: str
    action_label: str
    eval_slice: str | None = None
    anchor_text: str | None = None
    speaker_role: str | None = None
    entity_type: str | None = None
    semantic_role: str | None = None
    intent_label: str | None = None
    dialogue_id: str | None = None
    cost: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
