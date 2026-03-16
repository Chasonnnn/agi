from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True, frozen=True)
class RedactionSpan:
    start: int
    end: int
    text: str
    entity_type: str


_ENTITY_ALIASES = {
    "PERSON": "NAME",
    "NAME": "NAME",
    "LOCATION": "LOCATION",
    "SCHOOL": "SCHOOL",
    "URL": "URL",
    "AGE": "AGE",
    "DATE": "DATE",
    "PHONE_NUMBER": "PHONE",
    "GRADE_LEVEL": "GRADE_LEVEL",
    "COURSE": "COURSE",
    "NRP": "ENTITY",
    "MISC_ID": "ID",
    "US_DRIVER_LICENSE": "ID",
    "IP_ADDRESS": "IP",
}

_FAKE_VALUES = {
    "NAME": ["Alex", "Jordan", "Casey", "Taylor", "Morgan", "Riley"],
    "LOCATION": ["Springfield", "Riverton", "Maple Grove", "Lakeview"],
    "SCHOOL": ["Riverside High", "Summit Academy", "Northfield Prep"],
    "URL": [
        "https://example.org/resource-1",
        "https://example.org/resource-2",
        "https://example.org/resource-3",
    ],
    "AGE": ["14", "15", "16", "17"],
    "DATE": ["January 1, 2000", "February 2, 2001", "March 3, 2002"],
    "PHONE": ["555-0101", "555-0102", "555-0103"],
    "GRADE_LEVEL": ["9th grade", "10th grade", "11th grade"],
    "COURSE": ["Algebra I", "Geometry", "World History"],
    "ID": ["ID-1001", "ID-1002", "ID-1003"],
    "IP": ["203.0.113.10", "203.0.113.11", "203.0.113.12"],
    "ENTITY": ["Item-A", "Item-B", "Item-C"],
}


def normalize_entity_type(entity_type: str | None) -> str:
    if not entity_type:
        return "ENTITY"
    return _ENTITY_ALIASES.get(entity_type.upper(), entity_type.upper())


class RedactionRenderer:
    def __init__(self, strategy: str = "typed_placeholder") -> None:
        self.strategy = strategy
        self._assignments: dict[tuple[str, str], str] = {}
        self._type_counters: dict[str, int] = {}
        self._fake_indices: dict[str, int] = {}

    def render(self, text: str, spans: Iterable[RedactionSpan]) -> str:
        ordered_spans = sorted(spans, key=lambda span: (span.start, span.end))
        rendered = text
        for span in reversed(ordered_spans):
            replacement = self._replacement_for(span)
            rendered = rendered[: span.start] + replacement + rendered[span.end :]
        return rendered

    def _replacement_for(self, span: RedactionSpan) -> str:
        normalized_type = normalize_entity_type(span.entity_type)
        if self.strategy == "delete":
            return ""
        if self.strategy == "mask":
            return "[REDACTED]"

        key = (normalized_type, span.text.casefold())
        if key in self._assignments:
            return self._assignments[key]

        if self.strategy == "typed_placeholder":
            replacement = self._next_placeholder(normalized_type)
        elif self.strategy == "fake_surrogate":
            replacement = self._next_fake_value(normalized_type)
        else:
            raise ValueError(f"Unsupported redaction strategy: {self.strategy}")

        self._assignments[key] = replacement
        return replacement

    def _next_placeholder(self, normalized_type: str) -> str:
        next_index = self._type_counters.get(normalized_type, 0) + 1
        self._type_counters[normalized_type] = next_index
        return f"[{normalized_type}_{next_index}]"

    def _next_fake_value(self, normalized_type: str) -> str:
        values = _FAKE_VALUES.get(normalized_type, _FAKE_VALUES["ENTITY"])
        next_index = self._fake_indices.get(normalized_type, 0)
        self._fake_indices[normalized_type] = next_index + 1
        if next_index < len(values):
            return values[next_index]
        return f"{values[-1]}-{next_index + 1}"
