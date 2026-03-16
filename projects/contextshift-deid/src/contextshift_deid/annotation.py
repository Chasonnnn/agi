from __future__ import annotations

import re
from typing import Sequence

CONTACT_HINTS = ("@", "http", "www", ".com", ".org", "gmail", "email", "discord", "instagram", "snap", "phone")
MATH_TERMS = {
    "add",
    "angle",
    "circle",
    "coefficient",
    "diameter",
    "equation",
    "expression",
    "fraction",
    "graph",
    "intercept",
    "line",
    "multiply",
    "radius",
    "slope",
    "solve",
    "subtract",
    "triangle",
    "variable",
}
TITLES = {"mr", "mrs", "ms", "dr", "prof", "teacher", "tutor"}
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def bio_spans(labels: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label == "B-SUSPECT":
            if start is not None:
                spans.append((start, index))
            start = index
        elif label != "I-SUSPECT":
            if start is not None:
                spans.append((start, index))
                start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_token(token: str) -> str:
    return normalize_whitespace(token).casefold()


def span_text(tokens: Sequence[str], span: tuple[int, int]) -> str:
    return " ".join(tokens[span[0] : span[1]])


def preview_tokens(tokens: Sequence[str], span: tuple[int, int], *, window: int = 8) -> str:
    start, end = span
    preview_start = max(0, start - window)
    preview_end = min(len(tokens), end + window)
    pieces: list[str] = []
    for index in range(preview_start, preview_end):
        token = str(tokens[index])
        if index == start:
            pieces.append("[[")
        pieces.append(token)
        if index + 1 == end:
            pieces.append("]]")
    return " ".join(pieces)


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def find_subsequence(tokens: Sequence[str], target_tokens: Sequence[str]) -> tuple[int, int] | None:
    normalized_tokens = [normalize_token(token) for token in tokens]
    normalized_target = [normalize_token(token) for token in target_tokens if normalize_token(token)]
    if not normalized_target:
        return None
    window = len(normalized_target)
    for start in range(0, len(normalized_tokens) - window + 1):
        if normalized_tokens[start : start + window] == normalized_target:
            return (start, start + window)
    return None


def is_contact_like(text: str) -> bool:
    lowered = text.casefold()
    return any(hint in lowered for hint in CONTACT_HINTS)


def is_grade_level(text: str) -> bool:
    lowered = text.casefold()
    return "grade" in lowered or bool(re.fullmatch(r"\d{1,2}(st|nd|rd|th)", lowered))


def is_probable_phone(text: str) -> bool:
    digits = "".join(character for character in text if character.isdigit())
    return len(digits) >= 7


def is_name_like(tokens: Sequence[str]) -> bool:
    cleaned = [token.strip(".,!?()[]{}:;\"'") for token in tokens if token.strip()]
    if not cleaned:
        return False
    lowered = [token.casefold() for token in cleaned]
    if lowered[0] in TITLES:
        return True
    word_like = [token for token in cleaned if WORD_RE.search(token)]
    if not word_like:
        return False
    alpha_initial = [token for token in word_like if token[:1].isalpha()]
    if not alpha_initial:
        return False
    return all(token[:1].isupper() for token in alpha_initial)


def is_math_like(tokens: Sequence[str]) -> bool:
    cleaned = [token.strip() for token in tokens if token.strip()]
    if not cleaned:
        return False
    allowed = {"x", "y", "z", "+", "-", "*", "/", "=", "(", ")", ".", ",", "%", "^", "<", ">"}
    if len(cleaned) == 1 and cleaned[0].casefold() in {"x", "y", "z"}:
        return False
    if not any(any(character.isalnum() for character in token) for token in cleaned):
        return False
    if all(token.isdigit() or token.casefold() in allowed for token in cleaned):
        return any(character.isdigit() for token in cleaned for character in token)
    if any(token.casefold() in MATH_TERMS for token in cleaned):
        return True
    if len(cleaned) == 1:
        return False
    joined = "".join(cleaned)
    return any(character.isdigit() for character in joined) and any(character in "+-*/=^<>" for character in joined)


def math_like_spans(tokens: Sequence[str], *, max_span_length: int = 4) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + max_span_length) + 1):
            if is_math_like(tokens[start:end]):
                spans.append((start, end))
                break
    return spans


def guess_entity_type(span_value: str, span_tokens: Sequence[str]) -> str | None:
    if is_contact_like(span_value):
        return "URL" if "http" in span_value.casefold() or "www" in span_value.casefold() else "CONTACT"
    if is_grade_level(span_value):
        return "GRADE_LEVEL"
    if is_probable_phone(span_value):
        return "PHONE_NUMBER"
    if is_name_like(span_tokens):
        return "PERSON"
    return None
