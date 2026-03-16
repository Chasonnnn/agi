from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any, Iterable, Sequence

TOPIC_TO_SUBJECT = {
    "readingWriting": "english",
    "socialStudies": "social_studies",
}
AMBIGUOUS_CHALLENGE_TAGS = {"PERSON", "LOCATION", "NRP"}
CHALLENGE_TAGS = AMBIGUOUS_CHALLENGE_TAGS | {"SCHOOL", "COURSE"}
CONTROL_TAGS = {
    "PHONE_NUMBER",
    "URL",
    "SOCIAL_HANDLE",
    "IP_ADDRESS",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_SSN",
}
SUPPORTED_TAGS = CHALLENGE_TAGS | CONTROL_TAGS
TAG_RE = re.compile(r"<([A-Z_]+)>")


def canonical_subject(topic_name: str | None) -> str | None:
    if topic_name is None:
        return None
    return TOPIC_TO_SUBJECT.get(str(topic_name))


def normalize_text(text: str) -> str:
    return " ".join(str(text).split())


def stable_hash(value: str, *, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()


def stable_sort_key(value: str, *, seed: int) -> tuple[str, str]:
    return (stable_hash(value, seed=seed), value)


def format_context_window(turns: Sequence[dict[str, Any]], index: int) -> str:
    snippets: list[str] = []
    if index > 0:
        prev_turn = turns[index - 1]
        snippets.append(f"{prev_turn['user_role']}: {normalize_text(prev_turn['message'])}")
    turn = turns[index]
    snippets.append(f"{turn['user_role']}: {normalize_text(turn['message'])}")
    if index + 1 < len(turns):
        next_turn = turns[index + 1]
        snippets.append(f"{next_turn['user_role']}: {normalize_text(next_turn['message'])}")
    return "\n".join(snippets)


def build_anchor_text(turns: Sequence[dict[str, Any]], *, max_segments: int = 3, max_chars: int = 180) -> str:
    snippets: list[str] = []
    for turn in turns:
        message = normalize_text(turn.get("message", ""))
        if not message:
            continue
        snippets.append(f"{turn.get('user_role', 'unknown')}: {message}")
        if len(snippets) >= max_segments:
            break
    anchor = " | ".join(snippets)
    if len(anchor) <= max_chars:
        return anchor
    return anchor[: max_chars - 3].rstrip() + "..."


def extract_supported_tags(text: str) -> list[dict[str, Any]]:
    occurrences_by_tag: Counter[str] = Counter()
    tags: list[dict[str, Any]] = []
    for match in TAG_RE.finditer(text):
        entity_type = match.group(1)
        if entity_type not in SUPPORTED_TAGS:
            continue
        span_text = match.group(0)
        occurrences_by_tag[span_text] += 1
        tags.append(
            {
                "span_text": span_text,
                "entity_type": entity_type,
                "tag_start": match.start(),
                "tag_end": match.end(),
                "tag_occurrence": occurrences_by_tag[span_text],
            }
        )
    return tags


def challenge_score(*, entity_type: str, qualifying_tag_count: int, speaker_role: str) -> int:
    score = 0
    if entity_type in AMBIGUOUS_CHALLENGE_TAGS:
        score += 3
    if qualifying_tag_count >= 2:
        score += 1
    if str(speaker_role) == "student":
        score += 1
    return score


def proportional_targets(counts: Counter[str], *, total: int) -> dict[str, int]:
    if total <= 0 or not counts:
        return {}
    total_count = sum(counts.values())
    base_targets: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for value, count in counts.items():
        raw = (count / total_count) * total
        base = int(raw)
        base_targets[value] = base
        assigned += base
        remainders.append((raw - base, value))
    remainder = total - assigned
    for _, value in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remainder]:
        base_targets[value] += 1
    return base_targets


def bucket_sessions(session_ids: Iterable[str], *, seed: int) -> dict[str, str]:
    ordered = sorted(session_ids, key=lambda value: stable_sort_key(value, seed=seed))
    midpoint = len(ordered) // 2
    split_by_session: dict[str, str] = {}
    for session_id in ordered[:midpoint]:
        split_by_session[session_id] = "dev"
    for session_id in ordered[midpoint:]:
        split_by_session[session_id] = "test"
    return split_by_session
