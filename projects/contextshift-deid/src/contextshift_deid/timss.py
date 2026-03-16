from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Sequence

TRANSCRIPT_NAME_RE = re.compile(r"^Science (?P<code>[A-Z]{2}\d+) transcript\.txt$")
TRANSCRIPT_LINE_RE = re.compile(r"^(?P<timestamp>\d{2}:\d{2}:\d{2})\t(?P<speaker>[A-Z]+)\t(?P<text>.*)$")
FORM_NOISE = {"Top of Form", "Bottom of Form"}


@dataclass(slots=True)
class TimssTurn:
    timestamp: str
    speaker_tag: str
    speaker_role: str
    text: str


@dataclass(slots=True)
class TimssTranscript:
    dialogue_id: str
    subject: str
    country: str
    transcript_code: str
    source_path: Path
    turns: list[TimssTurn]


def normalize_timss_text(text: str) -> str:
    cleaned = text.replace("\ufeff", " ").replace("\ufffd", " ").replace("\xa0", " ")
    return " ".join(cleaned.split())


def sanitize_timss_line(text: str) -> str:
    return text.replace("\ufeff", "").replace("\ufffd", " ").replace("\xa0", " ").strip()


def speaker_role_for_tag(tag: str) -> str:
    normalized = tag.strip().upper()
    if normalized == "T":
        return "teacher"
    if normalized == "O":
        return "observer"
    return "student"


def transcript_code_from_path(path: Path) -> str:
    match = TRANSCRIPT_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"{path}: unsupported TIMSS transcript filename")
    return match.group("code")


def discover_timss_transcript_paths(root: Path, countries: Sequence[str]) -> list[Path]:
    allowed_countries = {country.strip().upper() for country in countries if country.strip()}
    if not allowed_countries:
        raise ValueError("No countries requested for TIMSS discovery.")

    chosen_paths: dict[str, tuple[str, Path]] = {}
    for path in sorted(root.rglob("*.txt")):
        if path.parent.name.upper() not in allowed_countries:
            continue
        match = TRANSCRIPT_NAME_RE.match(path.name)
        if match is None:
            continue
        transcript_code = match.group("code")
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        existing = chosen_paths.get(transcript_code)
        if existing is None:
            chosen_paths[transcript_code] = (digest, path)
            continue
        existing_digest, existing_path = existing
        if existing_digest != digest:
            raise ValueError(
                f"Conflicting TIMSS duplicates for {transcript_code}: {existing_path} and {path}"
            )
    return [chosen_paths[code][1] for code in sorted(chosen_paths)]


def parse_timss_transcript(path: Path, *, subject: str) -> TimssTranscript:
    transcript_code = transcript_code_from_path(path)
    country = transcript_code[:2]
    raw_text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    turns: list[TimssTurn] = []
    for raw_line in raw_text.splitlines():
        line = sanitize_timss_line(raw_line)
        if not line or line in FORM_NOISE:
            continue
        match = TRANSCRIPT_LINE_RE.match(line)
        if match is None:
            continue
        text = normalize_timss_text(match.group("text"))
        if not text or text in FORM_NOISE:
            continue
        speaker_tag = match.group("speaker")
        turns.append(
            TimssTurn(
                timestamp=match.group("timestamp"),
                speaker_tag=speaker_tag,
                speaker_role=speaker_role_for_tag(speaker_tag),
                text=text,
            )
        )
    if not turns:
        raise ValueError(f"{path}: no timestamped transcript turns found")
    return TimssTranscript(
        dialogue_id=f"timss-science-{transcript_code.casefold()}",
        subject=subject,
        country=country,
        transcript_code=transcript_code,
        source_path=path,
        turns=turns,
    )


def format_context_window(turns: Sequence[TimssTurn], index: int, *, radius: int = 1) -> str:
    window_start = max(0, index - radius)
    window_end = min(len(turns), index + radius + 1)
    snippets = [
        f"{turns[current_index].speaker_role}: {turns[current_index].text}"
        for current_index in range(window_start, window_end)
    ]
    return "\n".join(snippets)
