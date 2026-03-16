from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ANNOTATION_DIR, DEFAULT_ACTION_LABELS
from contextshift_deid.data import load_jsonl

COMMAND_TO_LABEL = {
    "r": "REDACT",
    "k": "KEEP",
    "v": "REVIEW",
}
DEFAULT_SHORTCUTS = {
    "R": ("REDACT", "PRIVATE"),
    "K": ("KEEP", "CURRICULAR"),
    "V": ("REVIEW", "AMBIGUOUS"),
}
DEFAULT_SEMANTIC_ROLE_BY_ACTION = {
    "REDACT": "PRIVATE",
    "KEEP": "CURRICULAR",
    "REVIEW": "AMBIGUOUS",
}
SEMANTIC_ROLE_ALIASES = {
    "p": "PRIVATE",
    "private": "PRIVATE",
    "c": "CURRICULAR",
    "curricular": "CURRICULAR",
    "a": "AMBIGUOUS",
    "ambiguous": "AMBIGUOUS",
}
EXCERPT_RADIUS = 120
FULL_TURN_RENDER_LIMIT = 600


def _default_annotator() -> str:
    return os.environ.get("USER", "annotator")


def _default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}.annotated{input_path.suffix}")
    return ANNOTATION_DIR / f"{input_path.name}.annotated.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary_path.replace(path)


def _load_rows(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(input_path)
    if not output_path.exists():
        return rows
    existing_rows = load_jsonl(output_path)
    existing_by_id = {str(row["id"]): row for row in existing_rows}
    merged_rows: list[dict[str, Any]] = []
    for row in rows:
        existing = existing_by_id.get(str(row["id"]))
        if existing is None:
            merged_rows.append(row)
            continue
        merged = dict(row)
        if "annotation" in existing:
            merged["annotation"] = existing["annotation"]
        merged_rows.append(merged)
    return merged_rows


def _annotation_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        annotation = row.get("annotation") or {}
        label = annotation.get("action_label")
        if label:
            counts[str(label)] += 1
    return counts


def _highlight_with_offsets(text: str, start: int | None, end: int | None) -> str:
    if start is None or end is None:
        return text
    bounded_start = max(0, min(int(start), len(text)))
    bounded_end = max(bounded_start, min(int(end), len(text)))
    return f"{text[:bounded_start]}[[{text[bounded_start:bounded_end]}]]{text[bounded_end:]}"


def _excerpt_with_offsets(text: str, start: int | None, end: int | None, *, radius: int = EXCERPT_RADIUS) -> str:
    if start is None or end is None:
        excerpt = text[: radius * 2]
        return excerpt if len(text) <= radius * 2 else f"{excerpt}..."
    bounded_start = max(0, min(int(start), len(text)))
    bounded_end = max(bounded_start, min(int(end), len(text)))
    excerpt_start = max(0, bounded_start - radius)
    excerpt_end = min(len(text), bounded_end + radius)
    excerpt = _highlight_with_offsets(text[excerpt_start:excerpt_end], bounded_start - excerpt_start, bounded_end - excerpt_start)
    prefix = "..." if excerpt_start > 0 else ""
    suffix = "..." if excerpt_end < len(text) else ""
    return f"{prefix}{excerpt}{suffix}"


def _highlight_context_text(
    context_text: str,
    *,
    speaker_role: str,
    turn_text: str,
    start: int | None,
    end: int | None,
) -> str:
    if not context_text or not turn_text:
        return context_text
    target_line = f"{speaker_role}: {turn_text}"
    highlighted_line = f"{speaker_role}: {_highlight_with_offsets(turn_text, start, end)}  <-- target"
    if target_line in context_text:
        return context_text.replace(target_line, highlighted_line, 1)
    return context_text


def _prompt_semantic_role(*, default: str) -> str:
    while True:
        raw = input(f"semantic_role [PRIVATE/CURRICULAR/AMBIGUOUS] default={default}> ").strip().casefold()
        if not raw:
            return default
        role = SEMANTIC_ROLE_ALIASES.get(raw)
        if role is not None:
            return role
        print("Unknown semantic role. Use PRIVATE, CURRICULAR, or AMBIGUOUS.")


def _next_unlabeled_index(rows: list[dict[str, Any]], start_index: int) -> int | None:
    for candidate_index in range(start_index, len(rows)):
        annotation = rows[candidate_index].get("annotation") or {}
        if annotation.get("action_label") not in DEFAULT_ACTION_LABELS:
            return candidate_index
    return None


def _first_pending_index(rows: list[dict[str, Any]]) -> int:
    for index, row in enumerate(rows):
        annotation = row.get("annotation") or {}
        if annotation.get("action_label") not in DEFAULT_ACTION_LABELS:
            return index
    return 0


def _render_row(index: int, rows: list[dict[str, Any]], *, output_path: Path) -> None:
    row = rows[index]
    counts = _annotation_counts(rows)
    labeled = sum(counts.values())
    print("\033[2J\033[H", end="")
    print(f"Action Annotation Tool  [{index + 1}/{len(rows)}]  labeled={labeled}  output={output_path}")
    print(f"Distribution: REDACT={counts['REDACT']} KEEP={counts['KEEP']} REVIEW={counts['REVIEW']}")
    print()
    print(f"id: {row['id']}")
    print(f"split: {row.get('split', 'unknown')}    source_row_id: {row.get('source_row_id', 'unknown')}")
    print(
        "subject: {subject}    eval_slice: {eval_slice}    speaker_role: {speaker}    entity_type: {entity}    pool_source: {source}".format(
            subject=row.get("subject", "unknown"),
            eval_slice=row.get("eval_slice", "unknown"),
            speaker=row.get("speaker_role", "unknown"),
            entity=row.get("entity_type", "unknown"),
            source=row.get("pool_source", "unknown"),
        )
    )
    print(f"span_text: {row.get('span_text', '')}")
    if row.get("token_preview"):
        print(f"token_preview: {row['token_preview']}")
    metadata = row.get("metadata") or {}
    turn_text = metadata.get("turn_text")
    if turn_text:
        turn_text_str = str(turn_text)
        tag_start = metadata.get("tag_start")
        tag_end = metadata.get("tag_end")
        highlighted_turn = _highlight_with_offsets(turn_text_str, tag_start, tag_end)
        target_excerpt = _excerpt_with_offsets(turn_text_str, tag_start, tag_end)
        print(
            "target_excerpt: {excerpt}    tag_occurrence: {occurrence}".format(
                excerpt=target_excerpt,
                occurrence=metadata.get("tag_occurrence", "unknown"),
            )
        )
        if len(turn_text_str) <= FULL_TURN_RENDER_LIMIT:
            print(f"target_turn: {highlighted_turn}")
        else:
            print(f"target_turn: [omitted full turn; {len(turn_text_str.split())} words, see excerpt above]")
    if row.get("anchor_text"):
        print(f"anchor_text: {row['anchor_text']}")
    print("context_text:")
    print(
        _highlight_context_text(
            str(row.get("context_text", "")),
            speaker_role=str(row.get("speaker_role", "unknown")),
            turn_text=str(turn_text or ""),
            start=metadata.get("tag_start"),
            end=metadata.get("tag_end"),
        )
    )
    print()
    print(
        "suggested_action: {action}    reason: {reason}".format(
            action=row.get("suggested_action", "unknown"),
            reason=row.get("suggested_reason", ""),
        )
    )
    annotation = row.get("annotation") or {}
    if annotation.get("action_label"):
        print(
            "current_annotation: {label} / {semantic_role} by {annotator} at {timestamp}".format(
                label=annotation.get("action_label"),
                semantic_role=annotation.get("semantic_role", "unknown"),
                annotator=annotation.get("annotator", "unknown"),
                timestamp=annotation.get("timestamp", "unknown"),
            )
        )
        if annotation.get("notes"):
            print(f"current_notes: {annotation['notes']}")
    print()
    print("Commands: [r]edact  [k]eep  re[v]iew  [R/K/V]=label+default role  [s]kip  [b]ack  [u]nlabel  [q]uit  [?]help")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manually annotate an action-label pool in the terminal.")
    parser.add_argument("--input-file", type=Path, default=ANNOTATION_DIR / "action_pool_dev.jsonl")
    parser.add_argument("--output-file", type=Path, help="Defaults to <input>.annotated.jsonl")
    parser.add_argument("--annotator", default=_default_annotator(), help="Short annotator identifier stored with each decision.")
    parser.add_argument("--start-index", type=int, help="Optional zero-based index override.")
    args = parser.parse_args()

    output_path = args.output_file or _default_output_path(args.input_file)
    rows = _load_rows(args.input_file, output_path)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_file}")

    index = args.start_index if args.start_index is not None else _first_pending_index(rows)
    index = min(max(index, 0), len(rows) - 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        _render_row(index, rows, output_path=output_path)
        raw_command = input("action> ").strip()
        if not raw_command:
            continue
        shortcut = DEFAULT_SHORTCUTS.get(raw_command)
        command = raw_command.lower()
        if command == "?":
            input("Use r/k/v to label, uppercase R/K/V to also accept the default semantic role, s to skip, b to go back, u to clear a label, q to quit. Press Enter.")
            continue
        if command == "q":
            _write_jsonl(output_path, rows)
            print(f"Saved {len(rows)} rows to {output_path}")
            return
        if command == "b":
            index = max(0, index - 1)
            continue
        if command == "s":
            index = min(len(rows) - 1, index + 1)
            continue
        if command == "u":
            rows[index].pop("annotation", None)
            _write_jsonl(output_path, rows)
            continue
        if shortcut is not None:
            label, semantic_role = shortcut
            note = None
        else:
            label = COMMAND_TO_LABEL.get(command)
            if label is None:
                input("Unknown command. Press Enter.")
                continue
            semantic_role = _prompt_semantic_role(default=DEFAULT_SEMANTIC_ROLE_BY_ACTION[label])
            note_text = input("notes (optional)> ").strip()
            note = note_text or None
        rows[index]["annotation"] = {
            "annotator": args.annotator,
            "action_label": label,
            "semantic_role": semantic_role,
            "notes": note,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        _write_jsonl(output_path, rows)
        if index >= len(rows) - 1:
            print(f"Saved final annotation to {output_path}")
            return
        next_pending = _next_unlabeled_index(rows, index + 1)
        index = next_pending if next_pending is not None else min(len(rows) - 1, index + 1)


if __name__ == "__main__":
    main()
