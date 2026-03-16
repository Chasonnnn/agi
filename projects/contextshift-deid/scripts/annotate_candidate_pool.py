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

from contextshift_deid.annotation import bio_spans, span_text
from contextshift_deid.constants import ANNOTATION_DIR
from contextshift_deid.data import load_jsonl, validate_candidate_records


def _default_annotator() -> str:
    return os.environ.get("USER", "annotator")


def _default_input_path() -> Path:
    return ANNOTATION_DIR / "timss_science" / "candidate_pool_dev.jsonl"


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
    validate_candidate_records(input_path)
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
        if "labels" in existing:
            merged["labels"] = [str(label) for label in existing["labels"]]
        merged_metadata = dict(row.get("metadata") or {})
        merged_metadata.update(existing.get("metadata") or {})
        merged["metadata"] = merged_metadata
        merged_rows.append(merged)
    return merged_rows


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata


def _is_completed(row: dict[str, Any]) -> bool:
    return bool(_metadata(row).get("annotation_completed"))


def _completed_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if _is_completed(row))


def _first_pending_index(rows: list[dict[str, Any]]) -> int:
    for index, row in enumerate(rows):
        if not _is_completed(row):
            return index
    return 0


def _label_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for label in row.get("labels", []):
            counts[str(label)] += 1
    return counts


def _rows_with_positive_spans(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if any(label != "O" for label in row.get("labels", [])))


def _labels_to_mask(labels: list[str]) -> list[bool]:
    return [label != "O" for label in labels]


def _mask_to_labels(mask: list[bool]) -> list[str]:
    labels: list[str] = []
    previous = False
    for current in mask:
        if not current:
            labels.append("O")
        elif previous:
            labels.append("I-SUSPECT")
        else:
            labels.append("B-SUSPECT")
        previous = current
    return labels


def _apply_range(labels: list[str], start: int, end: int, *, value: bool) -> list[str]:
    mask = _labels_to_mask(labels)
    for index in range(start, end + 1):
        mask[index] = value
    return _mask_to_labels(mask)


def _span_summary(row: dict[str, Any]) -> str:
    spans = bio_spans([str(label) for label in row.get("labels", [])])
    if not spans:
        return "none"
    tokens = [str(token) for token in row.get("tokens", [])]
    return "; ".join(
        f"{start}-{end - 1}: {span_text(tokens, (start, end))}"
        for start, end in spans
    )


def _render_token_lines(tokens: list[str], labels: list[str], *, line_width: int = 110) -> list[str]:
    rendered_lines: list[str] = []
    current_line = ""
    for index, token in enumerate(tokens):
        piece = f"{index:03d}:{token}"
        if labels[index] != "O":
            piece = f"{index:03d}:[[{token}]]"
        if current_line and len(current_line) + len(piece) + 1 > line_width:
            rendered_lines.append(current_line)
            current_line = piece
            continue
        if current_line:
            current_line = f"{current_line} {piece}"
        else:
            current_line = piece
    if current_line:
        rendered_lines.append(current_line)
    return rendered_lines


def _render_row(index: int, rows: list[dict[str, Any]], *, output_path: Path) -> None:
    row = rows[index]
    metadata = _metadata(row)
    completed = _completed_count(rows)
    label_counts = _label_counts(rows)
    print("\033[2J\033[H", end="")
    print(
        f"Candidate Annotation Tool  [{index + 1}/{len(rows)}]  "
        f"completed={completed}  positive_rows={_rows_with_positive_spans(rows)}  output={output_path}"
    )
    print(
        "Token labels: O={o} B-SUSPECT={b} I-SUSPECT={i}".format(
            o=label_counts["O"],
            b=label_counts["B-SUSPECT"],
            i=label_counts["I-SUSPECT"],
        )
    )
    print()
    print(f"id: {row['id']}")
    print(
        "subject: {subject}    speaker_role: {speaker_role}    dialogue_id: {dialogue_id}".format(
            subject=row.get("subject", "unknown"),
            speaker_role=row.get("speaker_role", "unknown"),
            dialogue_id=row.get("dialogue_id", "unknown"),
        )
    )
    print(
        "country: {country}    transcript_code: {transcript_code}    timestamp: {timestamp}".format(
            country=metadata.get("country", "unknown"),
            transcript_code=metadata.get("transcript_code", "unknown"),
            timestamp=metadata.get("timestamp", "unknown"),
        )
    )
    if row.get("anchor_text"):
        print(f"anchor_text: {row['anchor_text']}")
    print("context_text:")
    print(row.get("context_text", ""))
    print()
    print(f"current_spans: {_span_summary(row)}")
    annotation_timestamp = metadata.get("annotation_timestamp")
    if _is_completed(row):
        print(
            "completed_by: {annotator} at {timestamp}".format(
                annotator=metadata.get("annotation_annotator", "unknown"),
                timestamp=annotation_timestamp or "unknown",
            )
        )
    print()
    print("tokens:")
    for line in _render_token_lines(
        [str(token) for token in row.get("tokens", [])],
        [str(label) for label in row["labels"]],
    ):
        print(line)
    print()
    print("Commands: a <start> <end>  r <start> <end>  c  d  s  b  u  q  ?")


def _parse_range(command: str, token_count: int) -> tuple[int, int]:
    pieces = command.split()
    if len(pieces) != 3:
        raise ValueError("Expected two token indices.")
    start = int(pieces[1])
    end = int(pieces[2])
    if start < 0 or end < 0 or start >= token_count or end >= token_count or end < start:
        raise ValueError(f"Invalid token range {start}..{end} for {token_count} tokens.")
    return start, end


def _advance_to_next_pending(rows: list[dict[str, Any]], current_index: int) -> int:
    for candidate_index in range(current_index + 1, len(rows)):
        if not _is_completed(rows[candidate_index]):
            return candidate_index
    return min(len(rows) - 1, current_index + 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually annotate candidate BIO labels on tokenized rows in the terminal."
    )
    parser.add_argument("--input-file", type=Path, default=_default_input_path())
    parser.add_argument("--output-file", type=Path, help="Defaults to <input>.annotated.jsonl")
    parser.add_argument("--annotator", default=_default_annotator())
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
        command = input("candidate> ").strip()
        if not command:
            continue
        lowered = command.lower()
        if lowered == "?":
            input(
                "Use 'a start end' to add a suspect span and 'r start end' to remove one. "
                "Use c to clear labels, d to mark complete and advance, s to skip, b to go back, "
                "u to mark the row incomplete again, and q to quit. Press Enter."
            )
            continue
        if lowered == "q":
            _write_jsonl(output_path, rows)
            print(f"Saved {len(rows)} rows to {output_path}")
            return
        if lowered == "b":
            index = max(0, index - 1)
            continue
        if lowered == "s":
            index = min(len(rows) - 1, index + 1)
            continue
        if lowered == "c":
            rows[index]["labels"] = ["O"] * len(rows[index]["tokens"])
            metadata = dict(_metadata(rows[index]))
            metadata["annotation_completed"] = False
            metadata.pop("annotation_annotator", None)
            metadata.pop("annotation_timestamp", None)
            metadata.pop("label_source", None)
            rows[index]["metadata"] = metadata
            _write_jsonl(output_path, rows)
            continue
        if lowered == "u":
            metadata = dict(_metadata(rows[index]))
            metadata["annotation_completed"] = False
            metadata.pop("annotation_annotator", None)
            metadata.pop("annotation_timestamp", None)
            metadata.pop("label_source", None)
            rows[index]["metadata"] = metadata
            _write_jsonl(output_path, rows)
            continue
        if lowered == "d":
            metadata = dict(_metadata(rows[index]))
            metadata.update(
                {
                    "annotation_completed": True,
                    "annotation_annotator": args.annotator,
                    "annotation_timestamp": datetime.now().astimezone().isoformat(),
                    "label_source": "manual_candidate_token_bio_v1",
                }
            )
            rows[index]["metadata"] = metadata
            _write_jsonl(output_path, rows)
            if index >= len(rows) - 1:
                print(f"Saved final annotation to {output_path}")
                return
            index = _advance_to_next_pending(rows, index)
            continue
        if lowered.startswith("a ") or lowered.startswith("r "):
            try:
                start, end = _parse_range(lowered, len(rows[index]["tokens"]))
            except ValueError as exc:
                input(f"{exc} Press Enter.")
                continue
            rows[index]["labels"] = _apply_range(
                [str(label) for label in rows[index]["labels"]],
                start,
                end,
                value=lowered.startswith("a "),
            )
            metadata = dict(_metadata(rows[index]))
            metadata["annotation_completed"] = False
            metadata.pop("annotation_annotator", None)
            metadata.pop("annotation_timestamp", None)
            metadata.pop("label_source", None)
            rows[index]["metadata"] = metadata
            _write_jsonl(output_path, rows)
            continue
        input("Unknown command. Press Enter.")


if __name__ == "__main__":
    main()
