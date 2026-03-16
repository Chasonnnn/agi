from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, CANDIDATE_DIR, INTERIM_DIR
from contextshift_deid.data import ensure_repo_layout

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
SUPPORTED_PII_TYPES = {
    "PERSON",
    "LOCATION",
    "SCHOOL",
    "URL",
    "AGE",
    "DATE",
    "PHONE_NUMBER",
    "GRADE_LEVEL",
    "COURSE",
    "NRP",
    "MISC_ID",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
        spans.append(match.span())
    return tokens, spans


def _build_bio_labels(token_spans: list[tuple[int, int]], annotations: list[dict]) -> list[str]:
    labels = ["O"] * len(token_spans)
    for annotation in sorted(annotations, key=lambda item: (item["start"], item["end"])):
        start = int(annotation["start"])
        end = int(annotation["end"])
        first = True
        for index, (token_start, token_end) in enumerate(token_spans):
            overlaps = token_start < end and token_end > start
            if not overlaps:
                continue
            labels[index] = "B-SUSPECT" if first else "I-SUSPECT"
            first = False
    return labels


def _context_window(turns: list[dict], index: int) -> str:
    snippets: list[str] = []
    if index > 0:
        prev_turn = turns[index - 1]
        snippets.append(f"{prev_turn['role']}: {prev_turn['content']}")
    current_turn = turns[index]
    snippets.append(f"{current_turn['role']}: {current_turn['content']}")
    if index + 1 < len(turns):
        next_turn = turns[index + 1]
        snippets.append(f"{next_turn['role']}: {next_turn['content']}")
    return "\n".join(snippets)


def _normalize_annotation(annotation: dict, content: str) -> dict | None:
    pii_type = str(annotation.get("pii_type", "")).upper()
    if pii_type not in SUPPORTED_PII_TYPES:
        return None
    start = int(annotation["start"])
    end = int(annotation["end"])
    return {
        "start": start,
        "end": end,
        "text": content[start:end],
        "pii_type": pii_type,
    }


def _split_sessions(session_ids: list[str], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, str]:
    shuffled = session_ids[:]
    random.Random(seed).shuffle(shuffled)
    train_cutoff = int(len(shuffled) * train_ratio)
    dev_cutoff = train_cutoff + int(len(shuffled) * dev_ratio)
    split_by_session: dict[str, str] = {}
    for session_id in shuffled[:train_cutoff]:
        split_by_session[session_id] = "train"
    for session_id in shuffled[train_cutoff:dev_cutoff]:
        split_by_session[session_id] = "dev"
    for session_id in shuffled[dev_cutoff:]:
        split_by_session[session_id] = "test"
    return split_by_session


def _downsample_negatives(rows: list[dict], ratio: float, seed: int) -> list[dict]:
    if ratio < 0:
        return rows
    positives = [row for row in rows if any(label != "O" for label in row["labels"])]
    negatives = [row for row in rows if all(label == "O" for label in row["labels"])]
    max_negatives = int(len(positives) * ratio)
    if max_negatives <= 0 or len(negatives) <= max_negatives:
        return rows
    sampled_negatives = random.Random(seed).sample(negatives, max_negatives)
    combined = positives + sampled_negatives
    combined.sort(key=lambda row: row["id"])
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Import UPchieve math annotations into repo JSONL schema.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/Users/chason/Downloads/ground-truth-2026-03-11T04-17-06-314Z/DeID_GT_UPchieve_math_1000transcripts (1).jsonl"),
    )
    parser.add_argument("--subject", default="math")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--train-negative-ratio", type=float, default=3.0)
    parser.add_argument(
        "--interim-output",
        type=Path,
        default=INTERIM_DIR / "upchieve_math_sessions.jsonl",
        help="Session-level normalized export used for previews and later conversion work.",
    )
    parser.add_argument(
        "--write-provisional-action",
        action="store_true",
        help="Also emit provisional action splits with action_label=REDACT for all labeled spans.",
    )
    args = parser.parse_args()

    ensure_repo_layout()

    sessions: list[dict] = []
    session_ids: list[str] = []
    pii_counts: Counter[str] = Counter()
    turn_count = 0
    positive_turn_count = 0
    action_rows_by_split = {"train": [], "dev": [], "test": []}
    candidate_rows_by_split = {"train": [], "dev": [], "test": []}

    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            turns: list[dict] = []
            transcript_turns = sorted(payload["transcript"], key=lambda item: int(item.get("sequence_id", 0)))
            if not transcript_turns:
                continue
            session_id = str(transcript_turns[0]["session_id"])
            session_ids.append(session_id)
            for turn in transcript_turns:
                turn_count += 1
                content = str(turn.get("content", ""))
                normalized_annotations = []
                for annotation in turn.get("annotations", []):
                    normalized = _normalize_annotation(annotation, content)
                    if normalized is None:
                        continue
                    normalized_annotations.append(normalized)
                    pii_counts[normalized["pii_type"]] += 1
                if normalized_annotations:
                    positive_turn_count += 1
                turns.append(
                    {
                        "_id": str(turn.get("_id", "")),
                        "role": str(turn.get("role", "unknown")),
                        "content": content,
                        "sequence_id": int(turn.get("sequence_id", 0)),
                        "annotations": normalized_annotations,
                    }
                )
            sessions.append(
                {
                    "session_id": session_id,
                    "subject": args.subject,
                    "lead_role": payload.get("leadRole"),
                    "turns": turns,
                }
            )

    split_by_session = _split_sessions(session_ids, args.seed, args.train_ratio, args.dev_ratio)

    for session in sessions:
        split = split_by_session[session["session_id"]]
        turns = session["turns"]
        for index, turn in enumerate(turns):
            tokens, token_spans = _tokenize_with_spans(turn["content"])
            labels = _build_bio_labels(token_spans, turn["annotations"])
            candidate_rows_by_split[split].append(
                {
                    "id": f"{session['session_id']}-turn-{turn['sequence_id']}",
                    "subject": args.subject,
                    "tokens": tokens,
                    "labels": labels,
                    "dialogue_id": session["session_id"],
                    "speaker_role": turn["role"],
                    "context_text": _context_window(turns, index),
                    "metadata": {
                        "source": "upchieve_math_jsonl",
                        "lead_role": session.get("lead_role"),
                        "has_positive_label": any(label != "O" for label in labels),
                        "raw_turn_id": turn["_id"],
                    },
                }
            )

            if not args.write_provisional_action:
                continue
            for annotation_index, annotation in enumerate(turn["annotations"], start=1):
                action_rows_by_split[split].append(
                    {
                        "id": f"{session['session_id']}-turn-{turn['sequence_id']}-span-{annotation_index}",
                        "subject": args.subject,
                        "span_text": annotation["text"],
                        "context_text": _context_window(turns, index),
                        "action_label": "REDACT",
                        "speaker_role": turn["role"],
                        "entity_type": annotation["pii_type"],
                        "semantic_role": "PRIVATE",
                        "dialogue_id": session["session_id"],
                        "metadata": {
                            "source": "upchieve_math_jsonl",
                            "provisional_action": True,
                            "label_source": "legacy_redact_only",
                            "start": annotation["start"],
                            "end": annotation["end"],
                        },
                    }
                )

    candidate_rows_by_split["train"] = _downsample_negatives(
        candidate_rows_by_split["train"],
        args.train_negative_ratio,
        args.seed,
    )

    for split, rows in candidate_rows_by_split.items():
        rows.sort(key=lambda row: row["id"])
        _write_jsonl(CANDIDATE_DIR / f"{split}.jsonl", rows)

    if args.write_provisional_action:
        for split, rows in action_rows_by_split.items():
            rows.sort(key=lambda row: row["id"])
            _write_jsonl(ACTION_DIR / f"{split}.jsonl", rows)

    sessions.sort(key=lambda session: session["session_id"])
    _write_jsonl(args.interim_output, sessions)

    summary = {
        "input": str(args.input),
        "subject": args.subject,
        "session_count": len(sessions),
        "turn_count": turn_count,
        "positive_turn_count": positive_turn_count,
        "pii_counts": dict(sorted(pii_counts.items())),
        "candidate_split_sizes": {split: len(rows) for split, rows in candidate_rows_by_split.items()},
        "provisional_action_split_sizes": {split: len(rows) for split, rows in action_rows_by_split.items()},
        "train_negative_ratio": args.train_negative_ratio,
        "write_provisional_action": args.write_provisional_action,
        "interim_output": str(args.interim_output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
