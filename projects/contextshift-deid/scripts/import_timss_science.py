from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ANNOTATION_DIR, INTERIM_DIR
from contextshift_deid.data import ensure_repo_layout, validate_candidate_records
from contextshift_deid.timss import (
    TimssTranscript,
    discover_timss_transcript_paths,
    format_context_window,
    parse_timss_transcript,
)
from contextshift_deid.tokenization import tokenize_text


def _default_input_root() -> Path:
    candidates = (
        ROOT / "data" / "raw" / "timss",
        ROOT / "TIMSS",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / "TIMSS"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _split_dialogues(dialogue_ids: list[str], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, str]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= dev_ratio <= 1.0:
        raise ValueError("dev_ratio must be between 0 and 1")
    if train_ratio + dev_ratio > 1.0:
        raise ValueError("train_ratio + dev_ratio must be <= 1")

    shuffled = dialogue_ids[:]
    random.Random(seed).shuffle(shuffled)
    train_cutoff = int(len(shuffled) * train_ratio)
    dev_cutoff = train_cutoff + int(len(shuffled) * dev_ratio)
    split_by_dialogue: dict[str, str] = {}
    for dialogue_id in shuffled[:train_cutoff]:
        split_by_dialogue[dialogue_id] = "train"
    for dialogue_id in shuffled[train_cutoff:dev_cutoff]:
        split_by_dialogue[dialogue_id] = "dev"
    for dialogue_id in shuffled[dev_cutoff:]:
        split_by_dialogue[dialogue_id] = "test"
    return split_by_dialogue


def _candidate_row(transcript: TimssTranscript, turn_index: int, split_name: str) -> dict:
    turn = transcript.turns[turn_index]
    tokens = tokenize_text(turn.text)
    return {
        "id": f"{transcript.dialogue_id}-turn-{turn_index + 1:04d}",
        "subject": transcript.subject,
        "tokens": tokens,
        "labels": ["O"] * len(tokens),
        "anchor_text": turn.text,
        "dialogue_id": transcript.dialogue_id,
        "speaker_role": turn.speaker_role,
        "context_text": format_context_window(transcript.turns, turn_index),
        "metadata": {
            "source": "timss_science_txt",
            "country": transcript.country,
            "transcript_code": transcript.transcript_code,
            "speaker_tag": turn.speaker_tag,
            "timestamp": turn.timestamp,
            "source_path": str(transcript.source_path),
            "split": split_name,
            "annotation_completed": False,
        },
    }


def _interim_row(transcript: TimssTranscript, turn_index: int, split_name: str) -> dict:
    turn = transcript.turns[turn_index]
    return {
        "id": f"{transcript.dialogue_id}-turn-{turn_index + 1:04d}",
        "subject": transcript.subject,
        "dialogue_id": transcript.dialogue_id,
        "country": transcript.country,
        "transcript_code": transcript.transcript_code,
        "split": split_name,
        "turn_index": turn_index,
        "timestamp": turn.timestamp,
        "speaker_tag": turn.speaker_tag,
        "speaker_role": turn.speaker_role,
        "anchor_text": turn.text,
        "context_text": format_context_window(transcript.turns, turn_index),
        "source_path": str(transcript.source_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize TIMSS science transcripts into candidate-annotation seed files."
    )
    parser.add_argument("--input-root", type=Path, default=_default_input_root())
    parser.add_argument("--subject", default="science")
    parser.add_argument("--countries", nargs="+", default=["AU", "US"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument(
        "--interim-output",
        type=Path,
        default=INTERIM_DIR / "timss_science_turns.jsonl",
        help="Normalized per-turn export for audit and downstream conversion work.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANNOTATION_DIR / "timss_science",
        help="Directory that will receive candidate_pool_<split>.jsonl files and summary.json.",
    )
    args = parser.parse_args()

    ensure_repo_layout()
    if not args.input_root.exists():
        raise SystemExit(f"TIMSS input root does not exist: {args.input_root}")

    transcript_paths = discover_timss_transcript_paths(args.input_root, args.countries)
    if not transcript_paths:
        raise SystemExit(
            f"No TIMSS transcripts found under {args.input_root} for countries: {', '.join(args.countries)}"
        )

    transcripts = [parse_timss_transcript(path, subject=args.subject) for path in transcript_paths]
    split_by_dialogue = _split_dialogues(
        [transcript.dialogue_id for transcript in transcripts],
        args.seed,
        args.train_ratio,
        args.dev_ratio,
    )

    interim_rows: list[dict] = []
    candidate_rows_by_split = {"train": [], "dev": [], "test": []}
    transcript_counts: Counter[str] = Counter()
    country_counts: Counter[str] = Counter()
    turn_counts: Counter[str] = Counter()

    for transcript in transcripts:
        split_name = split_by_dialogue[transcript.dialogue_id]
        transcript_counts[split_name] += 1
        country_counts[transcript.country] += 1
        for turn_index, turn in enumerate(transcript.turns):
            tokens = tokenize_text(turn.text)
            if not tokens:
                continue
            candidate_rows_by_split[split_name].append(_candidate_row(transcript, turn_index, split_name))
            interim_rows.append(_interim_row(transcript, turn_index, split_name))
            turn_counts[split_name] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path] = {}
    for split_name, rows in candidate_rows_by_split.items():
        rows.sort(key=lambda row: row["id"])
        output_path = args.output_dir / f"candidate_pool_{split_name}.jsonl"
        _write_jsonl(output_path, rows)
        validate_candidate_records(output_path)
        output_files[split_name] = output_path

    interim_rows.sort(key=lambda row: row["id"])
    _write_jsonl(args.interim_output, interim_rows)

    summary = {
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "interim_output": str(args.interim_output),
        "countries": [country.upper() for country in args.countries],
        "transcripts": len(transcripts),
        "transcript_counts_by_split": dict(transcript_counts),
        "turn_counts_by_split": dict(turn_counts),
        "country_counts": dict(country_counts),
        "output_files": {split_name: str(path) for split_name, path in output_files.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
