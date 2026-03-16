from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import INTERIM_DIR
from contextshift_deid.data import ensure_repo_layout
from contextshift_deid.surrogates import SessionSurrogateMapper, replace_tags
from contextshift_deid.upchieve_pilot import (
    SUPPORTED_TAGS,
    build_anchor_text,
    canonical_subject,
    challenge_score,
    extract_supported_tags,
    format_context_window,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize anonymized UPChieve English/Social Studies turns for the action pilot.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=ROOT / "UPChive-all" / "sessions_transcript_20260126_113353_CC-BY.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=INTERIM_DIR / "upchieve_context_pilot_turns.jsonl",
    )
    parser.add_argument(
        "--surrogate-mode",
        choices=["replace", "none"],
        default="replace",
        help="'replace' swaps <TAG> placeholders with neutral surrogates; 'none' keeps raw tags (backward compatible).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_repo_layout()

    rows: list[dict[str, Any]] = []
    session_counts: Counter[str] = Counter()
    tagged_turn_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    use_surrogates = args.surrogate_mode == "replace"
    source_label = "upchieve_surrogate_view" if use_surrogates else "upchieve_all_anonymized"
    surrogate_source = "derived_from_anonymized_tags" if use_surrogates else "raw_anonymized_tags"

    with args.input_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            topic_name = str(payload.get("topic_name", ""))
            subject = canonical_subject(topic_name)
            if subject is None:
                continue

            transcript = payload.get("anonymized_transcript")
            if not isinstance(transcript, list) or not transcript:
                continue

            normalized_turns: list[dict[str, Any]] = [
                {
                    "message": str(turn.get("message", "")),
                    "message_type": str(turn.get("message_type", "unknown")),
                    "timestamp_seconds": turn.get("timestamp_seconds"),
                    "user_role": str(turn.get("user_role", "unknown")),
                }
                for turn in transcript
            ]
            session_id = str(payload.get("session_id"))

            # ----- surrogate replacement (or pass-through) -----
            if use_surrogates:
                mapper = SessionSurrogateMapper(seed=args.seed, session_id=session_id)
                display_turns: list[dict[str, Any]] = []
                turn_spans: list[list[dict[str, Any]]] = []
                for turn in normalized_turns:
                    replaced_text, spans = replace_tags(turn["message"], mapper)
                    display_turns.append({
                        **turn,
                        "message": replaced_text,
                        "message_original": turn["message"],
                    })
                    turn_spans.append(
                        [s for s in spans if s["entity_type"] in SUPPORTED_TAGS]
                    )
            else:
                display_turns = normalized_turns
                turn_spans = [
                    extract_supported_tags(turn["message"])
                    for turn in normalized_turns
                ]

            anchor_text = build_anchor_text(display_turns)
            session_recorded = False

            for index, turn in enumerate(display_turns):
                tags = turn_spans[index]
                if not tags:
                    continue
                if not session_recorded:
                    session_counts[subject] += 1
                    session_recorded = True

                qualifying_tag_count = sum(
                    1 for tag in tags
                    if tag["entity_type"] in {"PERSON", "LOCATION", "NRP", "SCHOOL", "COURSE"}
                )
                row: dict[str, Any] = {
                    "id": f"{session_id}-turn-{index}",
                    "session_id": session_id,
                    "subject": subject,
                    "topic_name": topic_name,
                    "subject_name": str(payload.get("subject_name", "")),
                    "turn_index": index,
                    "speaker_role": turn["user_role"],
                    "message_type": turn["message_type"],
                    "turn_text": turn["message"],
                    "context_text": format_context_window(display_turns, index),
                    "anchor_text": anchor_text,
                    "tags": [
                        {
                            **tag,
                            "challenge_score": challenge_score(
                                entity_type=str(tag["entity_type"]),
                                qualifying_tag_count=qualifying_tag_count,
                                speaker_role=turn["user_role"],
                            ),
                        }
                        for tag in tags
                    ],
                    "metadata": {
                        "source": source_label,
                        "surrogate_mode": args.surrogate_mode,
                        "surrogate_source": surrogate_source,
                        "session_line_number": line_number,
                        "timestamp_seconds": turn.get("timestamp_seconds"),
                        "qualifying_tag_count": qualifying_tag_count,
                    },
                }
                if use_surrogates:
                    row["turn_text_original"] = turn.get("message_original", turn["message"])
                    row["metadata"]["surrogate_seed"] = args.seed
                rows.append(row)
                tagged_turn_counts[subject] += 1
                for tag in tags:
                    tag_counts[f"{subject}:{tag['entity_type']}"] += 1

    _write_jsonl(args.output_file, rows)
    print(
        json.dumps(
            {
                "input_file": str(args.input_file),
                "output_file": str(args.output_file),
                "surrogate_mode": args.surrogate_mode,
                "seed": args.seed,
                "row_count": len(rows),
                "sessions_by_subject": session_counts,
                "tagged_turns_by_subject": tagged_turn_counts,
                "tags_by_subject_entity": tag_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
