from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ANNOTATION_DIR, INTERIM_DIR
from contextshift_deid.data import ensure_repo_layout, load_jsonl
from contextshift_deid.upchieve_pilot import (
    AMBIGUOUS_CHALLENGE_TAGS,
    CHALLENGE_TAGS,
    CONTROL_TAGS,
    bucket_sessions,
    proportional_targets,
    stable_sort_key,
)

PILOT_SUBJECTS = ("english", "social_studies")
SPLIT_TARGETS = {
    "natural": 50,
    "challenge": 50,
}
DEFAULT_MAX_TURN_WORDS = 350
DEFAULT_MAX_QUALIFYING_TAGS = 20
NON_ENGLISH_SESSION_MARKER_RE = re.compile(r"\b(spanish|espanol|español)\b", re.IGNORECASE)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _word_count(text: str) -> int:
    return len(str(text).split())


def _load_items(
    path: Path,
    *,
    max_turn_words: int,
    max_qualifying_tags: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    loaded_turns = 0
    skipped_turns = 0
    skipped_by_subject: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    for row in load_jsonl(path):
        loaded_turns += 1
        metadata = row.get("metadata") or {}
        qualifying_tag_count = int(metadata.get("qualifying_tag_count", 0))
        turn_text = str(row["turn_text"])
        turn_word_count = _word_count(turn_text)
        skip_reasons: list[str] = []
        if qualifying_tag_count > max_qualifying_tags:
            skip_reasons.append("qualifying_tag_count")
        if turn_word_count > max_turn_words:
            skip_reasons.append("turn_word_count")
        if skip_reasons:
            skipped_turns += 1
            skipped_by_subject[str(row["subject"])] += 1
            skip_reason_counts.update(skip_reasons)
            continue
        for tag in row.get("tags", []):
            items.append(
                {
                    "id": f"{row['id']}-tag-{tag['tag_start']}-{tag['tag_end']}",
                    "source_row_id": str(row["id"]),
                    "session_id": str(row["session_id"]),
                    "subject": str(row["subject"]),
                    "topic_name": str(row["topic_name"]),
                    "subject_name": str(row.get("subject_name", "")),
                    "turn_index": int(row["turn_index"]),
                    "speaker_role": str(row["speaker_role"]),
                    "message_type": str(row["message_type"]),
                    "turn_text": turn_text,
                    "context_text": str(row["context_text"]),
                    "anchor_text": str(row["anchor_text"]),
                    "span_text": str(tag["span_text"]),
                    "original_tag": str(tag.get("original_tag", tag["span_text"])),
                    "entity_type": str(tag["entity_type"]),
                    "tag_start": int(tag["tag_start"]),
                    "tag_end": int(tag["tag_end"]),
                    "tag_occurrence": int(tag["tag_occurrence"]),
                    "challenge_score": int(tag["challenge_score"]),
                    "qualifying_tag_count": qualifying_tag_count,
                    "turn_word_count": turn_word_count,
                    "session_line_number": metadata.get("session_line_number"),
                    "turn_text_original": str(row.get("turn_text_original", row["turn_text"])),
                    "source": str(metadata.get("source", "upchieve_all_anonymized")),
                    "surrogate_mode": str(metadata.get("surrogate_mode", "none")),
                    "surrogate_source": str(metadata.get("surrogate_source", "raw_anonymized_tags")),
                    "surrogate_seed": metadata.get("surrogate_seed"),
                }
            )
    return items, {
        "loaded_turns": loaded_turns,
        "kept_turns": loaded_turns - skipped_turns,
        "skipped_turns": skipped_turns,
        "skipped_turns_by_subject": dict(skipped_by_subject),
        "skip_reason_counts": dict(skip_reason_counts),
        "max_turn_words": max_turn_words,
        "max_qualifying_tags": max_qualifying_tags,
    }


def _language_marker_text(item: dict[str, Any]) -> str:
    return " ".join(
        (
            str(item.get("anchor_text", "")),
            str(item.get("context_text", "")),
            str(item.get("turn_text", "")),
            str(item.get("turn_text_original", "")),
        )
    )


def _exclude_marked_english_sessions(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_session_ids = {
        str(item["session_id"])
        for item in items
        if str(item["subject"]) == "english" and NON_ENGLISH_SESSION_MARKER_RE.search(_language_marker_text(item))
    }
    filtered_items = [
        item
        for item in items
        if not (str(item["subject"]) == "english" and str(item["session_id"]) in excluded_session_ids)
    ]
    removed_items = len(items) - len(filtered_items)
    removed_sessions = len(excluded_session_ids)
    return filtered_items, {
        "excluded_english_sessions": removed_sessions,
        "excluded_english_session_ids": sorted(excluded_session_ids),
        "removed_items": removed_items,
        "marker_pattern": NON_ENGLISH_SESSION_MARKER_RE.pattern,
    }


def _natural_targets(items: list[dict[str, Any]], *, total: int) -> dict[str, int]:
    counts: Counter[str] = Counter(str(item["entity_type"]) for item in items)
    targets: dict[str, int] = {}
    reserved_total = 0
    for entity_type in sorted(CONTROL_TAGS):
        if counts.get(entity_type, 0) > 0 and reserved_total < total:
            targets[entity_type] = 1
            reserved_total += 1

    remaining_total = total - reserved_total
    remaining_counts = Counter(
        {
            entity_type: count
            for entity_type, count in counts.items()
            if entity_type not in targets
        }
    )
    proportional = proportional_targets(remaining_counts, total=remaining_total)
    for entity_type, count in proportional.items():
        targets[entity_type] = targets.get(entity_type, 0) + count
    return targets


def _pick_next_available(
    items_by_entity: dict[str, list[dict[str, Any]]],
    *,
    entity_type: str,
    used_ids: set[str],
    session_counts: Counter[str],
    session_cap: int,
) -> dict[str, Any] | None:
    for item in items_by_entity.get(entity_type, []):
        if item["id"] in used_ids:
            continue
        if session_counts[item["session_id"]] >= session_cap:
            continue
        return item
    return None


def _select_natural(items: list[dict[str, Any]], *, total: int, session_cap: int, seed: int) -> list[dict[str, Any]]:
    targets = _natural_targets(items, total=total)
    items_by_entity: dict[str, list[dict[str, Any]]] = {
        entity_type: sorted(
            [item for item in items if item["entity_type"] == entity_type],
            key=lambda item: stable_sort_key(item["id"], seed=seed),
        )
        for entity_type in sorted({item["entity_type"] for item in items})
    }
    source_counts: Counter[str] = Counter(item["entity_type"] for item in items)
    current_counts: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    used_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    while len(selected) < total:
        progress = False
        for entity_type in sorted(
            targets,
            key=lambda value: (
                -(targets[value] - current_counts[value]),
                -source_counts[value],
                value,
            ),
        ):
            if current_counts[entity_type] >= targets[entity_type]:
                continue
            item = _pick_next_available(
                items_by_entity,
                entity_type=entity_type,
                used_ids=used_ids,
                session_counts=session_counts,
                session_cap=session_cap,
            )
            if item is None:
                continue
            selected.append(item)
            used_ids.add(item["id"])
            session_counts[item["session_id"]] += 1
            current_counts[entity_type] += 1
            progress = True
            if len(selected) >= total:
                break
        if not progress:
            break

    if len(selected) < total:
        remaining = sorted(items, key=lambda item: stable_sort_key(item["id"], seed=seed))
        for item in remaining:
            if item["id"] in used_ids:
                continue
            if session_counts[item["session_id"]] >= session_cap:
                continue
            selected.append(item)
            used_ids.add(item["id"])
            session_counts[item["session_id"]] += 1
            current_counts[item["entity_type"]] += 1
            if len(selected) >= total:
                break

    if len(selected) != total:
        raise SystemExit(f"Natural slice selection failed: expected {total} rows, found {len(selected)}")
    return selected


def _select_challenge(
    items: list[dict[str, Any]],
    *,
    total: int,
    session_cap: int,
    seed: int,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    challenge_items = [
        item
        for item in items
        if item["entity_type"] in CHALLENGE_TAGS and item["id"] not in excluded_ids
    ]
    ordered = sorted(
        challenge_items,
        key=lambda item: (
            -int(item["challenge_score"]),
            0 if item["entity_type"] in AMBIGUOUS_CHALLENGE_TAGS else 1,
            stable_sort_key(item["id"], seed=seed),
        ),
    )
    session_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for item in ordered:
        if session_counts[item["session_id"]] >= session_cap:
            continue
        selected.append(item)
        session_counts[item["session_id"]] += 1
        if len(selected) >= total:
            break
    if len(selected) != total:
        raise SystemExit(f"Challenge slice selection failed: expected {total} rows, found {len(selected)}")
    return selected


def _suggested_action(item: dict[str, Any], *, eval_slice: str) -> tuple[str, str]:
    entity_type = str(item["entity_type"])
    subject = str(item["subject"])
    if entity_type in CONTROL_TAGS:
        return ("REDACT", "Direct privacy-control tag in anonymized text; confirm it remains a true REDACT case.")
    if subject == "social_studies" and entity_type in {"PERSON", "LOCATION", "NRP"}:
        return ("KEEP", "Historical or civic entity in Social Studies is often curricular and should usually remain visible.")
    if subject == "english" and entity_type == "COURSE":
        return ("KEEP", "Course or assignment references in English are usually instructional rather than private.")
    if subject == "english" and entity_type in {"PERSON", "LOCATION"}:
        return ("REVIEW", "English person/location tags can be literary, curricular, or private depending on context.")
    if eval_slice == "challenge":
        return ("REVIEW", "Challenge slice prioritizes ambiguous curricular entities that need action adjudication.")
    return ("REVIEW", "Natural slice sample from anonymized UPChieve tags; verify whether this tag is curricular or private.")


def _build_pool_row(item: dict[str, Any], *, split_name: str, eval_slice: str) -> dict[str, Any]:
    suggested_action, suggested_reason = _suggested_action(item, eval_slice=eval_slice)
    pool_source = f"upchieve_{eval_slice}"
    row = {
        "id": item["id"],
        "split": split_name,
        "source_row_id": item["source_row_id"],
        "subject": item["subject"],
        "eval_slice": eval_slice,
        "span_text": item["span_text"],
        "original_tag": item["original_tag"],
        "context_text": item["context_text"],
        "anchor_text": item["anchor_text"],
        "dialogue_id": item["session_id"],
        "speaker_role": item["speaker_role"],
        "entity_type": item["entity_type"],
        "pool_source": pool_source,
        "suggested_action": suggested_action,
        "suggested_reason": suggested_reason,
        "metadata": {
            "source": item["source"],
            "surrogate_mode": item["surrogate_mode"],
            "surrogate_source": item["surrogate_source"],
            "topic_name": item["topic_name"],
            "subject_name": item["subject_name"],
            "turn_index": item["turn_index"],
            "message_type": item["message_type"],
            "turn_text": item["turn_text"],
            "turn_text_original": item["turn_text_original"],
            "tag_start": item["tag_start"],
            "tag_end": item["tag_end"],
            "tag_occurrence": item["tag_occurrence"],
            "challenge_score": item["challenge_score"],
            "qualifying_tag_count": item["qualifying_tag_count"],
            "turn_word_count": item["turn_word_count"],
            "session_line_number": item["session_line_number"],
            "eval_slice": eval_slice,
        },
    }
    if item["surrogate_seed"] is not None:
        row["metadata"]["surrogate_seed"] = int(item["surrogate_seed"])
    return row


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_subject = Counter(str(row["subject"]) for row in rows)
    counts_by_eval_slice = Counter(str(row["eval_slice"]) for row in rows)
    counts_by_entity = Counter(f"{row['subject']}:{row['entity_type']}" for row in rows)
    sessions_per_slice: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sessions_per_slice[str(row["eval_slice"])].add(str(row["dialogue_id"]))
    return {
        "row_count": len(rows),
        "by_subject": counts_by_subject,
        "by_eval_slice": counts_by_eval_slice,
        "by_subject_entity": counts_by_entity,
        "sessions_per_eval_slice": {key: len(value) for key, value in sessions_per_slice.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build English/Social Studies UPChieve action pilot annotation pools.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INTERIM_DIR / "upchieve_context_pilot_turns.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANNOTATION_DIR / "upchieve_context_pilot",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turn-words", type=int, default=DEFAULT_MAX_TURN_WORDS)
    parser.add_argument("--max-qualifying-tags", type=int, default=DEFAULT_MAX_QUALIFYING_TAGS)
    args = parser.parse_args()

    ensure_repo_layout()
    items, filter_summary = _load_items(
        args.input_file,
        max_turn_words=args.max_turn_words,
        max_qualifying_tags=args.max_qualifying_tags,
    )
    items, language_filter_summary = _exclude_marked_english_sessions(items)

    split_by_session: dict[str, str] = {}
    for subject in PILOT_SUBJECTS:
        subject_sessions = {item["session_id"] for item in items if item["subject"] == subject}
        split_by_session.update(bucket_sessions(subject_sessions, seed=args.seed))

    split_rows: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    manifest: dict[str, Any] = {
        "input_file": str(args.input_file),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "filter_summary": filter_summary,
        "language_filter_summary": language_filter_summary,
        "splits": {},
    }

    for split_name in ("dev", "test"):
        pool_rows: list[dict[str, Any]] = []
        for subject in PILOT_SUBJECTS:
            subject_items = [
                item
                for item in items
                if item["subject"] == subject and split_by_session.get(item["session_id"]) == split_name
            ]
            natural_rows = _select_natural(
                subject_items,
                total=SPLIT_TARGETS["natural"],
                session_cap=1,
                seed=args.seed,
            )
            challenge_rows = _select_challenge(
                subject_items,
                total=SPLIT_TARGETS["challenge"],
                session_cap=2,
                seed=args.seed,
                excluded_ids={row["id"] for row in natural_rows},
            )
            pool_rows.extend(_build_pool_row(row, split_name=f"upchieve_english_social_{split_name}", eval_slice="natural") for row in natural_rows)
            pool_rows.extend(_build_pool_row(row, split_name=f"upchieve_english_social_{split_name}", eval_slice="challenge") for row in challenge_rows)

        ordered_rows = sorted(
            pool_rows,
            key=lambda row: (
                str(row["subject"]),
                str(row["eval_slice"]),
                stable_sort_key(str(row["id"]), seed=args.seed),
            ),
        )
        split_rows[split_name] = ordered_rows
        output_file = args.output_dir / f"action_pool_upchieve_english_social_{split_name}.jsonl"
        _write_jsonl(output_file, ordered_rows)
        manifest["splits"][split_name] = {
            "output_file": str(output_file),
            **_split_summary(ordered_rows),
        }

    _write_json(args.output_dir / "summary.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
