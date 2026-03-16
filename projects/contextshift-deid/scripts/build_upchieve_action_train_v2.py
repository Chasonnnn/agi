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

from contextshift_deid.constants import ACTION_DIR, ANNOTATION_DIR, INTERIM_DIR
from contextshift_deid.data import ensure_repo_layout, load_jsonl, validate_action_records
from contextshift_deid.upchieve_pilot import (
    AMBIGUOUS_CHALLENGE_TAGS,
    CHALLENGE_TAGS,
    CONTROL_TAGS,
    proportional_targets,
    stable_sort_key,
)

TRAIN_SUBJECTS = ("english", "social_studies")
DEFAULT_SUBJECT_TOTAL = 200
DEFAULT_SLICE_TOTAL = 100
DEFAULT_PRIVATE_PRIOR_RESERVE = 25
DEFAULT_ENGLISH_PERSON_PRIVATE_RESERVE = 15
DEFAULT_MAX_TURN_WORDS = 350
DEFAULT_MAX_QUALIFYING_TAGS = 20
NON_ENGLISH_SESSION_MARKER_RE = re.compile(
    r"\b(spanish|espanol|español|arabic|persian|farsi|urdu)\b",
    re.IGNORECASE,
)
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
GREETING_SPAN_TEMPLATE = r"\b(?:hi|hello|hey)\s+{span}\b"
SELF_IDENTIFY_TEMPLATE = r"\b(?:my name is|i am|i'm|im)\s+{span}\b"
DIRECT_ADDRESS_TEMPLATE = r"(?:,\s*{span}\b|\b{span}\s*[,!?:])"
CONTACT_CUE_RE = re.compile(r"\b(?:text|email|call)\b", re.IGNORECASE)
SCHOOL_LOCATION_PRIVATE_RE = re.compile(
    r"\b(?:my school|i go to|i live in|i'm from|im from|i am from)\b",
    re.IGNORECASE,
)
ENGLISH_CURRICULAR_RE = re.compile(
    r"\b(?:essay|author|character|novel|story|poem|book|passage|article|quote|speaker|paragraph|chapter|text)\b",
    re.IGNORECASE,
)
SOCIAL_STUDIES_CURRICULAR_RE = re.compile(
    r"\b(?:war|treaty|empire|colony|president|king|queen|nation|state|history|revolution|conference|government|foreign|world war|civil war)\b",
    re.IGNORECASE,
)
CHALLENGE_MINIMUMS: dict[str, dict[str, int]] = {
    "english": {"PERSON": 60, "LOCATION": 15},
    "social_studies": {"LOCATION": 45, "NRP": 25, "PERSON": 15},
}


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


def _turn_text_for_script_filter(item: dict[str, Any]) -> str:
    return " ".join(
        (
            str(item.get("turn_text", "")),
            str(item.get("turn_text_original", "")),
        )
    )


def _exclude_non_english_english_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
    arabic_script_item_ids = {
        str(item["id"])
        for item in filtered_items
        if str(item["subject"]) == "english" and ARABIC_SCRIPT_RE.search(_turn_text_for_script_filter(item))
    }
    filtered_items = [
        item
        for item in filtered_items
        if not (str(item["subject"]) == "english" and str(item["id"]) in arabic_script_item_ids)
    ]
    return filtered_items, {
        "excluded_english_sessions": len(excluded_session_ids),
        "excluded_english_session_ids": sorted(excluded_session_ids),
        "removed_by_session_marker": len(excluded_session_ids),
        "removed_english_items_by_arabic_script": len(arabic_script_item_ids),
        "removed_english_item_ids_by_arabic_script": sorted(arabic_script_item_ids),
        "marker_pattern": NON_ENGLISH_SESSION_MARKER_RE.pattern,
        "arabic_script_pattern": ARABIC_SCRIPT_RE.pattern,
    }


def _load_excluded_dialogue_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for record in validate_action_records(path):
            if record.dialogue_id is None:
                raise SystemExit(f"{path} is missing dialogue_id for row {record.id}")
            excluded.add(str(record.dialogue_id))
    return excluded


def _combined_selection_text(item: dict[str, Any]) -> str:
    return " ".join(
        (
            str(item.get("turn_text", "")),
            str(item.get("context_text", "")),
            str(item.get("anchor_text", "")),
            str(item.get("turn_text_original", "")),
        )
    )


def _span_pattern(template: str, span_text: str) -> re.Pattern[str]:
    span = re.escape(span_text.strip())
    return re.compile(template.format(span=span), re.IGNORECASE)


def _english_person_private_score(item: dict[str, Any]) -> int:
    if str(item["subject"]) != "english" or str(item["entity_type"]) != "PERSON":
        return 0
    span_text = str(item["span_text"]).strip()
    if not span_text:
        return 0
    combined_text = _combined_selection_text(item)
    turn_text = str(item["turn_text"])
    greeting_match = _span_pattern(GREETING_SPAN_TEMPLATE, span_text).search(combined_text)
    self_identify_match = _span_pattern(SELF_IDENTIFY_TEMPLATE, span_text).search(combined_text)
    direct_address_match = _span_pattern(DIRECT_ADDRESS_TEMPLATE, span_text).search(turn_text)
    qualifies = greeting_match or self_identify_match or direct_address_match
    if not qualifies:
        return 0
    score = 0
    if greeting_match:
        score += 5
    if self_identify_match:
        score += 5
    if direct_address_match:
        score += 2
    if int(item["turn_index"]) < 5:
        score += 1
    return score


def _private_prior_score(item: dict[str, Any]) -> int:
    entity_type = str(item["entity_type"])
    text = _combined_selection_text(item)
    score = _english_person_private_score(item)
    if entity_type in {"SCHOOL", "LOCATION"} and SCHOOL_LOCATION_PRIVATE_RE.search(text):
        score += 2
    if CONTACT_CUE_RE.search(text):
        score += 1
    if entity_type in CONTROL_TAGS:
        score += 1
    return score


def _curricular_cue_bonus(item: dict[str, Any]) -> int:
    subject = str(item["subject"])
    text = _combined_selection_text(item)
    if subject == "english":
        matches = ENGLISH_CURRICULAR_RE.findall(text)
    elif subject == "social_studies":
        matches = SOCIAL_STUDIES_CURRICULAR_RE.findall(text)
    else:
        matches = []
    return min(2, len(matches))


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


def _select_natural(
    items: list[dict[str, Any]],
    *,
    subject: str,
    total: int,
    private_prior_reserve: int,
    english_person_private_reserve: int,
    session_cap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    english_person_private_candidates = [
        item
        for item in items
        if _english_person_private_score(item) > 0
    ]
    english_person_private_candidates = sorted(
        english_person_private_candidates,
        key=lambda item: (
            -_english_person_private_score(item),
            stable_sort_key(item["id"], seed=seed),
        ),
    )
    english_person_private_selected = 0
    if subject == "english":
        for item in english_person_private_candidates:
            if english_person_private_selected >= english_person_private_reserve:
                break
            if item["id"] in used_ids:
                continue
            if session_counts[item["session_id"]] >= session_cap:
                continue
            selected.append(item)
            used_ids.add(item["id"])
            session_counts[item["session_id"]] += 1
            english_person_private_selected += 1

    private_candidates = [
        item
        for item in items
        if _private_prior_score(item) > 0 and item["id"] not in used_ids
    ]
    private_candidates = sorted(
        private_candidates,
        key=lambda item: (
            -_private_prior_score(item),
            stable_sort_key(item["id"], seed=seed),
        ),
    )
    private_selected = len(selected)
    for item in private_candidates:
        if private_selected >= private_prior_reserve:
            break
        if item["id"] in used_ids:
            continue
        if session_counts[item["session_id"]] >= session_cap:
            continue
        selected.append(item)
        used_ids.add(item["id"])
        session_counts[item["session_id"]] += 1
        private_selected += 1

    if private_selected < private_prior_reserve:
        control_candidates = [
            item
            for item in items
            if item["id"] not in used_ids and item["entity_type"] in CONTROL_TAGS
        ]
        control_candidates = sorted(
            control_candidates,
            key=lambda item: stable_sort_key(item["id"], seed=seed),
        )
        for item in control_candidates:
            if private_selected >= private_prior_reserve:
                break
            if session_counts[item["session_id"]] >= session_cap:
                continue
            selected.append(item)
            used_ids.add(item["id"])
            session_counts[item["session_id"]] += 1
            private_selected += 1

    remaining_total = total - len(selected)
    remaining_items = [item for item in items if item["id"] not in used_ids]
    source_counts: Counter[str] = Counter(item["entity_type"] for item in remaining_items)
    targets = proportional_targets(source_counts, total=remaining_total)
    current_counts: Counter[str] = Counter()
    items_by_entity: dict[str, list[dict[str, Any]]] = {
        entity_type: sorted(
            [item for item in remaining_items if item["entity_type"] == entity_type],
            key=lambda item: stable_sort_key(item["id"], seed=seed),
        )
        for entity_type in sorted({item["entity_type"] for item in remaining_items})
    }

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
            if len(selected) >= total:
                break

    if len(selected) != total:
        raise SystemExit(f"Natural train slice selection failed: expected {total} rows, found {len(selected)}")

    return selected, {
        "row_count": len(selected),
        "private_prior_available": len([item for item in items if _private_prior_score(item) > 0]),
        "private_prior_selected": private_selected,
        "english_person_private_available": len(english_person_private_candidates),
        "english_person_private_selected": english_person_private_selected,
        "entity_targets": dict(targets),
        "entity_selected": dict(Counter(item["entity_type"] for item in selected)),
        "session_count": len(session_counts),
    }


def _select_challenge(
    items: list[dict[str, Any]],
    *,
    subject: str,
    total: int,
    session_cap: int,
    seed: int,
    excluded_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    challenge_items = [
        item
        for item in items
        if item["entity_type"] in CHALLENGE_TAGS and item["id"] not in excluded_ids
    ]
    ordered = sorted(
        challenge_items,
        key=lambda item: (
            -(int(item["challenge_score"]) + _curricular_cue_bonus(item)),
            0 if item["entity_type"] in AMBIGUOUS_CHALLENGE_TAGS else 1,
            stable_sort_key(item["id"], seed=seed),
        ),
    )
    ordered_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        ordered_by_entity[str(item["entity_type"])].append(item)

    quota_targets = CHALLENGE_MINIMUMS.get(subject, {})
    quota_selected: Counter[str] = Counter()
    session_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for entity_type, quota in quota_targets.items():
        for item in ordered_by_entity.get(entity_type, []):
            if quota_selected[entity_type] >= quota:
                break
            if item["id"] in selected_ids:
                continue
            if session_counts[item["session_id"]] >= session_cap:
                continue
            selected.append(item)
            selected_ids.add(item["id"])
            session_counts[item["session_id"]] += 1
            quota_selected[entity_type] += 1

    quota_underfill = {
        entity_type: quota_targets[entity_type] - quota_selected[entity_type]
        for entity_type in quota_targets
        if quota_selected[entity_type] < quota_targets[entity_type]
    }
    if quota_underfill:
        fill_candidates = sorted(challenge_items, key=lambda item: stable_sort_key(item["id"], seed=seed))
    else:
        fill_candidates = ordered

    for item in fill_candidates:
        if len(selected) >= total:
            break
        if item["id"] in selected_ids:
            continue
        if session_counts[item["session_id"]] >= session_cap:
            continue
        selected.append(item)
        selected_ids.add(item["id"])
        session_counts[item["session_id"]] += 1

    if len(selected) < total:
        backfill = sorted(challenge_items, key=lambda item: stable_sort_key(item["id"], seed=seed))
        for item in backfill:
            if len(selected) >= total:
                break
            if item["id"] in selected_ids:
                continue
            if session_counts[item["session_id"]] >= session_cap:
                continue
            selected.append(item)
            selected_ids.add(item["id"])
            session_counts[item["session_id"]] += 1

    if len(selected) != total:
        raise SystemExit(f"Challenge train slice selection failed: expected {total} rows, found {len(selected)}")

    cue_bonus_counts: Counter[int] = Counter(_curricular_cue_bonus(item) for item in selected)
    return selected, {
        "row_count": len(selected),
        "quota_targets": quota_targets,
        "quota_selected": dict(quota_selected),
        "quota_underfill": quota_underfill,
        "entity_selected": dict(Counter(item["entity_type"] for item in selected)),
        "curricular_cue_bonus_counts": dict(cue_bonus_counts),
        "session_count": len(session_counts),
        "used_stable_backfill_for_underfill": bool(quota_underfill),
    }


def _suggested_action(item: dict[str, Any], *, eval_slice: str) -> tuple[str, str]:
    entity_type = str(item["entity_type"])
    subject = str(item["subject"])
    if _english_person_private_score(item) > 0:
        return ("REDACT", "Greeting, direct address, or self-identification suggests this PERSON refers to a real participant.")
    if entity_type in CONTROL_TAGS:
        return ("REDACT", "Direct privacy-control tag in anonymized text; confirm it remains a true REDACT case.")
    if subject == "social_studies" and entity_type in {"PERSON", "LOCATION", "NRP"}:
        return ("KEEP", "Historical or civic entity in Social Studies is often curricular and should usually remain visible.")
    if subject == "english" and entity_type == "COURSE":
        return ("KEEP", "Course or assignment references in English are usually instructional rather than private.")
    if subject == "english" and entity_type in {"PERSON", "LOCATION"}:
        return ("REVIEW", "English person/location tags can be literary, curricular, or private depending on context.")
    if eval_slice == "challenge":
        return ("REVIEW", "Challenge slice targets ambiguous curricular and private entities that need action adjudication.")
    return ("REVIEW", "Natural slice sample from anonymized UPChieve tags; verify whether this tag is curricular or private.")


def _build_pool_row(item: dict[str, Any], *, split_name: str, eval_slice: str) -> dict[str, Any]:
    suggested_action, suggested_reason = _suggested_action(item, eval_slice=eval_slice)
    private_prior_score = _private_prior_score(item)
    curricular_cue_bonus = _curricular_cue_bonus(item)
    english_person_private_score = _english_person_private_score(item)
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
        "pool_source": f"upchieve_{eval_slice}",
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
            "private_prior_score": private_prior_score,
            "english_person_private_score": english_person_private_score,
            "curricular_cue_bonus": curricular_cue_bonus,
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
    max_session_rows_by_slice: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        eval_slice = str(row["eval_slice"])
        dialogue_id = str(row["dialogue_id"])
        sessions_per_slice[eval_slice].add(dialogue_id)
        max_session_rows_by_slice[eval_slice][dialogue_id] += 1
    return {
        "row_count": len(rows),
        "by_subject": dict(counts_by_subject),
        "by_eval_slice": dict(counts_by_eval_slice),
        "by_subject_entity": dict(counts_by_entity),
        "sessions_per_eval_slice": {key: len(value) for key, value in sessions_per_slice.items()},
        "max_rows_per_session_by_eval_slice": {
            key: max(counter.values()) if counter else 0
            for key, counter in max_session_rows_by_slice.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the targeted UPChieve English/Social Studies train v2 annotation pool.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INTERIM_DIR / "upchieve_context_pilot_turns.jsonl",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        action="append",
        default=[
            ACTION_DIR / "upchieve_english_social_dev.jsonl",
            ACTION_DIR / "upchieve_english_social_test.jsonl",
            ACTION_DIR / "upchieve_english_social_train.jsonl",
        ],
        help="Processed action exports whose dialogue_id values should be excluded from train_v2 sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANNOTATION_DIR / "upchieve_context_train_v2",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turn-words", type=int, default=DEFAULT_MAX_TURN_WORDS)
    parser.add_argument("--max-qualifying-tags", type=int, default=DEFAULT_MAX_QUALIFYING_TAGS)
    parser.add_argument("--subject-total", type=int, default=DEFAULT_SUBJECT_TOTAL)
    parser.add_argument("--slice-total", type=int, default=DEFAULT_SLICE_TOTAL)
    parser.add_argument("--private-prior-reserve", type=int, default=DEFAULT_PRIVATE_PRIOR_RESERVE)
    parser.add_argument(
        "--english-person-private-reserve",
        type=int,
        default=DEFAULT_ENGLISH_PERSON_PRIVATE_RESERVE,
    )
    args = parser.parse_args()

    ensure_repo_layout()
    if args.subject_total != args.slice_total * 2:
        raise SystemExit("--subject-total must equal 2 * --slice-total for the fixed natural/challenge split.")

    items, filter_summary = _load_items(
        args.input_file,
        max_turn_words=args.max_turn_words,
        max_qualifying_tags=args.max_qualifying_tags,
    )
    items, language_filter_summary = _exclude_non_english_english_items(items)
    excluded_dialogue_ids = _load_excluded_dialogue_ids(args.exclude_file)
    items = [item for item in items if item["session_id"] not in excluded_dialogue_ids]

    split_name = "upchieve_english_social_train_v2"
    pool_rows: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {}
    for subject in TRAIN_SUBJECTS:
        subject_items = [item for item in items if item["subject"] == subject]
        natural_rows, natural_summary = _select_natural(
            subject_items,
            subject=subject,
            total=args.slice_total,
            private_prior_reserve=args.private_prior_reserve,
            english_person_private_reserve=args.english_person_private_reserve,
            session_cap=1,
            seed=args.seed,
        )
        challenge_rows, challenge_summary = _select_challenge(
            subject_items,
            subject=subject,
            total=args.slice_total,
            session_cap=2,
            seed=args.seed,
            excluded_ids={row["id"] for row in natural_rows},
        )
        pool_rows.extend(
            _build_pool_row(row, split_name=split_name, eval_slice="natural")
            for row in natural_rows
        )
        pool_rows.extend(
            _build_pool_row(row, split_name=split_name, eval_slice="challenge")
            for row in challenge_rows
        )
        selection_summary[subject] = {
            "natural": natural_summary,
            "challenge": challenge_summary,
        }

    ordered_rows = sorted(
        pool_rows,
        key=lambda row: (
            str(row["subject"]),
            str(row["eval_slice"]),
            stable_sort_key(str(row["id"]), seed=args.seed),
        ),
    )
    output_file = args.output_dir / "action_pool_upchieve_english_social_train_v2.jsonl"
    _write_jsonl(output_file, ordered_rows)

    manifest: dict[str, Any] = {
        "input_file": str(args.input_file),
        "output_file": str(output_file),
        "seed": args.seed,
        "filter_summary": filter_summary,
        "language_filter_summary": language_filter_summary,
        "excluded_dialogue_ids_count": len(excluded_dialogue_ids),
        "excluded_dialogue_ids": sorted(excluded_dialogue_ids),
        "selection_summary": selection_summary,
        "pool_summary": _split_summary(ordered_rows),
    }
    _write_json(args.output_dir / "summary.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
