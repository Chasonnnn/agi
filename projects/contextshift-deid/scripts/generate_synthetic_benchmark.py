from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, CANDIDATE_DIR
from contextshift_deid.data import ensure_repo_layout


CANDIDATE_SPLITS = {
    "train": [
        {
            "id": "math-train-001",
            "subject": "math",
            "tokens": ["Ava", "has", "3", "apples", "."],
            "labels": ["B-SUSPECT", "O", "O", "O", "O"],
            "anchor_text": "Solve the word problem about apples.",
        },
        {
            "id": "math-train-002",
            "subject": "math",
            "tokens": ["Email", "me", "at", "ava@example.com", "."],
            "labels": ["O", "O", "O", "B-SUSPECT", "O"],
            "anchor_text": "Discuss homework logistics.",
        },
        {
            "id": "hist-train-001",
            "subject": "history",
            "tokens": ["Napoleon", "invaded", "Russia", "."],
            "labels": ["B-SUSPECT", "O", "B-SUSPECT", "O"],
            "anchor_text": "Lesson on Napoleonic wars.",
        },
        {
            "id": "hist-train-002",
            "subject": "history",
            "tokens": ["My", "teacher", "Mr.", "Shah", "mentioned", "Napoleon", "."],
            "labels": ["O", "O", "B-SUSPECT", "I-SUSPECT", "O", "B-SUSPECT", "O"],
            "anchor_text": "Lesson on Napoleon.",
        },
        {
            "id": "lit-train-001",
            "subject": "literature",
            "tokens": ["Macbeth", "is", "ambitious", "."],
            "labels": ["B-SUSPECT", "O", "O", "O"],
            "anchor_text": "Read Macbeth Act 1.",
        },
        {
            "id": "lit-train-002",
            "subject": "literature",
            "tokens": ["My", "phone", "is", "555-111-2222", "."],
            "labels": ["O", "O", "O", "B-SUSPECT", "O"],
            "anchor_text": "Off-topic logistics chat.",
        },
    ],
    "dev": [
        {
            "id": "math-dev-001",
            "subject": "math",
            "tokens": ["Call", "me", "at", "555-444-1000", "."],
            "labels": ["O", "O", "O", "B-SUSPECT", "O"],
            "anchor_text": "Scheduling tutoring.",
        },
        {
            "id": "hist-dev-001",
            "subject": "history",
            "tokens": ["Shakespeare", "is", "not", "history", "."],
            "labels": ["B-SUSPECT", "O", "O", "O", "O"],
            "anchor_text": "Compare literature and history figures.",
        },
        {
            "id": "lit-dev-001",
            "subject": "literature",
            "tokens": ["Mr.", "Lopez", "likes", "Macbeth", "."],
            "labels": ["B-SUSPECT", "I-SUSPECT", "O", "B-SUSPECT", "O"],
            "anchor_text": "Discuss Macbeth.",
        },
    ],
    "test": [
        {
            "id": "math-test-001",
            "subject": "math",
            "tokens": ["Sophia", "solved", "the", "equation", "."],
            "labels": ["B-SUSPECT", "O", "O", "O", "O"],
            "anchor_text": "Equation solving word problem.",
        },
        {
            "id": "hist-test-001",
            "subject": "history",
            "tokens": ["Text", "me", "the", "link", "."],
            "labels": ["O", "O", "O", "O", "O"],
            "anchor_text": "Class forum reminder.",
        },
        {
            "id": "lit-test-001",
            "subject": "literature",
            "tokens": ["Macbeth", "and", "Lady", "Macbeth", "argue", "."],
            "labels": ["B-SUSPECT", "O", "B-SUSPECT", "I-SUSPECT", "O", "O"],
            "anchor_text": "Analyze Macbeth scenes.",
        },
    ],
}


ACTION_SPLITS = {
    "train": [
        {
            "id": "math-action-train-001",
            "subject": "math",
            "span_text": "Ava",
            "context_text": "Ava has 3 apples.",
            "action_label": "KEEP",
            "anchor_text": "Solve the word problem about apples.",
            "entity_type": "PERSON",
            "semantic_role": "CURRICULAR",
        },
        {
            "id": "math-action-train-002",
            "subject": "math",
            "span_text": "ava@example.com",
            "context_text": "Email me at ava@example.com.",
            "action_label": "REDACT",
            "anchor_text": "Discuss homework logistics.",
            "entity_type": "CONTACT",
            "semantic_role": "PRIVATE",
        },
        {
            "id": "hist-action-train-001",
            "subject": "history",
            "span_text": "Napoleon",
            "context_text": "Napoleon invaded Russia.",
            "action_label": "KEEP",
            "anchor_text": "Lesson on Napoleonic wars.",
            "entity_type": "PERSON",
            "semantic_role": "CURRICULAR",
        },
        {
            "id": "hist-action-train-002",
            "subject": "history",
            "span_text": "Mr. Shah",
            "context_text": "My teacher Mr. Shah mentioned Napoleon.",
            "action_label": "REDACT",
            "anchor_text": "Lesson on Napoleon.",
            "entity_type": "PERSON",
            "semantic_role": "PRIVATE",
        },
        {
            "id": "lit-action-train-001",
            "subject": "literature",
            "span_text": "Macbeth",
            "context_text": "Macbeth is ambitious.",
            "action_label": "KEEP",
            "anchor_text": "Read Macbeth Act 1.",
            "entity_type": "PERSON",
            "semantic_role": "CURRICULAR",
        },
        {
            "id": "lit-action-train-002",
            "subject": "literature",
            "span_text": "555-111-2222",
            "context_text": "My phone is 555-111-2222.",
            "action_label": "REDACT",
            "anchor_text": "Off-topic logistics chat.",
            "entity_type": "PHONE",
            "semantic_role": "PRIVATE",
        },
    ],
    "dev": [
        {
            "id": "math-action-dev-001",
            "subject": "math",
            "span_text": "555-444-1000",
            "context_text": "Call me at 555-444-1000.",
            "action_label": "REDACT",
            "anchor_text": "Scheduling tutoring.",
            "entity_type": "PHONE",
            "semantic_role": "PRIVATE",
        },
        {
            "id": "hist-action-dev-001",
            "subject": "history",
            "span_text": "Shakespeare",
            "context_text": "Shakespeare is not history.",
            "action_label": "KEEP",
            "anchor_text": "Compare literature and history figures.",
            "entity_type": "PERSON",
            "semantic_role": "CURRICULAR",
        },
        {
            "id": "lit-action-dev-001",
            "subject": "literature",
            "span_text": "Mr. Lopez",
            "context_text": "Mr. Lopez likes Macbeth.",
            "action_label": "REVIEW",
            "anchor_text": "Discuss Macbeth.",
            "entity_type": "PERSON",
            "semantic_role": "AMBIGUOUS",
        },
    ],
    "test": [
        {
            "id": "math-action-test-001",
            "subject": "math",
            "span_text": "Sophia",
            "context_text": "Sophia solved the equation.",
            "action_label": "KEEP",
            "anchor_text": "Equation solving word problem.",
            "entity_type": "PERSON",
            "semantic_role": "CURRICULAR",
        },
        {
            "id": "hist-action-test-001",
            "subject": "history",
            "span_text": "link",
            "context_text": "Text me the link.",
            "action_label": "REVIEW",
            "anchor_text": "Class forum reminder.",
            "entity_type": "OTHER",
            "semantic_role": "AMBIGUOUS",
        },
        {
            "id": "lit-action-test-001",
            "subject": "literature",
            "span_text": "Lady Macbeth",
            "context_text": "Macbeth and Lady Macbeth argue.",
            "action_label": "KEEP",
            "anchor_text": "Analyze Macbeth scenes.",
            "entity_type": "PERSON",
            "semantic_role": "CURRICULAR",
        },
    ],
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def main() -> None:
    ensure_repo_layout()
    for split, records in CANDIDATE_SPLITS.items():
        _write_jsonl(CANDIDATE_DIR / f"{split}.jsonl", records)
    for split, records in ACTION_SPLITS.items():
        _write_jsonl(ACTION_DIR / f"{split}.jsonl", records)
    print("Synthetic benchmark written to data/processed/{candidate,action}/")


if __name__ == "__main__":
    main()
