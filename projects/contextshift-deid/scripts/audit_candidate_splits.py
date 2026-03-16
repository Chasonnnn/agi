from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import CANDIDATE_DIR, EXPECTED_SPLITS
from contextshift_deid.data import validate_candidate_records


def _row_has_positive(labels: list[str]) -> bool:
    return any(label != "O" for label in labels)


def _split_summary(path: Path) -> dict[str, object]:
    records = validate_candidate_records(path)
    row_count = len(records)
    token_count = sum(len(record.labels) for record in records)
    positive_rows = sum(_row_has_positive(record.labels) for record in records)
    positive_tokens = sum(label != "O" for record in records for label in record.labels)
    speaker_counts = Counter(record.speaker_role or "unknown" for record in records)
    subject_counts = Counter(record.subject for record in records)
    token_lengths = [len(record.labels) for record in records]
    return {
        "rows": row_count,
        "tokens": token_count,
        "positive_rows": positive_rows,
        "positive_row_rate": round(positive_rows / row_count, 6) if row_count else 0.0,
        "positive_tokens": positive_tokens,
        "positive_token_rate": round(positive_tokens / token_count, 6) if token_count else 0.0,
        "avg_tokens_per_row": round(sum(token_lengths) / row_count, 3) if row_count else 0.0,
        "max_tokens_per_row": max(token_lengths) if token_lengths else 0,
        "subject_distribution": dict(subject_counts),
        "speaker_role_distribution": dict(speaker_counts),
    }


def main() -> None:
    payload = {
        split: _split_summary(CANDIDATE_DIR / f"{split}.jsonl")
        for split in EXPECTED_SPLITS
        if (CANDIDATE_DIR / f"{split}.jsonl").exists()
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
