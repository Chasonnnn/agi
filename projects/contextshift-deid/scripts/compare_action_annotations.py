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

from contextshift_deid.constants import ANNOTATION_DIR, DEFAULT_ACTION_LABELS
from contextshift_deid.data import load_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _annotation_label(row: dict[str, Any]) -> str | None:
    annotation = row.get("annotation") or {}
    label = annotation.get("action_label")
    if label in DEFAULT_ACTION_LABELS:
        return str(label)
    return None


def _annotation_semantic_role(row: dict[str, Any]) -> str | None:
    annotation = row.get("annotation") or {}
    role = annotation.get("semantic_role")
    if role is None:
        return None
    return str(role)


def _default_disagreement_path(left_path: Path) -> Path:
    stem = left_path.stem.replace(".annotated", "")
    return ANNOTATION_DIR / f"{stem}.disagreements.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two action annotation files and emit a disagreement subset.")
    parser.add_argument("--left-file", type=Path, required=True)
    parser.add_argument("--right-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, help="Defaults to artifacts/annotation/<stem>.disagreements.jsonl")
    args = parser.parse_args()

    left_rows = load_jsonl(args.left_file)
    right_rows = load_jsonl(args.right_file)
    left_by_id = {str(row["id"]): row for row in left_rows}
    right_by_id = {str(row["id"]): row for row in right_rows}

    left_ids = set(left_by_id)
    right_ids = set(right_by_id)
    if left_ids != right_ids:
        missing_left = sorted(right_ids - left_ids)[:10]
        missing_right = sorted(left_ids - right_ids)[:10]
        raise SystemExit(
            f"Annotation id mismatch between files. missing_left={missing_left} missing_right={missing_right}"
        )

    disagreements: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    shared_labeled = 0
    shared_agree = 0

    for record_id in sorted(left_ids):
        left_row = left_by_id[record_id]
        right_row = right_by_id[record_id]
        left_label = _annotation_label(left_row)
        right_label = _annotation_label(right_row)
        left_role = _annotation_semantic_role(left_row)
        right_role = _annotation_semantic_role(right_row)
        if left_label and right_label:
            shared_labeled += 1
            pair_counts[f"{left_label}:{left_role}->{right_label}:{right_role}"] += 1
            if left_label == right_label and left_role == right_role:
                shared_agree += 1
                continue

        if left_label == right_label and left_role == right_role and left_label is not None:
            continue

        disagreement_row = dict(left_row)
        disagreement_row["comparison"] = {
            "left_file": str(args.left_file),
            "right_file": str(args.right_file),
            "left_label": left_label,
            "right_label": right_label,
            "left_semantic_role": left_role,
            "right_semantic_role": right_role,
            "left_notes": (left_row.get("annotation") or {}).get("notes"),
            "right_notes": (right_row.get("annotation") or {}).get("notes"),
        }
        disagreement_row.pop("annotation", None)
        disagreements.append(disagreement_row)

    output_file = args.output_file or _default_disagreement_path(args.left_file)
    _write_jsonl(output_file, disagreements)
    agreement = shared_agree / shared_labeled if shared_labeled else None
    print(
        json.dumps(
            {
                "left_file": str(args.left_file),
                "right_file": str(args.right_file),
                "output_file": str(output_file),
                "row_count": len(disagreements),
                "shared_labeled": shared_labeled,
                "shared_agreement": agreement,
                "label_pair_counts": pair_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
