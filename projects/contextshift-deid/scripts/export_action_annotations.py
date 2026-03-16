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

from contextshift_deid.constants import ACTION_DIR, ANNOTATION_DIR, DEFAULT_ACTION_LABELS
from contextshift_deid.data import load_jsonl, validate_action_records

DEFAULT_SEMANTIC_ROLE_BY_ACTION = {
    "REDACT": "PRIVATE",
    "KEEP": "CURRICULAR",
    "REVIEW": "AMBIGUOUS",
}


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem.replace(".annotated", "")
    split_name = stem.removeprefix("action_pool_")
    return ACTION_DIR / f"{split_name}.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotated pool rows into repo-native action JSONL.")
    parser.add_argument("--input-file", type=Path, default=ANNOTATION_DIR / "action_pool_dev.annotated.jsonl")
    parser.add_argument("--output-file", type=Path, help="Defaults to data/processed/action/<split>.jsonl")
    parser.add_argument(
        "--label-source",
        default="codebook_v2_manual",
        help="Manual label provenance written into row metadata.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_file}")

    unlabeled_ids: list[str] = []
    output_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()

    for row in rows:
        annotation = row.get("annotation") or {}
        label = annotation.get("action_label")
        if label not in DEFAULT_ACTION_LABELS:
            unlabeled_ids.append(str(row["id"]))
            continue
        label = str(label)
        semantic_role = str(
            annotation.get("semantic_role")
            or row.get("semantic_role")
            or DEFAULT_SEMANTIC_ROLE_BY_ACTION[label]
        )
        metadata = dict(row.get("metadata") or {})
        metadata.update(
            {
                "label_source": args.label_source,
                "provisional_action": False,
                "annotation_pool_source": row.get("pool_source"),
                "annotation_suggested_action": row.get("suggested_action"),
                "annotator": annotation.get("annotator"),
                "annotation_notes": annotation.get("notes"),
                "annotation_timestamp": annotation.get("timestamp"),
                "annotation_semantic_role": semantic_role,
                "split": row.get("split"),
            }
        )
        if row.get("token_start") is not None:
            metadata["token_start"] = int(row["token_start"])
        if row.get("token_end") is not None:
            metadata["token_end"] = int(row["token_end"])

        output_row: dict[str, Any] = {
            "id": str(row["id"]),
            "subject": str(row["subject"]),
            "span_text": str(row["span_text"]),
            "context_text": str(row["context_text"]),
            "action_label": label,
            "semantic_role": semantic_role,
            "metadata": metadata,
        }
        if row.get("eval_slice") is not None:
            output_row["eval_slice"] = str(row["eval_slice"])
        for field in ("anchor_text", "speaker_role", "entity_type", "intent_label", "dialogue_id"):
            value = row.get(field)
            if value is not None:
                output_row[field] = value
        output_rows.append(output_row)
        label_counts[label] += 1

    if unlabeled_ids:
        preview = ", ".join(unlabeled_ids[:10])
        raise SystemExit(
            f"{args.input_file}: found {len(unlabeled_ids)} unlabeled rows. "
            f"Annotate or remove them before export. First ids: {preview}"
        )

    output_file = args.output_file or _default_output_path(args.input_file)
    _write_jsonl(output_file, output_rows)
    validate_action_records(output_file)
    print(
        json.dumps(
            {
                "input_file": str(args.input_file),
                "output_file": str(output_file),
                "row_count": len(output_rows),
                "label_distribution": label_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
