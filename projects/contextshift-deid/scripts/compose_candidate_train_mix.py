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

from contextshift_deid.constants import CANDIDATE_DIR
from contextshift_deid.data import load_jsonl, validate_candidate_records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _summary_path_for(output_file: Path) -> Path:
    if output_file.suffix == ".jsonl":
        return output_file.with_name(f"{output_file.stem}_summary.json")
    return output_file.with_suffix(".summary.json")


def _load_and_validate(path: Path) -> list[dict[str, Any]]:
    validate_candidate_records(path)
    return load_jsonl(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compose the math candidate train split with a proxy UpChieve candidate train split.")
    parser.add_argument("--math-train-file", type=Path, default=CANDIDATE_DIR / "train.jsonl")
    parser.add_argument(
        "--upchieve-train-file",
        type=Path,
        default=CANDIDATE_DIR / "upchieve_english_social_proxy_train.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=CANDIDATE_DIR / "train_mixed_upchieve_english_social_proxy.jsonl",
    )
    parser.add_argument("--summary-file", type=Path, help="Defaults to <output>_summary.json")
    args = parser.parse_args(argv)

    math_rows = _load_and_validate(args.math_train_file)
    upchieve_rows = _load_and_validate(args.upchieve_train_file)

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    merged_rows: list[dict[str, Any]] = []
    for row in math_rows + upchieve_rows:
        row_id = str(row["id"])
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
            continue
        seen_ids.add(row_id)
        merged_rows.append(row)

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise SystemExit(
            f"Duplicate ids across candidate train sources: found {len(duplicate_ids)} duplicates. First ids: {preview}"
        )

    _write_jsonl(args.output_file, merged_rows)
    validate_candidate_records(args.output_file)

    summary = {
        "math_train_file": str(args.math_train_file),
        "upchieve_train_file": str(args.upchieve_train_file),
        "output_file": str(args.output_file),
        "row_count": len(merged_rows),
        "counts_by_source": {
            "math": len(math_rows),
            "upchieve": len(upchieve_rows),
        },
        "counts_by_subject": dict(Counter(str(row.get("subject", "unknown")) for row in merged_rows)),
        "counts_by_source_subject": {
            "math": dict(Counter(str(row.get("subject", "unknown")) for row in math_rows)),
            "upchieve": dict(Counter(str(row.get("subject", "unknown")) for row in upchieve_rows)),
        },
        "positive_rows_by_source": {
            "math": sum(any(label != "O" for label in row["labels"]) for row in math_rows),
            "upchieve": sum(any(label != "O" for label in row["labels"]) for row in upchieve_rows),
        },
    }
    summary_path = args.summary_file or _summary_path_for(args.output_file)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
