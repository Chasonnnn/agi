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

from contextshift_deid.data import load_jsonl, validate_action_records


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
    validate_action_records(path)
    return load_jsonl(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge exported action JSONL files and fail on duplicate ids.")
    parser.add_argument("--input-file", type=Path, action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path, help="Defaults to <output>_summary.json")
    args = parser.parse_args(argv)

    merged_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    counts_by_input: dict[str, int] = {}
    counts_by_input_subject: dict[str, Counter[str]] = {}

    for path in args.input_file:
        rows = _load_and_validate(path)
        counts_by_input[str(path)] = len(rows)
        counts_by_input_subject[str(path)] = Counter(str(row.get("subject", "unknown")) for row in rows)
        for row in rows:
            row_id = str(row["id"])
            if row_id in seen_ids:
                duplicate_ids.append(row_id)
                continue
            seen_ids.add(row_id)
            merged_rows.append(row)

    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise SystemExit(
            f"Duplicate ids across exported action files: found {len(duplicate_ids)} duplicates. First ids: {preview}"
        )

    _write_jsonl(args.output_file, merged_rows)
    validate_action_records(args.output_file)

    summary = {
        "input_files": [str(path) for path in args.input_file],
        "output_file": str(args.output_file),
        "row_count": len(merged_rows),
        "counts_by_input": counts_by_input,
        "counts_by_input_subject": {
            path: dict(counter) for path, counter in counts_by_input_subject.items()
        },
        "counts_by_subject": dict(Counter(str(row.get("subject", "unknown")) for row in merged_rows)),
        "counts_by_label": dict(Counter(str(row.get("action_label", "unknown")) for row in merged_rows)),
    }
    summary_path = args.summary_file or _summary_path_for(args.output_file)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
