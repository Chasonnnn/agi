"""
Initialize and validate the contextshift-deid repository layout.

This script creates the expected local directories, initializes the results log,
and validates any benchmark split files that already exist under data/processed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, CANDIDATE_DIR, EXPECTED_SPLITS, RESULTS_HEADER
from contextshift_deid.data import ensure_repo_layout, validate_action_records, validate_candidate_records


def _validate_optional_file(path: Path, validator) -> int:
    if not path.exists():
        print(f"- missing (ok for now): {path}")
        return 0
    print(f"- validating: {path}")
    records = validator(path)
    print(f"  records: {len(records)}")
    return len(records)


def _summarize_action_split(path: Path) -> dict[str, object]:
    records = validate_action_records(path)
    action_labels = sorted({record.action_label for record in records})
    provisional_rows = [
        record for record in records
        if record.metadata.get("provisional_action") is True
        or record.metadata.get("label_source") == "legacy_redact_only"
    ]
    label_sources = sorted(
        {
            str(record.metadata.get("label_source"))
            for record in records
            if record.metadata.get("label_source") is not None
        }
    )
    return {
        "records": len(records),
        "action_labels": action_labels,
        "label_sources": label_sources,
        "provisional_rows": len(provisional_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize and validate the repo.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any expected benchmark split is missing.",
    )
    parser.add_argument(
        "--allow-provisional-action-splits",
        nargs="*",
        choices=EXPECTED_SPLITS,
        default=(),
        help=(
            "When running a dev/test relabeling sprint, allow provisional action labels to remain on these splits "
            "without failing strict mode."
        ),
    )
    parser.add_argument(
        "--extra-action-file",
        type=Path,
        action="append",
        default=[],
        help="Optional extra action JSONL file to validate alongside train/dev/test.",
    )
    args = parser.parse_args()
    allowed_provisional_splits = set(args.allow_provisional_action_splits)

    layout = ensure_repo_layout()
    print("Repository layout ready:")
    for name, path in layout.items():
        print(f"  {name:>14s}: {path}")

    results_path = Path("results.tsv")
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")
        print(f"Created {results_path}")
    else:
        print(f"Found {results_path}")

    missing = []
    strict_failures: list[str] = []
    summary = {"candidate": {}, "action": {}, "extra_action": {}}

    print()
    print("Candidate split validation:")
    for split in EXPECTED_SPLITS:
        path = CANDIDATE_DIR / f"{split}.jsonl"
        if not path.exists():
            missing.append(path)
        summary["candidate"][split] = _validate_optional_file(path, validate_candidate_records)

    print()
    print("Action split validation:")
    for split in EXPECTED_SPLITS:
        path = ACTION_DIR / f"{split}.jsonl"
        if not path.exists():
            missing.append(path)
            summary["action"][split] = 0
            continue
        print(f"- validating: {path}")
        action_summary = _summarize_action_split(path)
        print(f"  records: {action_summary['records']}")
        print(f"  action_labels: {action_summary['action_labels']}")
        if action_summary["label_sources"]:
            print(f"  label_sources: {action_summary['label_sources']}")
        print(f"  provisional_rows: {action_summary['provisional_rows']}")
        summary["action"][split] = action_summary["records"]
        if action_summary["provisional_rows"]:
            if split in allowed_provisional_splits:
                print(f"  note: {split} still contains provisional action labels, but this strict run allows that split")
            else:
                print(f"  warning: {split} contains provisional action labels")
                strict_failures.append(
                    f"{path}: contains {action_summary['provisional_rows']} provisional action labels"
                )

    print()
    print("Validation summary:")
    print(json.dumps(summary, indent=2))

    if args.extra_action_file:
        print()
        print("Extra action file validation:")
        for path in args.extra_action_file:
            if not path.exists():
                raise SystemExit(f"Missing extra action file: {path}")
            print(f"- validating: {path}")
            action_summary = _summarize_action_split(path)
            print(f"  records: {action_summary['records']}")
            print(f"  action_labels: {action_summary['action_labels']}")
            if action_summary["label_sources"]:
                print(f"  label_sources: {action_summary['label_sources']}")
            print(f"  provisional_rows: {action_summary['provisional_rows']}")
            summary["extra_action"][str(path)] = action_summary["records"]
            if action_summary["provisional_rows"]:
                strict_failures.append(
                    f"{path}: contains {action_summary['provisional_rows']} provisional action labels"
                )

        print()
        print("Updated validation summary:")
        print(json.dumps(summary, indent=2))

    if args.strict and missing:
        raise SystemExit("Missing expected split files:\n" + "\n".join(str(path) for path in missing))
    if args.strict and strict_failures:
        raise SystemExit(
            "Strict validation failed due to provisional action supervision:\n"
            + "\n".join(strict_failures)
        )


if __name__ == "__main__":
    main()
