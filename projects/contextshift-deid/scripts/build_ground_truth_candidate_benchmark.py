from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import CANDIDATE_DIR
from contextshift_deid.data import validate_candidate_records
from contextshift_deid.ground_truth_candidate import (
    build_saga_segment_candidate_rows,
    build_upchieve_turn_candidate_rows,
    split_upchieve_dialogues,
    summarize_candidate_rows,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dialogue_ids(path: Path) -> set[str]:
    return {str(record.dialogue_id) for record in validate_candidate_records(path)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build UpChieve and Saga ground-truth candidate benchmark splits.")
    parser.add_argument(
        "--upchieve-raw-file",
        type=Path,
        default=Path("/Users/chason/Downloads/DeID_GT_UPchieve_math_1000transcripts.jsonl"),
    )
    parser.add_argument(
        "--saga-raw-dir",
        type=Path,
        default=Path("/Users/chason/Downloads/DeID_GT_Saga_math_27_transcripts"),
    )
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--train-dialogues", type=int, default=700)
    parser.add_argument("--dev-dialogues", type=int, default=150)
    parser.add_argument("--test-dialogues", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    upchieve_rows, upchieve_summary = build_upchieve_turn_candidate_rows(args.upchieve_raw_file)
    upchieve_splits, upchieve_split_summary = split_upchieve_dialogues(
        upchieve_rows,
        train_dialogues=args.train_dialogues,
        dev_dialogues=args.dev_dialogues,
        test_dialogues=args.test_dialogues,
        seed=args.seed,
    )
    saga_rows, saga_summary = build_saga_segment_candidate_rows(args.saga_raw_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    upchieve_paths = {
        "train": output_dir / "upchieve_math_ground_truth_train.jsonl",
        "dev": output_dir / "upchieve_math_ground_truth_dev.jsonl",
        "test": output_dir / "upchieve_math_ground_truth_test.jsonl",
    }
    for split_name, split_rows in upchieve_splits.items():
        _write_jsonl(upchieve_paths[split_name], split_rows)
        validate_candidate_records(upchieve_paths[split_name])

    saga_path = output_dir / "saga27_math_ground_truth_test.jsonl"
    _write_jsonl(saga_path, saga_rows)
    validate_candidate_records(saga_path)

    train_dialogue_ids = _dialogue_ids(upchieve_paths["train"])
    dev_dialogue_ids = _dialogue_ids(upchieve_paths["dev"])
    test_dialogue_ids = _dialogue_ids(upchieve_paths["test"])
    if train_dialogue_ids & dev_dialogue_ids or train_dialogue_ids & test_dialogue_ids or dev_dialogue_ids & test_dialogue_ids:
        raise SystemExit("Detected dialogue leakage across UpChieve train/dev/test splits.")

    if saga_summary["file_count"] != 27:
        raise SystemExit(f"Expected 27 Saga files, found {saga_summary['file_count']}.")

    summary = {
        "upchieve_raw_file": str(args.upchieve_raw_file),
        "saga_raw_dir": str(args.saga_raw_dir),
        "seed": args.seed,
        "upchieve": {
            "dataset": upchieve_summary,
            "splits": upchieve_split_summary,
            "split_paths": {split: str(path) for split, path in upchieve_paths.items()},
            "split_row_summaries": {
                split: summarize_candidate_rows(rows)
                for split, rows in upchieve_splits.items()
            },
        },
        "saga27": {
            "dataset": saga_summary,
            "path": str(saga_path),
            "row_summary": summarize_candidate_rows(saga_rows),
        },
        "validation": {
            "upchieve_dialogue_leakage": False,
            "saga_file_count": saga_summary["file_count"],
        },
    }

    summary_path = output_dir / "ground_truth_candidate_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
