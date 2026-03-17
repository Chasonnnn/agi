from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.candidate_adaptation import (
    annotate_baseline_misses,
    build_candidate_proxy_rows_from_action,
    sample_balanced_proxy_splits,
    summarize_proxy_rows,
)
from contextshift_deid.constants import ACTION_DIR, ANNOTATION_DIR, CANDIDATE_DIR, RUNS_DIR
from contextshift_deid.data import validate_candidate_records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _run_candidate_predictions(
    *,
    model_path: Path,
    input_file: Path,
    output_file: Path,
    batch_size: int,
    max_length: int,
    context_mode: str,
) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "predict_candidate.py"),
        "--model",
        str(model_path),
        "--input-file",
        str(input_file),
        "--output-file",
        str(output_file),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--context-mode",
        context_mode,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    with output_file.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build proxy UpChieve candidate splits from the action-first English/Social dataset."
    )
    parser.add_argument(
        "--action-file",
        type=Path,
        default=ACTION_DIR / "upchieve_english_social_train_v1_v2.jsonl",
    )
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=RUNS_DIR / "candidate",
        help="Frozen math candidate checkpoint used to prioritize missed turns.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--dev-count", type=int, default=80)
    parser.add_argument("--test-count", type=int, default=80)
    parser.add_argument(
        "--prefix",
        default="upchieve_english_social_proxy",
        help="Filename prefix for processed candidate splits.",
    )
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=ANNOTATION_DIR / "upchieve_candidate_adaptation",
    )
    parser.add_argument("--baseline-batch-size", type=int, default=16)
    parser.add_argument("--baseline-max-length", type=int, default=256)
    parser.add_argument(
        "--baseline-context-mode",
        choices=("none", "pair"),
        default="none",
    )
    args = parser.parse_args(argv)

    proxy_rows = build_candidate_proxy_rows_from_action(args.action_file)
    if not proxy_rows:
        raise SystemExit(f"No candidate proxy rows were built from {args.action_file}")

    args.annotation_dir.mkdir(parents=True, exist_ok=True)
    all_turns_file = args.annotation_dir / f"{args.prefix}_all_turns.jsonl"
    _write_jsonl(all_turns_file, proxy_rows)

    baseline_predictions_file = None
    if args.baseline_model.exists():
        baseline_predictions_file = args.annotation_dir / f"{args.prefix}_baseline_predictions.jsonl"
        baseline_predictions = _run_candidate_predictions(
            model_path=args.baseline_model,
            input_file=all_turns_file,
            output_file=baseline_predictions_file,
            batch_size=args.baseline_batch_size,
            max_length=args.baseline_max_length,
            context_mode=args.baseline_context_mode,
        )
        predictions_by_id = {str(row["id"]): row for row in baseline_predictions}
        proxy_rows = annotate_baseline_misses(proxy_rows, predictions_by_id)
        _write_jsonl(all_turns_file, proxy_rows)

    sampled_splits = sample_balanced_proxy_splits(
        proxy_rows,
        counts_by_split={
            "train": args.train_count,
            "dev": args.dev_count,
            "test": args.test_count,
        },
        seed=args.seed,
    )

    split_files: dict[str, str] = {}
    annotation_pool_files: dict[str, str] = {}
    split_summaries: dict[str, Any] = {}
    for split, rows in sampled_splits.items():
        processed_path = args.output_dir / f"{args.prefix}_{split}.jsonl"
        annotation_path = args.annotation_dir / f"candidate_pool_{args.prefix}_{split}.jsonl"
        _write_jsonl(processed_path, rows)
        _write_jsonl(annotation_path, rows)
        validate_candidate_records(processed_path)
        split_files[split] = str(processed_path)
        annotation_pool_files[split] = str(annotation_path)
        split_summaries[split] = summarize_proxy_rows(rows)

    summary = {
        "action_file": str(args.action_file),
        "baseline_model": str(args.baseline_model) if args.baseline_model.exists() else None,
        "baseline_predictions_file": str(baseline_predictions_file) if baseline_predictions_file else None,
        "seed": args.seed,
        "prefix": args.prefix,
        "all_turn_count": len(proxy_rows),
        "all_turn_summary": summarize_proxy_rows(proxy_rows),
        "split_files": split_files,
        "annotation_pool_files": annotation_pool_files,
        "split_summaries": split_summaries,
    }
    summary_path = args.annotation_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
