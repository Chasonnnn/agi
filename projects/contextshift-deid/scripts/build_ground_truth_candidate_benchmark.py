from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import (
    CANDIDATE_DIR,
    DEFAULT_SAGA27_MATH_GROUND_TRUTH_DIR,
    DEFAULT_UPCHIEVE_MATH_GROUND_TRUTH_DIR,
)
from contextshift_deid.data import validate_candidate_records
from contextshift_deid.ground_truth_candidate import (
    build_saga_segment_candidate_rows,
    build_upchieve_turn_candidate_rows,
    canonicalize_pii_type,
    canonicalize_pii_type_counts,
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_counts(mapping: Mapping[str, Any] | None) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in (mapping or {}).items()
    }


def _dataset_metrics(summary: Mapping[str, Any], source: str) -> dict[str, Any]:
    return dict((summary.get(source) or {}).get("dataset") or {})


def _positive_counts(summary: Mapping[str, Any], source: str) -> dict[str, int]:
    dataset = _dataset_metrics(summary, source)
    return {
        "row_count": int(dataset.get("row_count") or 0),
        "positive_row_count": int(dataset.get("positive_row_count") or 0),
        "positive_dialogue_count": int(dataset.get("positive_dialogue_count") or 0),
    }


def _span_delta_breakdown(old_counts: Mapping[str, int], new_counts: Mapping[str, int]) -> dict[str, int]:
    labels = sorted(set(old_counts) | set(new_counts))
    overlap = sum(min(old_counts.get(label, 0), new_counts.get(label, 0)) for label in labels)
    removed = sum(max(old_counts.get(label, 0) - new_counts.get(label, 0), 0) for label in labels)
    added = sum(max(new_counts.get(label, 0) - old_counts.get(label, 0), 0) for label in labels)
    return {
        "overlap_count": overlap,
        "removed_count": removed,
        "added_count": added,
    }


def _compare_source(previous_summary: Mapping[str, Any], current_summary: Mapping[str, Any], source: str) -> dict[str, Any]:
    previous_dataset = _dataset_metrics(previous_summary, source)
    current_dataset = _dataset_metrics(current_summary, source)
    previous_raw_counts = _coerce_counts(previous_dataset.get("pii_type_counts"))
    previous_normalized_counts = canonicalize_pii_type_counts(previous_raw_counts)
    current_canonical_counts = canonicalize_pii_type_counts(_coerce_counts(current_dataset.get("pii_type_counts")))
    label_union = sorted(set(previous_normalized_counts) | set(current_canonical_counts))
    renamed_from_old = {
        old_label: {
            "canonical_label": canonicalize_pii_type(old_label),
            "count": count,
        }
        for old_label, count in sorted(previous_raw_counts.items())
        if canonicalize_pii_type(old_label) != old_label
    }
    label_deltas = {
        label: current_canonical_counts.get(label, 0) - previous_normalized_counts.get(label, 0)
        for label in label_union
    }
    return {
        "positive_counts": {
            "previous": _positive_counts(previous_summary, source),
            "current": _positive_counts(current_summary, source),
        },
        "span_counts": {
            "previous_total": sum(previous_normalized_counts.values()),
            "current_total": sum(current_canonical_counts.values()),
        },
        "canonical_label_counts": {
            "previous": previous_normalized_counts,
            "current": current_canonical_counts,
            "delta": label_deltas,
        },
        "span_delta_after_normalization": _span_delta_breakdown(previous_normalized_counts, current_canonical_counts),
        "renamed_from_previous_labels": renamed_from_old,
        "removed_labels_after_normalization": {
            label: previous_normalized_counts[label]
            for label in label_union
            if previous_normalized_counts.get(label, 0) and not current_canonical_counts.get(label, 0)
        },
        "new_labels_after_normalization": {
            label: current_canonical_counts[label]
            for label in label_union
            if current_canonical_counts.get(label, 0) and not previous_normalized_counts.get(label, 0)
        },
    }


def _build_comparison(previous_summary: Mapping[str, Any], current_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "notes": "Count-based comparison after normalizing the previous benchmark label inventory into the current math taxonomy.",
        "sources": {
            "upchieve": _compare_source(previous_summary, current_summary, "upchieve"),
            "saga27": _compare_source(previous_summary, current_summary, "saga27"),
        },
    }


def _render_comparison_report(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Ground-Truth Benchmark Comparison",
        "",
        str(comparison.get("notes") or ""),
        "",
    ]
    for source in ("upchieve", "saga27"):
        payload = dict((comparison.get("sources") or {}).get(source) or {})
        lines.extend(
            [
                f"## {source}",
                "",
                "### Positive Counts",
                "",
                f"- previous: `{json.dumps(payload.get('positive_counts', {}).get('previous', {}), sort_keys=True)}`",
                f"- current: `{json.dumps(payload.get('positive_counts', {}).get('current', {}), sort_keys=True)}`",
                "",
                "### Span Counts",
                "",
                f"- previous_total: `{payload.get('span_counts', {}).get('previous_total')}`",
                f"- current_total: `{payload.get('span_counts', {}).get('current_total')}`",
                "",
                "### Span Delta After Normalization",
                "",
                f"- overlap_count: `{payload.get('span_delta_after_normalization', {}).get('overlap_count')}`",
                f"- removed_count: `{payload.get('span_delta_after_normalization', {}).get('removed_count')}`",
                f"- added_count: `{payload.get('span_delta_after_normalization', {}).get('added_count')}`",
                "",
                "### Canonical Label Counts",
                "",
                "```json",
                json.dumps(payload.get("canonical_label_counts") or {}, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build UpChieve and Saga ground-truth candidate benchmark splits.")
    parser.add_argument(
        "--upchieve-raw-dir",
        "--upchieve-raw-file",
        dest="upchieve_raw_dir",
        type=Path,
        default=DEFAULT_UPCHIEVE_MATH_GROUND_TRUTH_DIR,
    )
    parser.add_argument(
        "--saga-raw-dir",
        type=Path,
        default=DEFAULT_SAGA27_MATH_GROUND_TRUTH_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--train-dialogues", type=int, default=700)
    parser.add_argument("--dev-dialogues", type=int, default=150)
    parser.add_argument("--test-dialogues", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ground_truth_candidate_benchmark_summary.json"
    previous_summary = _read_json(summary_path) if summary_path.exists() else None

    upchieve_rows, upchieve_summary = build_upchieve_turn_candidate_rows(args.upchieve_raw_dir)
    upchieve_splits, upchieve_split_summary = split_upchieve_dialogues(
        upchieve_rows,
        train_dialogues=args.train_dialogues,
        dev_dialogues=args.dev_dialogues,
        test_dialogues=args.test_dialogues,
        seed=args.seed,
    )
    saga_rows, saga_summary = build_saga_segment_candidate_rows(args.saga_raw_dir)

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
        "upchieve_raw_dir": str(args.upchieve_raw_dir),
        "upchieve_raw_file": str(args.upchieve_raw_dir),
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

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if previous_summary is not None:
        comparison = _build_comparison(previous_summary, summary)
        comparison_json_path = output_dir / "ground_truth_candidate_benchmark_comparison.json"
        comparison_md_path = output_dir / "ground_truth_candidate_benchmark_comparison.md"
        comparison_json_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        comparison_md_path.write_text(_render_comparison_report(comparison), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
