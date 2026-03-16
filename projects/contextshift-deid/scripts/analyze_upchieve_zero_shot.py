from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, PREDICTIONS_DIR
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import create_experiment_run, write_run_metadata

DEFAULT_TARGET_REVIEW_RATE = 0.05
FIELD_TO_METRIC_KEY = {
    "subject": "by_context",
    "entity_type": "by_entity_type",
    "semantic_role": "by_semantic_role",
    "eval_slice": "by_eval_slice",
}
DEFAULT_MODEL_SPECS = {
    "distilroberta": {
        "label": "distilroberta-base",
        "summary_file": ROOT
        / "artifacts/experiments/20260313_154629_upchieve-english-social-zero-shot-distilroberta/summary.json",
        "base_prediction_file": PREDICTIONS_DIR / "upchieve_english_social_test_predictions.jsonl",
    },
    "roberta_base": {
        "label": "roberta-base",
        "summary_file": ROOT
        / "artifacts/experiments/20260313_154907_upchieve-english-social-zero-shot-roberta-base/summary.json",
        "base_prediction_file": PREDICTIONS_DIR / "upchieve_english_social_test_predictions_roberta_base.jsonl",
    },
    "modernbert": {
        "label": "ModernBERT-base",
        "summary_file": ROOT
        / "artifacts/experiments/20260313_154943_upchieve-english-social-zero-shot-modernbert/summary.json",
        "base_prediction_file": PREDICTIONS_DIR / "upchieve_english_social_test_predictions_modernbert.jsonl",
    },
}


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _normalize_target_key(value: float) -> str:
    return f"{value:.2f}"


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing prediction file: {path}")
    return load_jsonl(path)


def _selected_target_entry(summary: dict[str, Any], *, target_review_rate: float) -> dict[str, Any]:
    for target in summary.get("selected_targets", []):
        rate = float(target["target_review_rate"])
        if abs(rate - target_review_rate) <= 1e-9:
            return target
    raise SystemExit(
        f"{summary.get('model_name', 'unknown summary')} is missing a selected target "
        f"for review rate {target_review_rate:.2f}"
    )


def _merge_gold_predictions(gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    merged: list[dict[str, Any]] = []
    for row in gold_rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row_id}")
        merged.append(
            {
                "id": row_id,
                "subject": str(row.get("subject", "unknown")),
                "entity_type": row.get("entity_type"),
                "semantic_role": row.get("semantic_role"),
                "eval_slice": row.get("eval_slice"),
                "span_text": row.get("span_text"),
                "context_text": row.get("context_text"),
                "gold_action": row["action_label"],
                "predicted_action": prediction["predicted_action"],
            }
        )
    return merged


def _base_error_bucket_examples(
    gold_rows: list[dict[str, Any]],
    *,
    merged_by_model: dict[str, dict[str, dict[str, Any]]],
    model_order: list[str],
    limit: int = 12,
) -> dict[str, dict[str, Any]]:
    all_three_wrong: list[dict[str, Any]] = []
    modernbert_only_correct: list[dict[str, Any]] = []
    all_three_redact_miss: list[dict[str, Any]] = []

    for row in gold_rows:
        row_id = str(row["id"])
        gold_action = str(row["action_label"])
        predictions = {
            model_key: merged_by_model[model_key][row_id]["predicted_action"]
            for model_key in model_order
        }
        example = {
            "id": row_id,
            "subject": str(row.get("subject", "unknown")),
            "entity_type": row.get("entity_type"),
            "semantic_role": row.get("semantic_role"),
            "eval_slice": row.get("eval_slice"),
            "span_text": row.get("span_text"),
            "gold_action": gold_action,
            "predictions": predictions,
            "context_preview": _context_preview(str(row.get("context_text", "")), span_text=str(row.get("span_text", ""))),
        }
        if all(prediction != gold_action for prediction in predictions.values()):
            all_three_wrong.append(example)
        if (
            predictions["modernbert"] == gold_action
            and predictions["distilroberta"] != gold_action
            and predictions["roberta_base"] != gold_action
        ):
            modernbert_only_correct.append(example)
        if gold_action != "REDACT" and all(prediction == "REDACT" for prediction in predictions.values()):
            all_three_redact_miss.append(example)

    ordered_buckets = {
        "all_three_wrong": all_three_wrong,
        "modernbert_only_correct": modernbert_only_correct,
        "all_three_redact_against_gold_keep_or_review": all_three_redact_miss,
    }
    return {
        bucket_name: {
            "count": len(rows),
            "examples": rows[:limit],
        }
        for bucket_name, rows in ordered_buckets.items()
    }


def _context_preview(context_text: str, *, span_text: str, radius: int = 72) -> str:
    compact = " ".join(context_text.split())
    if not compact:
        return ""
    needle = span_text.strip()
    lowered = compact.lower()
    lowered_needle = needle.lower()
    if needle and lowered_needle in lowered:
        start = lowered.index(lowered_needle)
        excerpt_start = max(0, start - radius)
        excerpt_end = min(len(compact), start + len(needle) + radius)
        excerpt = compact[excerpt_start:excerpt_end]
        if excerpt_start > 0:
            excerpt = "..." + excerpt
        if excerpt_end < len(compact):
            excerpt = excerpt + "..."
        return excerpt
    if len(compact) <= radius * 2:
        return compact
    return compact[: radius * 2 - 3].rstrip() + "..."


def _metric_row(model_label: str, slice_name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {model_label} | {slice_name} | {metrics['count']} | "
        f"{_format_metric(float(metrics['accuracy']))} | "
        f"{_format_metric(float(metrics['macro_f1']))} | "
        f"{_format_metric(float(metrics['redact_recall']))} |"
    )


def _slice_table_lines(
    *,
    model_payloads: list[dict[str, Any]],
    field: str,
    metric_key: str,
    section_title: str,
) -> list[str]:
    lines = [f"### {section_title}", "", "| model | slice | count | accuracy | macro_f1 | redact_recall |", "| --- | --- | --- | --- | --- | --- |"]
    known_slices: set[str] = set()
    for payload in model_payloads:
        known_slices.update(payload["metrics"].get(metric_key, {}).keys())
    for slice_name in sorted(known_slices):
        for payload in model_payloads:
            metrics = payload["metrics"].get(metric_key, {}).get(slice_name)
            if metrics is None:
                continue
            lines.append(_metric_row(payload["label"], slice_name, metrics))
    if len(lines) == 4:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return lines


def _top_level_table_lines(
    *,
    rows: list[dict[str, Any]],
    section_title: str,
) -> list[str]:
    lines = [
        f"## {section_title}",
        "",
        "| model | accuracy | macro_f1 | redact_recall | review_rate | orr |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {accuracy} | {macro_f1} | {redact_recall} | {review_rate} | {orr} |".format(
                label=row["label"],
                accuracy=_format_metric(row["metrics"].get("accuracy")),
                macro_f1=_format_metric(row["metrics"].get("macro_f1")),
                redact_recall=_format_metric(row["metrics"].get("redact_recall")),
                review_rate=_format_percent(row["metrics"].get("review_rate")),
                orr=_format_percent(row["metrics"].get("orr")),
            )
        )
    lines.append("")
    return lines


def _target_table_lines(model_rows: list[dict[str, Any]], *, target_review_rate: float) -> list[str]:
    lines = [
        f"## Selected Policy Metrics ({target_review_rate * 100:.1f}% Target)",
        "",
        "| model | strategy | accuracy | macro_f1 | redact_recall | review_rate | gold_review_coverage | curricular_accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in model_rows:
        metrics = row["metrics"]
        curricular_metrics = metrics.get("by_semantic_role", {}).get("CURRICULAR", {})
        lines.append(
            "| {label} | `{strategy}` | {accuracy} | {macro_f1} | {redact_recall} | {review_rate} | {coverage} | {curricular_accuracy} |".format(
                label=row["label"],
                strategy=row["strategy"],
                accuracy=_format_metric(metrics.get("accuracy")),
                macro_f1=_format_metric(metrics.get("macro_f1")),
                redact_recall=_format_metric(metrics.get("redact_recall")),
                review_rate=_format_percent(metrics.get("review_rate")),
                coverage=_format_percent(metrics.get("gold_review_coverage")),
                curricular_accuracy=_format_metric(curricular_metrics.get("accuracy")),
            )
        )
    lines.append("")
    return lines


def _bucket_lines(bucket_name: str, bucket: dict[str, Any]) -> list[str]:
    lines = [
        f"### {bucket_name.replace('_', ' ').title()}",
        "",
        f"- Count: {bucket['count']}",
        "",
        "| id | subject | entity | role | gold | distil | roberta | modernbert | span | context |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for example in bucket["examples"]:
        predictions = example["predictions"]
        lines.append(
            "| {id} | {subject} | {entity_type} | {semantic_role} | {gold_action} | {distil} | {roberta} | {modernbert} | {span_text} | {context_preview} |".format(
                id=example["id"],
                subject=example["subject"],
                entity_type=example.get("entity_type") or "n/a",
                semantic_role=example.get("semantic_role") or "n/a",
                gold_action=example["gold_action"],
                distil=predictions["distilroberta"],
                roberta=predictions["roberta_base"],
                modernbert=predictions["modernbert"],
                span_text=example.get("span_text") or "",
                context_preview=example["context_preview"],
            )
        )
    if not bucket["examples"]:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return lines


def _build_report(
    *,
    gold_file: Path,
    base_rows: list[dict[str, Any]],
    selected_005_rows: list[dict[str, Any]],
    selected_010_rows: list[dict[str, Any]],
    overlap_buckets: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# UPChieve Zero-Shot Analysis",
        "",
        f"- Gold file: `{gold_file}`",
        f"- Benchmark rows: {sum(1 for _ in load_jsonl(gold_file))}",
        "",
    ]
    lines.extend(_top_level_table_lines(rows=base_rows, section_title="Base Eval Metrics"))
    lines.extend(_target_table_lines(selected_005_rows, target_review_rate=0.05))
    lines.extend(_target_table_lines(selected_010_rows, target_review_rate=0.10))

    lines.append("## Base Eval Slice Tables")
    lines.append("")
    for field, metric_key in FIELD_TO_METRIC_KEY.items():
        lines.extend(
            _slice_table_lines(
                model_payloads=base_rows,
                field=field,
                metric_key=metric_key,
                section_title=field.replace("_", " ").title(),
            )
        )

    lines.append("## Selected Policy Slice Tables (5.0% Target)")
    lines.append("")
    for field, metric_key in FIELD_TO_METRIC_KEY.items():
        lines.extend(
            _slice_table_lines(
                model_payloads=selected_005_rows,
                field=field,
                metric_key=metric_key,
                section_title=field.replace("_", " ").title(),
            )
        )

    lines.append("## Base Error Overlap")
    lines.append("")
    for bucket_name in (
        "all_three_wrong",
        "modernbert_only_correct",
        "all_three_redact_against_gold_keep_or_review",
    ):
        lines.extend(_bucket_lines(bucket_name, overlap_buckets[bucket_name]))

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze the three UPChieve English/Social zero-shot runs.")
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=ACTION_DIR / "upchieve_english_social_test.jsonl",
    )
    parser.add_argument(
        "--run-name",
        default="upchieve-english-social-zero-shot-analysis",
    )
    parser.add_argument(
        "--selected-review-rate",
        type=float,
        default=DEFAULT_TARGET_REVIEW_RATE,
        help="Which selected-policy target review rate to use for detailed slice tables.",
    )
    args = parser.parse_args(argv)

    gold_rows = load_jsonl(args.gold_file)
    model_payloads: dict[str, dict[str, Any]] = {}
    merged_by_model: dict[str, dict[str, dict[str, Any]]] = {}

    for model_key, spec in DEFAULT_MODEL_SPECS.items():
        summary = _load_summary(Path(spec["summary_file"]))
        base_prediction_rows = _load_prediction_rows(Path(spec["base_prediction_file"]))
        base_merged_rows = _merge_gold_predictions(gold_rows, base_prediction_rows)
        merged_by_model[model_key] = {row["id"]: row for row in base_merged_rows}

        selected_targets = {
            _normalize_target_key(float(target["target_review_rate"])): target
            for target in summary.get("selected_targets", [])
        }
        target_005 = _selected_target_entry(summary, target_review_rate=0.05)
        target_010 = _selected_target_entry(summary, target_review_rate=0.10)
        selected_default = _selected_target_entry(summary, target_review_rate=args.selected_review_rate)

        model_payloads[model_key] = {
            "key": model_key,
            "label": str(spec["label"]),
            "summary_path": str(spec["summary_file"]),
            "base_prediction_file": str(spec["base_prediction_file"]),
            "metrics": summary["eval_base_metrics"],
            "selected_005": target_005,
            "selected_010": target_010,
            "selected_default": selected_default,
            "selected_targets": selected_targets,
        }

    base_rows = sorted(
        (
            {
                "key": payload["key"],
                "label": payload["label"],
                "metrics": payload["metrics"],
            }
            for payload in model_payloads.values()
        ),
        key=lambda row: (-float(row["metrics"]["macro_f1"]), row["label"]),
    )
    selected_005_rows = sorted(
        (
            {
                "key": payload["key"],
                "label": payload["label"],
                "strategy": payload["selected_005"]["evaluation"]["strategy"],
                "metrics": payload["selected_005"]["evaluation"]["metrics"],
            }
            for payload in model_payloads.values()
        ),
        key=lambda row: (-float(row["metrics"]["macro_f1"]), row["label"]),
    )
    selected_010_rows = sorted(
        (
            {
                "key": payload["key"],
                "label": payload["label"],
                "strategy": payload["selected_010"]["evaluation"]["strategy"],
                "metrics": payload["selected_010"]["evaluation"]["metrics"],
            }
            for payload in model_payloads.values()
        ),
        key=lambda row: (-float(row["metrics"]["macro_f1"]), row["label"]),
    )

    overlap_buckets = _base_error_bucket_examples(
        gold_rows,
        merged_by_model=merged_by_model,
        model_order=["distilroberta", "roberta_base", "modernbert"],
    )

    experiment = create_experiment_run(args.run_name)
    metadata = {
        "run_name": args.run_name,
        "gold_file": str(args.gold_file),
        "selected_review_rate": args.selected_review_rate,
        "model_specs": {
            model_key: {
                "label": payload["label"],
                "summary_path": payload["summary_path"],
                "base_prediction_file": payload["base_prediction_file"],
            }
            for model_key, payload in model_payloads.items()
        },
    }
    write_run_metadata(experiment.metadata_path, metadata)

    summary = {
        "gold_file": str(args.gold_file),
        "benchmark_row_count": len(gold_rows),
        "ranking": {
            "base_eval_macro_f1": [row["key"] for row in base_rows],
            "selected_005_macro_f1": [row["key"] for row in selected_005_rows],
            "selected_010_macro_f1": [row["key"] for row in selected_010_rows],
        },
        "models": {
            model_key: {
                "label": payload["label"],
                "summary_path": payload["summary_path"],
                "base_prediction_file": payload["base_prediction_file"],
                "eval_base_metrics": payload["metrics"],
                "selected_default_target": payload["selected_default"],
                "selected_005_target": payload["selected_005"],
                "selected_010_target": payload["selected_010"],
            }
            for model_key, payload in model_payloads.items()
        },
        "overlap_buckets": overlap_buckets,
    }
    experiment.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    experiment.report_path.write_text(
        _build_report(
            gold_file=args.gold_file,
            base_rows=base_rows,
            selected_005_rows=selected_005_rows,
            selected_010_rows=selected_010_rows,
            overlap_buckets=overlap_buckets,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "experiment_root": str(experiment.root),
                "summary_path": str(experiment.summary_path),
                "report_path": str(experiment.report_path),
                "base_ranking": summary["ranking"]["base_eval_macro_f1"],
                "selected_005_ranking": summary["ranking"]["selected_005_macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
