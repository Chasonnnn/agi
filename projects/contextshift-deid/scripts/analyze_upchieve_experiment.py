from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
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
from contextshift_deid.metrics import compute_action_metrics

DEFAULT_GOLD_FILE = ACTION_DIR / "upchieve_english_social_test.jsonl"
DEFAULT_BASE_PREDICTION_FILE = PREDICTIONS_DIR / "upchieve_english_social_test_predictions_mixed_modernbert_v2_b4_l384.jsonl"
DEFAULT_SUMMARY_FILE = (
    ROOT / "artifacts/experiments/20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384/summary.json"
)
DEFAULT_SELECTED_REVIEW_RATE = 0.10
DEFAULT_RUN_NAME = "upchieve-modernbert-v2-analysis"
CURRICULAR_ACCEPTABLE_ACCURACY = 0.90
METRIC_ROUNDING_DIGITS = 4
BUCKET_EXAMPLE_LIMIT = 12
FIELD_TO_METRIC_KEY = {
    "subject": "by_context",
    "entity_type": "by_entity_type",
    "semantic_role": "by_semantic_role",
    "eval_slice": "by_eval_slice",
}


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Missing prediction file: {path}")
    return load_jsonl(path)


def _selected_target_entry(summary: Mapping[str, Any], *, target_review_rate: float) -> dict[str, Any]:
    for target in summary.get("selected_targets", []):
        rate = float(target["target_review_rate"])
        if abs(rate - target_review_rate) <= 1e-9:
            return dict(target)
    raise SystemExit(
        f"{summary.get('model_name', 'unknown summary')} is missing a selected target "
        f"for review rate {target_review_rate:.2f}"
    )


def _context_preview(context_text: str, *, span_text: str, radius: int = 72) -> str:
    compact = " ".join(context_text.split())
    if not compact:
        return ""
    lowered = compact.lower()
    needle = span_text.strip()
    lowered_needle = needle.lower()
    if needle and lowered_needle in lowered:
        start = lowered.index(lowered_needle)
        excerpt_start = max(0, start - radius)
        excerpt_end = min(len(compact), start + len(needle) + radius)
        excerpt = compact[excerpt_start:excerpt_end]
        if excerpt_start > 0:
            excerpt = "..." + excerpt
        if excerpt_end < len(compact):
            excerpt += "..."
        return excerpt
    if len(compact) <= radius * 2:
        return compact
    return compact[: radius * 2 - 3].rstrip() + "..."


def _merge_gold_predictions(gold_rows: Sequence[Mapping[str, Any]], prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
                "speaker_role": row.get("speaker_role"),
                "span_text": row.get("span_text"),
                "context_text": row.get("context_text"),
                "gold_action": row["action_label"],
                "predicted_action": prediction["predicted_action"],
            }
        )
    return merged


def _build_comparison_rows(
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    base_rows: Sequence[Mapping[str, Any]],
    selected_prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    base_by_id = {str(row["id"]): row for row in base_rows}
    selected_by_id = {str(row["id"]): row for row in selected_prediction_rows}
    rows: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        row_id = str(gold_row["id"])
        base_row = base_by_id.get(row_id)
        selected_row = selected_by_id.get(row_id)
        if base_row is None:
            raise SystemExit(f"Missing base prediction for id={row_id}")
        if selected_row is None:
            raise SystemExit(f"Missing selected-policy prediction for id={row_id}")
        selected_base_predicted = selected_row.get("base_predicted_action")
        if selected_base_predicted is not None and selected_base_predicted != base_row["predicted_action"]:
            raise SystemExit(
                f"Selected-policy base prediction mismatch for id={row_id}: "
                f"{selected_base_predicted!r} != {base_row['predicted_action']!r}"
            )
        rows.append(
            {
                "id": row_id,
                "subject": str(gold_row.get("subject", "unknown")),
                "entity_type": gold_row.get("entity_type"),
                "semantic_role": gold_row.get("semantic_role"),
                "eval_slice": gold_row.get("eval_slice"),
                "speaker_role": gold_row.get("speaker_role"),
                "span_text": gold_row.get("span_text"),
                "context_text": gold_row.get("context_text"),
                "gold_action": gold_row["action_label"],
                "base_predicted_action": base_row["predicted_action"],
                "selected_predicted_action": selected_row["predicted_action"],
                "deferred": bool(selected_row.get("deferred", False)),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row)) + "\n")


def _compare_value(recorded: Any, computed: Any, *, path: str) -> list[str]:
    mismatches: list[str] = []
    if isinstance(computed, Mapping):
        if not isinstance(recorded, Mapping):
            return [f"{path}: recorded value is not an object"]
        for key, value in computed.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in recorded:
                mismatches.append(f"{child_path}: missing in recorded metrics")
                continue
            mismatches.extend(_compare_value(recorded[key], value, path=child_path))
        return mismatches
    if isinstance(computed, list):
        if not isinstance(recorded, list):
            return [f"{path}: recorded value is not a list"]
        if len(recorded) != len(computed):
            return [f"{path}: recorded length {len(recorded)} != computed length {len(computed)}"]
        for index, value in enumerate(computed):
            mismatches.extend(_compare_value(recorded[index], value, path=f"{path}[{index}]"))
        return mismatches
    if isinstance(computed, float):
        try:
            recorded_float = float(recorded)
        except (TypeError, ValueError):
            return [f"{path}: recorded value {recorded!r} is not numeric"]
        if round(recorded_float, METRIC_ROUNDING_DIGITS) != round(computed, METRIC_ROUNDING_DIGITS):
            mismatches.append(
                f"{path}: recorded {recorded_float:.6f} != computed {computed:.6f} "
                f"at {METRIC_ROUNDING_DIGITS}-decimal rounding"
            )
        return mismatches
    if recorded != computed:
        mismatches.append(f"{path}: recorded {recorded!r} != computed {computed!r}")
    return mismatches


def _validate_metric_reproduction(
    *,
    label: str,
    recorded_metrics: Mapping[str, Any],
    computed_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches = _compare_value(recorded_metrics, computed_metrics, path="")
    if mismatches:
        mismatch_preview = "\n".join(mismatches[:20])
        raise SystemExit(f"{label} metrics did not reproduce:\n{mismatch_preview}")
    return {
        "label": label,
        "matches_recorded_summary": True,
        "rounding_digits": METRIC_ROUNDING_DIGITS,
    }


def _bucket_definition_matches(bucket_name: str, row: Mapping[str, Any], *, action_key: str) -> bool:
    action = str(row[action_key])
    gold_action = str(row["gold_action"])
    if bucket_name == "gold_keep_or_review_to_redact":
        return gold_action in {"KEEP", "REVIEW"} and action == "REDACT"
    if bucket_name == "gold_private_not_redact":
        return row.get("semantic_role") == "PRIVATE" and action != "REDACT"
    if bucket_name == "gold_review_not_review":
        return gold_action == "REVIEW" and action != "REVIEW"
    raise ValueError(f"Unsupported bucket: {bucket_name}")


def _bucket_examples(bucket_name: str, comparison_rows: Sequence[Mapping[str, Any]], *, limit: int) -> dict[str, Any]:
    base_count = sum(
        1 for row in comparison_rows if _bucket_definition_matches(bucket_name, row, action_key="base_predicted_action")
    )
    selected_count = sum(
        1 for row in comparison_rows if _bucket_definition_matches(bucket_name, row, action_key="selected_predicted_action")
    )
    candidate_rows = [
        {
            **row,
            "context_preview": _context_preview(
                str(row.get("context_text", "")),
                span_text=str(row.get("span_text", "")),
            ),
            "base_bucket_match": _bucket_definition_matches(bucket_name, row, action_key="base_predicted_action"),
            "selected_bucket_match": _bucket_definition_matches(bucket_name, row, action_key="selected_predicted_action"),
        }
        for row in comparison_rows
        if _bucket_definition_matches(bucket_name, row, action_key="base_predicted_action")
        or _bucket_definition_matches(bucket_name, row, action_key="selected_predicted_action")
    ]
    candidate_rows.sort(
        key=lambda row: (
            not bool(row["selected_bucket_match"]),
            not bool(row["base_bucket_match"]),
            str(row["id"]),
        )
    )
    return {
        "base_count": base_count,
        "selected_count": selected_count,
        "examples": candidate_rows[:limit],
    }


def _slice_table_lines(
    *,
    section_title: str,
    base_metrics: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    metric_key: str,
) -> list[str]:
    lines = [
        f"### {section_title}",
        "",
        "| variant | slice | count | accuracy | macro_f1 | redact_recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    known_slices = set(base_metrics.get(metric_key, {}).keys()) | set(selected_metrics.get(metric_key, {}).keys())
    for slice_name in sorted(known_slices):
        base_slice = base_metrics.get(metric_key, {}).get(slice_name)
        selected_slice = selected_metrics.get(metric_key, {}).get(slice_name)
        if base_slice is not None:
            lines.append(
                "| base | {slice_name} | {count} | {accuracy} | {macro_f1} | {redact_recall} |".format(
                    slice_name=slice_name,
                    count=base_slice["count"],
                    accuracy=_format_metric(base_slice.get("accuracy")),
                    macro_f1=_format_metric(base_slice.get("macro_f1")),
                    redact_recall=_format_metric(base_slice.get("redact_recall")),
                )
            )
        if selected_slice is not None:
            lines.append(
                "| selected_10pct | {slice_name} | {count} | {accuracy} | {macro_f1} | {redact_recall} |".format(
                    slice_name=slice_name,
                    count=selected_slice["count"],
                    accuracy=_format_metric(selected_slice.get("accuracy")),
                    macro_f1=_format_metric(selected_slice.get("macro_f1")),
                    redact_recall=_format_metric(selected_slice.get("redact_recall")),
                )
            )
    if len(lines) == 4:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return lines


def _top_level_table_lines(
    *,
    base_metrics: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
) -> list[str]:
    curricular_base = base_metrics.get("by_semantic_role", {}).get("CURRICULAR", {})
    curricular_selected = selected_metrics.get("by_semantic_role", {}).get("CURRICULAR", {})
    return [
        "## Top-Line Metrics",
        "",
        "| variant | accuracy | macro_f1 | redact_recall | review_rate | ORR | curricular_accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| base | {accuracy} | {macro_f1} | {redact_recall} | {review_rate} | {orr} | {curricular_accuracy} |".format(
            accuracy=_format_metric(base_metrics.get("accuracy")),
            macro_f1=_format_metric(base_metrics.get("macro_f1")),
            redact_recall=_format_metric(base_metrics.get("redact_recall")),
            review_rate=_format_percent(base_metrics.get("review_rate")),
            orr=_format_percent(base_metrics.get("orr")),
            curricular_accuracy=_format_metric(curricular_base.get("accuracy")),
        ),
        "| selected_10pct | {accuracy} | {macro_f1} | {redact_recall} | {review_rate} | {orr} | {curricular_accuracy} |".format(
            accuracy=_format_metric(selected_metrics.get("accuracy")),
            macro_f1=_format_metric(selected_metrics.get("macro_f1")),
            redact_recall=_format_metric(selected_metrics.get("redact_recall")),
            review_rate=_format_percent(selected_metrics.get("review_rate")),
            orr=_format_percent(selected_metrics.get("orr")),
            curricular_accuracy=_format_metric(curricular_selected.get("accuracy")),
        ),
        "",
    ]


def _bucket_lines(bucket_name: str, bucket: Mapping[str, Any]) -> list[str]:
    label = {
        "gold_keep_or_review_to_redact": "Gold KEEP/REVIEW Predicted REDACT",
        "gold_private_not_redact": "Gold PRIVATE Not Predicted REDACT",
        "gold_review_not_review": "Gold REVIEW Not Sent To REVIEW",
    }[bucket_name]
    lines = [
        f"### {label}",
        "",
        f"- Base count: {bucket['base_count']}",
        f"- Selected count: {bucket['selected_count']}",
        "",
        "| id | subject | entity | role | gold | base | selected | deferred | span | context |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for example in bucket["examples"]:
        lines.append(
            "| {id} | {subject} | {entity_type} | {semantic_role} | {gold_action} | {base_predicted_action} | "
            "{selected_predicted_action} | {deferred} | {span_text} | {context_preview} |".format(
                id=example["id"],
                subject=example["subject"],
                entity_type=example.get("entity_type") or "n/a",
                semantic_role=example.get("semantic_role") or "n/a",
                gold_action=example["gold_action"],
                base_predicted_action=example["base_predicted_action"],
                selected_predicted_action=example["selected_predicted_action"],
                deferred="yes" if example["deferred"] else "no",
                span_text=example.get("span_text") or "",
                context_preview=example["context_preview"],
            )
        )
    if not bucket["examples"]:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return lines


def _slice_question_payload(
    *,
    slice_name: str,
    base_metrics: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    metric_key: str,
) -> dict[str, Any]:
    base_slice = dict(base_metrics.get(metric_key, {}).get(slice_name, {}))
    selected_slice = dict(selected_metrics.get(metric_key, {}).get(slice_name, {}))
    if not base_slice or not selected_slice:
        return {
            "slice_name": slice_name,
            "available": False,
            "assessment": "slice missing from one or both variants",
        }
    delta_macro_f1 = float(selected_slice["macro_f1"]) - float(base_slice["macro_f1"])
    delta_accuracy = float(selected_slice["accuracy"]) - float(base_slice["accuracy"])
    if delta_macro_f1 >= 0.10 or delta_accuracy >= 0.10:
        assessment = "improved materially"
    elif delta_macro_f1 > 0.0 or delta_accuracy > 0.0:
        assessment = "improved slightly"
    else:
        assessment = "did not improve"
    return {
        "slice_name": slice_name,
        "available": True,
        "assessment": assessment,
        "base_accuracy": float(base_slice["accuracy"]),
        "selected_accuracy": float(selected_slice["accuracy"]),
        "base_macro_f1": float(base_slice["macro_f1"]),
        "selected_macro_f1": float(selected_slice["macro_f1"]),
        "base_redact_recall": float(base_slice["redact_recall"]),
        "selected_redact_recall": float(selected_slice["redact_recall"]),
        "count": int(selected_slice["count"]),
    }


def _residual_distribution_payload(comparison_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_errors = [row for row in comparison_rows if row["selected_predicted_action"] != row["gold_action"]]
    if not selected_errors:
        return {
            "available": True,
            "assessment": "no residual selected-policy errors",
            "concentrated": True,
            "selected_error_count": 0,
            "top_entity_types": [],
            "top_two_entity_share": 1.0,
        }
    counts = Counter(str(row.get("entity_type") or "n/a") for row in selected_errors)
    top_entity_types = counts.most_common(3)
    top_two_entity_share = sum(count for _, count in counts.most_common(2)) / len(selected_errors)
    concentrated = top_two_entity_share >= 0.60
    return {
        "available": True,
        "assessment": "concentrated" if concentrated else "diffuse",
        "concentrated": concentrated,
        "selected_error_count": len(selected_errors),
        "top_entity_types": [{"entity_type": label, "count": count} for label, count in top_entity_types],
        "top_two_entity_share": top_two_entity_share,
    }


def _core_questions_payload(
    *,
    base_metrics: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    curricular_selected = dict(selected_metrics.get("by_semantic_role", {}).get("CURRICULAR", {}))
    curricular_accuracy = float(curricular_selected.get("accuracy")) if curricular_selected else None
    return {
        "nrp": _slice_question_payload(
            slice_name="NRP",
            base_metrics=base_metrics,
            selected_metrics=selected_metrics,
            metric_key="by_entity_type",
        ),
        "location": _slice_question_payload(
            slice_name="LOCATION",
            base_metrics=base_metrics,
            selected_metrics=selected_metrics,
            metric_key="by_entity_type",
        ),
        "curricular_accuracy": {
            "available": curricular_accuracy is not None,
            "selected_accuracy": curricular_accuracy,
            "acceptable_threshold": CURRICULAR_ACCEPTABLE_ACCURACY,
            "acceptable": curricular_accuracy is not None and curricular_accuracy >= CURRICULAR_ACCEPTABLE_ACCURACY,
        },
        "residual_distribution": _residual_distribution_payload(comparison_rows),
    }


def _core_question_lines(core_questions: Mapping[str, Any]) -> list[str]:
    nrp = core_questions["nrp"]
    location = core_questions["location"]
    curricular = core_questions["curricular_accuracy"]
    residual = core_questions["residual_distribution"]
    lines = ["## Core Questions", ""]
    if nrp["available"]:
        lines.append(
            "- NRP: {assessment}; accuracy {base_accuracy:.4f} -> {selected_accuracy:.4f}, "
            "macro F1 {base_macro_f1:.4f} -> {selected_macro_f1:.4f}.".format(**nrp)
        )
    else:
        lines.append(f"- NRP: {nrp['assessment']}.")
    if location["available"]:
        lines.append(
            "- LOCATION: {assessment}; accuracy {base_accuracy:.4f} -> {selected_accuracy:.4f}, "
            "macro F1 {base_macro_f1:.4f} -> {selected_macro_f1:.4f}.".format(**location)
        )
    else:
        lines.append(f"- LOCATION: {location['assessment']}.")
    if curricular["available"]:
        lines.append(
            "- CURRICULAR accuracy: {status}; selected accuracy {selected_accuracy:.4f} "
            "vs threshold {acceptable_threshold:.2f}.".format(
                status="acceptable" if curricular["acceptable"] else "not yet acceptable",
                selected_accuracy=curricular["selected_accuracy"],
                acceptable_threshold=curricular["acceptable_threshold"],
            )
        )
    else:
        lines.append("- CURRICULAR accuracy: slice unavailable.")
    if residual["selected_error_count"] == 0:
        lines.append("- Residual errors: none under the selected policy.")
    else:
        top_labels = ", ".join(
            f"{row['entity_type']} ({row['count']})" for row in residual.get("top_entity_types", [])
        )
        lines.append(
            "- Residual errors are {assessment}; top entity types {top_labels} cover {share:.1f}% of "
            "{count} selected-policy errors.".format(
                assessment=residual["assessment"],
                top_labels=top_labels or "n/a",
                share=residual["top_two_entity_share"] * 100,
                count=residual["selected_error_count"],
            )
        )
    lines.append("")
    return lines


def _build_report(
    *,
    gold_file: Path,
    base_prediction_file: Path,
    summary_file: Path,
    selected_prediction_file: Path,
    selected_review_rate: float,
    base_metrics: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    buckets: Mapping[str, Any],
    core_questions: Mapping[str, Any],
    selected_policy: Mapping[str, Any],
) -> str:
    lines = [
        "# UPChieve Experiment Analysis",
        "",
        f"- Gold file: `{gold_file}`",
        f"- Base prediction file: `{base_prediction_file}`",
        f"- Summary file: `{summary_file}`",
        f"- Selected prediction file: `{selected_prediction_file}`",
        f"- Selected review target: {selected_review_rate * 100:.1f}%",
        f"- Selected strategy: `{selected_policy['strategy']}`",
        "",
    ]
    lines.extend(_top_level_table_lines(base_metrics=base_metrics, selected_metrics=selected_metrics))
    lines.append("## Slice Tables")
    lines.append("")
    for field, metric_key in FIELD_TO_METRIC_KEY.items():
        lines.extend(
            _slice_table_lines(
                section_title=field.replace("_", " ").title(),
                base_metrics=base_metrics,
                selected_metrics=selected_metrics,
                metric_key=metric_key,
            )
        )
    lines.append("## Error Buckets")
    lines.append("")
    for bucket_name in (
        "gold_keep_or_review_to_redact",
        "gold_private_not_redact",
        "gold_review_not_review",
    ):
        lines.extend(_bucket_lines(bucket_name, buckets[bucket_name]))
    lines.extend(_core_question_lines(core_questions))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze a single UPChieve action experiment against a frozen benchmark.")
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD_FILE)
    parser.add_argument("--base-prediction-file", type=Path, default=DEFAULT_BASE_PREDICTION_FILE)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--selected-review-rate", type=float, default=DEFAULT_SELECTED_REVIEW_RATE)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args(argv)

    summary = _load_summary(args.summary_file)
    gold_rows = load_jsonl(args.gold_file)
    base_prediction_rows = _load_prediction_rows(args.base_prediction_file)
    selected_target = _selected_target_entry(summary, target_review_rate=args.selected_review_rate)
    selected_prediction_file = Path(selected_target["output_file"])
    selected_prediction_rows = _load_prediction_rows(selected_prediction_file)

    base_rows = _merge_gold_predictions(gold_rows, base_prediction_rows)
    selected_rows = _merge_gold_predictions(gold_rows, selected_prediction_rows)
    base_metrics = compute_action_metrics(base_rows)
    selected_metrics = compute_action_metrics(selected_rows)

    validation_results = [
        _validate_metric_reproduction(
            label="base_eval",
            recorded_metrics=summary["eval_base_metrics"],
            computed_metrics=base_metrics,
        ),
        _validate_metric_reproduction(
            label="selected_eval",
            recorded_metrics=selected_target["evaluation"]["metrics"],
            computed_metrics=selected_metrics,
        ),
    ]

    comparison_rows = _build_comparison_rows(
        gold_rows,
        base_rows=base_rows,
        selected_prediction_rows=selected_prediction_rows,
    )
    buckets = {
        bucket_name: _bucket_examples(bucket_name, comparison_rows, limit=BUCKET_EXAMPLE_LIMIT)
        for bucket_name in (
            "gold_keep_or_review_to_redact",
            "gold_private_not_redact",
            "gold_review_not_review",
        )
    }
    core_questions = _core_questions_payload(
        base_metrics=base_metrics,
        selected_metrics=selected_metrics,
        comparison_rows=comparison_rows,
    )

    experiment = create_experiment_run(args.run_name)
    comparison_path = experiment.predictions_dir / "selected_policy_comparison.jsonl"
    _write_jsonl(comparison_path, comparison_rows)

    metadata = {
        "run_name": args.run_name,
        "gold_file": str(args.gold_file),
        "base_prediction_file": str(args.base_prediction_file),
        "summary_file": str(args.summary_file),
        "selected_review_rate": args.selected_review_rate,
        "selected_prediction_file": str(selected_prediction_file),
        "comparison_rows_file": str(comparison_path),
        "curricular_acceptability_threshold": CURRICULAR_ACCEPTABLE_ACCURACY,
    }
    write_run_metadata(experiment.metadata_path, metadata)

    summary_payload = {
        "gold_file": str(args.gold_file),
        "base_prediction_file": str(args.base_prediction_file),
        "summary_file": str(args.summary_file),
        "selected_review_rate": args.selected_review_rate,
        "selected_prediction_file": str(selected_prediction_file),
        "benchmark_row_count": len(gold_rows),
        "selected_policy": {
            "strategy": selected_target["evaluation"]["strategy"],
            "parameters": selected_target["evaluation"]["parameters"],
            "temperature": selected_target["evaluation"]["temperature"],
            "recorded_metrics": selected_target["evaluation"]["metrics"],
        },
        "metric_validation": validation_results,
        "base_metrics": base_metrics,
        "selected_metrics": selected_metrics,
        "bucket_counts": {
            bucket_name: {
                "base_count": payload["base_count"],
                "selected_count": payload["selected_count"],
            }
            for bucket_name, payload in buckets.items()
        },
        "core_questions": core_questions,
        "comparison_rows_file": str(comparison_path),
    }
    experiment.summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    experiment.report_path.write_text(
        _build_report(
            gold_file=args.gold_file,
            base_prediction_file=args.base_prediction_file,
            summary_file=args.summary_file,
            selected_prediction_file=selected_prediction_file,
            selected_review_rate=args.selected_review_rate,
            base_metrics=base_metrics,
            selected_metrics=selected_metrics,
            buckets=buckets,
            core_questions=core_questions,
            selected_policy=summary_payload["selected_policy"],
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "experiment_root": str(experiment.root),
                "summary_path": str(experiment.summary_path),
                "report_path": str(experiment.report_path),
                "comparison_rows_file": str(comparison_path),
                "selected_macro_f1": selected_metrics["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
