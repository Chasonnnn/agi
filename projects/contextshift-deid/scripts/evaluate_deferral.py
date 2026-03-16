from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
from math import ceil
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.data import load_jsonl
from contextshift_deid.deferral import apply_deferral_policy, compute_deferral_metrics, fit_temperature, probability_features, temperature_scale_probability_map
from contextshift_deid.experiment_runs import create_experiment_run, write_run_metadata
from contextshift_deid.metrics import compute_action_metrics

_SIMPLE_ANCHORS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
_COMPOSITE_ANCHORS = (0.01, 0.03, 0.05, 0.10)
_ASYMMETRIC_ANCHORS = (0.01, 0.03, 0.05, 0.10, 0.15)


def _parse_target_review_rates(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = float(chunk)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"Target review rates must be in [0, 1). Invalid value: {chunk}")
        values.append(value)
    if not values:
        raise ValueError("At least one target review rate is required.")
    return sorted(set(values))


def _merge_action_rows(gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    merged: list[dict[str, Any]] = []
    for row in gold_rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row_id}")
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise SystemExit(
                f"Prediction file is missing probabilities for id={row_id}. "
                "Run a probability-export capable action inference path first."
            )
        merged.append(
            {
                "id": row_id,
                "subject": row.get("subject", "unknown"),
                "eval_slice": row.get("eval_slice"),
                "gold_action": row["action_label"],
                "predicted_action": prediction["predicted_action"],
                "confidence": prediction.get("confidence"),
                "probabilities": probabilities,
                "speaker_role": row.get("speaker_role"),
                "entity_type": row.get("entity_type"),
                "semantic_role": row.get("semantic_role"),
                "span_text": row.get("span_text"),
                "context_text": row.get("context_text"),
                "cost": prediction.get("cost"),
                "latency_ms": prediction.get("latency_ms"),
            }
        )
    return merged


def _base_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluation_rows = [
        {
            "id": record["id"],
            "subject": record.get("subject", "unknown"),
            "eval_slice": record.get("eval_slice"),
            "gold_action": record["gold_action"],
            "predicted_action": record["predicted_action"],
            "speaker_role": record.get("speaker_role"),
            "entity_type": record.get("entity_type"),
            "semantic_role": record.get("semantic_role"),
            "cost": record.get("cost"),
            "latency_ms": record.get("latency_ms"),
        }
        for record in records
    ]
    return compute_action_metrics(evaluation_rows)


def _feature_rows(records: list[dict[str, Any]], *, temperature: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        scaled_probabilities = temperature_scale_probability_map(record["probabilities"], temperature)
        features = probability_features(scaled_probabilities)
        rows.append(
            {
                "id": record["id"],
                "predicted_action": record["predicted_action"],
                "confidence": float(features["confidence"]),
                "review_probability": float(features["review_probability"]),
                "redact_keep_margin": float(features["redact_keep_margin"]),
                "entropy": float(features["entropy"]),
            }
        )
    return rows


def _threshold_for_lower_tail(values: Iterable[float], rate: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot derive a threshold from an empty value set.")
    index = max(0, ceil(rate * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _threshold_for_upper_tail(values: Iterable[float], rate: float) -> float:
    ordered = sorted(values, reverse=True)
    if not ordered:
        raise ValueError("Cannot derive a threshold from an empty value set.")
    index = max(0, ceil(rate * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def _dedupe_parameter_sets(parameter_sets: Iterable[dict[str, float]]) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for parameter_set in parameter_sets:
        key = tuple(sorted((name, round(value, 12)) for name, value in parameter_set.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(parameter_set)
    return unique


def _generate_simple_candidates(feature_rows: list[dict[str, Any]], strategy: str) -> list[dict[str, float]]:
    if strategy == "confidence":
        return _dedupe_parameter_sets(
            {"max_confidence": _threshold_for_lower_tail((row["confidence"] for row in feature_rows), rate)}
            for rate in _SIMPLE_ANCHORS
        )
    if strategy == "margin":
        return _dedupe_parameter_sets(
            {
                "max_redact_keep_margin": _threshold_for_lower_tail(
                    (row["redact_keep_margin"] for row in feature_rows),
                    rate,
                )
            }
            for rate in _SIMPLE_ANCHORS
        )
    if strategy == "entropy":
        return _dedupe_parameter_sets(
            {"min_entropy": _threshold_for_upper_tail((row["entropy"] for row in feature_rows), rate)}
            for rate in _SIMPLE_ANCHORS
        )
    if strategy == "review_probability":
        return _dedupe_parameter_sets(
            {
                "min_review_probability": _threshold_for_upper_tail(
                    (row["review_probability"] for row in feature_rows),
                    rate,
                )
            }
            for rate in _SIMPLE_ANCHORS
        )
    raise ValueError(f"Unsupported simple strategy: {strategy}")


def _generate_composite_candidates(feature_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    # Composite is an OR of uncertainty signals, so keeping all thresholds on the
    # same anchor rate produces a smaller, more interpretable search space than a
    # full cartesian product over four mostly overlapping conditions.
    return _dedupe_parameter_sets(
        {
            "max_confidence": _threshold_for_lower_tail((row["confidence"] for row in feature_rows), rate),
            "max_redact_keep_margin": _threshold_for_lower_tail(
                (row["redact_keep_margin"] for row in feature_rows),
                rate,
            ),
            "min_entropy": _threshold_for_upper_tail((row["entropy"] for row in feature_rows), rate),
            "min_review_probability": _threshold_for_upper_tail(
                (row["review_probability"] for row in feature_rows),
                rate,
            ),
        }
        for rate in _COMPOSITE_ANCHORS
    )


def _generate_asymmetric_candidates(feature_rows: list[dict[str, Any]], strategy: str) -> list[dict[str, float]]:
    redact_rows = [row for row in feature_rows if row["predicted_action"] == "REDACT"]
    keep_rows = [row for row in feature_rows if row["predicted_action"] == "KEEP"]
    if not redact_rows or not keep_rows:
        return []

    if strategy == "asymmetric_confidence":
        redact_thresholds = _dedupe_parameter_sets(
            {"redact_max_confidence": _threshold_for_lower_tail((row["confidence"] for row in redact_rows), rate)}
            for rate in _ASYMMETRIC_ANCHORS
        )
        keep_thresholds = _dedupe_parameter_sets(
            {"keep_max_confidence": _threshold_for_lower_tail((row["confidence"] for row in keep_rows), rate)}
            for rate in _ASYMMETRIC_ANCHORS
        )
    elif strategy == "asymmetric_margin":
        redact_thresholds = _dedupe_parameter_sets(
            {
                "redact_max_redact_keep_margin": _threshold_for_lower_tail(
                    (row["redact_keep_margin"] for row in redact_rows),
                    rate,
                )
            }
            for rate in _ASYMMETRIC_ANCHORS
        )
        keep_thresholds = _dedupe_parameter_sets(
            {
                "keep_max_redact_keep_margin": _threshold_for_lower_tail(
                    (row["redact_keep_margin"] for row in keep_rows),
                    rate,
                )
            }
            for rate in _ASYMMETRIC_ANCHORS
        )
    else:
        raise ValueError(f"Unsupported asymmetric strategy: {strategy}")

    return _dedupe_parameter_sets(
        {**redact, **keep}
        for redact in redact_thresholds
        for keep in keep_thresholds
    )


def _evaluate_policy_result(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    parameters: Mapping[str, float],
    temperature: float,
) -> dict[str, Any]:
    remapped = apply_deferral_policy(
        records,
        strategy=strategy,
        parameters=parameters,
        temperature=temperature,
    )
    metrics = compute_deferral_metrics(remapped)
    return {
        "strategy": strategy,
        "parameters": dict(parameters),
        "temperature": temperature,
        "metrics": metrics,
    }


def _selection_key(result: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = result["metrics"]
    return (
        float(metrics.get("gold_review_coverage") or 0.0),
        float(metrics.get("protected_redact_rate") or 0.0),
        float(metrics.get("macro_f1") or 0.0),
        -(float(metrics.get("orr") or 0.0)),
        -(float(metrics.get("review_rate") or 0.0)),
    )


def _select_best_for_target(results: list[dict[str, Any]], *, target_review_rate: float) -> dict[str, Any] | None:
    eligible = [
        result
        for result in results
        if float(result["metrics"]["review_rate"]) <= target_review_rate + 1e-9
    ]
    if not eligible:
        return None
    return max(eligible, key=_selection_key)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _report_top_review_errors(
    records: list[dict[str, Any]],
    *,
    temperature: float,
    limit: int = 8,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        if record["gold_action"] != "REVIEW":
            continue
        scaled_probabilities = temperature_scale_probability_map(record["probabilities"], temperature)
        features = probability_features(scaled_probabilities)
        base_predicted_action = str(record["predicted_action"])
        if base_predicted_action == "REVIEW":
            continue
        examples.append(
            {
                "id": record["id"],
                "base_predicted_action": base_predicted_action,
                "confidence": float(features["confidence"]),
                "review_probability": float(features["review_probability"]),
                "redact_keep_margin": float(features["redact_keep_margin"]),
                "span_text": record.get("span_text"),
                "context_preview": str(record.get("context_text", "")).replace("\n", " ")[:180],
            }
        )
    examples.sort(key=lambda row: (-row["confidence"], row["id"]))
    return examples[:limit]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _base_review_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "base_predicted_action": record["predicted_action"],
            "deferred": False,
        }
        for record in records
    ]


def _evaluate_selected_policy(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    parameters: Mapping[str, float],
    temperature: float,
) -> list[dict[str, Any]]:
    if strategy == "none":
        return _base_review_records(records)
    return apply_deferral_policy(
        records,
        strategy=strategy,
        parameters=parameters,
        temperature=temperature,
    )


def _build_report(
    *,
    model_name: str,
    calibration_base_metrics: Mapping[str, Any],
    eval_base_metrics: Mapping[str, Any],
    temperature: float,
    target_summaries: list[dict[str, Any]],
    top_review_errors: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Deferral Evaluation: {model_name}",
        "",
        "## Base Action Metrics",
        "",
        f"- Calibration macro F1: {_format_metric(float(calibration_base_metrics['macro_f1']))}",
        f"- Calibration accuracy: {_format_metric(float(calibration_base_metrics['accuracy']))}",
        f"- Calibration redact recall: {_format_metric(float(calibration_base_metrics['redact_recall']))}",
        f"- Eval macro F1: {_format_metric(float(eval_base_metrics['macro_f1']))}",
        f"- Eval accuracy: {_format_metric(float(eval_base_metrics['accuracy']))}",
        f"- Eval redact recall: {_format_metric(float(eval_base_metrics['redact_recall']))}",
        f"- Temperature: {temperature:.4f}",
        "",
        "## Selected Policies",
        "",
    ]
    for summary in target_summaries:
        target = summary["target_review_rate"]
        calibration = summary["calibration"]
        evaluation = summary["evaluation"]
        lines.extend(
            [
                f"### Target review rate <= {target * 100:.1f}%",
                "",
                f"- Strategy: `{calibration['strategy']}`",
                f"- Parameters: `{json.dumps(calibration['parameters'], sort_keys=True)}`",
                f"- Calibration review rate: {_format_percent(float(calibration['metrics']['review_rate']))}",
                f"- Calibration gold REVIEW coverage: {_format_percent(calibration['metrics'].get('gold_review_coverage'))}",
                f"- Eval review rate: {_format_percent(float(evaluation['metrics']['review_rate']))}",
                f"- Eval gold REVIEW coverage: {_format_percent(evaluation['metrics'].get('gold_review_coverage'))}",
                f"- Eval protected REDACT rate: {_format_percent(evaluation['metrics'].get('protected_redact_rate'))}",
                f"- Eval macro F1: {_format_metric(float(evaluation['metrics']['macro_f1']))}",
                f"- Eval ORR: {_format_percent(evaluation['metrics'].get('orr'))}",
                "",
            ]
        )

    lines.extend(
        [
            "## High-Confidence Gold REVIEW Misses",
            "",
            "| id | base | conf | p(review) | margin | span | context |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for example in top_review_errors:
        lines.append(
            "| {id} | {base_predicted_action} | {confidence:.3f} | {review_probability:.3f} | "
            "{redact_keep_margin:.3f} | {span_text} | {context_preview} |".format(**example)
        )
    if not top_review_errors:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate confidence-based REVIEW deferral policies.")
    parser.add_argument("--calibration-gold", type=Path, required=True)
    parser.add_argument("--calibration-predictions", type=Path, required=True)
    parser.add_argument("--eval-gold", type=Path, required=True)
    parser.add_argument("--eval-predictions", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--target-review-rates", default="0.05,0.10")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--temperature-min", type=float, default=0.25)
    parser.add_argument("--temperature-max", type=float, default=5.0)
    parser.add_argument("--temperature-steps", type=int, default=96)
    args = parser.parse_args(argv)

    target_review_rates = _parse_target_review_rates(args.target_review_rates)
    model_name = args.model_name or args.run_name

    calibration_records = _merge_action_rows(
        load_jsonl(args.calibration_gold),
        load_jsonl(args.calibration_predictions),
    )
    eval_records = _merge_action_rows(
        load_jsonl(args.eval_gold),
        load_jsonl(args.eval_predictions),
    )

    calibration_base_metrics = _base_metrics(calibration_records)
    eval_base_metrics = _base_metrics(eval_records)
    temperature = 1.0
    if args.fit_temperature:
        temperature = fit_temperature(
            calibration_records,
            min_temperature=args.temperature_min,
            max_temperature=args.temperature_max,
            steps=args.temperature_steps,
        )

    calibration_feature_rows = _feature_rows(calibration_records, temperature=temperature)
    strategy_parameter_sets: dict[str, list[dict[str, float]]] = {
        "confidence": _generate_simple_candidates(calibration_feature_rows, "confidence"),
        "margin": _generate_simple_candidates(calibration_feature_rows, "margin"),
        "entropy": _generate_simple_candidates(calibration_feature_rows, "entropy"),
        "review_probability": _generate_simple_candidates(calibration_feature_rows, "review_probability"),
        "composite": _generate_composite_candidates(calibration_feature_rows),
        "asymmetric_confidence": _generate_asymmetric_candidates(calibration_feature_rows, "asymmetric_confidence"),
        "asymmetric_margin": _generate_asymmetric_candidates(calibration_feature_rows, "asymmetric_margin"),
    }

    calibration_results: list[dict[str, Any]] = [
        {
            "strategy": "none",
            "parameters": {},
            "temperature": temperature,
            "metrics": compute_deferral_metrics(_base_review_records(calibration_records)),
        }
    ]
    for strategy, parameter_sets in strategy_parameter_sets.items():
        for parameter_set in parameter_sets:
            calibration_results.append(
                _evaluate_policy_result(
                    calibration_records,
                    strategy=strategy,
                    parameters=parameter_set,
                    temperature=temperature,
                )
            )

    target_summaries: list[dict[str, Any]] = []
    target_outputs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    selected_prediction_paths: list[str] = []
    for target_review_rate in target_review_rates:
        calibration_selection = _select_best_for_target(
            calibration_results,
            target_review_rate=target_review_rate,
        )
        if calibration_selection is None:
            continue
        eval_remapped = _evaluate_selected_policy(
            eval_records,
            strategy=calibration_selection["strategy"],
            parameters=calibration_selection["parameters"],
            temperature=temperature,
        )
        eval_metrics = compute_deferral_metrics(eval_remapped)
        output_name = f"eval_target_{int(target_review_rate * 1000):03d}_predictions.jsonl"
        target_summary = (
            {
                "target_review_rate": target_review_rate,
                "calibration": calibration_selection,
                "evaluation": {
                    "strategy": calibration_selection["strategy"],
                    "parameters": calibration_selection["parameters"],
                    "temperature": temperature,
                    "metrics": eval_metrics,
                },
                "output_name": output_name,
            }
        )
        target_summaries.append(target_summary)
        target_outputs.append((target_summary, eval_remapped))

    experiment = create_experiment_run(args.run_name)
    metadata = {
        "run_name": args.run_name,
        "model_name": model_name,
        "calibration_gold": str(args.calibration_gold),
        "calibration_predictions": str(args.calibration_predictions),
        "eval_gold": str(args.eval_gold),
        "eval_predictions": str(args.eval_predictions),
        "target_review_rates": target_review_rates,
        "fit_temperature": args.fit_temperature,
        "temperature": temperature,
        "strategy_counts": {strategy: len(parameter_sets) for strategy, parameter_sets in strategy_parameter_sets.items()},
    }
    write_run_metadata(experiment.metadata_path, metadata)

    written_targets: list[dict[str, Any]] = []
    for target_summary, eval_remapped in target_outputs:
        output_path = experiment.predictions_dir / target_summary["output_name"]
        _write_jsonl(output_path, eval_remapped)
        selected_prediction_paths.append(str(output_path))
        written_targets.append(
            {
                **target_summary,
                "output_file": str(output_path),
            }
        )

    sweep_path = experiment.root / "sweep_results.json"
    sweep_path.write_text(json.dumps(calibration_results, indent=2), encoding="utf-8")

    top_review_errors = _report_top_review_errors(
        eval_records,
        temperature=temperature,
    )
    summary = {
        "model_name": model_name,
        "temperature": temperature,
        "calibration_base_metrics": calibration_base_metrics,
        "eval_base_metrics": eval_base_metrics,
        "selected_targets": written_targets,
        "selected_prediction_files": selected_prediction_paths,
        "sweep_result_count": len(calibration_results),
    }
    experiment.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    experiment.report_path.write_text(
        _build_report(
            model_name=model_name,
            calibration_base_metrics=calibration_base_metrics,
            eval_base_metrics=eval_base_metrics,
            temperature=temperature,
            target_summaries=written_targets,
            top_review_errors=top_review_errors,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "experiment_root": str(experiment.root),
                "metadata_path": str(experiment.metadata_path),
                "summary_path": str(experiment.summary_path),
                "report_path": str(experiment.report_path),
                "sweep_path": str(sweep_path),
                "temperature": temperature,
                "selected_prediction_files": selected_prediction_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
