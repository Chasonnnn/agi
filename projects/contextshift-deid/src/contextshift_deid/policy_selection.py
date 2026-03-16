from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import ceil
from typing import Any

from .deferral import (
    apply_deferral_policy,
    compute_deferral_metrics,
    fit_temperature,
    probability_features,
    temperature_scale_probability_map,
)
from .direct_id_rules import apply_direct_id_overrides

DEFAULT_POLICY_SELECTION_TARGET_REVIEW_RATE = 0.10
_SIMPLE_ANCHORS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
_COMPOSITE_ANCHORS = (0.01, 0.03, 0.05, 0.10)
_ASYMMETRIC_ANCHORS = (0.01, 0.03, 0.05, 0.10, 0.15)


def merge_action_rows(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    merged: list[dict[str, Any]] = []
    for row in gold_rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise ValueError(f"Missing prediction for id={row_id}")
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, Mapping):
            raise ValueError(f"Prediction for id={row_id} is missing per-class probabilities.")
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


def _base_review_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "base_predicted_action": record["predicted_action"],
            "deferred": False,
        }
        for record in records
    ]


def _feature_rows(records: Sequence[Mapping[str, Any]], *, temperature: float) -> list[dict[str, Any]]:
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


def _generate_simple_candidates(feature_rows: Sequence[Mapping[str, Any]], strategy: str) -> list[dict[str, float]]:
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


def _generate_composite_candidates(feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
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


def _generate_asymmetric_candidates(feature_rows: Sequence[Mapping[str, Any]], strategy: str) -> list[dict[str, float]]:
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
    records: Sequence[Mapping[str, Any]],
    *,
    strategy: str,
    parameters: Mapping[str, float],
    temperature: float,
) -> dict[str, Any]:
    remapped = _base_review_records(records) if strategy == "none" else apply_deferral_policy(
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
        "records": remapped,
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


def _select_best_for_target(
    results: Sequence[Mapping[str, Any]],
    *,
    target_review_rate: float,
) -> dict[str, Any] | None:
    eligible = [
        result
        for result in results
        if float(result["metrics"]["review_rate"]) <= target_review_rate + 1e-9
    ]
    if not eligible:
        return dict(
            max(
                results,
                key=lambda result: (
                    -abs(float(result["metrics"]["review_rate"]) - target_review_rate),
                    float(result["metrics"].get("protected_redact_rate") or 0.0),
                    float(result["metrics"].get("worst_context_redact_recall") or 0.0),
                    float(result["metrics"].get("macro_f1") or 0.0),
                    -(float(result["metrics"].get("orr") or 0.0)),
                ),
            )
        )
    return dict(max(eligible, key=_selection_key))


def select_policy_for_target(
    records: Sequence[Mapping[str, Any]],
    *,
    target_review_rate: float = DEFAULT_POLICY_SELECTION_TARGET_REVIEW_RATE,
    fit_temperature_on_records: bool = True,
) -> dict[str, Any]:
    temperature = 1.0
    if fit_temperature_on_records:
        temperature = fit_temperature(records)

    feature_rows = _feature_rows(records, temperature=temperature)
    strategy_parameter_sets: dict[str, list[dict[str, float]]] = {
        "confidence": _generate_simple_candidates(feature_rows, "confidence"),
        "margin": _generate_simple_candidates(feature_rows, "margin"),
        "entropy": _generate_simple_candidates(feature_rows, "entropy"),
        "review_probability": _generate_simple_candidates(feature_rows, "review_probability"),
        "composite": _generate_composite_candidates(feature_rows),
        "asymmetric_confidence": _generate_asymmetric_candidates(feature_rows, "asymmetric_confidence"),
        "asymmetric_margin": _generate_asymmetric_candidates(feature_rows, "asymmetric_margin"),
    }
    calibration_results: list[dict[str, Any]] = [
        _evaluate_policy_result(records, strategy="none", parameters={}, temperature=temperature)
    ]
    for strategy, parameter_sets in strategy_parameter_sets.items():
        for parameter_set in parameter_sets:
            calibration_results.append(
                _evaluate_policy_result(
                    records,
                    strategy=strategy,
                    parameters=parameter_set,
                    temperature=temperature,
                )
            )

    selected = _select_best_for_target(calibration_results, target_review_rate=target_review_rate)
    if selected is None:
        raise ValueError(f"No policy satisfied target review rate <= {target_review_rate:.3f}")
    remapped_records = list(selected.pop("records"))
    return {
        "temperature": temperature,
        "target_review_rate": target_review_rate,
        "selected_target": {
            "target_review_rate": target_review_rate,
            "calibration": {
                "strategy": selected["strategy"],
                "parameters": selected["parameters"],
                "temperature": temperature,
                "metrics": selected["metrics"],
            },
            "evaluation": {
                "strategy": selected["strategy"],
                "parameters": selected["parameters"],
                "temperature": temperature,
                "metrics": selected["metrics"],
            },
        },
        "sweep_results": [
            {
                key: value
                for key, value in result.items()
                if key != "records"
            }
            for result in calibration_results
        ],
        "evaluation_rows": remapped_records,
    }


def evaluate_direct_id_policy(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    target_review_rate: float = DEFAULT_POLICY_SELECTION_TARGET_REVIEW_RATE,
    fit_temperature_on_records: bool = True,
) -> dict[str, Any]:
    patched_prediction_rows, override_summary = apply_direct_id_overrides(gold_rows, prediction_rows)
    records = merge_action_rows(gold_rows, patched_prediction_rows)
    selected = select_policy_for_target(
        records,
        target_review_rate=target_review_rate,
        fit_temperature_on_records=fit_temperature_on_records,
    )
    return {
        **selected,
        "patched_prediction_rows": patched_prediction_rows,
        "override_summary": override_summary,
    }
