from __future__ import annotations

from collections import Counter
from math import exp, log
from typing import Any, Mapping, Sequence

from .constants import DEFAULT_ACTION_LABELS
from .metrics import compute_action_metrics

_EPSILON = 1e-12


def normalize_probability_map(probabilities: Mapping[str, float]) -> dict[str, float]:
    normalized = {label: max(0.0, float(probabilities.get(label, 0.0))) for label in DEFAULT_ACTION_LABELS}
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("Probability map must contain at least one positive value.")
    return {label: value / total for label, value in normalized.items()}


def probability_argmax(probabilities: Mapping[str, float]) -> str:
    normalized = normalize_probability_map(probabilities)
    return max(DEFAULT_ACTION_LABELS, key=lambda label: normalized[label])


def temperature_scale_probability_map(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    normalized = normalize_probability_map(probabilities)
    if abs(temperature - 1.0) < 1e-9:
        return normalized

    scaled_logits = {
        label: log(max(probability, _EPSILON)) / temperature
        for label, probability in normalized.items()
    }
    max_logit = max(scaled_logits.values())
    weights = {
        label: exp(value - max_logit)
        for label, value in scaled_logits.items()
    }
    total = sum(weights.values())
    return {label: value / total for label, value in weights.items()}


def fit_temperature(
    records: Sequence[Mapping[str, Any]],
    *,
    gold_field: str = "gold_action",
    probability_field: str = "probabilities",
    min_temperature: float = 0.25,
    max_temperature: float = 5.0,
    steps: int = 96,
) -> float:
    if not records:
        raise ValueError("Cannot fit temperature without calibration records.")
    if min_temperature <= 0.0 or max_temperature <= 0.0:
        raise ValueError("Temperature range must be positive.")
    if steps < 2:
        raise ValueError("steps must be at least 2.")

    temperatures = [
        exp(log(min_temperature) + ((log(max_temperature) - log(min_temperature)) * index / (steps - 1)))
        for index in range(steps)
    ]
    temperatures.append(1.0)

    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in sorted(dict.fromkeys(temperatures)):
        losses: list[float] = []
        for record in records:
            gold_label = str(record[gold_field])
            scaled_probabilities = temperature_scale_probability_map(record[probability_field], temperature)
            losses.append(-log(max(scaled_probabilities[gold_label], _EPSILON)))
        nll = sum(losses) / len(losses)
        if nll < best_nll:
            best_temperature = temperature
            best_nll = nll
    return best_temperature


def probability_features(probabilities: Mapping[str, float]) -> dict[str, float | str]:
    normalized = normalize_probability_map(probabilities)
    predicted_action = probability_argmax(normalized)
    confidence = max(normalized.values())
    redact_keep_margin = abs(normalized["REDACT"] - normalized["KEEP"])
    entropy = -sum(
        probability * log(max(probability, _EPSILON))
        for probability in normalized.values()
    )
    normalized_entropy = entropy / log(len(DEFAULT_ACTION_LABELS))
    return {
        "predicted_action": predicted_action,
        "confidence": confidence,
        "review_probability": normalized["REVIEW"],
        "redact_keep_margin": redact_keep_margin,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def _should_defer(
    *,
    predicted_action: str,
    features: Mapping[str, float | str],
    strategy: str,
    parameters: Mapping[str, float],
) -> bool:
    if predicted_action == "REVIEW":
        # Preserve model-emitted REVIEW labels instead of passing them through
        # post-hoc deferral thresholds intended for REDACT/KEEP decisions.
        return False

    confidence = float(features["confidence"])
    review_probability = float(features["review_probability"])
    redact_keep_margin = float(features["redact_keep_margin"])
    entropy = float(features["entropy"])

    if strategy == "confidence":
        return confidence <= float(parameters["max_confidence"])
    if strategy == "margin":
        return redact_keep_margin <= float(parameters["max_redact_keep_margin"])
    if strategy == "entropy":
        return entropy >= float(parameters["min_entropy"])
    if strategy == "review_probability":
        return review_probability >= float(parameters["min_review_probability"])
    if strategy == "composite":
        return (
            confidence <= float(parameters["max_confidence"])
            or redact_keep_margin <= float(parameters["max_redact_keep_margin"])
            or entropy >= float(parameters["min_entropy"])
            or review_probability >= float(parameters["min_review_probability"])
        )
    if strategy == "asymmetric_confidence":
        if predicted_action == "REDACT":
            return confidence <= float(parameters["redact_max_confidence"])
        return confidence <= float(parameters["keep_max_confidence"])
    if strategy == "asymmetric_margin":
        if predicted_action == "REDACT":
            return redact_keep_margin <= float(parameters["redact_max_redact_keep_margin"])
        return redact_keep_margin <= float(parameters["keep_max_redact_keep_margin"])
    raise ValueError(f"Unsupported deferral strategy: {strategy}")


def apply_deferral_policy(
    records: Sequence[Mapping[str, Any]],
    *,
    strategy: str,
    parameters: Mapping[str, float],
    temperature: float = 1.0,
    probability_field: str = "probabilities",
    predicted_field: str = "predicted_action",
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for record in records:
        calibrated_probabilities = temperature_scale_probability_map(record[probability_field], temperature)
        features = probability_features(calibrated_probabilities)
        base_predicted_action = str(record.get(predicted_field) or features["predicted_action"])
        defer = _should_defer(
            predicted_action=base_predicted_action,
            features=features,
            strategy=strategy,
            parameters=parameters,
        )
        remapped.append(
            {
                **record,
                "base_predicted_action": base_predicted_action,
                "predicted_action": "REVIEW" if defer else base_predicted_action,
                "deferred": defer,
                "calibrated_probabilities": calibrated_probabilities,
                "calibrated_confidence": float(features["confidence"]),
                "calibrated_review_probability": float(features["review_probability"]),
                "calibrated_redact_keep_margin": float(features["redact_keep_margin"]),
                "calibrated_entropy": float(features["entropy"]),
            }
        )
    return remapped


def compute_deferral_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No action records supplied for deferral evaluation.")

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
    metrics = compute_action_metrics(evaluation_rows)

    total = len(records)
    deferred_count = sum(1 for record in records if bool(record.get("deferred")))
    gold_redact = [record for record in records if record["gold_action"] == "REDACT"]
    gold_review = [record for record in records if record["gold_action"] == "REVIEW"]
    predicted_review = [record for record in records if record["predicted_action"] == "REVIEW"]
    base_errors = [
        record
        for record in records
        if record.get("base_predicted_action", record["predicted_action"]) != record["gold_action"]
    ]
    base_correct = [
        record
        for record in records
        if record.get("base_predicted_action", record["predicted_action"]) == record["gold_action"]
    ]

    metrics["automation_rate"] = 1.0 - metrics["review_rate"]
    metrics["deferred_count"] = deferred_count
    metrics["base_prediction_label_distribution"] = Counter(
        str(record.get("base_predicted_action", record["predicted_action"]))
        for record in records
    )

    if gold_redact:
        protected = sum(1 for record in gold_redact if record["predicted_action"] != "KEEP")
        deferred_gold_redact = sum(1 for record in gold_redact if record["predicted_action"] == "REVIEW")
        metrics["protected_redact_rate"] = protected / len(gold_redact)
        metrics["redact_leak_rate"] = 1.0 - metrics["protected_redact_rate"]
        metrics["gold_redact_defer_rate"] = deferred_gold_redact / len(gold_redact)
    else:
        metrics["protected_redact_rate"] = None
        metrics["redact_leak_rate"] = None
        metrics["gold_redact_defer_rate"] = None

    if gold_review:
        covered = sum(1 for record in gold_review if record["predicted_action"] == "REVIEW")
        metrics["gold_review_coverage"] = covered / len(gold_review)
    else:
        metrics["gold_review_coverage"] = None

    if predicted_review:
        review_precision = sum(1 for record in predicted_review if record["gold_action"] == "REVIEW") / len(predicted_review)
        metrics["review_precision"] = review_precision
    else:
        metrics["review_precision"] = None

    if base_errors:
        deferred_errors = sum(1 for record in base_errors if bool(record.get("deferred")))
        metrics["base_error_rate"] = len(base_errors) / total
        metrics["deferred_error_coverage"] = deferred_errors / len(base_errors)
    else:
        metrics["base_error_rate"] = 0.0
        metrics["deferred_error_coverage"] = None

    if base_correct:
        deferred_correct = sum(1 for record in base_correct if bool(record.get("deferred")))
        metrics["deferred_correct_rate"] = deferred_correct / len(base_correct)
    else:
        metrics["deferred_correct_rate"] = None

    return metrics
