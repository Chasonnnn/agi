from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score


def _count_positive_rows(label_rows: list[list[str]]) -> int:
    return sum(1 for row in label_rows if any(label != "O" for label in row))


def _compute_candidate_slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold_labels"] for row in rows]
    pred = [row["predicted_labels"] for row in rows]
    return {
        "count": len(rows),
        "precision": seq_precision_score(gold, pred),
        "recall": seq_recall_score(gold, pred),
        "f1": seq_f1_score(gold, pred),
        "gold_positive_rows": _count_positive_rows(gold),
        "predicted_positive_rows": _count_positive_rows(pred),
    }


def _compute_action_slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [row["gold_action"] for row in rows]
    y_pred = [row["predicted_action"] for row in rows]
    redact_recall = recall_score(
        y_true,
        y_pred,
        labels=["REDACT"],
        average=None,
        zero_division=0,
    )[0]
    return {
        "count": len(rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "redact_recall": redact_recall,
    }


def _group_non_null(records: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        grouped[str(value)].append(record)
    return grouped


def compute_candidate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No candidate records supplied for evaluation.")

    gold = [record["gold_labels"] for record in records]
    pred = [record["predicted_labels"] for record in records]
    metrics: dict[str, Any] = _compute_candidate_slice_metrics(records)
    metrics["token_count"] = sum(len(row) for row in gold)
    metrics["row_count"] = len(records)

    context_metrics: dict[str, Any] = {}
    recalls: list[float] = []
    for context, rows in _group_non_null(records, "subject").items():
        slice_metrics = _compute_candidate_slice_metrics(rows)
        context_recall = slice_metrics["recall"]
        recalls.append(context_recall)
        context_metrics[context] = slice_metrics
    metrics["by_context"] = context_metrics
    metrics["worst_context_recall"] = min(recalls) if recalls else None

    positive_rows = [record for record in records if record.get("has_positive_label")]
    if positive_rows:
        metrics["positive_rows_only"] = _compute_candidate_slice_metrics(positive_rows)

    speaker_metrics: dict[str, Any] = {}
    for speaker_role, rows in _group_non_null(records, "speaker_role").items():
        speaker_metrics[speaker_role] = _compute_candidate_slice_metrics(rows)
    if speaker_metrics:
        metrics["by_speaker_role"] = speaker_metrics

    length_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        token_count = int(record.get("token_count", len(record["gold_labels"])))
        if token_count <= 4:
            bucket = "1-4"
        elif token_count <= 8:
            bucket = "5-8"
        elif token_count <= 16:
            bucket = "9-16"
        else:
            bucket = "17+"
        length_buckets[bucket].append(record)
    metrics["by_token_count_bucket"] = {
        bucket: _compute_candidate_slice_metrics(rows)
        for bucket, rows in length_buckets.items()
    }
    return metrics


def compute_action_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("No action records supplied for evaluation.")

    y_true = [record["gold_action"] for record in records]
    y_pred = [record["predicted_action"] for record in records]
    metrics: dict[str, Any] = _compute_action_slice_metrics(records)
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    context_metrics: dict[str, Any] = {}
    redact_recalls: list[float] = []
    for context, rows in _group_non_null(records, "subject").items():
        slice_metrics = _compute_action_slice_metrics(rows)
        context_recall = slice_metrics["redact_recall"]
        redact_recalls.append(context_recall)
        context_metrics[context] = slice_metrics
    metrics["by_context"] = context_metrics
    metrics["worst_context_redact_recall"] = min(redact_recalls) if redact_recalls else None

    for field in ("speaker_role", "entity_type", "semantic_role"):
        grouped = _group_non_null(records, field)
        if grouped:
            metrics[f"by_{field}"] = {
                value: _compute_action_slice_metrics(rows)
                for value, rows in grouped.items()
            }
    eval_slice_groups = _group_non_null(records, "eval_slice")
    if eval_slice_groups:
        metrics["by_eval_slice"] = {
            value: _compute_action_slice_metrics(rows)
            for value, rows in eval_slice_groups.items()
        }

    gold_keep = [record for record in records if record["gold_action"] == "KEEP"]
    if gold_keep:
        kept_correctly = sum(1 for record in gold_keep if record["predicted_action"] == "KEEP")
        wrong_redactions = sum(1 for record in gold_keep if record["predicted_action"] == "REDACT")
        metrics["cerr"] = kept_correctly / len(gold_keep)
        metrics["orr"] = wrong_redactions / len(gold_keep)
    else:
        metrics["cerr"] = None
        metrics["orr"] = None

    review_count = sum(1 for record in records if record["predicted_action"] == "REVIEW")
    metrics["review_rate"] = review_count / len(records)
    metrics["prediction_label_distribution"] = Counter(y_pred)
    metrics["gold_label_distribution"] = Counter(y_true)

    costs = [float(record["cost"]) for record in records if record.get("cost") is not None]
    if costs:
        metrics["avg_cost"] = mean(costs)

    latencies = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]
    if latencies:
        metrics["avg_latency_ms"] = mean(latencies)

    return metrics
