from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .annotation import bio_spans, preview_tokens, spans_overlap
from .direct_id_rules import detect_direct_id
from .ground_truth_candidate import CANONICAL_RARE_TYPES
from .metrics import compute_candidate_metrics


def merge_candidate_predictions(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
                "subject": row.get("subject", "unknown"),
                "speaker_role": row.get("speaker_role"),
                "dialogue_id": row.get("dialogue_id"),
                "token_count": len(row["labels"]),
                "has_positive_label": any(label != "O" for label in row["labels"]),
                "tokens": [str(token) for token in row["tokens"]],
                "gold_labels": [str(label) for label in row["labels"]],
                "predicted_labels": [str(label) for label in prediction["predicted_labels"]],
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return merged


def compute_candidate_audit_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged_records = list(records)
    base_metrics = compute_candidate_metrics(merged_records)

    action_seed_total = 0
    action_seed_covered = 0
    redact_seed_total = 0
    candidate_redact_covered = 0
    protected_redact_covered = 0
    direct_id_redact_covered = 0

    by_subject_action: dict[str, Counter[str]] = defaultdict(Counter)
    protected_miss_entity_type = Counter()
    protected_miss_speaker_role = Counter()
    protected_miss_span_length = Counter()
    protected_miss_subject = Counter()
    protected_miss_examples: list[dict[str, Any]] = []
    action_seed_miss_examples: list[dict[str, Any]] = []
    gold_span_count = 0
    predicted_span_count = 0
    pii_type_total = Counter()
    pii_type_hit = Counter()
    span_speaker_total = Counter()
    span_speaker_hit = Counter()

    for row in merged_records:
        predicted_spans = bio_spans(row["predicted_labels"])
        gold_spans = bio_spans(row["gold_labels"])
        metadata = row.get("metadata") or {}
        subject = str(row.get("subject", "unknown"))
        speaker_role = str(row.get("speaker_role", "unknown"))
        seed_spans = list(metadata.get("action_seed_spans") or [])
        gold_span_records = list(metadata.get("gold_spans") or [])

        gold_span_count += len(gold_spans)
        predicted_span_count += len(predicted_spans)

        for gold_span_record in gold_span_records:
            gold_span = (int(gold_span_record["token_start"]), int(gold_span_record["token_end"]))
            pii_type = str(gold_span_record.get("label") or "unknown")
            candidate_hit = any(spans_overlap(gold_span, predicted_span) for predicted_span in predicted_spans)
            pii_type_total[pii_type] += 1
            pii_type_hit[pii_type] += int(candidate_hit)
            span_speaker_total[speaker_role] += 1
            span_speaker_hit[speaker_role] += int(candidate_hit)

        for seed in seed_spans:
            action_seed_total += 1
            gold_span = (int(seed["token_start"]), int(seed["token_end"]))
            candidate_hit = any(spans_overlap(gold_span, predicted_span) for predicted_span in predicted_spans)
            if candidate_hit:
                action_seed_covered += 1
                by_subject_action[subject]["covered"] += 1
            else:
                by_subject_action[subject]["missed"] += 1
                action_seed_miss_examples.append(
                    {
                        "id": row["id"],
                        "subject": subject,
                        "speaker_role": speaker_role,
                        "action_label": seed.get("action_label"),
                        "entity_type": seed.get("entity_type"),
                        "span_text": seed.get("span_text"),
                        "preview": preview_tokens(row["tokens"], gold_span),
                    }
                )

            if str(seed.get("action_label")) != "REDACT":
                continue

            redact_seed_total += 1
            if candidate_hit:
                candidate_redact_covered += 1

            direct_id_hit = detect_direct_id(
                str(seed.get("span_text") or ""),
                entity_type=seed.get("entity_type"),
            ) is not None
            if direct_id_hit:
                direct_id_redact_covered += 1

            protected_hit = candidate_hit or direct_id_hit
            if protected_hit:
                protected_redact_covered += 1
            else:
                protected_miss_entity_type[str(seed.get("entity_type") or "unknown")] += 1
                protected_miss_speaker_role[speaker_role] += 1
                protected_miss_span_length[_span_length_bucket(gold_span[1] - gold_span[0])] += 1
                protected_miss_subject[subject] += 1
                protected_miss_examples.append(
                    {
                        "id": row["id"],
                        "subject": subject,
                        "speaker_role": speaker_role,
                        "entity_type": seed.get("entity_type"),
                        "span_text": seed.get("span_text"),
                        "preview": preview_tokens(row["tokens"], gold_span),
                    }
                )

    action_seed_coverage = action_seed_covered / action_seed_total if action_seed_total else None
    candidate_redact_recall = candidate_redact_covered / redact_seed_total if redact_seed_total else None
    protected_redact_recall = protected_redact_covered / redact_seed_total if redact_seed_total else None
    direct_id_redact_recall = direct_id_redact_covered / redact_seed_total if redact_seed_total else None
    positive_row_recall = ((base_metrics.get("positive_rows_only") or {}).get("recall"))
    candidate_volume_multiplier = predicted_span_count / gold_span_count if gold_span_count else None

    by_subject_seed_coverage = {
        subject: {
            "count": counter["covered"] + counter["missed"],
            "action_seed_span_coverage": (
                counter["covered"] / (counter["covered"] + counter["missed"])
                if (counter["covered"] + counter["missed"])
                else None
            ),
        }
        for subject, counter in sorted(by_subject_action.items())
    }
    recall_by_pii_type = {
        pii_type: (pii_type_hit[pii_type] / pii_type_total[pii_type] if pii_type_total[pii_type] else None)
        for pii_type in sorted(pii_type_total)
    }
    recall_by_span_speaker_role = {
        role: (span_speaker_hit[role] / span_speaker_total[role] if span_speaker_total[role] else None)
        for role in sorted(span_speaker_total)
    }
    rare_type_labels = CANONICAL_RARE_TYPES
    rare_type_recalls = [recall_by_pii_type[label] for label in rare_type_labels if recall_by_pii_type.get(label) is not None]
    rare_type_mean_recall = (
        sum(float(value) for value in rare_type_recalls) / len(rare_type_recalls)
        if rare_type_recalls
        else None
    )

    return {
        **base_metrics,
        "action_seed_span_count": action_seed_total,
        "redact_seed_span_count": redact_seed_total,
        "action_seed_span_coverage": action_seed_coverage,
        "candidate_redact_recall": candidate_redact_recall,
        "protected_redact_recall": protected_redact_recall,
        "direct_id_redact_recall": direct_id_redact_recall,
        "positive_row_recall": positive_row_recall,
        "gold_span_count": gold_span_count,
        "predicted_span_count": predicted_span_count,
        "candidate_volume_multiplier": candidate_volume_multiplier,
        "recall_by_pii_type": recall_by_pii_type,
        "recall_by_span_speaker_role": recall_by_span_speaker_role,
        "rare_type_mean_recall": rare_type_mean_recall,
        "by_subject_action_seed_coverage": by_subject_seed_coverage,
        "protected_redact_miss_buckets": {
            "by_entity_type": dict(sorted(protected_miss_entity_type.items(), key=lambda item: (-item[1], item[0]))),
            "by_speaker_role": dict(sorted(protected_miss_speaker_role.items(), key=lambda item: (-item[1], item[0]))),
            "by_span_length_bucket": dict(
                sorted(protected_miss_span_length.items(), key=lambda item: (-item[1], item[0]))
            ),
            "by_subject": dict(sorted(protected_miss_subject.items(), key=lambda item: (-item[1], item[0]))),
        },
        "protected_redact_miss_examples": protected_miss_examples,
        "action_seed_miss_examples": action_seed_miss_examples,
    }


def _span_length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 4:
        return "3-4"
    return "5+"
