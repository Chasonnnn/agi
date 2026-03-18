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

from contextshift_deid.candidate_audit import compute_candidate_audit_metrics, merge_candidate_predictions
from contextshift_deid.data import load_jsonl


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _top_examples(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    return rows[:limit]


def _render_report(
    *,
    gold_file: Path,
    prediction_file: Path,
    metrics: dict[str, Any],
    sample_limit: int,
) -> str:
    lines = [
        "# Candidate Audit Report",
        "",
        f"- gold_file: `{gold_file}`",
        f"- prediction_file: `{prediction_file}`",
        "",
        "## Headline Metrics",
        "",
        f"- candidate_recall: `{_format_metric(metrics.get('recall'))}`",
        f"- candidate_f1: `{_format_metric(metrics.get('f1'))}`",
        f"- positive_row_recall: `{_format_metric(metrics.get('positive_row_recall'))}`",
        f"- worst_context_recall: `{_format_metric(metrics.get('worst_context_recall'))}`",
        f"- gold_span_count: `{metrics.get('gold_span_count')}`",
        f"- predicted_span_count: `{metrics.get('predicted_span_count')}`",
        f"- candidate_volume_multiplier: `{_format_metric(metrics.get('candidate_volume_multiplier'))}`",
        f"- action_seed_span_coverage: `{_format_metric(metrics.get('action_seed_span_coverage'))}`",
        f"- candidate_redact_recall: `{_format_metric(metrics.get('candidate_redact_recall'))}`",
        f"- direct_id_redact_recall: `{_format_metric(metrics.get('direct_id_redact_recall'))}`",
        f"- protected_redact_recall: `{_format_metric(metrics.get('protected_redact_recall'))}`",
        "",
        "## Recall By PII Type",
        "",
        "```json",
        json.dumps(metrics.get("recall_by_pii_type") or {}, indent=2),
        "```",
        "",
        "## Protected Miss Buckets",
        "",
        "```json",
        json.dumps(metrics.get("protected_redact_miss_buckets") or {}, indent=2),
        "```",
        "",
        "## Protected Miss Examples",
        "",
    ]

    protected_examples = _top_examples(metrics.get("protected_redact_miss_examples") or [], limit=sample_limit)
    if protected_examples:
        for example in protected_examples:
            lines.extend(
                [
                    f"- `{example['id']}` `{example.get('subject', 'unknown')}` `{example.get('entity_type', 'unknown')}` `{example.get('span_text', '')}`",
                    f"  - preview: `{example.get('preview', '')}`",
                ]
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Action-Seed Miss Examples",
            "",
        ]
    )
    action_seed_examples = _top_examples(metrics.get("action_seed_miss_examples") or [], limit=sample_limit)
    if action_seed_examples:
        for example in action_seed_examples:
            lines.extend(
                [
                    f"- `{example['id']}` `{example.get('subject', 'unknown')}` `{example.get('action_label', 'unknown')}` `{example.get('span_text', '')}`",
                    f"  - preview: `{example.get('preview', '')}`",
                ]
            )
    else:
        lines.append("- none")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate candidate predictions with UpChieve audit metrics.")
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--report-file", type=Path)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args(argv)

    gold_rows = load_jsonl(args.gold_file)
    prediction_rows = load_jsonl(args.prediction_file)
    merged = merge_candidate_predictions(gold_rows, prediction_rows)
    metrics = compute_candidate_audit_metrics(merged)

    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(
            _render_report(
                gold_file=args.gold_file,
                prediction_file=args.prediction_file,
                metrics=metrics,
                sample_limit=args.sample_limit,
            ),
            encoding="utf-8",
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
