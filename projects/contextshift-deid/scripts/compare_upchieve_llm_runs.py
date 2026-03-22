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

from contextshift_deid.constants import LEGACY_EXPERIMENTS_DIR
from contextshift_deid.experiment_runs import create_experiment_run, write_run_metadata

DEFAULT_BASELINE_SUMMARY_FILE = (
    LEGACY_EXPERIMENTS_DIR / "20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384" / "summary.json"
)
DEFAULT_SELECTED_REVIEW_RATE = 0.10
DEFAULT_RUN_NAME = "upchieve-llm-compare"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_target_entry(summary: dict[str, Any], *, target_review_rate: float) -> dict[str, Any]:
    for target in summary.get("selected_targets", []):
        if abs(float(target["target_review_rate"]) - target_review_rate) <= 1e-9:
            return dict(target)
    raise SystemExit(f"Baseline summary is missing a selected target for review rate {target_review_rate:.2f}")


def _parse_summary_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise SystemExit(f"Invalid --summary spec {raw!r}; expected label=/absolute/or/relative/path.json")
    label, raw_path = raw.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip())
    if not label:
        raise SystemExit(f"Invalid --summary spec {raw!r}; label must not be empty")
    return label, path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare multiple UPChieve LLM/local-open summary artifacts against the frozen ModernBERT v2 dev baseline.")
    parser.add_argument("--summary", action="append", required=True, help="Repeated label=summary.json entry")
    parser.add_argument("--baseline-summary-file", type=Path, default=DEFAULT_BASELINE_SUMMARY_FILE)
    parser.add_argument("--selected-review-rate", type=float, default=DEFAULT_SELECTED_REVIEW_RATE)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args(argv)

    baseline_summary = _load_json(args.baseline_summary_file)
    selected_target = _selected_target_entry(
        baseline_summary,
        target_review_rate=args.selected_review_rate,
    )
    selected_dev_metrics = selected_target["calibration"]["metrics"]
    selected_dev_macro_f1 = float(selected_dev_metrics["macro_f1"])

    rows: list[dict[str, Any]] = []
    summary_files: list[str] = []
    for raw_spec in args.summary:
        label, summary_path = _parse_summary_spec(raw_spec)
        payload = _load_json(summary_path)
        summary_files.append(str(summary_path))
        metrics = payload["llm_metrics"]
        usage_and_latency = payload.get("usage_and_latency", {})
        usage_totals = usage_and_latency.get("usage_totals", {})
        latency = usage_and_latency.get("latency_ms", {})
        rows.append(
            {
                "label": label,
                "summary_file": str(summary_path),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "redact_recall": float(metrics["redact_recall"]),
                "review_rate": float(metrics["review_rate"]),
                "gap_vs_selected_dev": float(metrics["macro_f1"]) - selected_dev_macro_f1,
                "thinking_mode": payload.get("thinking_mode"),
                "reasoning_effort": payload.get("reasoning_effort"),
                "repair_request_count": int(payload.get("repair_request_count") or 0),
                "latency_avg_ms": latency.get("avg"),
                "reasoning_tokens": int(usage_totals.get("reasoning_tokens") or 0),
            }
        )

    best_row = max(rows, key=lambda row: row["macro_f1"])
    train_v3_recommended = best_row["gap_vs_selected_dev"] >= 0.10

    experiment = create_experiment_run(args.run_name)
    metadata = {
        "baseline_summary_file": str(args.baseline_summary_file),
        "selected_review_rate": args.selected_review_rate,
        "selected_dev_baseline_macro_f1": selected_dev_macro_f1,
        "summary_files": summary_files,
    }
    write_run_metadata(experiment.metadata_path, metadata)

    summary_payload = {
        "baseline_summary_file": str(args.baseline_summary_file),
        "selected_review_rate": args.selected_review_rate,
        "selected_dev_baseline_macro_f1": selected_dev_macro_f1,
        "rows": rows,
        "best_model": best_row,
        "train_v3_recommended": train_v3_recommended,
        "writeup_recommended": not train_v3_recommended,
    }
    experiment.summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    lines = [
        "# UPChieve LLM Comparison",
        "",
        f"- ModernBERT v2 selected {args.selected_review_rate * 100:.1f}% dev baseline macro F1: {selected_dev_macro_f1:.4f}",
        "",
        "## Model Comparison",
        "",
        "| model | accuracy | macro_f1 | gap_vs_selected_dev | redact_recall | review_rate | thinking_mode | reasoning_effort | repair_requests | reasoning_tokens | avg_latency_ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        latency_text = "n/a" if row["latency_avg_ms"] is None else f"{float(row['latency_avg_ms']):.1f}"
        lines.append(
            "| {label} | {accuracy:.4f} | {macro_f1:.4f} | {gap:+.4f} | {redact_recall:.4f} | {review_rate:.1f}% | {thinking_mode} | {reasoning_effort} | {repair_requests} | {reasoning_tokens} | {latency} |".format(
                label=row["label"],
                accuracy=row["accuracy"],
                macro_f1=row["macro_f1"],
                gap=row["gap_vs_selected_dev"],
                redact_recall=row["redact_recall"],
                review_rate=row["review_rate"] * 100,
                thinking_mode=row["thinking_mode"] or "n/a",
                reasoning_effort=row["reasoning_effort"] or "n/a",
                repair_requests=row["repair_request_count"],
                reasoning_tokens=row["reasoning_tokens"],
                latency=latency_text,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Move to writeup." if not train_v3_recommended else "Train_v3 cleared the comparison gate.",
            "",
            "## Why",
            "",
            f"- Best model in this comparison: {best_row['label']} at {best_row['macro_f1']:.4f} macro F1 ({best_row['gap_vs_selected_dev']:+.4f} vs selected-dev baseline).",
            "- The comparison rule is unchanged: only treat a follow-on training branch as justified if the best LLM/open-model comparison clears the local selected-dev baseline by at least +0.10 macro F1.",
            "",
        ]
    )
    experiment.report_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "experiment_root": str(experiment.root),
                "summary_path": str(experiment.summary_path),
                "report_path": str(experiment.report_path),
                "best_model": best_row["label"],
                "best_macro_f1": best_row["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
