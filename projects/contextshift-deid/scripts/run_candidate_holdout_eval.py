from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import RESULTS_HEADER
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, create_experiment_run, slugify, write_run_metadata

RESULTS_PATH = ROOT / "results.tsv"
DEFAULT_MODEL = ROOT / "runs" / "candidate_math_distilbert_v3"
DEFAULT_BASELINE_PYTHON = ROOT / ".baseline-py311" / "bin" / "python"
DEFAULT_GOLD = ROOT / "data" / "processed" / "candidate" / "test.jsonl"
DEFAULT_INTERIM = ROOT / "data" / "interim" / "upchieve_math_sessions.jsonl"
DEFAULT_RUN_NAME = "candidate_math_holdout"

MATH_TERMS = {
    "add",
    "subtract",
    "solve",
    "graph",
    "equation",
    "slope",
    "intercept",
    "radius",
    "diameter",
    "triangle",
    "fraction",
    "variable",
    "expression",
}
CONTACT_HINTS = ("@", "http", "www", ".com", ".org", "gmail", "email", "discord", "instagram", "snap", "phone")
FORMAT_HINTS = ("\u200e", "\u200f", "\ufeff", "\t", "\n")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "teacher", "tutor"}


def _run(command: list[str], *, capture_json: bool = False) -> dict | None:
    print("+", " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=capture_json, text=capture_json)
    if not capture_json:
        return None
    print(result.stdout)
    return json.loads(result.stdout)


def _default_model_alias(model_path: Path) -> str:
    alias_source = model_path.name
    if alias_source == "model":
        alias_source = model_path.parent.name
    return slugify(alias_source)


def _ensure_results_header() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")


def _append_result(metric: float, status: str, description: str) -> None:
    _ensure_results_header()
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"working\tcandidate\t{metric:.6f}\t{status}\t{description}\n")


def _read_rows(path: Path) -> list[dict]:
    return load_jsonl(path)


def _check_prediction_alignment(gold_rows: list[dict], prediction_rows: list[dict], model_name: str) -> dict:
    gold_ids = [row["id"] for row in gold_rows]
    prediction_ids = [row["id"] for row in prediction_rows]
    if len(gold_ids) != len(prediction_ids):
        raise SystemExit(f"{model_name}: row-count mismatch ({len(gold_ids)} != {len(prediction_ids)})")
    if set(gold_ids) != set(prediction_ids):
        missing = sorted(set(gold_ids) - set(prediction_ids))[:5]
        extra = sorted(set(prediction_ids) - set(gold_ids))[:5]
        raise SystemExit(f"{model_name}: id mismatch; missing={missing} extra={extra}")
    predictions_by_id = {row["id"]: row for row in prediction_rows}
    for row in gold_rows:
        prediction = predictions_by_id[row["id"]]
        if len(row["labels"]) != len(prediction["predicted_labels"]):
            raise SystemExit(
                f"{model_name}: token-length mismatch for {row['id']} "
                f"({len(row['labels'])} != {len(prediction['predicted_labels'])})"
            )
    return {
        "row_count_matches": True,
        "id_set_matches": True,
        "token_length_matches": True,
        "row_count": len(gold_rows),
    }


def _positive_rate(prediction_rows: list[dict]) -> dict[str, float]:
    positive_rows = sum(1 for row in prediction_rows if any(label != "O" for label in row["predicted_labels"]))
    positive_tokens = sum(label != "O" for row in prediction_rows for label in row["predicted_labels"])
    total_tokens = sum(len(row["predicted_labels"]) for row in prediction_rows)
    return {
        "predicted_positive_row_rate": positive_rows / len(prediction_rows) if prediction_rows else 0.0,
        "predicted_positive_token_rate": positive_tokens / total_tokens if total_tokens else 0.0,
    }


def _bio_spans(labels: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for index, label in enumerate(labels):
        if label == "B-SUSPECT":
            if start is not None:
                spans.append((start, index))
            start = index
        elif label != "I-SUSPECT":
            if start is not None:
                spans.append((start, index))
                start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def _span_text(tokens: list[str], span: tuple[int, int]) -> str:
    return " ".join(tokens[span[0] : span[1]])


def _overlaps(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in others)


def _is_math_expression(tokens: list[str], span: tuple[int, int]) -> bool:
    span_tokens = tokens[span[0] : span[1]]
    if not span_tokens:
        return False
    joined = "".join(span_tokens)
    if all(token.isdigit() or token in {"x", "y", "z", "+", "-", "*", "/", "=", "(", ")", ".", ",", "%", "^", "<", ">"} for token in span_tokens):
        return True
    if any(token.lower() in MATH_TERMS for token in span_tokens):
        return True
    return any(character.isdigit() for character in joined) and any(character in "+-*/=^<>" for character in joined)


def _is_name_like(tokens: list[str], span: tuple[int, int]) -> bool:
    span_tokens = [token for token in tokens[span[0] : span[1]] if token.strip()]
    if not span_tokens:
        return False
    normalized = [token.strip(".,!?").lower() for token in span_tokens]
    if normalized[0] in TITLES:
        return True
    alpha_tokens = [token for token in span_tokens if any(char.isalpha() for char in token)]
    if not alpha_tokens:
        return False
    return all(token[:1].isupper() for token in alpha_tokens if token[:1].isalpha())


def _has_formatting_issue(text: str) -> bool:
    return any(hint in text for hint in FORMAT_HINTS)


def _looks_like_contact(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in CONTACT_HINTS)


def _preview_tokens(
    tokens: list[str],
    target_span: tuple[int, int],
    *,
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    window: int = 6,
) -> str:
    preview_start = max(0, target_span[0] - window)
    preview_end = min(len(tokens), target_span[1] + window)
    masked_segments: list[tuple[int, int, str]] = [(target_span[0], target_span[1], "[TARGET_SPAN]")]
    for start, end in gold_spans:
        if (start, end) == target_span or end <= preview_start or start >= preview_end:
            continue
        masked_segments.append((start, end, "[GOLD_SPAN]"))
    for start, end in pred_spans:
        if (start, end) == target_span or end <= preview_start or start >= preview_end:
            continue
        masked_segments.append((start, end, "[PRED_SPAN]"))
    masked_segments.sort(key=lambda item: (item[0], item[1]))

    pieces: list[str] = []
    index = preview_start
    while index < preview_end:
        replacement = None
        for start, end, marker in masked_segments:
            if start == index:
                replacement = (end, marker)
                break
        if replacement is not None:
            end, marker = replacement
            pieces.append(marker)
            index = end
            continue
        pieces.append(tokens[index])
        index += 1
    return " ".join(pieces)


def _classify_bucket(
    *,
    error_type: str,
    target_span: tuple[int, int],
    tokens: list[str],
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    raw_text: str,
) -> str:
    other_spans = gold_spans if error_type == "fp" else pred_spans
    span_text = _span_text(tokens, target_span)
    if _overlaps(target_span, other_spans):
        return "boundary error"
    if _has_formatting_issue(raw_text) or _has_formatting_issue(span_text):
        return "tokenization/formatting issue"
    if error_type == "fn" and _looks_like_contact(span_text or raw_text):
        return "URL/contact miss"
    if error_type == "fp" and _is_math_expression(tokens, target_span):
        return "math-expression false positive"
    if error_type == "fp" and _is_name_like(tokens, target_span):
        return "curricular/person-name false positive"
    return "annotation ambiguity"


def _collect_errors(gold_rows: list[dict], prediction_rows: list[dict], text_lookup: dict[str, str]) -> dict[str, list[dict]]:
    predictions_by_id = {row["id"]: row for row in prediction_rows}
    false_positives: list[dict] = []
    false_negatives: list[dict] = []

    for gold_row in gold_rows:
        prediction = predictions_by_id[gold_row["id"]]
        gold_spans = _bio_spans(gold_row["labels"])
        pred_spans = _bio_spans(prediction["predicted_labels"])
        raw_text = text_lookup.get(gold_row["id"], gold_row.get("context_text", ""))

        for span in pred_spans:
            if span in gold_spans:
                continue
            false_positives.append(
                {
                    "id": gold_row["id"],
                    "dialogue_id": gold_row.get("dialogue_id"),
                    "speaker_role": gold_row.get("speaker_role"),
                    "span_text": _span_text(gold_row["tokens"], span),
                    "bucket": _classify_bucket(
                        error_type="fp",
                        target_span=span,
                        tokens=gold_row["tokens"],
                        gold_spans=gold_spans,
                        pred_spans=pred_spans,
                        raw_text=raw_text,
                    ),
                    "preview": _preview_tokens(
                        gold_row["tokens"],
                        span,
                        gold_spans=gold_spans,
                        pred_spans=pred_spans,
                    ),
                }
            )

        for span in gold_spans:
            if span in pred_spans:
                continue
            false_negatives.append(
                {
                    "id": gold_row["id"],
                    "dialogue_id": gold_row.get("dialogue_id"),
                    "speaker_role": gold_row.get("speaker_role"),
                    "span_text": _span_text(gold_row["tokens"], span),
                    "bucket": _classify_bucket(
                        error_type="fn",
                        target_span=span,
                        tokens=gold_row["tokens"],
                        gold_spans=gold_spans,
                        pred_spans=pred_spans,
                        raw_text=raw_text,
                    ),
                    "preview": _preview_tokens(
                        gold_row["tokens"],
                        span,
                        gold_spans=gold_spans,
                        pred_spans=pred_spans,
                    ),
                }
            )

    return {"false_positives": false_positives, "false_negatives": false_negatives}


def _sample_errors(errors: list[dict], limit: int) -> list[dict]:
    return sorted(errors, key=lambda item: (len(item["preview"]), item["id"], item["span_text"]))[:limit]


def _bucket_counts(errors: list[dict]) -> dict[str, int]:
    counts = Counter(error["bucket"] for error in errors)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _top_failure_patterns(false_positives: list[dict], false_negatives: list[dict], limit: int = 2) -> list[dict]:
    counts = Counter(error["bucket"] for error in false_positives + false_negatives)
    return [{"bucket": bucket, "count": count} for bucket, count in counts.most_common(limit)]


def _build_text_lookup(interim_rows: list[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for session in interim_rows:
        session_id = str(session["session_id"])
        for turn in session["turns"]:
            lookup[f"{session_id}-turn-{turn['sequence_id']}"] = str(turn["content"])
    return lookup


def _write_report(summary: dict, markdown_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Math Candidate Holdout Report",
        "",
        "Strict holdout comparison on the imported math `test` split using the frozen local candidate detector and untuned spaCy/Presidio baselines.",
        "",
        "## Comparison",
        "",
        "| model | precision | recall | f1 | worst_context_recall | predicted_positive_row_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, model_summary in summary["models"].items():
        metrics = model_summary["metrics"]
        lines.append(
            f"| {model_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['worst_context_recall']:.4f} | "
            f"{model_summary['positive_rates']['predicted_positive_row_rate']:.4f} |"
        )

    for model_name, model_summary in summary["models"].items():
        top_failure_text = ", ".join(
            f"{item['bucket']} ({item['count']})" for item in model_summary["top_failure_patterns"]
        ) or "none"
        lines.extend(
            [
                "",
                f"## {model_name}",
                "",
                f"- Top failure patterns: {top_failure_text}",
                f"- False-positive buckets: {json.dumps(model_summary['false_positive_bucket_counts'], ensure_ascii=False)}",
                f"- False-negative buckets: {json.dumps(model_summary['false_negative_bucket_counts'], ensure_ascii=False)}",
                "",
                "### Sample False Positives",
                "",
            ]
        )
        for sample in model_summary["sample_false_positives"]:
            lines.append(
                f"- `{sample['id']}` [{sample['bucket']}] `{sample['preview']}`"
            )
        lines.extend(["", "### Sample False Negatives", ""])
        for sample in model_summary["sample_false_negatives"]:
            lines.append(
                f"- `{sample['id']}` [{sample['bucket']}] `{sample['preview']}`"
            )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict holdout candidate evaluation on the math test split.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-alias", help="Short label used for the frozen local model in filenames and reports.")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--interim-file", type=Path, default=DEFAULT_INTERIM)
    parser.add_argument("--baseline-python", type=Path, default=DEFAULT_BASELINE_PYTHON)
    parser.add_argument("--run-root", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Missing frozen candidate checkpoint: {args.model}")
    if not args.baseline_python.exists():
        raise SystemExit(f"Missing baseline sidecar Python: {args.baseline_python}")

    started_at = datetime.now().astimezone()
    experiment = create_experiment_run(args.run_name, root_dir=args.run_root)
    model_alias = args.model_alias or _default_model_alias(args.model)
    local_predictions = experiment.predictions_dir / f"candidate_test_{model_alias}.jsonl"
    spacy_predictions = experiment.predictions_dir / "candidate_test_spacy.jsonl"
    presidio_predictions = experiment.predictions_dir / "candidate_test_presidio.jsonl"

    _run(
        [
            sys.executable,
            "scripts/predict_candidate.py",
            "--model",
            str(args.model),
            "--input-file",
            str(args.gold),
            "--output-file",
            str(local_predictions),
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
        ]
    )

    _run(
        [
            str(args.baseline_python),
            "scripts/run_candidate_baselines.py",
            "--split-file",
            str(args.gold),
            "--interim-file",
            str(args.interim_file),
            "--spacy-output",
            str(spacy_predictions),
            "--presidio-output",
            str(presidio_predictions),
        ]
    )

    gold_rows = _read_rows(args.gold)
    interim_rows = _read_rows(args.interim_file)
    text_lookup = _build_text_lookup(interim_rows)

    model_specs = {
        model_alias: {
            "prediction_path": local_predictions,
            "description": f"strict holdout test using frozen {model_alias} on imported math candidate test",
        },
        "spacy_en_core_web_sm": {
            "prediction_path": spacy_predictions,
            "description": "strict holdout test using spaCy en_core_web_sm candidate baseline on imported math test",
        },
        "presidio_default": {
            "prediction_path": presidio_predictions,
            "description": "strict holdout test using Presidio default analyzer candidate baseline on imported math test",
        },
    }

    summary = {
        "gold_file": str(args.gold),
        "interim_file": str(args.interim_file),
        "frozen_model": str(args.model),
        "frozen_model_alias": model_alias,
        "run_dir": str(experiment.root),
        "started_at": started_at.isoformat(),
        "models": {},
    }

    pending_results: list[tuple[float, str, str]] = []
    for model_name, spec in model_specs.items():
        prediction_rows = _read_rows(spec["prediction_path"])
        sanity = _check_prediction_alignment(gold_rows, prediction_rows, model_name)
        metrics = _run(
            [
                sys.executable,
                "eval.py",
                "--stage",
                "candidate",
                "--gold",
                str(args.gold),
                "--predictions",
                str(spec["prediction_path"]),
            ],
            capture_json=True,
        )
        errors = _collect_errors(gold_rows, prediction_rows, text_lookup)
        positive_rates = _positive_rate(prediction_rows)
        model_summary = {
            "sanity": sanity,
            "metrics": metrics,
            "positive_rates": positive_rates,
            "false_positive_bucket_counts": _bucket_counts(errors["false_positives"]),
            "false_negative_bucket_counts": _bucket_counts(errors["false_negatives"]),
            "top_failure_patterns": _top_failure_patterns(errors["false_positives"], errors["false_negatives"]),
            "sample_false_positives": _sample_errors(errors["false_positives"], args.sample_limit),
            "sample_false_negatives": _sample_errors(errors["false_negatives"], args.sample_limit),
            "prediction_path": str(spec["prediction_path"]),
        }
        summary["models"][model_name] = model_summary
        pending_results.append((float(metrics["f1"]), "keep", f"{spec['description']} [{experiment.root.name}]"))

    summary["completed_at"] = datetime.now().astimezone().isoformat()
    _write_report(summary, experiment.report_path, experiment.summary_path)
    write_run_metadata(
        experiment.metadata_path,
        {
            "run_name": args.run_name,
            "run_dir": str(experiment.root),
            "started_at": started_at.isoformat(),
            "completed_at": summary["completed_at"],
            "model": str(args.model),
            "model_alias": model_alias,
            "gold": str(args.gold),
            "interim_file": str(args.interim_file),
            "baseline_python": str(args.baseline_python),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "sample_limit": args.sample_limit,
            "artifacts": {
                "report": str(experiment.report_path),
                "summary": str(experiment.summary_path),
                "predictions_dir": str(experiment.predictions_dir),
            },
        },
    )
    for metric, status, description in pending_results:
        _append_result(metric, status, description)

    print(
        json.dumps(
            {
                "run_dir": str(experiment.root),
                "summary_json": str(experiment.summary_path),
                "report_md": str(experiment.report_path),
                "metadata_json": str(experiment.metadata_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
