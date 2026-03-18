from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.candidate_adaptation import char_span_to_token_span, labels_from_token_spans
from contextshift_deid.candidate_audit import compute_candidate_audit_metrics, merge_candidate_predictions
from contextshift_deid.constants import CANDIDATE_DIR, RESULTS_HEADER
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, create_experiment_run, slugify, write_run_metadata
from contextshift_deid.ground_truth_candidate import UPCHIEVE_RARE_TYPES
from contextshift_deid.tokenization import tokenize_with_offsets

RESULTS_PATH = ROOT / "results.tsv"
DEFAULT_RUN_ROOT = EXPERIMENTS_DIR / "candidate_model_selection"
DEFAULT_VOLUME_CAP = 3.0
GLINER_ENTITY_LABELS = [
    "person name",
    "location",
    "school",
    "grade level",
    "course name",
    "age",
    "date",
    "phone number",
    "email address",
    "website or url",
    "ip address",
    "user or account identifier",
]
GLINER_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    return env


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True, env=_command_env())


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _metric_value(value: Any, *, default: float = -1.0) -> float:
    if value is None:
        return default
    return float(value)


def _append_result(metric: float, *, status: str, description: str) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"working\tcandidate\t{metric:.6f}\t{status}\t{description}\n")


def _evaluate_audit(gold_file: Path, prediction_file: Path) -> dict[str, Any]:
    merged = merge_candidate_predictions(load_jsonl(gold_file), load_jsonl(prediction_file))
    return compute_candidate_audit_metrics(merged)


def _render_audit_report(name: str, metrics: Mapping[str, Any], *, gold_file: Path, prediction_file: Path) -> str:
    return "\n".join(
        [
            f"# {name}",
            "",
            f"- gold_file: `{gold_file}`",
            f"- prediction_file: `{prediction_file}`",
            "",
            "## Headline",
            "",
            f"- recall: `{_fmt(metrics.get('recall'))}`",
            f"- f1: `{_fmt(metrics.get('f1'))}`",
            f"- positive_row_recall: `{_fmt(metrics.get('positive_row_recall'))}`",
            f"- worst_context_recall: `{_fmt(metrics.get('worst_context_recall'))}`",
            f"- gold_span_count: `{metrics.get('gold_span_count')}`",
            f"- predicted_span_count: `{metrics.get('predicted_span_count')}`",
            f"- candidate_volume_multiplier: `{_fmt(metrics.get('candidate_volume_multiplier'))}`",
            f"- rare_type_mean_recall: `{_fmt(metrics.get('rare_type_mean_recall'))}`",
            "",
            "## Recall By PII Type",
            "",
            "```json",
            json.dumps(metrics.get("recall_by_pii_type") or {}, indent=2),
            "```",
            "",
        ]
    ) + "\n"


def _predict_candidate(
    *,
    model_path: Path | str,
    input_file: Path,
    output_file: Path,
    batch_size: int,
    max_length: int,
    context_mode: str,
) -> None:
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "predict_candidate.py"),
            "--model",
            str(model_path),
            "--input-file",
            str(input_file),
            "--output-file",
            str(output_file),
            "--batch-size",
            str(batch_size),
            "--max-length",
            str(max_length),
            "--context-mode",
            context_mode,
        ]
    )


def _train_candidate(
    *,
    model: Path | str,
    train_file: Path,
    dev_file: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    max_length: int,
    context_mode: str,
) -> None:
    _run(
        [
            sys.executable,
            str(ROOT / "train_candidate.py"),
            "--model",
            str(model),
            "--train-file",
            str(train_file),
            "--dev-file",
            str(dev_file),
            "--output-dir",
            str(output_dir),
            "--epochs",
            str(epochs),
            "--lr",
            str(lr),
            "--batch-size",
            str(batch_size),
            "--max-length",
            str(max_length),
            "--context-mode",
            context_mode,
            "--selection-metric",
            "recall",
            "--prediction-file",
            str(output_dir / "dev_predictions.jsonl"),
        ]
    )


def _evaluate_prediction_run(
    *,
    label: str,
    gold_file: Path,
    prediction_file: Path,
    summary_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    metrics = _evaluate_audit(gold_file, prediction_file)
    _write_json(summary_path, metrics)
    report_path.write_text(
        _render_audit_report(label, metrics, gold_file=gold_file, prediction_file=prediction_file),
        encoding="utf-8",
    )
    return metrics


def _alias_from_model(model_value: str | Path) -> str:
    candidate = Path(model_value)
    if candidate.exists():
        name = candidate.name
        if name == "model":
            name = candidate.parent.name
        return slugify(name)
    return slugify(str(model_value))


def _make_dropin_result(
    *,
    family: str,
    label: str,
    model_value: str | Path,
    trained_model_path: Path,
    config: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
    volume_cap: float,
) -> dict[str, Any]:
    dev_metrics = evaluations["upchieve_dev"]
    return {
        "family": family,
        "label": label,
        "source_model": str(model_value),
        "trained_model_path": str(trained_model_path),
        "config": dict(config),
        "evaluations": {name: dict(metrics) for name, metrics in evaluations.items()},
        "eligible_on_dev": _metric_value(dev_metrics.get("candidate_volume_multiplier"), default=1e9) <= volume_cap,
        "dev_recall": _metric_value(dev_metrics.get("recall")),
        "dev_volume_multiplier": _metric_value(dev_metrics.get("candidate_volume_multiplier"), default=1e9),
        "math_recall": _metric_value(evaluations["math_test"].get("recall")),
    }


def _dev_rank_key(entry: Mapping[str, Any]) -> tuple[int, float, float, float]:
    return (
        int(bool(entry.get("eligible_on_dev"))),
        _metric_value(entry.get("dev_recall")),
        -_metric_value(entry.get("dev_volume_multiplier"), default=1e9),
        _metric_value(((entry.get("evaluations") or {}).get("upchieve_dev") or {}).get("f1")),
    )


def _choose_top_entries(entries: Sequence[Mapping[str, Any]], *, count: int) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=_dev_rank_key, reverse=True)
    return [dict(entry) for entry in ordered[:count]]


def _heldout_rare_type_mean(entry: Mapping[str, Any]) -> float:
    evaluations = entry.get("evaluations") or {}
    values = [
        _metric_value((evaluations.get("upchieve_test") or {}).get("rare_type_mean_recall"), default=-1.0),
        _metric_value((evaluations.get("saga27_test") or {}).get("rare_type_mean_recall"), default=-1.0),
    ]
    values = [value for value in values if value >= 0.0]
    if not values:
        return -1.0
    return float(statistics.mean(values))


def _final_rank_key(entry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    evaluations = entry.get("evaluations") or {}
    upchieve_test = evaluations.get("upchieve_test") or {}
    saga_test = evaluations.get("saga27_test") or {}
    return (
        min(
            _metric_value(upchieve_test.get("recall")),
            _metric_value(saga_test.get("recall")),
        ),
        _heldout_rare_type_mean(entry),
        -_metric_value((evaluations.get("upchieve_dev") or {}).get("candidate_volume_multiplier"), default=1e9),
        _metric_value(upchieve_test.get("f1")),
    )


def _upchieve_files(output_dir: Path) -> dict[str, Path]:
    return {
        "train": output_dir / "upchieve_math_ground_truth_train.jsonl",
        "dev": output_dir / "upchieve_math_ground_truth_dev.jsonl",
        "test": output_dir / "upchieve_math_ground_truth_test.jsonl",
    }


def _run_build_benchmark(
    *,
    upchieve_raw_file: Path,
    saga_raw_dir: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_ground_truth_candidate_benchmark.py"),
            "--upchieve-raw-file",
            str(upchieve_raw_file),
            "--saga-raw-dir",
            str(saga_raw_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
        ]
    )
    summary_path = output_dir / "ground_truth_candidate_benchmark_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _evaluate_hf_model(
    *,
    label: str,
    model_path: Path,
    datasets: Mapping[str, Path],
    predictions_dir: Path,
    batch_size: int,
    max_length: int,
    context_mode: str,
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for split_name, gold_file in datasets.items():
        prediction_file = predictions_dir / f"{slugify(label)}_{split_name}_predictions.jsonl"
        _predict_candidate(
            model_path=model_path,
            input_file=gold_file,
            output_file=prediction_file,
            batch_size=batch_size,
            max_length=max_length,
            context_mode=context_mode,
        )
        evaluations[split_name] = _evaluate_prediction_run(
            label=f"{label} {split_name}",
            gold_file=gold_file,
            prediction_file=prediction_file,
            summary_path=prediction_file.with_suffix(".summary.json"),
            report_path=prediction_file.with_suffix(".report.md"),
        )
    return evaluations


def _predict_gliner_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    model_name: str,
) -> list[dict[str, Any]]:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError("GLiNER is not installed") from exc

    model = GLiNER.from_pretrained(model_name)
    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        raw_text = str((row.get("metadata") or {}).get("raw_text") or " ".join(row["tokens"]))
        token_offsets = [span for _, span in tokenize_with_offsets(raw_text)]
        if len(token_offsets) != len(row["tokens"]):
            raise RuntimeError(f"Raw text/token mismatch for {row['id']}")
        entities = model.predict_entities(raw_text, GLINER_ENTITY_LABELS, threshold=threshold)
        token_spans = set()
        for entity in entities:
            token_span = char_span_to_token_span(token_offsets, int(entity["start"]), int(entity["end"]))
            if token_span is not None:
                token_spans.add(token_span)
        prediction_rows.append(
            {
                "id": row["id"],
                "predicted_labels": labels_from_token_spans(len(row["tokens"]), sorted(token_spans)),
            }
        )
    return prediction_rows


def _run_gliner_selection(
    *,
    model_name: str,
    datasets: Mapping[str, Path],
    output_dir: Path,
    volume_cap: float,
) -> dict[str, Any]:
    dev_rows = load_jsonl(datasets["upchieve_dev"])
    dev_runs: list[dict[str, Any]] = []
    for threshold in GLINER_THRESHOLDS:
        threshold_dir = output_dir / f"threshold-{str(threshold).replace('.', '-')}"
        prediction_rows = _predict_gliner_rows(rows=dev_rows, threshold=threshold, model_name=model_name)
        prediction_file = threshold_dir / "upchieve_dev_predictions.jsonl"
        _write_jsonl(prediction_file, prediction_rows)
        metrics = _evaluate_prediction_run(
            label=f"GLiNER threshold {threshold}",
            gold_file=datasets["upchieve_dev"],
            prediction_file=prediction_file,
            summary_path=prediction_file.with_suffix(".summary.json"),
            report_path=prediction_file.with_suffix(".report.md"),
        )
        dev_runs.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "eligible": _metric_value(metrics.get("candidate_volume_multiplier"), default=1e9) <= volume_cap,
            }
        )

    best_dev_run = sorted(
        dev_runs,
        key=lambda entry: (
            int(bool(entry["eligible"])),
            _metric_value(entry["metrics"].get("recall")),
            -_metric_value(entry["metrics"].get("candidate_volume_multiplier"), default=1e9),
            _metric_value(entry["metrics"].get("f1")),
        ),
        reverse=True,
    )[0]

    result = {
        "family": "gliner",
        "label": "gliner-medium-v2-1",
        "source_model": model_name,
        "trained_model_path": None,
        "config": {"threshold": best_dev_run["threshold"]},
        "evaluations": {"upchieve_dev": best_dev_run["metrics"]},
        "eligible_on_dev": bool(best_dev_run["eligible"]),
        "dev_recall": _metric_value(best_dev_run["metrics"].get("recall")),
        "dev_volume_multiplier": _metric_value(best_dev_run["metrics"].get("candidate_volume_multiplier"), default=1e9),
        "math_recall": -1.0,
    }
    return result


def _evaluate_gliner_finalist(
    *,
    model_name: str,
    threshold: float,
    datasets: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for split_name, gold_file in datasets.items():
        rows = load_jsonl(gold_file)
        prediction_rows = _predict_gliner_rows(rows=rows, threshold=threshold, model_name=model_name)
        prediction_file = output_dir / f"gliner_{split_name}_predictions.jsonl"
        _write_jsonl(prediction_file, prediction_rows)
        evaluations[split_name] = _evaluate_prediction_run(
            label=f"GLiNER {split_name}",
            gold_file=gold_file,
            prediction_file=prediction_file,
            summary_path=prediction_file.with_suffix(".summary.json"),
            report_path=prediction_file.with_suffix(".report.md"),
        )
    return evaluations


def _spanmarker_datasets(rows: Sequence[Mapping[str, Any]]):
    try:
        from datasets import ClassLabel, Dataset, Features, Sequence as DatasetSequence, Value
    except ImportError as exc:
        raise RuntimeError("datasets is required for SpanMarker") from exc

    label_names = ["O", "B-SUSPECT", "I-SUSPECT"]
    features = Features(
        {
            "tokens": DatasetSequence(Value("string")),
            "ner_tags": DatasetSequence(ClassLabel(names=label_names)),
        }
    )
    data = [
        {
            "tokens": list(row["tokens"]),
            "ner_tags": [label_names.index(label) for label in row["labels"]],
        }
        for row in rows
    ]
    return Dataset.from_list(data).cast(features)


def _train_spanmarker(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    max_length: int,
) -> Path:
    try:
        from span_marker import SpanMarkerModel, Trainer as SpanMarkerTrainer
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("SpanMarker is not installed") from exc

    train_dataset = _spanmarker_datasets(train_rows)
    dev_dataset = _spanmarker_datasets(dev_rows)
    model = SpanMarkerModel.from_pretrained(
        "roberta-base",
        labels=["SUSPECT"],
        model_max_length=max_length,
        entity_max_length=8,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=1e-5,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_recall",
        greater_is_better=True,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to=[],
    )
    trainer = SpanMarkerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    return output_dir


def _predict_spanmarker_rows(
    *,
    model_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    try:
        from span_marker import SpanMarkerModel
    except ImportError as exc:
        raise RuntimeError("SpanMarker is not installed") from exc

    model = SpanMarkerModel.from_pretrained(str(model_path))
    entities_per_row = model.predict([list(row["tokens"]) for row in rows], batch_size=4)
    prediction_rows: list[dict[str, Any]] = []
    for row, entities in zip(rows, entities_per_row):
        token_spans = set()
        for entity in entities:
            token_spans.add((int(entity["word_start_index"]), int(entity["word_end_index"])))
        prediction_rows.append(
            {
                "id": row["id"],
                "predicted_labels": labels_from_token_spans(len(row["tokens"]), sorted(token_spans)),
            }
        )
    return prediction_rows


def _evaluate_spanmarker_model(
    *,
    model_path: Path,
    datasets: Mapping[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for split_name, gold_file in datasets.items():
        rows = load_jsonl(gold_file)
        prediction_rows = _predict_spanmarker_rows(model_path=model_path, rows=rows)
        prediction_file = output_dir / f"spanmarker_{split_name}_predictions.jsonl"
        _write_jsonl(prediction_file, prediction_rows)
        evaluations[split_name] = _evaluate_prediction_run(
            label=f"SpanMarker {split_name}",
            gold_file=gold_file,
            prediction_file=prediction_file,
            summary_path=prediction_file.with_suffix(".summary.json"),
            report_path=prediction_file.with_suffix(".report.md"),
        )
    return evaluations


def _report_comparison_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| label | family | dev recall | dev volume | eligible | math recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=_dev_rank_key, reverse=True):
        lines.append(
            "| {label} | {family} | {dev_recall} | {dev_volume} | {eligible} | {math_recall} |".format(
                label=row["label"],
                family=row["family"],
                dev_recall=_fmt(row.get("dev_recall")),
                dev_volume=_fmt(row.get("dev_volume_multiplier")),
                eligible="yes" if row.get("eligible_on_dev") else "no",
                math_recall=_fmt((row.get("evaluations") or {}).get("math_test", {}).get("recall")),
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run candidate backbone selection on ground-truth math corpora.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="candidate-model-selection")
    parser.add_argument(
        "--upchieve-raw-file",
        type=Path,
        default=Path("/Users/chason/Downloads/DeID_GT_UPchieve_math_1000transcripts.jsonl"),
    )
    parser.add_argument(
        "--saga-raw-dir",
        type=Path,
        default=Path("/Users/chason/Downloads/DeID_GT_Saga_math_27_transcripts"),
    )
    parser.add_argument("--candidate-output-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--incumbent-model", type=Path, default=ROOT / "runs" / "candidate_math_distilbert_rebuilt")
    parser.add_argument("--math-test-file", type=Path, default=CANDIDATE_DIR / "test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--volume-cap", type=float, default=DEFAULT_VOLUME_CAP)
    parser.add_argument("--skip-external-models", action="store_true")
    args = parser.parse_args(argv)

    experiment = create_experiment_run(args.run_name, root_dir=args.run_root)
    benchmark_summary = _run_build_benchmark(
        upchieve_raw_file=args.upchieve_raw_file,
        saga_raw_dir=args.saga_raw_dir,
        output_dir=args.candidate_output_dir,
        seed=args.seed,
    )
    upchieve_files = _upchieve_files(args.candidate_output_dir)
    saga_file = args.candidate_output_dir / "saga27_math_ground_truth_test.jsonl"
    benchmark_datasets = {
        "upchieve_dev": upchieve_files["dev"],
        "upchieve_test": upchieve_files["test"],
        "saga27_test": saga_file,
        "math_test": args.math_test_file,
    }

    incumbent_dir = experiment.root / "incumbent"
    incumbent_dir.mkdir(parents=True, exist_ok=True)
    incumbent_evaluations = _evaluate_hf_model(
        label="incumbent-distilbert-rebuilt",
        model_path=args.incumbent_model,
        datasets=benchmark_datasets,
        predictions_dir=incumbent_dir,
        batch_size=16,
        max_length=384,
        context_mode="none",
    )
    incumbent_result = _make_dropin_result(
        family="drop_in",
        label="incumbent-distilbert-rebuilt",
        model_value=args.incumbent_model,
        trained_model_path=args.incumbent_model,
        config={"lr": None, "epochs": None, "max_length": 384, "context_mode": "none", "batch_size": 16},
        evaluations=incumbent_evaluations,
        volume_cap=args.volume_cap,
    )

    scout_model_specs = [
        {"label": "distilbert-from-incumbent", "model": args.incumbent_model, "batch_size": 8},
        {"label": "modernbert-base", "model": "answerdotai/ModernBERT-base", "batch_size": 4},
        {"label": "deberta-v3-base", "model": "microsoft/deberta-v3-base", "batch_size": 8},
        {"label": "xlm-roberta-base", "model": "xlm-roberta-base", "batch_size": 8},
    ]
    training_root = experiment.root / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    scout_results: list[dict[str, Any]] = []
    trained_entries: dict[tuple[str, float, int, str], dict[str, Any]] = {}

    for spec in scout_model_specs:
        config = {
            "lr": 2e-5,
            "epochs": 3,
            "max_length": 384,
            "context_mode": "none",
            "batch_size": spec["batch_size"],
        }
        key = (spec["label"], config["lr"], config["max_length"], config["context_mode"])
        output_dir = training_root / spec["label"] / "scout"
        _train_candidate(
            model=spec["model"],
            train_file=upchieve_files["train"],
            dev_file=upchieve_files["dev"],
            output_dir=output_dir,
            epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            max_length=config["max_length"],
            context_mode=config["context_mode"],
        )
        evaluations = _evaluate_hf_model(
            label=spec["label"],
            model_path=output_dir,
            datasets={"upchieve_dev": upchieve_files["dev"], "math_test": args.math_test_file},
            predictions_dir=output_dir / "predictions",
            batch_size=16,
            max_length=config["max_length"],
            context_mode=config["context_mode"],
        )
        result = _make_dropin_result(
            family="drop_in",
            label=spec["label"],
            model_value=spec["model"],
            trained_model_path=output_dir,
            config=config,
            evaluations=evaluations,
            volume_cap=args.volume_cap,
        )
        scout_results.append(result)
        trained_entries[key] = result

    top_dropins = _choose_top_entries(scout_results, count=2)
    tuned_results: list[dict[str, Any]] = list(scout_results)
    for entry in top_dropins:
        model_label = str(entry["label"])
        source_model = str(entry["source_model"])
        model_arg: str | Path = Path(source_model) if Path(source_model).exists() else source_model
        batch_size = int((entry.get("config") or {}).get("batch_size") or 8)
        for lr in (2e-5, 3e-5):
            for max_length in (256, 384):
                for context_mode in ("none", "pair"):
                    key = (model_label, lr, max_length, context_mode)
                    if key in trained_entries:
                        continue
                    output_dir = training_root / model_label / f"lr-{lr}-len-{max_length}-ctx-{context_mode}"
                    _train_candidate(
                        model=model_arg,
                        train_file=upchieve_files["train"],
                        dev_file=upchieve_files["dev"],
                        output_dir=output_dir,
                        epochs=3,
                        lr=lr,
                        batch_size=batch_size,
                        max_length=max_length,
                        context_mode=context_mode,
                    )
                    evaluations = _evaluate_hf_model(
                        label=f"{model_label}-lr-{lr}-len-{max_length}-ctx-{context_mode}",
                        model_path=output_dir,
                        datasets={"upchieve_dev": upchieve_files["dev"], "math_test": args.math_test_file},
                        predictions_dir=output_dir / "predictions",
                        batch_size=16,
                        max_length=max_length,
                        context_mode=context_mode,
                    )
                    result = _make_dropin_result(
                        family="drop_in",
                        label=f"{model_label}-lr-{lr}-len-{max_length}-ctx-{context_mode}",
                        model_value=model_arg,
                        trained_model_path=output_dir,
                        config={
                            "lr": lr,
                            "epochs": 3,
                            "max_length": max_length,
                            "context_mode": context_mode,
                            "batch_size": batch_size,
                        },
                        evaluations=evaluations,
                        volume_cap=args.volume_cap,
                    )
                    tuned_results.append(result)
                    trained_entries[key] = result

    all_candidates: list[dict[str, Any]] = list(tuned_results)
    if not args.skip_external_models:
        external_root = experiment.root / "external"
        external_root.mkdir(parents=True, exist_ok=True)
        try:
            gliner_result = _run_gliner_selection(
                model_name="urchade/gliner_medium-v2.1",
                datasets={"upchieve_dev": upchieve_files["dev"]},
                output_dir=external_root / "gliner",
                volume_cap=args.volume_cap,
            )
            all_candidates.append(gliner_result)
        except Exception as exc:  # pragma: no cover - dependency/runtime variability
            all_candidates.append(
                {
                    "family": "gliner",
                    "label": "gliner-medium-v2-1",
                    "source_model": "urchade/gliner_medium-v2.1",
                    "trained_model_path": None,
                    "config": {"error": str(exc)},
                    "evaluations": {},
                    "eligible_on_dev": False,
                    "dev_recall": -1.0,
                    "dev_volume_multiplier": 1e9,
                    "math_recall": -1.0,
                    "error": str(exc),
                }
            )
        try:
            spanmarker_output = external_root / "spanmarker-roberta-base"
            spanmarker_model_path = _train_spanmarker(
                train_rows=load_jsonl(upchieve_files["train"]),
                dev_rows=load_jsonl(upchieve_files["dev"]),
                output_dir=spanmarker_output / "model",
                max_length=384,
            )
            spanmarker_evaluations = _evaluate_spanmarker_model(
                model_path=spanmarker_model_path,
                datasets={"upchieve_dev": upchieve_files["dev"], "math_test": args.math_test_file},
                output_dir=spanmarker_output / "predictions",
            )
            spanmarker_result = {
                "family": "spanmarker",
                "label": "spanmarker-roberta-base",
                "source_model": "roberta-base",
                "trained_model_path": str(spanmarker_model_path),
                "config": {"lr": 1e-5, "epochs": 3, "max_length": 384, "entity_max_length": 8},
                "evaluations": spanmarker_evaluations,
                "eligible_on_dev": _metric_value(
                    (spanmarker_evaluations.get("upchieve_dev") or {}).get("candidate_volume_multiplier"),
                    default=1e9,
                )
                <= args.volume_cap,
                "dev_recall": _metric_value((spanmarker_evaluations.get("upchieve_dev") or {}).get("recall")),
                "dev_volume_multiplier": _metric_value(
                    (spanmarker_evaluations.get("upchieve_dev") or {}).get("candidate_volume_multiplier"),
                    default=1e9,
                ),
                "math_recall": _metric_value((spanmarker_evaluations.get("math_test") or {}).get("recall")),
            }
            all_candidates.append(spanmarker_result)
        except Exception as exc:  # pragma: no cover - dependency/runtime variability
            all_candidates.append(
                {
                    "family": "spanmarker",
                    "label": "spanmarker-roberta-base",
                    "source_model": "roberta-base",
                    "trained_model_path": None,
                    "config": {"error": str(exc)},
                    "evaluations": {},
                    "eligible_on_dev": False,
                    "dev_recall": -1.0,
                    "dev_volume_multiplier": 1e9,
                    "math_recall": -1.0,
                    "error": str(exc),
                }
            )

    finalists = _choose_top_entries(all_candidates, count=3)
    final_datasets = {
        "upchieve_test": upchieve_files["test"],
        "saga27_test": saga_file,
        "math_test": args.math_test_file,
        "upchieve_dev": upchieve_files["dev"],
    }
    final_results: list[dict[str, Any]] = [incumbent_result]
    for finalist in finalists:
        finalist_copy = dict(finalist)
        family = str(finalist["family"])
        if family == "drop_in":
            trained_model_path = Path(str(finalist["trained_model_path"]))
            evaluations = _evaluate_hf_model(
                label=str(finalist["label"]),
                model_path=trained_model_path,
                datasets=final_datasets,
                predictions_dir=training_root / str(finalist["label"]) / "final_predictions",
                batch_size=16,
                max_length=int((finalist.get("config") or {}).get("max_length") or 384),
                context_mode=str((finalist.get("config") or {}).get("context_mode") or "none"),
            )
            finalist_copy["evaluations"] = evaluations
        elif family == "gliner" and not finalist.get("error"):
            threshold = float((finalist.get("config") or {}).get("threshold"))
            evaluations = _evaluate_gliner_finalist(
                model_name=str(finalist["source_model"]),
                threshold=threshold,
                datasets=final_datasets,
                output_dir=experiment.root / "external" / "gliner" / "final_predictions",
            )
            finalist_copy["evaluations"] = evaluations
        elif family == "spanmarker" and not finalist.get("error"):
            evaluations = _evaluate_spanmarker_model(
                model_path=Path(str(finalist["trained_model_path"])),
                datasets=final_datasets,
                output_dir=experiment.root / "external" / "spanmarker-roberta-base" / "final_predictions",
            )
            finalist_copy["evaluations"] = evaluations
        final_results.append(finalist_copy)

    incumbent_min_recall = min(
        _metric_value((incumbent_result["evaluations"]["upchieve_test"]).get("recall")),
        _metric_value((incumbent_result["evaluations"]["saga27_test"]).get("recall")),
    )
    eligible_finalists = []
    for result in final_results:
        evaluations = result.get("evaluations") or {}
        upchieve_test = evaluations.get("upchieve_test") or {}
        saga_test = evaluations.get("saga27_test") or {}
        math_test = evaluations.get("math_test") or {}
        if not upchieve_test or not saga_test or not math_test:
            continue
        result["final_min_recall"] = min(
            _metric_value(upchieve_test.get("recall")),
            _metric_value(saga_test.get("recall")),
        )
        result["math_recall_drop_vs_incumbent"] = _metric_value(
            incumbent_result["evaluations"]["math_test"].get("recall")
        ) - _metric_value(math_test.get("recall"))
        result["passes_math_guard"] = result["math_recall_drop_vs_incumbent"] <= 0.01
        result["passes_volume_guard"] = _metric_value(
            (result.get("evaluations") or {}).get("upchieve_dev", {}).get("candidate_volume_multiplier"),
            default=1e9,
        ) <= args.volume_cap
        if result["passes_math_guard"] and result["passes_volume_guard"]:
            eligible_finalists.append(result)

    winning_candidate = sorted(eligible_finalists, key=_final_rank_key, reverse=True)[0] if eligible_finalists else incumbent_result
    promotion_delta = _metric_value(winning_candidate.get("final_min_recall")) - incumbent_min_recall
    promoted = winning_candidate.get("label") != incumbent_result.get("label") and promotion_delta >= 0.02

    compact_benchmark_summary = {
        "upchieve": {
            "dataset": {
                "dialogue_count": benchmark_summary["upchieve"]["dataset"]["dialogue_count"],
                "row_count": benchmark_summary["upchieve"]["dataset"]["row_count"],
                "positive_row_count": benchmark_summary["upchieve"]["dataset"]["positive_row_count"],
                "positive_dialogue_count": benchmark_summary["upchieve"]["dataset"]["positive_dialogue_count"],
            },
            "split_row_summaries": benchmark_summary["upchieve"]["split_row_summaries"],
        },
        "saga27": {
            "dataset": {
                "file_count": benchmark_summary["saga27"]["dataset"]["file_count"],
                "row_count": benchmark_summary["saga27"]["dataset"]["row_count"],
                "positive_row_count": benchmark_summary["saga27"]["dataset"]["positive_row_count"],
            },
            "row_summary": benchmark_summary["saga27"]["row_summary"],
        },
        "validation": benchmark_summary["validation"],
    }

    summary = {
        "benchmark_summary": compact_benchmark_summary,
        "incumbent": incumbent_result,
        "scout_results": scout_results,
        "all_candidate_results": all_candidates,
        "finalists": final_results,
        "winning_candidate": winning_candidate,
        "promotion_delta_vs_incumbent_min_recall": promotion_delta,
        "promoted": promoted,
        "volume_cap": args.volume_cap,
    }
    _write_json(experiment.summary_path, summary)

    report_lines = [
        "# Candidate Model Selection",
        "",
        "## Benchmark",
        "",
        f"- UpChieve raw file: `{args.upchieve_raw_file}`",
        f"- Saga raw dir: `{args.saga_raw_dir}`",
        f"- Volume cap: `{args.volume_cap:.2f}`",
        f"- Incumbent: `{args.incumbent_model}`",
        "",
        "## Scout Comparison",
        "",
        _report_comparison_table(all_candidates),
        "",
        "## Incumbent",
        "",
        f"- UpChieve dev recall: `{_fmt((incumbent_result['evaluations']['upchieve_dev']).get('recall'))}`",
        f"- UpChieve test recall: `{_fmt((incumbent_result['evaluations']['upchieve_test']).get('recall'))}`",
        f"- Saga recall: `{_fmt((incumbent_result['evaluations']['saga27_test']).get('recall'))}`",
        f"- Math recall: `{_fmt((incumbent_result['evaluations']['math_test']).get('recall'))}`",
        "",
        "## Winner",
        "",
        f"- label: `{winning_candidate.get('label')}`",
        f"- family: `{winning_candidate.get('family')}`",
        f"- final_min_recall: `{_fmt(winning_candidate.get('final_min_recall'))}`",
        f"- promotion_delta_vs_incumbent: `{_fmt(promotion_delta)}`",
        f"- promoted: `{promoted}`",
        f"- math_recall_drop_vs_incumbent: `{_fmt(winning_candidate.get('math_recall_drop_vs_incumbent'))}`",
        f"- upchieve_dev_volume_multiplier: `{_fmt((winning_candidate.get('evaluations') or {}).get('upchieve_dev', {}).get('candidate_volume_multiplier'))}`",
        "",
    ]
    experiment.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_run_metadata(
        experiment.metadata_path,
        {
            "run_name": args.run_name,
            "upchieve_raw_file": str(args.upchieve_raw_file),
            "saga_raw_dir": str(args.saga_raw_dir),
            "incumbent_model": str(args.incumbent_model),
            "candidate_output_dir": str(args.candidate_output_dir),
            "volume_cap": args.volume_cap,
            "promoted": promoted,
        },
    )

    status = "keep" if promoted else "discard"
    description = (
        f"candidate model selection kept {winning_candidate.get('label')} with final min recall "
        f"{_metric_value(winning_candidate.get('final_min_recall'), default=0.0):.4f} and "
        f"promotion delta {promotion_delta:.4f} [{experiment.root.name}]"
    )
    _append_result(_metric_value(winning_candidate.get("final_min_recall"), default=0.0), status=status, description=description)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
