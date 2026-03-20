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

from contextshift_deid.candidate_input import INPUT_FORMATS, resolve_candidate_input_format
from contextshift_deid.candidate_audit import compute_candidate_audit_metrics, merge_candidate_predictions
from contextshift_deid.constants import CANDIDATE_DIR, RESULTS_HEADER
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, create_experiment_run, slugify, write_run_metadata

RESULTS_PATH = ROOT / "results.tsv"
DEFAULT_RUN_ROOT = EXPERIMENTS_DIR / "candidate_input_branch_suite"
DEFAULT_VOLUME_CAP = 3.0
DEFAULT_CONFIRMATION_SEEDS = (13, 42, 101)


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
            f"- candidate_volume_multiplier: `{_fmt(metrics.get('candidate_volume_multiplier'))}`",
            f"- rare_type_mean_recall: `{_fmt(metrics.get('rare_type_mean_recall'))}`",
            "",
        ]
    ) + "\n"


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


def _predict_candidate(
    *,
    model_path: Path | str,
    input_file: Path,
    output_file: Path,
    batch_size: int,
    max_length: int,
    input_format: str,
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
            "--input-format",
            input_format,
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
    input_format: str,
    seed: int,
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
            "--input-format",
            input_format,
            "--selection-metric",
            "recall",
            "--seed",
            str(seed),
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


def _evaluate_model(
    *,
    label: str,
    model_path: Path,
    datasets: Mapping[str, Path],
    predictions_dir: Path,
    max_length: int,
    input_format: str,
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    for split_name, gold_file in datasets.items():
        prediction_file = predictions_dir / f"{slugify(label)}_{split_name}_predictions.jsonl"
        _predict_candidate(
            model_path=model_path,
            input_file=gold_file,
            output_file=prediction_file,
            batch_size=16,
            max_length=max_length,
            input_format=input_format,
        )
        evaluations[split_name] = _evaluate_prediction_run(
            label=f"{label} {split_name}",
            gold_file=gold_file,
            prediction_file=prediction_file,
            summary_path=prediction_file.with_suffix(".summary.json"),
            report_path=prediction_file.with_suffix(".report.md"),
        )
    return evaluations


def _make_result(
    *,
    backbone: str,
    label: str,
    source_model: str | Path,
    trained_model_path: Path,
    config: Mapping[str, Any],
    evaluations: Mapping[str, Mapping[str, Any]],
    incumbent_math_recall: float,
    volume_cap: float,
) -> dict[str, Any]:
    dev_metrics = evaluations["upchieve_dev"]
    math_metrics = evaluations["math_test"]
    math_recall = _metric_value(math_metrics.get("recall"))
    math_recall_drop = incumbent_math_recall - math_recall
    return {
        "family": "drop_in",
        "backbone": backbone,
        "label": label,
        "source_model": str(source_model),
        "trained_model_path": str(trained_model_path),
        "config": dict(config),
        "evaluations": {name: dict(metrics) for name, metrics in evaluations.items()},
        "dev_recall": _metric_value(dev_metrics.get("recall")),
        "dev_volume_multiplier": _metric_value(dev_metrics.get("candidate_volume_multiplier"), default=1e9),
        "math_recall": math_recall,
        "math_recall_drop_vs_incumbent": math_recall_drop,
        "passes_volume_guard": _metric_value(dev_metrics.get("candidate_volume_multiplier"), default=1e9) <= volume_cap,
        "passes_math_guard": math_recall_drop <= 0.01,
        "eligible_on_dev": (
            _metric_value(dev_metrics.get("candidate_volume_multiplier"), default=1e9) <= volume_cap
            and math_recall_drop <= 0.01
        ),
    }


def _scout_rank_key(entry: Mapping[str, Any]) -> tuple[int, float, float, float]:
    return (
        int(bool(entry.get("eligible_on_dev"))),
        _metric_value(entry.get("dev_recall")),
        -_metric_value(entry.get("dev_volume_multiplier"), default=1e9),
        _metric_value(((entry.get("evaluations") or {}).get("upchieve_dev") or {}).get("f1")),
    )


def _final_rank_key(entry: Mapping[str, Any]) -> tuple[float, float, float, float]:
    evaluations = entry.get("evaluations") or {}
    upchieve_test = evaluations.get("upchieve_test") or {}
    saga_test = evaluations.get("saga27_test") or {}
    return (
        min(_metric_value(upchieve_test.get("recall")), _metric_value(saga_test.get("recall"))),
        _metric_value((upchieve_test.get("rare_type_mean_recall")), default=-1.0),
        -_metric_value((evaluations.get("upchieve_dev") or {}).get("candidate_volume_multiplier"), default=1e9),
        _metric_value(upchieve_test.get("f1")),
    )


def _report_scout_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| label | backbone | input_format | dev recall | dev volume | math recall | eligible |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=_scout_rank_key, reverse=True):
        config = row.get("config") or {}
        lines.append(
            "| {label} | {backbone} | {input_format} | {dev_recall} | {dev_volume} | {math_recall} | {eligible} |".format(
                label=row["label"],
                backbone=row["backbone"],
                input_format=config.get("input_format"),
                dev_recall=_fmt(row.get("dev_recall")),
                dev_volume=_fmt(row.get("dev_volume_multiplier")),
                math_recall=_fmt(row.get("math_recall")),
                eligible="yes" if row.get("eligible_on_dev") else "no",
            )
        )
    return "\n".join(lines)


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run candidate input-format branch experiments.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="candidate-input-branch-suite")
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
    parser.add_argument("--modernbert-model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--math-test-file", type=Path, default=CANDIDATE_DIR / "test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirmation-seeds", default="13,42,101")
    parser.add_argument("--volume-cap", type=float, default=DEFAULT_VOLUME_CAP)
    args = parser.parse_args(argv)

    experiment = create_experiment_run(args.run_name, root_dir=args.run_root)
    benchmark_summary = _run_build_benchmark(
        upchieve_raw_file=args.upchieve_raw_file,
        saga_raw_dir=args.saga_raw_dir,
        output_dir=args.candidate_output_dir,
        seed=args.seed,
    )
    datasets = {
        "upchieve_train": args.candidate_output_dir / "upchieve_math_ground_truth_train.jsonl",
        "upchieve_dev": args.candidate_output_dir / "upchieve_math_ground_truth_dev.jsonl",
        "upchieve_test": args.candidate_output_dir / "upchieve_math_ground_truth_test.jsonl",
        "saga27_test": args.candidate_output_dir / "saga27_math_ground_truth_test.jsonl",
        "math_test": args.math_test_file,
    }

    incumbent_evaluations = _evaluate_model(
        label="incumbent-distilbert-rebuilt",
        model_path=args.incumbent_model,
        datasets={
            "upchieve_dev": datasets["upchieve_dev"],
            "upchieve_test": datasets["upchieve_test"],
            "saga27_test": datasets["saga27_test"],
            "math_test": datasets["math_test"],
        },
        predictions_dir=experiment.root / "incumbent",
        max_length=384,
        input_format="turn_only_v1",
    )
    incumbent_final_min_recall = min(
        _metric_value((incumbent_evaluations["upchieve_test"]).get("recall")),
        _metric_value((incumbent_evaluations["saga27_test"]).get("recall")),
    )
    incumbent_math_recall = _metric_value((incumbent_evaluations["math_test"]).get("recall"))

    backbone_specs = [
        {
            "backbone": "distilbert",
            "label": "distilbert-from-incumbent",
            "model": args.incumbent_model,
            "lr": 2e-5,
            "epochs": 3,
            "batch_size": 8,
            "max_length": 256,
        },
        {
            "backbone": "modernbert",
            "label": "modernbert-base",
            "model": args.modernbert_model,
            "lr": 3e-5,
            "epochs": 3,
            "batch_size": 4,
            "max_length": 256,
        },
    ]

    scout_results: list[dict[str, Any]] = []
    for spec in backbone_specs:
        for input_format in INPUT_FORMATS:
            output_dir = experiment.root / "training" / spec["label"] / f"{input_format}-seed-{args.seed}"
            _train_candidate(
                model=spec["model"],
                train_file=datasets["upchieve_train"],
                dev_file=datasets["upchieve_dev"],
                output_dir=output_dir,
                epochs=int(spec["epochs"]),
                lr=float(spec["lr"]),
                batch_size=int(spec["batch_size"]),
                max_length=int(spec["max_length"]),
                input_format=resolve_candidate_input_format(input_format=input_format),
                seed=args.seed,
            )
            evaluations = _evaluate_model(
                label=f"{spec['label']}-{input_format}-seed-{args.seed}",
                model_path=output_dir,
                datasets={
                    "upchieve_dev": datasets["upchieve_dev"],
                    "math_test": datasets["math_test"],
                },
                predictions_dir=output_dir / "predictions",
                max_length=int(spec["max_length"]),
                input_format=input_format,
            )
            scout_results.append(
                _make_result(
                    backbone=str(spec["backbone"]),
                    label=f"{spec['label']}-{input_format}-seed-{args.seed}",
                    source_model=spec["model"],
                    trained_model_path=output_dir,
                    config={
                        "input_format": input_format,
                        "seed": args.seed,
                        "epochs": int(spec["epochs"]),
                        "lr": float(spec["lr"]),
                        "batch_size": int(spec["batch_size"]),
                        "max_length": int(spec["max_length"]),
                    },
                    evaluations=evaluations,
                    incumbent_math_recall=incumbent_math_recall,
                    volume_cap=args.volume_cap,
                )
            )

    finalists: list[dict[str, Any]] = []
    for backbone in ("distilbert", "modernbert"):
        backbone_entries = [entry for entry in scout_results if entry["backbone"] == backbone]
        finalists.append(sorted(backbone_entries, key=_scout_rank_key, reverse=True)[0])

    final_results: list[dict[str, Any]] = []
    for finalist in finalists:
        finalist_copy = dict(finalist)
        evaluations = _evaluate_model(
            label=str(finalist["label"]),
            model_path=Path(str(finalist["trained_model_path"])),
            datasets={
                "upchieve_dev": datasets["upchieve_dev"],
                "upchieve_test": datasets["upchieve_test"],
                "saga27_test": datasets["saga27_test"],
                "math_test": datasets["math_test"],
            },
            predictions_dir=experiment.root / "finalists" / str(finalist["label"]),
            max_length=int((finalist.get("config") or {}).get("max_length") or 256),
            input_format=str((finalist.get("config") or {}).get("input_format") or "turn_only_v1"),
        )
        finalist_copy["evaluations"] = evaluations
        finalist_copy["final_min_recall"] = min(
            _metric_value((evaluations["upchieve_test"]).get("recall")),
            _metric_value((evaluations["saga27_test"]).get("recall")),
        )
        finalist_copy["math_recall_drop_vs_incumbent"] = incumbent_math_recall - _metric_value(
            (evaluations["math_test"]).get("recall")
        )
        finalist_copy["passes_math_guard"] = finalist_copy["math_recall_drop_vs_incumbent"] <= 0.01
        finalist_copy["passes_volume_guard"] = _metric_value(
            (evaluations["upchieve_dev"]).get("candidate_volume_multiplier"),
            default=1e9,
        ) <= args.volume_cap
        final_results.append(finalist_copy)

    winning_branch = sorted(final_results, key=_final_rank_key, reverse=True)[0]
    promotion_delta = _metric_value(winning_branch.get("final_min_recall")) - incumbent_final_min_recall
    promoted_after_scout = (
        winning_branch.get("passes_math_guard")
        and winning_branch.get("passes_volume_guard")
        and promotion_delta >= 0.02
    )

    confirmation_summary: dict[str, Any] | None = None
    promoted_after_confirmation = False
    if promoted_after_scout:
        confirmation_seed_results: list[dict[str, Any]] = []
        confirmation_seeds = _parse_seeds(args.confirmation_seeds)
        config = dict(winning_branch.get("config") or {})
        for seed in confirmation_seeds:
            seed_output_dir = experiment.root / "confirmation" / str(winning_branch["label"]) / f"seed-{seed}"
            _train_candidate(
                model=winning_branch["source_model"],
                train_file=datasets["upchieve_train"],
                dev_file=datasets["upchieve_dev"],
                output_dir=seed_output_dir,
                epochs=int(config.get("epochs") or 3),
                lr=float(config.get("lr") or 2e-5),
                batch_size=int(config.get("batch_size") or 8),
                max_length=int(config.get("max_length") or 256),
                input_format=str(config.get("input_format") or "turn_only_v1"),
                seed=seed,
            )
            evaluations = _evaluate_model(
                label=f"{winning_branch['label']}-confirmation-seed-{seed}",
                model_path=seed_output_dir,
                datasets={
                    "upchieve_dev": datasets["upchieve_dev"],
                    "upchieve_test": datasets["upchieve_test"],
                    "saga27_test": datasets["saga27_test"],
                    "math_test": datasets["math_test"],
                },
                predictions_dir=seed_output_dir / "predictions",
                max_length=int(config.get("max_length") or 256),
                input_format=str(config.get("input_format") or "turn_only_v1"),
            )
            seed_result = {
                "seed": seed,
                "evaluations": evaluations,
                "final_min_recall": min(
                    _metric_value((evaluations["upchieve_test"]).get("recall")),
                    _metric_value((evaluations["saga27_test"]).get("recall")),
                ),
                "math_recall_drop_vs_incumbent": incumbent_math_recall - _metric_value(
                    (evaluations["math_test"]).get("recall")
                ),
                "passes_math_guard": (
                    incumbent_math_recall - _metric_value((evaluations["math_test"]).get("recall"))
                )
                <= 0.01,
                "passes_volume_guard": _metric_value(
                    (evaluations["upchieve_dev"]).get("candidate_volume_multiplier"),
                    default=1e9,
                )
                <= args.volume_cap,
            }
            confirmation_seed_results.append(seed_result)

        mean_final_min_recall = statistics.mean(float(item["final_min_recall"]) for item in confirmation_seed_results)
        promoted_after_confirmation = (
            all(item["passes_math_guard"] and item["passes_volume_guard"] for item in confirmation_seed_results)
            and mean_final_min_recall >= incumbent_final_min_recall + 0.02
        )
        confirmation_summary = {
            "config": config,
            "seeds": confirmation_seeds,
            "seed_results": confirmation_seed_results,
            "mean_final_min_recall": mean_final_min_recall,
            "promoted": promoted_after_confirmation,
        }

    summary = {
        "benchmark_summary": benchmark_summary,
        "incumbent": {
            "label": "incumbent-distilbert-rebuilt",
            "source_model": str(args.incumbent_model),
            "input_format": "turn_only_v1",
            "evaluations": incumbent_evaluations,
            "final_min_recall": incumbent_final_min_recall,
        },
        "scout_results": scout_results,
        "finalists": final_results,
        "winning_branch": winning_branch,
        "promotion_delta_vs_incumbent_min_recall": promotion_delta,
        "promoted_after_scout": promoted_after_scout,
        "confirmation": confirmation_summary,
        "promoted": promoted_after_confirmation if confirmation_summary is not None else False,
        "volume_cap": args.volume_cap,
    }
    _write_json(experiment.summary_path, summary)

    report_lines = [
        "# Candidate Input Branch Suite",
        "",
        "## Benchmark",
        "",
        f"- UpChieve raw file: `{args.upchieve_raw_file}`",
        f"- Saga raw dir: `{args.saga_raw_dir}`",
        f"- Volume cap: `{args.volume_cap:.2f}`",
        f"- Incumbent model: `{args.incumbent_model}`",
        f"- ModernBERT model: `{args.modernbert_model}`",
        "",
        "## Scout Results",
        "",
        _report_scout_table(scout_results),
        "",
        "## Incumbent",
        "",
        f"- final_min_recall: `{_fmt(incumbent_final_min_recall)}`",
        f"- UpChieve test recall: `{_fmt((incumbent_evaluations['upchieve_test']).get('recall'))}`",
        f"- Saga recall: `{_fmt((incumbent_evaluations['saga27_test']).get('recall'))}`",
        f"- Math recall: `{_fmt((incumbent_evaluations['math_test']).get('recall'))}`",
        "",
        "## Winning Scout Branch",
        "",
        f"- label: `{winning_branch.get('label')}`",
        f"- input_format: `{(winning_branch.get('config') or {}).get('input_format')}`",
        f"- final_min_recall: `{_fmt(winning_branch.get('final_min_recall'))}`",
        f"- promotion_delta_vs_incumbent: `{_fmt(promotion_delta)}`",
        f"- promoted_after_scout: `{promoted_after_scout}`",
        "",
    ]
    if confirmation_summary is not None:
        report_lines.extend(
            [
                "## Seed Confirmation",
                "",
                f"- mean_final_min_recall: `{_fmt(confirmation_summary.get('mean_final_min_recall'))}`",
                f"- promoted: `{confirmation_summary.get('promoted')}`",
                "",
            ]
        )
    experiment.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_run_metadata(
        experiment.metadata_path,
        {
            "run_name": args.run_name,
            "upchieve_raw_file": str(args.upchieve_raw_file),
            "saga_raw_dir": str(args.saga_raw_dir),
            "candidate_output_dir": str(args.candidate_output_dir),
            "incumbent_model": str(args.incumbent_model),
            "modernbert_model": str(args.modernbert_model),
            "scout_seed": args.seed,
            "confirmation_seeds": list(_parse_seeds(args.confirmation_seeds)),
            "volume_cap": args.volume_cap,
            "promoted": promoted_after_confirmation if confirmation_summary is not None else False,
        },
    )

    final_status = "keep" if (confirmation_summary is not None and promoted_after_confirmation) else "discard"
    final_metric = (
        confirmation_summary.get("mean_final_min_recall")
        if confirmation_summary is not None
        else winning_branch.get("final_min_recall")
    )
    description = (
        f"candidate input branch suite winner {winning_branch.get('label')} "
        f"reached final min recall {_metric_value(final_metric, default=0.0):.4f} "
        f"with promotion delta {promotion_delta:.4f} [{experiment.root.name}]"
    )
    _append_result(_metric_value(final_metric, default=0.0), status=final_status, description=description)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
