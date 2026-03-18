from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.candidate_audit import compute_candidate_audit_metrics, merge_candidate_predictions
from contextshift_deid.constants import CANDIDATE_DIR, RESULTS_HEADER
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, create_experiment_run, write_run_metadata

RESULTS_PATH = ROOT / "results.tsv"
DEFAULT_RUN_ROOT = EXPERIMENTS_DIR / "candidate_checkpoint_benchmark"
DEFAULT_VOLUME_CAP = 3.0


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    return env


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True, env=_command_env())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluate(gold_file: Path, prediction_file: Path) -> dict[str, Any]:
    merged = merge_candidate_predictions(load_jsonl(gold_file), load_jsonl(prediction_file))
    return compute_candidate_audit_metrics(merged)


def _predict_and_evaluate(
    *,
    label: str,
    model_path: Path,
    gold_file: Path,
    output_dir: Path,
    batch_size: int,
    max_length: int,
    context_mode: str,
) -> dict[str, Any]:
    prediction_file = output_dir / f"{label}_{gold_file.stem}_predictions.jsonl"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "predict_candidate.py"),
            "--model",
            str(model_path),
            "--input-file",
            str(gold_file),
            "--output-file",
            str(prediction_file),
            "--batch-size",
            str(batch_size),
            "--max-length",
            str(max_length),
            "--context-mode",
            context_mode,
        ]
    )
    metrics = _evaluate(gold_file, prediction_file)
    _write_json(prediction_file.with_suffix(".summary.json"), metrics)
    return metrics


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


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    evaluations = result["evaluations"]
    return (
        result["final_min_recall"],
        _metric_value(evaluations["upchieve_test"].get("rare_type_mean_recall")),
        -_metric_value(evaluations["upchieve_dev"].get("candidate_volume_multiplier"), default=1e9),
        _metric_value(evaluations["upchieve_test"].get("f1")),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark existing candidate checkpoints on the new ground-truth splits.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="candidate-checkpoint-benchmark")
    parser.add_argument("--upchieve-dev-file", type=Path, default=CANDIDATE_DIR / "upchieve_math_ground_truth_dev.jsonl")
    parser.add_argument("--upchieve-test-file", type=Path, default=CANDIDATE_DIR / "upchieve_math_ground_truth_test.jsonl")
    parser.add_argument("--saga-file", type=Path, default=CANDIDATE_DIR / "saga27_math_ground_truth_test.jsonl")
    parser.add_argument("--math-test-file", type=Path, default=CANDIDATE_DIR / "test.jsonl")
    parser.add_argument("--volume-cap", type=float, default=DEFAULT_VOLUME_CAP)
    args = parser.parse_args(argv)

    experiment = create_experiment_run(args.run_name, root_dir=args.run_root)
    models = [
        {
            "label": "distilbert-incumbent",
            "path": ROOT / "runs" / "candidate_math_distilbert_rebuilt",
            "batch_size": 16,
            "max_length": 384,
            "context_mode": "none",
        },
        {
            "label": "roberta-locked",
            "path": Path("/Users/chason/contextshift-deid/workspaces/candidate/runs/locked_roberta_math_best_f1"),
            "batch_size": 16,
            "max_length": 384,
            "context_mode": "none",
        },
        {
            "label": "modernbert-1ep",
            "path": Path("/Users/chason/contextshift-deid/artifacts/experiments/20260311_115847_candidate-math-modernbert-1ep-dev/model"),
            "batch_size": 8,
            "max_length": 384,
            "context_mode": "none",
        },
        {
            "label": "deberta-v3-small",
            "path": Path("/Users/chason/contextshift-deid/runs/candidate_math_deberta_v1"),
            "batch_size": 16,
            "max_length": 384,
            "context_mode": "none",
        },
    ]
    datasets = {
        "upchieve_dev": args.upchieve_dev_file,
        "upchieve_test": args.upchieve_test_file,
        "saga27_test": args.saga_file,
        "math_test": args.math_test_file,
    }

    results: list[dict[str, Any]] = []
    for model in models:
        model_output_dir = experiment.root / model["label"]
        model_output_dir.mkdir(parents=True, exist_ok=True)
        evaluations: dict[str, Any] = {}
        for split_name, gold_file in datasets.items():
            evaluations[split_name] = _predict_and_evaluate(
                label=model["label"],
                model_path=model["path"],
                gold_file=gold_file,
                output_dir=model_output_dir,
                batch_size=int(model["batch_size"]),
                max_length=int(model["max_length"]),
                context_mode=str(model["context_mode"]),
            )
        result = {
            "label": model["label"],
            "path": str(model["path"]),
            "evaluations": evaluations,
        }
        result["final_min_recall"] = min(
            _metric_value(evaluations["upchieve_test"].get("recall")),
            _metric_value(evaluations["saga27_test"].get("recall")),
        )
        results.append(result)

    incumbent = next(result for result in results if result["label"] == "distilbert-incumbent")
    incumbent_math_recall = _metric_value(incumbent["evaluations"]["math_test"].get("recall"))
    eligible_results: list[dict[str, Any]] = []
    for result in results:
        dev_volume = _metric_value(result["evaluations"]["upchieve_dev"].get("candidate_volume_multiplier"), default=1e9)
        math_recall = _metric_value(result["evaluations"]["math_test"].get("recall"))
        result["passes_volume_guard"] = dev_volume <= args.volume_cap
        result["math_recall_drop_vs_incumbent"] = incumbent_math_recall - math_recall
        result["passes_math_guard"] = result["math_recall_drop_vs_incumbent"] <= 0.01
        if result["passes_volume_guard"] and result["passes_math_guard"]:
            eligible_results.append(result)

    winner = sorted(eligible_results or results, key=_rank_key, reverse=True)[0]
    promoted = winner["label"] != incumbent["label"] and (
        winner["final_min_recall"] - incumbent["final_min_recall"] >= 0.02
    )

    summary = {
        "datasets": {name: str(path) for name, path in datasets.items()},
        "results": results,
        "winner": winner,
        "promoted": promoted,
        "volume_cap": args.volume_cap,
    }
    _write_json(experiment.summary_path, summary)

    report_lines = [
        "# Candidate Checkpoint Benchmark",
        "",
        "| label | eligible | dev recall | dev volume | upchieve test recall | saga recall | math recall | math drop | final min recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(
        results,
        key=lambda item: (
            int(bool(item.get("passes_volume_guard")) and bool(item.get("passes_math_guard"))),
            *_rank_key(item),
        ),
        reverse=True,
    ):
        evaluations = result["evaluations"]
        eligible = bool(result.get("passes_volume_guard")) and bool(result.get("passes_math_guard"))
        report_lines.append(
            "| {label} | {eligible} | {dev_recall} | {dev_volume} | {up_test} | {saga} | {math} | {math_drop} | {final_min} |".format(
                label=result["label"],
                eligible="yes" if eligible else "no",
                dev_recall=_fmt(evaluations["upchieve_dev"].get("recall")),
                dev_volume=_fmt(evaluations["upchieve_dev"].get("candidate_volume_multiplier")),
                up_test=_fmt(evaluations["upchieve_test"].get("recall")),
                saga=_fmt(evaluations["saga27_test"].get("recall")),
                math=_fmt(evaluations["math_test"].get("recall")),
                math_drop=_fmt(result.get("math_recall_drop_vs_incumbent")),
                final_min=_fmt(result["final_min_recall"]),
            )
        )
    report_lines.extend(
        [
            "",
            "## Winner",
            "",
            f"- label: `{winner['label']}`",
            f"- promoted: `{promoted}`",
            f"- final_min_recall: `{_fmt(winner['final_min_recall'])}`",
            f"- math_recall_drop_vs_incumbent: `{_fmt(winner.get('math_recall_drop_vs_incumbent'))}`",
            f"- upchieve_dev_volume_multiplier: `{_fmt(winner['evaluations']['upchieve_dev'].get('candidate_volume_multiplier'))}`",
        ]
    )
    experiment.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_run_metadata(
        experiment.metadata_path,
        {
            "run_name": args.run_name,
            "datasets": {name: str(path) for name, path in datasets.items()},
            "volume_cap": args.volume_cap,
            "promoted": promoted,
        },
    )

    _append_result(
        winner["final_min_recall"],
        status="keep" if promoted else "discard",
        description=(
            f"candidate checkpoint benchmark selected {winner['label']} with final min recall "
            f"{winner['final_min_recall']:.4f} [{experiment.root.name}]"
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
