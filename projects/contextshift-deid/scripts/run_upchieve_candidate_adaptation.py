from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.candidate_audit import compute_candidate_audit_metrics, merge_candidate_predictions
from contextshift_deid.constants import (
    ACTION_DIR,
    CANDIDATE_DIR,
    LEGACY_CANDIDATE_DIR,
    LEGACY_EXPERIMENTS_DIR,
    LEGACY_RUNS_DIR,
    RESULTS_HEADER,
)
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, create_experiment_run, slugify, write_run_metadata

RESULTS_PATH = ROOT / "results.tsv"
DEFAULT_RUN_ROOT = LEGACY_EXPERIMENTS_DIR / "candidate_adaptation"


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_audit(gold_file: Path, prediction_file: Path) -> dict[str, Any]:
    merged = merge_candidate_predictions(load_jsonl(gold_file), load_jsonl(prediction_file))
    return compute_candidate_audit_metrics(merged)


def _evaluate_and_write(
    *,
    gold_file: Path,
    prediction_file: Path,
    summary_path: Path,
    report_path: Path,
    sample_limit: int = 10,
) -> dict[str, Any]:
    metrics = _evaluate_audit(gold_file, prediction_file)
    _write_json(summary_path, metrics)
    report_lines = [
        "# Candidate Audit Report",
        "",
        f"- gold_file: `{gold_file}`",
        f"- prediction_file: `{prediction_file}`",
        "",
        "## Headline",
        "",
        f"- recall: `{_fmt(metrics.get('recall'))}`",
        f"- f1: `{_fmt(metrics.get('f1'))}`",
        f"- worst_context_recall: `{_fmt(metrics.get('worst_context_recall'))}`",
        f"- action_seed_span_coverage: `{_fmt(metrics.get('action_seed_span_coverage'))}`",
        f"- protected_redact_recall: `{_fmt(metrics.get('protected_redact_recall'))}`",
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
    protected_examples = (metrics.get("protected_redact_miss_examples") or [])[:sample_limit]
    if protected_examples:
        for example in protected_examples:
            report_lines.append(
                f"- `{example['id']}` `{example.get('subject', 'unknown')}` `{example.get('entity_type', 'unknown')}` `{example.get('span_text', '')}`"
            )
            report_lines.append(f"  - preview: `{example.get('preview', '')}`")
    else:
        report_lines.append("- none")
    report_lines.append("")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return metrics


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _metric_value(value: Any, *, default: float = -1.0) -> float:
    if value is None:
        return default
    return float(value)


def _result_line(metric: float, status: str, description: str) -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"working\tcandidate\t{metric:.6f}\t{status}\t{description}\n")


def _predict(
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


def _train(
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


def _config_sort_key(entry: Mapping[str, Any]) -> tuple[float, float, float]:
    dev = entry["evaluations"]["upchieve_dev"]
    math_test = entry["evaluations"]["math_test"]
    return (
        _metric_value(dev.get("protected_redact_recall")),
        _metric_value(math_test.get("recall")),
        _metric_value(dev.get("recall")),
    )


def _local_model_max_length(model_path: Path) -> int | None:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = config.get("max_position_embeddings")
    if value is None:
        return None
    return int(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the bounded UpChieve candidate recall adaptation sweep.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", default="upchieve-candidate-recall-adaptation")
    parser.add_argument("--baseline-model", type=Path, default=LEGACY_RUNS_DIR / "candidate")
    parser.add_argument("--fallback-model", default="distilroberta-base")
    parser.add_argument("--skip-fallback", action="store_true")
    parser.add_argument(
        "--action-file",
        type=Path,
        default=ACTION_DIR / "upchieve_english_social_train_v1_v2.jsonl",
    )
    parser.add_argument("--proxy-prefix", default="upchieve_english_social_proxy")
    parser.add_argument("--train-count", type=int, default=200)
    parser.add_argument("--dev-count", type=int, default=80)
    parser.add_argument("--test-count", type=int, default=80)
    parser.add_argument("--math-train-file", type=Path, default=CANDIDATE_DIR / "train.jsonl")
    parser.add_argument("--math-test-file", type=Path, default=CANDIDATE_DIR / "test.jsonl")
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--prediction-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lrs", default="2e-5,3e-5")
    parser.add_argument("--max-lengths", default="256,384")
    parser.add_argument("--context-modes", default="none,pair")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    experiment = create_experiment_run(args.run_name, root_dir=args.run_root)
    training_root = experiment.root / "training"
    training_root.mkdir(parents=True, exist_ok=True)

    proxy_annotation_dir = experiment.root / "proxy_build"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_upchieve_candidate_proxy.py"),
            "--action-file",
            str(args.action_file),
            "--baseline-model",
            str(args.baseline_model),
            "--seed",
            str(args.seed),
            "--train-count",
            str(args.train_count),
            "--dev-count",
            str(args.dev_count),
            "--test-count",
            str(args.test_count),
            "--prefix",
            args.proxy_prefix,
            "--output-dir",
            str(LEGACY_CANDIDATE_DIR),
            "--annotation-dir",
            str(proxy_annotation_dir),
        ]
    )
    proxy_summary = _load_json(proxy_annotation_dir / f"{args.proxy_prefix}_summary.json")

    upchieve_train_file = Path(proxy_summary["split_files"]["train"])
    upchieve_dev_file = Path(proxy_summary["split_files"]["dev"])
    upchieve_test_file = Path(proxy_summary["split_files"]["test"])

    mixed_train_file = LEGACY_CANDIDATE_DIR / "train_mixed_upchieve_english_social_proxy.jsonl"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compose_candidate_train_mix.py"),
            "--math-train-file",
            str(args.math_train_file),
            "--upchieve-train-file",
            str(upchieve_train_file),
            "--output-file",
            str(mixed_train_file),
        ]
    )

    metadata = {
        "run_name": args.run_name,
        "baseline_model": str(args.baseline_model),
        "fallback_model": args.fallback_model,
        "proxy_summary_file": str(proxy_annotation_dir / f"{args.proxy_prefix}_summary.json"),
        "mixed_train_file": str(mixed_train_file),
        "upchieve_train_file": str(upchieve_train_file),
        "upchieve_dev_file": str(upchieve_dev_file),
        "upchieve_test_file": str(upchieve_test_file),
        "math_test_file": str(args.math_test_file),
        "grid": {
            "lrs": args.lrs,
            "max_lengths": args.max_lengths,
            "context_modes": args.context_modes,
            "epochs": args.epochs,
        },
    }
    write_run_metadata(experiment.metadata_path, metadata)

    baseline_outputs: dict[str, dict[str, Any]] = {}
    for name, gold_file, max_length, context_mode in (
        ("upchieve_dev", upchieve_dev_file, 256, "none"),
        ("upchieve_test", upchieve_test_file, 256, "none"),
        ("math_test", args.math_test_file, 256, "none"),
    ):
        prediction_file = experiment.predictions_dir / f"baseline_{name}.jsonl"
        _predict(
            model_path=args.baseline_model,
            input_file=gold_file,
            output_file=prediction_file,
            batch_size=args.prediction_batch_size,
            max_length=max_length,
            context_mode=context_mode,
        )
        baseline_outputs[name] = {
            "prediction_file": str(prediction_file),
            "metrics": _evaluate_and_write(
                gold_file=gold_file,
                prediction_file=prediction_file,
                summary_path=experiment.root / f"baseline_{name}_summary.json",
                report_path=experiment.root / f"baseline_{name}_report.md",
            ),
        }

    compatible_lengths = [int(item.strip()) for item in args.max_lengths.split(",") if item.strip()]
    baseline_max_length = _local_model_max_length(args.baseline_model)
    if baseline_max_length is not None:
        compatible_lengths = [length for length in compatible_lengths if length <= baseline_max_length]
    if not compatible_lengths:
        raise SystemExit(
            f"No compatible max lengths remain for baseline model {args.baseline_model}; "
            f"requested={args.max_lengths} baseline_max_position_embeddings={baseline_max_length}"
        )

    configs: list[dict[str, Any]] = []
    failed_configs: list[dict[str, Any]] = []
    for context_mode in [item.strip() for item in args.context_modes.split(",") if item.strip()]:
        for max_length in compatible_lengths:
            for lr_text in [item.strip() for item in args.lrs.split(",") if item.strip()]:
                lr = float(lr_text)
                slug = slugify(f"distilbert-context-{context_mode}-l{max_length}-lr-{lr_text}")
                training_dir = training_root / slug
                try:
                    _train(
                        model=args.baseline_model,
                        train_file=mixed_train_file,
                        dev_file=upchieve_dev_file,
                        output_dir=training_dir,
                        epochs=args.epochs,
                        lr=lr,
                        batch_size=args.candidate_batch_size,
                        max_length=max_length,
                        context_mode=context_mode,
                    )
                    config_evaluations: dict[str, Any] = {}
                    for name, gold_file in (
                        ("upchieve_dev", upchieve_dev_file),
                        ("upchieve_test", upchieve_test_file),
                        ("math_test", args.math_test_file),
                    ):
                        prediction_file = experiment.predictions_dir / f"{slug}_{name}.jsonl"
                        _predict(
                            model_path=training_dir,
                            input_file=gold_file,
                            output_file=prediction_file,
                            batch_size=args.prediction_batch_size,
                            max_length=max_length,
                            context_mode=context_mode,
                        )
                        config_evaluations[name] = _evaluate_and_write(
                            gold_file=gold_file,
                            prediction_file=prediction_file,
                            summary_path=training_dir / f"{name}_summary.json",
                            report_path=training_dir / f"{name}_report.md",
                        )
                    configs.append(
                        {
                            "label": slug,
                            "model": str(args.baseline_model),
                            "training_dir": str(training_dir),
                            "config": {
                                "context_mode": context_mode,
                                "max_length": max_length,
                                "lr": lr,
                                "epochs": args.epochs,
                            },
                            "evaluations": config_evaluations,
                        }
                    )
                except subprocess.CalledProcessError as exc:
                    failed_configs.append(
                        {
                            "label": slug,
                            "model": str(args.baseline_model),
                            "config": {
                                "context_mode": context_mode,
                                "max_length": max_length,
                                "lr": lr,
                                "epochs": args.epochs,
                            },
                            "error": str(exc),
                        }
                    )

    if not configs:
        raise SystemExit("No same-backbone candidate runs completed successfully.")

    configs.sort(key=_config_sort_key, reverse=True)
    best_same_backbone = configs[0]

    candidates = list(configs)
    baseline_dev_protected = _metric_value(
        baseline_outputs["upchieve_dev"]["metrics"].get("protected_redact_recall"),
        default=0.0,
    )
    best_same_backbone_dev = _metric_value(
        best_same_backbone["evaluations"]["upchieve_dev"].get("protected_redact_recall"),
        default=0.0,
    )

    fallback_result: dict[str, Any] | None = None
    if not args.skip_fallback and best_same_backbone_dev < baseline_dev_protected + 0.05:
        fallback_config = dict(best_same_backbone["config"])
        fallback_slug = slugify(
            f"distilroberta-context-{fallback_config['context_mode']}-l{fallback_config['max_length']}-lr-{fallback_config['lr']}"
        )
        fallback_training_dir = training_root / fallback_slug
        _train(
            model=args.fallback_model,
            train_file=mixed_train_file,
            dev_file=upchieve_dev_file,
            output_dir=fallback_training_dir,
            epochs=int(fallback_config["epochs"]),
            lr=float(fallback_config["lr"]),
            batch_size=args.candidate_batch_size,
            max_length=int(fallback_config["max_length"]),
            context_mode=str(fallback_config["context_mode"]),
        )
        fallback_evaluations: dict[str, Any] = {}
        for name, gold_file in (
            ("upchieve_dev", upchieve_dev_file),
            ("upchieve_test", upchieve_test_file),
            ("math_test", args.math_test_file),
        ):
            prediction_file = experiment.predictions_dir / f"{fallback_slug}_{name}.jsonl"
            _predict(
                model_path=fallback_training_dir,
                input_file=gold_file,
                output_file=prediction_file,
                batch_size=args.prediction_batch_size,
                max_length=int(fallback_config["max_length"]),
                context_mode=str(fallback_config["context_mode"]),
            )
            fallback_evaluations[name] = _evaluate_and_write(
                gold_file=gold_file,
                prediction_file=prediction_file,
                summary_path=fallback_training_dir / f"{name}_summary.json",
                report_path=fallback_training_dir / f"{name}_report.md",
            )
        fallback_result = {
            "label": fallback_slug,
            "model": args.fallback_model,
            "training_dir": str(fallback_training_dir),
            "config": fallback_config,
            "evaluations": fallback_evaluations,
        }
        candidates.append(fallback_result)

    candidates.sort(key=_config_sort_key, reverse=True)
    winner = candidates[0]

    baseline_test = baseline_outputs["upchieve_test"]["metrics"]
    winner_test = winner["evaluations"]["upchieve_test"]
    baseline_math = baseline_outputs["math_test"]["metrics"]
    winner_math = winner["evaluations"]["math_test"]
    protected_gain = _metric_value(winner_test.get("protected_redact_recall"), default=0.0) - _metric_value(
        baseline_test.get("protected_redact_recall"),
        default=0.0,
    )
    action_seed_gain = _metric_value(winner_test.get("action_seed_span_coverage"), default=0.0) - _metric_value(
        baseline_test.get("action_seed_span_coverage"),
        default=0.0,
    )
    math_recall_drop = _metric_value(baseline_math.get("recall"), default=0.0) - _metric_value(
        winner_math.get("recall"),
        default=0.0,
    )
    accepted = protected_gain >= 0.05 and action_seed_gain > 0.0 and math_recall_drop <= 0.01

    summary = {
        "baseline": baseline_outputs,
        "best_same_backbone": best_same_backbone,
        "fallback": fallback_result,
        "failed_configs": failed_configs,
        "candidates_ranked": candidates,
        "winner": winner,
        "decision": {
            "accepted": accepted,
            "protected_redact_recall_gain": protected_gain,
            "action_seed_span_coverage_gain": action_seed_gain,
            "math_recall_drop": math_recall_drop,
        },
    }
    _write_json(experiment.summary_path, summary)

    report_lines = [
        "# UpChieve Candidate Recall Adaptation",
        "",
        "## Baseline",
        "",
        f"- upchieve_dev protected_redact_recall: `{_fmt(baseline_outputs['upchieve_dev']['metrics'].get('protected_redact_recall'))}`",
        f"- upchieve_test protected_redact_recall: `{_fmt(baseline_outputs['upchieve_test']['metrics'].get('protected_redact_recall'))}`",
        f"- math_test recall: `{_fmt(baseline_outputs['math_test']['metrics'].get('recall'))}`",
        "",
        "## Best Same-Backbone",
        "",
        f"- config: `{best_same_backbone['label']}`",
        f"- upchieve_dev protected_redact_recall: `{_fmt(best_same_backbone['evaluations']['upchieve_dev'].get('protected_redact_recall'))}`",
        f"- upchieve_test protected_redact_recall: `{_fmt(best_same_backbone['evaluations']['upchieve_test'].get('protected_redact_recall'))}`",
        f"- math_test recall: `{_fmt(best_same_backbone['evaluations']['math_test'].get('recall'))}`",
        "",
    ]
    if fallback_result is not None:
        report_lines.extend(
            [
                "## Fallback",
                "",
                f"- config: `{fallback_result['label']}`",
                f"- upchieve_dev protected_redact_recall: `{_fmt(fallback_result['evaluations']['upchieve_dev'].get('protected_redact_recall'))}`",
                f"- upchieve_test protected_redact_recall: `{_fmt(fallback_result['evaluations']['upchieve_test'].get('protected_redact_recall'))}`",
                f"- math_test recall: `{_fmt(fallback_result['evaluations']['math_test'].get('recall'))}`",
                "",
            ]
        )
    report_lines.extend(
        [
            "## Winner",
            "",
            f"- label: `{winner['label']}`",
            f"- accepted: `{accepted}`",
            f"- protected_redact_recall_gain: `{protected_gain:.4f}`",
            f"- action_seed_span_coverage_gain: `{action_seed_gain:.4f}`",
            f"- math_recall_drop: `{math_recall_drop:.4f}`",
            "",
            "## Ranked Candidates",
            "",
        ]
    )
    for candidate in candidates:
        report_lines.append(
            f"- `{candidate['label']}` dev_protected=`{_fmt(candidate['evaluations']['upchieve_dev'].get('protected_redact_recall'))}` "
            f"test_protected=`{_fmt(candidate['evaluations']['upchieve_test'].get('protected_redact_recall'))}` "
            f"math_recall=`{_fmt(candidate['evaluations']['math_test'].get('recall'))}`"
        )
    experiment.report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    status = "keep" if accepted else "discard"
    _result_line(
        _metric_value(winner_test.get("protected_redact_recall"), default=0.0),
        status,
        (
            f"UpChieve candidate adaptation winner {winner['label']} reached protected_redact_recall "
            f"{_fmt(winner_test.get('protected_redact_recall'))} on proxy test with math recall "
            f"{_fmt(winner_math.get('recall'))} [{experiment.root.name}]"
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
