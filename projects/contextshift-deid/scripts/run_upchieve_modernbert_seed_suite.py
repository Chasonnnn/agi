from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_deferral as deferral_eval

from contextshift_deid.action_features import ACTION_INPUT_FORMAT_CHOICES, DEFAULT_ACTION_INPUT_FORMAT
from contextshift_deid.constants import RESULTS_HEADER
from contextshift_deid.data import load_jsonl
from contextshift_deid.direct_id_rules import apply_direct_id_overrides
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, ExperimentRunPaths, create_experiment_run, write_run_metadata
from contextshift_deid.metrics import compute_action_metrics

RESULTS_PATH = ROOT / "results.tsv"
SUITE_KIND = "upchieve_modernbert_seed_suite"
DEFAULT_MODEL = "ModernBERT-base"
DEFAULT_RUN_NAME = "upchieve-modernbert-v2-seed-suite"
DEFAULT_RUN_ROOT = EXPERIMENTS_DIR / "modernbert_recipe_tuning"
DEFAULT_SEEDS = (13, 42, 101)
DEFAULT_TARGET_REVIEW_RATES = (0.05, 0.10)
POLICY_SELECTION_METRIC = "selected_policy_10pct_direct_id"
SEMANTIC_ROLE_HEAD_MODE_CHOICES = ("none", "multitask")
SAMPLER_MODE_CHOICES = ("none", "subject_action_balanced")
DEFAULT_TRAIN_FILE = ROOT / "data/processed/action/train_mixed_upchieve_english_social_v2.jsonl"
DEFAULT_DEV_FILE = ROOT / "data/processed/action/upchieve_english_social_dev.jsonl"
DEFAULT_TEST_FILE = ROOT / "data/processed/action/upchieve_english_social_test.jsonl"
DEFAULT_LEGACY_MATH_FILE = ROOT / "data/processed/action/test.jsonl"
STABILITY_TARGET_REVIEW_RATE = 0.10
STABILITY_VARIANT = "no_rules"
STABILITY_REVIEW_RATE_STD_THRESHOLD = 0.0315
STABILITY_REVIEW_RATE_IMPROVEMENT_RATIO = 0.75
STABILITY_MACRO_F1_DROP_TOLERANCE = 0.005
HISTORICAL_MODEL_REFERENCES = {
    "ModernBERT-base": {
        "label": "ModernBERT-base v2 (historical, uncontrolled seed)",
        "path": ROOT / "artifacts/experiments/20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384/summary.json",
        "note": "The historical ModernBERT v2 result came from a single earlier run before this seeded suite existed. It is kept as reference context only, because the original training path did not control all randomness before model initialization.",
    },
    "roberta-base": {
        "label": "roberta-base v2 (historical, uncontrolled seed)",
        "path": ROOT / "artifacts/experiments/20260314_225018_upchieve-english-social-mixed-roberta-base-v2-b4-l384-fresh/summary.json",
        "note": "The historical roberta-base v2 result came from a single earlier run before this seeded suite existed. It is kept as reference context only, because the original training path did not control all randomness before model initialization.",
    },
    "distilroberta-base": {
        "label": "distilroberta-base v2 (historical, uncontrolled seed)",
        "path": ROOT / "artifacts/experiments/20260314_225845_upchieve-english-social-mixed-distilroberta-v2-b4-l384-fresh/summary.json",
        "note": "The historical distilroberta-base v2 result came from a single earlier run before this seeded suite existed. It is kept as reference context only, because the original training path did not control all randomness before model initialization.",
    },
}
HISTORICAL_BACKBONE_SUMMARIES = (
    {
        "label": "roberta-base v1 from math",
        "path": ROOT / "artifacts/experiments/20260314_193820_upchieve-english-social-mixed-roberta-base-v1-from-math-b4-l384/summary.json",
    },
    {
        "label": "roberta-base v1 fresh",
        "path": ROOT / "artifacts/experiments/20260314_201224_upchieve-english-social-mixed-roberta-base-v1-b4-l384-fresh/summary.json",
    },
    {
        "label": "roberta-base v2",
        "path": ROOT / "artifacts/experiments/20260314_225018_upchieve-english-social-mixed-roberta-base-v2-b4-l384-fresh/summary.json",
    },
    {
        "label": "distilroberta-base v1 from math",
        "path": ROOT / "artifacts/experiments/20260314_193849_upchieve-english-social-mixed-distilroberta-v1-from-math-b4-l384/summary.json",
    },
    {
        "label": "distilroberta-base v1 fresh",
        "path": ROOT / "artifacts/experiments/20260314_201314_upchieve-english-social-mixed-distilroberta-v1-b4-l384-fresh/summary.json",
    },
    {
        "label": "distilroberta-base v2",
        "path": ROOT / "artifacts/experiments/20260314_225845_upchieve-english-social-mixed-distilroberta-v2-b4-l384-fresh/summary.json",
    },
)


@dataclass(frozen=True)
class SeedRunLayout:
    root: Path
    metadata_path: Path
    summary_path: Path
    report_path: Path
    training_dir: Path
    predictions_dir: Path
    base_dev_predictions: Path
    base_test_predictions: Path
    base_math_predictions: Path
    direct_id_dev_predictions: Path
    direct_id_test_predictions: Path
    direct_id_math_predictions: Path
    no_rules_dir: Path
    direct_id_dir: Path


@dataclass(frozen=True)
class VariantRunLayout:
    root: Path
    metadata_path: Path
    summary_path: Path
    report_path: Path
    predictions_dir: Path
    sweep_path: Path


def _parse_target_review_rates(raw: str) -> tuple[float, ...]:
    return tuple(deferral_eval._parse_target_review_rates(raw))


def _canonical_model_name(model_name_or_path: str | Path) -> str:
    model_text = str(model_name_or_path)
    candidate = Path(model_text)
    if candidate.exists():
        return candidate.name
    return model_text.split("/")[-1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _run(command: list[str], *, capture_json: bool = False) -> dict[str, Any] | None:
    print("+", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=capture_json,
            text=capture_json,
        )
    except subprocess.CalledProcessError as exc:
        if capture_json:
            if exc.stdout:
                print(exc.stdout)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
        raise
    if not capture_json:
        return None
    stdout = completed.stdout.strip()
    if stdout:
        print(stdout)
    if not stdout:
        return {}
    return _extract_last_json_object(stdout)


def _extract_last_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate_starts = [index for index, character in enumerate(text) if character == "{"]  # noqa: C416
    for start in reversed(candidate_starts):
        candidate = text[start:].strip()
        try:
            payload, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if candidate[end:].strip():
            continue
        if isinstance(payload, dict):
            return payload
    raise SystemExit("Expected a trailing JSON object in command output, but none was found.")


def _ensure_results_header() -> None:
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(RESULTS_HEADER, encoding="utf-8")


def _append_result(metric: float, status: str, description: str, *, stage: str = "action") -> None:
    _ensure_results_header()
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"working\t{stage}\t{metric:.6f}\t{status}\t{description}\n")


def _find_existing_suite_root(run_name: str, *, run_root: Path) -> ExperimentRunPaths | None:
    candidates: list[Path] = []
    for metadata_path in sorted(run_root.glob("*/metadata.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("suite_kind") != SUITE_KIND:
            continue
        if payload.get("run_name") != run_name:
            continue
        candidates.append(metadata_path.parent)
    if not candidates:
        return None
    root = sorted(candidates)[-1]
    return ExperimentRunPaths(
        root=root,
        predictions_dir=root / "predictions",
        metadata_path=root / "metadata.json",
        summary_path=root / "summary.json",
        report_path=root / "report.md",
    )


def _ensure_suite_root(run_name: str, *, run_root: Path) -> ExperimentRunPaths:
    existing = _find_existing_suite_root(run_name, run_root=run_root)
    if existing is not None:
        return existing
    return create_experiment_run(run_name, root_dir=run_root)


def _suite_paths_from_root(root: Path) -> ExperimentRunPaths:
    return ExperimentRunPaths(
        root=root,
        predictions_dir=root / "predictions",
        metadata_path=root / "metadata.json",
        summary_path=root / "summary.json",
        report_path=root / "report.md",
    )


def _seed_run_paths_from_suite_root(suite_root: Path) -> dict[str, Path]:
    metadata_path = suite_root / "metadata.json"
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        seed_runs = metadata.get("seed_runs")
        if isinstance(seed_runs, Mapping):
            return {str(seed): Path(path) for seed, path in seed_runs.items()}

    summary_path = suite_root / "summary.json"
    if not summary_path.exists():
        return {}
    summary = _load_json(summary_path)
    seed_run_paths = summary.get("seed_run_paths")
    if not isinstance(seed_run_paths, list):
        return {}

    resolved: dict[str, Path] = {}
    for raw_path in seed_run_paths:
        path = Path(raw_path)
        seed_token = path.name.rsplit("-seed-", maxsplit=1)
        if len(seed_token) != 2 or not seed_token[1].isdigit():
            continue
        resolved[seed_token[1]] = path
    return resolved


def _seed_layout(root: Path) -> SeedRunLayout:
    predictions_dir = root / "predictions"
    return SeedRunLayout(
        root=root,
        metadata_path=root / "metadata.json",
        summary_path=root / "summary.json",
        report_path=root / "report.md",
        training_dir=root / "training" / "model",
        predictions_dir=predictions_dir,
        base_dev_predictions=predictions_dir / "dev_base_predictions.jsonl",
        base_test_predictions=predictions_dir / "test_base_predictions.jsonl",
        base_math_predictions=predictions_dir / "legacy_math_base_predictions.jsonl",
        direct_id_dev_predictions=predictions_dir / "dev_direct_id_predictions.jsonl",
        direct_id_test_predictions=predictions_dir / "test_direct_id_predictions.jsonl",
        direct_id_math_predictions=predictions_dir / "legacy_math_direct_id_predictions.jsonl",
        no_rules_dir=root / "deferral_no_rules",
        direct_id_dir=root / "deferral_direct_id",
    )


def _variant_layout(root: Path) -> VariantRunLayout:
    predictions_dir = root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return VariantRunLayout(
        root=root,
        metadata_path=root / "metadata.json",
        summary_path=root / "summary.json",
        report_path=root / "report.md",
        predictions_dir=predictions_dir,
        sweep_path=root / "sweep_results.json",
    )


def _remove_if_force(path: Path, *, force: bool) -> None:
    if not force or not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _checkpoint_complete(training_dir: Path) -> bool:
    if not training_dir.exists():
        return False
    if not (training_dir / "config.json").exists():
        return False
    if not ((training_dir / "model.safetensors").exists() or (training_dir / "pytorch_model.bin").exists()):
        return False
    return (training_dir / "metrics.json").exists() and (training_dir / "metadata.json").exists()


def _prediction_complete(path: Path, *, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = load_jsonl(path)
    except Exception:
        return False
    return len(rows) == expected_count


def _variant_complete(path: Path) -> bool:
    if not path.exists():
        return False
    summary_path = path / "summary.json"
    report_path = path / "report.md"
    metadata_path = path / "metadata.json"
    if not summary_path.exists() or not report_path.exists() or not metadata_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    selected_targets = summary.get("selected_targets", [])
    if not selected_targets:
        return False
    for target in selected_targets:
        output_file = target.get("output_file")
        if not output_file:
            return False
        if not Path(output_file).exists():
            return False
    return True


def _evaluate_action_metrics(gold_file: Path, prediction_file: Path) -> dict[str, Any]:
    gold_rows = load_jsonl(gold_file)
    prediction_rows = load_jsonl(prediction_file)
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    merged: list[dict[str, Any]] = []
    for row in gold_rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row_id} in {prediction_file}")
        merged.append(
            {
                "id": row_id,
                "subject": row.get("subject", "unknown"),
                "eval_slice": row.get("eval_slice"),
                "gold_action": row["action_label"],
                "predicted_action": prediction["predicted_action"],
                "speaker_role": row.get("speaker_role"),
                "entity_type": row.get("entity_type"),
                "semantic_role": row.get("semantic_role"),
                "cost": prediction.get("cost"),
                "latency_ms": prediction.get("latency_ms"),
            }
        )
    return compute_action_metrics(merged)


def _curricular_accuracy(metrics: Mapping[str, Any]) -> float | None:
    by_semantic_role = metrics.get("by_semantic_role")
    if not isinstance(by_semantic_role, Mapping):
        return None
    curricular = by_semantic_role.get("CURRICULAR")
    if not isinstance(curricular, Mapping):
        return None
    value = curricular.get("accuracy")
    return None if value is None else float(value)


def _read_training_metrics(training_dir: Path) -> dict[str, Any]:
    metrics = _load_json(training_dir / "metrics.json")
    metadata = _load_json(training_dir / "metadata.json")
    return {
        "metrics": metrics,
        "metadata": metadata,
    }


def _historical_reference(model_name: str) -> dict[str, Any] | None:
    reference = HISTORICAL_MODEL_REFERENCES.get(model_name)
    if reference is None:
        return None
    summary = _load_json(reference["path"])
    target = next(
        target
        for target in summary["selected_targets"]
        if abs(float(target["target_review_rate"]) - 0.10) <= 1e-9
    )
    return {
        "label": reference["label"],
        "path": str(reference["path"]),
        "note": reference["note"],
        "macro_f1": float(target["evaluation"]["metrics"]["macro_f1"]),
        "review_rate": float(target["evaluation"]["metrics"]["review_rate"]),
    }


def _historical_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for reference in HISTORICAL_MODEL_REFERENCES.values():
        reference_path = str(reference["path"])
        if reference_path in seen_paths:
            continue
        summary = _load_json(reference["path"])
        selected_target = next(
            target
            for target in summary["selected_targets"]
            if abs(float(target["target_review_rate"]) - 0.10) <= 1e-9
        )
        rows.append(
            {
                "label": reference["label"],
                "summary_file": str(reference["path"]),
                "macro_f1": float(selected_target["evaluation"]["metrics"]["macro_f1"]),
                "review_rate": float(selected_target["evaluation"]["metrics"]["review_rate"]),
            }
        )
        seen_paths.add(reference_path)
    for item in HISTORICAL_BACKBONE_SUMMARIES:
        item_path = str(item["path"])
        if item_path in seen_paths:
            continue
        summary = _load_json(item["path"])
        selected_target = next(
            target
            for target in summary["selected_targets"]
            if abs(float(target["target_review_rate"]) - 0.10) <= 1e-9
        )
        rows.append(
            {
                "label": item["label"],
                "summary_file": str(item["path"]),
                "macro_f1": float(selected_target["evaluation"]["metrics"]["macro_f1"]),
                "review_rate": float(selected_target["evaluation"]["metrics"]["review_rate"]),
            }
        )
        seen_paths.add(item_path)
    return rows


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _metric_mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _resolve_seed_run(
    suite_root: ExperimentRunPaths,
    suite_metadata: dict[str, Any],
    *,
    run_name: str,
    seed: int,
    run_root: Path,
) -> SeedRunLayout:
    seed_runs = dict(suite_metadata.get("seed_runs") or {})
    existing_path = seed_runs.get(str(seed))
    if existing_path:
        return _seed_layout(Path(existing_path))

    experiment = create_experiment_run(f"{run_name}-seed-{seed}", root_dir=run_root)
    seed_runs[str(seed)] = str(experiment.root)
    suite_metadata["seed_runs"] = seed_runs
    write_run_metadata(suite_root.metadata_path, suite_metadata)
    return _seed_layout(experiment.root)


def _write_seed_metadata(layout: SeedRunLayout, payload: Mapping[str, Any]) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.predictions_dir.mkdir(parents=True, exist_ok=True)
    _write_json(layout.metadata_path, payload)


def _write_variant_artifacts(
    layout: VariantRunLayout,
    *,
    run_name: str,
    model_name: str,
    calibration_gold: Path,
    calibration_predictions: Path,
    eval_gold: Path,
    eval_predictions: Path,
    target_review_rates: tuple[float, ...],
    fit_temperature: bool,
    extra_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_records = deferral_eval._merge_action_rows(
        load_jsonl(calibration_gold),
        load_jsonl(calibration_predictions),
    )
    eval_records = deferral_eval._merge_action_rows(
        load_jsonl(eval_gold),
        load_jsonl(eval_predictions),
    )
    calibration_base_metrics = deferral_eval._base_metrics(calibration_records)
    eval_base_metrics = deferral_eval._base_metrics(eval_records)
    temperature = 1.0
    if fit_temperature:
        temperature = deferral_eval.fit_temperature(calibration_records)

    calibration_feature_rows = deferral_eval._feature_rows(calibration_records, temperature=temperature)
    strategy_parameter_sets: dict[str, list[dict[str, float]]] = {
        "confidence": deferral_eval._generate_simple_candidates(calibration_feature_rows, "confidence"),
        "margin": deferral_eval._generate_simple_candidates(calibration_feature_rows, "margin"),
        "entropy": deferral_eval._generate_simple_candidates(calibration_feature_rows, "entropy"),
        "review_probability": deferral_eval._generate_simple_candidates(calibration_feature_rows, "review_probability"),
        "composite": deferral_eval._generate_composite_candidates(calibration_feature_rows),
        "asymmetric_confidence": deferral_eval._generate_asymmetric_candidates(calibration_feature_rows, "asymmetric_confidence"),
        "asymmetric_margin": deferral_eval._generate_asymmetric_candidates(calibration_feature_rows, "asymmetric_margin"),
    }

    calibration_results: list[dict[str, Any]] = [
        {
            "strategy": "none",
            "parameters": {},
            "temperature": temperature,
            "metrics": deferral_eval.compute_deferral_metrics(deferral_eval._base_review_records(calibration_records)),
        }
    ]
    for strategy, parameter_sets in strategy_parameter_sets.items():
        for parameter_set in parameter_sets:
            calibration_results.append(
                deferral_eval._evaluate_policy_result(
                    calibration_records,
                    strategy=strategy,
                    parameters=parameter_set,
                    temperature=temperature,
                )
            )

    selected_prediction_files: list[str] = []
    selected_targets: list[dict[str, Any]] = []
    for target_review_rate in target_review_rates:
        calibration_selection = deferral_eval._select_best_for_target(
            calibration_results,
            target_review_rate=target_review_rate,
        )
        if calibration_selection is None:
            continue
        eval_remapped = deferral_eval._evaluate_selected_policy(
            eval_records,
            strategy=calibration_selection["strategy"],
            parameters=calibration_selection["parameters"],
            temperature=temperature,
        )
        eval_metrics = deferral_eval.compute_deferral_metrics(eval_remapped)
        output_name = f"eval_target_{int(target_review_rate * 1000):03d}_predictions.jsonl"
        output_path = layout.predictions_dir / output_name
        deferral_eval._write_jsonl(output_path, eval_remapped)
        selected_prediction_files.append(str(output_path))
        selected_targets.append(
            {
                "target_review_rate": target_review_rate,
                "calibration": calibration_selection,
                "evaluation": {
                    "strategy": calibration_selection["strategy"],
                    "parameters": calibration_selection["parameters"],
                    "temperature": temperature,
                    "metrics": eval_metrics,
                },
                "output_name": output_name,
                "output_file": str(output_path),
            }
        )

    top_review_errors = deferral_eval._report_top_review_errors(eval_records, temperature=temperature)
    metadata = {
        "suite_component": "deferral_variant",
        "run_name": run_name,
        "model_name": model_name,
        "calibration_gold": str(calibration_gold),
        "calibration_predictions": str(calibration_predictions),
        "eval_gold": str(eval_gold),
        "eval_predictions": str(eval_predictions),
        "target_review_rates": list(target_review_rates),
        "fit_temperature": fit_temperature,
        "temperature": temperature,
        "strategy_counts": {strategy: len(parameter_sets) for strategy, parameter_sets in strategy_parameter_sets.items()},
        **extra_metadata,
    }
    summary = {
        "model_name": model_name,
        "temperature": temperature,
        "calibration_base_metrics": calibration_base_metrics,
        "eval_base_metrics": eval_base_metrics,
        "selected_targets": selected_targets,
        "selected_prediction_files": selected_prediction_files,
        "sweep_result_count": len(calibration_results),
    }
    _write_json(layout.metadata_path, metadata)
    _write_json(layout.summary_path, summary)
    layout.sweep_path.write_text(json.dumps(calibration_results, indent=2), encoding="utf-8")
    layout.report_path.write_text(
        deferral_eval._build_report(
            model_name=model_name,
            calibration_base_metrics=calibration_base_metrics,
            eval_base_metrics=eval_base_metrics,
            temperature=temperature,
            target_summaries=selected_targets,
            top_review_errors=top_review_errors,
        ),
        encoding="utf-8",
    )
    return summary


def _selected_target(summary: Mapping[str, Any], *, target_review_rate: float) -> Mapping[str, Any]:
    for target in summary.get("selected_targets", []):
        if abs(float(target["target_review_rate"]) - target_review_rate) <= 1e-9:
            return target
    raise SystemExit(f"Missing selected target {target_review_rate:.2f} in summary")


def _find_selected_target(summary: Mapping[str, Any], *, target_review_rate: float) -> Mapping[str, Any] | None:
    for target in summary.get("selected_targets", []):
        if abs(float(target["target_review_rate"]) - target_review_rate) <= 1e-9:
            return target
    return None


def _build_seed_report(
    *,
    model_name: str,
    seed: int,
    training_dir: Path,
    no_rules_summary: Mapping[str, Any],
    direct_id_summary: Mapping[str, Any],
    legacy_math_metrics: Mapping[str, Any],
    direct_id_override_counts: Mapping[str, Any],
    direct_id_test_spotcheck: list[Mapping[str, Any]],
) -> str:
    lines = [
        f"# {model_name} Seed Run: {seed}",
        "",
        f"- Training dir: `{training_dir}`",
        "",
        "## Selected 10% Results",
        "",
        "| variant | strategy | temperature | parameters | calibration review | eval review | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, summary, math_key in (
        ("no_rules", no_rules_summary, "no_rules"),
        ("direct_id", direct_id_summary, "direct_id"),
    ):
        target = _selected_target(summary, target_review_rate=0.10)
        calibration_metrics = target["calibration"]["metrics"]
        eval_metrics = target["evaluation"]["metrics"]
        lines.append(
            "| {label} | {strategy} | {temperature} | `{parameters}` | {calibration_review} | {eval_review} | {macro_f1} | {accuracy} | {redact_recall} | {curricular_accuracy} | {review_coverage} | {protected_redact_rate} | {redact_leak_rate} | {legacy_math_recall} |".format(
                label=label,
                strategy=target["calibration"]["strategy"],
                temperature=_format_metric(float(target["calibration"]["temperature"])),
                parameters=json.dumps(target["calibration"]["parameters"], sort_keys=True),
                calibration_review=_format_percent(float(calibration_metrics["review_rate"])),
                eval_review=_format_percent(float(eval_metrics["review_rate"])),
                macro_f1=_format_metric(float(eval_metrics["macro_f1"])),
                accuracy=_format_metric(float(eval_metrics["accuracy"])),
                redact_recall=_format_metric(float(eval_metrics["redact_recall"])),
                curricular_accuracy=_format_metric(_curricular_accuracy(eval_metrics)),
                review_coverage=_format_percent(eval_metrics.get("gold_review_coverage")),
                protected_redact_rate=_format_percent(eval_metrics.get("protected_redact_rate")),
                redact_leak_rate=_format_percent(eval_metrics.get("redact_leak_rate")),
                legacy_math_recall=_format_metric(float(legacy_math_metrics[math_key]["redact_recall"])),
            )
        )
    lines.extend(
        [
            "",
            "## Direct-ID Overrides",
            "",
            "| split | override_count | by_reason |",
            "| --- | --- | --- |",
        ]
    )
    for split in ("dev", "test", "math"):
        override_summary = direct_id_override_counts[split]
        lines.append(
            f"| {split} | {override_summary['override_count']} | `{json.dumps(override_summary['by_reason'], sort_keys=True)}` |"
        )
    if direct_id_test_spotcheck:
        lines.extend(
            [
                "",
                "## Direct-ID Test Spot-Check",
                "",
                "| id | span_text | entity_type | gold_action | base_predicted_action | patched_predicted_action | reason | source |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in direct_id_test_spotcheck:
            span_text = str(row.get("span_text", "")).replace("|", "\\|")
            lines.append(
                f"| {row['id']} | {span_text} | {row.get('entity_type') or ''} | {row['gold_action']} | "
                f"{row['base_predicted_action']} | {row['patched_predicted_action']} | {row['reason']} | {row['source']} |"
            )
    return "\n".join(lines) + "\n"


def _build_suite_report(
    *,
    model_name: str,
    run_name: str,
    run_root: Path,
    seeds: tuple[int, ...],
    target_review_rates: tuple[float, ...],
    action_input_format: str,
    semantic_role_head_mode: str,
    semantic_role_loss_weight: float,
    sampler_mode: str,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    selection_metric: str,
    warmup_ratio: float,
    weight_decay: float,
    fit_temperature: bool,
    recompute_deferral_only: bool,
    source_suite_root: Path | None,
    suite_rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    base_vs_selected_rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
    historical_reference: Mapping[str, Any] | None,
    anchor_gap: float | None,
    selected_10_no_rules_stability: Mapping[str, Any] | None,
    source_suite_comparison: Mapping[str, Any] | None,
) -> str:
    lines = [
        f"# {model_name} Seed Suite",
        "",
        f"- Run name: `{run_name}`",
        f"- Run root: `{run_root}`",
        f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- Target review rates: `{', '.join(f'{rate:.2f}' for rate in target_review_rates)}`",
        f"- Action input format: `{action_input_format}`",
        f"- Semantic-role head mode: `{semantic_role_head_mode}`",
        f"- Semantic-role loss weight: `{semantic_role_loss_weight}`",
        f"- Sampler mode: `{sampler_mode}`",
        f"- Gradient accumulation steps: `{gradient_accumulation_steps}`",
        f"- Gradient checkpointing: `{gradient_checkpointing}`",
        f"- Checkpoint selection metric: `{selection_metric}`",
        f"- Warmup ratio: `{warmup_ratio}`",
        f"- Weight decay: `{weight_decay}`",
        f"- Fit temperature: `{fit_temperature}`",
        f"- Recompute deferral only: `{recompute_deferral_only}`",
        f"- Source suite root: `{source_suite_root}`" if source_suite_root is not None else "- Source suite root: `n/a`",
        "## Seed-Level Selected Policies",
        "",
        "| seed | variant | target | strategy | temperature | parameters | dev_review | test_review | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    if historical_reference is not None and anchor_gap is not None:
        insert_at = lines.index("## Seed-Level Selected Policies")
        lines[insert_at:insert_at] = [
            f"- Seed 42 anchor gap vs historical 10% test macro F1 {historical_reference['macro_f1']:.4f}: {anchor_gap:+.4f} (reference only — see note)",
            "",
            f"> {historical_reference['note']}",
            "",
        ]
    for row in suite_rows:
        lines.append(
            "| {seed} | {variant} | {target:.2f} | {strategy} | {temperature} | `{parameters}` | {dev_review} | {test_review} | {macro_f1} | {accuracy} | {redact_recall} | {curricular_accuracy} | {gold_review_coverage} | {protected_redact_rate} | {redact_leak_rate} | {legacy_math_redact_recall} |".format(
                seed=row["seed"],
                variant=row["variant"],
                target=row["target_review_rate"],
                strategy=row["strategy"],
                temperature=_format_metric(row["temperature"]),
                parameters=json.dumps(row["parameters"], sort_keys=True),
                dev_review=_format_percent(row["dev_review_rate"]),
                test_review=_format_percent(row["test_review_rate"]),
                macro_f1=_format_metric(row["macro_f1"]),
                accuracy=_format_metric(row["accuracy"]),
                redact_recall=_format_metric(row["redact_recall"]),
                curricular_accuracy=_format_metric(row["curricular_accuracy"]),
                gold_review_coverage=_format_percent(row["gold_review_coverage"]),
                protected_redact_rate=_format_percent(row["protected_redact_rate"]),
                redact_leak_rate=_format_percent(row["redact_leak_rate"]),
                legacy_math_redact_recall=_format_metric(row["legacy_math_redact_recall"]),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Mean ± Std",
            "",
            "| variant | target | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | review_rate | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {variant} | {target:.2f} | {macro_f1} | {accuracy} | {redact_recall} | {curricular_accuracy} | {gold_review_coverage} | {review_rate} | {protected_redact_rate} | {redact_leak_rate} | {legacy_math_redact_recall} |".format(
                variant=row["variant"],
                target=row["target_review_rate"],
                macro_f1=row["macro_f1_text"],
                accuracy=row["accuracy_text"],
                redact_recall=row["redact_recall_text"],
                curricular_accuracy=row["curricular_accuracy_text"],
                gold_review_coverage=row["gold_review_coverage_text"],
                review_rate=row["review_rate_text"],
                protected_redact_rate=row["protected_redact_rate_text"],
                redact_leak_rate=row["redact_leak_rate_text"],
                legacy_math_redact_recall=row["legacy_math_redact_recall_text"],
            )
        )
    lines.extend(
        [
            "",
            "## Selected 10% No-Rules Stability",
            "",
        ]
    )
    if selected_10_no_rules_stability is None:
        lines.append("No selected 10% no-rules stability summary was available.")
    else:
        lines.extend(
            [
                "| metric | value |",
                "| --- | --- |",
                f"| unique_strategy_count | {selected_10_no_rules_stability['unique_strategy_count']} |",
                f"| strategy_labels | `{json.dumps(selected_10_no_rules_stability['strategy_labels'])}` |",
                f"| test_review_rate_mean | {_format_percent(float(selected_10_no_rules_stability['test_review_rate_mean']))} |",
                f"| test_review_rate_stdev | {_format_metric(float(selected_10_no_rules_stability['test_review_rate_stdev']))} |",
                f"| macro_f1_mean | {_format_metric(float(selected_10_no_rules_stability['macro_f1_mean']))} |",
                f"| macro_f1_stdev | {_format_metric(float(selected_10_no_rules_stability['macro_f1_stdev']))} |",
                f"| all_legacy_math_redact_recall_one | `{bool(selected_10_no_rules_stability['all_legacy_math_redact_recall_one'])}` |",
            ]
        )
    if source_suite_comparison is not None:
        lines.extend(
            [
                "",
                "## Source-Suite Comparison",
                "",
                "| metric | value |",
                "| --- | --- |",
                f"| baseline_unique_strategy_count | {source_suite_comparison['baseline']['unique_strategy_count']} |",
                f"| current_unique_strategy_count | {source_suite_comparison['current']['unique_strategy_count']} |",
                f"| baseline_test_review_rate_stdev | {_format_metric(float(source_suite_comparison['baseline']['test_review_rate_stdev']))} |",
                f"| current_test_review_rate_stdev | {_format_metric(float(source_suite_comparison['current']['test_review_rate_stdev']))} |",
                f"| delta_test_review_rate_stdev | {_format_metric(float(source_suite_comparison['delta_test_review_rate_stdev']))} |",
                f"| baseline_macro_f1_mean | {_format_metric(float(source_suite_comparison['baseline']['macro_f1_mean']))} |",
                f"| current_macro_f1_mean | {_format_metric(float(source_suite_comparison['current']['macro_f1_mean']))} |",
                f"| delta_macro_f1_mean | {_format_metric(float(source_suite_comparison['delta_macro_f1_mean']))} |",
                f"| strategy_collapsed | `{bool(source_suite_comparison['strategy_collapsed'])}` |",
                f"| review_rate_stdev_improved | `{bool(source_suite_comparison['review_rate_stdev_improved'])}` |",
                f"| quality_preserved | `{bool(source_suite_comparison['quality_preserved'])}` |",
                f"| safety_preserved | `{bool(source_suite_comparison['safety_preserved'])}` |",
                f"| carry_fit_temperature_forward | `{bool(source_suite_comparison['carry_fit_temperature_forward'])}` |",
            ]
        )
    lines.extend(
        [
            "",
            "## Base Classifier vs Selected Policy",
            "",
            "| variant | target | base_macro_f1 | selected_macro_f1 | delta_macro_f1 | base_accuracy | selected_accuracy | delta_accuracy | base_redact_recall | selected_redact_recall | delta_redact_recall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in base_vs_selected_rows:
        lines.append(
            "| {variant} | {target:.2f} | {base_macro_f1} | {selected_macro_f1} | {delta_macro_f1} | {base_accuracy} | {selected_accuracy} | {delta_accuracy} | {base_redact_recall} | {selected_redact_recall} | {delta_redact_recall} |".format(
                variant=row["variant"],
                target=row["target_review_rate"],
                base_macro_f1=row["base_macro_f1_text"],
                selected_macro_f1=row["selected_macro_f1_text"],
                delta_macro_f1=row["delta_macro_f1_text"],
                base_accuracy=row["base_accuracy_text"],
                selected_accuracy=row["selected_accuracy_text"],
                delta_accuracy=row["delta_accuracy_text"],
                base_redact_recall=row["base_redact_recall_text"],
                selected_redact_recall=row["selected_redact_recall_text"],
                delta_redact_recall=row["delta_redact_recall_text"],
            )
        )
    lines.extend(
        [
            "",
            "## Historical Frozen Backbone Results (single-run, uncontrolled seed — reference context only)",
            "",
            "| backbone | selected 10% macro_f1 | selected 10% review_rate | source |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in historical_rows:
        lines.append(
            f"| {row['label']} | {_format_metric(row['macro_f1'])} | {_format_percent(row['review_rate'])} | `{row['summary_file']}` |"
        )
    return "\n".join(lines) + "\n"


def _format_mean_std(values: list[float]) -> str:
    avg, std = _metric_mean_std(values)
    return f"{avg:.4f} ± {std:.4f}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    return _metric_mean_std(values)


def _policy_stability_rows(
    suite_rows: list[dict[str, Any]],
    *,
    variant: str = STABILITY_VARIANT,
    target_review_rate: float = STABILITY_TARGET_REVIEW_RATE,
) -> list[dict[str, Any]]:
    return [
        row
        for row in suite_rows
        if row["variant"] == variant and abs(row["target_review_rate"] - target_review_rate) <= 1e-9
    ]


def _policy_stability_summary(
    suite_rows: list[dict[str, Any]],
    *,
    variant: str = STABILITY_VARIANT,
    target_review_rate: float = STABILITY_TARGET_REVIEW_RATE,
) -> dict[str, Any] | None:
    rows = _policy_stability_rows(suite_rows, variant=variant, target_review_rate=target_review_rate)
    if not rows:
        return None

    macro_f1_mean, macro_f1_stdev = _mean_std([row["macro_f1"] for row in rows])
    review_rate_mean, review_rate_stdev = _mean_std([row["test_review_rate"] for row in rows])
    strategy_labels = sorted({str(row["strategy"]) for row in rows})
    return {
        "variant": variant,
        "target_review_rate": target_review_rate,
        "seed_count": len(rows),
        "strategy_labels": strategy_labels,
        "unique_strategy_count": len(strategy_labels),
        "test_review_rate_mean": review_rate_mean,
        "test_review_rate_stdev": review_rate_stdev,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_stdev": macro_f1_stdev,
        "all_legacy_math_redact_recall_one": all(
            abs(float(row["legacy_math_redact_recall"]) - 1.0) <= 1e-9 for row in rows
        ),
    }


def _policy_stability_summary_from_suite(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    existing = summary.get("selected_10_no_rules_stability")
    if isinstance(existing, Mapping):
        return dict(existing)
    seed_rows = summary.get("seed_rows")
    if not isinstance(seed_rows, list):
        return None
    normalized_rows = [row for row in seed_rows if isinstance(row, Mapping)]
    return _policy_stability_summary(normalized_rows)


def _temperature_gate_decision(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    current_macro = float(current["macro_f1_mean"])
    baseline_macro = float(baseline["macro_f1_mean"])
    current_strategy_count = int(current["unique_strategy_count"])
    baseline_strategy_count = int(baseline["unique_strategy_count"])
    current_review_std = float(current["test_review_rate_stdev"])
    baseline_review_std = float(baseline["test_review_rate_stdev"])
    macro_drop = current_macro - baseline_macro
    strategy_collapsed = current_strategy_count < baseline_strategy_count
    review_std_improved = (
        current_review_std <= baseline_review_std * STABILITY_REVIEW_RATE_IMPROVEMENT_RATIO
        or current_review_std <= STABILITY_REVIEW_RATE_STD_THRESHOLD
    )
    mean_improved = current_macro > baseline_macro
    quality_preserved = macro_drop >= -STABILITY_MACRO_F1_DROP_TOLERANCE
    safety_preserved = bool(current.get("all_legacy_math_redact_recall_one"))
    carry_forward = mean_improved or (
        (strategy_collapsed or review_std_improved) and quality_preserved and safety_preserved
    )
    return {
        "baseline": dict(baseline),
        "current": dict(current),
        "delta_macro_f1_mean": macro_drop,
        "delta_test_review_rate_stdev": current_review_std - baseline_review_std,
        "delta_unique_strategy_count": current_strategy_count - baseline_strategy_count,
        "review_rate_stdev_threshold": STABILITY_REVIEW_RATE_STD_THRESHOLD,
        "review_rate_stdev_ratio_threshold": STABILITY_REVIEW_RATE_IMPROVEMENT_RATIO,
        "macro_f1_drop_tolerance": STABILITY_MACRO_F1_DROP_TOLERANCE,
        "strategy_collapsed": strategy_collapsed,
        "review_rate_stdev_improved": review_std_improved,
        "quality_preserved": quality_preserved,
        "safety_preserved": safety_preserved,
        "carry_fit_temperature_forward": carry_forward,
    }


def _results_logged_signature(
    *,
    seeds: tuple[int, ...],
    target_review_rates: tuple[float, ...],
    action_input_format: str,
    semantic_role_head_mode: str,
    semantic_role_loss_weight: float,
    sampler_mode: str,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    fit_temperature: bool,
    selection_metric: str,
    warmup_ratio: float,
    weight_decay: float,
    recompute_deferral_only: bool,
) -> dict[str, Any]:
    return {
        "seeds": list(seeds),
        "target_review_rates": list(target_review_rates),
        "action_input_format": action_input_format,
        "semantic_role_head_mode": semantic_role_head_mode,
        "semantic_role_loss_weight": semantic_role_loss_weight,
        "sampler_mode": sampler_mode,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": gradient_checkpointing,
        "fit_temperature": fit_temperature,
        "selection_metric": selection_metric,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "recompute_deferral_only": recompute_deferral_only,
    }


def _collect_direct_id_spotcheck(
    *,
    gold_file: Path,
    base_prediction_file: Path,
    patched_prediction_file: Path,
    limit: int = 10,
) -> list[dict[str, Any]]:
    gold_rows = {str(row["id"]): row for row in load_jsonl(gold_file)}
    base_rows = {str(row["id"]): row for row in load_jsonl(base_prediction_file)}
    patched_rows = {str(row["id"]): row for row in load_jsonl(patched_prediction_file)}
    spotcheck: list[dict[str, Any]] = []
    for row_id, patched_row in patched_rows.items():
        override = patched_row.get("direct_id_override")
        if not isinstance(override, Mapping):
            continue
        gold_row = gold_rows[row_id]
        base_row = base_rows[row_id]
        spotcheck.append(
            {
                "id": row_id,
                "span_text": gold_row.get("span_text", ""),
                "entity_type": gold_row.get("entity_type"),
                "gold_action": gold_row.get("action_label"),
                "base_predicted_action": base_row.get("predicted_action"),
                "patched_predicted_action": patched_row.get("predicted_action"),
                "reason": override.get("reason"),
                "source": override.get("source"),
            }
        )
        if len(spotcheck) >= limit:
            break
    return spotcheck


def _run_seed(
    *,
    seed: int,
    seed_layout: SeedRunLayout,
    model_display_name: str,
    model: str,
    train_file: Path,
    dev_file: Path,
    test_file: Path,
    legacy_math_file: Path,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    max_length: int,
    action_input_format: str,
    semantic_role_head_mode: str,
    semantic_role_loss_weight: float,
    sampler_mode: str,
    selection_metric: str,
    warmup_ratio: float,
    weight_decay: float,
    target_review_rates: tuple[float, ...],
    fit_temperature: bool,
    force: bool,
    recompute_deferral_only: bool,
    source_seed_layout: SeedRunLayout | None,
    suite_root: ExperimentRunPaths,
) -> dict[str, Any]:
    input_layout = source_seed_layout or seed_layout
    source_seed_root = None if source_seed_layout is None else str(source_seed_layout.root)
    _write_seed_metadata(
        seed_layout,
        {
            "suite_kind": SUITE_KIND,
            "suite_component": "seed_run",
            "suite_root": str(suite_root.root),
            "run_root": str(suite_root.root.parent),
            "seed": seed,
            "model": model,
            "train_file": str(train_file),
            "dev_file": str(dev_file),
            "test_file": str(test_file),
            "legacy_math_file": str(legacy_math_file),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "max_length": max_length,
            "action_input_format": action_input_format,
            "semantic_role_head_mode": semantic_role_head_mode,
            "semantic_role_loss_weight": semantic_role_loss_weight,
            "sampler_mode": sampler_mode,
            "selection_metric": selection_metric,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "target_review_rates": list(target_review_rates),
            "fit_temperature": fit_temperature,
            "recompute_deferral_only": recompute_deferral_only,
            "source_seed_run_root": source_seed_root,
            "training_dir": str(input_layout.training_dir),
        },
    )

    gold_counts = {
        "dev": len(load_jsonl(dev_file)),
        "test": len(load_jsonl(test_file)),
        "math": len(load_jsonl(legacy_math_file)),
    }

    if recompute_deferral_only:
        if force:
            for path in (
                seed_layout.direct_id_dev_predictions,
                seed_layout.direct_id_test_predictions,
                seed_layout.direct_id_math_predictions,
            ):
                _remove_if_force(path, force=True)
            _remove_if_force(seed_layout.no_rules_dir, force=True)
            _remove_if_force(seed_layout.direct_id_dir, force=True)

        if not _checkpoint_complete(input_layout.training_dir):
            raise SystemExit(
                f"Missing reusable checkpoint for seed {seed}: {input_layout.training_dir}. "
                "Provide --source-suite-root pointing at an existing completed suite or rerun without "
                "--recompute-deferral-only."
            )
        for split_name, prediction_file in (
            ("dev", input_layout.base_dev_predictions),
            ("test", input_layout.base_test_predictions),
            ("math", input_layout.base_math_predictions),
        ):
            if not _prediction_complete(prediction_file, expected_count=gold_counts[split_name]):
                raise SystemExit(
                    f"Missing reusable {split_name} predictions for seed {seed}: {prediction_file}. "
                    "The reevaluation-only path requires existing probability exports."
                )
    else:
        _remove_if_force(seed_layout.training_dir, force=force)
        if force:
            for path in (
                seed_layout.base_dev_predictions,
                seed_layout.base_test_predictions,
                seed_layout.base_math_predictions,
                seed_layout.direct_id_dev_predictions,
                seed_layout.direct_id_test_predictions,
                seed_layout.direct_id_math_predictions,
            ):
                _remove_if_force(path, force=True)
            _remove_if_force(seed_layout.no_rules_dir, force=True)
            _remove_if_force(seed_layout.direct_id_dir, force=True)

        if not _checkpoint_complete(seed_layout.training_dir):
            seed_layout.training_dir.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    sys.executable,
                    str(ROOT / "train_action.py"),
                    "--model",
                    model,
                    "--train-file",
                    str(train_file),
                    "--dev-file",
                    str(dev_file),
                    "--output-dir",
                    str(seed_layout.training_dir),
                    "--prediction-file",
                    str(seed_layout.base_dev_predictions),
                    "--epochs",
                    str(epochs),
                    "--lr",
                    str(learning_rate),
                    "--batch-size",
                    str(batch_size),
                    "--gradient-accumulation-steps",
                    str(gradient_accumulation_steps),
                    *(
                        ["--gradient-checkpointing"]
                        if gradient_checkpointing
                        else []
                    ),
                    "--max-length",
                    str(max_length),
                    "--action-input-format",
                    action_input_format,
                    "--semantic-role-head-mode",
                    semantic_role_head_mode,
                    "--semantic-role-loss-weight",
                    str(semantic_role_loss_weight),
                    "--sampler-mode",
                    sampler_mode,
                    "--selection-metric",
                    selection_metric,
                    "--warmup-ratio",
                    str(warmup_ratio),
                    "--weight-decay",
                    str(weight_decay),
                    "--seed",
                    str(seed),
                ],
                capture_json=True,
            )

        prediction_specs = (
            (dev_file, seed_layout.base_dev_predictions),
            (test_file, seed_layout.base_test_predictions),
            (legacy_math_file, seed_layout.base_math_predictions),
        )
        for gold_file, prediction_file in prediction_specs:
            split_name = "dev" if gold_file == dev_file else "test" if gold_file == test_file else "math"
            if _prediction_complete(prediction_file, expected_count=gold_counts[split_name]):
                continue
            _run(
                [
                    sys.executable,
                    str(ROOT / "scripts/predict_action.py"),
                    "--model",
                    str(seed_layout.training_dir),
                    "--input-file",
                    str(gold_file),
                    "--output-file",
                    str(prediction_file),
                    "--batch-size",
                    str(batch_size),
                    "--max-length",
                    str(max_length),
                    "--action-input-format",
                    action_input_format,
                ],
                capture_json=True,
            )

    direct_id_override_counts: dict[str, Any] = {}
    direct_id_specs = (
        ("dev", dev_file, input_layout.base_dev_predictions, seed_layout.direct_id_dev_predictions),
        ("test", test_file, input_layout.base_test_predictions, seed_layout.direct_id_test_predictions),
        ("math", legacy_math_file, input_layout.base_math_predictions, seed_layout.direct_id_math_predictions),
    )
    for split_name, gold_file, base_predictions, direct_id_predictions in direct_id_specs:
        if not _prediction_complete(direct_id_predictions, expected_count=gold_counts[split_name]):
            gold_rows = load_jsonl(gold_file)
            prediction_rows = load_jsonl(base_predictions)
            patched_rows, override_summary = apply_direct_id_overrides(gold_rows, prediction_rows)
            _write_jsonl(direct_id_predictions, patched_rows)
            direct_id_override_counts[split_name] = override_summary
        else:
            gold_rows = load_jsonl(gold_file)
            prediction_rows = load_jsonl(direct_id_predictions)
            _, override_summary = apply_direct_id_overrides(gold_rows, prediction_rows)
            direct_id_override_counts[split_name] = override_summary

    no_rules_variant = _variant_layout(seed_layout.no_rules_dir)
    if not _variant_complete(no_rules_variant.root):
        no_rules_summary = _write_variant_artifacts(
            no_rules_variant,
            run_name=f"seed-{seed}-no-rules",
            model_name=f"{model_display_name} seed {seed} no_rules",
            calibration_gold=dev_file,
            calibration_predictions=input_layout.base_dev_predictions,
            eval_gold=test_file,
            eval_predictions=input_layout.base_test_predictions,
            target_review_rates=target_review_rates,
            fit_temperature=fit_temperature,
            extra_metadata={
                "seed": seed,
                "variant": "no_rules",
                "training_dir": str(input_layout.training_dir),
                "recompute_deferral_only": recompute_deferral_only,
                "source_seed_run_root": source_seed_root,
            },
        )
    else:
        no_rules_summary = _load_json(no_rules_variant.summary_path)

    direct_id_variant = _variant_layout(seed_layout.direct_id_dir)
    if not _variant_complete(direct_id_variant.root):
        direct_id_summary = _write_variant_artifacts(
            direct_id_variant,
            run_name=f"seed-{seed}-direct-id",
            model_name=f"{model_display_name} seed {seed} direct_id",
            calibration_gold=dev_file,
            calibration_predictions=seed_layout.direct_id_dev_predictions,
            eval_gold=test_file,
            eval_predictions=seed_layout.direct_id_test_predictions,
            target_review_rates=target_review_rates,
            fit_temperature=fit_temperature,
            extra_metadata={
                "seed": seed,
                "variant": "direct_id",
                "training_dir": str(input_layout.training_dir),
                "direct_id_overrides": direct_id_override_counts,
                "recompute_deferral_only": recompute_deferral_only,
                "source_seed_run_root": source_seed_root,
            },
        )
    else:
        direct_id_summary = _load_json(direct_id_variant.summary_path)

    legacy_math_metrics = {
        "no_rules": _evaluate_action_metrics(legacy_math_file, input_layout.base_math_predictions),
        "direct_id": _evaluate_action_metrics(legacy_math_file, seed_layout.direct_id_math_predictions),
    }
    direct_id_test_spotcheck = _collect_direct_id_spotcheck(
        gold_file=test_file,
        base_prediction_file=input_layout.base_test_predictions,
        patched_prediction_file=seed_layout.direct_id_test_predictions,
    )

    training_info = _read_training_metrics(input_layout.training_dir)
    selected_10_no_rules = _selected_target(no_rules_summary, target_review_rate=STABILITY_TARGET_REVIEW_RATE)
    selected_10_direct_id = _selected_target(direct_id_summary, target_review_rate=STABILITY_TARGET_REVIEW_RATE)

    seed_summary = {
        "seed": seed,
        "seed_artifact_root": str(seed_layout.root),
        "training_dir": str(input_layout.training_dir),
        "source_seed_run_root": source_seed_root,
        "recompute_deferral_only": recompute_deferral_only,
        "recipe": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "max_length": max_length,
            "action_input_format": action_input_format,
            "semantic_role_head_mode": semantic_role_head_mode,
            "semantic_role_loss_weight": semantic_role_loss_weight,
            "sampler_mode": sampler_mode,
            "selection_metric": selection_metric,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "fit_temperature": fit_temperature,
        },
        "training": training_info,
        "files": {
            "base_dev_predictions": str(input_layout.base_dev_predictions),
            "base_test_predictions": str(input_layout.base_test_predictions),
            "base_math_predictions": str(input_layout.base_math_predictions),
            "direct_id_dev_predictions": str(seed_layout.direct_id_dev_predictions),
            "direct_id_test_predictions": str(seed_layout.direct_id_test_predictions),
            "direct_id_math_predictions": str(seed_layout.direct_id_math_predictions),
        },
        "direct_id_overrides": direct_id_override_counts,
        "direct_id_test_spotcheck": direct_id_test_spotcheck,
        "variants": {
            "no_rules": {
                "summary_path": str(no_rules_variant.summary_path),
                "report_path": str(no_rules_variant.report_path),
                "summary": no_rules_summary,
            },
            "direct_id": {
                "summary_path": str(direct_id_variant.summary_path),
                "report_path": str(direct_id_variant.report_path),
                "summary": direct_id_summary,
            },
        },
        "legacy_math_metrics": legacy_math_metrics,
        "base_test_macro_f1": {
            "no_rules": float(no_rules_summary["eval_base_metrics"]["macro_f1"]),
            "direct_id": float(direct_id_summary["eval_base_metrics"]["macro_f1"]),
        },
        "base_test_accuracy": {
            "no_rules": float(no_rules_summary["eval_base_metrics"]["accuracy"]),
            "direct_id": float(direct_id_summary["eval_base_metrics"]["accuracy"]),
        },
        "base_test_redact_recall": {
            "no_rules": float(no_rules_summary["eval_base_metrics"]["redact_recall"]),
            "direct_id": float(direct_id_summary["eval_base_metrics"]["redact_recall"]),
        },
        "selected_10_no_rules_macro_f1": float(selected_10_no_rules["evaluation"]["metrics"]["macro_f1"]),
        "selected_10_direct_id_macro_f1": float(selected_10_direct_id["evaluation"]["metrics"]["macro_f1"]),
    }
    _write_json(seed_layout.summary_path, seed_summary)
    seed_layout.report_path.write_text(
        _build_seed_report(
            model_name=model_display_name,
            seed=seed,
            training_dir=input_layout.training_dir,
            no_rules_summary=no_rules_summary,
            direct_id_summary=direct_id_summary,
            legacy_math_metrics=legacy_math_metrics,
            direct_id_override_counts=direct_id_override_counts,
            direct_id_test_spotcheck=direct_id_test_spotcheck,
        ),
        encoding="utf-8",
    )
    return seed_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the three-seed UPChieve backbone cascade package.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DEV_FILE)
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--legacy-math-file", type=Path, default=DEFAULT_LEGACY_MATH_FILE)
    parser.add_argument("--seeds", default="13,42,101")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument(
        "--action-input-format",
        choices=ACTION_INPUT_FORMAT_CHOICES,
        default=DEFAULT_ACTION_INPUT_FORMAT,
    )
    parser.add_argument(
        "--semantic-role-head-mode",
        choices=SEMANTIC_ROLE_HEAD_MODE_CHOICES,
        default="none",
    )
    parser.add_argument("--semantic-role-loss-weight", type=float, default=0.3)
    parser.add_argument(
        "--sampler-mode",
        choices=SAMPLER_MODE_CHOICES,
        default="none",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["redact_recall", "macro_f1", "accuracy", POLICY_SELECTION_METRIC],
        default="redact_recall",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-review-rates", default="0.05,0.10")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--recompute-deferral-only", action="store_true")
    parser.add_argument("--source-suite-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    run_root = args.run_root
    if args.selection_metric == POLICY_SELECTION_METRIC and not args.fit_temperature:
        args.fit_temperature = True
    seeds = tuple(int(chunk.strip()) for chunk in args.seeds.split(",") if chunk.strip())
    if not seeds:
        raise SystemExit("At least one seed is required.")
    target_review_rates = _parse_target_review_rates(args.target_review_rates)
    if 0.10 not in target_review_rates:
        raise SystemExit("The seeded suite requires a 0.10 target review rate for the final headline comparison.")
    if args.source_suite_root is not None and not args.recompute_deferral_only:
        raise SystemExit("--source-suite-root is only supported together with --recompute-deferral-only.")
    model_display_name = _canonical_model_name(args.model)
    historical_reference = _historical_reference(model_display_name)
    source_suite_root = args.source_suite_root
    source_seed_runs = (
        _seed_run_paths_from_suite_root(source_suite_root)
        if source_suite_root is not None
        else {}
    )

    suite_root = _ensure_suite_root(args.run_name, run_root=run_root)
    suite_metadata = _load_json(suite_root.metadata_path) if suite_root.metadata_path.exists() else {}
    suite_metadata.update(
        {
            "suite_kind": SUITE_KIND,
            "run_name": args.run_name,
            "run_root": str(run_root),
            "model": args.model,
            "train_file": str(args.train_file),
            "dev_file": str(args.dev_file),
            "test_file": str(args.test_file),
            "legacy_math_file": str(args.legacy_math_file),
            "seeds": list(seeds),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_length": args.max_length,
            "action_input_format": args.action_input_format,
            "semantic_role_head_mode": args.semantic_role_head_mode,
            "semantic_role_loss_weight": args.semantic_role_loss_weight,
            "sampler_mode": args.sampler_mode,
            "selection_metric": args.selection_metric,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "target_review_rates": list(target_review_rates),
            "fit_temperature": args.fit_temperature,
            "recompute_deferral_only": args.recompute_deferral_only,
            "source_suite_root": None if source_suite_root is None else str(source_suite_root),
        }
    )
    write_run_metadata(suite_root.metadata_path, suite_metadata)

    seed_summaries: list[dict[str, Any]] = []
    seed_run_paths: list[str] = []
    for seed in seeds:
        seed_layout = _resolve_seed_run(
            suite_root,
            suite_metadata,
            run_name=args.run_name,
            seed=seed,
            run_root=run_root,
        )
        suite_metadata = _load_json(suite_root.metadata_path)
        seed_run_paths.append(str(seed_layout.root))
        seed_summary = _run_seed(
            seed=seed,
            seed_layout=seed_layout,
            model_display_name=model_display_name,
            model=args.model,
            train_file=args.train_file,
            dev_file=args.dev_file,
            test_file=args.test_file,
            legacy_math_file=args.legacy_math_file,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            max_length=args.max_length,
            action_input_format=args.action_input_format,
            semantic_role_head_mode=args.semantic_role_head_mode,
            semantic_role_loss_weight=args.semantic_role_loss_weight,
            sampler_mode=args.sampler_mode,
            selection_metric=args.selection_metric,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            target_review_rates=target_review_rates,
            fit_temperature=args.fit_temperature,
            force=args.force,
            recompute_deferral_only=args.recompute_deferral_only,
            source_seed_layout=None if str(seed) not in source_seed_runs else _seed_layout(source_seed_runs[str(seed)]),
            suite_root=suite_root,
        )
        seed_summaries.append(seed_summary)

    suite_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for seed_summary in seed_summaries:
        seed = int(seed_summary["seed"])
        for variant in ("no_rules", "direct_id"):
            variant_summary = seed_summary["variants"][variant]["summary"]
            math_metrics = seed_summary["legacy_math_metrics"][variant]
            base_metrics = variant_summary["eval_base_metrics"]
            base_rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "macro_f1": float(base_metrics["macro_f1"]),
                    "accuracy": float(base_metrics["accuracy"]),
                    "redact_recall": float(base_metrics["redact_recall"]),
                }
            )
            for target_review_rate in target_review_rates:
                selected = _find_selected_target(variant_summary, target_review_rate=target_review_rate)
                if selected is None:
                    continue
                evaluation_metrics = selected["evaluation"]["metrics"]
                calibration_metrics = selected["calibration"]["metrics"]
                suite_rows.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "target_review_rate": target_review_rate,
                        "strategy": selected["calibration"]["strategy"],
                        "parameters": selected["calibration"]["parameters"],
                        "temperature": float(selected["calibration"]["temperature"]),
                        "dev_review_rate": float(calibration_metrics["review_rate"]),
                        "test_review_rate": float(evaluation_metrics["review_rate"]),
                        "macro_f1": float(evaluation_metrics["macro_f1"]),
                        "accuracy": float(evaluation_metrics["accuracy"]),
                        "redact_recall": float(evaluation_metrics["redact_recall"]),
                        "curricular_accuracy": _curricular_accuracy(evaluation_metrics),
                        "gold_review_coverage": (
                            None
                            if evaluation_metrics.get("gold_review_coverage") is None
                            else float(evaluation_metrics["gold_review_coverage"])
                        ),
                        "protected_redact_rate": (
                            None
                            if evaluation_metrics.get("protected_redact_rate") is None
                            else float(evaluation_metrics["protected_redact_rate"])
                        ),
                        "redact_leak_rate": (
                            None
                            if evaluation_metrics.get("redact_leak_rate") is None
                            else float(evaluation_metrics["redact_leak_rate"])
                        ),
                        "legacy_math_redact_recall": float(math_metrics["redact_recall"]),
                        "seed_summary_path": str(seed_summary["training_dir"]),
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    for variant in ("no_rules", "direct_id"):
        for target_review_rate in target_review_rates:
            subset = [
                row
                for row in suite_rows
                if row["variant"] == variant and abs(row["target_review_rate"] - target_review_rate) <= 1e-9
            ]
            if not subset:
                continue
            aggregate_rows.append(
                {
                    "variant": variant,
                    "target_review_rate": target_review_rate,
                    "macro_f1_values": [row["macro_f1"] for row in subset],
                    "accuracy_values": [row["accuracy"] for row in subset],
                    "redact_recall_values": [row["redact_recall"] for row in subset],
                    "curricular_accuracy_values": [row["curricular_accuracy"] for row in subset if row["curricular_accuracy"] is not None],
                    "gold_review_coverage_values": [row["gold_review_coverage"] for row in subset if row["gold_review_coverage"] is not None],
                    "review_rate_values": [row["test_review_rate"] for row in subset],
                    "protected_redact_rate_values": [row["protected_redact_rate"] for row in subset if row["protected_redact_rate"] is not None],
                    "redact_leak_rate_values": [row["redact_leak_rate"] for row in subset if row["redact_leak_rate"] is not None],
                    "legacy_math_redact_recall_values": [row["legacy_math_redact_recall"] for row in subset],
                }
            )

    for row in aggregate_rows:
        row["macro_f1_text"] = _format_mean_std(row["macro_f1_values"])
        row["accuracy_text"] = _format_mean_std(row["accuracy_values"])
        row["redact_recall_text"] = _format_mean_std(row["redact_recall_values"])
        row["curricular_accuracy_text"] = (
            _format_mean_std(row["curricular_accuracy_values"]) if row["curricular_accuracy_values"] else "n/a"
        )
        row["gold_review_coverage_text"] = (
            _format_mean_std(row["gold_review_coverage_values"]) if row["gold_review_coverage_values"] else "n/a"
        )
        row["review_rate_text"] = _format_mean_std(row["review_rate_values"])
        row["protected_redact_rate_text"] = (
            _format_mean_std(row["protected_redact_rate_values"]) if row["protected_redact_rate_values"] else "n/a"
        )
        row["redact_leak_rate_text"] = (
            _format_mean_std(row["redact_leak_rate_values"]) if row["redact_leak_rate_values"] else "n/a"
        )
        row["legacy_math_redact_recall_text"] = _format_mean_std(row["legacy_math_redact_recall_values"])

    base_vs_selected_rows: list[dict[str, Any]] = []
    for variant in ("no_rules", "direct_id"):
        base_subset = [row for row in base_rows if row["variant"] == variant]
        base_macro_avg, base_macro_std = _mean_std([row["macro_f1"] for row in base_subset])
        base_accuracy_avg, base_accuracy_std = _mean_std([row["accuracy"] for row in base_subset])
        base_redact_avg, base_redact_std = _mean_std([row["redact_recall"] for row in base_subset])
        for target_review_rate in target_review_rates:
            selected_row = next(
                (
                    row
                    for row in aggregate_rows
                    if row["variant"] == variant and abs(row["target_review_rate"] - target_review_rate) <= 1e-9
                ),
                None,
            )
            if selected_row is None:
                continue
            selected_macro_avg, _ = _mean_std(selected_row["macro_f1_values"])
            selected_accuracy_avg, _ = _mean_std(selected_row["accuracy_values"])
            selected_redact_avg, _ = _mean_std(selected_row["redact_recall_values"])
            base_vs_selected_rows.append(
                {
                    "variant": variant,
                    "target_review_rate": target_review_rate,
                    "base_macro_f1_text": f"{base_macro_avg:.4f} ± {base_macro_std:.4f}",
                    "selected_macro_f1_text": selected_row["macro_f1_text"],
                    "delta_macro_f1_text": f"{selected_macro_avg - base_macro_avg:+.4f}",
                    "base_accuracy_text": f"{base_accuracy_avg:.4f} ± {base_accuracy_std:.4f}",
                    "selected_accuracy_text": selected_row["accuracy_text"],
                    "delta_accuracy_text": f"{selected_accuracy_avg - base_accuracy_avg:+.4f}",
                    "base_redact_recall_text": f"{base_redact_avg:.4f} ± {base_redact_std:.4f}",
                    "selected_redact_recall_text": selected_row["redact_recall_text"],
                    "delta_redact_recall_text": f"{selected_redact_avg - base_redact_avg:+.4f}",
                }
            )

    historical_rows = _historical_rows()
    anchor_gap: float | None = None
    if historical_reference is not None:
        seed_42_row = next(
            (
                row
                for row in suite_rows
                if row["seed"] == 42
                and row["variant"] == STABILITY_VARIANT
                and abs(row["target_review_rate"] - STABILITY_TARGET_REVIEW_RATE) <= 1e-9
            ),
            None,
        )
        if seed_42_row is not None:
            anchor_gap = seed_42_row["macro_f1"] - float(historical_reference["macro_f1"])

    selected_10_no_rules_stability = _policy_stability_summary(suite_rows)
    source_suite_comparison = None
    if source_suite_root is not None and selected_10_no_rules_stability is not None:
        baseline_suite_summary = _load_json(source_suite_root / "summary.json")
        baseline_stability = _policy_stability_summary_from_suite(baseline_suite_summary)
        if baseline_stability is not None:
            source_suite_comparison = _temperature_gate_decision(
                selected_10_no_rules_stability,
                baseline_stability,
            )

    suite_summary = {
        "suite_kind": SUITE_KIND,
        "run_name": args.run_name,
        "suite_root": str(suite_root.root),
        "run_root": str(run_root),
        "seed_run_paths": seed_run_paths,
        "model": args.model,
        "model_display_name": model_display_name,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "test_file": str(args.test_file),
        "legacy_math_file": str(args.legacy_math_file),
        "seeds": list(seeds),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "action_input_format": args.action_input_format,
        "semantic_role_head_mode": args.semantic_role_head_mode,
        "semantic_role_loss_weight": args.semantic_role_loss_weight,
        "sampler_mode": args.sampler_mode,
        "selection_metric": args.selection_metric,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "target_review_rates": list(target_review_rates),
        "fit_temperature": args.fit_temperature,
        "recompute_deferral_only": args.recompute_deferral_only,
        "source_suite_root": None if source_suite_root is None else str(source_suite_root),
        "historical_anchor": historical_reference,
        "seed_42_anchor_gap": anchor_gap,
        "base_rows": base_rows,
        "seed_rows": suite_rows,
        "aggregate_rows": aggregate_rows,
        "base_vs_selected_rows": base_vs_selected_rows,
        "selected_10_no_rules_stability": selected_10_no_rules_stability,
        "source_suite_comparison": source_suite_comparison,
        "historical_backbone_rows": historical_rows,
    }
    _write_json(suite_root.summary_path, suite_summary)
    suite_root.report_path.write_text(
        _build_suite_report(
            model_name=model_display_name,
            run_name=args.run_name,
            run_root=run_root,
            seeds=seeds,
            target_review_rates=target_review_rates,
            action_input_format=args.action_input_format,
            semantic_role_head_mode=args.semantic_role_head_mode,
            semantic_role_loss_weight=args.semantic_role_loss_weight,
            sampler_mode=args.sampler_mode,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            selection_metric=args.selection_metric,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            fit_temperature=args.fit_temperature,
            recompute_deferral_only=args.recompute_deferral_only,
            source_suite_root=source_suite_root,
            suite_rows=suite_rows,
            aggregate_rows=aggregate_rows,
            base_vs_selected_rows=base_vs_selected_rows,
            historical_rows=historical_rows,
            historical_reference=historical_reference,
            anchor_gap=anchor_gap,
            selected_10_no_rules_stability=selected_10_no_rules_stability,
            source_suite_comparison=source_suite_comparison,
        ),
        encoding="utf-8",
    )

    headline_row = next(
        row
        for row in aggregate_rows
        if row["variant"] == STABILITY_VARIANT and abs(row["target_review_rate"] - STABILITY_TARGET_REVIEW_RATE) <= 1e-9
    )
    suite_metadata = _load_json(suite_root.metadata_path)
    results_signature = _results_logged_signature(
        seeds=seeds,
        target_review_rates=target_review_rates,
        action_input_format=args.action_input_format,
        semantic_role_head_mode=args.semantic_role_head_mode,
        semantic_role_loss_weight=args.semantic_role_loss_weight,
        sampler_mode=args.sampler_mode,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fit_temperature=args.fit_temperature,
        selection_metric=args.selection_metric,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        recompute_deferral_only=args.recompute_deferral_only,
    )
    if len(seeds) > 1 and (args.force or suite_metadata.get("results_logged_signature") != results_signature):
        result_stage = "eval" if args.recompute_deferral_only else "action"
        for seed_summary in seed_summaries:
            _append_result(
                float(seed_summary["selected_10_no_rules_macro_f1"]),
                "keep",
                (
                    f"{model_display_name} seed {seed_summary['seed']} "
                    f"{'deferral reevaluation' if args.recompute_deferral_only else 'mixed-context v2 seeded cascade package'} "
                    f"completed with no-rules 10% macro F1 {float(seed_summary['selected_10_no_rules_macro_f1']):.4f} "
                    f"and direct-id 10% macro F1 {float(seed_summary['selected_10_direct_id_macro_f1']):.4f} "
                    f"[{Path(seed_summary['seed_artifact_root']).name}]"
                ),
                stage=result_stage,
            )
        anchor_clause = (
            f" and seed-42 anchor gap {anchor_gap:+.4f} (reference only)"
            if anchor_gap is not None
            else ""
        )
        _append_result(
            mean(headline_row["macro_f1_values"]),
            "keep",
            (
                f"{model_display_name} 3-seed "
                f"{'deferral reevaluation' if args.recompute_deferral_only else 'mixed-context v2 seeded cascade aggregate'} "
                f"completed with no-rules 10% macro F1 {headline_row['macro_f1_text']}{anchor_clause} "
                f"[{suite_root.root.name}]"
            ),
            stage=result_stage,
        )
        suite_metadata["results_logged"] = True
        suite_metadata["results_logged_signature"] = results_signature
        write_run_metadata(suite_root.metadata_path, suite_metadata)

    print(
        json.dumps(
            {
                "suite_root": str(suite_root.root),
                "summary_path": str(suite_root.summary_path),
                "report_path": str(suite_root.report_path),
                "seed_run_paths": seed_run_paths,
                "seed_42_anchor_gap": anchor_gap,
                "selected_10_no_rules_stability": selected_10_no_rules_stability,
                "source_suite_comparison": source_suite_comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
