from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, ANNOTATION_DIR, CANDIDATE_DIR
from contextshift_deid.data import ensure_repo_layout
from contextshift_deid.experiment_runs import EXPERIMENTS_DIR, slugify

DEFAULT_SUMMARY_PATHS = (
    ROOT / "artifacts" / "experiments" / "20260311_024035_candidate-math-distilbert-3ep-test" / "summary.json",
    ROOT / "artifacts" / "experiments" / "20260311_115739_candidate-math-roberta-3ep-test" / "summary.json",
)
PACKAGES_DIR = ANNOTATION_DIR / "packages"
HEADLINE_CONFIG = {
    "privacy_recall": {
        "metric_key": "worst_context_recall",
        "label": "privacy recall",
    },
    "overall_f1": {
        "metric_key": "f1",
        "label": "overall F1",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_candidate_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise SystemExit(f"{path}: missing metrics block")
    source_checkpoint = payload.get("source_checkpoint")
    if not source_checkpoint:
        raise SystemExit(f"{path}: missing source_checkpoint")
    return {
        "summary_path": path,
        "experiment_dir": path.parent,
        "model": str(payload.get("model", "unknown")),
        "source_checkpoint": _resolve_repo_path(str(source_checkpoint)),
        "selection_metric": str(payload.get("selection_metric", "unknown")),
        "metrics": metrics,
        "status": str(payload.get("status", "unknown")),
    }


def _select_operating_point(summaries: list[dict[str, Any]], *, headline: str) -> dict[str, Any]:
    config = HEADLINE_CONFIG[headline]
    metric_key = config["metric_key"]
    missing = [summary["summary_path"] for summary in summaries if metric_key not in summary["metrics"]]
    if missing:
        raise SystemExit(f"Missing {metric_key} in candidate summaries: {missing}")
    return max(
        summaries,
        key=lambda summary: (
            float(summary["metrics"][metric_key]),
            float(summary["metrics"].get("f1", 0.0)),
            float(summary["metrics"].get("worst_context_recall", 0.0)),
        ),
    )


def _unique_dir(path: Path) -> Path:
    candidate = path
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = path.with_name(f"{path.name}_{suffix}")
    return candidate


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _latest_child_dir(path: Path) -> Path:
    child_dirs = sorted(directory for directory in path.iterdir() if directory.is_dir())
    if not child_dirs:
        raise SystemExit(f"No run directory created under {path}")
    return child_dirs[-1]


def _new_child_dir(path: Path, before: set[Path]) -> Path:
    new_dirs = sorted(directory for directory in path.iterdir() if directory.is_dir() and directory not in before)
    if new_dirs:
        return new_dirs[-1]
    return _latest_child_dir(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _build_holdout_block(
    *,
    checkpoint: Path,
    model_alias: str,
    run_holdout: bool,
    holdout_run_root: Path,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    holdout_run_root.mkdir(parents=True, exist_ok=True)
    if not run_holdout:
        return {
            "status": "skipped",
            "reason": "Requested via --skip-holdout",
            "run_root": str(holdout_run_root),
        }

    existing_run_dirs = {directory for directory in holdout_run_root.iterdir() if directory.is_dir()}
    _run(
        [
            sys.executable,
            "scripts/run_candidate_holdout_eval.py",
            "--model",
            str(checkpoint),
            "--model-alias",
            model_alias,
            "--run-root",
            str(holdout_run_root),
            "--run-name",
            f"{model_alias}-holdout",
            "--batch-size",
            str(batch_size),
            "--max-length",
            str(max_length),
        ]
    )
    run_dir = _new_child_dir(holdout_run_root, existing_run_dirs)
    return {
        "status": "completed",
        "run_root": str(holdout_run_root),
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "summary.json"),
        "report_md": str(run_dir / "report.md"),
        "metadata_json": str(run_dir / "metadata.json"),
    }


def _build_split_pool(
    *,
    split_name: str,
    checkpoint: Path,
    model_alias: str,
    candidate_file: Path,
    provisional_action_file: Path,
    predictions_dir: Path,
    pools_dir: Path,
    context_mode: str,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    prediction_file = predictions_dir / f"candidate_{split_name}_{model_alias}.jsonl"
    pool_file = pools_dir / f"action_pool_{split_name}.jsonl"
    _run(
        [
            sys.executable,
            "scripts/predict_candidate.py",
            "--model",
            str(checkpoint),
            "--input-file",
            str(candidate_file),
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
    _run(
        [
            sys.executable,
            "scripts/build_action_annotation_pool.py",
            "--candidate-file",
            str(candidate_file),
            "--provisional-action-file",
            str(provisional_action_file),
            "--prediction-file",
            str(prediction_file),
            "--split-name",
            split_name,
            "--output-file",
            str(pool_file),
        ]
    )
    return {
        "candidate_file": str(candidate_file),
        "provisional_action_file": str(provisional_action_file),
        "prediction_file": str(prediction_file),
        "pool_file": str(pool_file),
    }


def _build_second_subject_pilot(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    model_alias: str,
    predictions_dir: Path,
    pools_dir: Path,
) -> dict[str, Any]:
    if args.pilot_candidate_file is None and args.pilot_action_file is None:
        return {
            "status": "blocked",
            "reason": "No second-subject processed candidate/action files were provided. The current processed benchmark is still math-only.",
        }

    if args.pilot_candidate_file is None or args.pilot_action_file is None:
        raise SystemExit("Provide both --pilot-candidate-file and --pilot-action-file when enabling the second-subject pilot.")

    split_name = args.pilot_name or args.pilot_candidate_file.stem
    payload = _build_split_pool(
        split_name=split_name,
        checkpoint=checkpoint,
        model_alias=model_alias,
        candidate_file=args.pilot_candidate_file,
        provisional_action_file=args.pilot_action_file,
        predictions_dir=predictions_dir,
        pools_dir=pools_dir,
        context_mode=args.context_mode,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    payload["status"] = "ready"
    return payload


def _instructions_text(manifest: dict[str, Any]) -> str:
    codebook_path = manifest["codebook_v2"]
    train_pool = manifest["splits"]["train"]["pool_file"]
    dev_pool = manifest["splits"]["dev"]["pool_file"]
    test_pool = manifest["splits"]["test"]["pool_file"]
    lines = [
        "# Action Work Package",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Headline: {manifest['headline']}",
        f"- Frozen candidate model: {manifest['selected_operating_point']['model']}",
        f"- Candidate checkpoint: {manifest['selected_operating_point']['source_checkpoint']}",
        f"- Selection rationale: {manifest['selected_operating_point']['rationale']}",
        f"- Codebook: {codebook_path}",
        "",
        "## Next Commands",
        "",
        "1. Read and calibrate against the codebook.",
        f"   `sed -n '1,260p' {codebook_path}`",
        "2. Annotate the frozen train pool.",
        f"   `uv run scripts/annotate_action_pool.py --input-file {train_pool} --annotator <annotator_id>`",
        "3. Export adjudicated train labels back into the benchmark once labeling is complete.",
        f"   `uv run scripts/export_action_annotations.py --input-file {train_pool.replace('.jsonl', '.annotated.jsonl')} --output-file data/processed/action/train.jsonl`",
        "4. Annotate the frozen dev pool.",
        f"   `uv run scripts/annotate_action_pool.py --input-file {dev_pool} --annotator <annotator_id>`",
        "5. Annotate the frozen test pool.",
        f"   `uv run scripts/annotate_action_pool.py --input-file {test_pool} --annotator <annotator_id>`",
        "6. Export adjudicated dev/test labels back into the benchmark once labeling is complete.",
        f"   `uv run scripts/export_action_annotations.py --input-file {dev_pool.replace('.jsonl', '.annotated.jsonl')} --output-file data/processed/action/dev.jsonl`",
        f"   `uv run scripts/export_action_annotations.py --input-file {test_pool.replace('.jsonl', '.annotated.jsonl')} --output-file data/processed/action/test.jsonl`",
        "7. Validate the benchmark with no provisional action labels remaining.",
        "   `uv run prepare.py --strict`",
        "8. Train the first real action pilot on manual train + manual dev, then score test.",
        "   `uv run train_action.py --train-file data/processed/action/train.jsonl --dev-file data/processed/action/dev.jsonl --output-dir runs/action_pilot_v1 --prediction-file artifacts/predictions/action_dev_predictions.jsonl`",
        "   `uv run scripts/predict_action.py --model runs/action_pilot_v1 --input-file data/processed/action/test.jsonl --output-file artifacts/predictions/action_test_predictions.jsonl`",
        "   `uv run eval.py --stage action --gold data/processed/action/test.jsonl --predictions artifacts/predictions/action_test_predictions.jsonl`",
    ]
    pilot = manifest["second_subject_pilot"]
    if pilot["status"] == "ready":
        lines.extend(
            [
                "9. Label the second-subject pilot immediately so the context-shift claim becomes testable.",
                f"   `uv run scripts/annotate_action_pool.py --input-file {pilot['pool_file']} --annotator <annotator_id>`",
            ]
        )
    else:
        lines.extend(
            [
                "9. Add a second-subject pilot as soon as processed candidate/action files exist.",
                f"   Current status: {pilot['reason']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one candidate operating point and bootstrap the action relabeling work package.")
    parser.add_argument("--headline", choices=sorted(HEADLINE_CONFIG), default="privacy_recall")
    parser.add_argument("--package-name", default="action-work-package")
    parser.add_argument(
        "--candidate-summary",
        action="append",
        type=Path,
        help="Repeatable candidate test summary.json files to consider. Defaults to the current distilbert and roberta test summaries.",
    )
    parser.add_argument("--train-candidate-file", type=Path, default=CANDIDATE_DIR / "train.jsonl")
    parser.add_argument("--train-action-file", type=Path, default=ACTION_DIR / "train.jsonl")
    parser.add_argument("--dev-candidate-file", type=Path, default=CANDIDATE_DIR / "dev.jsonl")
    parser.add_argument("--dev-action-file", type=Path, default=ACTION_DIR / "dev.jsonl")
    parser.add_argument("--test-candidate-file", type=Path, default=CANDIDATE_DIR / "test.jsonl")
    parser.add_argument("--test-action-file", type=Path, default=ACTION_DIR / "test.jsonl")
    parser.add_argument("--pilot-candidate-file", type=Path, help="Optional processed candidate split for a small second-subject pilot.")
    parser.add_argument("--pilot-action-file", type=Path, help="Optional provisional action split for the second-subject pilot.")
    parser.add_argument("--pilot-name", help="Split label used for the second-subject pilot pool.")
    parser.add_argument("--skip-holdout", action="store_true", help="Skip the clean holdout rerun and only freeze the supporting manifest + pools.")
    parser.add_argument("--holdout-run-root", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--context-mode", choices=("none", "pair"), default="none")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    ensure_repo_layout()
    package_root = PACKAGES_DIR
    package_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_dir = _unique_dir(package_root / f"{timestamp}_{slugify(args.package_name)}-{slugify(args.headline)}")
    package_dir.mkdir(parents=True, exist_ok=False)

    summary_paths = args.candidate_summary or list(DEFAULT_SUMMARY_PATHS)
    summaries = [_load_candidate_summary(path) for path in summary_paths]
    selected = _select_operating_point(summaries, headline=args.headline)
    selected_metric_key = HEADLINE_CONFIG[args.headline]["metric_key"]
    model_alias = slugify(f"{selected['model']}-{args.headline}")

    predictions_dir = package_dir / "predictions"
    pools_dir = package_dir / "pools"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pools_dir.mkdir(parents=True, exist_ok=True)

    holdout = _build_holdout_block(
        checkpoint=selected["source_checkpoint"],
        model_alias=model_alias,
        run_holdout=not args.skip_holdout,
        holdout_run_root=args.holdout_run_root,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    split_payload = {
        "train": _build_split_pool(
            split_name="train",
            checkpoint=selected["source_checkpoint"],
            model_alias=model_alias,
            candidate_file=args.train_candidate_file,
            provisional_action_file=args.train_action_file,
            predictions_dir=predictions_dir,
            pools_dir=pools_dir,
            context_mode=args.context_mode,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        "dev": _build_split_pool(
            split_name="dev",
            checkpoint=selected["source_checkpoint"],
            model_alias=model_alias,
            candidate_file=args.dev_candidate_file,
            provisional_action_file=args.dev_action_file,
            predictions_dir=predictions_dir,
            pools_dir=pools_dir,
            context_mode=args.context_mode,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        "test": _build_split_pool(
            split_name="test",
            checkpoint=selected["source_checkpoint"],
            model_alias=model_alias,
            candidate_file=args.test_candidate_file,
            provisional_action_file=args.test_action_file,
            predictions_dir=predictions_dir,
            pools_dir=pools_dir,
            context_mode=args.context_mode,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
    }
    second_subject_pilot = _build_second_subject_pilot(
        args=args,
        checkpoint=selected["source_checkpoint"],
        model_alias=model_alias,
        predictions_dir=predictions_dir,
        pools_dir=pools_dir,
    )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "headline": args.headline,
        "codebook_v2": str(ROOT / "docs" / "codebook_v2.md"),
        "package_dir": str(package_dir),
        "selected_operating_point": {
            "summary_path": str(selected["summary_path"]),
            "experiment_dir": str(selected["experiment_dir"]),
            "model": selected["model"],
            "source_checkpoint": str(selected["source_checkpoint"]),
            "selection_metric": selected["selection_metric"],
            "selected_metric_key": selected_metric_key,
            "selected_metric_value": float(selected["metrics"][selected_metric_key]),
            "f1": float(selected["metrics"].get("f1", 0.0)),
            "worst_context_recall": float(selected["metrics"].get("worst_context_recall", 0.0)),
            "rationale": (
                f"Selected by the highest {selected_metric_key} across {len(summaries)} locked candidate test summaries "
                f"for a {HEADLINE_CONFIG[args.headline]['label']} headline."
            ),
        },
        "available_candidate_summaries": [
            {
                "summary_path": str(summary["summary_path"]),
                "model": summary["model"],
                "source_checkpoint": str(summary["source_checkpoint"]),
                "f1": float(summary["metrics"].get("f1", 0.0)),
                "worst_context_recall": float(summary["metrics"].get("worst_context_recall", 0.0)),
                "selected": str(summary["summary_path"]) == str(selected["summary_path"]),
            }
            for summary in summaries
        ],
        "holdout": holdout,
        "splits": split_payload,
        "second_subject_pilot": second_subject_pilot,
    }

    manifest_path = package_dir / "manifest.json"
    instructions_path = package_dir / "instructions.md"
    _write_json(manifest_path, manifest)
    _write_text(instructions_path, _instructions_text(manifest))

    print(
        json.dumps(
            {
                "package_dir": str(package_dir),
                "manifest_json": str(manifest_path),
                "instructions_md": str(instructions_path),
                "headline": args.headline,
                "selected_checkpoint": str(selected["source_checkpoint"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
