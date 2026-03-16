from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results.tsv"
ARTIFACTS_DIR = ROOT / "artifacts"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
SMOKE_MODELS_DIR = ARTIFACTS_DIR / "smoke_models"


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _append_result(stage: str, metric: float, status: str, description: str) -> None:
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"smoke\t{stage}\t{metric:.6f}\t{status}\t{description}\n")


def _read_json_output(command: list[str]) -> dict:
    print("+", " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    print(result.stdout)
    return json.loads(result.stdout)


def main() -> None:
    candidate_model = SMOKE_MODELS_DIR / "candidate-bert"
    action_model = SMOKE_MODELS_DIR / "action-bert"
    pending_results: list[tuple[str, float, str, str]] = []

    _run([sys.executable, "scripts/generate_synthetic_benchmark.py"])
    _run([sys.executable, "scripts/generate_local_smoke_models.py"])
    _run([sys.executable, "prepare.py", "--strict"])

    candidate_prediction_file = PREDICTIONS_DIR / "candidate_dev_predictions.jsonl"
    action_prediction_file = PREDICTIONS_DIR / "action_dev_predictions.jsonl"

    _run(
        [
            sys.executable,
            "train.py",
            "--stage",
            "candidate",
            "--model",
            str(candidate_model),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--prediction-file",
            str(candidate_prediction_file),
        ]
    )
    candidate_metrics = _read_json_output(
        [
            sys.executable,
            "eval.py",
            "--stage",
            "candidate",
            "--gold",
            "data/processed/candidate/dev.jsonl",
            "--predictions",
            str(candidate_prediction_file),
        ]
    )
    pending_results.append(
        (
            "candidate",
            float(candidate_metrics.get("worst_context_recall") or 0.0),
            "keep",
            "smoke test candidate stage",
        )
    )

    _run(
        [
            sys.executable,
            "train.py",
            "--stage",
            "action",
            "--model",
            str(action_model),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--prediction-file",
            str(action_prediction_file),
        ]
    )
    action_metrics = _read_json_output(
        [
            sys.executable,
            "eval.py",
            "--stage",
            "action",
            "--gold",
            "data/processed/action/dev.jsonl",
            "--predictions",
            str(action_prediction_file),
        ]
    )
    pending_results.append(
        (
            "action",
            float(action_metrics.get("worst_context_redact_recall") or 0.0),
            "keep",
            "smoke test action stage",
        )
    )

    summary = {
        "candidate": candidate_metrics,
        "action": action_metrics,
    }
    summary_path = ARTIFACTS_DIR / "smoke_test_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for stage, metric, status, description in pending_results:
        _append_result(stage, metric, status, description)
    print(f"Smoke test summary written to {summary_path}")


if __name__ == "__main__":
    main()
