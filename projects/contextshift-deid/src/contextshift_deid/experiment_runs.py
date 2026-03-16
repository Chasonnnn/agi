from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re

from .constants import ARTIFACTS_DIR

EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "experiment"


@dataclass(frozen=True)
class ExperimentRunPaths:
    root: Path
    predictions_dir: Path
    metadata_path: Path
    summary_path: Path
    report_path: Path


def create_experiment_run(run_name: str, *, root_dir: Path = EXPERIMENTS_DIR) -> ExperimentRunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(run_name)
    candidate = root_dir / f"{timestamp}_{slug}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = root_dir / f"{timestamp}_{slug}_{suffix}"
    predictions_dir = candidate / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=False)
    return ExperimentRunPaths(
        root=candidate,
        predictions_dir=predictions_dir,
        metadata_path=candidate / "metadata.json",
        summary_path=candidate / "summary.json",
        report_path=candidate / "report.md",
    )


def write_run_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
