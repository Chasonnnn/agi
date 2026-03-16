from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CANDIDATE_DIR = PROCESSED_DIR / "candidate"
ACTION_DIR = PROCESSED_DIR / "action"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ANNOTATION_DIR = ARTIFACTS_DIR / "annotation"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
RUNS_DIR = REPO_ROOT / "runs"
DOCS_DIR = REPO_ROOT / "docs"

EXPECTED_SPLITS = ("train", "dev", "test")
DEFAULT_CANDIDATE_LABELS = ("O", "B-SUSPECT", "I-SUSPECT")
DEFAULT_ACTION_LABELS = ("REDACT", "KEEP", "REVIEW")
DEFAULT_SEMANTIC_ROLE_LABELS = ("PRIVATE", "CURRICULAR", "AMBIGUOUS")

RESULTS_HEADER = "commit\tstage\tprimary_metric\tstatus\tdescription\n"
