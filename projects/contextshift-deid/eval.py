from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.data import load_jsonl
from contextshift_deid.metrics import compute_action_metrics, compute_candidate_metrics


def _evaluate_action(gold_rows: list[dict], prediction_rows: list[dict]) -> dict:
    predictions_by_id = {row["id"]: row for row in prediction_rows}
    merged = []
    for row in gold_rows:
        prediction = predictions_by_id.get(row["id"])
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row['id']}")
        merged.append(
            {
                "id": row["id"],
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


def _evaluate_candidate(gold_rows: list[dict], prediction_rows: list[dict]) -> dict:
    predictions_by_id = {row["id"]: row for row in prediction_rows}
    merged = []
    for row in gold_rows:
        prediction = predictions_by_id.get(row["id"])
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row['id']}")
        merged.append(
            {
                "id": row["id"],
                "subject": row.get("subject", "unknown"),
                "gold_labels": row["labels"],
                "predicted_labels": prediction["predicted_labels"],
                "speaker_role": row.get("speaker_role"),
                "dialogue_id": row.get("dialogue_id"),
                "token_count": len(row["labels"]),
                "has_positive_label": any(label != "O" for label in row["labels"]),
            }
        )
    return compute_candidate_metrics(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold labels.")
    parser.add_argument("--stage", choices=["candidate", "action"], required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    prediction_rows = load_jsonl(args.predictions)

    if args.stage == "action":
        metrics = _evaluate_action(gold_rows, prediction_rows)
    else:
        metrics = _evaluate_candidate(gold_rows, prediction_rows)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
