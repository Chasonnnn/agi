from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.action_features import ACTION_INPUT_FORMAT_CHOICES, build_action_prompt
from contextshift_deid.action_inference import inference_device, predict_action_rows
from contextshift_deid.constants import ACTION_DIR, PREDICTIONS_DIR
from contextshift_deid.data import validate_action_records


def _metadata_candidates(model_path: Path) -> list[Path]:
    candidates = [model_path / "metadata.json"]
    if model_path.name.startswith("checkpoint-"):
        candidates.append(model_path.parent / "metadata.json")
    return candidates


def _checkpoint_action_input_format(model_path: Path) -> str | None:
    for metadata_path in _metadata_candidates(model_path):
        if not metadata_path.exists():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        value = payload.get("action_input_format")
        if value in ACTION_INPUT_FORMAT_CHOICES:
            return value
    return None


def _resolve_action_input_format(model_path: Path, configured: str | None) -> str:
    if configured is not None:
        return configured
    inferred = _checkpoint_action_input_format(model_path)
    if inferred is not None:
        return inferred
    return "flat_v1"


def _load_split(path: Path, *, action_input_format: str) -> list[dict]:
    return [
        {
            "id": record.id,
            "text": build_action_prompt(record, input_format=action_input_format),
        }
        for record in validate_action_records(path)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run frozen action-model inference on a split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved sequence-classification checkpoint.")
    parser.add_argument("--input-file", type=Path, default=ACTION_DIR / "test.jsonl")
    parser.add_argument("--output-file", type=Path, default=PREDICTIONS_DIR / "action_test_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--action-input-format", choices=ACTION_INPUT_FORMAT_CHOICES, default=None)
    args = parser.parse_args(argv)

    action_input_format = _resolve_action_input_format(args.model, args.action_input_format)
    records = _load_split(args.input_file, action_input_format=action_input_format)
    prediction_rows = predict_action_rows(
        records,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row) + "\n")

    print(
        json.dumps(
            {
                "model": str(args.model),
                "input_file": str(args.input_file),
                "output_file": str(args.output_file),
                "record_count": len(records),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "action_input_format": action_input_format,
                "device": str(inference_device()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
