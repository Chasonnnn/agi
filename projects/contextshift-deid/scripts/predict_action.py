from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.action_features import build_action_prompt
from contextshift_deid.constants import ACTION_DIR, DEFAULT_ACTION_LABELS, PREDICTIONS_DIR
from contextshift_deid.data import validate_action_records
from contextshift_deid.hf import load_sequence_classification_model, load_tokenizer, resolve_model_name_or_path


def _load_split(path: Path) -> list[dict]:
    return [
        {
            "id": record.id,
            "text": build_action_prompt(record),
        }
        for record in validate_action_records(path)
    ]


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run frozen action-model inference on a split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved sequence-classification checkpoint.")
    parser.add_argument("--input-file", type=Path, default=ACTION_DIR / "test.jsonl")
    parser.add_argument("--output-file", type=Path, default=PREDICTIONS_DIR / "action_test_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args(argv)

    records = _load_split(args.input_file)
    resolved_model = resolve_model_name_or_path(args.model)
    tokenizer = load_tokenizer(resolved_model)
    model = load_sequence_classification_model(resolved_model)
    id_to_label = {int(index): label for index, label in model.config.id2label.items()}
    device = _device()
    model.to(device)
    model.eval()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            encoding = tokenizer(
                [record["text"] for record in batch],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1).detach().cpu()
            predicted_label_ids = probabilities.argmax(dim=-1).tolist()
            confidences = probabilities.max(dim=-1).values.tolist()
            for record, predicted_label_id, confidence, probability_row in zip(
                batch,
                predicted_label_ids,
                confidences,
                probabilities.tolist(),
            ):
                probability_map = {label: 0.0 for label in DEFAULT_ACTION_LABELS}
                probability_map.update(
                    {
                        id_to_label[int(label_id)]: float(probability)
                        for label_id, probability in enumerate(probability_row)
                    }
                )
                handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "predicted_action": id_to_label[int(predicted_label_id)],
                            "confidence": float(confidence),
                            "probabilities": probability_map,
                        }
                    )
                    + "\n"
                )

    print(
        json.dumps(
            {
                "model": str(args.model),
                "input_file": str(args.input_file),
                "output_file": str(args.output_file),
                "record_count": len(records),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
