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

from contextshift_deid.candidate_input import (
    INPUT_FORMATS,
    build_candidate_model_inputs,
    decode_candidate_word_level_predictions,
    resolve_candidate_input_format,
    tokenize_candidate_batch,
)
from contextshift_deid.constants import CANDIDATE_DIR, PREDICTIONS_DIR
from contextshift_deid.data import validate_candidate_records
from contextshift_deid.hf import load_token_classification_model, load_tokenizer, resolve_model_name_or_path


def _load_split(path: Path, *, input_format: str) -> list[dict]:
    return build_candidate_model_inputs(validate_candidate_records(path), input_format=input_format)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run frozen candidate detector inference on a split.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved token-classification checkpoint.")
    parser.add_argument("--input-file", type=Path, default=CANDIDATE_DIR / "test.jsonl")
    parser.add_argument("--output-file", type=Path, default=PREDICTIONS_DIR / "candidate_test_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--context-mode",
        choices=("none", "pair"),
        default="none",
        help="Legacy compatibility flag. Maps to turn_only_v1 or pair_context_v1 when --input-format is omitted.",
    )
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        help="Candidate input representation. Defaults to a mapping from --context-mode for backward compatibility.",
    )
    args = parser.parse_args(argv)

    resolved_input_format = resolve_candidate_input_format(
        input_format=args.input_format,
        context_mode=args.context_mode,
    )

    records = _load_split(args.input_file, input_format=resolved_input_format)
    resolved_model = resolve_model_name_or_path(args.model)
    tokenizer = load_tokenizer(resolved_model)
    model = load_token_classification_model(resolved_model)
    id_to_label = {int(index): label for index, label in model.config.id2label.items()}

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            encoding = tokenize_candidate_batch(
                tokenizer,
                {
                    "model_input_tokens": [record["model_input_tokens"] for record in batch],
                    "model_input_pair_tokens": [record["model_input_pair_tokens"] for record in batch],
                },
                max_length=args.max_length,
                input_format=resolved_input_format,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = model(**model_inputs).logits
            predicted_token_ids = logits.argmax(dim=-1).detach().cpu().numpy()
            for batch_index, (record, token_prediction_row) in enumerate(zip(batch, predicted_token_ids)):
                predicted_labels = decode_candidate_word_level_predictions(
                    encoding,
                    batch_index,
                    original_token_count=len(record["tokens"]),
                    predicted_ids=token_prediction_row,
                    id_to_label=id_to_label,
                    decode_map=record["decode_map"],
                )
                if len(predicted_labels) != len(record["labels"]):
                    raise RuntimeError(
                        f"Prediction length mismatch for {record['id']}: "
                        f"expected {len(record['labels'])}, got {len(predicted_labels)}"
                    )
                handle.write(
                    json.dumps(
                        {
                            "id": record["id"],
                            "predicted_labels": predicted_labels,
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
                "context_mode": args.context_mode,
                "input_format": resolved_input_format,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
