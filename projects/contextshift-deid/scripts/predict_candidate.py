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

from contextshift_deid.constants import CANDIDATE_DIR, PREDICTIONS_DIR
from contextshift_deid.data import validate_candidate_records
from contextshift_deid.hf import load_token_classification_model, load_tokenizer, resolve_model_name_or_path
from contextshift_deid.tokenization import tokenize_text


def _load_split(path: Path) -> list[dict]:
    return [
        {
            "id": record.id,
            "tokens": record.tokens,
            "labels": record.labels,
            "subject": record.subject,
            "context_text": record.context_text or "",
            "context_tokens": tokenize_text(record.context_text or ""),
        }
        for record in validate_candidate_records(path)
    ]


def _decode_word_level_predictions(encoding, batch_index: int, tokens: list[str], predicted_ids, id_to_label: dict[int, str]) -> list[str]:
    word_ids = encoding.word_ids(batch_index=batch_index)
    sequence_ids = encoding.sequence_ids(batch_index=batch_index)
    word_predictions = ["O"] * len(tokens)
    previous_word_idx = None
    for token_idx, (word_idx, sequence_id) in enumerate(zip(word_ids, sequence_ids)):
        if sequence_id != 0 or word_idx is None or word_idx == previous_word_idx or word_idx >= len(tokens):
            previous_word_idx = word_idx
            continue
        word_predictions[word_idx] = id_to_label[int(predicted_ids[token_idx])]
        previous_word_idx = word_idx
    return word_predictions


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
        help="Whether to append turn-window context as a second encoder sequence.",
    )
    args = parser.parse_args(argv)

    records = _load_split(args.input_file)
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
            batch_tokens = [record["tokens"] for record in batch]
            if args.context_mode == "pair":
                batch_context_tokens = [record["context_tokens"] for record in batch]
                encoding = tokenizer(
                    batch_tokens,
                    batch_context_tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                encoding = tokenizer(
                    batch_tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=args.max_length,
                    padding=True,
                    return_tensors="pt",
                )
            model_inputs = {key: value.to(device) for key, value in encoding.items()}
            with torch.no_grad():
                logits = model(**model_inputs).logits
            predicted_token_ids = logits.argmax(dim=-1).detach().cpu().numpy()
            for batch_index, (record, token_prediction_row) in enumerate(zip(batch, predicted_token_ids)):
                predicted_labels = _decode_word_level_predictions(
                    encoding,
                    batch_index,
                    record["tokens"],
                    token_prediction_row,
                    id_to_label,
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
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
