from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification, TrainingArguments

from contextshift_deid.constants import CANDIDATE_DIR, PREDICTIONS_DIR, RUNS_DIR
from contextshift_deid.data import validate_candidate_records
from contextshift_deid.hf import (
    OriginalFormatSaveTrainer,
    load_token_classification_model,
    load_tokenizer,
    resolve_model_name_or_path,
)
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


def _build_label_maps(records: list[dict]) -> tuple[list[str], dict[str, int]]:
    labels: list[str] = []
    for record in records:
        for label in record["labels"]:
            if label not in labels:
                labels.append(label)
    return labels, {label: index for index, label in enumerate(labels)}


def _tokenize_batch(tokenizer, batch: dict, *, max_length: int, context_mode: str):
    if context_mode == "pair":
        return tokenizer(
            batch["tokens"],
            batch["context_tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
    return tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )


def _decode_word_level_predictions(
    tokenizer,
    tokens: list[str],
    predicted_ids,
    *,
    max_length: int,
    id_to_label: dict[int, str],
    context_tokens: list[str],
    context_mode: str,
) -> list[str]:
    tokenized = _tokenize_batch(
        tokenizer,
        {"tokens": tokens, "context_tokens": context_tokens},
        max_length=max_length,
        context_mode=context_mode,
    )
    word_ids = tokenized.word_ids()
    sequence_ids = tokenized.sequence_ids()
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
    parser = argparse.ArgumentParser(description="Train the suspicious-span candidate detector.")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--train-file", type=Path, default=CANDIDATE_DIR / "train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=CANDIDATE_DIR / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "candidate")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--context-mode",
        choices=("none", "pair"),
        default="none",
        help="Whether to append the turn-window context as a second encoder sequence.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("recall", "f1", "precision"),
        default="recall",
        help="Metric used to select the best checkpoint. Defaults to recall for recall-first candidate training.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=PREDICTIONS_DIR / "candidate_dev_predictions.jsonl",
        help="Where to write dev-set predictions after training.",
    )
    args = parser.parse_args(argv)

    train_records = _load_split(args.train_file)
    dev_records = _load_split(args.dev_file)
    label_names, label_to_id = _build_label_maps(train_records + dev_records)
    id_to_label = {index: label for label, index in label_to_id.items()}

    resolved_model = resolve_model_name_or_path(args.model)
    tokenizer = load_tokenizer(resolved_model)
    model = load_token_classification_model(
        resolved_model,
        num_labels=len(label_names),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_dataset = Dataset.from_list(train_records)
    dev_dataset = Dataset.from_list(dev_records)
    train_columns = train_dataset.column_names
    dev_columns = dev_dataset.column_names

    def tokenize_and_align_labels(examples):
        tokenized = _tokenize_batch(
            tokenizer,
            examples,
            max_length=args.max_length,
            context_mode=args.context_mode,
        )
        aligned_labels = []
        for batch_index, labels in enumerate(examples["labels"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            sequence_ids = tokenized.sequence_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids = []
            for word_idx, sequence_id in zip(word_ids, sequence_ids):
                if sequence_id != 0 or word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[labels[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=train_columns)
    dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dev_columns)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        predicted_ids = predictions.argmax(axis=-1)
        true_predictions = []
        true_labels = []
        for prediction_row, label_row in zip(predicted_ids, labels):
            pred_labels = []
            gold_labels = []
            for predicted_id, label_id in zip(prediction_row, label_row):
                if label_id == -100:
                    continue
                pred_labels.append(id_to_label[int(predicted_id)])
                gold_labels.append(id_to_label[int(label_id)])
            true_predictions.append(pred_labels)
            true_labels.append(gold_labels)
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.selection_metric}",
        greater_is_better=True,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to=[],
    )

    trainer = OriginalFormatSaveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    prediction_output = trainer.predict(dev_dataset)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": str(args.model),
                "resolved_model": str(resolved_model),
                "train_file": str(args.train_file),
                "dev_file": str(args.dev_file),
                "output_dir": str(args.output_dir),
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "context_mode": args.context_mode,
                "selection_metric": args.selection_metric,
                "prediction_file": str(args.prediction_file),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.prediction_file.parent.mkdir(parents=True, exist_ok=True)
    predicted_token_ids = prediction_output.predictions.argmax(axis=-1)
    with args.prediction_file.open("w", encoding="utf-8") as handle:
        for record, token_prediction_row in zip(dev_records, predicted_token_ids):
            predicted_labels = _decode_word_level_predictions(
                tokenizer,
                record["tokens"],
                token_prediction_row,
                max_length=args.max_length,
                id_to_label=id_to_label,
                context_tokens=record["context_tokens"],
                context_mode=args.context_mode,
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
