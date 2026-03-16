from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import DataCollatorWithPadding, TrainingArguments, set_seed

from contextshift_deid.action_features import build_action_prompt
from contextshift_deid.constants import ACTION_DIR, DEFAULT_ACTION_LABELS, PREDICTIONS_DIR, RUNS_DIR
from contextshift_deid.data import validate_action_records
from contextshift_deid.hf import (
    OriginalFormatSaveTrainer,
    load_sequence_classification_model,
    load_tokenizer,
    resolve_model_name_or_path,
)


def _binary_label_names() -> tuple[str, ...]:
    return tuple(label for label in DEFAULT_ACTION_LABELS if label != "REVIEW")


def _map_action_label(label: str, *, label_mode: str, binary_review_handling: str) -> str | None:
    if label_mode == "multiclass":
        return label
    if label != "REVIEW":
        return label
    if binary_review_handling == "drop":
        return None
    if binary_review_handling == "keep":
        return "KEEP"
    return "REDACT"


def _load_split(path: Path, *, label_mode: str, binary_review_handling: str) -> list[dict]:
    rows: list[dict] = []
    for record in validate_action_records(path):
        mapped_label = _map_action_label(
            record.action_label,
            label_mode=label_mode,
            binary_review_handling=binary_review_handling,
        )
        if mapped_label is None:
            continue
        rows.append(
            {
                "id": record.id,
                "text": build_action_prompt(record),
                "label": mapped_label,
                "subject": record.subject,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the REDACT/KEEP/REVIEW action model.")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--train-file", type=Path, default=ACTION_DIR / "train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=ACTION_DIR / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "action")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--selection-metric",
        choices=["redact_recall", "macro_f1", "accuracy"],
        default="redact_recall",
        help="Dev metric used to select the best checkpoint.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--label-mode", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument(
        "--binary-review-handling",
        choices=["drop", "keep", "redact"],
        default="drop",
        help="How REVIEW rows are handled when --label-mode=binary.",
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=PREDICTIONS_DIR / "action_dev_predictions.jsonl",
        help="Where to write dev-set predictions after training.",
    )
    args = parser.parse_args(argv)
    set_seed(args.seed)

    if args.label_mode == "multiclass":
        label_names = list(DEFAULT_ACTION_LABELS)
    else:
        label_names = list(_binary_label_names())
    label_to_id = {label: index for index, label in enumerate(label_names)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    train_records = _load_split(
        args.train_file,
        label_mode=args.label_mode,
        binary_review_handling=args.binary_review_handling,
    )
    dev_records = _load_split(
        args.dev_file,
        label_mode=args.label_mode,
        binary_review_handling=args.binary_review_handling,
    )

    resolved_model = resolve_model_name_or_path(args.model)
    tokenizer = load_tokenizer(resolved_model)
    model = load_sequence_classification_model(
        resolved_model,
        num_labels=len(label_names),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    train_dataset = Dataset.from_list(train_records)
    dev_dataset = Dataset.from_list(dev_records)
    train_columns = train_dataset.column_names
    dev_columns = dev_dataset.column_names

    def tokenize_batch(examples):
        encoded = tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        encoded["labels"] = [label_to_id[label] for label in examples["label"]]
        return encoded

    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=train_columns)
    dev_dataset = dev_dataset.map(tokenize_batch, batched=True, remove_columns=dev_columns)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_prediction):
        predictions, labels = eval_prediction
        predicted_ids = predictions.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, predicted_ids),
            "macro_f1": f1_score(labels, predicted_ids, average="macro", zero_division=0),
            "redact_recall": recall_score(
                labels,
                predicted_ids,
                labels=[label_to_id["REDACT"]],
                average=None,
                zero_division=0,
            )[0],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{args.selection_metric}",
        greater_is_better=True,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to=[],
    )
    if args.data_seed is not None:
        training_args.data_seed = args.data_seed

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
    metrics_with_seed = {
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        **metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_with_seed, indent=2), encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": str(args.model),
                "resolved_model": resolved_model,
                "train_file": str(args.train_file),
                "dev_file": str(args.dev_file),
                "output_dir": str(args.output_dir),
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "selection_metric": args.selection_metric,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "label_mode": args.label_mode,
                "binary_review_handling": args.binary_review_handling,
                "prediction_file": str(args.prediction_file),
                "seed": args.seed,
                "data_seed": args.data_seed,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.prediction_file.parent.mkdir(parents=True, exist_ok=True)
    predicted_label_ids = prediction_output.predictions.argmax(axis=-1)
    probability_rows = torch.softmax(torch.tensor(prediction_output.predictions), dim=-1).tolist()
    with args.prediction_file.open("w", encoding="utf-8") as handle:
        for record, predicted_label_id, probability_row in zip(dev_records, predicted_label_ids, probability_rows):
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
                        "confidence": float(max(probability_row)),
                        "probabilities": probability_map,
                    }
                )
                + "\n"
            )
    print(json.dumps(metrics_with_seed, indent=2))


if __name__ == "__main__":
    main()
