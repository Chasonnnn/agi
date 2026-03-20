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

from contextshift_deid.candidate_input import (
    decode_candidate_word_level_predictions,
    IGNORE_WORD_LABEL,
    INPUT_FORMATS,
    build_candidate_model_inputs,
    resolve_candidate_input_format,
    tokenize_candidate_batch,
)
from contextshift_deid.constants import CANDIDATE_DIR, PREDICTIONS_DIR, RUNS_DIR
from contextshift_deid.data import validate_candidate_records
from contextshift_deid.hf import (
    OriginalFormatSaveTrainer,
    load_token_classification_model,
    load_tokenizer,
    resolve_model_name_or_path,
)


def _load_split(path: Path, *, input_format: str) -> list[dict]:
    return build_candidate_model_inputs(validate_candidate_records(path), input_format=input_format)


def _build_label_maps(records: list[dict]) -> tuple[list[str], dict[str, int]]:
    labels: list[str] = []
    for record in records:
        for label in record["labels"]:
            if label not in labels:
                labels.append(label)
    return labels, {label: index for index, label in enumerate(labels)}


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
        help="Legacy compatibility flag. Maps to turn_only_v1 or pair_context_v1 when --input-format is omitted.",
    )
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        help="Candidate input representation. Defaults to a mapping from --context-mode for backward compatibility.",
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    resolved_input_format = resolve_candidate_input_format(
        input_format=args.input_format,
        context_mode=args.context_mode,
    )

    train_records = _load_split(args.train_file, input_format=resolved_input_format)
    dev_records = _load_split(args.dev_file, input_format=resolved_input_format)
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
        tokenized = tokenize_candidate_batch(
            tokenizer,
            examples,
            max_length=args.max_length,
            input_format=resolved_input_format,
        )
        aligned_labels: list[list[int]] = []
        for batch_index, word_labels in enumerate(examples["model_word_labels"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            sequence_ids = tokenized.sequence_ids(batch_index=batch_index)
            previous_word_idx = None
            label_ids = []
            for word_idx, sequence_id in zip(word_ids, sequence_ids):
                if sequence_id != 0 or word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    word_label = str(word_labels[word_idx])
                    label_ids.append(label_to_id[word_label] if word_label != IGNORE_WORD_LABEL else -100)
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
        seed=args.seed,
        data_seed=args.seed,
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
                "input_format": resolved_input_format,
                "selection_metric": args.selection_metric,
                "prediction_file": str(args.prediction_file),
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    args.prediction_file.parent.mkdir(parents=True, exist_ok=True)
    predicted_token_ids = prediction_output.predictions.argmax(axis=-1)
    with args.prediction_file.open("w", encoding="utf-8") as handle:
        for record, token_prediction_row in zip(dev_records, predicted_token_ids):
            tokenized = tokenize_candidate_batch(
                tokenizer,
                {
                    "model_input_tokens": [record["model_input_tokens"]],
                    "model_input_pair_tokens": [record["model_input_pair_tokens"]],
                },
                max_length=args.max_length,
                input_format=resolved_input_format,
            )
            predicted_labels = decode_candidate_word_level_predictions(
                tokenized,
                0,
                original_token_count=len(record["labels"]),
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
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
