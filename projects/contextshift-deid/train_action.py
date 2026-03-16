from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DataCollatorWithPadding, TrainingArguments, set_seed

from contextshift_deid.action_features import ACTION_INPUT_FORMAT_CHOICES, DEFAULT_ACTION_INPUT_FORMAT, build_action_prompt
from contextshift_deid.action_inference import predict_action_rows
from contextshift_deid.action_model import SEMANTIC_ROLE_IGNORE_INDEX, load_action_model
from contextshift_deid.constants import (
    ACTION_DIR,
    DEFAULT_ACTION_LABELS,
    DEFAULT_SEMANTIC_ROLE_LABELS,
    PREDICTIONS_DIR,
    RUNS_DIR,
)
from contextshift_deid.data import load_jsonl, validate_action_records
from contextshift_deid.hf import OriginalFormatSaveTrainer, load_tokenizer, resolve_model_name_or_path
from contextshift_deid.metrics import compute_action_metrics
from contextshift_deid.policy_selection import (
    DEFAULT_POLICY_SELECTION_TARGET_REVIEW_RATE,
    evaluate_direct_id_policy,
)

POLICY_SELECTION_METRIC = "selected_policy_10pct_direct_id"
SELECTION_METRIC_CHOICES = ("redact_recall", "macro_f1", "accuracy", POLICY_SELECTION_METRIC)
SEMANTIC_ROLE_HEAD_MODE_CHOICES = ("none", "multitask")
SAMPLER_MODE_CHOICES = ("none", "subject_action_balanced")
CHECKPOINT_SELECTION_DIRNAME = "checkpoint_selection"


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


def _normalize_semantic_role(label: str | None) -> str | None:
    if label is None:
        return None
    normalized = str(label).strip().upper()
    if not normalized:
        return None
    if normalized not in DEFAULT_SEMANTIC_ROLE_LABELS:
        raise ValueError(f"Unsupported semantic role label: {label}")
    return normalized


def _load_split(
    path: Path,
    *,
    label_mode: str,
    binary_review_handling: str,
    action_input_format: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
                "text": build_action_prompt(record, input_format=action_input_format),
                "label": mapped_label,
                "subject": record.subject,
                "semantic_role_label": _normalize_semantic_role(record.semantic_role),
            }
        )
    return rows


def _compute_sample_weights(records: list[dict[str, Any]], *, sampler_mode: str) -> list[float] | None:
    if sampler_mode == "none":
        return None
    if sampler_mode != "subject_action_balanced":
        raise ValueError(f"Unsupported sampler mode: {sampler_mode}")
    subject_counts = Counter(record["subject"] for record in records)
    action_counts = Counter(record["label"] for record in records)
    return [
        (1.0 / subject_counts[record["subject"]]) * (1.0 / action_counts[record["label"]])
        for record in records
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _action_metrics_from_predictions(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    merged: list[dict[str, Any]] = []
    for row in gold_rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise ValueError(f"Missing prediction for id={row_id}")
        merged.append(
            {
                "id": row_id,
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


def _checkpoint_step(checkpoint_dir: Path) -> int:
    suffix = checkpoint_dir.name.removeprefix("checkpoint-")
    return int(suffix) if suffix.isdigit() else -1


def _selection_score(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["score"]["protected_redact_rate"]),
        float(summary["score"]["worst_context_redact_recall"]),
        float(summary["score"]["selected_policy_macro_f1_10pct"]),
        -float(summary["score"]["review_rate_abs_error"]),
    )


def _copy_checkpoint_to_output_dir(checkpoint_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = {"optimizer.pt", "scheduler.pt", "rng_state.pth"}
    for item in checkpoint_dir.iterdir():
        if item.name in skipped or item.name.startswith("checkpoint-"):
            continue
        target = output_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _evaluate_checkpoint_with_policy_selection(
    *,
    checkpoint_dir: Path,
    selection_dir: Path,
    dev_gold_rows: list[dict[str, Any]],
    dev_prediction_records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    target_review_rate: float,
) -> dict[str, Any]:
    checkpoint_selection_dir = selection_dir / checkpoint_dir.name
    checkpoint_selection_dir.mkdir(parents=True, exist_ok=True)

    base_prediction_file = checkpoint_selection_dir / "dev_base_predictions.jsonl"
    direct_id_prediction_file = checkpoint_selection_dir / "dev_direct_id_predictions.jsonl"
    policy_summary_file = checkpoint_selection_dir / "policy_summary.json"
    policy_sweep_file = checkpoint_selection_dir / "policy_sweep.json"

    prediction_rows = predict_action_rows(
        dev_prediction_records,
        model_name_or_path=checkpoint_dir,
        batch_size=batch_size,
        max_length=max_length,
        device=torch.device("cpu"),
    )
    _write_jsonl(base_prediction_file, prediction_rows)
    base_metrics = _action_metrics_from_predictions(dev_gold_rows, prediction_rows)

    policy_selection = evaluate_direct_id_policy(
        dev_gold_rows,
        prediction_rows,
        target_review_rate=target_review_rate,
        fit_temperature_on_records=True,
    )
    _write_jsonl(direct_id_prediction_file, policy_selection["patched_prediction_rows"])
    policy_sweep_file.write_text(json.dumps(policy_selection["sweep_results"], indent=2), encoding="utf-8")

    selected_target = policy_selection["selected_target"]
    selected_metrics = selected_target["evaluation"]["metrics"]
    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_step": _checkpoint_step(checkpoint_dir),
        "base_prediction_file": str(base_prediction_file),
        "direct_id_prediction_file": str(direct_id_prediction_file),
        "policy_sweep_file": str(policy_sweep_file),
        "target_review_rate": target_review_rate,
        "base_metrics": base_metrics,
        "selected_target": selected_target,
        "override_summary": policy_selection["override_summary"],
        "score": {
            "protected_redact_rate": float(selected_metrics.get("protected_redact_rate") or 0.0),
            "worst_context_redact_recall": float(selected_metrics.get("worst_context_redact_recall") or 0.0),
            "selected_policy_macro_f1_10pct": float(selected_metrics["macro_f1"]),
            "review_rate_abs_error": abs(float(selected_metrics["review_rate"]) - target_review_rate),
        },
    }
    _write_json(policy_summary_file, summary)
    return summary


def _evaluate_checkpoint_with_base_metric(
    *,
    checkpoint_dir: Path,
    selection_dir: Path,
    dev_gold_rows: list[dict[str, Any]],
    dev_prediction_records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    selection_metric: str,
) -> dict[str, Any]:
    checkpoint_selection_dir = selection_dir / checkpoint_dir.name
    checkpoint_selection_dir.mkdir(parents=True, exist_ok=True)

    base_prediction_file = checkpoint_selection_dir / "dev_base_predictions.jsonl"
    summary_file = checkpoint_selection_dir / "base_metric_summary.json"
    prediction_rows = predict_action_rows(
        dev_prediction_records,
        model_name_or_path=checkpoint_dir,
        batch_size=batch_size,
        max_length=max_length,
        device=torch.device("cpu"),
    )
    _write_jsonl(base_prediction_file, prediction_rows)
    base_metrics = _action_metrics_from_predictions(dev_gold_rows, prediction_rows)
    summary = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_step": _checkpoint_step(checkpoint_dir),
        "base_prediction_file": str(base_prediction_file),
        "base_metrics": base_metrics,
        "selection_metric": selection_metric,
        "score": float(base_metrics[selection_metric]),
    }
    _write_json(summary_file, summary)
    return summary


def _select_best_checkpoint_by_base_metric(
    *,
    output_dir: Path,
    dev_gold_rows: list[dict[str, Any]],
    dev_prediction_records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    selection_metric: str,
) -> dict[str, Any]:
    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=_checkpoint_step,
    )
    if not checkpoint_dirs:
        raise ValueError(f"No epoch checkpoints found in {output_dir}")

    selection_dir = output_dir / CHECKPOINT_SELECTION_DIRNAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_summaries = [
        _evaluate_checkpoint_with_base_metric(
            checkpoint_dir=checkpoint_dir,
            selection_dir=selection_dir,
            dev_gold_rows=dev_gold_rows,
            dev_prediction_records=dev_prediction_records,
            batch_size=batch_size,
            max_length=max_length,
            selection_metric=selection_metric,
        )
        for checkpoint_dir in checkpoint_dirs
    ]
    best_checkpoint = max(
        checkpoint_summaries,
        key=lambda summary: (float(summary["score"]), float(summary["checkpoint_step"])),
    )
    selection_summary = {
        "selection_metric": selection_metric,
        "best_checkpoint": best_checkpoint,
        "checkpoints": checkpoint_summaries,
    }
    _write_json(selection_dir / "summary.json", selection_summary)
    return selection_summary


def _select_best_checkpoint_by_policy(
    *,
    output_dir: Path,
    dev_gold_rows: list[dict[str, Any]],
    dev_prediction_records: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    target_review_rate: float,
) -> dict[str, Any]:
    checkpoint_dirs = sorted(
        [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
        key=_checkpoint_step,
    )
    if not checkpoint_dirs:
        raise ValueError(f"No epoch checkpoints found in {output_dir}")

    selection_dir = output_dir / CHECKPOINT_SELECTION_DIRNAME
    selection_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_summaries = [
        _evaluate_checkpoint_with_policy_selection(
            checkpoint_dir=checkpoint_dir,
            selection_dir=selection_dir,
            dev_gold_rows=dev_gold_rows,
            dev_prediction_records=dev_prediction_records,
            batch_size=batch_size,
            max_length=max_length,
            target_review_rate=target_review_rate,
        )
        for checkpoint_dir in checkpoint_dirs
    ]
    best_checkpoint = max(checkpoint_summaries, key=_selection_score)
    selection_summary = {
        "selection_metric": POLICY_SELECTION_METRIC,
        "target_review_rate": target_review_rate,
        "best_checkpoint": best_checkpoint,
        "checkpoints": checkpoint_summaries,
    }
    _write_json(selection_dir / "summary.json", selection_summary)
    return selection_summary


class ActionTrainer(OriginalFormatSaveTrainer):
    def __init__(self, *args: Any, sample_weights: list[float] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer requires a train dataset before constructing the dataloader.")
        if not self.sample_weights:
            return super().get_train_dataloader()
        if len(self.sample_weights) != len(self.train_dataset):
            raise ValueError("Sample-weight count does not match the train dataset size.")
        sampler = WeightedRandomSampler(
            weights=torch.tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(self.sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the REDACT/KEEP/REVIEW action model.")
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--train-file", type=Path, default=ACTION_DIR / "train.jsonl")
    parser.add_argument("--dev-file", type=Path, default=ACTION_DIR / "dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "action")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default="redact_recall",
        help="Dev metric used to select the best checkpoint.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--label-mode", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument(
        "--action-input-format",
        choices=ACTION_INPUT_FORMAT_CHOICES,
        default=DEFAULT_ACTION_INPUT_FORMAT,
    )
    parser.add_argument(
        "--semantic-role-head-mode",
        choices=SEMANTIC_ROLE_HEAD_MODE_CHOICES,
        default="none",
    )
    parser.add_argument("--semantic-role-loss-weight", type=float, default=0.3)
    parser.add_argument(
        "--sampler-mode",
        choices=SAMPLER_MODE_CHOICES,
        default="subject_action_balanced",
    )
    parser.add_argument(
        "--policy-selection-target-review-rate",
        type=float,
        default=DEFAULT_POLICY_SELECTION_TARGET_REVIEW_RATE,
    )
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
    semantic_role_to_id = {
        label: index
        for index, label in enumerate(DEFAULT_SEMANTIC_ROLE_LABELS)
    }

    train_records = _load_split(
        args.train_file,
        label_mode=args.label_mode,
        binary_review_handling=args.binary_review_handling,
        action_input_format=args.action_input_format,
    )
    dev_records = _load_split(
        args.dev_file,
        label_mode=args.label_mode,
        binary_review_handling=args.binary_review_handling,
        action_input_format=args.action_input_format,
    )

    resolved_model = resolve_model_name_or_path(args.model)
    tokenizer = load_tokenizer(resolved_model)
    model = load_action_model(
        resolved_model,
        num_labels=len(label_names),
        id2label=id_to_label,
        label2id=label_to_id,
        enable_semantic_role_head=args.semantic_role_head_mode == "multitask",
        semantic_role_label_names=DEFAULT_SEMANTIC_ROLE_LABELS,
        semantic_role_loss_weight=args.semantic_role_loss_weight,
    )

    train_dataset = Dataset.from_list(train_records)
    dev_dataset = Dataset.from_list(dev_records)
    train_columns = train_dataset.column_names
    dev_columns = dev_dataset.column_names

    def tokenize_batch(examples: dict[str, list[Any]]) -> dict[str, Any]:
        encoded = tokenizer(examples["text"], truncation=True, max_length=args.max_length)
        encoded["labels"] = [label_to_id[label] for label in examples["label"]]
        encoded["semantic_role_labels"] = [
            SEMANTIC_ROLE_IGNORE_INDEX if label is None else semantic_role_to_id[label]
            for label in examples["semantic_role_label"]
        ]
        return encoded

    train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=train_columns)
    dev_dataset = dev_dataset.map(tokenize_batch, batched=True, remove_columns=dev_columns)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    sample_weights = _compute_sample_weights(train_records, sampler_mode=args.sampler_mode)

    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]
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
    training_args_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": False,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_strategy": "epoch",
        "save_total_limit": None,
        "report_to": [],
    }
    training_args = TrainingArguments(**training_args_kwargs)
    if args.data_seed is not None:
        training_args.data_seed = args.data_seed

    trainer = ActionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        sample_weights=sample_weights,
    )
    trainer.train()

    checkpoint_selection_summary = None
    dev_record_ids = {record["id"] for record in dev_records}
    dev_gold_rows = [
        row
        for row in load_jsonl(args.dev_file)
        if str(row["id"]) in dev_record_ids
    ]
    base_metrics: dict[str, Any]
    prediction_rows: list[dict[str, Any]]

    trainer.model.to(torch.device("cpu"))
    del model
    del trainer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if args.selection_metric == POLICY_SELECTION_METRIC:
        checkpoint_selection_summary = _select_best_checkpoint_by_policy(
            output_dir=args.output_dir,
            dev_gold_rows=dev_gold_rows,
            dev_prediction_records=dev_records,
            batch_size=args.batch_size,
            max_length=args.max_length,
            target_review_rate=args.policy_selection_target_review_rate,
        )
    else:
        checkpoint_selection_summary = _select_best_checkpoint_by_base_metric(
            output_dir=args.output_dir,
            dev_gold_rows=dev_gold_rows,
            dev_prediction_records=dev_records,
            batch_size=args.batch_size,
            max_length=args.max_length,
            selection_metric=args.selection_metric,
        )
    best_checkpoint = checkpoint_selection_summary["best_checkpoint"]
    _copy_checkpoint_to_output_dir(Path(best_checkpoint["checkpoint"]), args.output_dir)
    prediction_rows = load_jsonl(Path(best_checkpoint["base_prediction_file"]))
    args.prediction_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint["base_prediction_file"], args.prediction_file)
    base_metrics = dict(best_checkpoint["base_metrics"])

    metrics_with_seed: dict[str, Any] = {
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "eval_accuracy": float(base_metrics["accuracy"]),
        "eval_macro_f1": float(base_metrics["macro_f1"]),
        "eval_redact_recall": float(base_metrics["redact_recall"]),
    }
    if args.selection_metric == POLICY_SELECTION_METRIC and checkpoint_selection_summary is not None:
        best_checkpoint = checkpoint_selection_summary["best_checkpoint"]
        selected_metrics = best_checkpoint["selected_target"]["evaluation"]["metrics"]
        metrics_with_seed.update(
            {
                "selected_policy_target_review_rate": args.policy_selection_target_review_rate,
                "selected_policy_best_checkpoint": best_checkpoint["checkpoint"],
                "selected_policy_protected_redact_rate": float(selected_metrics.get("protected_redact_rate") or 0.0),
                "selected_policy_worst_context_redact_recall": float(
                    selected_metrics.get("worst_context_redact_recall") or 0.0
                ),
                "selected_policy_macro_f1": float(selected_metrics["macro_f1"]),
                "selected_policy_review_rate": float(selected_metrics["review_rate"]),
                "selected_policy_review_rate_abs_error": float(best_checkpoint["score"]["review_rate_abs_error"]),
            }
        )

    _write_json(args.output_dir / "metrics.json", metrics_with_seed)
    _write_json(
        args.output_dir / "metadata.json",
        {
            "model": str(args.model),
            "resolved_model": resolved_model,
            "train_file": str(args.train_file),
            "dev_file": str(args.dev_file),
            "output_dir": str(args.output_dir),
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_length": args.max_length,
            "selection_metric": args.selection_metric,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "action_input_format": args.action_input_format,
            "label_mode": args.label_mode,
            "binary_review_handling": args.binary_review_handling,
            "semantic_role_head_mode": args.semantic_role_head_mode,
            "semantic_role_loss_weight": args.semantic_role_loss_weight,
            "sampler_mode": args.sampler_mode,
            "policy_selection_target_review_rate": args.policy_selection_target_review_rate,
            "prediction_file": str(args.prediction_file),
            "seed": args.seed,
            "data_seed": args.data_seed,
            "metrics": metrics_with_seed,
            "checkpoint_selection_summary": (
                None
                if checkpoint_selection_summary is None
                else str(args.output_dir / CHECKPOINT_SELECTION_DIRNAME / "summary.json")
            ),
        },
    )
    print(json.dumps(metrics_with_seed, indent=2))


if __name__ == "__main__":
    main()
