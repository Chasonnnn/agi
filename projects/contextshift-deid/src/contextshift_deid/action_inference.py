from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .action_model import load_action_model
from .constants import DEFAULT_ACTION_LABELS
from .hf import load_tokenizer, resolve_model_name_or_path


def inference_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _probability_map(probability_row: list[float], *, id_to_label: dict[int, str]) -> dict[str, float]:
    probability_map = {label: 0.0 for label in DEFAULT_ACTION_LABELS}
    probability_map.update(
        {
            id_to_label[int(label_id)]: float(probability)
            for label_id, probability in enumerate(probability_row)
        }
    )
    return probability_map


def predict_action_rows(
    records: Sequence[dict[str, Any]],
    *,
    model_name_or_path: str | Path,
    batch_size: int,
    max_length: int,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    resolved_model = resolve_model_name_or_path(model_name_or_path)
    tokenizer = load_tokenizer(resolved_model)
    model = load_action_model(resolved_model)
    device = device or inference_device()
    model.to(device)
    model.eval()

    action_id_to_label = {
        int(index): label
        for index, label in model.base_model.config.id2label.items()
    }
    semantic_id_to_label = dict(model.semantic_role_id_to_label) if model.supports_semantic_role_head else {}

    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = list(records[start : start + batch_size])
            encoding = tokenizer(
                [record["text"] for record in batch],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = {key: value.to(device) for key, value in encoding.items()}
            outputs = model(**model_inputs)
            action_probabilities = torch.softmax(outputs.logits, dim=-1).detach().cpu()
            predicted_label_ids = action_probabilities.argmax(dim=-1).tolist()
            confidences = action_probabilities.max(dim=-1).values.tolist()
            semantic_probabilities = None
            semantic_predicted_ids: list[int | None] = [None] * len(batch)
            if outputs.semantic_role_logits is not None:
                semantic_probabilities = torch.softmax(outputs.semantic_role_logits, dim=-1).detach().cpu()
                semantic_predicted_ids = semantic_probabilities.argmax(dim=-1).tolist()

            for index, (record, predicted_label_id, confidence, probability_row) in enumerate(
                zip(batch, predicted_label_ids, confidences, action_probabilities.tolist())
            ):
                semantic_probability_map = None
                predicted_semantic_role = None
                if semantic_probabilities is not None:
                    semantic_probability_row = semantic_probabilities[index].tolist()
                    semantic_probability_map = {
                        semantic_id_to_label[int(label_id)]: float(probability)
                        for label_id, probability in enumerate(semantic_probability_row)
                    }
                    predicted_semantic_role = semantic_id_to_label[int(semantic_predicted_ids[index])]
                prediction_rows.append(
                    {
                        "id": record["id"],
                        "predicted_action": action_id_to_label[int(predicted_label_id)],
                        "confidence": float(confidence),
                        "probabilities": _probability_map(probability_row, id_to_label=action_id_to_label),
                        "predicted_semantic_role": predicted_semantic_role,
                        "semantic_role_probabilities": semantic_probability_map,
                    }
                )
    return prediction_rows
