from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from torch import nn
from transformers.modeling_outputs import ModelOutput

from .constants import DEFAULT_SEMANTIC_ROLE_LABELS
from .hf import load_sequence_classification_model, resolve_model_name_or_path

SEMANTIC_ROLE_HEAD_FILENAME = "semantic_role_head.safetensors"
SEMANTIC_ROLE_HEAD_METADATA_FILENAME = "semantic_role_head.json"
SEMANTIC_ROLE_IGNORE_INDEX = -100


@dataclass
class ActionModelOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    semantic_role_logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


def _hidden_size_from_config(config: Any) -> int:
    for key in ("hidden_size", "dim", "d_model"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(f"Unable to infer encoder hidden size from config type {type(config).__name__}")


def _dropout_probability_from_config(config: Any) -> float:
    for key in ("classifier_dropout", "cls_dropout", "hidden_dropout_prob", "dropout"):
        value = getattr(config, key, None)
        if isinstance(value, (int, float)) and value >= 0.0:
            return float(value)
    return 0.1


def _semantic_head_metadata_path(model_path: Path) -> Path:
    return model_path / SEMANTIC_ROLE_HEAD_METADATA_FILENAME


def _semantic_head_weights_path(model_path: Path) -> Path:
    return model_path / SEMANTIC_ROLE_HEAD_FILENAME


def semantic_role_head_available(model_name_or_path: str | Path) -> bool:
    resolved = Path(resolve_model_name_or_path(model_name_or_path))
    if not resolved.exists() or not resolved.is_dir():
        return False
    return _semantic_head_metadata_path(resolved).exists() and _semantic_head_weights_path(resolved).exists()


class MultitaskActionModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        enable_semantic_role_head: bool,
        semantic_role_label_names: tuple[str, ...] = DEFAULT_SEMANTIC_ROLE_LABELS,
        semantic_role_loss_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.main_input_name = getattr(base_model, "main_input_name", "input_ids")
        self._keys_to_ignore_on_save = getattr(base_model, "_keys_to_ignore_on_save", None)
        self._keys_to_ignore_on_load_missing = getattr(base_model, "_keys_to_ignore_on_load_missing", None)
        self._keys_to_ignore_on_load_unexpected = getattr(base_model, "_keys_to_ignore_on_load_unexpected", None)
        self.supports_gradient_checkpointing = bool(
            getattr(base_model, "supports_gradient_checkpointing", False)
        )
        self.semantic_role_loss_weight = float(semantic_role_loss_weight)
        self.semantic_role_label_names = tuple(semantic_role_label_names)
        self.semantic_role_label_to_id = {
            label: index
            for index, label in enumerate(self.semantic_role_label_names)
        }
        self.semantic_role_id_to_label = {
            index: label
            for label, index in self.semantic_role_label_to_id.items()
        }
        self._supports_semantic_role_head = bool(enable_semantic_role_head)
        if enable_semantic_role_head:
            hidden_size = _hidden_size_from_config(self.config)
            dropout_probability = _dropout_probability_from_config(self.config)
            self.semantic_role_dropout = nn.Dropout(dropout_probability)
            self.semantic_role_classifier = nn.Linear(hidden_size, len(self.semantic_role_label_names))
        else:
            self.semantic_role_dropout = None
            self.semantic_role_classifier = None

    @property
    def supports_semantic_role_head(self) -> bool:
        return self.semantic_role_classifier is not None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.base_model.gradient_checkpointing_enable()
            else:
                self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        semantic_role_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ActionModelOutput:
        requested_hidden_states = bool(kwargs.pop("output_hidden_states", False))
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            labels=None,
            output_hidden_states=requested_hidden_states or self.supports_semantic_role_head,
            return_dict=True,
            **kwargs,
        )
        action_logits = outputs.logits
        action_loss = None
        if labels is not None:
            action_loss = F.cross_entropy(action_logits, labels)

        semantic_role_logits = None
        semantic_role_loss = None
        if self.supports_semantic_role_head:
            if outputs.hidden_states is None:
                raise ValueError("Semantic-role head requires hidden states, but the base model did not return them.")
            pooled_output = outputs.hidden_states[-1][:, 0]
            semantic_role_logits = self.semantic_role_classifier(self.semantic_role_dropout(pooled_output))
            if semantic_role_labels is not None:
                semantic_role_loss = F.cross_entropy(
                    semantic_role_logits,
                    semantic_role_labels,
                    ignore_index=SEMANTIC_ROLE_IGNORE_INDEX,
                )

        loss = action_loss
        if semantic_role_loss is not None:
            loss = semantic_role_loss * self.semantic_role_loss_weight if loss is None else loss + (
                self.semantic_role_loss_weight * semantic_role_loss
            )

        return ActionModelOutput(
            loss=loss,
            logits=action_logits,
            semantic_role_logits=semantic_role_logits,
            hidden_states=outputs.hidden_states if requested_hidden_states else None,
            attentions=outputs.attentions,
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        state_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_state = self.state_dict() if state_dict is None else state_dict

        base_state_dict = {
            key.removeprefix("base_model."): value
            for key, value in combined_state.items()
            if key.startswith("base_model.")
        }
        self.base_model.save_pretrained(
            output_dir,
            state_dict=base_state_dict,
            **kwargs,
        )

        head_metadata_path = _semantic_head_metadata_path(output_dir)
        head_weights_path = _semantic_head_weights_path(output_dir)
        if not self.supports_semantic_role_head:
            if head_metadata_path.exists():
                head_metadata_path.unlink()
            if head_weights_path.exists():
                head_weights_path.unlink()
            return

        head_state = {
            key.removeprefix("semantic_role_classifier."): value
            for key, value in combined_state.items()
            if key.startswith("semantic_role_classifier.")
        }
        save_safetensors_file(head_state, str(head_weights_path))
        head_metadata_path.write_text(
            json.dumps(
                {
                    "label_names": list(self.semantic_role_label_names),
                    "semantic_role_loss_weight": self.semantic_role_loss_weight,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if state_dict and not any(str(key).startswith("base_model.") for key in state_dict):
            state_dict = {
                (
                    key
                    if str(key).startswith("semantic_role_classifier.")
                    else f"base_model.{key}"
                ): value
                for key, value in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def load_action_model(
    model_name_or_path: str | Path,
    *,
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    enable_semantic_role_head: bool = False,
    semantic_role_label_names: tuple[str, ...] = DEFAULT_SEMANTIC_ROLE_LABELS,
    semantic_role_loss_weight: float = 0.3,
) -> MultitaskActionModel:
    resolved_model = resolve_model_name_or_path(model_name_or_path)
    resolved_path = Path(resolved_model)
    head_metadata: dict[str, Any] | None = None
    if resolved_path.exists() and resolved_path.is_dir():
        metadata_path = _semantic_head_metadata_path(resolved_path)
        if metadata_path.exists():
            head_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label_names = tuple(head_metadata.get("label_names") or semantic_role_label_names)
            if label_names:
                semantic_role_label_names = label_names
            if head_metadata.get("semantic_role_loss_weight") is not None:
                semantic_role_loss_weight = float(head_metadata["semantic_role_loss_weight"])

    base_model = load_sequence_classification_model(
        resolved_model,
        **{
            key: value
            for key, value in {
                "num_labels": num_labels,
                "id2label": id2label,
                "label2id": label2id,
            }.items()
            if value is not None
        },
    )
    model = MultitaskActionModel(
        base_model,
        enable_semantic_role_head=enable_semantic_role_head or head_metadata is not None,
        semantic_role_label_names=semantic_role_label_names,
        semantic_role_loss_weight=semantic_role_loss_weight,
    )

    weights_path = _semantic_head_weights_path(resolved_path)
    if model.supports_semantic_role_head and weights_path.exists():
        head_state = load_safetensors_file(str(weights_path))
        model.semantic_role_classifier.load_state_dict(dict(head_state))
    return model
