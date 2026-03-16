from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, Trainer
from transformers.trainer import TRAINING_ARGS_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = REPO_ROOT / "models"
LOCAL_MODEL_ALIASES = {
    "roberta-base": "roberta-base",
    "FacebookAI/roberta-base": "roberta-base",
    "distilroberta-base": "distilroberta-base",
    "ModernBERT-base": "ModernBERT-base",
    "answerdotai/ModernBERT-base": "ModernBERT-base",
}


def resolve_model_name_or_path(model_name_or_path: str | Path) -> str:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate)

    model_key = str(model_name_or_path)
    local_name = LOCAL_MODEL_ALIASES.get(model_key)
    if local_name is None and "/" not in model_key:
        fallback_candidate = LOCAL_MODEL_DIR / model_key
        if fallback_candidate.exists():
            return str(fallback_candidate)
        return model_key

    if local_name is None:
        return model_key

    local_candidate = LOCAL_MODEL_DIR / local_name
    if local_candidate.exists():
        return str(local_candidate)
    return model_key


def load_tokenizer(model_name_or_path: str | Path):
    model_name_or_path = resolve_model_name_or_path(model_name_or_path)
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception as exc:
        message = str(exc).lower()
        if "sentencepiece" in message or "tiktoken" in message or "spm.model" in message:
            return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        raise


def _normalize_layernorm_key(name: str) -> str:
    return name.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")


def _normalize_layernorm_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized: dict[str, Any] = {}
    changed = False
    for key, value in state_dict.items():
        new_key = _normalize_layernorm_key(key)
        if new_key in normalized:
            raise ValueError(f"Duplicate key after LayerNorm normalization: {new_key}")
        normalized[new_key] = value
        changed = changed or new_key != key
    return normalized, changed


def _load_checkpoint_state_dict(model_path: Path) -> dict[str, Any] | None:
    if not model_path.exists() or not model_path.is_dir():
        return None
    safetensors_path = model_path / "model.safetensors"
    if safetensors_path.exists():
        return dict(load_safetensors_file(str(safetensors_path)))
    bin_path = model_path / "pytorch_model.bin"
    if bin_path.exists():
        return dict(torch.load(bin_path, map_location="cpu", weights_only=True))
    return None


def _compatible_state_dict_for(model_name_or_path: str | Path) -> tuple[str, dict[str, Any] | None]:
    resolved = resolve_model_name_or_path(model_name_or_path)
    state_dict = _load_checkpoint_state_dict(Path(resolved))
    if state_dict is None:
        return resolved, None
    normalized_state_dict, changed = _normalize_layernorm_state_dict(state_dict)
    if not changed:
        return resolved, None
    return resolved, normalized_state_dict


def _load_model_from_config_with_state_dict(model_cls, resolved: str, state_dict: dict[str, Any], **kwargs: Any):
    config = AutoConfig.from_pretrained(resolved)
    for key in ("num_labels", "id2label", "label2id"):
        if key in kwargs:
            setattr(config, key, kwargs.pop(key))
    model = model_cls.from_config(config, **kwargs)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_sequence_classification_model(model_name_or_path: str | Path, **kwargs: Any):
    resolved, state_dict = _compatible_state_dict_for(model_name_or_path)
    if state_dict is None:
        return AutoModelForSequenceClassification.from_pretrained(resolved, **kwargs)
    return _load_model_from_config_with_state_dict(
        AutoModelForSequenceClassification,
        resolved,
        state_dict,
        **kwargs,
    )


def load_token_classification_model(model_name_or_path: str | Path, **kwargs: Any):
    resolved, state_dict = _compatible_state_dict_for(model_name_or_path)
    if state_dict is None:
        return AutoModelForTokenClassification.from_pretrained(resolved, **kwargs)
    return _load_model_from_config_with_state_dict(
        AutoModelForTokenClassification,
        resolved,
        state_dict,
        **kwargs,
    )


class OriginalFormatSaveTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if state_dict is None:
            state_dict = model_to_save.state_dict()
        model_to_save.save_pretrained(
            output_dir,
            state_dict=state_dict,
            save_original_format=False,
        )
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
