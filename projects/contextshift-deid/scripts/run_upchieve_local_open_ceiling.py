from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from statistics import mean, median
import sys
from time import perf_counter, time
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, LEGACY_EXPERIMENTS_DIR, PREDICTIONS_DIR
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import create_experiment_run, slugify, write_run_metadata
from contextshift_deid.metrics import compute_action_metrics

DEFAULT_GOLD_FILE = ACTION_DIR / "upchieve_english_social_dev.jsonl"
DEFAULT_MODEL_PATH = ROOT / "models/Qwen3.5-0.8B"
DEFAULT_RUN_NAME = "upchieve-local-open-ceiling"
DEFAULT_SELECTED_REVIEW_RATE = 0.10
DEFAULT_COMPARISON_SUMMARY_FILE = (
    LEGACY_EXPERIMENTS_DIR / "20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384" / "summary.json"
)
DEFAULT_THINKING_MODE = "non-thinking"
DEFAULT_TORCH_DTYPE = "auto"
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_SEED = 17
ACTION_LABELS = ("REDACT", "KEEP", "REVIEW")
SEMANTIC_ROLES = ("PRIVATE", "CURRICULAR", "AMBIGUOUS")
INPUT_FIELD_SPECS = (
    {"source_field": "subject", "prompt_label": "subject", "required": True},
    {"source_field": "context_text", "prompt_label": "context", "required": True},
    {"source_field": "span_text", "prompt_label": "span", "required": True},
    {"source_field": "anchor_text", "prompt_label": "anchor", "required": False},
    {"source_field": "speaker_role", "prompt_label": "speaker_role", "required": False},
    {"source_field": "entity_type", "prompt_label": "entity_type", "required": False},
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action_label": {
            "type": "string",
            "enum": list(ACTION_LABELS),
        },
        "semantic_role": {
            "type": "string",
            "enum": list(SEMANTIC_ROLES),
        },
    },
    "required": ["action_label", "semantic_role"],
    "additionalProperties": False,
}
CODEBOOK_SOURCE = "docs/codebook_v3.md"
CODEBOOK_EXCERPT = """The action task is:

Given a suspicious span in context, should the system REDACT it, KEEP it, or send it to REVIEW?

The pilot also requires one semantic role:
- PRIVATE when the span refers to a real participant or identifying detail
- CURRICULAR when the span is part of lesson content
- AMBIGUOUS when the context does not support a confident binary reading

Label definitions:

REDACT:
- Use REDACT when the tag stands for a real participant, school, location, or other identifying detail in the tutoring exchange.
- Typical REDACT cases: student or volunteer self-introductions; real teacher or counselor names; school names tied to the current student; direct contact or account-style tags; local places, hometowns, or other personal locations.
- Default semantic role: PRIVATE

KEEP:
- Use KEEP when the tag stands for curricular content that is necessary to preserve educational meaning.
- Typical KEEP cases: authors, characters, books, poems, and fictional settings in English; historical figures, countries, wars, movements, and textbook places in Social Studies; course or assignment references used as instructional framing.
- Default semantic role: CURRICULAR

REVIEW:
- Use REVIEW when the context honestly supports both a private and curricular reading.
- Typical REVIEW cases: a <PERSON> tag that could be either a classmate/teacher or an author/character; a <LOCATION> tag that could be either a historical place or a student's local place; a <SCHOOL> tag that could be either a real school or a named program/concept.
- Default semantic role: AMBIGUOUS

Subject rules:

English:
- <PERSON> is often KEEP when it stands for an author, character, or speaker inside the assigned text.
- <PERSON> is usually REDACT when it is used in greetings, direct address, self-identification, turn-taking, or school logistics.
- <LOCATION> is often KEEP when it is a fictional or literary setting.
- <SCHOOL> is often REDACT when it names the student's actual school, but may be REVIEW if the context is about admissions, rankings, or essay prompts.
- <COURSE> is usually KEEP when it refers to classwork, readings, or an assignment.

Social Studies:
- <PERSON> is often KEEP when it stands for a historical figure, politician, ruler, or public figure under discussion.
- <LOCATION> is often KEEP when it stands for a country, state, city, battlefield, colony, or other textbook place.
- NRP is often KEEP when it stands for nationality, religion, or political affiliation used as lesson content.
- <SCHOOL> is usually REDACT when it identifies the real student, but may be REVIEW if it appears inside a comparative or institutional discussion.
- <COURSE> is usually KEEP when it is part of the assignment or lesson description.

Fast decision order:
1. If the tag is direct contact or account information, label REDACT.
2. If the tag appears in a direct greeting, self-introduction, or direct address to a real participant, label REDACT.
3. If the tag clearly refers to the real student, volunteer, teacher, school, or local place in this exchange, label REDACT.
4. If the tag clearly belongs to English or Social Studies content, label KEEP.
5. If both readings still fit after reading context_text and anchor_text, label REVIEW.

Edge rules:
- Do not label REDACT just because the underlying entity type sounds personal.
- Do not label KEEP just because the conversation is educational.
- Use REVIEW when the ambiguity is real, not when the example is merely unfamiliar.
- Prefer the highlighted target occurrence over nearby repeated tags.
"""
SAMPLING_PRESETS = {
    "non-thinking": {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "repetition_penalty": 1.0,
    },
    "thinking": {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "repetition_penalty": 1.0,
    },
}


class RowPredictionError(RuntimeError):
    def __init__(self, *, row_id: str, message: str, attempts_log: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.row_id = row_id
        self.attempts_log = attempts_log


def _normalize_enum_value(value: Any, *, allowed: tuple[str, ...]) -> Any:
    if not isinstance(value, str):
        return value
    normalized = re.sub(r"[^A-Z_]", "", value.upper())
    if normalized in allowed:
        return normalized
    prefix_matches = [candidate for candidate in allowed if normalized.startswith(candidate)]
    if len(prefix_matches) == 1 and len(normalized) - len(prefix_matches[0]) <= 1:
        return prefix_matches[0]
    return value


def _default_output_file(model_path: Path, *, thinking_mode: str) -> Path:
    suffix = "nothink" if thinking_mode == "non-thinking" else "think"
    return PREDICTIONS_DIR / f"upchieve_english_social_dev_local_open_{slugify(model_path.name)}_{suffix}.jsonl"


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing comparison summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_target_entry(summary: dict[str, Any], *, target_review_rate: float) -> dict[str, Any]:
    for target in summary.get("selected_targets", []):
        if abs(float(target["target_review_rate"]) - target_review_rate) <= 1e-9:
            return dict(target)
    raise SystemExit(f"Comparison summary is missing a selected target for review rate {target_review_rate:.2f}")


def _build_user_prompt(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for field_spec in INPUT_FIELD_SPECS:
        value = row.get(field_spec["source_field"])
        if value is None or (isinstance(value, str) and not value.strip()):
            if field_spec["required"]:
                raise SystemExit(
                    f"Gold row {row.get('id')} is missing required field {field_spec['source_field']!r}"
                )
            continue
        lines.append(f"{field_spec['prompt_label']}: {str(value)}")
    return "\n".join(lines)


def _build_system_prompt() -> str:
    return (
        "You are labeling one UPChieve anonymization-tag occurrence.\n"
        "Apply the frozen codebook excerpt below.\n"
        "Return exactly one JSON object with keys action_label and semantic_role.\n"
        f"Allowed action_label values: {', '.join(ACTION_LABELS)}.\n"
        f"Allowed semantic_role values: {', '.join(SEMANTIC_ROLES)}.\n"
        "Do not add extra keys, prose, markdown, or explanations.\n\n"
        "Frozen codebook excerpt:\n"
        f"{CODEBOOK_EXCERPT.rstrip()}\n"
    )


def _prompt_template_text() -> str:
    return "\n".join(
        [
            "System prompt:",
            "",
            _build_system_prompt().rstrip(),
            "",
            "Per-row user prompt template:",
            "subject: {subject}",
            "context: {context_text}",
            "span: {span_text}",
            "anchor: {anchor_text}  # only when present",
            "speaker_role: {speaker_role}  # only when present",
            "entity_type: {entity_type}  # only when present",
            "",
            "Expected JSON object:",
            json.dumps(OUTPUT_SCHEMA, indent=2),
            "",
        ]
    )


def _prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _parse_model_output(raw_text: str) -> dict[str, str]:
    normalized_text = raw_text.strip()
    if "</think>" in normalized_text:
        normalized_text = normalized_text.rsplit("</think>", maxsplit=1)[-1].strip()
    if normalized_text.startswith("```"):
        lines = normalized_text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            normalized_text = "\n".join(lines[1:-1]).strip()
    first_brace = normalized_text.find("{")
    last_brace = normalized_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        normalized_text = normalized_text[first_brace : last_brace + 1]
    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError as exc:
        repaired_text = re.sub(
            r'("action_label"\s*:\s*)(REDACT|KEEP|REVIEW)(\s*[,}])',
            r'\1"\2"\3',
            normalized_text,
        )
        repaired_text = re.sub(
            r'("semantic_role"\s*:\s*)(PRIVATE|CURRICULAR|AMBIGUOUS)(\s*[,}])',
            r'\1"\2"\3',
            repaired_text,
        )
        try:
            payload = json.loads(repaired_text)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    if set(payload.keys()) != {"action_label", "semantic_role"}:
        raise ValueError(
            "response must contain exactly action_label and semantic_role keys, "
            f"got {sorted(payload.keys())}"
        )
    action_label = _normalize_enum_value(payload.get("action_label"), allowed=ACTION_LABELS)
    semantic_role = _normalize_enum_value(payload.get("semantic_role"), allowed=SEMANTIC_ROLES)
    if action_label not in ACTION_LABELS:
        raise ValueError(f"action_label must be one of {ACTION_LABELS}, got {action_label!r}")
    if semantic_role not in SEMANTIC_ROLES:
        raise ValueError(f"semantic_role must be one of {SEMANTIC_ROLES}, got {semantic_role!r}")
    return {
        "action_label": str(action_label),
        "semantic_role": str(semantic_role),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _load_existing_predictions(
    path: Path,
    *,
    model_label: str,
    prompt_hash_value: str,
    thinking_mode: str,
    sampling_preset: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    predictions_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row["id"])
        if row_id in predictions_by_id:
            raise SystemExit(f"Duplicate prediction id in existing output file: {row_id}")
        predicted_action = row.get("predicted_action")
        predicted_role = row.get("predicted_semantic_role")
        if predicted_action not in ACTION_LABELS or predicted_role not in SEMANTIC_ROLES:
            raise SystemExit(
                f"Existing output file {path} contains invalid prediction values for id={row_id}; "
                "delete the file or choose a new output path."
            )
        metadata_match = (
            row.get("model_label") == model_label
            and row.get("prompt_hash") == prompt_hash_value
            and row.get("thinking_mode") == thinking_mode
            and row.get("sampling_preset") == sampling_preset
        )
        if not metadata_match:
            raise SystemExit(
                f"Existing output file {path} was produced with different model/prompt/thinking metadata; "
                "choose a new output file instead of mixing runs."
            )
        predictions_by_id[row_id] = dict(row)
    return predictions_by_id


def _validate_prediction_coverage(*, gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    expected_ids = {str(row["id"]) for row in gold_rows}
    actual_ids = {str(row["id"]) for row in prediction_rows}
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing:
        raise SystemExit(f"Prediction file is missing ids: {missing[:10]}")
    if extra:
        raise SystemExit(f"Prediction file contains unexpected ids: {extra[:10]}")


def _merge_gold_predictions(gold_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_by_id = {str(row["id"]): row for row in prediction_rows}
    merged: list[dict[str, Any]] = []
    for row in gold_rows:
        row_id = str(row["id"])
        prediction = predictions_by_id.get(row_id)
        if prediction is None:
            raise SystemExit(f"Missing prediction for id={row_id}")
        merged.append(
            {
                "id": row_id,
                "subject": str(row.get("subject", "unknown")),
                "eval_slice": row.get("eval_slice"),
                "gold_action": row["action_label"],
                "predicted_action": prediction["predicted_action"],
                "speaker_role": row.get("speaker_role"),
                "entity_type": row.get("entity_type"),
                "semantic_role": row.get("semantic_role"),
                "latency_ms": prediction.get("latency_ms"),
            }
        )
    return merged


def _aggregate_run_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    repair_count = sum(1 for row in rows if bool(row.get("repair_used")))
    attempt_count = sum(int(row.get("attempt_count") or 1) for row in rows)
    payload: dict[str, Any] = {
        "prediction_count": len(rows),
        "repair_count": repair_count,
        "attempt_count": attempt_count,
    }
    if latencies:
        payload["latency_ms"] = {
            "avg": mean(latencies),
            "median": median(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }
    else:
        payload["latency_ms"] = {
            "avg": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return payload


def _top_level_table_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Dev Comparison",
        "",
        "| variant | accuracy | macro_f1 | redact_recall | review_rate | gold_review_coverage | curricular_accuracy |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        metrics = row["metrics"]
        curricular_metrics = metrics.get("by_semantic_role", {}).get("CURRICULAR", {})
        lines.append(
            "| {label} | {accuracy} | {macro_f1} | {redact_recall} | {review_rate} | {gold_review_coverage} | {curricular_accuracy} |".format(
                label=row["label"],
                accuracy=_format_metric(metrics.get("accuracy")),
                macro_f1=_format_metric(metrics.get("macro_f1")),
                redact_recall=_format_metric(metrics.get("redact_recall")),
                review_rate=_format_percent(metrics.get("review_rate")),
                gold_review_coverage=_format_percent(metrics.get("gold_review_coverage")),
                curricular_accuracy=_format_metric(curricular_metrics.get("accuracy")),
            )
        )
    lines.append("")
    return lines


def _build_report(
    *,
    gold_file: Path,
    model_path: Path,
    model_label: str,
    model_backend: str,
    output_file: Path,
    prompt_hash_value: str,
    input_field_specs: tuple[dict[str, Any], ...],
    aggregate_stats: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    thinking_mode: str,
    device_label: str,
    torch_dtype_label: str,
    max_new_tokens: int,
    seed: int,
    sampling_preset: dict[str, Any],
) -> str:
    latency = aggregate_stats["latency_ms"]
    lines = [
        "# UPChieve Local Open-Model Ceiling",
        "",
        f"- Gold file: `{gold_file}`",
        f"- Model path: `{model_path}`",
        f"- Model label: `{model_label}`",
        f"- Model backend: `{model_backend}`",
        f"- Output file: `{output_file}`",
        f"- Prompt hash: `{prompt_hash_value}`",
        f"- Thinking mode: `{thinking_mode}`",
        f"- Device: `{device_label}`",
        f"- Torch dtype: `{torch_dtype_label}`",
        f"- Max new tokens: {max_new_tokens}",
        f"- Seed: {seed}",
        f"- Sampling preset: `{json.dumps(sampling_preset, sort_keys=True)}`",
        f"- Prediction rows: {aggregate_stats['prediction_count']}",
        f"- Repair retries used: {aggregate_stats['repair_count']}",
        "",
        "## Frozen Input Fields",
        "",
    ]
    for field_spec in input_field_specs:
        required_label = "required" if field_spec["required"] else "optional when present"
        lines.append(f"- `{field_spec['source_field']}` -> `{field_spec['prompt_label']}` ({required_label})")
    lines.append("")
    lines.extend(_top_level_table_lines(comparison_rows))
    lines.extend(
        [
            "## Run Stats",
            "",
            f"- Latency avg / median / min / max: "
            f"{_format_metric(latency['avg'])} / {_format_metric(latency['median'])} / "
            f"{_format_metric(latency['min'])} / {_format_metric(latency['max'])} ms",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _download_hint(model_path: Path) -> str:
    model_name = model_path.name
    if model_name.startswith("Qwen3.5-"):
        return (
            "Download example:\n"
            f"  HF_HUB_OFFLINE=0 uv run hf download Qwen/{model_name} --local-dir {model_path}"
        )
    return (
        "Model directory is missing. Point --model-path at a local Hugging Face snapshot "
        "directory or download the weights into models/ first."
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "mps":
        return torch.device("mps")
    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_torch_dtype(dtype_name: str, *, device: torch.device) -> tuple[torch.dtype, str]:
    if dtype_name == "float16":
        return torch.float16, "float16"
    if dtype_name == "bfloat16":
        return torch.bfloat16, "bfloat16"
    if dtype_name == "float32":
        return torch.float32, "float32"
    if device.type in {"mps", "cuda"}:
        return torch.float16, "float16"
    return torch.float32, "float32"


def _load_tokenizer_and_model(
    *,
    model_path: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[Any, Any, str]:
    if not model_path.exists():
        raise SystemExit(f"Missing local model path: {model_path}\n{_download_hint(model_path)}")
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        architecture_names = tuple(config.architectures or ())
        if getattr(config, "model_type", None) == "qwen3_5" or "Qwen3_5ForConditionalGeneration" in architecture_names:
            model = Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
                dtype=torch_dtype,
            )
            model_backend = "Qwen3_5ForConditionalGeneration"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                low_cpu_mem_usage=True,
                dtype=torch_dtype,
            )
            model_backend = type(model).__name__
    except (AttributeError, OSError, ValueError) as exc:
        raise SystemExit(f"Unable to load local model from {model_path}: {exc}\n{_download_hint(model_path)}") from exc
    model.to(device)
    model.eval()
    return tokenizer, model, model_backend


def _generate_once(
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    messages: list[dict[str, str]],
    thinking_mode: str,
    max_new_tokens: int,
    sampling_preset: dict[str, Any],
) -> tuple[str, float]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=thinking_mode == "thinking",
    )
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        **sampling_preset,
    }
    started = perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generate_kwargs)
    latency_ms = (perf_counter() - started) * 1000.0
    generated_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text, latency_ms


def _repair_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    invalid_output: str,
    validation_error: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": invalid_output},
        {
            "role": "user",
            "content": (
                "The previous response was invalid.\n"
                f"Validation error: {validation_error}\n"
                "semantic_role is not the entity_type. It must be one of PRIVATE, CURRICULAR, or AMBIGUOUS.\n"
                "Return only a valid JSON object with exactly these keys:\n"
                '{"action_label":"REDACT|KEEP|REVIEW","semantic_role":"PRIVATE|CURRICULAR|AMBIGUOUS"}'
            ),
        },
    ]


def _run_one_row(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    thinking_mode: str,
    max_new_tokens: int,
    sampling_preset: dict[str, Any],
    system_prompt: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    user_prompt = _build_user_prompt(row)
    attempts_log: list[dict[str, Any]] = []
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = "unknown validation error"
    last_raw_text = ""
    for attempt_number in (1, 2):
        if attempt_number == 1:
            messages = base_messages
            repair = False
            attempt_sampling_preset = sampling_preset
        else:
            messages = _repair_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                invalid_output=last_raw_text,
                validation_error=last_error,
            )
            repair = True
            attempt_sampling_preset = {"do_sample": False}
        raw_text, latency_ms = _generate_once(
            tokenizer=tokenizer,
            model=model,
            device=device,
            messages=messages,
            thinking_mode=thinking_mode,
            max_new_tokens=max_new_tokens,
            sampling_preset=attempt_sampling_preset,
        )
        try:
            parsed = _parse_model_output(raw_text)
        except ValueError as exc:
            last_error = str(exc)
            last_raw_text = raw_text
            attempts_log.append(
                {
                    "id": str(row["id"]),
                    "attempt": attempt_number,
                    "repair": repair,
                    "latency_ms": latency_ms,
                    "raw_text": raw_text,
                    "valid": False,
                    "validation_error": str(exc),
                }
            )
            if attempt_number == 2:
                raise RowPredictionError(
                    row_id=str(row["id"]),
                    message=f"Local model output stayed invalid for id={row['id']} after one repair retry: {exc}",
                    attempts_log=attempts_log,
                ) from exc
            continue
        attempts_log.append(
            {
                "id": str(row["id"]),
                "attempt": attempt_number,
                "repair": repair,
                "latency_ms": latency_ms,
                "raw_text": raw_text,
                "valid": True,
            }
        )
        prediction_row = {
            "id": str(row["id"]),
            "predicted_action": parsed["action_label"],
            "predicted_semantic_role": parsed["semantic_role"],
            "attempt_count": attempt_number,
            "repair_used": repair,
            "latency_ms": latency_ms,
        }
        return prediction_row, attempts_log
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a local open-model UPChieve action eval on the frozen dev split.")
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD_FILE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-label")
    parser.add_argument("--thinking-mode", choices=tuple(SAMPLING_PRESETS.keys()), default=DEFAULT_THINKING_MODE)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default=DEFAULT_TORCH_DTYPE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args(argv)

    device = _resolve_device(args.device)
    torch_dtype, torch_dtype_label = _resolve_torch_dtype(args.torch_dtype, device=device)
    model_label = args.model_label or args.model_path.name
    output_file = args.output_file or _default_output_file(args.model_path, thinking_mode=args.thinking_mode)
    sampling_preset = dict(SAMPLING_PRESETS[args.thinking_mode])

    comparison_summary = _load_summary(DEFAULT_COMPARISON_SUMMARY_FILE)
    selected_target = _selected_target_entry(
        comparison_summary,
        target_review_rate=DEFAULT_SELECTED_REVIEW_RATE,
    )
    gold_rows = load_jsonl(args.gold_file)
    system_prompt = _build_system_prompt()
    prompt_text = _prompt_template_text()
    prompt_hash_value = _prompt_hash(prompt_text)

    experiment = create_experiment_run(args.run_name)
    prompt_path = experiment.root / "prompt.txt"
    schema_path = experiment.root / "schema.json"
    input_fields_path = experiment.root / "input_fields.json"
    raw_responses_path = experiment.predictions_dir / "raw_responses.jsonl"

    prompt_path.write_text(prompt_text, encoding="utf-8")
    schema_path.write_text(json.dumps(OUTPUT_SCHEMA, indent=2), encoding="utf-8")
    input_fields_path.write_text(json.dumps(list(INPUT_FIELD_SPECS), indent=2), encoding="utf-8")

    existing_predictions = _load_existing_predictions(
        output_file,
        model_label=model_label,
        prompt_hash_value=prompt_hash_value,
        thinking_mode=args.thinking_mode,
        sampling_preset=sampling_preset,
    )

    set_seed(args.seed)
    tokenizer, model, model_backend = _load_tokenizer_and_model(
        model_path=args.model_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    completed_predictions: dict[str, dict[str, Any]] = dict(existing_predictions)
    executed_generation_count = 0
    repair_request_count = 0
    started_at_epoch = time()

    for index, row in enumerate(gold_rows, start=1):
        row_id = str(row["id"])
        if row_id in completed_predictions:
            continue
        try:
            prediction_row, attempts_log = _run_one_row(
                row=row,
                tokenizer=tokenizer,
                model=model,
                device=device,
                thinking_mode=args.thinking_mode,
                max_new_tokens=args.max_new_tokens,
                sampling_preset=sampling_preset,
                system_prompt=system_prompt,
            )
        except RowPredictionError as exc:
            for attempt in exc.attempts_log:
                _append_jsonl(raw_responses_path, attempt)
            raise SystemExit(str(exc)) from exc
        prediction_row.update(
            {
                "model_label": model_label,
                "model_path": str(args.model_path),
                "model_backend": model_backend,
                "thinking_mode": args.thinking_mode,
                "sampling_preset": sampling_preset,
                "device": str(device),
                "torch_dtype": torch_dtype_label,
                "prompt_hash": prompt_hash_value,
            }
        )
        completed_predictions[row_id] = prediction_row
        _append_jsonl(output_file, prediction_row)
        for attempt in attempts_log:
            _append_jsonl(raw_responses_path, attempt)
        executed_generation_count += len(attempts_log)
        repair_request_count += sum(1 for attempt in attempts_log if bool(attempt["repair"]))
        if index % 10 == 0 or index == len(gold_rows):
            print(f"[local-open-ceiling] completed {index}/{len(gold_rows)} rows", flush=True)

    ordered_predictions = [completed_predictions[str(row["id"])] for row in gold_rows]
    _validate_prediction_coverage(gold_rows=gold_rows, prediction_rows=ordered_predictions)

    prediction_copy_path = experiment.predictions_dir / output_file.name
    prediction_copy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, prediction_copy_path)

    aggregate_stats = _aggregate_run_stats(ordered_predictions)
    merged_rows = _merge_gold_predictions(gold_rows, ordered_predictions)
    llm_metrics = compute_action_metrics(merged_rows)
    comparison_rows = [
        {"label": "ModernBERT v2 base (dev)", "metrics": comparison_summary["calibration_base_metrics"]},
        {"label": "ModernBERT v2 selected 10% (dev)", "metrics": selected_target["calibration"]["metrics"]},
        {"label": f"{model_label} local", "metrics": llm_metrics},
    ]

    completed_at_epoch = time()
    metadata = {
        "run_name": args.run_name,
        "gold_file": str(args.gold_file),
        "model_path": str(args.model_path),
        "model_label": model_label,
        "model_backend": model_backend,
        "thinking_mode": args.thinking_mode,
        "device": str(device),
        "torch_dtype": torch_dtype_label,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling_preset": sampling_preset,
        "output_file": str(output_file),
        "prediction_copy_file": str(prediction_copy_path),
        "raw_responses_file": str(raw_responses_path),
        "comparison_summary_file": str(DEFAULT_COMPARISON_SUMMARY_FILE),
        "comparison_selected_review_rate": DEFAULT_SELECTED_REVIEW_RATE,
        "prompt_hash": prompt_hash_value,
        "prompt_file": str(prompt_path),
        "schema_file": str(schema_path),
        "input_fields_file": str(input_fields_path),
        "codebook_source": CODEBOOK_SOURCE,
        "started_at_epoch": started_at_epoch,
        "completed_at_epoch": completed_at_epoch,
        "existing_prediction_count": len(existing_predictions),
        "executed_generation_count": executed_generation_count,
        "repair_request_count": repair_request_count,
    }
    write_run_metadata(experiment.metadata_path, metadata)

    summary_payload = {
        "gold_file": str(args.gold_file),
        "model_path": str(args.model_path),
        "model_label": model_label,
        "model_backend": model_backend,
        "thinking_mode": args.thinking_mode,
        "device": str(device),
        "torch_dtype": torch_dtype_label,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling_preset": sampling_preset,
        "output_file": str(output_file),
        "prediction_copy_file": str(prediction_copy_path),
        "raw_responses_file": str(raw_responses_path),
        "benchmark_row_count": len(gold_rows),
        "prediction_count": len(ordered_predictions),
        "prompt_hash": prompt_hash_value,
        "codebook_source": CODEBOOK_SOURCE,
        "input_fields": list(INPUT_FIELD_SPECS),
        "usage_and_latency": aggregate_stats,
        "executed_generation_count": executed_generation_count,
        "repair_request_count": repair_request_count,
        "existing_prediction_count": len(existing_predictions),
        "comparison_baselines": {
            "base_dev_metrics": comparison_summary["calibration_base_metrics"],
            "selected_dev_metrics": selected_target["calibration"]["metrics"],
        },
        "llm_metrics": llm_metrics,
    }
    experiment.summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    experiment.report_path.write_text(
        _build_report(
            gold_file=args.gold_file,
            model_path=args.model_path,
            model_label=model_label,
            model_backend=model_backend,
            output_file=output_file,
            prompt_hash_value=prompt_hash_value,
            input_field_specs=INPUT_FIELD_SPECS,
            aggregate_stats=aggregate_stats,
            comparison_rows=comparison_rows,
            thinking_mode=args.thinking_mode,
            device_label=str(device),
            torch_dtype_label=torch_dtype_label,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            sampling_preset=sampling_preset,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "experiment_root": str(experiment.root),
                "summary_path": str(experiment.summary_path),
                "report_path": str(experiment.report_path),
                "prediction_file": str(output_file),
                "macro_f1": llm_metrics["macro_f1"],
                "executed_generation_count": executed_generation_count,
                "repair_request_count": repair_request_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
