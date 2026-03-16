from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
from statistics import mean, median
import sys
from time import perf_counter, time
from typing import Any
from urllib import error, parse, request

import certifi

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ACTION_DIR, PREDICTIONS_DIR
from contextshift_deid.data import load_jsonl
from contextshift_deid.experiment_runs import create_experiment_run, slugify, write_run_metadata
from contextshift_deid.metrics import compute_action_metrics

DEFAULT_GOLD_FILE = ACTION_DIR / "upchieve_english_social_dev.jsonl"
DEFAULT_BASE_URL = "https://api.ai.it.cornell.edu"
DEFAULT_ENDPOINT_PATH = "/v1/chat/completions"
DEFAULT_MODEL = "openai.gpt-5.2"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_RUN_NAME = "upchieve-llm-ceiling"
DEFAULT_SELECTED_REVIEW_RATE = 0.10
DEFAULT_COMPARISON_SUMMARY_FILE = (
    ROOT / "artifacts/experiments/20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384/summary.json"
)
GATEWAY_KEY_ENV = "CORNELL_AI_GATEWAY_KEY"
ACTION_LABELS = ("REDACT", "KEEP", "REVIEW")
SEMANTIC_ROLES = ("PRIVATE", "CURRICULAR", "AMBIGUOUS")
REASONING_EFFORT_VALUES = ("none", "minimal", "low", "medium", "high", "xhigh")
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


def _default_output_file(model: str) -> Path:
    return PREDICTIONS_DIR / f"upchieve_english_social_dev_llm_ceiling_{slugify(model)}.jsonl"


def _resolved_reasoning_effort(model: str, raw_value: str | None) -> str | None:
    if raw_value is not None:
        return raw_value
    if model.startswith("openai.gpt-5.2"):
        return DEFAULT_REASONING_EFFORT
    return None


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


def _validate_base_url(base_url: str) -> str:
    parsed = parse.urlparse(base_url)
    normalized_path = parsed.path.rstrip("/")
    if parsed.scheme != "https" or parsed.netloc != "api.ai.it.cornell.edu":
        raise SystemExit(
            "This runner only supports the Cornell AI Gateway. "
            f"Expected https://api.ai.it.cornell.edu, got {base_url!r}."
        )
    if normalized_path:
        raise SystemExit(
            "Pass only the Cornell gateway base URL without an endpoint path, "
            f"got {base_url!r}."
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def _build_user_prompt(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for field_spec in INPUT_FIELD_SPECS:
        source_field = field_spec["source_field"]
        prompt_label = field_spec["prompt_label"]
        required = bool(field_spec["required"])
        value = row.get(source_field)
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                raise SystemExit(f"Gold row {row.get('id')} is missing required field {source_field!r}")
            continue
        lines.append(f"{prompt_label}: {str(value)}")
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


def _normalize_usage(raw_usage: Any) -> dict[str, int]:
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    normalized: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None:
            normalized[key] = 0
            continue
        normalized[key] = int(value)
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        normalized["reasoning_tokens"] = int(completion_details.get("reasoning_tokens") or 0)
    else:
        normalized["reasoning_tokens"] = 0
    return normalized


def _response_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        text_chunks: list[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value is not None:
                    text_chunks.append(str(text_value))
        return "".join(text_chunks)
    raise ValueError(f"Unsupported message content type: {type(message_content)!r}")


def _parse_model_output(raw_text: str) -> dict[str, str]:
    normalized_text = raw_text.strip()
    if normalized_text.startswith("```"):
        lines = normalized_text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            normalized_text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    if set(payload.keys()) != {"action_label", "semantic_role"}:
        raise ValueError(
            "response must contain exactly action_label and semantic_role keys, "
            f"got {sorted(payload.keys())}"
        )
    action_label = payload.get("action_label")
    semantic_role = payload.get("semantic_role")
    if action_label not in ACTION_LABELS:
        raise ValueError(f"action_label must be one of {ACTION_LABELS}, got {action_label!r}")
    if semantic_role not in SEMANTIC_ROLES:
        raise ValueError(f"semantic_role must be one of {SEMANTIC_ROLES}, got {semantic_role!r}")
    return {
        "action_label": str(action_label),
        "semantic_role": str(semantic_role),
    }


def _request_chat_completion(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    use_json_object_mode: bool,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
    }
    if reasoning_effort is None:
        payload["temperature"] = 0
    else:
        payload["reasoning_effort"] = reasoning_effort
    if use_json_object_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    started = perf_counter()
    try:
        with request.urlopen(req, context=ssl.create_default_context(cafile=certifi.where())) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Gateway request failed with HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:
        raise SystemExit(f"Gateway request failed: {exc}") from exc
    latency_ms = (perf_counter() - started) * 1000.0
    response_payload = json.loads(raw_body)
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit(f"Gateway response is missing choices: {response_payload}")
    message = choices[0].get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise SystemExit(f"Gateway response is missing message content: {response_payload}")
    text = _response_text(message["content"])
    return {
        "response_id": response_payload.get("id"),
        "latency_ms": latency_ms,
        "raw_response": response_payload,
        "raw_text": text,
        "usage": _normalize_usage(response_payload.get("usage")),
    }


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
                "Return only a valid JSON object with exactly these keys:\n"
                '{"action_label":"REDACT|KEEP|REVIEW","semantic_role":"PRIVATE|CURRICULAR|AMBIGUOUS"}'
            ),
        },
    ]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _load_existing_predictions(
    path: Path,
    *,
    model: str,
    base_url: str,
    prompt_hash_value: str,
    reasoning_effort: str | None,
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
        if (
            row.get("model") != model
            or row.get("base_url") != base_url
            or row.get("prompt_hash") != prompt_hash_value
            or row.get("reasoning_effort") != reasoning_effort
        ):
            raise SystemExit(
                f"Existing output file {path} was produced with different model/base_url/prompt_hash/reasoning metadata; "
                "choose a new output file instead of mixing runs."
            )
        predictions_by_id[row_id] = dict(row)
    return predictions_by_id


def _validate_prediction_coverage(
    *,
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
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
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in rows)
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in rows)
    reasoning_tokens = sum(int(row.get("reasoning_tokens") or 0) for row in rows)
    repair_count = sum(1 for row in rows if bool(row.get("repair_used")))
    attempt_count = sum(int(row.get("attempt_count") or 1) for row in rows)
    response_ids = [str(row["response_id"]) for row in rows if row.get("response_id")]
    payload: dict[str, Any] = {
        "prediction_count": len(rows),
        "repair_count": repair_count,
        "attempt_count": attempt_count,
        "usage_totals": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        },
        "response_ids": response_ids,
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
    output_file: Path,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    comparison_summary_file: Path,
    prompt_hash_value: str,
    input_field_specs: tuple[dict[str, Any], ...],
    aggregate_stats: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
) -> str:
    latency = aggregate_stats["latency_ms"]
    lines = [
        "# UPChieve LLM Ceiling",
        "",
        f"- Gold file: `{gold_file}`",
        f"- Output file: `{output_file}`",
        f"- Comparison summary: `{comparison_summary_file}`",
        f"- Base URL: `{base_url}`",
        f"- Model: `{model}`",
        f"- Reasoning effort: `{reasoning_effort or 'n/a'}`",
        f"- Endpoint: `{DEFAULT_ENDPOINT_PATH}`",
        f"- Prompt hash: `{prompt_hash_value}`",
        f"- Prediction rows: {aggregate_stats['prediction_count']}",
        f"- Repair retries used: {aggregate_stats['repair_count']}",
        "",
        "## Frozen Input Fields",
        "",
    ]
    for field_spec in input_field_specs:
        required_label = "required" if field_spec["required"] else "optional when present"
        lines.append(
            f"- `{field_spec['source_field']}` -> `{field_spec['prompt_label']}` ({required_label})"
        )
    lines.append("")
    lines.extend(_top_level_table_lines(comparison_rows))
    lines.extend(
        [
            "## Run Stats",
            "",
            f"- Prompt tokens: {aggregate_stats['usage_totals']['prompt_tokens']}",
            f"- Completion tokens: {aggregate_stats['usage_totals']['completion_tokens']}",
            f"- Total tokens: {aggregate_stats['usage_totals']['total_tokens']}",
            f"- Reasoning tokens: {aggregate_stats['usage_totals']['reasoning_tokens']}",
            f"- Latency avg / median / min / max: "
            f"{_format_metric(latency['avg'])} / {_format_metric(latency['median'])} / "
            f"{_format_metric(latency['min'])} / {_format_metric(latency['max'])} ms",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_one_row(
    *,
    row: dict[str, Any],
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    reasoning_effort: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    user_prompt = _build_user_prompt(row)
    attempts_log: list[dict[str, Any]] = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = "unknown validation error"
    last_raw_text = ""
    for attempt_number in (1, 2):
        if attempt_number == 1:
            current_messages = messages
            repair = False
            use_json_object_mode = True
        else:
            current_messages = _repair_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                invalid_output=last_raw_text,
                validation_error=last_error,
            )
            repair = True
            use_json_object_mode = not model.startswith("anthropic.")
        response_payload = _request_chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=current_messages,
            use_json_object_mode=use_json_object_mode,
            reasoning_effort=reasoning_effort,
        )
        raw_text = response_payload["raw_text"]
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
                    "response_id": response_payload.get("response_id"),
                    "latency_ms": response_payload["latency_ms"],
                    "usage": response_payload["usage"],
                    "raw_text": raw_text,
                    "valid": False,
                    "validation_error": str(exc),
                }
            )
            if attempt_number == 2:
                raise SystemExit(
                    f"LLM output stayed invalid for id={row['id']} after one repair retry: {exc}"
                ) from exc
            continue
        attempts_log.append(
            {
                "id": str(row["id"]),
                "attempt": attempt_number,
                "repair": repair,
                "response_id": response_payload.get("response_id"),
                "latency_ms": response_payload["latency_ms"],
                "usage": response_payload["usage"],
                "raw_text": raw_text,
                "valid": True,
            }
        )
        prediction_row = {
            "id": str(row["id"]),
            "predicted_action": parsed["action_label"],
            "predicted_semantic_role": parsed["semantic_role"],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "base_url": DEFAULT_BASE_URL if url.startswith(DEFAULT_BASE_URL) else url.removesuffix(DEFAULT_ENDPOINT_PATH),
            "endpoint_path": DEFAULT_ENDPOINT_PATH,
            "prompt_hash": "",
            "response_id": response_payload.get("response_id"),
            "attempt_count": attempt_number,
            "repair_used": repair,
            "latency_ms": response_payload["latency_ms"],
            "prompt_tokens": response_payload["usage"]["prompt_tokens"],
            "completion_tokens": response_payload["usage"]["completion_tokens"],
            "total_tokens": response_payload["usage"]["total_tokens"],
            "reasoning_tokens": response_payload["usage"]["reasoning_tokens"],
        }
        return prediction_row, attempts_log
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a dev-only LLM ceiling baseline for UPChieve action classification.")
    parser.add_argument("--gold-file", type=Path, default=DEFAULT_GOLD_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORT_VALUES)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args(argv)

    api_key = os.environ.get(GATEWAY_KEY_ENV)
    if not api_key:
        raise SystemExit(
            f"Missing {GATEWAY_KEY_ENV}. This runner only supports the Cornell AI Gateway."
        )

    base_url = _validate_base_url(args.base_url)
    reasoning_effort = _resolved_reasoning_effort(args.model, args.reasoning_effort)
    output_file = args.output_file or _default_output_file(args.model)
    comparison_summary = _load_summary(DEFAULT_COMPARISON_SUMMARY_FILE)
    selected_target = _selected_target_entry(
        comparison_summary,
        target_review_rate=DEFAULT_SELECTED_REVIEW_RATE,
    )

    gold_rows = load_jsonl(args.gold_file)
    system_prompt = _build_system_prompt()
    prompt_text = _prompt_template_text()
    prompt_hash_value = _prompt_hash(prompt_text)
    request_url = f"{base_url}{DEFAULT_ENDPOINT_PATH}"

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
        model=args.model,
        base_url=base_url,
        prompt_hash_value=prompt_hash_value,
        reasoning_effort=reasoning_effort,
    )
    completed_predictions: dict[str, dict[str, Any]] = dict(existing_predictions)
    executed_request_count = 0
    repair_request_count = 0
    started_at_epoch = time()

    for index, row in enumerate(gold_rows, start=1):
        row_id = str(row["id"])
        if row_id in completed_predictions:
            continue
        prediction_row, attempts_log = _run_one_row(
            row=row,
            url=request_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
        )
        prediction_row["prompt_hash"] = prompt_hash_value
        completed_predictions[row_id] = prediction_row
        _append_jsonl(output_file, prediction_row)
        for attempt in attempts_log:
            _append_jsonl(raw_responses_path, attempt)
        executed_request_count += len(attempts_log)
        repair_request_count += sum(1 for attempt in attempts_log if bool(attempt["repair"]))
        if index % 10 == 0 or index == len(gold_rows):
            print(f"[llm-ceiling] completed {index}/{len(gold_rows)} rows", flush=True)

    ordered_predictions = [completed_predictions[str(row["id"])] for row in gold_rows]
    _validate_prediction_coverage(gold_rows=gold_rows, prediction_rows=ordered_predictions)

    prediction_copy_path = experiment.predictions_dir / output_file.name
    prediction_copy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_file, prediction_copy_path)

    aggregate_stats = _aggregate_run_stats(ordered_predictions)
    merged_rows = _merge_gold_predictions(gold_rows, ordered_predictions)
    llm_metrics = compute_action_metrics(merged_rows)

    comparison_rows = [
        {
            "label": "ModernBERT v2 base (dev)",
            "metrics": comparison_summary["calibration_base_metrics"],
        },
        {
            "label": "ModernBERT v2 selected 10% (dev)",
            "metrics": selected_target["calibration"]["metrics"],
        },
        {
            "label": f"{args.model} ceiling",
            "metrics": llm_metrics,
        },
    ]

    completed_at_epoch = time()
    metadata = {
        "run_name": args.run_name,
        "gold_file": str(args.gold_file),
        "output_file": str(output_file),
        "prediction_copy_file": str(prediction_copy_path),
        "raw_responses_file": str(raw_responses_path),
        "comparison_summary_file": str(DEFAULT_COMPARISON_SUMMARY_FILE),
        "comparison_selected_review_rate": DEFAULT_SELECTED_REVIEW_RATE,
        "base_url": base_url,
        "endpoint_path": DEFAULT_ENDPOINT_PATH,
        "model": args.model,
        "reasoning_effort": reasoning_effort,
        "prompt_hash": prompt_hash_value,
        "prompt_file": str(prompt_path),
        "schema_file": str(schema_path),
        "input_fields_file": str(input_fields_path),
        "codebook_source": CODEBOOK_SOURCE,
        "started_at_epoch": started_at_epoch,
        "completed_at_epoch": completed_at_epoch,
        "existing_prediction_count": len(existing_predictions),
        "executed_request_count": executed_request_count,
        "repair_request_count": repair_request_count,
    }
    write_run_metadata(experiment.metadata_path, metadata)

    summary_payload = {
        "gold_file": str(args.gold_file),
        "output_file": str(output_file),
        "prediction_copy_file": str(prediction_copy_path),
        "raw_responses_file": str(raw_responses_path),
        "benchmark_row_count": len(gold_rows),
        "prediction_count": len(ordered_predictions),
        "base_url": base_url,
        "endpoint_path": DEFAULT_ENDPOINT_PATH,
        "model": args.model,
        "reasoning_effort": reasoning_effort,
        "prompt_hash": prompt_hash_value,
        "codebook_source": CODEBOOK_SOURCE,
        "input_fields": list(INPUT_FIELD_SPECS),
        "usage_and_latency": aggregate_stats,
        "executed_request_count": executed_request_count,
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
            output_file=output_file,
            base_url=base_url,
            model=args.model,
            reasoning_effort=reasoning_effort,
            comparison_summary_file=DEFAULT_COMPARISON_SUMMARY_FILE,
            prompt_hash_value=prompt_hash_value,
            input_field_specs=INPUT_FIELD_SPECS,
            aggregate_stats=aggregate_stats,
            comparison_rows=comparison_rows,
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
                "executed_request_count": executed_request_count,
                "repair_request_count": repair_request_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
