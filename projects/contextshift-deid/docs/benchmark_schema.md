# Benchmark Schema

This repository uses two benchmark views:

- `candidate` for suspicious-span detection
- `action` for `REDACT / KEEP / REVIEW` classification

## Candidate Splits

Files:

```text
data/processed/candidate/train.jsonl
data/processed/candidate/dev.jsonl
data/processed/candidate/test.jsonl
```

Each line is one JSON object:

```json
{
  "id": "lit-001",
  "subject": "literature",
  "tokens": ["My", "teacher", "Mr.", "Shah", "said", "Macbeth", "is", "ambitious", "."],
  "labels": ["O", "O", "B-SUSPECT", "I-SUSPECT", "O", "B-SUSPECT", "O", "O", "O"]
}
```

Optional fields:

- `anchor_text`
- `dialogue_id`
- `speaker_role`
- `context_text`
- `metadata`

Notes:

- `tokens` and `labels` must have the same length.
- Recommended labels are `O`, `B-SUSPECT`, and `I-SUSPECT`.
- This stage is recall-first.

Prediction schema:

```json
{
  "id": "lit-001",
  "predicted_labels": ["O", "O", "B-SUSPECT", "I-SUSPECT", "O", "B-SUSPECT", "O", "O", "O"]
}
```

## Action Splits

Files:

```text
data/processed/action/train.jsonl
data/processed/action/dev.jsonl
data/processed/action/test.jsonl
```

Each line is one JSON object:

```json
{
  "id": "lit-001-span-1",
  "subject": "literature",
  "span_text": "Macbeth",
  "context_text": "My teacher Mr. Shah said Macbeth is ambitious.",
  "action_label": "KEEP",
  "eval_slice": "natural"
}
```

Recommended optional fields:

- `eval_slice`
- `anchor_text`
- `speaker_role`
- `entity_type`
- `semantic_role`
- `intent_label`
- `dialogue_id`
- `cost`
- `latency_ms`

Prediction schema:

```json
{
  "id": "lit-001-span-1",
  "predicted_action": "KEEP",
  "confidence": 0.93,
  "probabilities": {
    "REDACT": 0.04,
    "KEEP": 0.93,
    "REVIEW": 0.03
  }
}
```

Optional prediction fields:

- `confidence`
- `probabilities`
- `cost`
- `latency_ms`

## Pilot Metadata

The anonymized UPChieve action pilot preserves exact tag-target provenance in `metadata`:

- `source`
- `surrogate_mode`
- `surrogate_source`
- `surrogate_seed` when surrogate replacement is enabled
- `tag_start`
- `tag_end`
- `tag_occurrence`

Those offsets refer to the current turn text stored in `metadata.turn_text`. The exported `span_text` may be an anonymization tag such as `<PERSON>` rather than the original surface form.
Those offsets refer to the current turn text stored in `metadata.turn_text`. Depending on `metadata.surrogate_mode`, the exported `span_text` may be either the original anonymization tag or a derived surrogate surface form.

## Default Action Labels

- `REDACT`
- `KEEP`
- `REVIEW`
