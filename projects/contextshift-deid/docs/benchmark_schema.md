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

Math ground-truth benchmark files:

```text
data/processed/candidate/upchieve_math_ground_truth_train.jsonl
data/processed/candidate/upchieve_math_ground_truth_dev.jsonl
data/processed/candidate/upchieve_math_ground_truth_test.jsonl
data/processed/candidate/saga27_math_ground_truth_test.jsonl
data/processed/candidate/ground_truth_candidate_benchmark_summary.json
data/processed/candidate/ground_truth_candidate_benchmark_comparison.json
data/processed/candidate/ground_truth_candidate_benchmark_comparison.md
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
- The math ground-truth benchmark keeps the same top-level schema and stores raw annotation provenance inside `metadata`.

### Math Ground-Truth Candidate Metadata

The math ground-truth benchmark is built from raw per-dialogue JSON files under:

- `/Users/chason/agi/ground-truth/ground-truth-math-upchieve-1000`
- `/Users/chason/agi/ground-truth/ground-truth-math-saga-27`

Both sources use the same raw format:

- one JSON file per dialogue
- `transcript` as the source text
- char-offset annotations from `pii_occurrences` or `spans`

UpChieve rows are projected to speaker turns using `student:` and `volunteer:` prefixes from the transcript string.
Saga rows are projected to text segments from the same transcript string.

Observed canonical math labels:

- `NAME`
- `ADDRESS`
- `URL`
- `SCHOOL`
- `AGE`
- `IDENTIFYING_NUMBER`
- `TUTOR_PROVIDER`
- `DATE`
- `PHONE_NUMBER`
- `IP_ADDRESS`

Ground-truth candidate rows may include these `metadata` fields:

- `source`
- `source_file`
- `ground_truth_source`
- `raw_filename`
- `dialogue_id`
- `turn_index` or `segment_index`
- `turn_start` and `turn_end`, or `segment_start` and `segment_end`
- `raw_text`
- `current_raw_text`
- `prev_turn_text`
- `next_turn_text`
- `prev_speaker_role`
- `next_speaker_role`
- `pii_types`
- `raw_pii_types`
- `has_positive_label`
- `benchmark_split`
- `gold_spans`

Each `metadata.gold_spans[]` entry contains token offsets plus raw annotation provenance:

- `char_start`
- `char_end`
- `token_start`
- `token_end`
- `text`
- `label`
- `raw_pii_type`
- `canonical_pii_type`
- `offset_resolution`
- `source_annotation_start`
- `source_annotation_end`
- `source_annotation_text`
- `transcript_char_start`
- `transcript_char_end`

Current offset resolution modes are attempted in this order:

1. exact `[start, end)`
2. inclusive end
3. trimmed whitespace
4. one-character right shift
5. bounded local exact-text search

The build fails if no resolution strategy succeeds.

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
Depending on `metadata.surrogate_mode`, the exported `span_text` may be either the original anonymization tag or a derived surrogate surface form.

## Default Action Labels

- `REDACT`
- `KEEP`
- `REVIEW`
