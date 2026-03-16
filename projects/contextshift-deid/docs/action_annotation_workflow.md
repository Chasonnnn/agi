# Action Annotation Workflow

This workflow turns candidate-stage evidence into adjudicated action labels without introducing a web service or network dependency.

## Files

The new action-labeling flow uses these files:

- pool file: `artifacts/annotation/action_pool_<split>.jsonl`
- per-annotator decisions: `artifacts/annotation/action_pool_<split>.annotated.jsonl`
- disagreement subset: `artifacts/annotation/action_pool_<split>.disagreements.jsonl`
- final exported action split: `data/processed/action/<split>.jsonl`

For the anonymized UPChieve pilot, also use:

- normalized turns: `data/interim/upchieve_context_pilot_turns.jsonl`
- pilot pools: `artifacts/annotation/upchieve_context_pilot/action_pool_upchieve_english_social_<split>.jsonl`
- pilot codebook: [codebook_v3.md](./codebook_v3.md)

## Recommended Order

1. Freeze the candidate checkpoint you want to trust for the sprint.
2. Export candidate predictions for the target split.
3. Build an annotation pool that mixes legacy positives with nontrivial probes.
4. Run one or two manual annotation passes.
5. Compare annotators and adjudicate disagreements.
6. Export final repo-native action JSONL.
7. Run `uv run prepare.py --strict --allow-provisional-action-splits train` while train is still provisional.
8. Repeat the same pool, annotation, and export loop for `train` so the action model is no longer learning from all-`REDACT` supervision.

If you want the repo to bootstrap that package for you, start here:

```bash
uv run scripts/freeze_action_work_package.py --headline privacy_recall
```

That creates a timestamped package under `artifacts/annotation/packages/` with a locked holdout artifact, frozen train/dev/test candidate predictions, frozen annotation pools, and an `instructions.md` file for the relabeling sprint.

## 1. Export Candidate Predictions

Example:

```bash
uv run scripts/predict_candidate.py \
  --model runs/candidate_math_distilbert_v3 \
  --input-file data/processed/candidate/dev.jsonl \
  --output-file artifacts/predictions/candidate_dev_predictions.jsonl
```

## 2. Build The Action Pool

Example with candidate predictions:

```bash
uv run scripts/build_action_annotation_pool.py \
  --candidate-file data/processed/candidate/dev.jsonl \
  --provisional-action-file data/processed/action/dev.jsonl \
  --prediction-file artifacts/predictions/candidate_dev_predictions.jsonl \
  --output-file artifacts/annotation/action_pool_dev.jsonl
```

Example without predictions:

```bash
uv run scripts/build_action_annotation_pool.py \
  --candidate-file data/processed/candidate/dev.jsonl \
  --provisional-action-file data/processed/action/dev.jsonl
```

The builder seeds:

- `legacy_positive` rows from the existing provisional action split
- `candidate_prediction` rows from predicted-only candidate spans
- `titlecase_probe` rows as extra ambiguous/name-like probes

The output keeps provenance, previews, and suggested actions, but it does not assign final gold labels.

### UPChieve English/Social Studies Pilot

Build the anonymized action-first pilot pools:

```bash
uv run scripts/import_upchieve_context_pilot.py
uv run scripts/build_upchieve_action_pilot.py
```

The pool builder now drops pathological turns by default before sampling:

- `--max-turn-words 350`
- `--max-qualifying-tags 20`

This keeps extremely long, tag-saturated surrogate rows out of the annotation pool while preserving the fixed per-subject and per-slice quotas. Override those flags only if you are intentionally rebuilding a harder stress pool.

For the `english` pilot slice, the builder also excludes `readingWriting` sessions that explicitly mention `spanish`, `espanol`, or `español` so bilingual or Spanish-writing-help sessions do not leak into the English-only annotation workload.

This produces:

- `artifacts/annotation/upchieve_context_pilot/action_pool_upchieve_english_social_dev.jsonl`
- `artifacts/annotation/upchieve_context_pilot/action_pool_upchieve_english_social_test.jsonl`

Each pool contains exactly 200 rows:

- 100 `english`
- 100 `social_studies`
- within each subject, 50 `natural` and 50 `challenge`

Treat the exported `upchieve_english_social_dev/test` files as the frozen `benchmark-v1` once you finish that pilot. The next package starts from unused sessions:

```bash
uv run scripts/analyze_upchieve_zero_shot.py
uv run scripts/build_upchieve_action_train.py
```

That train builder writes:

- `artifacts/annotation/upchieve_context_train/action_pool_upchieve_english_social_train.jsonl`
- `artifacts/annotation/upchieve_context_train/summary.json`

The fixed train-pool quotas are:

- 300 `english`
- 300 `social_studies`
- within each subject, 150 `natural` and 150 `challenge`

The train builder reuses the same path filters as the pilot:

- max turn words `350`
- max qualifying tags `20`
- exclude English-side sessions with explicit `spanish`, `espanol`, or `español`
- exclude any `dialogue_id` already used in `upchieve_english_social_dev/test`

For the targeted additive scaling round, build the v2 pool from sessions not used in `benchmark-v1` or train v1:

```bash
uv run scripts/build_upchieve_action_train_v2.py
```

That builder writes:

- `artifacts/annotation/upchieve_context_train_v2/action_pool_upchieve_english_social_train_v2.jsonl`
- `artifacts/annotation/upchieve_context_train_v2/summary.json`

The fixed v2 quotas are:

- 200 `english`
- 200 `social_studies`
- within each subject, 100 `natural` and 100 `challenge`

The v2 builder keeps the same turn filters and adds stronger failure-bucket targeting:

- excludes any `dialogue_id` already present in `upchieve_english_social_dev/test/train`
- excludes English-topic sessions with explicit `spanish`, `espanol`, `español`, `arabic`, `persian`, `farsi`, or `urdu`
- excludes English-topic rows whose target turn contains Arabic-script text
- reserves English natural `PERSON` rows with greeting, direct-address, and self-identification cues
- enforces challenge minimums for English `PERSON` and `LOCATION`, plus Social Studies `LOCATION`, `NRP`, and `PERSON`

## 3. Run Manual Annotation

One annotator:

```bash
uv run scripts/annotate_action_pool.py \
  --input-file artifacts/annotation/action_pool_dev.jsonl \
  --annotator chason
```

Terminal controls:

- `r` -> `REDACT`
- `k` -> `KEEP`
- `v` -> `REVIEW`
- `s` -> skip
- `b` -> back
- `u` -> clear the current label
- `q` -> save and quit

The tool writes after every decision, so it is safe to stop and resume.

For the UPChieve pilot, the annotator also records `semantic_role` on every row and highlights the exact anonymization-tag occurrence inside the target turn.

## 4. Double Annotation And Disagreements

For a calibration round, have two annotators label the same pool into separate files.

Then compare them:

```bash
uv run scripts/compare_action_annotations.py \
  --left-file artifacts/annotation/action_pool_dev.annotated.jsonl \
  --right-file artifacts/annotation/action_pool_dev_second.annotated.jsonl \
  --output-file artifacts/annotation/action_pool_dev.disagreements.jsonl
```

The disagreement file is another pool file. Re-run `annotate_action_pool.py` on it during adjudication.

## 5. Export Final Action JSONL

Once every row in the adjudicated file has a label:

```bash
uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/action_pool_dev.annotated.jsonl \
  --output-file data/processed/action/dev.jsonl
```

The exporter writes repo-native action rows and stamps metadata with manual-label provenance.

For the UPChieve pilot, export with the new codebook provenance:

```bash
uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/upchieve_context_pilot/action_pool_upchieve_english_social_dev.annotated.jsonl \
  --output-file data/processed/action/upchieve_english_social_dev.jsonl \
  --label-source codebook_v3_manual
```

## 6. Validate

Run:

```bash
uv run prepare.py --strict --allow-provisional-action-splits train
```

To validate nonstandard pilot action files alongside the default splits:

```bash
uv run prepare.py \
  --strict \
  --allow-provisional-action-splits train \
  --extra-action-file data/processed/action/upchieve_english_social_dev.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_test.jsonl
```

Full `uv run prepare.py --strict` should still fail until the train split has been relabeled. That is the intended guardrail.

After you relabel `train`, export it the same way:

```bash
uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/upchieve_context_train/action_pool_upchieve_english_social_train.annotated.jsonl \
  --output-file data/processed/action/upchieve_english_social_train.jsonl \
  --label-source codebook_v3_manual
```

Then compose the mixed train file:

```bash
uv run scripts/compose_action_train_mix.py \
  --upchieve-train-file data/processed/action/upchieve_english_social_train.jsonl
```

For the additive v2 round, export the new split, merge the exported UPChieve train files, then compose the mixed train file:

```bash
uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/upchieve_context_train_v2/action_pool_upchieve_english_social_train_v2.annotated.jsonl \
  --output-file data/processed/action/upchieve_english_social_train_v2.jsonl \
  --label-source codebook_v3_manual
```

```bash
uv run scripts/merge_action_exports.py \
  --input-file data/processed/action/upchieve_english_social_train.jsonl \
  --input-file data/processed/action/upchieve_english_social_train_v2.jsonl \
  --output-file data/processed/action/upchieve_english_social_train_v1_v2.jsonl
```

```bash
uv run scripts/compose_action_train_mix.py \
  --upchieve-train-file data/processed/action/upchieve_english_social_train_v1_v2.jsonl \
  --output-file data/processed/action/train_mixed_upchieve_english_social_v2.jsonl
```

Then validate the mixed-context benchmark files:

```bash
uv run prepare.py \
  --strict \
  --allow-provisional-action-splits train \
  --extra-action-file data/processed/action/upchieve_english_social_dev.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_test.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_train.jsonl
```

For the additive v2 round, validate the new exported train files alongside the frozen benchmark:

```bash
uv run prepare.py \
  --strict \
  --allow-provisional-action-splits train \
  --extra-action-file data/processed/action/upchieve_english_social_dev.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_test.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_train.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_train_v2.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_train_v1_v2.jsonl
```

Go back to the normal validation gate only after every train split in the repo is fully relabeled:

```bash
uv run prepare.py --strict
```

## Calibration Advice

- Double-annotate the first 100-200 rows before scaling up.
- Keep `REVIEW` small but real.
- Revise [codebook_v2.md](./codebook_v2.md) or [codebook_v3.md](./codebook_v3.md) when the same disagreement pattern repeats.
- Do not overwrite the original provisional files until the new annotated export is ready.
- If you have a second-subject candidate/action split, add it to the same sprint immediately with `freeze_action_work_package.py --pilot-candidate-file ... --pilot-action-file ...`.
