# contextshift-deid

> Isolation note: this is a vendored experiment workspace copied from `/Users/chason/contextshift-deid` into the AGI repo. It includes code, docs, processed benchmark splits, and lightweight baseline summaries needed for the privacy-first action-model sprint. Heavy local state such as `models/`, `runs/`, and full experiment artifacts was intentionally not copied; point `--model` at a local checkpoint or place a compatible snapshot under `projects/contextshift-deid/models/`.

Context-aware educational de-identification research scaffold for **privacy-preserving tutoring and instructional text** under subject shift.

This repository is a fresh project inspired by the small, agent-friendly `autoresearch` workflow, but it is built for a different problem:

- educational tutoring transcripts and related text
- privacy protection under subject and genre shift
- explicit `REDACT / KEEP / REVIEW` action decisions
- local-first experimentation with small encoder and classifier models
- benchmark-driven optimization for a paper-ready system

The current goal is to support a research program around the following claim:

> Named entities change semantic role across educational contexts, so de-identification systems must jointly optimize privacy recall, curricular utility, and cost under context shift.

For the execution plan, see [roadmap/milestone.md](./roadmap/milestone.md).

## Problem Statement

De-identification in education is not just a generic NER problem.

The same surface form can play very different roles:

- in **math**, names in word problems are often incidental and can be replaced
- in **history** or **literature**, names are often part of the lesson content and should be preserved
- in **group chat**, **office hours**, or **support conversations**, references may be personal, contextual, or identifying only in combination

That means the right decision is not just:

> "Is this a person or location mention?"

It is:

> "In this subject, speaker context, and anchor context, should this span be redacted, preserved, or deferred?"

This repo is designed to support that decision boundary as an empirical research problem.

## Research Questions

The working research questions are:

1. How badly do existing de-identification systems fail under educational context shift?
2. Can a local-first cascade reduce over-redaction while keeping privacy recall high?
3. How much does anchor text help preserve curricular entities?
4. Can a selective `REVIEW` option improve risk control compared with forced binary decisions?
5. Can a mostly local system approach stronger API-LLM baselines at lower cost and latency?

## Intended Contribution

The project is aimed at a paper where the main contribution is **not** “we fine-tuned another model.”

The target contribution is:

- a sharper task definition for educational de-identification under context shift
- a benchmark with subject-aware failure modes
- a local, cost-aware cascade for `REDACT / KEEP / REVIEW`
- an evaluation protocol that treats privacy and utility jointly

The method contribution should be centered on the **action decision**, not only the first-pass detector.

## System Overview

The intended cascade is:

1. **Direct-ID rules**
   For email, phone, URL, DOB, handles, and obvious identifiers.
2. **Candidate span detector**
   A high-recall local encoder that proposes suspicious spans.
3. **Intent and context features**
   Examples: ask-for-PII, offer-PII, off-platform move, anchor overlap, speaker role, subject prior.
4. **Action model**
   Predict `REDACT`, `KEEP`, or `REVIEW`.
5. **Escalation**
   Only ambiguous cases go to a larger local model or human review.
6. **Replacement**
   Happens only after the action is settled.

This separation is deliberate:

- the candidate detector should be recall-first
- the action model should handle semantic role and ambiguity
- the review path should be treated as a feature, not a failure

## Repo Layout

```text
README.md
program.md
pyproject.toml
Makefile
prepare.py
train.py
train_candidate.py
train_action.py
eval.py
results.tsv
docs/benchmark_schema.md
roadmap/milestone.md
src/contextshift_deid/constants.py
src/contextshift_deid/data.py
src/contextshift_deid/metrics.py
src/contextshift_deid/schemas.py
```

## Current Workflow

The repo is designed around a small set of repeatable commands:

Project Python version:

```bash
uv run python --version
```

This repo is currently intended to run under the managed `uv` environment using Python 3.14. The checked-in [.python-version](/Users/chason/contextshift-deid/.python-version) file makes that explicit so the project interpreter stays separate from your system Python.

Install dependencies:

```bash
uv sync
```

Initialize local directories and validate any prepared data:

```bash
uv run prepare.py
```

Use `uv run prepare.py --strict` before benchmarked action runs. Strict mode now fails when action splits still contain provisional `legacy_redact_only` supervision.
During the dev/test relabeling sprint, use `uv run prepare.py --strict --allow-provisional-action-splits train` so manual dev/test labels can be validated before train is relabeled.

Train the candidate detector:

```bash
uv run train.py --stage candidate
```

The candidate trainer selects the best dev checkpoint by recall by default. Override with `--selection-metric f1` or `--selection-metric precision` only when you explicitly want a different tradeoff.
Use `--context-mode pair` when you want to feed the turn-window context as a second encoder sequence during candidate training and frozen inference.

Train the action model:

```bash
uv run train.py --stage action
```

Train a binary `REDACT / KEEP` action model and reserve `REVIEW` for post-hoc deferral:

```bash
uv run train_action.py \
  --model models/ModernBERT-base \
  --label-mode binary \
  --binary-review-handling drop \
  --train-file data/processed/action/train.jsonl \
  --dev-file data/processed/action/dev.jsonl \
  --output-dir runs/action_binary \
  --seed 42
```

`train_action.py` also supports recipe-tuning knobs for seeded ModernBERT experiments. Use
`--selection-metric macro_f1` to select checkpoints by dev macro F1 instead of redact recall, and
use `--warmup-ratio` / `--weight-decay` to sweep a bounded training recipe without editing code.

When offline, the training and prediction scripts automatically resolve common backbone names like `roberta-base`, `distilroberta-base`, and `ModernBERT-base` to repo-local copies under `models/` when those directories exist.

Evaluate candidate predictions:

```bash
uv run eval.py \
  --stage candidate \
  --gold data/processed/candidate/test.jsonl \
  --predictions artifacts/predictions/candidate_test_predictions.jsonl
```

Candidate evaluation also reports speaker-role and token-length slices so dev/test differences are easier to interpret.

Evaluate action predictions:

```bash
uv run scripts/predict_action.py \
  --model runs/action \
  --input-file data/processed/action/test.jsonl \
  --output-file artifacts/predictions/action_test_predictions.jsonl

uv run eval.py \
  --stage action \
  --gold data/processed/action/test.jsonl \
  --predictions artifacts/predictions/action_test_predictions.jsonl
```

`scripts/predict_action.py` writes the predicted label, max-confidence score, and the full per-class probability vector for `REDACT`, `KEEP`, and `REVIEW`.

Evaluate `REVIEW` as a dev-tuned deferral policy instead of a directly learned class:

```bash
uv run scripts/evaluate_deferral.py \
  --calibration-gold data/processed/action/dev.jsonl \
  --calibration-predictions artifacts/predictions/action_dev_predictions.jsonl \
  --eval-gold data/processed/action/test.jsonl \
  --eval-predictions artifacts/predictions/action_test_predictions.jsonl \
  --run-name action-deferral \
  --fit-temperature
```

The deferral runner creates a fresh experiment folder under `artifacts/experiments/` with:

- `predictions/` containing the selected remapped test predictions for each target review rate
- `summary.json` with base metrics and selected policy metrics
- `metadata.json` with the calibrated temperature and source files
- `report.md` with a human-readable review-rate vs. coverage summary
- `sweep_results.json` with all calibration sweep points

Analyze one frozen UPChieve experiment at the selected review target:

```bash
uv run scripts/analyze_upchieve_experiment.py \
  --gold-file data/processed/action/upchieve_english_social_test.jsonl \
  --base-prediction-file artifacts/predictions/upchieve_english_social_test_predictions_mixed_modernbert_v2_b4_l384.jsonl \
  --summary-file artifacts/experiments/20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384/summary.json \
  --selected-review-rate 0.10 \
  --run-name upchieve-modernbert-v2-analysis
```

Run the seeded backbone mixed-context cascade package with resumable per-seed artifacts:

```bash
uv run scripts/run_upchieve_modernbert_seed_suite.py \
  --model ModernBERT-base \
  --run-root artifacts/experiments/modernbert_recipe_tuning \
  --train-file data/processed/action/train_mixed_upchieve_english_social_v2.jsonl \
  --dev-file data/processed/action/upchieve_english_social_dev.jsonl \
  --test-file data/processed/action/upchieve_english_social_test.jsonl \
  --legacy-math-file data/processed/action/test.jsonl \
  --seeds 13,42,101 \
  --selection-metric macro_f1 \
  --warmup-ratio 0.1 \
  --weight-decay 0.0 \
  --target-review-rates 0.05,0.10 \
  --run-name upchieve-modernbert-v2-seed-suite
```

The suite reruns the requested mixed-context recipe for the requested backbone, exports dev/test/math predictions, applies the optional direct-ID override profile, calibrates deferral on dev only, and writes one seed artifact per run plus one aggregate comparison artifact. By default, recipe-tuning suites now live under `artifacts/experiments/modernbert_recipe_tuning/` so they do not mix with older experiment folders. Re-running the same command resumes the latest incomplete suite for that run name inside the selected `--run-root`; pass `--force` to recompute completed seed steps.

For temperature-only reevaluation, reuse an existing suite's checkpoints and probability exports without retraining:

```bash
uv run scripts/run_upchieve_modernbert_seed_suite.py \
  --run-root artifacts/experiments/modernbert_recipe_tuning \
  --run-name upchieve-modernbert-v2-temp-reeval \
  --model ModernBERT-base \
  --seeds 13,42,101 \
  --target-review-rates 0.05,0.10 \
  --fit-temperature \
  --recompute-deferral-only \
  --source-suite-root artifacts/experiments/20260315_214342_upchieve-modernbert-v2-seed-suite
```

The reevaluation-only path writes a fresh suite under the selected `--run-root`, points each seed artifact back to the reused training checkpoint, regenerates deferral artifacts, and reports selected-policy stability for the 10% no-rules operating point:

- `unique_strategy_count`
- `strategy_labels`
- `test_review_rate_mean`
- `test_review_rate_stdev`
- `macro_f1_mean`
- `macro_f1_stdev`

When `--source-suite-root` is provided, the suite summary also compares the new stability profile against the source suite and records whether the temperature-only pass clears the carry-forward gate.

Use a different `--model` and `--run-name` to compare other backbones, for example `roberta-base` or `distilroberta-base`.

Run the dev-only Cornell-gateway LLM ceiling baseline:

```bash
uv run scripts/run_upchieve_llm_ceiling.py \
  --gold-file data/processed/action/upchieve_english_social_dev.jsonl \
  --model openai.gpt-5.2 \
  --reasoning-effort high \
  --output-file artifacts/predictions/upchieve_english_social_dev_llm_ceiling_openai-gpt-5-2.jsonl \
  --run-name upchieve-llm-ceiling
```

Run a local open-model dev ceiling baseline with the frozen prompt/schema:

```bash
uv run scripts/run_upchieve_local_open_ceiling.py \
  --gold-file data/processed/action/upchieve_english_social_dev.jsonl \
  --model-path models/Qwen3.5-0.8B \
  --thinking-mode non-thinking \
  --output-file artifacts/predictions/upchieve_english_social_dev_local_open_qwen3-5-0-8b_nothink.jsonl \
  --run-name upchieve-local-open-qwen35-08b
```

Compare multiple LLM or local-open summary artifacts against the frozen ModernBERT v2 dev baseline:

```bash
uv run scripts/compare_upchieve_llm_runs.py \
  --summary "Qwen3.5-0.8B=artifacts/experiments/<run_08b>/summary.json" \
  --summary "Qwen3.5-4B=artifacts/experiments/<run_4b>/summary.json" \
  --summary "Qwen3.5-9B=artifacts/experiments/<run_9b>/summary.json" \
  --run-name upchieve-qwen35-compare
```

Run the synthetic smoke test:

```bash
uv run scripts/smoke_test.py
```

The smoke runner is fully offline. It creates tiny local BERT checkpoints under `artifacts/smoke_models/`, trains both stages on synthetic JSONL splits, exports dev predictions, evaluates them, and appends rows to `results.tsv`.

Run a paper-ready holdout experiment:

```bash
uv run scripts/run_candidate_holdout_eval.py --run-name candidate-math-holdout
```

Bootstrap the current 1-2 week action work package:

```bash
uv run scripts/freeze_action_work_package.py --headline privacy_recall
```

The bootstrap script picks the best locked candidate checkpoint for the chosen headline, reruns one clean holdout artifact under `artifacts/experiments/`, exports frozen train/dev/test candidate predictions, and builds frozen action annotation pools under `artifacts/annotation/packages/`.

Each holdout invocation creates a new timestamped directory under `artifacts/experiments/` with:

- `predictions/`
- `summary.json`
- `metadata.json`
- `report.md`

Use `results.tsv` as the compact experiment index, and use the per-run folder for the full artifact trail.

Import the provided UPchieve math annotations into repo-native JSONL splits:

```bash
uv run scripts/import_upchieve_math.py --write-provisional-action
```

Render deterministic redacted previews from the imported sessions:

```bash
uv run scripts/render_redactions.py --strategy typed_placeholder
```

Build an action annotation pool:

```bash
uv run scripts/build_action_annotation_pool.py \
  --candidate-file data/processed/candidate/dev.jsonl \
  --provisional-action-file data/processed/action/dev.jsonl \
  --prediction-file artifacts/predictions/candidate_dev_predictions.jsonl
```

Run the local manual annotation tool:

```bash
uv run scripts/annotate_action_pool.py \
  --input-file artifacts/annotation/action_pool_dev.jsonl \
  --annotator your_name
```

Compare two annotation passes and export final action JSONL:

```bash
uv run scripts/compare_action_annotations.py \
  --left-file artifacts/annotation/action_pool_dev.annotated.jsonl \
  --right-file artifacts/annotation/action_pool_dev_second.annotated.jsonl

uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/action_pool_dev.annotated.jsonl \
  --output-file data/processed/action/dev.jsonl
```

After `dev` and `test` are manual, repeat the same pool and export flow for `train` so the action model is no longer trained on provisional all-`REDACT` labels.

Bootstrap a local TIMSS science pilot from raw AU/US transcripts:

```bash
uv run scripts/import_timss_science.py \
  --input-root TIMSS \
  --countries AU US
```

This creates normalized turn records under `data/interim/timss_science_turns.jsonl` and candidate-annotation seed files under `artifacts/annotation/timss_science/`.
The raw TIMSS source should stay local-only under either `TIMSS/` or `data/raw/timss/`; do not commit it.

Annotate candidate BIO labels in the terminal:

```bash
uv run scripts/annotate_candidate_pool.py \
  --input-file artifacts/annotation/timss_science/candidate_pool_test.jsonl \
  --annotator your_name
```

The annotated output file is valid candidate JSONL and can be used directly for evaluation, candidate-model training, or action-pool construction.
For example, build a manual action pool from the annotated candidate test split:

```bash
uv run scripts/build_action_annotation_pool.py \
  --candidate-file artifacts/annotation/timss_science/candidate_pool_test.annotated.jsonl \
  --output-file artifacts/annotation/timss_science/action_pool_test.jsonl
```

Then label `REDACT / KEEP / REVIEW` with the existing action workflow:

```bash
uv run scripts/annotate_action_pool.py \
  --input-file artifacts/annotation/timss_science/action_pool_test.jsonl \
  --annotator your_name

uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/timss_science/action_pool_test.annotated.jsonl \
  --output-file data/processed/action/timss_test.jsonl
```

Bootstrap the anonymized UPChieve English/Social Studies action pilot:

```bash
uv run scripts/import_upchieve_context_pilot.py
uv run scripts/build_upchieve_action_pilot.py
```

This creates normalized tagged turns under `data/interim/upchieve_context_pilot_turns.jsonl` and two 200-row annotation pools under `artifacts/annotation/upchieve_context_pilot/`.

Annotate the pilot with the exact-tag highlighter and semantic-role prompt:

```bash
uv run scripts/annotate_action_pool.py \
  --input-file artifacts/annotation/upchieve_context_pilot/action_pool_upchieve_english_social_dev.jsonl \
  --annotator your_name
```

Export the pilot with `codebook_v3` provenance:

```bash
uv run scripts/export_action_annotations.py \
  --input-file artifacts/annotation/upchieve_context_pilot/action_pool_upchieve_english_social_dev.annotated.jsonl \
  --output-file data/processed/action/upchieve_english_social_dev.jsonl \
  --label-source codebook_v3_manual
```

Validate the nonstandard pilot split alongside the default benchmark files:

```bash
uv run prepare.py \
  --strict \
  --allow-provisional-action-splits train \
  --extra-action-file data/processed/action/upchieve_english_social_dev.jsonl \
  --extra-action-file data/processed/action/upchieve_english_social_test.jsonl
```

Treat those exported pilot files as the frozen `benchmark-v1` for UPChieve cross-context work. Build the next mixed-context package from unused sessions:

```bash
uv run scripts/analyze_upchieve_zero_shot.py
uv run scripts/build_upchieve_action_train.py
```

That writes:

- a timestamped zero-shot analysis artifact under `artifacts/experiments/`
- a 600-row train annotation pool under `artifacts/annotation/upchieve_context_train/`

After you finish annotating and exporting `data/processed/action/upchieve_english_social_train.jsonl`, compose the mixed train file:

```bash
uv run scripts/compose_action_train_mix.py \
  --upchieve-train-file data/processed/action/upchieve_english_social_train.jsonl
```

For the targeted additive scaling round, build the v2 pool from sessions not used in `benchmark-v1` or train v1:

```bash
uv run scripts/build_upchieve_action_train_v2.py
```

That writes a 400-row targeted pool under `artifacts/annotation/upchieve_context_train_v2/` with:

- 200 `english`
- 200 `social_studies`
- within each subject, 100 `natural` and 100 `challenge`

The v2 builder broadens the English-side language filter to exclude explicit `spanish`, `espanol`, `español`, `arabic`, `persian`, `farsi`, and `urdu` markers, excludes English-topic target turns with Arabic-script text, and overweights the observed failure buckets:

- English natural `PERSON` rows with direct greetings or self-identification
- English challenge `PERSON` and `LOCATION`
- Social Studies challenge `LOCATION`, `NRP`, and `PERSON`

After you annotate and export `data/processed/action/upchieve_english_social_train_v2.jsonl`, merge it with train v1 and compose the v2 mixed train file:

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

## Data Model

This repo currently expects **prepared JSONL benchmark splits** under `data/processed/`:

```text
data/processed/candidate/train.jsonl
data/processed/candidate/dev.jsonl
data/processed/candidate/test.jsonl
data/processed/action/train.jsonl
data/processed/action/dev.jsonl
data/processed/action/test.jsonl
```

The schema details live in [docs/benchmark_schema.md](./docs/benchmark_schema.md).

In short:

- **candidate** examples are token-level records with `tokens` and `labels`
- **action** examples are span-level records with `context_text`, `span_text`, and `action_label`

The repo does not assume that raw transcripts can be released publicly. It supports internal-only or governed benchmark workflows.

The currently processed real benchmark is still math-only, so a second-subject pilot needs its own processed candidate/action files before the context-shift claim can be tested on real held-out data.

## Metrics

The paper-facing metrics this repo is designed around are:

- worst-context `REDACT` recall
- macro action F1
- curricular entity retention rate (CERR)
- over-redaction rate (ORR)
- review rate
- optional cost and latency summaries

The current evaluation philosophy is:

- optimize for **privacy failures first**
- explicitly measure **over-redaction**
- never report privacy metrics without **utility preservation**
- compare systems by **worst context**, not only macro average

Suggested dev objective:

```text
min-context REDACT recall
- lambda_1 * ORR
- lambda_2 * utility_drop
- lambda_3 * normalized_cost
```

## What Goes Where

Use the repo like this:

- `prepare.py`
  Initializes directories and validates splits.
- `train_candidate.py`
  Trains the suspicious-span candidate detector.
- `train_action.py`
  Trains the action classifier for `REDACT / KEEP / REVIEW`.
- `eval.py`
  Scores predictions against gold labels.
- `results.tsv`
  Logs experiments in a simple table.
- `program.md`
  Contains the autoresearch-style agent loop instructions.
- `scripts/generate_synthetic_benchmark.py`
  Creates a tiny synthetic benchmark for pipeline smoke tests.
- `scripts/generate_local_smoke_models.py`
  Creates tiny local transformer checkpoints so smoke tests do not depend on network access.
- `scripts/import_upchieve_math.py`
  Converts the provided UPchieve math annotation export into candidate splits and optional provisional action splits.
- `scripts/import_upchieve_context_pilot.py`
  Normalizes anonymized UPChieve English/Social Studies turns into tagged-turn pilot rows for the action benchmark.
- `scripts/build_upchieve_action_pilot.py`
  Builds fixed-size English/Social Studies `natural` and `challenge` action annotation pools from the anonymized UPChieve pilot rows.
- `scripts/analyze_upchieve_zero_shot.py`
  Compares the three frozen zero-shot UPChieve pilot runs, writes a timestamped analysis artifact, and reports the cross-context failure slices.
- `scripts/build_upchieve_action_train.py`
  Builds a 600-row English/Social Studies UPChieve train annotation pool from sessions not used in the frozen pilot benchmark.
- `scripts/build_upchieve_action_train_v2.py`
  Builds a 400-row targeted UPChieve train-v2 pool with stronger multilingual exclusion and failure-bucket sampling.
- `scripts/compose_action_train_mix.py`
  Merges the relabeled math action train split with the exported UPChieve train split for mixed-context training.
- `scripts/merge_action_exports.py`
  Validates and merges multiple exported action JSONL files, failing on duplicate ids before mixed-train composition.
- `scripts/import_timss_science.py`
  Normalizes raw TIMSS science transcripts into split-scoped candidate annotation seed files for local AU/US pilots.
- `scripts/annotate_candidate_pool.py`
  Runs a local terminal annotator for candidate BIO labels on tokenized transcript turns.
- `scripts/build_action_annotation_pool.py`
  Builds a manual action-label pool from candidate rows, provisional positives, and optional candidate predictions.
- `scripts/annotate_action_pool.py`
  Runs a local terminal annotator for `REDACT / KEEP / REVIEW` labeling.
- `scripts/compare_action_annotations.py`
  Compares two annotation files and emits a disagreement subset for adjudication.
- `scripts/export_action_annotations.py`
  Exports annotated pool rows into repo-native action JSONL splits.
- `scripts/predict_action.py`
  Runs frozen action-model inference on any action split so dev/test learnability can be checked after relabeling.
- `scripts/freeze_action_work_package.py`
  Freezes one candidate operating point, reruns a clean holdout artifact, and writes dev/test action-label pools for the current annotation sprint.
- `scripts/render_redactions.py`
  Renders imported sessions with typed placeholders, masks, deletes, or fake surrogate replacements for comparison.
- `scripts/smoke_test.py`
  Exercises synthetic data generation, tiny training runs, prediction export, evaluation, and results logging.

## Autoresearch-Style Loop

This project borrows a few ideas from `autoresearch`, but only at the workflow level:

- a small number of core files
- a plain-text experiment loop
- git-friendly changes
- a persistent `results.tsv` log

What is carried over:

- lightweight experimentation rhythm
- one-change-at-a-time iteration
- explicit keep/discard logic

What is **not** carried over:

- scratch pretraining
- mutable model architecture as the main research axis
- one-file training systems

Here, the outer loop is useful for tuning:

- context window sizes
- anchor formatting
- hard-negative sampling
- class weighting
- review thresholds
- model choice within a constrained family

## Project Boundaries

This repo intentionally does **not** assume:

- a public benchmark release from day one
- a frontier API dependency in the core method
- scratch pretraining
- a hard subject router
- a production-ready replacement engine in v1

This is a benchmark-and-pipeline research repo, not a foundation-model repo and not yet a production de-identification service.

## Immediate Next Steps

The next concrete tasks are:

1. finalize the benchmark label inventory and codebook
2. export candidate and action JSONL splits from your annotated data
3. run baseline models on the pilot benchmark
4. implement the first real utility evaluation tasks
5. start the candidate detector and action model baselines

The milestone plan for doing that is in [roadmap/milestone.md](./roadmap/milestone.md).
