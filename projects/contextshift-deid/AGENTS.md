# AGENTS.md — contextshift-deid

This file is the repository-specific operating guide for contributors in this repository.

The goal is to keep the repo optimized for reproducible, benchmark-driven de-identification research. Favor rules that protect methodological clarity, data safety, and local repeatability. Do not import web-app conventions that do not apply here.

## 1) Project Identity

This repository is:

- a Python research scaffold for context-aware educational de-identification
- organized around prepared JSONL benchmark splits, local model training, and evaluation
- local-first, with an offline smoke path

This repository is not:

- a Next.js frontend
- a FastAPI service
- a multi-tenant SaaS app
- a database migration project

Do not carry over app-specific rules about routers, CSRF, org scoping, React hooks, shadcn/ui, or accessibility checklists unless the repo actually grows those components.

## 2) Quick Commands

Setup:

```bash
uv sync
```

Initialize directories and validate prepared data:

```bash
uv run prepare.py
uv run prepare.py --strict
uv run prepare.py --strict --allow-provisional-action-splits train
```

Train the candidate detector:

```bash
uv run train.py --stage candidate
```

Real training may require `--model` pointing to a local or cached checkpoint. Only the smoke workflow is guaranteed to run fully offline on a clean machine.

Train the action model:

```bash
uv run train.py --stage action
```

Evaluate candidate predictions:

```bash
uv run eval.py \
  --stage candidate \
  --gold data/processed/candidate/test.jsonl \
  --predictions artifacts/predictions/candidate_test_predictions.jsonl
```

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

Offline smoke workflow:

```bash
uv run scripts/generate_synthetic_benchmark.py
uv run scripts/generate_local_smoke_models.py
uv run scripts/smoke_test.py
```

Make targets:

```bash
make prepare
make generate-synthetic
make generate-smoke-models
make smoke
make train-candidate
make train-action
make eval-action
```

## 3) Non-Negotiable Boundaries

### Never

- Never commit secrets.
- Never commit governed raw data unless the repo owner explicitly intends that data to live here.
- Never paste raw PII into logs, docs, commit messages, or `results.tsv`.
- Never silently change benchmark splits, label semantics, or metric definitions.
- Never turn the smoke path into a network-dependent workflow.

### Fix Immediately

- Broken `prepare.py` validation
- Schema drift between code and `docs/benchmark_schema.md`
- Missing predictions for evaluation IDs
- Corrupt JSONL outputs
- Regressions in worst-context recall reporting or other headline metrics

### Breaking Changes

Backward compatibility is not the priority here. Clean research design is.

Breaking changes are acceptable when they improve the benchmark or pipeline, but they must come with the matching updates to:

- `README.md`
- `docs/benchmark_schema.md`
- relevant scripts or validators
- any affected commands in this file

## 4) Repo Layout

```text
README.md
AGENTS.md
program.md
pyproject.toml
Makefile
prepare.py
train.py
train_candidate.py
train_action.py
eval.py
results.tsv
docs/
roadmap/
scripts/
src/contextshift_deid/
data/
artifacts/
runs/
```

Use directories consistently:

- `src/contextshift_deid/` for reusable Python logic
- top-level `*.py` files for thin CLI entrypoints
- `data/processed/` for prepared benchmark splits
- `artifacts/experiments/` for one-folder-per-run research artifacts
- `artifacts/predictions/` for exported predictions
- `artifacts/smoke_models/` for offline smoke checkpoints
- `runs/` for training outputs
- `results.tsv` for experiment logging

## 5) Data And Schema Rules

Two benchmark views are first-class:

- `candidate`: token-level suspicious-span detection
- `action`: span-level `REDACT / KEEP / REVIEW` classification

Required candidate fields:

- `id`
- `subject`
- `tokens`
- `labels`

Required action fields:

- `id`
- `subject`
- `span_text`
- `context_text`
- `action_label`

Schema discipline:

- Keep JSONL as one object per line in UTF-8.
- Preserve required fields expected by validators in `src/contextshift_deid/data.py`.
- If you add or reinterpret fields, update both validators and [docs/benchmark_schema.md](/Users/chason/contextshift-deid/docs/benchmark_schema.md).
- Do not change default labels casually. `O/B-SUSPECT/I-SUSPECT` and `REDACT/KEEP/REVIEW` are baseline assumptions throughout the repo.
- Legacy redact-only annotations are acceptable for candidate experiments, but do not present them as full `REDACT/KEEP/REVIEW` supervision unless that mapping is explicitly justified.

Import workflow:

- Prefer importer scripts under `scripts/` for converting raw annotations into `data/processed/*.jsonl`.
- Do not hand-edit benchmark splits except for tiny synthetic smoke fixtures.
- Keep raw source files outside committed benchmark outputs unless the repo owner explicitly wants them tracked here.

## 6) Coding Conventions

Prefer the current repo style:

- Python 3.14
- typed functions
- `pathlib.Path` over stringly-typed paths
- small reusable helpers in `src/contextshift_deid/`
- thin command-line wrappers in top-level scripts

Implementation rules:

- Keep training and evaluation scripts explicit and readable.
- Prefer local, deterministic helpers over hidden framework magic.
- Avoid introducing heavy orchestration layers unless the repo complexity actually demands them.
- Keep offline smoke scripts fully self-contained.
- Do not leave placeholder pipeline code or TODO-only branches in core paths.

When fixing bugs:

- Prefer a regression-first change.
- In this repo, that may mean a small failing synthetic example, a validator check, or a smoke-script repro before the implementation change.

## 7) Experiment Workflow

Use the repo as a controlled experiment engine.

Default loop:

1. Inspect current code and `results.tsv`.
2. Make one concrete change.
3. Run the smallest meaningful validation.
4. Evaluate on the fixed dev split.
5. Record the outcome.
6. Keep the change only if it improves the target objective or clearly simplifies the system without hiding regressions.

Experiment artifact rule:

- Every comparable experiment run should create a fresh output folder under `artifacts/experiments/`.
- Comparable experiment scripts should use the shared helper in [experiment_runs.py](/Users/chason/contextshift-deid/src/contextshift_deid/experiment_runs.py) instead of inventing one-off artifact paths.
- A run folder should contain exact predictions, `summary.json`, `metadata.json`, and a human-readable `report.md`.
- This applies to future action-model runners, tuning/sweep scripts, replacement-policy evaluations, and any other dev/test experiment entrypoint.
- Treat `results.tsv` as the compact index and the per-run folder as the full record for paper figures, tables, and error analysis.

Primary optimization mindset:

- protect privacy failures first
- report worst-context performance, not only averages
- track utility preservation alongside privacy metrics
- treat cost and latency as part of the system, not an afterthought

## 8) Verification Expectations

There is not yet a formal pytest suite in this repository. The default verification surface is script-based.

Minimum checks by change type:

- Data/schema/layout changes: `uv run prepare.py --strict`
- Dev/test-only action relabeling sprints while train stays provisional: `uv run prepare.py --strict --allow-provisional-action-splits train`
- Candidate-stage logic changes: candidate train/eval path on the relevant split
- Action-stage logic changes: action train/eval path on the relevant split
- End-to-end pipeline changes: `uv run scripts/smoke_test.py`

If a change is too heavy to fully run, say exactly what was not run and why.

## 9) Results Logging

`results.tsv` is a project artifact, not scratch output.

Expected columns:

```text
commit	stage	primary_metric	status	description
```

Conventions:

- use `candidate`, `action`, or `eval` for `stage`
- use `keep`, `discard`, or `crash` for `status`
- log every comparable dev/test experiment run and every crash
- do not log schema-only checks, exploratory shell commands, or one-off debugging runs
- keep descriptions short and specific

Do not claim an improvement without a comparable evaluation artifact or logged result.

## 10) Git And Change Hygiene

Use conventional commit prefixes:

- `feat:`
- `fix:`
- `docs:`
- `refactor:`
- `test:`
- `chore:`

Before committing:

- run the relevant validation commands for the files you changed
- update docs when behavior, schema, or commands change
- avoid bundling unrelated experiments into one commit

## 11) What To Exclude From This Repo

Do not add rules here just because they are good in another project.

The following sample-project concerns are currently out of scope:

- frontend design system rules
- accessibility acceptance criteria
- API auth and CSRF rules
- org-scoped database queries
- migration naming policies
- React, Next.js, Zustand, or TanStack Query conventions
- deployment checklists for a web service

If this repo later grows those surfaces, add those rules when they become real.
