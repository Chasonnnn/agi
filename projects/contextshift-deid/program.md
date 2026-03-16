# contextshift-deid

This repository uses an autoresearch-style loop for **benchmark-driven de-identification experiments**.

## Setup

To start a run:

1. Agree on a run tag such as `mar10-candidate` or `mar10-action`.
2. Create a branch: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files:
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `train_candidate.py`
   - `train_action.py`
   - `eval.py`
   - `docs/benchmark_schema.md`
4. Verify split files exist under:
   - `data/processed/candidate/`
   - `data/processed/action/`
5. Verify `results.tsv` exists or initialize it with `prepare.py`.

## Goal

Improve the de-identification system under context shift.

Primary optimization targets:

- maximize worst-context `REDACT` recall
- minimize over-redaction rate
- preserve curricular entities
- keep cost and latency controlled

## What You Can Change

- candidate detector training recipe
- action model training recipe
- feature formatting
- anchor text formatting
- class weighting
- hard-negative sampling
- review thresholds
- routing thresholds
- evaluation summaries

## What You Should Not Change Silently

- benchmark splits
- codebook assumptions
- release governance assumptions
- primary metric definitions without recording the change

## Logging

Every run goes into `results.tsv`:

```text
commit	stage	primary_metric	status	description
```

Use:

- `candidate`, `action`, or `eval` for stage
- `keep`, `discard`, or `crash` for status

## Loop

1. Inspect current best result and git state.
2. Make one concrete change.
3. Commit it.
4. Run the relevant stage.
5. Evaluate on dev.
6. Append the result to `results.tsv`.
7. Keep the change only if it improves the target objective or meaningfully simplifies the system without harming key metrics.

The loop should optimize the **paper claim**, not just one aggregate score.

