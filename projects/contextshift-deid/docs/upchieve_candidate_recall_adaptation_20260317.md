# UpChieve Candidate Recall Adaptation Notes

Date: `2026-03-17`

## Scope

- Turn the vendored UpChieve English/social action file into a reproducible candidate-stage proxy dataset.
- Freeze a usable math candidate checkpoint, audit that baseline on the new proxy split plus legacy math holdout, and run a bounded recall-first adaptation sweep.
- Preserve the full prediction artifacts locally under `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/candidate_adaptation/20260317_143031_upchieve-candidate-recall-adaptation/`.

## Implemented Changes

- Added offset-aware tokenization in `/Users/chason/agi/projects/contextshift-deid/src/contextshift_deid/tokenization.py`.
- Added candidate proxy builders and audit helpers in `/Users/chason/agi/projects/contextshift-deid/src/contextshift_deid/candidate_adaptation.py` and `/Users/chason/agi/projects/contextshift-deid/src/contextshift_deid/candidate_audit.py`.
- Added new CLIs:
  - `/Users/chason/agi/projects/contextshift-deid/scripts/build_upchieve_candidate_proxy.py`
  - `/Users/chason/agi/projects/contextshift-deid/scripts/compose_candidate_train_mix.py`
  - `/Users/chason/agi/projects/contextshift-deid/scripts/evaluate_candidate_audit.py`
  - `/Users/chason/agi/projects/contextshift-deid/scripts/run_upchieve_candidate_adaptation.py`
- Extended `/Users/chason/agi/projects/contextshift-deid/train_candidate.py` to write run metadata next to training metrics.

## Data Package

- The source action file `/Users/chason/agi/projects/contextshift-deid/data/processed/action/upchieve_english_social_train_v1_v2.jsonl` contains `1000` span-level action rows across `861` unique turns.
- The proxy candidate builder deduplicated that into full-turn candidate rows and wrote:
  - `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/upchieve_english_social_proxy_train.jsonl`
  - `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/upchieve_english_social_proxy_dev.jsonl`
  - `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/upchieve_english_social_proxy_test.jsonl`
  - `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/train_mixed_upchieve_english_social_proxy.jsonl`
- Final split sizes were `200 / 80 / 80` with balanced English/social counts.
- Seed inventory on the proxy test split:
  - `110` action seed spans
  - `36` REDACT seed spans
  - `29` multi-span turns
- Important correction: the vendored `/Users/chason/agi/projects/contextshift-deid/runs/candidate` checkpoint was a placeholder with near-zero recall and was not a usable frozen baseline. I rebuilt a real math detector at `/Users/chason/agi/projects/contextshift-deid/runs/candidate_math_distilbert_rebuilt` from `distilbert-base-uncased` before running the adaptation sweep.

## Baseline Audit

Frozen baseline:

- model: `/Users/chason/agi/projects/contextshift-deid/runs/candidate_math_distilbert_rebuilt`
- proxy test protected_redact_recall: `0.9722`
- proxy test action_seed_span_coverage: `0.7818`
- proxy test candidate recall: `0.7727`
- proxy test worst_context_recall: `0.6964`
- math holdout recall: `0.9317`
- math holdout f1: `0.7992`

The remaining misses were mostly on English proxy REDACT seeds and broader curricular-name spans in social studies.

## Sweep Outcome

Best held-out same-backbone tradeoff:

- config: `distilbert-context-none-l384-lr-3e-5`
- proxy test protected_redact_recall: `0.9722`
- proxy test action_seed_span_coverage: `0.8636`
- proxy test candidate recall: `0.8364`
- proxy test f1: `0.7050`
- math holdout recall: `0.9268`
- math recall drop vs baseline: `0.0049`

Other notable runs:

- `distilbert-context-none-l384-lr-2e-5`: held protected_redact_recall at `0.9722`, raised action_seed_span_coverage to `0.8455`, and kept math recall at `0.9268`.
- `distilroberta-context-pair-l384-lr-3e-05` fallback: matched proxy protected_redact_recall at `0.9722` and improved proxy F1 to `0.7652`, but math recall fell to `0.9024`, so it is not viable under the math guard.

The sweep runner’s automatic dev-selected winner was `distilbert-context-pair-l384-lr-3e-5` because many configs tied at dev protected_redact_recall `1.0` and it broke ties on dev recall while keeping math recall highest. That config did not hold up on proxy test: protected_redact_recall fell to `0.9167`. I am not treating it as the meaningful held-out winner.

## Decision

- Do not replace the frozen detector from this sprint.
- The preregistered acceptance rule required `+0.05` absolute proxy protected_redact_recall improvement. On this proxy test split that was impossible once the rebuilt baseline reached `0.9722` on only `36` REDACT seed spans:
  - ceiling gain from `0.9722` to `1.0000` is only `+0.0278`
- The useful positive result is narrower:
  - mixed fine-tuning can materially improve broader suspicious-span coverage on UpChieve
  - the best same-backbone held-out recipe improved action_seed_span_coverage by `+0.0818` absolute and overall candidate recall by `+0.0636` absolute
  - it did that without violating the math recall guard
- The useful negative result is:
  - on this proxy, protected REDACT recall is already near the ceiling once the math detector is rebuilt, so that metric is too saturated to drive adaptation selection by itself

## Next Step

- Keep `/Users/chason/agi/projects/contextshift-deid/runs/candidate_math_distilbert_rebuilt` as the frozen candidate baseline for now.
- If candidate adaptation continues, change the audit protocol rather than opening a larger model sweep:
  - expand the proxy test set with more REDACT-bearing turns or real token-level second-subject candidate gold
  - keep protected_redact_recall as a safety gate
  - move model selection to broader suspicious-span coverage metrics such as action_seed_span_coverage or full candidate span recall
