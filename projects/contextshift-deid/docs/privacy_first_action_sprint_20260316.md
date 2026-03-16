# Privacy-First Action Sprint Notes

Date: `2026-03-16`

## Scope

- Freeze the candidate detector and work only on the action stage.
- Isolate the experiment in the vendored AGI workspace.
- Keep full prediction exports locally for error analysis even though `artifacts/` stays gitignored.

## Implemented Changes

- Added `marked_turn_v1` action inputs that wrap the exact target span as `[TGT]...[/TGT]` when UPChieve offsets are available, with flat-prompt fallback otherwise.
- Added an optional auxiliary semantic-role head for `PRIVATE / CURRICULAR / AMBIGUOUS`.
- Added `subject_action_balanced` sampling with per-row weight `1 / subject_count * 1 / action_label_count`.
- Added shared inference helpers so training and prediction produce aligned probability exports, including semantic-role outputs when present.
- Added direct-ID 10% policy-based checkpoint selection and preserved per-checkpoint dev prediction files under `training/model/checkpoint_selection/`.

## Outcome

Kept recipe:

- Branch A, marked-turn only: `--action-input-format marked_turn_v1 --semantic-role-head-mode none --sampler-mode none`
- 3-seed direct-ID 10% aggregate from `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/modernbert_recipe_tuning/20260316_014131_upchieve-modernbert-v2-branch-a-marked-turn-gc/report.md`
- `macro_f1 0.7156 ± 0.0140`
- `accuracy 0.8700 ± 0.0050`
- `redact_recall 0.5490 ± 0.0340`
- `protected_redact_rate 0.7647 ± 0.0588`
- `review_rate 0.1050 ± 0.0087`
- `legacy_math_redact_recall 0.9960 ± 0.0070`

Rejected probes:

- Branch B, marked-turn + multitask + balanced sampler: direct-ID 10% `macro_f1 0.6181`, `redact_recall 0.6471`, but `legacy_math_redact_recall 0.9639`. Report: `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/modernbert_recipe_tuning/20260316_020927_upchieve-modernbert-v2-branch-b-multitask-gc/report.md`
- Branch D, marked-turn + balanced sampler only: direct-ID 10% `macro_f1 0.7329`, `redact_recall 0.5882`, but `legacy_math_redact_recall 0.9699`. Report: `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/modernbert_recipe_tuning/20260316_031733_upchieve-modernbert-v2-branch-d-sampler-only-gc/report.md`
- Branch E scout, marked-turn + best flat-prompt recipe on seed 42: direct-ID 10% `macro_f1 0.7125`, `review_rate 0.1000`, `legacy_math_redact_recall 0.9880`. This preserved the relaxed seed-42 math gate but missed the `> 0.73` promotion threshold and underperformed the current marked-turn seed-42 result (`0.7178`), so it was not promoted to 3 seeds. Report: `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/modernbert_recipe_tuning/20260316_133741_upchieve-modernbert-v2-branch-e-marked-turn-best-recipe/report.md`
- Policy-selector probe on Branch A later checkpoints recovered recall on two seeds but reduced held-out macro F1 versus the kept Branch A aggregate. Probe summaries were written under the relevant `checkpoint_selection/checkpoint-2295/policy_probe/` directories.

## Interpretation

- Marking the exact target turn helps enough to keep.
- The auxiliary semantic-role head is not helping in its current form.
- The balanced sampler is probably increasing privacy sensitivity, but it breaks the legacy-math guard badly enough to reject for now.
- The combined marked-turn plus best-flat-recipe scout did not show clear additivity. Because it used `macro_f1` checkpoint selection and a recipe copied from the flat prompt setting, the right interpretation is inconclusive / likely overlap, not a hard negative on stacking gains.
- The next useful step is not another broad backbone sweep. It is targeted error analysis on the kept Branch A predictions to recover some REDACT recall without giving back the stability and utility gains.
