# Workplan: Mixed-Context De-Identification Cascade

Last updated: 2026-03-16 (post 3-backbone seeded cascade comparison)

## Decision Summary

**Active mainline backbone:** `distilroberta-base v2` (mixed-context, math + UPChieve `train_v1_v2`)
- Controlled 3-seed comparison reversed the preliminary single-run backbone ranking.
- Primary mainline result is now the seeded aggregate, not any earlier historical single run.
- Headline result to carry into the paper:
  - direct-ID rules + encoder + deferral @ `10%`: `0.6229 ± 0.0263` macro F1
  - encoder + deferral @ `10%`: `0.5910 ± 0.0149` macro F1
- Legacy math REDACT recall stayed high at `0.9980 ± 0.0035`.

**Backbone conclusion:** single-run results were misleading.
- `distilroberta-base` is the best seeded backbone on the frozen English/Social test set.
- `roberta-base` is second.
- `ModernBERT-base` is third and also the highest-variance seeded backbone.
- The paper should report all three backbones as a controlled seeded comparison, not as isolated single runs.

**LLM baseline conclusion:** keep LLMs as reference baselines, not as the mainline system.
- Best frontier direct-classifier baseline on dev: Gemini 3.1 Pro at `0.5643`.
- GPT-5.2 with reasoning on dev: `0.5273`.
- Best local direct-classifier Qwen baseline on dev: `0.4475`.
- These remain useful reference points, but the main paper result is now the seeded local encoder cascade.

**System architecture:**
Rules -> Mixed-context `distilroberta-base` encoder -> Calibrated REVIEW deferral -> Optional narrow reviewer -> Human fallback

**Paper thesis:** Math-trained de-identification breaks under subject shift; targeted mixed-context supervision repairs much of the loss; controlled seeded comparison reveals `distilroberta-base` is the strongest and most stable local backbone for this task; and a local-first cascade with sparse direct-ID rules and calibrated REVIEW is a strong system baseline.

---

## Active Work

### Phase 1: Seeded Backbone Comparison And Deferral Calibration
**Status:** Complete (2026-03-16)  
**Priority:** Completed

Run the frozen mixed-context v2 recipe through the cascade stack with seeded evaluation,
then compare all three encoder backbones under the same protocol.

Tasks:
- [x] Run 3-seed suites for `ModernBERT-base`, `roberta-base`, and `distilroberta-base`
- [x] Reuse the same dev-only policy selection protocol at `5%` and `10%` review budgets
- [x] Record seed-level selected strategies, thresholds, temperatures, and realized review rates
- [x] Compare base classifier vs selected-policy performance
- [x] Compare legacy math REDACT recall across the seeded suites
- [x] Treat historical single-run numbers as reference context only

Completed artifacts:
- `artifacts/experiments/20260315_214342_upchieve-modernbert-v2-seed-suite/`
- `artifacts/experiments/20260315_232942_upchieve-roberta-base-v2-seed-suite/`
- `artifacts/experiments/20260315_234849_upchieve-distilroberta-base-v2-seed-suite/`

Primary seeded comparison to cite at selected `10%`:
- `distilroberta-base` direct-ID: `0.6229 ± 0.0263`
- `roberta-base` direct-ID: `0.5842 ± 0.0234`
- `ModernBERT-base` direct-ID: `0.5611 ± 0.0526`

Base vs selected-policy deltas at selected `10%`:
- `distilroberta-base` direct-ID: `0.5136 ± 0.0437` base -> `0.6229 ± 0.0263` selected
- `roberta-base` direct-ID: `0.5055 ± 0.0579` base -> `0.5842 ± 0.0234` selected
- `ModernBERT-base` direct-ID: `0.4901 ± 0.0428` base -> `0.5611 ± 0.0526` selected

Takeaway:
- Deferral helped all three backbones.
- `distilroberta-base` benefited the most and remained the most stable.
- Backbone selection should now be frozen around `distilroberta-base`, not `ModernBERT-base`.

### Phase 2: Direct-ID Rules Front End
**Status:** Complete (2026-03-16)  
**Priority:** Completed

Hard-rule deterministic REDACT for obvious direct identifiers that never need model
judgment.

Tasks:
- [x] Implement rules layer for the validated allowlist: `EMAIL_ADDRESS`, `URL`, `IP_ADDRESS`, `US_BANK_NUMBER`, `US_PASSPORT`, `US_SSN`, `US_DRIVER_LICENSE`, and direct DOB/date-of-birth pattern hits
- [x] Evaluate cascade with the rules front end on frozen dev/test
- [x] Re-run the seeded Phase 1 operating points with rules enabled
- [x] Report marginal improvement from rules vs encoder-only for all three backbones

Notes:
- `PHONE_NUMBER` and `SOCIAL_HANDLE` were intentionally excluded from the final rules profile after spot-checking showed mixed gold labels in the English/Social benchmark.
- The rules layer stayed sparse and behaved sensibly across the seeded suites.
- At selected `10%`, the rules layer helped every backbone:
  - `ModernBERT-base`: `0.5401 ± 0.0286` -> `0.5611 ± 0.0526`
  - `roberta-base`: `0.5517 ± 0.0254` -> `0.5842 ± 0.0234`
  - `distilroberta-base`: `0.5910 ± 0.0149` -> `0.6229 ± 0.0263`

### Phase 3: DistilRoBERTa Recipe Tuning
**Status:** Not started  
**Priority:** Highest remaining modeling experiment

Tune only the winning backbone. Do not reopen a broad multi-backbone search.

Goals:
- Test whether the `distilroberta-base` recipe can beat the current seeded `direct_id` `10%` result of `0.6229 ± 0.0263`
- Preserve the same safety bar on legacy math
- Keep the evaluation protocol fixed and benchmark-driven

Bounded plan:
- [ ] Freeze the current `distilroberta-base` seeded suite as the baseline
- [ ] Run a small dev-driven pilot sweep on `distilroberta-base` only
- [ ] Start with the highest-leverage knobs:
  - learning rate (`1e-5`, `2e-5`, `3e-5`)
  - max length (`256`, `384`, `512`) only if memory permits
  - epochs (`3`, `4`) if dev curves suggest undertraining
- [ ] Use a single seed for pilot ranking, then confirm only the top recipe on all 3 seeds
- [ ] Keep direct-ID rules + selected `10%` as the primary comparison row
- [ ] Reject any candidate that regresses legacy math REDACT recall materially

### Phase 4: Codebook Cleanup / Adjudication Pass
**Status:** Not started  
**Priority:** High, parallel to Phase 3

150–300 row targeted adjudication on `NRP` and `CURRICULAR` edge cases. This is a documented
audit pass, not a silent benchmark edit.

Tasks:
- [ ] Pull 30–50 row disagreement packet from concentrated residual error buckets
- [ ] Review `NRP` cases: distinguish cultural-reference `KEEP` from identity-revealing `REDACT`
- [ ] Review `CURRICULAR` cases: distinguish subject-matter `KEEP` from institution-specific `REDACT`
- [ ] Document adjudication decisions in codebook v4
- [ ] Decide: fold into a later benchmark-v2 update or keep as qualitative paper characterization only

### Phase 5: Optional Qwen-9B Reviewer Ablation
**Status:** Not started  
**Priority:** Medium, paper enrichment only

Run `Qwen3.5-9B` or similar as a reviewer on deferred/hard-bucket cases only. This is a narrow
ablation, not the mainline system.

Tasks:
- [ ] Define the deferred case set from the seeded `distilroberta-base` results
- [ ] Run the reviewer on deferred cases only
- [ ] Compare: encoder-only vs encoder+human review vs encoder+Qwen reviewer+human fallback on the same budget
- [ ] Report whether the reviewer helps on the narrow hard bucket or whether human fallback remains strictly better

### Phase 6: Paper Writing
**Status:** Ready / unblocked  
**Priority:** High, parallel with Phase 3 and Phase 4

Write around the seeded cascade system architecture.

Structure:
1. Problem: math-trained de-identification collapses under subject shift
2. Method: mixed-context supervision + calibrated REVIEW deferral + sparse direct-ID rules
3. Controlled comparison: seeded backbone evaluation reveals `distilroberta-base` is best
4. Results: local seeded encoder cascade is a strong benchmarked baseline against frontier and local LLM references
5. Analysis: error characterization, deferral composition, rules front end value, and reviewer limits

Key tables/figures needed:
- [ ] Zero-shot collapse table (3 architectures × math vs English/Social)
- [ ] Seeded backbone comparison table (ModernBERT vs roberta vs distilroberta)
- [ ] LLM baseline comparison table (frontier + local vs seeded encoder cascade, clearly labeled by split/setting)
- [ ] Deferral calibration curves at `5%` and `10%` review budgets for the winning backbone
- [ ] Error bucket characterization (`NRP`, `CURRICULAR`, institution-boundary)
- [ ] Cascade ablation: rules-only -> encoder -> encoder+deferral -> encoder+reviewer

---

## Deferred / Future Work

These are explicitly out of scope for the current arc:

- **RL-based reward optimization** — No stable reward design for privacy/utility/review-load tradeoff yet. Revisit only after the supervised cascade is fully characterized.
- **Fine-tuning a local LLM as primary classifier** — Evidence does not support this. The local encoder cascade is still the better main path.
- **Autonomous relabeling / auto-research loops** — Keep any future automation bounded to proposal generation or hyperparameter suggestion. No unsupervised benchmark edits.
- **Further backbone proliferation** — The seeded comparison is now complete. Future modeling work should tune `distilroberta-base`, not reopen a broad backbone sweep.
- **Scaling to additional subjects** — Current scope is math + English/Social Studies. Other UPChieve subjects remain future work.

---

## Key Reference Numbers

These numbers come from different splits and settings. Treat them as reference anchors,
not as one apples-to-apples leaderboard.

| Experiment | Split | Setting | Macro F1 | Review % | Math REDACT Recall |
| --- | --- | --- | --- | --- | --- |
| Zero-shot ModernBERT on English/Social | English/Social test | direct classifier | 0.1531 | 1.0% | — |
| Mixed-context v2 ModernBERT (historical single run, reference only) | English/Social test | selected policy | 0.5872 | 11.5% | 1.0000 |
| Mixed-context v2 ModernBERT seeded suite | English/Social test | selected policy, direct-ID rules, 10% | 0.5611 ± 0.0526 | 10.2% ± 4.6% | 0.9960 ± 0.0070 |
| Mixed-context v2 roberta-base seeded suite | English/Social test | selected policy, direct-ID rules, 10% | 0.5842 ± 0.0234 | 12.0% ± 0.5% | 0.9960 ± 0.0035 |
| Mixed-context v2 distilroberta-base seeded suite | English/Social test | selected policy, direct-ID rules, 10% | 0.6229 ± 0.0263 | 10.8% ± 1.6% | 0.9980 ± 0.0035 |
| Math-only ModernBERT deferral (5%) | math test | multiclass deferral | 0.7735 | 5.0% | — |
| Gemini 3.1 Pro (best frontier LLM) | English/Social dev | direct classifier | 0.5643 | 7.5% | — |
| GPT-5.2 with reasoning | English/Social dev | direct classifier | 0.5273 | 11.5% | — |
| Qwen3.5-9B (best local LLM) | English/Social dev | direct classifier | 0.4475 | 4.0% | — |

## Frozen Evaluation Assets

- **Dev set:** `data/processed/action/upchieve_english_social_dev.jsonl`
- **Test set:** `data/processed/action/upchieve_english_social_test.jsonl`
- **Primary mainline artifact:** `artifacts/experiments/20260315_234849_upchieve-distilroberta-base-v2-seed-suite/`
- **Comparison artifacts:**
  - `artifacts/experiments/20260315_232942_upchieve-roberta-base-v2-seed-suite/`
  - `artifacts/experiments/20260315_214342_upchieve-modernbert-v2-seed-suite/`
- **Codebook:** `docs/codebook_v3.md`
