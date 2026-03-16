# ModernBERT-base Seed Suite

- Run name: `upchieve-modernbert-v2-seed-suite`
- Run root: `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/modernbert_recipe_tuning`
- Seeds: `13, 42, 101`
- Target review rates: `0.05, 0.10`
- Checkpoint selection metric: `redact_recall`
- Warmup ratio: `0.0`
- Weight decay: `0.0`
- Fit temperature: `True`
- Recompute deferral only: `False`
- Source suite root: `n/a`
- Seed 42 anchor gap vs historical 10% test macro F1 0.5872: +0.0286 (reference only — see note)

> The historical ModernBERT v2 result came from a single earlier run before this seeded suite existed. It is kept as reference context only, because the original training path did not control all randomness before model initialization.

## Seed-Level Selected Policies

| seed | variant | target | strategy | temperature | parameters | dev_review | test_review | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | no_rules | 0.05 | review_probability | 2.3458 | `{"min_review_probability": 0.31188481644711863}` | 5.0% | 4.5% | 0.4914 | 0.7850 | 0.5294 | 0.9355 | 10.7% | 64.7% | 35.3% | 1.0000 |
| 13 | no_rules | 0.10 | entropy | 2.3458 | `{"min_entropy": 1.0122923149741567}` | 10.0% | 11.0% | 0.5633 | 0.8000 | 0.4706 | 0.9226 | 32.1% | 70.6% | 29.4% | 1.0000 |
| 13 | direct_id | 0.05 | review_probability | 2.3458 | `{"min_review_probability": 0.31188481644711863}` | 5.0% | 4.5% | 0.5039 | 0.7900 | 0.5882 | 0.9355 | 10.7% | 70.6% | 29.4% | 1.0000 |
| 13 | direct_id | 0.10 | entropy | 2.3458 | `{"min_entropy": 1.0122923149741567}` | 10.0% | 9.5% | 0.6088 | 0.8150 | 0.6471 | 0.9226 | 32.1% | 70.6% | 29.4% | 1.0000 |
| 42 | no_rules | 0.05 | review_probability | 2.5785 | `{"min_review_probability": 0.26942987711020705}` | 5.0% | 3.5% | 0.5654 | 0.8200 | 0.7059 | 0.9548 | 14.3% | 76.5% | 23.5% | 1.0000 |
| 42 | no_rules | 0.10 | review_probability | 2.5785 | `{"min_review_probability": 0.24494085855420053}` | 7.5% | 8.5% | 0.6158 | 0.8250 | 0.5882 | 0.9419 | 32.1% | 76.5% | 23.5% | 1.0000 |
| 42 | direct_id | 0.05 | review_probability | 2.6611 | `{"min_review_probability": 0.2715912994142245}` | 5.0% | 3.5% | 0.5654 | 0.8200 | 0.7059 | 0.9548 | 14.3% | 76.5% | 23.5% | 1.0000 |
| 42 | direct_id | 0.10 | review_probability | 2.6611 | `{"min_review_probability": 0.24794279108427783}` | 7.5% | 8.0% | 0.6314 | 0.8300 | 0.6471 | 0.9419 | 32.1% | 76.5% | 23.5% | 1.0000 |
| 101 | no_rules | 0.10 | none | 5.0000 | `{}` | 9.5% | 8.0% | 0.5796 | 0.8250 | 0.4118 | 0.9742 | 25.0% | 70.6% | 29.4% | 0.9940 |
| 101 | direct_id | 0.10 | none | 5.0000 | `{}` | 9.5% | 7.0% | 0.6340 | 0.8400 | 0.5882 | 0.9742 | 25.0% | 76.5% | 23.5% | 0.9940 |

## Aggregate Mean ± Std

| variant | target | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | review_rate | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_rules | 0.05 | 0.5284 ± 0.0523 | 0.8025 ± 0.0247 | 0.6176 ± 0.1248 | 0.9452 ± 0.0137 | 0.1250 ± 0.0253 | 0.0400 ± 0.0071 | 0.7059 ± 0.0832 | 0.2941 ± 0.0832 | 1.0000 ± 0.0000 |
| no_rules | 0.10 | 0.5862 ± 0.0269 | 0.8167 ± 0.0144 | 0.4902 ± 0.0899 | 0.9462 ± 0.0261 | 0.2976 ± 0.0412 | 0.0917 ± 0.0161 | 0.7255 ± 0.0340 | 0.2745 ± 0.0340 | 0.9980 ± 0.0035 |
| direct_id | 0.05 | 0.5347 ± 0.0435 | 0.8050 ± 0.0212 | 0.6471 ± 0.0832 | 0.9452 ± 0.0137 | 0.1250 ± 0.0253 | 0.0400 ± 0.0071 | 0.7353 ± 0.0416 | 0.2647 ± 0.0416 | 1.0000 ± 0.0000 |
| direct_id | 0.10 | 0.6248 ± 0.0139 | 0.8283 ± 0.0126 | 0.6275 ± 0.0340 | 0.9462 ± 0.0261 | 0.2976 ± 0.0412 | 0.0817 ± 0.0126 | 0.7451 ± 0.0340 | 0.2549 ± 0.0340 | 0.9980 ± 0.0035 |

## Selected 10% No-Rules Stability

| metric | value |
| --- | --- |
| unique_strategy_count | 3 |
| strategy_labels | `["entropy", "none", "review_probability"]` |
| test_review_rate_mean | 9.2% |
| test_review_rate_stdev | 0.0161 |
| macro_f1_mean | 0.5862 |
| macro_f1_stdev | 0.0269 |
| all_legacy_math_redact_recall_one | `False` |

## Base Classifier vs Selected Policy

| variant | target | base_macro_f1 | selected_macro_f1 | delta_macro_f1 | base_accuracy | selected_accuracy | delta_accuracy | base_redact_recall | selected_redact_recall | delta_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_rules | 0.05 | 0.5044 ± 0.0700 | 0.5284 ± 0.0523 | +0.0240 | 0.8083 ± 0.0208 | 0.8025 ± 0.0247 | -0.0058 | 0.5882 ± 0.1765 | 0.6176 ± 0.1248 | +0.0294 |
| no_rules | 0.10 | 0.5044 ± 0.0700 | 0.5862 ± 0.0269 | +0.0819 | 0.8083 ± 0.0208 | 0.8167 ± 0.0144 | +0.0083 | 0.5882 ± 0.1765 | 0.4902 ± 0.0899 | -0.0980 |
| direct_id | 0.05 | 0.5263 ± 0.0953 | 0.5347 ± 0.0435 | +0.0083 | 0.8150 ± 0.0250 | 0.8050 ± 0.0212 | -0.0100 | 0.6667 ± 0.0899 | 0.6471 ± 0.0832 | -0.0196 |
| direct_id | 0.10 | 0.5263 ± 0.0953 | 0.6248 ± 0.0139 | +0.0984 | 0.8150 ± 0.0250 | 0.8283 ± 0.0126 | +0.0133 | 0.6667 ± 0.0899 | 0.6275 ± 0.0340 | -0.0392 |

## Historical Frozen Backbone Results (single-run, uncontrolled seed — reference context only)

| backbone | selected 10% macro_f1 | selected 10% review_rate | source |
| --- | --- | --- | --- |
| ModernBERT-base v2 (historical, uncontrolled seed) | 0.5872 | 11.5% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384/summary.json` |
| roberta-base v2 (historical, uncontrolled seed) | 0.4090 | 8.5% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_225018_upchieve-english-social-mixed-roberta-base-v2-b4-l384-fresh/summary.json` |
| distilroberta-base v2 (historical, uncontrolled seed) | 0.4708 | 9.0% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_225845_upchieve-english-social-mixed-distilroberta-v2-b4-l384-fresh/summary.json` |
| roberta-base v1 from math | 0.4882 | 8.5% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_193820_upchieve-english-social-mixed-roberta-base-v1-from-math-b4-l384/summary.json` |
| roberta-base v1 fresh | 0.5647 | 14.0% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_201224_upchieve-english-social-mixed-roberta-base-v1-b4-l384-fresh/summary.json` |
| distilroberta-base v1 from math | 0.5164 | 5.0% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_193849_upchieve-english-social-mixed-distilroberta-v1-from-math-b4-l384/summary.json` |
| distilroberta-base v1 fresh | 0.5120 | 14.5% | `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/20260314_201314_upchieve-english-social-mixed-distilroberta-v1-b4-l384-fresh/summary.json` |
