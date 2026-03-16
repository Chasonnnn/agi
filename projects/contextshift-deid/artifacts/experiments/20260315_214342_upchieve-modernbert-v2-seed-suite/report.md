# ModernBERT Seed Suite

- Run name: `upchieve-modernbert-v2-seed-suite`
- Seeds: `13, 42, 101`
- Target review rates: `0.05, 0.10`
- Seed 42 anchor gap vs historical 10% test macro F1 0.5872: -0.0696 (reference only — see note)

> The historical ModernBERT v2 result came from a single earlier run before this seeded suite existed. It is kept as reference context only, because the original training path did not control all randomness before model initialization.

## Seed-Level Selected Policies

| seed | variant | target | strategy | temperature | parameters | dev_review | test_review | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | no_rules | 0.05 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.6202674004475385, "redact_max_confidence": 0.47617938972588714}` | 5.0% | 4.5% | 0.5193 | 0.7850 | 0.6471 | 0.9161 | 14.3% | 70.6% | 29.4% | 1.0000 |
| 13 | no_rules | 0.10 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.8802208724873031, "redact_max_confidence": 0.4841496492668882}` | 9.5% | 15.5% | 0.5303 | 0.7350 | 0.6471 | 0.8258 | 28.6% | 88.2% | 11.8% | 1.0000 |
| 13 | direct_id | 0.05 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.6202674004475385, "redact_max_confidence": 0.47617938972588714}` | 5.0% | 4.5% | 0.5193 | 0.7850 | 0.6471 | 0.9161 | 14.3% | 70.6% | 29.4% | 1.0000 |
| 13 | direct_id | 0.10 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.8802208724873031, "redact_max_confidence": 0.4841496492668882}` | 9.5% | 15.5% | 0.5303 | 0.7350 | 0.6471 | 0.8258 | 28.6% | 88.2% | 11.8% | 1.0000 |
| 42 | no_rules | 0.05 | asymmetric_margin | 1.0000 | `{"keep_max_redact_keep_margin": 0.6174182744311405, "redact_max_redact_keep_margin": 0.3415769458484752}` | 4.5% | 5.5% | 0.4980 | 0.7950 | 0.5882 | 0.9548 | 3.6% | 88.2% | 11.8% | 1.0000 |
| 42 | no_rules | 0.10 | margin | 1.0000 | `{"max_redact_keep_margin": 0.6981004804089738}` | 7.5% | 8.0% | 0.5176 | 0.7850 | 0.5882 | 0.9355 | 7.1% | 88.2% | 11.8% | 1.0000 |
| 42 | direct_id | 0.05 | asymmetric_margin | 1.0000 | `{"keep_max_redact_keep_margin": 0.6174182744311405, "redact_max_redact_keep_margin": 0.3415769458484752}` | 4.5% | 5.0% | 0.5110 | 0.8000 | 0.6471 | 0.9548 | 3.6% | 88.2% | 11.8% | 1.0000 |
| 42 | direct_id | 0.10 | margin | 1.0000 | `{"max_redact_keep_margin": 0.6981004804089738}` | 7.5% | 7.5% | 0.5313 | 0.7900 | 0.6471 | 0.9355 | 7.1% | 88.2% | 11.8% | 1.0000 |
| 101 | no_rules | 0.05 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.9086927248041381, "redact_max_confidence": 0.5019325913150663}` | 4.0% | 5.5% | 0.5905 | 0.8400 | 0.4706 | 0.9935 | 21.4% | 70.6% | 29.4% | 0.9880 |
| 101 | no_rules | 0.10 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.9808227609396654, "redact_max_confidence": 0.5019325913150663}` | 10.0% | 8.5% | 0.5723 | 0.8150 | 0.4706 | 0.9613 | 21.4% | 76.5% | 23.5% | 0.9880 |
| 101 | direct_id | 0.05 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.9086927248041381, "redact_max_confidence": 0.5019325913150663}` | 4.0% | 5.0% | 0.6395 | 0.8550 | 0.6471 | 0.9935 | 21.4% | 82.4% | 17.6% | 0.9880 |
| 101 | direct_id | 0.10 | asymmetric_confidence | 1.0000 | `{"keep_max_confidence": 0.9808227609396654, "redact_max_confidence": 0.5019325913150663}` | 10.0% | 7.5% | 0.6218 | 0.8300 | 0.6471 | 0.9613 | 21.4% | 82.4% | 17.6% | 0.9880 |

## Aggregate Mean ± Std

| variant | target | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | review_rate | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_rules | 0.05 | 0.5359 ± 0.0485 | 0.8067 ± 0.0293 | 0.5686 ± 0.0899 | 0.9548 ± 0.0387 | 0.1310 ± 0.0899 | 0.0517 ± 0.0058 | 0.7647 ± 0.1019 | 0.2353 ± 0.1019 | 0.9960 ± 0.0070 |
| no_rules | 0.10 | 0.5401 ± 0.0286 | 0.7783 ± 0.0404 | 0.5686 ± 0.0899 | 0.9075 ± 0.0719 | 0.1905 ± 0.1091 | 0.1067 ± 0.0419 | 0.8431 ± 0.0679 | 0.1569 ± 0.0679 | 0.9960 ± 0.0070 |
| direct_id | 0.05 | 0.5566 ± 0.0719 | 0.8133 ± 0.0369 | 0.6471 ± 0.0000 | 0.9548 ± 0.0387 | 0.1310 ± 0.0899 | 0.0483 ± 0.0029 | 0.8039 ± 0.0899 | 0.1961 ± 0.0899 | 0.9960 ± 0.0070 |
| direct_id | 0.10 | 0.5611 ± 0.0526 | 0.7850 ± 0.0477 | 0.6471 ± 0.0000 | 0.9075 ± 0.0719 | 0.1905 ± 0.1091 | 0.1017 ± 0.0462 | 0.8627 ± 0.0340 | 0.1373 ± 0.0340 | 0.9960 ± 0.0070 |

## Base Classifier vs Selected Policy

| variant | target | base_macro_f1 | selected_macro_f1 | delta_macro_f1 | base_accuracy | selected_accuracy | delta_accuracy | base_redact_recall | selected_redact_recall | delta_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_rules | 0.05 | 0.4747 ± 0.0331 | 0.5359 ± 0.0485 | +0.0613 | 0.8050 ± 0.0218 | 0.8067 ± 0.0293 | +0.0017 | 0.6275 ± 0.1480 | 0.5686 ± 0.0899 | -0.0588 |
| no_rules | 0.10 | 0.4747 ± 0.0331 | 0.5401 ± 0.0286 | +0.0654 | 0.8050 ± 0.0218 | 0.7783 ± 0.0404 | -0.0267 | 0.6275 ± 0.1480 | 0.5686 ± 0.0899 | -0.0588 |
| direct_id | 0.05 | 0.4901 ± 0.0428 | 0.5566 ± 0.0719 | +0.0665 | 0.8100 ± 0.0265 | 0.8133 ± 0.0369 | +0.0033 | 0.6863 ± 0.0679 | 0.6471 ± 0.0000 | -0.0392 |
| direct_id | 0.10 | 0.4901 ± 0.0428 | 0.5611 ± 0.0526 | +0.0710 | 0.8100 ± 0.0265 | 0.7850 ± 0.0477 | -0.0250 | 0.6863 ± 0.0679 | 0.6471 ± 0.0000 | -0.0392 |

## Historical Frozen Backbone Results (single-run, uncontrolled seed — reference context only)

| backbone | selected 10% macro_f1 | selected 10% review_rate | source |
| --- | --- | --- | --- |
| ModernBERT-base v2 (historical, uncontrolled seed) | 0.5872 | 11.5% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_224940_upchieve-english-social-mixed-modernbert-v2-b4-l384/summary.json` |
| roberta-base v1 from math | 0.4882 | 8.5% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_193820_upchieve-english-social-mixed-roberta-base-v1-from-math-b4-l384/summary.json` |
| roberta-base v1 fresh | 0.5647 | 14.0% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_201224_upchieve-english-social-mixed-roberta-base-v1-b4-l384-fresh/summary.json` |
| roberta-base v2 | 0.4090 | 8.5% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_225018_upchieve-english-social-mixed-roberta-base-v2-b4-l384-fresh/summary.json` |
| distilroberta-base v1 from math | 0.5164 | 5.0% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_193849_upchieve-english-social-mixed-distilroberta-v1-from-math-b4-l384/summary.json` |
| distilroberta-base v1 fresh | 0.5120 | 14.5% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_201314_upchieve-english-social-mixed-distilroberta-v1-b4-l384-fresh/summary.json` |
| distilroberta-base v2 | 0.4708 | 9.0% | `/Users/chason/contextshift-deid/artifacts/experiments/20260314_225845_upchieve-english-social-mixed-distilroberta-v2-b4-l384-fresh/summary.json` |
