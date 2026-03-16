# ModernBERT-base Seed Run: 42

- Training dir: `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/modernbert_recipe_tuning/20260316_003740_upchieve-modernbert-v2-seed-suite-seed-42/training/model`

## Selected 10% Results

| variant | strategy | temperature | parameters | calibration review | eval review | macro_f1 | accuracy | redact_recall | curricular_accuracy | gold_review_coverage | protected_redact_rate | redact_leak_rate | legacy_math_redact_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_rules | review_probability | 2.5785 | `{"min_review_probability": 0.24494085855420053}` | 7.5% | 8.5% | 0.6158 | 0.8250 | 0.5882 | 0.9419 | 32.1% | 76.5% | 23.5% | 1.0000 |
| direct_id | review_probability | 2.6611 | `{"min_review_probability": 0.24794279108427783}` | 7.5% | 8.0% | 0.6314 | 0.8300 | 0.6471 | 0.9419 | 32.1% | 76.5% | 23.5% | 1.0000 |

## Direct-ID Overrides

| split | override_count | by_reason |
| --- | --- | --- |
| dev | 2 | `{"URL": 2}` |
| test | 5 | `{"IP_ADDRESS": 1, "URL": 2, "US_BANK_NUMBER": 1, "US_PASSPORT": 1}` |
| math | 13 | `{"URL": 13}` |

## Direct-ID Test Spot-Check

| id | span_text | entity_type | gold_action | base_predicted_action | patched_predicted_action | reason | source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1273-turn-149-tag-182-207 | https://example.com/page1 | URL | REDACT | REDACT | REDACT | URL | entity_type |
| 3517-turn-13-tag-0-20 | [REDACTED-BANK-0001] | US_BANK_NUMBER | REDACT | REDACT | REDACT | US_BANK_NUMBER | entity_type |
| 4088-turn-51-tag-0-24 | [REDACTED-PASSPORT-0001] | US_PASSPORT | REDACT | REDACT | REDACT | US_PASSPORT | entity_type |
| 13766-turn-35-tag-29-54 | https://example.com/page1 | URL | REDACT | REDACT | REDACT | URL | entity_type |
| 13543-turn-180-tag-0-9 | 192.0.2.1 | IP_ADDRESS | REDACT | REDACT | REDACT | IP_ADDRESS | entity_type |
