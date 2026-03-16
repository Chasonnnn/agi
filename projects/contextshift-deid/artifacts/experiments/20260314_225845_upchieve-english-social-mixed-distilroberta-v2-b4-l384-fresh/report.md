# Deferral Evaluation: upchieve-english-social-mixed-distilroberta-v2-b4-l384-fresh

## Base Action Metrics

- Calibration macro F1: 0.4750
- Calibration accuracy: 0.7400
- Calibration redact recall: 0.6889
- Eval macro F1: 0.4046
- Eval accuracy: 0.7100
- Eval redact recall: 0.7647
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `composite`
- Parameters: `{"max_confidence": 0.5531038518301282, "max_redact_keep_margin": 0.18407824224058744, "min_entropy": 0.8942117270184011, "min_review_probability": 0.09009910166854272}`
- Calibration review rate: 5.0%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 6.5%
- Eval gold REVIEW coverage: 10.7%
- Eval protected REDACT rate: 88.2%
- Eval macro F1: 0.4537
- Eval ORR: 14.2%

### Target review rate <= 10.0%

- Strategy: `review_probability`
- Parameters: `{"min_review_probability": 0.06259034480764948}`
- Calibration review rate: 10.0%
- Calibration gold REVIEW coverage: 23.5%
- Eval review rate: 9.0%
- Eval gold REVIEW coverage: 14.3%
- Eval protected REDACT rate: 94.1%
- Eval macro F1: 0.4708
- Eval ORR: 12.3%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 13205-turn-6-tag-0-5 | REDACT | 0.998 | 0.001 | 0.998 | Keith | student: yes student: Keith volunteer: Reading |
| 7166-turn-28-tag-3-10 | REDACT | 0.998 | 0.002 | 0.997 | Brandon | volunteer: a) Brittany volunteer: b) Brandon volunteer: c) Chelsea |
| 411-turn-51-tag-0-6 | REDACT | 0.997 | 0.002 | 0.997 | Andrea | volunteer: breath to breathe their to they're student: Andrea volunteer: ofc |
| 15188-turn-0-tag-6-14 | REDACT | 0.997 | 0.002 | 0.997 | Edgewood | volunteer: Hello Edgewood! What do you need help with today? student: Hello |
| 14685-turn-413-tag-0-8 | REDACT | 0.997 | 0.002 | 0.996 | 555-0101 | student: 23 + 10. student: 555-0101 student: OK. |
| 9779-turn-64-tag-5-12 | REDACT | 0.997 | 0.002 | 0.996 | Rebecca | student: oh wait mr wickhams Amber name is Andrea? student: like Rebecca volunteer: Yeah lol |
| 6188-turn-49-tag-0-7 | REDACT | 0.997 | 0.002 | 0.996 | Gregory | student: OK ! volunteer: Gregory what you think! volunteer: change whatever |
| 6410-turn-54-tag-34-42 | REDACT | 0.996 | 0.003 | 0.995 | Courtney | volunteer: :) * student: Also do you know the feature that Courtney has where you can like text in the old session with the coach? volunteer: yes, all session history is available  |
