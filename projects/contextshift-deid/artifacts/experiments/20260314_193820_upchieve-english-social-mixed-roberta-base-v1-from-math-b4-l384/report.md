# Deferral Evaluation: upchieve-english-social-mixed-roberta-base-v1-from-math-b4-l384

## Base Action Metrics

- Calibration macro F1: 0.5065
- Calibration accuracy: 0.7800
- Calibration redact recall: 0.7556
- Eval macro F1: 0.4282
- Eval accuracy: 0.7350
- Eval redact recall: 0.8824
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `asymmetric_margin`
- Parameters: `{"keep_max_redact_keep_margin": 0.007827163209150001, "redact_max_redact_keep_margin": 0.2224978374315677}`
- Calibration review rate: 4.5%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 3.0%
- Eval gold REVIEW coverage: 10.7%
- Eval protected REDACT rate: 88.2%
- Eval macro F1: 0.4847
- Eval ORR: 14.2%

### Target review rate <= 10.0%

- Strategy: `margin`
- Parameters: `{"max_redact_keep_margin": 0.4770803586888527}`
- Calibration review rate: 10.0%
- Calibration gold REVIEW coverage: 17.6%
- Eval review rate: 8.5%
- Eval gold REVIEW coverage: 14.3%
- Eval protected REDACT rate: 94.1%
- Eval macro F1: 0.4882
- Eval ORR: 11.6%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 13205-turn-6-tag-0-5 | REDACT | 0.998 | 0.001 | 0.998 | Keith | student: yes student: Keith volunteer: Reading |
| 6188-turn-49-tag-0-7 | REDACT | 0.998 | 0.002 | 0.997 | Gregory | student: OK ! volunteer: Gregory what you think! volunteer: change whatever |
| 9779-turn-64-tag-5-12 | REDACT | 0.998 | 0.002 | 0.997 | Rebecca | student: oh wait mr wickhams Amber name is Andrea? student: like Rebecca volunteer: Yeah lol |
| 411-turn-51-tag-0-6 | REDACT | 0.998 | 0.002 | 0.997 | Andrea | volunteer: breath to breathe their to they're student: Andrea volunteer: ofc |
| 14685-turn-413-tag-0-8 | REDACT | 0.997 | 0.002 | 0.996 | 555-0101 | student: 23 + 10. student: 555-0101 student: OK. |
| 10968-turn-42-tag-86-107 | REDACT | 0.994 | 0.004 | 0.993 | Westfield High School | volunteer: I think you should and we can remove less relevant stuff once you're done writing volunteer: Like depending on how you write it, it might be more relevant and impressive |
| 16375-turn-27-tag-96-121 | REDACT | 0.992 | 0.006 | 0.990 | Harborfield Middle School | student: that was supposed to be the opening sentence volunteer: ok that is good, i also have As the cost of higher education skyrockets far beyond wage growth, Harborfield Middle  |
| 6410-turn-54-tag-34-42 | REDACT | 0.991 | 0.005 | 0.988 | Courtney | volunteer: :) * student: Also do you know the feature that Courtney has where you can like text in the old session with the coach? volunteer: yes, all session history is available  |
