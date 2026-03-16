# Deferral Evaluation: upchieve-english-social-mixed-distilroberta-v1-from-math-b4-l384

## Base Action Metrics

- Calibration macro F1: 0.5119
- Calibration accuracy: 0.8000
- Calibration redact recall: 0.6222
- Eval macro F1: 0.4610
- Eval accuracy: 0.7950
- Eval redact recall: 0.7647
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `asymmetric_margin`
- Parameters: `{"keep_max_redact_keep_margin": 0.8855345330195273, "redact_max_redact_keep_margin": 0.7946927178880038}`
- Calibration review rate: 5.0%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 2.5%
- Eval gold REVIEW coverage: 3.6%
- Eval protected REDACT rate: 76.5%
- Eval macro F1: 0.4751
- Eval ORR: 5.2%

### Target review rate <= 10.0%

- Strategy: `asymmetric_margin`
- Parameters: `{"keep_max_redact_keep_margin": 0.9890036360066315, "redact_max_redact_keep_margin": 0.6247965297548024}`
- Calibration review rate: 9.0%
- Calibration gold REVIEW coverage: 23.5%
- Eval review rate: 5.0%
- Eval gold REVIEW coverage: 10.7%
- Eval protected REDACT rate: 82.4%
- Eval macro F1: 0.5164
- Eval ORR: 5.2%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 13205-turn-6-tag-0-5 | REDACT | 0.999 | 0.001 | 0.999 | Keith | student: yes student: Keith volunteer: Reading |
| 411-turn-51-tag-0-6 | REDACT | 0.999 | 0.001 | 0.999 | Andrea | volunteer: breath to breathe their to they're student: Andrea volunteer: ofc |
| 28-turn-71-tag-0-6 | REDACT | 0.999 | 0.001 | 0.999 | Kelsey | student: what is the word student: Kelsey student: what does it mean |
| 7166-turn-28-tag-3-10 | REDACT | 0.999 | 0.001 | 0.998 | Brandon | volunteer: a) Brittany volunteer: b) Brandon volunteer: c) Chelsea |
| 9779-turn-64-tag-5-12 | REDACT | 0.998 | 0.001 | 0.998 | Rebecca | student: oh wait mr wickhams Amber name is Andrea? student: like Rebecca volunteer: Yeah lol |
| 6410-turn-54-tag-34-42 | REDACT | 0.998 | 0.001 | 0.998 | Courtney | volunteer: :) * student: Also do you know the feature that Courtney has where you can like text in the old session with the coach? volunteer: yes, all session history is available  |
| 18019-turn-270-tag-34-41 | KEEP | 0.998 | 0.001 | 0.998 | Shelton | volunteer: oh God, I'm not really volunteer: I don't know what curriculum, um, Shelton has, so this Valley, you, you might volunteer: ah, try like asking your parents |
| 2831-turn-122-tag-8-17 | KEEP | 0.998 | 0.001 | 0.998 | Glenfield | volunteer: 8th, OK student: uh from Glenfield, it is only 90 miles about 145 kilometers to the coast of Oakmont. student: so Summitville is like there and then. |
