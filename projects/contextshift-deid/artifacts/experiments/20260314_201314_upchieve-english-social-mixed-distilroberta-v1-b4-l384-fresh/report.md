# Deferral Evaluation: upchieve-english-social-mixed-distilroberta-v1-b4-l384-fresh

## Base Action Metrics

- Calibration macro F1: 0.5180
- Calibration accuracy: 0.8050
- Calibration redact recall: 0.6000
- Eval macro F1: 0.4662
- Eval accuracy: 0.7800
- Eval redact recall: 0.8824
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `asymmetric_confidence`
- Parameters: `{"keep_max_confidence": 0.5854744736708858, "redact_max_confidence": 0.7658450603485107}`
- Calibration review rate: 4.5%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 5.0%
- Eval gold REVIEW coverage: 3.6%
- Eval protected REDACT rate: 88.2%
- Eval macro F1: 0.4767
- Eval ORR: 6.5%

### Target review rate <= 10.0%

- Strategy: `confidence`
- Parameters: `{"max_confidence": 0.8341508818473995}`
- Calibration review rate: 10.0%
- Calibration gold REVIEW coverage: 23.5%
- Eval review rate: 14.5%
- Eval gold REVIEW coverage: 17.9%
- Eval protected REDACT rate: 94.1%
- Eval macro F1: 0.5120
- Eval ORR: 4.5%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 6188-turn-49-tag-0-7 | REDACT | 0.997 | 0.001 | 0.996 | Gregory | student: OK ! volunteer: Gregory what you think! volunteer: change whatever |
| 7166-turn-28-tag-3-10 | REDACT | 0.997 | 0.001 | 0.996 | Brandon | volunteer: a) Brittany volunteer: b) Brandon volunteer: c) Chelsea |
| 13205-turn-6-tag-0-5 | REDACT | 0.996 | 0.002 | 0.995 | Keith | student: yes student: Keith volunteer: Reading |
| 15188-turn-0-tag-6-14 | REDACT | 0.992 | 0.005 | 0.989 | Edgewood | volunteer: Hello Edgewood! What do you need help with today? student: Hello |
| 9779-turn-64-tag-5-12 | REDACT | 0.991 | 0.004 | 0.987 | Rebecca | student: oh wait mr wickhams Amber name is Andrea? student: like Rebecca volunteer: Yeah lol |
| 14685-turn-413-tag-0-8 | REDACT | 0.989 | 0.005 | 0.984 | 555-0101 | student: 23 + 10. student: 555-0101 student: OK. |
| 411-turn-51-tag-0-6 | REDACT | 0.984 | 0.007 | 0.974 | Andrea | volunteer: breath to breathe their to they're student: Andrea volunteer: ofc |
| 1544-turn-120-tag-3-13 | KEEP | 0.982 | 0.013 | 0.976 | Haverfield | volunteer: and so it's very similar volunteer: to Haverfield. So Father just me or potter is the Mountain word for father. student: Oh |
