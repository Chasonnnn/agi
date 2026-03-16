# Deferral Evaluation: upchieve-english-social-mixed-roberta-base-v1-b4-l384-fresh

## Base Action Metrics

- Calibration macro F1: 0.4911
- Calibration accuracy: 0.7850
- Calibration redact recall: 0.5333
- Eval macro F1: 0.4320
- Eval accuracy: 0.7650
- Eval redact recall: 0.6471
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `review_probability`
- Parameters: `{"min_review_probability": 0.1100966831275072}`
- Calibration review rate: 5.0%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 6.5%
- Eval gold REVIEW coverage: 21.4%
- Eval protected REDACT rate: 64.7%
- Eval macro F1: 0.5441
- Eval ORR: 5.8%

### Target review rate <= 10.0%

- Strategy: `confidence`
- Parameters: `{"max_confidence": 0.7144539249769758}`
- Calibration review rate: 10.0%
- Calibration gold REVIEW coverage: 29.4%
- Eval review rate: 14.0%
- Eval gold REVIEW coverage: 39.3%
- Eval protected REDACT rate: 70.6%
- Eval macro F1: 0.5647
- Eval ORR: 3.2%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 6188-turn-49-tag-0-7 | REDACT | 0.997 | 0.002 | 0.997 | Gregory | student: OK ! volunteer: Gregory what you think! volunteer: change whatever |
| 15188-turn-0-tag-6-14 | REDACT | 0.997 | 0.002 | 0.995 | Edgewood | volunteer: Hello Edgewood! What do you need help with today? student: Hello |
| 14685-turn-413-tag-0-8 | REDACT | 0.995 | 0.003 | 0.993 | 555-0101 | student: 23 + 10. student: 555-0101 student: OK. |
| 10968-turn-42-tag-86-107 | REDACT | 0.995 | 0.003 | 0.993 | Westfield High School | volunteer: I think you should and we can remove less relevant stuff once you're done writing volunteer: Like depending on how you write it, it might be more relevant and impressive |
| 16375-turn-27-tag-96-121 | REDACT | 0.976 | 0.011 | 0.964 | Harborfield Middle School | student: that was supposed to be the opening sentence volunteer: ok that is good, i also have As the cost of higher education skyrockets far beyond wage growth, Harborfield Middle  |
| 13205-turn-6-tag-0-5 | REDACT | 0.973 | 0.013 | 0.959 | Keith | student: yes student: Keith volunteer: Reading |
| 1544-turn-120-tag-3-13 | KEEP | 0.964 | 0.019 | 0.946 | Haverfield | volunteer: and so it's very similar volunteer: to Haverfield. So Father just me or potter is the Mountain word for father. student: Oh |
| 2878-turn-74-tag-0-9 | REDACT | 0.949 | 0.021 | 0.919 | Midlander | volunteer: Oh student: Midlander student: desk. Did you oh no, Clearview High School here. I actually wrote my first poem, poem in Mr. Sarah's class right across the hall. |
