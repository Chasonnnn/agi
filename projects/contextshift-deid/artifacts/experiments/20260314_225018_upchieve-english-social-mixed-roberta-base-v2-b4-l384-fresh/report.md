# Deferral Evaluation: upchieve-english-social-mixed-roberta-base-v2-b4-l384-fresh

## Base Action Metrics

- Calibration macro F1: 0.4533
- Calibration accuracy: 0.7050
- Calibration redact recall: 0.6889
- Eval macro F1: 0.3858
- Eval accuracy: 0.6750
- Eval redact recall: 0.7647
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `review_probability`
- Parameters: `{"min_review_probability": 0.1613907665014267}`
- Calibration review rate: 5.0%
- Calibration gold REVIEW coverage: 5.9%
- Eval review rate: 5.5%
- Eval gold REVIEW coverage: 7.1%
- Eval protected REDACT rate: 76.5%
- Eval macro F1: 0.4088
- Eval ORR: 18.7%

### Target review rate <= 10.0%

- Strategy: `asymmetric_confidence`
- Parameters: `{"keep_max_confidence": 0.8422136997010611, "redact_max_confidence": 0.6727771959536432}`
- Calibration review rate: 10.0%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 8.5%
- Eval gold REVIEW coverage: 7.1%
- Eval protected REDACT rate: 82.4%
- Eval macro F1: 0.4090
- Eval ORR: 16.1%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 10968-turn-42-tag-86-107 | REDACT | 0.996 | 0.003 | 0.995 | Westfield High School | volunteer: I think you should and we can remove less relevant stuff once you're done writing volunteer: Like depending on how you write it, it might be more relevant and impressive |
| 1494-turn-7-tag-0-7 | REDACT | 0.996 | 0.003 | 0.994 | Heather | student: On October 22d 2025,Luke Shelton reprt for the NY times,wrote It's part of the Islander lore that Travis rebuffed proposas for a presidentals pallace.e believed a fledging |
| 16375-turn-27-tag-96-121 | REDACT | 0.995 | 0.004 | 0.994 | Harborfield Middle School | student: that was supposed to be the opening sentence volunteer: ok that is good, i also have As the cost of higher education skyrockets far beyond wage growth, Harborfield Middle  |
| 2255-turn-465-tag-0-23 | REDACT | 0.994 | 0.005 | 0.992 | Riverside Middle School | student: Yeah volunteer: Riverside Middle School, like maybe saying some research organization conducted a survey with teenagers and more teenagers said that with a curfew, they sp |
| 6410-turn-54-tag-34-42 | REDACT | 0.993 | 0.005 | 0.991 | Courtney | volunteer: :) * student: Also do you know the feature that Courtney has where you can like text in the old session with the coach? volunteer: yes, all session history is available  |
| 1544-turn-120-tag-3-13 | KEEP | 0.992 | 0.006 | 0.991 | Haverfield | volunteer: and so it's very similar volunteer: to Haverfield. So Father just me or potter is the Mountain word for father. student: Oh |
| 411-turn-51-tag-0-6 | REDACT | 0.990 | 0.007 | 0.988 | Andrea | volunteer: breath to breathe their to they're student: Andrea volunteer: ofc |
| 2831-turn-122-tag-8-17 | KEEP | 0.990 | 0.007 | 0.988 | Glenfield | volunteer: 8th, OK student: uh from Glenfield, it is only 90 miles about 145 kilometers to the coast of Oakmont. student: so Summitville is like there and then. |
