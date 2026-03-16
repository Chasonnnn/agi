# Deferral Evaluation: upchieve-english-social-mixed-modernbert-v2-b4-l384

## Base Action Metrics

- Calibration macro F1: 0.5098
- Calibration accuracy: 0.8000
- Calibration redact recall: 0.5778
- Eval macro F1: 0.4892
- Eval accuracy: 0.8150
- Eval redact recall: 0.7647
- Temperature: 1.0000

## Selected Policies

### Target review rate <= 5.0%

- Strategy: `asymmetric_confidence`
- Parameters: `{"keep_max_confidence": 0.7771726250648499, "redact_max_confidence": 0.5199789559834884}`
- Calibration review rate: 5.0%
- Calibration gold REVIEW coverage: 5.9%
- Eval review rate: 7.5%
- Eval gold REVIEW coverage: 17.9%
- Eval protected REDACT rate: 82.4%
- Eval macro F1: 0.5703
- Eval ORR: 2.6%

### Target review rate <= 10.0%

- Strategy: `asymmetric_confidence`
- Parameters: `{"keep_max_confidence": 0.9126039778516977, "redact_max_confidence": 0.5867771637464426}`
- Calibration review rate: 9.5%
- Calibration gold REVIEW coverage: 11.8%
- Eval review rate: 11.5%
- Eval gold REVIEW coverage: 28.6%
- Eval protected REDACT rate: 88.2%
- Eval macro F1: 0.5872
- Eval ORR: 2.6%

## High-Confidence Gold REVIEW Misses

| id | base | conf | p(review) | margin | span | context |
| --- | --- | --- | --- | --- | --- | --- |
| 13205-turn-6-tag-0-5 | REDACT | 0.998 | 0.001 | 0.998 | Keith | student: yes student: Keith volunteer: Reading |
| 10968-turn-42-tag-86-107 | REDACT | 0.996 | 0.004 | 0.996 | Westfield High School | volunteer: I think you should and we can remove less relevant stuff once you're done writing volunteer: Like depending on how you write it, it might be more relevant and impressive |
| 18019-turn-270-tag-34-41 | KEEP | 0.995 | 0.003 | 0.994 | Shelton | volunteer: oh God, I'm not really volunteer: I don't know what curriculum, um, Shelton has, so this Valley, you, you might volunteer: ah, try like asking your parents |
| 2831-turn-122-tag-8-17 | KEEP | 0.995 | 0.004 | 0.994 | Glenfield | volunteer: 8th, OK student: uh from Glenfield, it is only 90 miles about 145 kilometers to the coast of Oakmont. student: so Summitville is like there and then. |
| 19272-turn-18-tag-12-20 | KEEP | 0.992 | 0.007 | 0.990 | Thornton | volunteer: If you can answer these questions, it will lead us to the answers to what the main question is asking. student: Bayview and Thornton student: cash and resources |
| 6188-turn-49-tag-0-7 | REDACT | 0.986 | 0.011 | 0.984 | Gregory | student: OK ! volunteer: Gregory what you think! volunteer: change whatever |
| 2255-turn-465-tag-0-23 | REDACT | 0.986 | 0.012 | 0.985 | Riverside Middle School | student: Yeah volunteer: Riverside Middle School, like maybe saying some research organization conducted a survey with teenagers and more teenagers said that with a curfew, they sp |
| 15188-turn-0-tag-6-14 | REDACT | 0.985 | 0.011 | 0.982 | Edgewood | volunteer: Hello Edgewood! What do you need help with today? student: Hello |
