# Candidate Model Selection on Math Ground Truth

Date: 2026-03-18

## Summary

We built the first real candidate-selection benchmark from two uploaded math-domain ground-truth corpora:

- UpChieve 1000 transcripts, normalized to turn-level candidate rows and split at the dialogue level into `700 train / 150 dev / 150 test`
- Saga 27 transcripts, normalized to segment-level candidate rows and kept as an untouched external holdout

The benchmark uses the existing candidate JSONL schema and canonicalizes label names across sources so Saga `NAME` is treated as `PERSON` in the audit reports.

## Benchmark Datasets

- UpChieve train: `78,644` rows, `1,340` positive rows
- UpChieve dev: `19,009` rows, `252` positive rows
- UpChieve test: `17,961` rows, `254` positive rows
- Saga 27 holdout: `13,685` rows, `576` positive rows

## Selection Rule

Winner guardrails:

- highest `min(UpChieve test recall, Saga recall)`
- UpChieve dev `candidate_volume_multiplier <= 3.0`
- legacy math recall drop `<= 0.01` versus the incumbent

## Results

| Model | UpChieve dev recall | UpChieve dev volume | UpChieve test recall | Saga recall | Imported math recall | Eligible |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT incumbent | `0.9442` | `1.2714` | `0.9474` | `0.8458` | `0.9317` | `yes` |
| RoBERTa locked math | `0.9294` | `1.2565` | `0.9549` | `0.8568` | `0.9171` | `no` |
| ModernBERT 1-epoch math | `0.9033` | `1.1561` | `0.9211` | `0.8458` | `0.8878` | `no` |
| DeBERTa-v3-small math | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `no` |

Raw held-out minimum recall was highest for the locked RoBERTa checkpoint at `0.8568`, but it failed the legacy math recall guard with a `0.0146` absolute drop versus the incumbent. Under the preregistered rule, the incumbent DistilBERT checkpoint remains the winner.

## Decision

Keep `runs/candidate_math_distilbert_rebuilt` as the candidate backbone for the next fine-tuning sprint.

Reason:

- it is the best fully eligible model under the gold benchmark
- it preserves legacy math recall
- it stays well below the proposal-volume cap
- its UpChieve rare-type recall is stronger than the other eligible options because the other candidates are not actually eligible

## Interpretation

The benchmark says two things at once:

1. DistilBERT is the safest backbone to continue fine-tuning right now.
2. RoBERTa is the strongest raw recall challenger and is worth revisiting in a fresh retraining sprint if we want to challenge the incumbent while explicitly re-optimizing for the math guard.

ModernBERT should not be ruled out from the candidate stage yet, because the available checkpoint is only a `1`-epoch math dev artifact and is not a fair tuned comparison. DeBERTa-v3-small is not competitive in its current form.

## Next Step

Use the new gold benchmark to run a bounded recall-first fine-tuning sprint on the DistilBERT incumbent:

- mixed math ground-truth training on UpChieve train plus existing imported math train
- selection by recall, not F1
- report UpChieve test, Saga holdout, and imported math test together
- preserve the same volume and math guards used here
