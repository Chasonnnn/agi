# Math Ground-Truth Benchmark Migration

Date: 2026-03-21

## Summary

The math candidate benchmark now uses the updated ground-truth JSON directories directly:

- `/Users/chason/agi/ground-truth/ground-truth-math-upchieve-1000`
- `/Users/chason/agi/ground-truth/ground-truth-math-saga-27`

Both corpora now share the same raw shape:

- one JSON file per dialogue
- `transcript` as a string
- char-offset annotations from `pii_occurrences` or `spans`

The candidate JSONL schema did not change at the top level. The benchmark now records raw annotation provenance inside `metadata.gold_spans[]`.

## Canonical Math Labels

Observed canonical labels in the rebuilt benchmark:

- `NAME`
- `ADDRESS`
- `URL`
- `SCHOOL`
- `AGE`
- `IDENTIFYING_NUMBER`
- `TUTOR_PROVIDER`
- `DATE`
- `PHONE_NUMBER`
- `IP_ADDRESS`

Compatibility normalization remains locked as:

- `PERSON`, `NAME` -> `NAME`
- `LOCATION` -> `ADDRESS`
- `DATE` -> `DATE`
- `PHONE`, `PHONE_NUMBER` -> `PHONE_NUMBER`
- `EMAIL_ADDRESS`, `EMAIL` -> `EMAIL`
- `US_SSN`, `SSN` -> `SSN`
- `US_BANK_NUMBER` -> `ACCOUNT_NUMBER`
- `IP_ADDRESS` -> `IP_ADDRESS`
- `URL` -> `URL`
- `AGE` -> `AGE`
- `SCHOOL` -> `SCHOOL`
- `US_DRIVER_LICENSE`, `US_PASSPORT`, `MISC_ID`, `SOCIAL_HANDLE` -> `IDENTIFYING_NUMBER`
- `TUTOR_PROVIDER` -> `TUTOR_PROVIDER`

Legacy action-only labels such as `COURSE`, `GRADE_LEVEL`, and `NRP` are not remapped into the current math benchmark.

## Rebuilt Benchmark

Generated files:

- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/upchieve_math_ground_truth_train.jsonl`
- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/upchieve_math_ground_truth_dev.jsonl`
- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/upchieve_math_ground_truth_test.jsonl`
- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/saga27_math_ground_truth_test.jsonl`
- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/ground_truth_candidate_benchmark_summary.json`
- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/ground_truth_candidate_benchmark_comparison.json`
- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/ground_truth_candidate_benchmark_comparison.md`

Row counts after rebuild:

| Split | Rows | Positive rows |
|---|---:|---:|
| UpChieve train | `79,502` | `1,182` |
| UpChieve dev | `18,413` | `218` |
| UpChieve test | `17,699` | `234` |
| Saga 27 holdout | `13,684` | `575` |

Validation checks:

- imported `1000` UpChieve files and `27` Saga files
- no UpChieve dialogue leakage across `train / dev / test`
- every span resolved successfully
- all observed resolutions were `exact`

Offset resolution counts:

- UpChieve: `1744 exact`
- Saga 27: `732 exact`

## Benchmark Drift vs 2026-03-18

Reference artifact:

- `/Users/chason/agi/projects/contextshift-deid/data/processed/candidate/ground_truth_candidate_benchmark_comparison.md`

High-level drift after normalizing the previous benchmark into the current taxonomy:

| Source | Previous spans | Current spans | Delta |
|---|---:|---:|---:|
| UpChieve | `1988` | `1744` | `-244` |
| Saga 27 | `733` | `732` | `-1` |

| Source | Previous positive rows | Current positive rows | Delta |
|---|---:|---:|---:|
| UpChieve | `1846` | `1634` | `-212` |
| Saga 27 | `576` | `575` | `-1` |

UpChieve label-count drift after normalization:

- removed: `COURSE 40`, `GRADE_LEVEL 103`, `NRP 25`
- added: `TUTOR_PROVIDER 3`
- reduced but retained: `ADDRESS -59`, `URL -38`, `SCHOOL -2`, `DATE -1`
- increased or renamed into surviving labels: `NAME +18`, `AGE +1`, `IDENTIFYING_NUMBER +1`, `PHONE_NUMBER +1`

Saga drift is minimal:

- removed: `ADDRESS 3`
- added: `TUTOR_PROVIDER 2`

The benchmark shift is therefore mostly an annotation-scheme change in UpChieve rather than an importer artifact.

## Incumbent Baseline on the Rebuilt Benchmark

The incumbent checkpoint remains:

- `/Users/chason/agi/projects/contextshift-deid/runs/candidate_math_distilbert_rebuilt`

Rebuilt-benchmark evaluation artifacts:

- `/Users/chason/agi/projects/contextshift-deid/artifacts/predictions/incumbent_upchieve_ground_truth_dev_predictions.summary.json`
- `/Users/chason/agi/projects/contextshift-deid/artifacts/predictions/incumbent_upchieve_ground_truth_test_predictions.summary.json`
- `/Users/chason/agi/projects/contextshift-deid/artifacts/predictions/incumbent_saga27_ground_truth_test_predictions.summary.json`

Side-by-side with the 2026-03-18 benchmark note:

| Benchmark | UpChieve dev recall | UpChieve dev volume | UpChieve test recall | Saga recall |
|---|---:|---:|---:|---:|
| 2026-03-18 benchmark | `0.9442` | `1.2714` | `0.9474` | `0.8458` |
| 2026-03-21 rebuilt benchmark | `0.9658` | `1.5043` | `0.9724` | `0.8470` |

Additional rebuilt-benchmark incumbent metrics:

- UpChieve dev F1: `0.7483`
- UpChieve test F1: `0.8316`
- Saga F1: `0.8115`
- UpChieve dev rare-type mean recall: `0.7862`
- UpChieve test rare-type mean recall: `0.7500`
- Saga rare-type mean recall: `0.6333`

Interpretation:

- raw recall is slightly higher on the rebuilt benchmark
- candidate volume is materially higher on UpChieve dev
- Saga is effectively flat

That is consistent with a benchmark-ontology shift rather than an obvious model improvement.

## Locked Checkpoint Benchmark

Official benchmark artifacts:

- `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/candidate_checkpoint_benchmark/20260321_224321_candidate-checkpoint-benchmark-ground-truth-20260321/summary.json`
- `/Users/chason/agi/projects/contextshift-deid/artifacts/experiments/candidate_checkpoint_benchmark/20260321_224321_candidate-checkpoint-benchmark-ground-truth-20260321/report.md`

Checkpoint benchmark result:

| Model | Eligible | Dev recall | Dev volume | UpChieve test recall | Saga recall | Math recall | Math drop | Final min recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DistilBERT incumbent | `yes` | `0.9658` | `1.5043` | `0.9724` | `0.8470` | `0.9317` | `0.0000` | `0.8470` |
| RoBERTa locked | `no` | `0.9701` | `1.4060` | `0.9646` | `0.8566` | `0.9171` | `0.0146` | `0.8566` |
| ModernBERT 1-epoch | `no` | `0.9615` | `1.2735` | `0.9488` | `0.8456` | `0.8878` | `0.0439` | `0.8456` |
| DeBERTa-v3-small | `no` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.9317` | `0.0000` |

Winner:

- label: `distilbert-incumbent`
- promoted: `false`
- final minimum recall: `0.8470`
- UpChieve dev volume multiplier: `1.5043`

The rebuilt benchmark still prefers the incumbent under the existing recall-first guardrails.

## Remaining Run

After rebuilding the benchmark and rerunning the locked checkpoints, the remaining execution step is:

1. rerun the candidate model-selection flow with the existing recall-first guards unchanged

That output should be appended to this migration trail once the longer-running training run finishes.
