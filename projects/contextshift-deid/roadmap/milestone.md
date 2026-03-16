# Milestones

This document defines the execution roadmap for `contextshift-deid`.

Each milestone includes:

- **objective**
- **work items**
- **deliverables**
- **evaluation criteria**
- **exit condition**

The goal is to move from idea to benchmark to baseline to paper-ready system without losing clarity on what counts as “done.”

## Milestone 0: Research Freeze

### Objective

Freeze the project scope so the repo and benchmark are aligned to one paper claim.

### Work Items

- lock the one-sentence paper claim
- lock the 3 main hypotheses
- decide the target subjects for v1
- decide whether the first submission target is benchmark-heavy or method-heavy
- define the main headline metrics

### Deliverables

- a written one-sentence claim in the paper notes
- 3 explicit hypotheses
- a short benchmark scope note
- a metric shortlist for the main table

### Evaluation Criteria

- the claim can be explained in 2-3 sentences without talking about implementation details
- the hypotheses are testable with the planned data
- the benchmark scope is small enough to finish
- the main metrics reflect both privacy and utility

### Exit Condition

You can answer:

- what the paper is about
- what the main dataset contexts are
- what the headline number will be

without changing the answer day to day.

## Milestone 1: Codebook v2 and Annotation Policy

### Objective

Define the new label/action policy for context-aware de-identification.

### Work Items

- create the composite label inventory
- define `REDACT`, `KEEP`, and `REVIEW`
- specify how curricular entities differ from private entities
- write 30+ canonical examples across math, history, and literature
- define adjudication rules for ambiguous cases

### Deliverables

- codebook v2
- example bank
- adjudication checklist

### Evaluation Criteria

- examples cover both under-redaction and over-redaction risks
- ambiguous person/place cases are explicitly addressed
- the codebook makes anchor text usage clear
- annotators can independently apply the labels to unseen examples

### Exit Condition

A second annotator can read the codebook and label pilot examples without needing oral clarification for every edge case.

## Milestone 2: Pilot Benchmark and Calibration

### Objective

Validate that the task and label policy are usable before scaling annotation.

### Work Items

- collect pilot data across required subjects
- normalize dialogue structure where needed
- attach anchor text and relevant metadata
- run 2-3 calibration rounds
- adjudicate disagreements
- revise the codebook after each round

### Deliverables

- pilot benchmark subset
- calibration report
- updated codebook v2
- agreement summary

### Evaluation Criteria

- direct-PII span agreement is strong enough to trust scale-up
- action-label disagreement is concentrated in meaningful ambiguous cases
- history/literature examples clearly expose over-redaction risk
- there is enough evidence that the task is materially harder than the math-only setting

### Exit Condition

Agreement is stable enough that scaling annotation will produce trustworthy dev/test splits.

## Milestone 3: Benchmark v1

### Objective

Build the first full benchmark version with stable train/dev/test splits.

### Work Items

- finalize subject coverage for v1
- sample the full benchmark using a mix of hard and ordinary cases
- annotate and adjudicate dev/test
- export benchmark JSONL files for candidate and action stages
- add benchmark documentation and data cards

### Deliverables

- `data/processed/candidate/{train,dev,test}.jsonl`
- `data/processed/action/{train,dev,test}.jsonl`
- benchmark composition summary
- benchmark documentation

### Evaluation Criteria

- all required files validate with `prepare.py`
- the subject distribution is balanced enough for per-context evaluation
- hard ambiguity cases are present in meaningful numbers
- dev and test are adjudicated and stable

### Exit Condition

The benchmark can support reproducible baseline runs and paper tables.

## Milestone 4: Baseline Systems

### Objective

Establish baseline performance and prove that context shift is a real failure mode.

### Work Items

- run regex-only and structured-rule baselines
- run Presidio/spaCy-style baselines
- run the current math-oriented system on the new benchmark
- add at least one stronger upper-bound baseline

### Deliverables

- baseline prediction files
- baseline metrics table
- error buckets by context

### Evaluation Criteria

- at least one baseline clearly over-redacts curricular entities
- at least one baseline clearly underperforms on worst-context privacy recall
- the benchmark shows a measurable math-to-humanities transfer failure
- baseline outputs are reproducible from scripts and configs

### Exit Condition

You can already tell a convincing “context shift breaks current systems” story from the baseline results.

## Milestone 5: Candidate Detector v1

### Objective

Train a recall-first local candidate detector for suspicious spans.

### Work Items

- finalize candidate labels
- train the first token-level encoder baseline
- compare context windows and truncation strategies
- examine hard-negative failure modes
- export candidate predictions for the action stage

### Deliverables

- trained candidate detector checkpoint
- candidate prediction files on dev/test
- candidate metrics report

### Evaluation Criteria

- candidate recall is high enough that the action model is not starved
- worst-context candidate recall is reported, not just average recall
- common over-generation error types are documented
- the detector is fast enough to be practical in the full cascade

### Exit Condition

The candidate detector can serve as a stable first-pass span proposer for later experiments.

## Milestone 6: Action Model v1

### Objective

Train the first action model for `REDACT / KEEP / REVIEW`.

### Work Items

- train the baseline action classifier
- test anchor-text formatting
- test subject priors as features
- compare with and without `REVIEW`
- analyze `KEEP` failures in history/literature

### Deliverables

- trained action model checkpoint
- dev/test action predictions
- action metrics report
- confusion analysis

### Evaluation Criteria

- worst-context `REDACT` recall improves over simpler baselines
- CERR and ORR are both reported and interpretable
- the model does not collapse into overusing `REVIEW`
- the paper claim about semantic role shift is visible in the errors

### Exit Condition

The first real subject-aware action baseline exists and is measurable against prior systems.

## Milestone 7: Utility Evaluation

### Objective

Make utility preservation part of the benchmark story rather than an afterthought.

### Work Items

- define two downstream utility tasks
- evaluate raw vs de-identified text on those tasks
- compute utility drop for all major baselines and models
- connect utility loss to specific over-redaction patterns

### Deliverables

- utility task definitions
- utility evaluation scripts
- utility results table

### Evaluation Criteria

- utility tasks are real enough to matter and reproducible enough to report
- at least one system shows the privacy/utility tradeoff clearly
- ORR correlates with measurable utility loss in a useful way
- utility metrics are ready for the main results section

### Exit Condition

The project can support the claim that privacy and utility must be evaluated jointly.

## Milestone 8: Cascade v1

### Objective

Assemble the first full local-first cascade.

### Work Items

- integrate direct-ID rules
- integrate candidate detector and action model
- add review/escalation routing
- define placeholder replacement behavior
- run end-to-end evaluation

### Deliverables

- first end-to-end cascade
- integrated prediction pipeline
- end-to-end metrics table

### Evaluation Criteria

- the cascade beats the best non-cascade baseline on worst-context `REDACT` recall or ORR
- end-to-end predictions are reproducible
- review routing is controlled and not excessive
- cost and latency are still practical

### Exit Condition

There is a coherent full system, not just independent component experiments.

## Milestone 9: Autoresearch Loop

### Objective

Use the repo as a controlled experiment engine rather than hand-tuning every variant.

### Work Items

- define the search space
- define the primary dev objective
- log all runs in `results.tsv`
- add nightly or batch experiment scripts
- keep only changes that improve the target objective

### Deliverables

- repeatable experiment loop
- populated `results.tsv`
- Pareto-style comparison of privacy, utility, and cost

### Evaluation Criteria

- experiments are comparable across runs
- the search space is constrained and meaningful
- improvements are measured against the same fixed dev benchmark
- the loop finds at least one nontrivial configuration improvement

### Exit Condition

The repo can support sustained iteration without losing methodological discipline.

## Milestone 10: Robustness and Bias

### Objective

Stress test the system on adversarial and fairness-sensitive cases.

### Work Items

- build an adversarial evaluation suite
- add bypass-like structured examples
- add culturally diverse matched templates
- measure review behavior across groups

### Deliverables

- robustness suite
- bias/fairness evaluation summary

### Evaluation Criteria

- the system does not fail silently on trivial adversarial variations
- failure patterns can be categorized and discussed honestly
- fairness slices are measurable and not ignored

### Exit Condition

The paper can include a credible robustness and limitations section.

## Milestone 11: Paper Packaging

### Objective

Turn the benchmark and method into a paper-ready artifact set.

### Work Items

- freeze the benchmark version
- freeze the main system version
- draft tables and figures
- write the benchmark section, method section, and experimental setup
- draft the ethics/release note

### Deliverables

- submission-ready draft sections
- figure/table inventory
- artifact checklist

### Evaluation Criteria

- the benchmark, method, and metrics tell one coherent story
- tables already support the main claim without extra interpretation
- release constraints are documented honestly
- the narrative does not depend on unfinished side experiments

### Exit Condition

The project can move from system-building to paper-polishing.

## Milestone 12: Submission Readiness

### Objective

Reach a point where the paper can be reviewed internally and submitted.

### Work Items

- run a final experiment freeze
- perform internal review
- rewrite introduction and discussion based on final results
- finalize the abstract and title
- verify reproducibility commands

### Deliverables

- final draft
- final figures and tables
- final experiment log

### Evaluation Criteria

- the main result is stable
- all headline tables are reproducible
- the paper clearly states limitations and governance constraints
- there is a defensible answer to “why is this new and why does it matter?”

### Exit Condition

The project is ready for a real submission decision.

## Current Starting Point

Right now, the highest-priority milestones are:

1. Milestone 0: Research Freeze
2. Milestone 1: Codebook v2 and Annotation Policy
3. Milestone 2: Pilot Benchmark and Calibration
4. Milestone 3: Benchmark v1

Until those are done, model work should stay lightweight and exploratory.
