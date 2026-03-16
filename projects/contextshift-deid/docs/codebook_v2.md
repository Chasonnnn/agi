# Codebook v2

This document defines the manual action-label policy for context-aware de-identification in `contextshift-deid`.

The action task is not "is this token ever person-like." The action task is:

`Given a suspicious span in context, should the system REDACT it, KEEP it, or send it to REVIEW?`

## Annotation Unit

Annotate one suspicious span at a time.

The annotator does not create new spans in this phase. The candidate stage proposes spans; the action stage decides what to do with each proposed span.

Read, in order:

1. `span_text`
2. the local token preview if present
3. `context_text`
4. `anchor_text` if present
5. any subject or speaker metadata

## Label Definitions

### `REDACT`

Use `REDACT` when the span is private or identifying in this tutoring context.

Typical `REDACT` cases:

- student or volunteer names used as real people in the conversation
- usernames, handles, emails, phone numbers, URLs, or profile links
- school names when they identify the student or volunteer
- grade level or age when it describes the real student or volunteer
- teacher names when they identify a real person from the student's school
- personal location references
- IDs or account-like strings

### `KEEP`

Use `KEEP` when the span is part of curricular content, task content, or harmless instructional framing and should remain visible.

Typical `KEEP` cases:

- historical, literary, or scientific entities that are part of the lesson
- names in synthetic word problems or worksheet prompts
- course or topic names that are instructional rather than identifying
- references required to preserve the educational meaning of the example

### `REVIEW`

Use `REVIEW` when the context is genuinely ambiguous and neither `REDACT` nor `KEEP` is defensible with confidence.

`REVIEW` is not a fatigue label. Use it only when:

- the span could plausibly be private or curricular
- the visible context is insufficient to resolve that ambiguity
- the span may require a higher-cost model or human adjudication

## Decision Order

1. If the span is direct contact info or a unique identifier, label `REDACT`.
2. If the span names a real participant or other real local person in the tutoring exchange, label `REDACT`.
3. If the span clearly belongs to lesson content, label `KEEP`.
4. If both readings remain plausible after reading the available context, label `REVIEW`.

## Subject Rules

### Math

- First names inside greetings, closings, or turn-taking are usually `REDACT`.
- Grade levels are usually `REDACT` when they describe the real student or volunteer.
- Names inside word problems are usually `KEEP`.
- Named mathematical objects are `KEEP`.
- Teacher names are usually `REDACT` unless the name is clearly part of a canned exercise.

### History

- historical figures, wars, countries, dynasties, and textbook place names are usually `KEEP`
- teacher, classmate, or school names are usually `REDACT`
- person names with too little context may be `REVIEW`

### Literature

- authors, characters, books, poems, and fictional places are usually `KEEP`
- real teacher names, student names, and local school references are usually `REDACT`
- ambiguous person names without anchor context may be `REVIEW`

## Canonical Examples

### `REDACT` examples

- `John` in `volunteer: Hello John`
- `Sarah` in `student: Hi Sarah!`
- `ava@example.com` in `Email me at ava@example.com`
- `555-444-1000` in `Call me at 555-444-1000`
- `Lincoln High` in `I go to Lincoln High`
- `eighth grade` in `im in eighth grade too`
- `Mr. Shah` in `my teacher Mr. Shah gave me this worksheet`
- `discord.gg/...` in `join my discord.gg/...`
- `https://example.com` in `here is my personal site https://example.com`
- `Queens` in `I live in Queens`

### `KEEP` examples

- `Macbeth` in `Macbeth is ambitious`
- `Lady Macbeth` in `Macbeth and Lady Macbeth argue`
- `Napoleon` in `Napoleon invaded Russia`
- `Shakespeare` in `Compare Shakespeare and Sophocles`
- `Pythagorean theorem` in `Use the Pythagorean theorem here`
- `Algebra II` in `This looks like Algebra II review`
- `triangle ABC` in `Find the area of triangle ABC`
- `Sophia` in `Sophia solved the equation` when this is clearly a worksheet name
- `Romeo` in `Why does Romeo act impulsively?`
- `World War II` in `Compare causes of World War II`

### `REVIEW` examples

- `Mr. Lopez` in `Mr. Lopez likes Macbeth`
- `Jordan` in `Jordan was part of the example` when it is unclear whether this is a country, a student, or a worksheet name
- `Lincoln` in `Lincoln was mentioned in class today`
- `Pat` in `wait Pat write it on the board`
- `the link` in `text me the link`
- `Ava` in `Ava told me to solve for x` when it is unclear whether Ava is a real participant or a problem character
- `Harvard` in `Harvard method` when it is unclear whether the span is a school reference or a named concept
- `Ms. Kim` in `Ms. Kim said to use this reading` when there is not enough context to tell whether this is a real teacher or part of supplied material
- `Washington` in `Washington wrote back` when the historical reading and local-person reading both fit
- `Room 204` in `meet me in Room 204 after class`

## Edge Rules

- Do not label a span `KEEP` just because it appears in an educational setting.
- Do not label a span `REDACT` just because it is capitalized.
- Do not convert an unclear case into forced binary supervision if `REVIEW` is the honest label.
- If the span is an obvious privacy risk, prefer `REDACT` over `REVIEW`.
- If the span is an obvious curricular entity, prefer `KEEP` over `REVIEW`.

## Adjudication Checklist

Before escalating or adjudicating a difficult item, check:

1. Does the span refer to a real participant in this conversation?
2. Does the span expose contact, location, school, grade, or identity information?
3. Does the span belong to lesson content, anchor text, or a synthetic example?
4. Would redacting the span damage educational meaning?
5. Is the ambiguity real, or is more careful reading enough to resolve it?

## Output Expectations

For exported benchmark rows:

- `REDACT` should be well-supported privacy supervision
- `KEEP` should preserve curricular meaning rather than act as a generic negative label
- `REVIEW` should be a small but real class for unresolved ambiguity

If a case keeps causing confusion, revise this codebook before scaling annotation.
