# Codebook v3

This document extends [codebook_v2.md](./codebook_v2.md) for the anonymized UPChieve English and Social Studies pilot.

The action task is still:

`Given a suspicious span in context, should the system REDACT it, KEEP it, or send it to REVIEW?`

The pilot adds one more required output per annotation:

- `semantic_role=PRIVATE` when the span refers to a real participant or identifying detail
- `semantic_role=CURRICULAR` when the span is part of lesson content
- `semantic_role=AMBIGUOUS` when the context does not support a confident binary reading

## Annotation Unit

Annotate one anonymization tag occurrence at a time.

For the UPChieve pilot:

1. Read `span_text` and the highlighted `target_turn`.
2. Read the local `context_text`.
3. Read `anchor_text`.
4. Check `subject`, `eval_slice`, `speaker_role`, and `entity_type`.

If a turn contains repeated tags such as multiple `<PERSON>` mentions, use the highlighted `target_turn` to decide the exact occurrence being labeled.

## Label Definitions

### `REDACT`

Use `REDACT` when the tag stands for a real participant, school, location, or other identifying detail in the tutoring exchange.

Typical `REDACT` cases:

- student or volunteer self-introductions
- real teacher or counselor names
- school names tied to the current student
- direct contact or account-style tags
- local places, hometowns, or other personal locations

Default semantic role: `PRIVATE`

### `KEEP`

Use `KEEP` when the tag stands for curricular content that is necessary to preserve educational meaning.

Typical `KEEP` cases:

- authors, characters, books, poems, and fictional settings in English
- historical figures, countries, wars, movements, and textbook places in Social Studies
- course or assignment references used as instructional framing

Default semantic role: `CURRICULAR`

### `REVIEW`

Use `REVIEW` when the context honestly supports both a private and curricular reading.

Typical `REVIEW` cases:

- a `<PERSON>` tag that could be either a classmate/teacher or an author/character
- a `<LOCATION>` tag that could be either a historical place or a student's local place
- a `<SCHOOL>` tag that could be either a real school or a named program/concept

Default semantic role: `AMBIGUOUS`

## Subject Rules

### English

- `<PERSON>` is often `KEEP` when it stands for an author, character, or speaker inside the assigned text.
- `<PERSON>` is usually `REDACT` when it is used in greetings, direct address, self-identification, turn-taking, or school logistics.
- `<LOCATION>` is often `KEEP` when it is a fictional or literary setting.
- `<SCHOOL>` is often `REDACT` when it names the student's actual school, but may be `REVIEW` if the context is about admissions, rankings, or essay prompts.
- `<COURSE>` is usually `KEEP` when it refers to classwork, readings, or an assignment.

### Social Studies

- `<PERSON>` is often `KEEP` when it stands for a historical figure, politician, ruler, or public figure under discussion.
- `<LOCATION>` is often `KEEP` when it stands for a country, state, city, battlefield, colony, or other textbook place.
- `NRP` is often `KEEP` when it stands for nationality, religion, or political affiliation used as lesson content.
- `<SCHOOL>` is usually `REDACT` when it identifies the real student, but may be `REVIEW` if it appears inside a comparative or institutional discussion.
- `<COURSE>` is usually `KEEP` when it is part of the assignment or lesson description.

## Fast Decision Order

1. If the tag is direct contact or account information, label `REDACT`.
2. If the tag appears in a direct greeting, self-introduction, or direct address to a real participant, label `REDACT`.
3. If the tag clearly refers to the real student, volunteer, teacher, school, or local place in this exchange, label `REDACT`.
4. If the tag clearly belongs to English or Social Studies content, label `KEEP`.
5. If both readings still fit after reading `context_text` and `anchor_text`, label `REVIEW`.

## Exclusion Rules

Some `readingWriting` sessions are not part of the intended English-only pool.

- Do not annotate rows from multilingual or non-English English-topic sessions when the pool builder should have filtered them out.
- Examples include explicit `spanish`, `espanol`, `español`, `arabic`, `persian`, `farsi`, or `urdu` markers in the English-topic session text.
- English-topic rows whose target turn contains Arabic-script text should also stay out of the pool.
- If one of those rows appears anyway, treat it as a pool bug and exclude it from the annotation round rather than forcing an action label.

## Canonical Pilot Examples

### `REDACT`

- `<PERSON>` in `volunteer: Hi <PERSON>, can you hear me?`
- `<PERSON>` in `student: My name is <PERSON>`
- `<PERSON>` in `student: Hi <PERSON> can you review my essay?`
- `<SCHOOL>` in `student: I go to <SCHOOL> and I want to transfer`
- `<LOCATION>` in `student: I live in <LOCATION> so commuting is hard`
- `<PHONE_NUMBER>` in `text me at <PHONE_NUMBER>`

### `KEEP`

- `<PERSON>` in `student: In The Scarlet Letter, <PERSON> is judged unfairly`
- `<PERSON>` in `student: Compare <PERSON> and <PERSON> in Chekhov`
- `<LOCATION>` in `student: Why did fighting start in <LOCATION>?`
- `NRP` in `student: Were the <NRP> or the <NRP> more prepared for war?`

### `REVIEW`

- `<PERSON>` in `student: <PERSON> told us to read this before class`
- `<LOCATION>` in `student: We talked about <LOCATION> today`
- `<SCHOOL>` in `student: Is <SCHOOL> better for this path?`

## Edge Rules

- Do not label a tag `REDACT` just because the underlying entity type sounds personal.
- Do not label a tag `KEEP` just because the conversation is educational.
- Use `REVIEW` when the ambiguity is real, not when the example is merely unfamiliar.
- Prefer the highlighted target occurrence over nearby repeated tags.

## Output Expectations

- `REDACT` rows should support privacy recall under non-math context shift.
- `KEEP` rows should preserve curricular meaning in English and Social Studies.
- `REVIEW` rows should stay limited to honest ambiguity cases.
- `semantic_role` should match the reasoning behind the action label.
