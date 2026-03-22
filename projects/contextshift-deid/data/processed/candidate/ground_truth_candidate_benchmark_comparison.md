# Ground-Truth Benchmark Comparison

Count-based comparison after normalizing the previous benchmark label inventory into the current math taxonomy.

## upchieve

### Positive Counts

- previous: `{"positive_dialogue_count": 609, "positive_row_count": 1634, "row_count": 115614}`
- current: `{"positive_dialogue_count": 609, "positive_row_count": 1634, "row_count": 115614}`

### Span Counts

- previous_total: `1744`
- current_total: `1744`

### Span Delta After Normalization

- overlap_count: `1744`
- removed_count: `0`
- added_count: `0`

### Canonical Label Counts

```json
{
  "current": {
    "ADDRESS": 62,
    "AGE": 9,
    "DATE": 2,
    "IDENTIFYING_NUMBER": 3,
    "IP_ADDRESS": 2,
    "NAME": 1440,
    "PHONE_NUMBER": 3,
    "SCHOOL": 71,
    "TUTOR_PROVIDER": 3,
    "URL": 149
  },
  "delta": {
    "ADDRESS": 0,
    "AGE": 0,
    "DATE": 0,
    "IDENTIFYING_NUMBER": 0,
    "IP_ADDRESS": 0,
    "NAME": 0,
    "PHONE_NUMBER": 0,
    "SCHOOL": 0,
    "TUTOR_PROVIDER": 0,
    "URL": 0
  },
  "previous": {
    "ADDRESS": 62,
    "AGE": 9,
    "DATE": 2,
    "IDENTIFYING_NUMBER": 3,
    "IP_ADDRESS": 2,
    "NAME": 1440,
    "PHONE_NUMBER": 3,
    "SCHOOL": 71,
    "TUTOR_PROVIDER": 3,
    "URL": 149
  }
}
```

## saga27

### Positive Counts

- previous: `{"positive_dialogue_count": 0, "positive_row_count": 575, "row_count": 13684}`
- current: `{"positive_dialogue_count": 0, "positive_row_count": 575, "row_count": 13684}`

### Span Counts

- previous_total: `732`
- current_total: `732`

### Span Delta After Normalization

- overlap_count: `732`
- removed_count: `0`
- added_count: `0`

### Canonical Label Counts

```json
{
  "current": {
    "ADDRESS": 21,
    "AGE": 6,
    "DATE": 1,
    "IDENTIFYING_NUMBER": 3,
    "NAME": 690,
    "SCHOOL": 4,
    "TUTOR_PROVIDER": 2,
    "URL": 5
  },
  "delta": {
    "ADDRESS": 0,
    "AGE": 0,
    "DATE": 0,
    "IDENTIFYING_NUMBER": 0,
    "NAME": 0,
    "SCHOOL": 0,
    "TUTOR_PROVIDER": 0,
    "URL": 0
  },
  "previous": {
    "ADDRESS": 21,
    "AGE": 6,
    "DATE": 1,
    "IDENTIFYING_NUMBER": 3,
    "NAME": 690,
    "SCHOOL": 4,
    "TUTOR_PROVIDER": 2,
    "URL": 5
  }
}
```

