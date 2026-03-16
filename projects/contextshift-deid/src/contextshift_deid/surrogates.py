"""Surrogate replacement for anonymized UPChieve data.

Produces a **derived view** where placeholder tags like ``<PERSON>`` are replaced
with realistic but neutral surface forms.  The canonical source remains the
anonymized tags; surrogate text is layered on top for two practical purposes:

1. Give the action model realistic surface forms at training time.
2. Give annotators concrete names to judge in context.

**Known limitation — co-reference is synthetic, not recovered.**  The sequential
per-session mapper assigns surrogates to each tag occurrence independently.
Repeated ``<PERSON>`` mentions in the same session may get different surrogates
even if the original entity was the same.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import random
import re
from typing import Any

# ---------------------------------------------------------------------------
# Tag pattern — matches ``<ENTITY_TYPE>`` placeholders in anonymized text.
# Intentionally duplicated from ``upchieve_pilot.TAG_RE`` to avoid coupling.
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r"<([A-Z_]+)>")

# ---------------------------------------------------------------------------
# Pool design constraints
# ---------------------------------------------------------------------------
# * PERSON — common US first names; no presidents, historical figures, authors,
#   scientists, or literary characters.  Census-representative diversity but
#   deliberately boring.
# * LOCATION — neutral small-city / geographic names; no capitals, famous
#   cities, or historically loaded locations.
# * SCHOOL — generic realistic school names; no real prestigious or
#   recognizable school names.
# * NRP — most conservative pool.  Broad generic geographic/regional
#   descriptors that avoid specific religious, ethnic, or political group
#   names.
# * COURSE — generic course names and codes.
# * GRADE_LEVEL — common grade references.
# ---------------------------------------------------------------------------

SURROGATE_POOLS: dict[str, list[str]] = {
    "PERSON": [
        # Female (~40) --------------------------------------------------
        "Sarah", "Emily", "Jessica", "Ashley", "Megan",
        "Lauren", "Rachel", "Nicole", "Amanda", "Stephanie",
        "Jennifer", "Michelle", "Christina", "Brittany", "Samantha",
        "Amber", "Heather", "Tiffany", "Melissa", "Rebecca",
        "Kayla", "Danielle", "Alyssa", "Courtney", "Kelsey",
        "Natalie", "Chelsea", "Kimberly", "Andrea", "Vanessa",
        "Monica", "Sandra", "Maria", "Angela", "Lisa",
        "Amy", "Hannah", "Olivia", "Sophia", "Chloe",
        # Male (~40) ---------------------------------------------------
        "Michael", "David", "James", "Daniel", "Matthew",
        "Christopher", "Andrew", "Joshua", "Ryan", "Brandon",
        "Tyler", "Nathan", "Kevin", "Justin", "Jonathan",
        "Brian", "Steven", "Robert", "William", "Thomas",
        "Anthony", "Patrick", "Eric", "Jacob", "Ethan",
        "Travis", "Derek", "Marcus", "Jason", "Kyle",
        "Aaron", "Sean", "Bradley", "Mitchell", "Gregory",
        "Raymond", "Russell", "Scott", "Timothy", "Keith",
    ],
    "LOCATION": [
        "Maplewood", "Cedar Falls", "Thornton", "Brookfield", "Fairview",
        "Greendale", "Oakwood", "Pine Ridge", "Lakeview", "Hillcrest",
        "Meadowbrook", "Springdale", "Willowbrook", "Elmwood", "Stonegate",
        "Westbrook", "Northfield", "Clearview", "Ridgewood", "Ashford",
        "Millbrook", "Pinehurst", "Bayview", "Harborfield", "Crestwood",
        "Woodlands", "Bridgewater", "Riverdale", "Glenwood", "Lakewood",
        "Forestville", "Clarkton", "Shelton", "Dalton", "Weston",
        "Clayton", "Eastfield", "Summitville", "Whitfield", "Mooreland",
        "Granville", "Bellmont", "Hartfield", "Rosewood", "Brentwood",
        "Ashland", "Ferndale", "Edgewood", "Haverfield", "Pennbrook",
        "Cedarville", "Oakmont", "Sterling", "Windham", "Rockport",
        "Stratton", "Waverly", "Birchwood", "Glenfield", "Marshfield",
    ],
    "SCHOOL": [
        "Westfield High School", "Riverside Middle School", "Oakwood Elementary",
        "Lakeview Academy", "Cedar Creek High School", "Brookside Middle School",
        "Fairview High School", "Hillcrest Middle School", "Meadow Lane Elementary",
        "Pine Valley High School", "Springdale Academy", "Northfield High School",
        "Greendale Middle School", "Elmwood High School", "Stonegate Academy",
        "Bridgewater High School", "Crestwood Middle School", "Summit High School",
        "Millbrook Academy", "Clearview High School", "Ridgewood Middle School",
        "Ashford Elementary", "Woodlands High School", "Bayview Academy",
        "Harborfield Middle School",
    ],
    "NRP": [
        # Most conservative pool — broad geographic/regional descriptors only.
        # Avoids specific religious, ethnic, or political group names.
        "Northerner", "Southerner", "Easterner", "Westerner",
        "Islander", "Highlander", "Lowlander", "Midlander",
        "Coastal", "Valley", "Plains", "River",
        "Mountain", "Prairie", "Lakeside",
    ],
    "COURSE": [
        "English 101", "History 201", "BIO-110", "MATH-150", "CHEM-101",
        "Literature 102", "World History 200", "Government 301", "PHYS-120",
        "Spanish 101", "Algebra II", "Earth Science", "Creative Writing",
        "Economics 201", "Psychology 101", "Sociology 110", "Art History 100",
        "Music Theory", "Computer Science 101", "Geography 200",
    ],
    "GRADE_LEVEL": [
        "9th grade", "10th grade", "11th grade", "12th grade",
        "sophomore", "junior", "senior", "freshman",
        "8th grade", "7th grade",
    ],
}

# Contact/ID types — use obviously fake reserved formats.
FORMAT_TEMPLATES: dict[str, str] = {
    "PHONE_NUMBER": "555-01{:02d}",               # FCC reserved
    "EMAIL_ADDRESS": "user{}@example.com",          # RFC 2606 reserved domain
    "URL": "https://example.com/page{}",
    "SOCIAL_HANDLE": "@user_{:05d}",
    "IP_ADDRESS": "192.0.2.{}",                     # RFC 5737 documentation range
    "US_DRIVER_LICENSE": "[REDACTED-DL-{:04d}]",
    "US_BANK_NUMBER": "[REDACTED-BANK-{:04d}]",
    "US_PASSPORT": "[REDACTED-PASSPORT-{:04d}]",
    "US_SSN": "[REDACTED-SSN-{:04d}]",
}


class SessionSurrogateMapper:
    """Deterministic per-session surrogate assignment.

    Each new occurrence of a tag type gets the next entry from a
    deterministically shuffled pool (or a sequential format string for
    contact/ID types).  Wraps around if a session has more occurrences than
    pool entries.

    Parameters
    ----------
    seed:
        Global seed for reproducibility across runs.
    session_id:
        Unique session identifier — the combination of *seed* and
        *session_id* fully determines the mapping.
    """

    def __init__(self, seed: int, session_id: str) -> None:
        combined = hashlib.sha256(f"{seed}:{session_id}".encode("utf-8")).hexdigest()
        self._rng = random.Random(combined)
        self._shuffled_pools: dict[str, list[str]] = {}
        self._counters: dict[str, int] = defaultdict(int)

        for entity_type, pool in SURROGATE_POOLS.items():
            shuffled = list(pool)
            self._rng.shuffle(shuffled)
            self._shuffled_pools[entity_type] = shuffled

    def next_surrogate(self, entity_type: str) -> tuple[str, int]:
        """Return ``(surrogate_text, occurrence_number)`` for *entity_type*.

        *occurrence_number* is 1-indexed and counts across the whole session.
        """
        self._counters[entity_type] += 1
        occurrence = self._counters[entity_type]

        if entity_type in self._shuffled_pools:
            pool = self._shuffled_pools[entity_type]
            return pool[(occurrence - 1) % len(pool)], occurrence

        if entity_type in FORMAT_TEMPLATES:
            return FORMAT_TEMPLATES[entity_type].format(occurrence), occurrence

        # Unknown entity type — keep the original tag visible.
        return f"<{entity_type}>", occurrence


def replace_tags(
    text: str,
    mapper: SessionSurrogateMapper,
) -> tuple[str, list[dict[str, Any]]]:
    """Replace ``<TAG>`` placeholders with surrogates and extract span metadata.

    All span extraction happens inside this function — no second regex pass
    is needed downstream.

    Returns
    -------
    replaced_text:
        The text with all recognised tags replaced by surrogates.
    spans:
        One dict per replaced tag, with keys ``span_text`` (the surrogate),
        ``original_tag``, ``entity_type``, ``tag_start``, ``tag_end`` (char
        positions in *replaced_text*), and ``tag_occurrence``.
    """
    spans: list[dict[str, Any]] = []
    parts: list[str] = []
    last_end = 0
    offset = 0

    for match in TAG_RE.finditer(text):
        entity_type = match.group(1)
        original_tag = match.group(0)
        surrogate, occurrence = mapper.next_surrogate(entity_type)

        parts.append(text[last_end:match.start()])

        new_start = match.start() + offset
        new_end = new_start + len(surrogate)

        parts.append(surrogate)

        spans.append({
            "span_text": surrogate,
            "original_tag": original_tag,
            "entity_type": entity_type,
            "tag_start": new_start,
            "tag_end": new_end,
            "tag_occurrence": occurrence,
        })

        offset += len(surrogate) - len(original_tag)
        last_end = match.end()

    parts.append(text[last_end:])
    return "".join(parts), spans
