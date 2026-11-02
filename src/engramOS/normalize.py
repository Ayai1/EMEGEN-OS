"""Value normalization for EngramOS fact comparison.

Normalizes extracted fact values so that semantically identical values
(e.g., "three bedrooms" and "3 bedrooms") compare as equal.

Used by the interval-stitching supersession algorithm to determine
whether a new fact confirms or supersedes an existing fact.
"""

from __future__ import annotations

import re
import unicodedata

# English number words → digits (zero through twenty, plus tens)
_NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000",
}

# Build regex pattern for whole-word matching
_NUMBER_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(w) for w in sorted(_NUMBER_WORDS.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE,
)

# Leading/trailing punctuation to strip (preserving internal punctuation like hyphens)
_STRIP_PUNCT = re.compile(r'^[\s\'".,;:!?()]+|[\s\'".,;:!?()]+$')

# Collapse internal whitespace
_COLLAPSE_WS = re.compile(r'\s+')

# Articles to remove at start
_ARTICLES = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)


def normalize_value(value: str | None) -> str:
    """Normalize a fact value for comparison.

    Steps:
    1. Lowercase.
    2. Strip leading/trailing whitespace and surrounding punctuation.
    3. Remove leading articles (the, a, an).
    4. Collapse internal whitespace to single spaces.
    5. Convert standalone English number words (zero–twenty, thirty, etc.)
       to digits so "three bedrooms" and "3 bedrooms" normalize identically.

    Args:
        value: Raw extracted value string.

    Returns:
        Normalized value string. Empty string if value is None or empty.
    """
    if not value:
        return ""

    # Step 1: Lowercase
    result = value.lower()

    # Step 2: Strip surrounding punctuation and whitespace
    result = _STRIP_PUNCT.sub('', result)

    # Step 3: Remove leading articles
    result = _ARTICLES.sub('', result)

    # Step 4: Collapse internal whitespace
    result = _COLLAPSE_WS.sub(' ', result).strip()

    # Step 5: Convert number words to digits
    result = _NUMBER_PATTERN.sub(lambda m: _NUMBER_WORDS[m.group(0).lower()], result)

    return result


def normalize_slot_name(entity: str, attribute: str) -> tuple[str, str]:
    """Normalize entity and attribute names for slot comparison.

    Converts to lowercase snake_case and strips extra whitespace.

    Args:
        entity: Entity name (e.g., "user", "database").
        attribute: Attribute name (e.g., "job_title", "port").

    Returns:
        Tuple of (normalized_entity, normalized_attribute).
    """
    def _normalize_part(s: str) -> str:
        s = s.lower().strip()
        # Replace spaces and hyphens with underscores
        s = re.sub(r'[\s\-]+', '_', s)
        # Remove non-alphanumeric except underscores
        s = re.sub(r'[^a-z0-9_]', '', s)
        # Collapse multiple underscores
        s = re.sub(r'_+', '_', s)
        # Strip leading/trailing underscores
        s = s.strip('_')
        return s

    return _normalize_part(entity), _normalize_part(attribute)
