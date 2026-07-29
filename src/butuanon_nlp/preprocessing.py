from __future__ import annotations

import re
from typing import Dict, List


def build_preprocessing_plan() -> List[Dict[str, str]]:
    """Create the data-preprocessing workflow described in the PDF."""
    return [
        {
            "name": "Digitization and transcription",
            "goal": "Convert physical documents and field audio into UTF-8 text and define a glottal-stop convention.",
        },
        {
            "name": "Audio slicing",
            "goal": "Slice recordings into 3-10 second WAV chunks that match single sentences or phrases.",
        },
        {
            "name": "Sentence alignment",
            "goal": "Align English and Butuanon sentences in a structured JSONL or TSV dataset.",
        },
        {
            "name": "Native speaker validation",
            "goal": "Have fluent speakers review the pairs for contextual and phonetic accuracy.",
        },
    ]


# ── Glottal stop detection ──────────────────────────────────────────────────

# Butuanon uses apostrophes and hyphens *between letters* to mark the
# glottal stop (ʔ).  Examples:
#   dal-a  →  dalʔa    (hyphen between two letters)
#   Hain'ka → Hainʔka  (apostrophe between two letters)
#
# We must NOT treat punctuation-adjacent hyphens (e.g. "—" em-dash) or
# apostrophes at word boundaries (e.g. English possessive "it's") as
# glottal markers.  The regex below matches a letter, then an apostrophe
# or hyphen, then another letter — capturing only mid-word markers.

_GLOTTAL_PATTERN = re.compile(
    r"(?<=[A-Za-zÀ-ÿ])(['\u2018\u2019\u02BC\-])(?=[A-Za-zÀ-ÿ])"
)

# IPA glottal stop symbol
_IPA_GLOTTAL = "ʔ"


def phoneticize_text(text: str) -> str:
    """Show where glottal stops occur in Butuanon text.

    Replaces mid-word apostrophes and hyphens with the IPA glottal stop
    symbol (ʔ) and returns an annotated reading guide.  If the text has
    no glottal markers, returns an empty string (no guide needed).

    Examples
    --------
    >>> phoneticize_text("Palihug dal-a ako kanang tubig.")
    "Palihug dalʔa ako kanang tubig.  ·  dal-a → dalʔa"

    >>> phoneticize_text("Maayong buntag!")
    ""
    """
    if not text:
        return ""

    # Find all mid-word glottal markers
    markers = list(_GLOTTAL_PATTERN.finditer(text))
    if not markers:
        return ""  # no glottal markers — nothing to annotate

    # Build the IPA version
    ipa_text = _GLOTTAL_PATTERN.sub(_IPA_GLOTTAL, text)

    # Build a list of specific transformations for the guide
    seen: Dict[str, str] = {}
    for match in markers:
        # Extract the word surrounding the marker
        start = match.start()
        end = match.end()
        # Expand to full word boundaries
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        while end < len(text) and text[end].isalpha():
            end += 1
        original_word = text[start:end]
        ipa_word = _GLOTTAL_PATTERN.sub(_IPA_GLOTTAL, original_word)
        if original_word != ipa_word and original_word not in seen:
            seen[original_word] = ipa_word

    # Format the guide
    transformations = "  ·  ".join(
        f"{orig} → {ipa}" for orig, ipa in seen.items()
    )

    return f"{ipa_text}  ·  {transformations}"
