"""Text normalisation shared by the extraction and cleaning stages."""
import re
import unicodedata

# Values the original annotators used to mean "not reported in the paper".
NOT_REPORTED = {"", "-", "--", "---", "n/a", "na", "nan", "none", "not reported"}

# The annotators prefixed each marked segment inside a cell with an underscore.
_LEADING_MARKER = re.compile(r"^[_\-\u2022*\s]+")
_WHITESPACE = re.compile(r"\s+")

# Enumeration prefixes the LLM adds to paraphrases ("1. ", "2) ", "- ", ...).
_ENUMERATION = re.compile(r"^\s*(?:\(?\d{1,2}[.)\]]|[-\u2013\u2022*])\s+")

# Characters that indicate the source cell was damaged by a previous
# encoding round-trip (mojibake). Reported, never silently repaired.
_MOJIBAKE = re.compile(r"[\ufffd]|Ã[\u0080-\u00bf]|â€")


def normalise(text: object) -> str:
    """Collapse a raw spreadsheet cell into a single-line excerpt.

    Applies Unicode NFC composition, drops the annotators' underscore markers,
    and squeezes every run of whitespace (including the newlines that separated
    segments inside a cell) into a single space.
    """
    if text is None:
        return ""
    value = unicodedata.normalize("NFC", str(text))
    value = _WHITESPACE.sub(" ", value).strip()
    value = _LEADING_MARKER.sub("", value)
    return value.strip()


def is_not_reported(text: str) -> bool:
    return text.strip().lower() in NOT_REPORTED


def strip_enumeration(text: str) -> str:
    """Remove a leading list marker left over from LLM generation."""
    return _ENUMERATION.sub("", text, count=1).strip()


def has_mojibake(text: str) -> bool:
    return bool(_MOJIBAKE.search(text))


def segments(cell: object) -> list[str]:
    """Split a cell into the segments the annotators separated by line breaks.

    Normalisation is applied per segment so that the underscore marker is
    removed from each one, not just from the first.
    """
    if cell is None:
        return []
    raw = unicodedata.normalize("NFC", str(cell))
    out = []
    for part in raw.splitlines():
        cleaned = normalise(part)
        if cleaned and not is_not_reported(cleaned):
            out.append(cleaned)
    return out
