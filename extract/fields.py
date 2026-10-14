"""Check spans against the PDF text and choose the text of each instance.

The paper lists the lack of an automated substring check after the LLM
grounding as a threat to validity ("trust-but-anchor"). Here every span is
anchored before it is used:

* ``exact``   - the span occurs in the PDF text (whitespace and Unicode
                compatibility forms normalised, so ligatures such as "ﬁ" match);
* ``compact`` - it occurs once case, punctuation and whitespace are ignored,
                which absorbs line-break hyphenation and PyMuPDF spacing;
* ``gapped``  - it occurs in order, in pieces of at least ``MIN_PIECE``
                characters, separated by short gaps (at most ``MAX_GAP``).
                The gaps are page furniture that the PDF text puts in the
                middle of a sentence and that the model rightly skips: the
                ACM permission block, IEEE "Authorized licensed use" footers,
                running heads and page numbers;
* ``partial`` - a span made of several passages joined by "..." where only
                some passages were found;
* ``missing`` - not found.

Only ``exact``, ``compact`` and ``gapped`` count as anchored. A grounded span
that is not anchored is never used: the instance keeps the spreadsheet text
and the report records why.
"""
from __future__ import annotations

import re
import unicodedata

from pipeline import textnorm

ANCHORED = {"exact", "compact", "gapped"}
# A multi-passage span is reported with the weakest method among its passages.
ANCHORED_RANK = {"exact": 0, "compact": 1, "gapped": 2}

# Compact characters (letters and digits only) for the gapped match. A gap can
# hold a whole table that PyMuPDF emits in the middle of a sentence, hence the
# generous MAX_GAP; the minimum piece size and the cap on the number of gaps
# keep a span stitched from scattered fragments from passing.
MIN_PIECE = 25
MAX_GAP = 3000
CHARS_PER_GAP = 400

_PASSAGE_SEPARATOR = re.compile(r"\s*(?:\.{3}|…)\s*")
_WHITESPACE = re.compile(r"\s+")


def clean_span(text: str) -> str:
    """Unicode compatibility form (ligatures, full-width forms) on one line."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    return _WHITESPACE.sub(" ", value).strip()


def compact(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _longest_prefix(needle: str, haystack: str) -> tuple[int, int]:
    """Length and position of the longest prefix of ``needle`` in ``haystack``."""
    low, high, where = 0, len(needle), 0
    while low < high:
        middle = (low + high + 1) // 2
        position = haystack.find(needle[:middle])
        if position >= 0:
            low, where = middle, position
        else:
            high = middle - 1
    return low, where


def gapped_match(needle: str, haystack: str) -> bool:
    """True when ``needle`` occurs in ``haystack`` in order, with short gaps."""
    max_gaps = 1 + len(needle) // CHARS_PER_GAP
    length, start = _longest_prefix(needle, haystack)
    if length < min(MIN_PIECE, len(needle)):
        return False
    done, end, gaps = length, start + length, 0
    while done < len(needle):
        gaps += 1
        window = haystack[end : end + MAX_GAP + len(needle) - done]
        length, offset = _longest_prefix(needle[done:], window)
        remaining = len(needle) - done
        # Only the closing piece of the span may be shorter than MIN_PIECE.
        if gaps > max_gaps or offset > MAX_GAP or length < min(MIN_PIECE, remaining):
            return False
        done += length
        end += offset + length
    return True


class PdfText:
    """The text of one PDF, pre-normalised for repeated lookups."""

    def __init__(self, raw: str) -> None:
        # Words hyphenated across a line break are joined before matching.
        joined = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
        self.flat = clean_span(joined)
        self.compact = compact(raw)

    def locate(self, span: str) -> str:
        span = clean_span(span)
        if not span or textnorm.is_not_reported(span):
            return "empty"
        passages = [p for p in _PASSAGE_SEPARATOR.split(span) if compact(p)]
        if not passages:
            return "empty"

        methods = []
        for passage in passages:
            if passage in self.flat:
                methods.append("exact")
            elif compact(passage) in self.compact:
                methods.append("compact")
            elif gapped_match(compact(passage), self.compact):
                methods.append("gapped")
            else:
                methods.append("missing")

        for method in ("exact", "compact", "gapped"):
            if all(m in ANCHORED and ANCHORED_RANK[m] <= ANCHORED_RANK[method] for m in methods):
                return method
        return "partial" if any(m in ANCHORED for m in methods) else "missing"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKD", text).lower()))


def jaccard(a: str, b: str) -> float:
    left, right = _words(a), _words(b)
    return len(left & right) / len(left | right) if left | right else 0.0


def resolve_label_conflicts(rows: list[dict]) -> int:
    """Keep a grounded span under one label only, within one study.

    Sometimes the model returns the same passage for two classes whose
    spreadsheet excerpts were different: either the sentence really covers
    both (the annotators cut different parts of it) or a vague excerpt such
    as "Experiments." was mapped to a neighbouring sentence. Training on the
    same text under two labels is contradictory, so the span stays with the
    label whose spreadsheet excerpt it resembles most (Jaccard over words) and
    the other labels fall back to their spreadsheet text.

    Conflicts already present in the spreadsheet (identical excerpts recorded
    under two labels by the annotators) are left untouched. Rows are updated
    in place; returns how many were reverted.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if row["text_source"] == "pdf":
            groups.setdefault(row["text"].casefold(), []).append(row)

    reverted = 0
    for group in groups.values():
        if len({row["label"] for row in group}) < 2:
            continue
        if len({row["text_sheet"].casefold().strip() for row in group}) < 2:
            continue  # the annotators themselves used one excerpt for both
        winner = max(group, key=lambda row: jaccard(row["text_sheet"], row["text"]))
        for row in group:
            if row is winner:
                row["conflict"] = "kept"
                continue
            row["text"] = row["text_sheet"]
            row["text_source"] = "sheet"
            row["conflict"] = f"reverted: span kept by {winner['label']}"
            reverted += 1
    return reverted


def choose_text(sheet_text: str, span: str, pdf: PdfText | None) -> dict:
    """Decide the text of one instance.

    Returns the chosen text, where it came from (``pdf`` or ``sheet``) and how
    the grounded span and the spreadsheet excerpt were located in the PDF.
    """
    if pdf is None:
        return {
            "text": sheet_text,
            "text_source": "sheet",
            "span_method": "no-pdf",
            "sheet_method": "no-pdf",
        }

    sheet_method = pdf.locate(sheet_text)
    span_method = pdf.locate(span) if span else "not-grounded"

    if span_method in ANCHORED:
        text, source = clean_span(span), "pdf"
    else:
        text, source = sheet_text, "sheet"
    return {
        "text": text,
        "text_source": source,
        "span_method": span_method,
        "sheet_method": sheet_method,
    }
