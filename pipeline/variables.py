"""Splitter for the single ``Variables`` column of the source spreadsheet.

Jedlitschka et al. (2008) define one ``Variables`` construct, and the source
spreadsheet follows that convention. The original annotators, however,
consistently recorded independent and dependent variables as separately marked
segments inside the same cell, which is what makes the split possible and is
why the study reports 11 planning classes from 10 planning columns.

Observed segment vocabulary across the 198 non-empty cells:

* 186 cells contain an explicit ``independent`` marker, 190 a ``dependent`` one;
* 7 cells use experimental-design vocabulary instead (``factor`` / ``treatment``,
  one of them in Portuguese: ``Fator`` / ``Tratamentos``);
* segments carrying no marker are continuations of the previous marked segment.

Anything that cannot be routed is returned as unresolved and written to a
report instead of being guessed.
"""
import re

from . import textnorm

# ``(?<![a-z])`` rather than ``\b``: the underscore the annotators used as a
# marker prefix counts as a word character in regex, so ``\bindependent`` never
# matches ``_Independent``. The lookbehind also keeps ``independent`` from
# matching inside itself when we later search for ``dependent``.
_INDEPENDENT = re.compile(r"(?<![a-z])independent", re.IGNORECASE)
_DEPENDENT = re.compile(r"(?<![a-z])dependent", re.IGNORECASE)

# Secondary vocabulary. A factor and its treatments are, by definition, the
# manipulated (independent) side of the design.
_FACTOR = re.compile(r"(?<![a-z])(factors?|fator(?:es)?|treatments?|tratamentos?)", re.IGNORECASE)

INDEPENDENT = "IndependentVariables"
DEPENDENT = "DependentVariables"


def split(cell: object) -> tuple[dict[str, str], list[str]]:
    """Route the segments of one ``Variables`` cell to the two classes.

    Returns a ``(assigned, unresolved)`` pair, where ``assigned`` maps a class
    name to the joined text of its segments. A class is absent from the mapping
    when the cell says nothing about it.
    """
    assigned: dict[str, list[str]] = {INDEPENDENT: [], DEPENDENT: []}
    unresolved: list[str] = []
    previous: str | None = None

    for segment in textnorm.segments(cell):
        independent = bool(_INDEPENDENT.search(segment))
        dependent = bool(_DEPENDENT.search(segment))

        if independent and dependent:
            # A single segment describing both sides is kept on both, because
            # dropping it would lose annotator-validated content.
            targets = [INDEPENDENT, DEPENDENT]
        elif independent:
            targets = [INDEPENDENT]
        elif dependent:
            targets = [DEPENDENT]
        elif _FACTOR.search(segment):
            targets = [INDEPENDENT]
        elif previous is not None:
            # Unmarked continuation of the preceding segment.
            targets = [previous]
        else:
            unresolved.append(segment)
            continue

        for target in targets:
            assigned[target].append(segment)
        previous = targets[-1]

    return {
        label: " ".join(parts) for label, parts in assigned.items() if parts
    }, unresolved
