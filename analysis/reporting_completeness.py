"""How many studies of the corpus report each element of the guideline.

    python analysis/reporting_completeness.py

For every class (an element of the Jedlitschka et al. reporting guideline) the
number and share of the corpus studies with at least one original excerpt of
that class, from the original rows of data/07_training/instances.csv.
A study without an excerpt is one in which the annotators found no text for
that element, so the share is a direct measure of reporting completeness in
the corpus (Section 5 of the paper).
Written to reports/reporting_completeness.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import config, io_utils, schema  # noqa: E402

FIELDS = ["task", "label", "studies_reporting", "studies_total", "share", "original_excerpts"]


def run() -> list[dict]:
    rows = [r for r in io_utils.read_csv(config.TRAINING_DIR / "instances.csv") if r["origin"] == "original"]
    studies_total = len({r["study_id"] for r in rows})
    table = []
    for task in schema.TASKS:
        for label in schema.LABELS_BY_TASK[task]:
            subset = [r for r in rows if r["task"] == task and r["label"] == label]
            reporting = len({r["study_id"] for r in subset})
            table.append(
                {
                    "task": task,
                    "label": label,
                    "studies_reporting": reporting,
                    "studies_total": studies_total,
                    "share": round(reporting / studies_total, 4),
                    "original_excerpts": len(subset),
                }
            )
    target = config.REPORTS / "reporting_completeness.csv"
    io_utils.write_csv(target, FIELDS, table)
    io_utils.banner("completeness", f"{studies_total} studies -> {target}")
    for row in sorted(table, key=lambda r: r["share"]):
        print(f"  {row['task']:16} {row['label']:22} {row['studies_reporting']:4}/{studies_total}  {100 * row['share']:5.1f}%")
    return table


if __name__ == "__main__":
    run()
