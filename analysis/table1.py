"""Compare the corpus with Table 1 of the paper.

Three columns per class, written to reports/table1_comparison.csv:

* ``before_original_rule`` - populated cells counted with the empty-cell rule of the
  original extraction script (only "", "---" and "nan" are empty), Variables
  counted once per study, as in the published "Before" column;
* ``before_clean``  - the original excerpts that actually enter the dataset
  (cells such as "----" are dropped; Variables split into two classes);
* ``after``         - instances after augmentation and balancing.
"""
from __future__ import annotations

from collections import Counter

from pipeline import config, io_utils, schema

FIELDS = [
    "task", "label", "before_original_rule", "paper_before", "diff_before",
    "before_clean", "after", "paper_after", "diff_after",
]


def _original_rule_filled(value) -> int:
    if value is None:
        return 0
    text = str(value).strip().replace("\n", " ").replace("_", "")
    return int(text not in ("", "---", "nan"))


def run() -> list[dict]:
    included = {int(s["study_id"]) for s in io_utils.read_csv(config.CORPUS_DIR / "studies.csv") if s["included"] == "yes"}
    rows = [r for r in io_utils.read_sheet(config.SOURCE_WORKBOOK, config.SOURCE_SHEET)[1:] if r[0] is not None]
    original_rule = Counter()
    for row in rows:
        if int(row[schema.COL_STUDY_ID]) not in included:
            continue
        for spec in schema.COLUMNS:
            key = "Variables" if spec.splitter == "variables" else spec.label
            original_rule[key] += _original_rule_filled(row[spec.index] if spec.index < len(row) else None)

    clean = Counter(r["label"] for r in io_utils.read_csv(config.CLEAN_DIR / "instances.csv"))
    final_path = config.FINAL_DIR / "instances.csv"
    after = Counter(r["label"] for r in io_utils.read_csv(final_path)) if final_path.exists() else Counter()

    table = []
    for task in schema.TASKS:
        for label in schema.LABELS_BY_TASK[task]:
            before_key = "Variables" if label in ("IndependentVariables", "DependentVariables") else label
            paper_before = config.PAPER_BEFORE[before_key]
            table.append(
                {
                    "task": task,
                    "label": label,
                    "before_original_rule": original_rule[before_key],
                    "paper_before": paper_before,
                    "diff_before": original_rule[before_key] - paper_before,
                    "before_clean": clean[label],
                    "after": after[label],
                    "paper_after": config.PAPER_AFTER[label],
                    "diff_after": after[label] - config.PAPER_AFTER[label],
                }
            )
    io_utils.write_csv(config.REPORTS / "table1_comparison.csv", FIELDS, table)

    # Variables appears on two rows; count it once in the Before total.
    seen, total_original_rule = set(), 0
    for row in table:
        key = "Variables" if row["label"] in ("IndependentVariables", "DependentVariables") else row["label"]
        if key not in seen:
            seen.add(key)
            total_original_rule += row["before_original_rule"]
    total_after = sum(row["after"] for row in table)
    mismatched = [f"{r['label']} {r['diff_before']:+d}" for r in table if r["diff_before"]]
    io_utils.banner("table1", f"Before (original rule) {total_original_rule} vs paper {config.PAPER_BEFORE_TOTAL}")
    io_utils.banner("table1", f"Before differences: {', '.join(mismatched) or 'none'}")
    io_utils.banner("table1", f"After {total_after} vs paper {config.PAPER_AFTER_TOTAL}")
    return table


if __name__ == "__main__":
    run()
