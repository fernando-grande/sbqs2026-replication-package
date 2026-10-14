"""Label schema: mapping from spreadsheet columns to tasks and classes.

This module is the single source of truth for the 23 classes used in the study.
Everything downstream derives from ``COLUMNS``, so adding or retiring a class is
a one-line change here.

Column indices are zero-based positions in the ``Extracted Data`` sheet of
``planilha_padronizada.xlsx``. They are pinned by index rather than by header
text because several headers embed the full guideline question from
Jedlitschka et al. (2008) and are therefore long and unstable.
"""
from dataclasses import dataclass, field

# Study-level metadata columns.
COL_STUDY_ID = 0
COL_TITLE = 1
COL_YEAR = 3
COL_VENUE = 5

TASKS = ("CT-Abstract", "CT-Introduction", "CT-Planning", "CT-Conclusion")


@dataclass(frozen=True)
class ColumnSpec:
    """One spreadsheet column and the class (or classes) it produces."""

    index: int
    task: str
    label: str
    header: str
    # When set, the cell is not a single excerpt: it holds several marked
    # segments that must be routed to different labels. See ``variables.py``.
    splitter: str | None = None
    extra_labels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def labels(self) -> tuple[str, ...]:
        return (self.label,) + self.extra_labels


COLUMNS: tuple[ColumnSpec, ...] = (
    # --- CT-Abstract -------------------------------------------------------
    ColumnSpec(12, "CT-Abstract", "Background", "Abstract - Background"),
    ColumnSpec(13, "CT-Abstract", "Objective", "Objective"),
    ColumnSpec(14, "CT-Abstract", "Methods", "Methods"),
    ColumnSpec(15, "CT-Abstract", "Results", "Results"),
    ColumnSpec(16, "CT-Abstract", "Limitations", "Limitations"),
    ColumnSpec(17, "CT-Abstract", "Conclusions", "Conclusions"),
    # --- CT-Introduction ---------------------------------------------------
    ColumnSpec(19, "CT-Introduction", "ProblemStatement", "Problem statement"),
    ColumnSpec(20, "CT-Introduction", "ResearchObjectives", "Research objective (GQM)"),
    ColumnSpec(21, "CT-Introduction", "Context", "Context"),
    # --- CT-Planning -------------------------------------------------------
    ColumnSpec(28, "CT-Planning", "Goals", "Goals"),
    ColumnSpec(29, "CT-Planning", "ExperimentalUnits", "Experimental Units"),
    ColumnSpec(30, "CT-Planning", "ExperimentalMaterial", "Experimental Material"),
    ColumnSpec(31, "CT-Planning", "Tasks", "Tasks"),
    ColumnSpec(32, "CT-Planning", "Hypotheses", "Hypotheses"),
    ColumnSpec(33, "CT-Planning", "Parameters", "Parameters"),
    ColumnSpec(
        34,
        "CT-Planning",
        "IndependentVariables",
        "Variables",
        splitter="variables",
        extra_labels=("DependentVariables",),
    ),
    ColumnSpec(35, "CT-Planning", "ExperimentDesign", "Experiment Design"),
    ColumnSpec(36, "CT-Planning", "Procedure", "Procedure"),
    ColumnSpec(37, "CT-Planning", "AnalysisProcedure", "Analysis Procedure"),
    # --- CT-Conclusion -----------------------------------------------------
    ColumnSpec(51, "CT-Conclusion", "Summary", "Summary"),
    ColumnSpec(52, "CT-Conclusion", "Impact", "Impact"),
    ColumnSpec(53, "CT-Conclusion", "FutureWork", "Future Work"),
)

# Presentation order used by the report tables, task by task.
LABELS_BY_TASK: dict[str, tuple[str, ...]] = {
    task: tuple(
        label for spec in COLUMNS if spec.task == task for label in spec.labels
    )
    for task in TASKS
}

ALL_LABELS: tuple[str, ...] = tuple(
    label for task in TASKS for label in LABELS_BY_TASK[task]
)

# Human-readable names for the paper tables.
DISPLAY_NAMES = {
    "Background": "Background",
    "Objective": "Objective",
    "Methods": "Methods",
    "Results": "Results",
    "Limitations": "Limitations",
    "Conclusions": "Conclusions",
    "ProblemStatement": "Problem Statement",
    "ResearchObjectives": "Research Objectives",
    "Context": "Context",
    "Goals": "Goals",
    "ExperimentalUnits": "Experimental Units",
    "ExperimentalMaterial": "Experimental Material",
    "Tasks": "Tasks",
    "Hypotheses": "Hypotheses",
    "Parameters": "Parameters",
    "IndependentVariables": "Independent Variables",
    "DependentVariables": "Dependent Variables",
    "ExperimentDesign": "Experiment Design",
    "Procedure": "Procedure",
    "AnalysisProcedure": "Analysis Procedure",
    "Summary": "Summary",
    "Impact": "Impact",
    "FutureWork": "Future Work",
}


def task_of(label: str) -> str:
    for task, labels in LABELS_BY_TASK.items():
        if label in labels:
            return task
    raise KeyError(f"unknown label: {label!r}")
