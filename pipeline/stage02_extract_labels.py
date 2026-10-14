"""Stage 02 - extraction and annotation mapping.

Walks the 22 methodological columns declared in ``schema.COLUMNS`` and turns
each populated cell into one ``(text, label)`` instance. This replaces
``extract-sheets/extract_data.py``, which had to be edited and re-run once per
class because the target column was hard-coded in the return dictionary.

The ``Variables`` column is the one exception to "one cell, one instance": it
is routed through ``variables.split`` and can yield an instance for
``IndependentVariables``, one for ``DependentVariables``, or both.
"""
from . import config, io_utils, schema, textnorm, variables

STAGE = "02-extract-labels"

FIELDS = [
    "instance_id",
    "study_id",
    "task",
    "label",
    "source_column",
    "source_header",
    "text",
]


def run() -> dict:
    config.ensure_dirs()

    studies = io_utils.read_csv(config.CORPUS_DIR / "studies.csv")
    included = {int(s["study_id"]) for s in studies if s["included"] == "yes"}

    rows = io_utils.read_sheet(config.SOURCE_WORKBOOK, config.SOURCE_SHEET)[1:]
    rows = [r for r in rows if r[schema.COL_STUDY_ID] is not None]

    instances = []
    unresolved = []
    not_reported = 0

    for row in rows:
        study_id = int(row[schema.COL_STUDY_ID])
        if study_id not in included:
            continue

        for spec in schema.COLUMNS:
            cell = row[spec.index] if spec.index < len(row) else None

            if spec.splitter == "variables":
                assigned, leftovers = variables.split(cell)
                for segment in leftovers:
                    unresolved.append(
                        {
                            "study_id": study_id,
                            "source_column": spec.index,
                            "text": segment,
                        }
                    )
                for label, text in assigned.items():
                    instances.append(_instance(study_id, spec, label, text))
                if not assigned:
                    not_reported += 1
                continue

            text = textnorm.normalise(cell)
            if not text or textnorm.is_not_reported(text):
                not_reported += 1
                continue
            instances.append(_instance(study_id, spec, spec.label, text))

    # Stable identifier: study order first, then schema order. Deterministic
    # across runs, which keeps diffs between pipeline versions readable.
    for position, instance in enumerate(instances):
        instance["instance_id"] = f"i{position:05d}"

    target = config.EXTRACTED_DIR / "instances.csv"
    io_utils.write_csv(target, FIELDS, instances)
    io_utils.write_csv(
        config.REPORTS / "02_variables_unresolved.csv",
        ["study_id", "source_column", "text"],
        unresolved,
    )

    io_utils.banner(STAGE, f"studies processed: {len(included)}")
    io_utils.banner(STAGE, f"columns walked: {len(schema.COLUMNS)}")
    io_utils.banner(STAGE, f"cells marked as not reported: {not_reported}")
    io_utils.banner(STAGE, f"unresolved Variables segments: {len(unresolved)}")
    io_utils.banner(STAGE, f"instances extracted: {len(instances)} -> {target.name}")

    return {"instances": len(instances), "unresolved": len(unresolved)}


def _instance(study_id: int, spec: schema.ColumnSpec, label: str, text: str) -> dict:
    return {
        "instance_id": "",
        "study_id": study_id,
        "task": spec.task,
        "label": label,
        "source_column": spec.index,
        "source_header": spec.header,
        "text": text,
    }


if __name__ == "__main__":
    run()
