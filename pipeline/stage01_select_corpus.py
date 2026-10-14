"""Stage 01 - corpus selection.

Reads the study-level metadata from the source spreadsheet, applies the
versioned exclusion list, and emits one row per primary study with an
``included`` flag. Nothing is deleted: excluded studies stay in the output so
that the selection is auditable.
"""
from . import config, io_utils, schema, textnorm

STAGE = "01-select-corpus"


def run() -> dict:
    config.ensure_dirs()

    rows = io_utils.read_sheet(config.SOURCE_WORKBOOK, config.SOURCE_SHEET)
    header, body = rows[0], rows[1:]

    # Trailing blank rows are common in hand-maintained sheets; a populated
    # study is one that carries an ID.
    studies = [row for row in body if row[schema.COL_STUDY_ID] is not None]

    excluded = {}
    if config.EXCLUDED_STUDIES.exists():
        for record in io_utils.read_csv(config.EXCLUDED_STUDIES):
            excluded[int(record["study_id"])] = record

    out = []
    for row in studies:
        study_id = int(row[schema.COL_STUDY_ID])
        exclusion = excluded.get(study_id)
        out.append(
            {
                "study_id": study_id,
                "title": textnorm.normalise(row[schema.COL_TITLE]),
                "year": int(row[schema.COL_YEAR]) if row[schema.COL_YEAR] else "",
                "venue": textnorm.normalise(row[schema.COL_VENUE]),
                "included": "no" if exclusion else "yes",
                "exclusion_reason": exclusion["reason"] if exclusion else "",
            }
        )

    target = config.CORPUS_DIR / "studies.csv"
    io_utils.write_csv(
        target,
        ["study_id", "title", "year", "venue", "included", "exclusion_reason"],
        out,
    )

    included = sum(1 for row in out if row["included"] == "yes")
    unknown = sorted(set(excluded) - {row["study_id"] for row in out})

    io_utils.banner(STAGE, f"source sheet: {len(header)} columns, {len(studies)} studies")
    io_utils.banner(STAGE, f"excluded: {len(studies) - included}")
    io_utils.banner(STAGE, f"corpus size: {included} studies -> {target.name}")
    if unknown:
        io_utils.banner(
            STAGE, f"WARNING exclusion list mentions unknown study ids: {unknown}"
        )

    return {"total": len(studies), "included": included}


if __name__ == "__main__":
    run()
