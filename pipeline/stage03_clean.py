"""Stage 03 - cleaning and quality assurance.

By default this stage is **non-destructive**: it measures quality problems and
writes them to ``reports/`` without removing anything, so that the resulting
distribution reproduces the dataset behind the published results. The original
pipeline applied no deduplication or length filter, and silently changing that
here would make the package disagree with the paper.

Two opt-in filters are available for a stricter rebuild:

``--dedupe``
    Drop exact duplicate excerpts within the same class. Worth knowing before
    enabling it: ``Experiment Design`` loses 75 of its instances, because the
    original annotators recorded it with a small closed vocabulary ("One factor
    with two treatments", "one factor with more than two treatments", ...). The
    class behaves as a categorical field rather than free text, and duplicates
    shared across papers put the same string in both the training and the test
    fold even under document-level grouping.

``--min-length N``
    Drop excerpts shorter than ``N`` characters. ``Methods`` is the main source
    of these: fourteen studies have it recorded as a single word such as
    "Experiment" or "Experiments.".

``--text-source {auto,pdf,sheet}``
    Which text each instance carries. ``pdf`` takes the span grounded in the
    PDF by ``run_extract.py --llm`` (``data/extract/grounded_instances.csv``);
    instances whose span could not be anchored keep the spreadsheet text.
    ``sheet`` keeps the spreadsheet text. ``auto`` (default) uses ``pdf`` when
    the grounded file exists. The set of instances, and therefore the class
    distribution, is the same in both cases: only the text changes.

The distribution written here is the "before augmentation" column of the paper
table.
"""
import argparse
from collections import defaultdict

from . import config, io_utils, stats, textnorm

STAGE = "03-clean"

FIELDS = [
    "instance_id",
    "study_id",
    "task",
    "label",
    "source_column",
    "source_header",
    "text",
]


def run(dedupe: bool = False, min_length: int = 0, text_source: str = "auto") -> dict:
    config.ensure_dirs()

    instances = io_utils.read_csv(config.EXTRACTED_DIR / "instances.csv")
    instances = _apply_text_source(instances, text_source)

    by_text: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in instances:
        by_text[(row["label"], row["text"].casefold())].append(row)

    duplicate_report = []
    for (label, _), group in sorted(by_text.items(), key=lambda kv: -len(kv[1])):
        if len(group) > 1:
            duplicate_report.append(
                {
                    "label": label,
                    "occurrences": len(group),
                    "study_ids": " ".join(r["study_id"] for r in group),
                    "text": group[0]["text"],
                }
            )

    kept: list[dict] = []
    dropped: list[dict] = []
    flagged: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for row in instances:
        text = row["text"]

        if min_length and len(text) < min_length:
            dropped.append({**row, "drop_reason": f"shorter than {min_length} chars"})
            continue

        key = (row["label"], text.casefold())
        if dedupe and key in seen:
            dropped.append({**row, "drop_reason": "duplicate within class"})
            continue
        seen.add(key)

        if textnorm.has_mojibake(text):
            # Kept on purpose: repairing annotator-validated text automatically
            # would silently rewrite the ground truth.
            flagged.append({**row, "flag": "mojibake in source cell"})

        kept.append({field: row[field] for field in FIELDS})

    target = config.CLEAN_DIR / "instances.csv"
    io_utils.write_csv(target, FIELDS, kept)
    io_utils.write_csv(
        config.REPORTS / "03_duplicates.csv",
        ["label", "occurrences", "study_ids", "text"],
        duplicate_report,
    )
    io_utils.write_csv(
        config.REPORTS / "03_dropped.csv", FIELDS + ["drop_reason"], dropped
    )
    io_utils.write_csv(config.REPORTS / "03_flagged.csv", FIELDS + ["flag"], flagged)

    counts = stats.write_table(config.REPORTS / "stats_raw.csv", kept)
    report = stats.render(counts, "Distribution before augmentation")
    (config.REPORTS / "stats_raw.txt").write_text(report, encoding="utf-8")

    redundant = sum(r["occurrences"] - 1 for r in duplicate_report)
    io_utils.banner(STAGE, f"mode: dedupe={dedupe} min_length={min_length}")
    io_utils.banner(
        STAGE,
        f"repeated excerpts detected: {redundant} across {len(duplicate_report)} texts",
    )
    io_utils.banner(STAGE, f"mojibake flagged: {len(flagged)}")
    io_utils.banner(STAGE, f"dropped: {len(dropped)}")
    io_utils.banner(STAGE, f"instances kept: {len(kept)} -> {target.name}")
    print(report)

    return {"instances": len(kept), "dropped": len(dropped), "duplicates": redundant}


def _apply_text_source(instances: list[dict], text_source: str) -> list[dict]:
    """Swap in the text grounded in the PDF, matched by instance id."""
    grounded_path = config.GROUNDED_INSTANCES
    if text_source == "sheet" or (text_source == "auto" and not grounded_path.exists()):
        io_utils.banner(STAGE, "text source: spreadsheet")
        return instances
    if not grounded_path.exists():
        raise SystemExit(f"[{STAGE}] --text-source pdf needs {grounded_path}; run run_extract.py --llm first")

    grounded = {row["instance_id"]: row for row in io_utils.read_csv(grounded_path)}
    out = []
    swapped = stale = 0
    for row in instances:
        match = grounded.get(row["instance_id"])
        # Instance ids are positional; a grounded row from an older stage 02
        # run is ignored rather than attached to the wrong excerpt.
        if match is None or (match["study_id"], match["label"], match["text_sheet"]) != (
            row["study_id"],
            row["label"],
            row["text"],
        ):
            stale += match is not None
            out.append(row)
            continue
        if match["text_source"] == "pdf":
            swapped += 1
        out.append({**row, "text": match["text"]})

    io_utils.banner(STAGE, f"text source: PDF spans for {swapped}/{len(instances)} instances, spreadsheet for the rest")
    if stale:
        io_utils.banner(STAGE, f"WARNING {stale} grounded rows do not match stage 02; re-run run_extract.py")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="drop exact duplicate excerpts within a class (off by default)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="drop excerpts shorter than N characters (0 disables the filter)",
    )
    parser.add_argument(
        "--text-source",
        choices=("auto", "pdf", "sheet"),
        default="auto",
        help="text of each instance: PDF span grounded by run_extract.py, or spreadsheet (default: auto)",
    )
    args = parser.parse_args()
    run(dedupe=args.dedupe, min_length=args.min_length, text_source=args.text_source)


if __name__ == "__main__":
    main()
