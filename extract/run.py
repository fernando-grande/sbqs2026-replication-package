"""PDF extraction and grounding entry point.

    python run_extract.py                 # anchor the spreadsheet excerpts only
    python run_extract.py --llm           # ground every excerpt with Gemini
    python run_extract.py --llm --limit 3 # try the LLM on the first 3 studies
    python run_extract.py --llm --cache-only  # rebuild outputs from the cache, no API calls

Reads the included studies (stage 01), their excerpts (stage 02) and the PDFs
in corpus/pdfs/, and writes:

    data/extract/texts/{study_id}.txt       full PDF text (PyMuPDF)
    data/extract/fields/{study_id}.json     spreadsheet value, span, anchoring
    data/extract/grounded_instances.csv     stage 02 instances with the text
                                            replaced by the anchored span
    reports/pdf_match.csv                   catalog study -> PDF file
    reports/span_verification.csv           one row per instance
    reports/extract_summary.txt             counts per class and method

Does nothing harmful when corpus/pdfs/ is empty: every instance keeps its
spreadsheet text and the reports say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract import grounding
from extract.fields import PdfText, choose_text, resolve_label_conflicts
from extract.match import match_pdfs
from extract.pdf_text import pdf_to_text
from pipeline import config, io_utils, schema

STAGE = "extract"

GROUNDED_FIELDS = [
    "instance_id",
    "study_id",
    "task",
    "label",
    "source_column",
    "source_header",
    "text",
    "text_source",
    "span_method",
    "sheet_method",
    "text_sheet",
    "text_pdf",
    "conflict",
]

VERIFICATION_FIELDS = [
    "instance_id",
    "study_id",
    "task",
    "label",
    "text_source",
    "span_method",
    "sheet_method",
    "conflict",
    "sheet_chars",
    "span_chars",
]


def _load_studies() -> list[dict]:
    catalog = config.CORPUS_DIR / "studies.csv"
    if not catalog.exists():
        from pipeline.stage01_select_corpus import run as select_corpus

        select_corpus()
    return io_utils.read_csv(catalog)


def _load_instances() -> dict[str, list[dict]]:
    path = config.EXTRACTED_DIR / "instances.csv"
    if not path.exists():
        from pipeline.stage02_extract_labels import run as extract_labels

        extract_labels()
    by_study: dict[str, list[dict]] = {}
    for row in io_utils.read_csv(path):
        by_study.setdefault(row["study_id"], []).append(row)
    return by_study


def _clear(directory: Path, pattern: str) -> None:
    """Remove outputs of a previous run so no stale study survives."""
    for path in directory.glob(pattern):
        path.unlink()


def _relative(path) -> str:
    """Path inside the package, so the report does not record the local folder."""
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(config.PACKAGE_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def run(
    use_llm: bool = False,
    limit: int | None = None,
    sleep: float = config.GROUNDING_SLEEP,
    cache_only: bool = False,
    provider: str = config.GROUNDING_PROVIDER,
) -> dict:
    config.ensure_dirs()
    studies = _load_studies()
    instances = _load_instances()

    matches = match_pdfs(studies, config.PDF_DIR)
    io_utils.write_csv(
        config.REPORTS / "pdf_match.csv",
        ["study_id", "title", "included", "status", "pdf_name", "pdf_path"],
        [{**row, "pdf_path": _relative(row["pdf_path"])} for row in matches],
    )
    pdf_by_study = {
        str(row["study_id"]): row for row in matches if row["status"] == "matched"
    }

    grounder = None
    if use_llm:
        if not grounding.has_api_key(provider):
            io_utils.banner(
                STAGE,
                "ERROR --llm needs OPENROUTER_API_KEY (OpenRouter) or GOOGLE_API_KEY (Google) "
                "in .env or in the environment",
            )
            raise SystemExit(1)
        try:
            grounder = grounding.Grounder(sleep=sleep, provider=provider)
            io_utils.banner(STAGE, f"grounding model: {config.GEMINI_MODEL} via {grounder.provider}")
            grounder.offline = cache_only
        except ImportError:
            io_utils.banner(STAGE, "ERROR --llm needs the google-genai package: python -m pip install google-genai")
            raise SystemExit(1)

    _clear(config.PDF_TEXT_DIR, "*.txt")
    _clear(config.PDF_FIELDS_DIR, "*.json")

    included = [s for s in studies if s["included"] == "yes"]
    io_utils.banner(STAGE, f"included studies: {len(included)}, with PDF: {sum(str(s['study_id']) in pdf_by_study for s in included)}")

    grounded_rows: list[dict] = []
    verification: list[dict] = []
    origins: Counter = Counter()
    read_failures: list[str] = []
    grounded_studies = 0
    conflicts_reverted = 0

    for study in included:
        study_id = str(study["study_id"])
        rows = instances.get(study_id, [])
        match = pdf_by_study.get(study_id)

        pdf = None
        if match is not None:
            try:
                raw = pdf_to_text(Path(match["pdf_path"]))
            except Exception as error:  # noqa: BLE001 - a broken PDF must not stop the run
                io_utils.banner(STAGE, f"ERROR reading {match['pdf_name']}: {error}")
                read_failures.append(study_id)
            else:
                (config.PDF_TEXT_DIR / f"{study_id}.txt").write_text(raw, encoding="utf-8")
                pdf = PdfText(raw)

        spans: dict[str, str] = {}
        origin = "no-pdf" if pdf is None else "off"
        if grounder is not None and pdf is not None and (limit is None or grounded_studies < limit):
            try:
                spans, origin = grounder.ground(study_id, grounding.guide_for(rows), raw)
            except grounding.GroundingConfigError as error:
                if "daily request quota" not in str(error):
                    io_utils.banner(STAGE, f"ERROR the API rejected the request, stopping: {error}")
                    raise SystemExit(1)
                # Finish the run from the cache only, so the reports reflect
                # everything grounded so far; the next run resumes here.
                io_utils.banner(STAGE, "daily quota exhausted: remaining studies use the cache only")
                grounder.offline = True
                spans, origin = grounder.ground(study_id, grounding.guide_for(rows), raw)
            grounded_studies += 1
        origins[origin.split(":")[0]] += 1

        study_rows = []
        for row in rows:
            span = spans.get(row["label"], "")
            study_rows.append(
                {
                    **{key: row[key] for key in GROUNDED_FIELDS[:6]},
                    **choose_text(row["text"], span, pdf),
                    "text_sheet": row["text"],
                    "text_pdf": span,
                    "conflict": "",
                }
            )
        conflicts_reverted += resolve_label_conflicts(study_rows)

        fields = {}
        for row in study_rows:
            grounded_rows.append(row)
            verification.append(
                {
                    "instance_id": row["instance_id"],
                    "study_id": study_id,
                    "task": row["task"],
                    "label": row["label"],
                    "text_source": row["text_source"],
                    "span_method": row["span_method"],
                    "sheet_method": row["sheet_method"],
                    "conflict": row["conflict"],
                    "sheet_chars": len(row["text_sheet"]),
                    "span_chars": len(row["text_pdf"]),
                }
            )
            fields[row["label"]] = {
                "sheet": row["text_sheet"],
                "span": row["text_pdf"],
                "text": row["text"],
                "text_source": row["text_source"],
                "span_method": row["span_method"],
                "sheet_method": row["sheet_method"],
                "conflict": row["conflict"],
            }

        (config.PDF_FIELDS_DIR / f"{study_id}.json").write_text(
            json.dumps(
                {
                    "study_id": study_id,
                    "title": study["title"],
                    "pdf_name": match["pdf_name"] if match else "",
                    "grounding": origin,
                    "model": config.GEMINI_MODEL if grounder else "",
                    "prompt_version": config.GROUNDING_PROMPT_VERSION if grounder else "",
                    "fields": fields,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if origin in {"api", "cache"} or origin.startswith("error"):
            io_utils.banner(STAGE, f"study {study_id}: {origin}")

    io_utils.write_csv(config.GROUNDED_INSTANCES, GROUNDED_FIELDS, grounded_rows)
    io_utils.write_csv(config.REPORTS / "span_verification.csv", VERIFICATION_FIELDS, verification)

    summary = render_summary(grounded_rows, origins, read_failures, conflicts_reverted)
    (config.REPORTS / "extract_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    if grounder is not None:
        io_utils.banner(STAGE, f"Gemini requests made in this run: {grounder.calls}")
    return {
        "instances": len(grounded_rows),
        "from_pdf": sum(r["text_source"] == "pdf" for r in grounded_rows),
        "origins": dict(origins),
    }


def render_summary(rows: list[dict], origins: Counter, read_failures: list[str], conflicts_reverted: int = 0) -> str:
    lines = ["Extraction summary", "------------------"]
    lines.append("grounding per study: " + ", ".join(f"{k}={v}" for k, v in sorted(origins.items())))
    lines.append(
        f"same span under two labels of one study: {conflicts_reverted} instances reverted to the "
        "spreadsheet text (span kept by the label it resembles most)"
    )
    if read_failures:
        lines.append(f"unreadable PDFs: {' '.join(read_failures)}")
    lines.append("")

    span_methods = ["exact", "compact", "gapped", "partial", "missing", "empty", "not-grounded", "no-pdf"]
    header = f"{'class':<24}{'n':>5}{'pdf':>6}" + "".join(f"{m:>13}" for m in span_methods) + f"{'sheet in pdf':>14}"
    lines += ["span anchoring per class (text_source=pdf only when the span is exact, compact or gapped)", header]
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    for label in schema.ALL_LABELS:
        group = by_label.get(label, [])
        methods = Counter(r["span_method"] for r in group)
        sheet_in_pdf = sum(r["sheet_method"] in {"exact", "compact"} for r in group)
        lines.append(
            f"{schema.DISPLAY_NAMES[label]:<24}{len(group):>5}{sum(r['text_source'] == 'pdf' for r in group):>6}"
            + "".join(f"{methods.get(m, 0):>13}" for m in span_methods)
            + f"{sheet_in_pdf:>14}"
        )
    total_pdf = sum(r["text_source"] == "pdf" for r in rows)
    lines.append(f"{'TOTAL':<24}{len(rows):>5}{total_pdf:>6}")
    lines.append("")
    lines.append(compare_with_paper(rows))
    return "\n".join(lines)


def compare_with_paper(rows: list[dict]) -> str:
    """Per-class counts next to the "Before" column of the paper table."""
    counts = Counter(r["label"] for r in rows)
    variable_studies = {
        r["study_id"] for r in rows if r["label"] in {"IndependentVariables", "DependentVariables"}
    }
    lines = ["Comparison with the paper (Table 1, column Before)"]
    lines.append(f"{'class':<24}{'here':>6}{'paper':>7}{'diff':>6}")
    total_here = 0
    for label, paper in config.PAPER_BEFORE.items():
        if label == "Variables":
            here = len(variable_studies)
            name = "Variables (studies)"
        else:
            here = counts.get(label, 0)
            name = schema.DISPLAY_NAMES[label]
        total_here += here
        lines.append(f"{name:<24}{here:>6}{paper:>7}{here - paper:>+6}")
    lines.append(f"{'TOTAL':<24}{total_here:>6}{config.PAPER_BEFORE_TOTAL:>7}{total_here - config.PAPER_BEFORE_TOTAL:>+6}")
    lines.append(
        "(the paper counts Variables once per study, before the split into "
        "Independent/Dependent Variables; its total uses that single count)"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="ground every excerpt with Gemini (needs GOOGLE_API_KEY)")
    parser.add_argument("--limit", type=int, default=None, help="with --llm, ground only the first N studies")
    parser.add_argument(
        "--provider",
        choices=("auto", "openrouter", "google"),
        default=config.GROUNDING_PROVIDER,
        help="route to the Gemini model (default auto: OpenRouter when OPENROUTER_API_KEY is set)",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="with --llm, use cached answers only and never call the API (rebuilds the outputs for free)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=config.GROUNDING_SLEEP,
        help=f"seconds between Gemini requests (default {config.GROUNDING_SLEEP})",
    )
    args = parser.parse_args()
    run(
        use_llm=args.llm,
        limit=args.limit,
        sleep=args.sleep,
        cache_only=args.cache_only,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
