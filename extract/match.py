"""Match PDF filenames to catalog titles.

Uses the same aggressive alphanumeric folding as extrai_jsons/extrai.py so that
"Hierarchical vs. constrained styles.pdf" still hits the spreadsheet title.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def fold(text: str) -> str:
    value = str(text).lower()
    # Only a real file extension is removed: titles such as
    # "Hierarchical vs. constrained styles" contain a dot of their own.
    if value.endswith(".pdf"):
        value = value[: -len(".pdf")]
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = value.replace("&", " and ")
    value = re.sub(r"\bversus\b", "vs", value)
    value = re.sub(r"\bvs\.", "vs", value)
    return re.sub(r"[^a-z0-9]", "", value)


def list_pdfs(pdf_dir: Path) -> list[Path]:
    if not pdf_dir.exists():
        return []
    return sorted(
        path
        for path in pdf_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf" and not path.name.startswith(".")
    )


def match_pdfs(studies: list[dict], pdf_dir: Path) -> list[dict]:
    """Return one report row per catalog study plus unmatched leftover files."""
    files = list_pdfs(pdf_dir)
    file_map: dict[str, Path] = {}
    for path in files:
        key = fold(path.name)
        file_map[key] = path

    used: set[str] = set()
    rows = []
    for study in studies:
        key = fold(study["title"])
        path = file_map.get(key) if key not in used else None
        if path is None:
            for file_key, candidate in file_map.items():
                if file_key in used:
                    continue
                if key and (key in file_key or file_key in key):
                    path = candidate
                    key = file_key
                    break
        if path is not None:
            used.add(key)
            rows.append(
                {
                    "study_id": study["study_id"],
                    "title": study["title"],
                    "included": study.get("included", "yes"),
                    "status": "matched",
                    "pdf_name": path.name,
                    "pdf_path": str(path),
                }
            )
        else:
            rows.append(
                {
                    "study_id": study["study_id"],
                    "title": study["title"],
                    "included": study.get("included", "yes"),
                    "status": "missing",
                    "pdf_name": "",
                    "pdf_path": "",
                }
            )

    leftover_keys = set(file_map) - used
    for key in sorted(leftover_keys):
        path = file_map[key]
        rows.append(
            {
                "study_id": "",
                "title": "",
                "included": "",
                "status": "unmatched_file",
                "pdf_name": path.name,
                "pdf_path": str(path),
            }
        )
    return rows
