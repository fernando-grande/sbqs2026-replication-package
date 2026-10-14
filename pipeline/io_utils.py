"""CSV and workbook helpers.

The pipeline deliberately depends only on the standard library plus
``openpyxl``, so that a reviewer can regenerate the whole dataset without
installing the training stack. Only stage 04 needs an extra package.
"""
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Excerpts are long; the default field limit can be exceeded by Summary cells.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

ENCODING = "utf-8-sig"


def read_sheet(path: Path, sheet: str) -> list[tuple]:
    """Return every populated row of a worksheet, header included."""
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet]
        rows = [row for row in worksheet.iter_rows(values_only=True)]
    finally:
        workbook.close()
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=ENCODING, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding=ENCODING, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    import json

    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, record: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def content_sha256(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF.

    Git stores these files with LF and checks them out with CRLF on Windows,
    so hashing raw bytes would give a different value per operating system.
    """
    import hashlib

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_files(root: Path) -> list[Path]:
    """Source files that determine a training run: every .py file of the
    package and the requirements files (documentation and data excluded)."""
    files = list(root.glob("*.py")) + list(root.glob("requirements-*.txt"))
    for folder in ("analysis", "extract", "pipeline", "train"):
        files += [p for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts]
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def code_sha256(root: Path) -> str:
    """Fingerprint of the code: SHA-256 over the relative path and the
    LF-normalised content hash of every file in ``code_files``. Recorded in
    every run_info.json and checked by verify.py, so a reader can confirm
    that the published runs were produced by the code in the package."""
    import hashlib

    digest = hashlib.sha256()
    for path in code_files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(content_sha256(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def banner(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)
