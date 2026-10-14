"""Extract plain text from a PDF (PyMuPDF, with pypdf as fallback)."""
from pathlib import Path


def pdf_to_text(pdf_path: Path) -> str:
    fitz = _import_fitz()
    if fitz is None:
        return _pdf_to_text_pypdf(pdf_path)

    document = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in document]
    finally:
        document.close()
    return "\n".join(pages)


def _import_fitz():
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            return None


def _pdf_to_text_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "Install pymupdf (preferred) or pypdf to extract text from PDFs: "
            "python -m pip install pymupdf"
        ) from error

    reader = PdfReader(str(pdf_path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
