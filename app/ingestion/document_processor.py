"""Lightweight PDF validation and text-layer extraction."""

from pathlib import Path

import pymupdf


class PDFProcessingError(RuntimeError):
    """Raised when a PDF cannot be validated or processed."""


def extract_page_texts(pdf_path: str | Path) -> list[str | None]:
    """Validate a PDF and return best-effort text for each page."""
    source_path = Path(pdf_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {source_path}")
    if not source_path.is_file():
        raise PDFProcessingError(f"PDF path is not a file: {source_path}")

    try:
        document = pymupdf.open(source_path)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise PDFProcessingError(f"Unable to open PDF: {source_path}") from exc

    with document:
        if document.needs_pass:
            raise PDFProcessingError(
                f"Password-protected PDFs are not supported: {source_path}"
            )
        if document.page_count == 0:
            raise PDFProcessingError(f"PDF contains no pages: {source_path}")

        page_texts: list[str | None] = []
        for page_number, page in enumerate(document, start=1):
            try:
                text = page.get_text("text").strip()
            except (RuntimeError, ValueError) as exc:
                raise PDFProcessingError(
                    f"Unable to extract text from page {page_number}: {source_path}"
                ) from exc
            page_texts.append(text or None)

    return page_texts
