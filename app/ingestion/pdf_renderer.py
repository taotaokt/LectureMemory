"""Render lecture PDF pages to consistently named PNG images."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pymupdf

from app.config import get_settings

DEFAULT_DPI = 180
MIN_DPI = 72
MAX_DPI = 600

logger = logging.getLogger(__name__)


class PDFRenderError(RuntimeError):
    """Raised when a PDF cannot be opened or rendered safely."""


def render_pdf_pages(
    pdf_path: str | Path,
    course_id: int,
    lecture_id: int,
    *,
    output_root: str | Path | None = None,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Render every page in ``pdf_path`` and return the generated PNG paths.

    Output files use the stable layout
    ``course_<id>/lecture_<id>/page_<number>.png``. Pages are staged in a
    temporary directory so a render failure does not leave partial new output.
    """
    source_path = _validate_source_path(pdf_path)
    _validate_identifier("course_id", course_id)
    _validate_identifier("lecture_id", lecture_id)
    _validate_dpi(dpi)

    destination_root = _resolve_output_root(output_root)
    destination = destination_root / f"course_{course_id}" / f"lecture_{lecture_id}"

    try:
        document = pymupdf.open(source_path)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise PDFRenderError(f"Unable to open PDF: {source_path}") from exc

    with document:
        if document.needs_pass:
            raise PDFRenderError(f"Password-protected PDFs are not supported: {source_path}")
        if document.page_count == 0:
            raise PDFRenderError(f"PDF contains no pages: {source_path}")

        destination.mkdir(parents=True, exist_ok=True)
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        final_paths: list[Path] = []

        with TemporaryDirectory(prefix=".render-", dir=destination) as temporary_directory:
            staging_directory = Path(temporary_directory)

            for page_index, page in enumerate(document):
                page_number = page_index + 1
                filename = f"page_{page_number:04d}.png"
                staged_path = staging_directory / filename
                final_path = destination / filename

                try:
                    pixmap = page.get_pixmap(
                        matrix=matrix,
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )
                    staged_path.write_bytes(pixmap.tobytes("png"))
                except (OSError, RuntimeError, ValueError) as exc:
                    raise PDFRenderError(
                        f"Failed to render page {page_number} from {source_path}"
                    ) from exc

                final_paths.append(final_path)

            for staged_path, final_path in zip(
                sorted(staging_directory.glob("page_*.png")),
                final_paths,
                strict=True,
            ):
                staged_path.replace(final_path)

    logger.info(
        "Rendered %s PDF pages at %s DPI to %s",
        len(final_paths),
        dpi,
        destination,
    )
    return final_paths


def _validate_source_path(pdf_path: str | Path) -> Path:
    source_path = Path(pdf_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {source_path}")
    if not source_path.is_file():
        raise PDFRenderError(f"PDF path is not a file: {source_path}")
    return source_path


def _validate_identifier(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_dpi(dpi: int) -> None:
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(f"dpi must be an integer between {MIN_DPI} and {MAX_DPI}")


def _resolve_output_root(output_root: str | Path | None) -> Path:
    if output_root is None:
        configured_root = get_settings().rendered_dir
        if configured_root is None:
            raise ValueError("RENDERED_DIR must be configured")
        return configured_root
    return Path(output_root).expanduser().resolve()
