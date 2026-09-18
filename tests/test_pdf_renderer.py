"""Tests for rendering lecture PDF pages to PNG files."""

from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.config import get_settings
from app.ingestion.pdf_renderer import PDFRenderError, render_pdf_pages


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample_lecture.pdf"
    document = canvas.Canvas(str(pdf_path), pagesize=letter)
    document.setTitle("Lecture Memory Renderer Test")

    document.setFont("Helvetica-Bold", 24)
    document.drawString(72, 700, "Lecture 03: Divide and Conquer")
    document.setFont("Helvetica", 14)
    document.drawString(72, 660, "Page one verifies stable rendering.")
    document.showPage()

    document.setFont("Helvetica-Bold", 24)
    document.drawString(72, 700, "Karatsuba Multiplication")
    document.setFont("Helvetica", 14)
    document.drawString(72, 660, "Three recursive products replace four.")
    document.save()
    return pdf_path


def test_render_pdf_pages_uses_stable_names_and_dimensions(
    sample_pdf: Path,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "rendered"

    rendered_pages = render_pdf_pages(
        sample_pdf,
        course_id=7,
        lecture_id=3,
        output_root=output_root,
        dpi=180,
    )

    expected_directory = output_root / "course_7/lecture_3"
    assert rendered_pages == [
        expected_directory / "page_0001.png",
        expected_directory / "page_0002.png",
    ]
    assert all(page_path.is_file() for page_path in rendered_pages)

    first_page = pymupdf.Pixmap(rendered_pages[0])
    assert (first_page.width, first_page.height) == (1530, 1980)
    assert first_page.alpha == 0


def test_render_pdf_pages_uses_configured_output_directory(
    sample_pdf: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    rendered_root = tmp_path / "configured-rendered"
    monkeypatch.setenv("RENDERED_DIR", str(rendered_root))
    get_settings.cache_clear()

    rendered_pages = render_pdf_pages(sample_pdf, course_id=1, lecture_id=2)

    assert rendered_pages[0].parent == rendered_root / "course_1/lecture_2"
    get_settings.cache_clear()


def test_rerender_replaces_existing_page(sample_pdf: Path, tmp_path: Path) -> None:
    output_root = tmp_path / "rendered"
    rendered_pages = render_pdf_pages(sample_pdf, 1, 1, output_root=output_root)
    rendered_pages[0].write_bytes(b"not a png")

    rerendered_pages = render_pdf_pages(sample_pdf, 1, 1, output_root=output_root)

    assert rerendered_pages == rendered_pages
    assert rerendered_pages[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_missing_pdf_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF does not exist"):
        render_pdf_pages(tmp_path / "missing.pdf", 1, 1, output_root=tmp_path)


def test_invalid_pdf_is_rejected(tmp_path: Path) -> None:
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("this is not a PDF", encoding="utf-8")

    with pytest.raises(PDFRenderError, match="Unable to open PDF"):
        render_pdf_pages(invalid_pdf, 1, 1, output_root=tmp_path / "rendered")


@pytest.mark.parametrize(
    ("course_id", "lecture_id"),
    [(0, 1), (-1, 1), (1, 0), (1, -1), (True, 1)],
)
def test_invalid_identifiers_are_rejected(
    sample_pdf: Path,
    tmp_path: Path,
    course_id: int,
    lecture_id: int,
) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        render_pdf_pages(
            sample_pdf,
            course_id,
            lecture_id,
            output_root=tmp_path / "rendered",
        )


@pytest.mark.parametrize("dpi", [71, 601, 180.5, True])
def test_invalid_dpi_is_rejected(sample_pdf: Path, tmp_path: Path, dpi: object) -> None:
    with pytest.raises(ValueError, match="dpi must be an integer"):
        render_pdf_pages(
            sample_pdf,
            1,
            1,
            output_root=tmp_path / "rendered",
            dpi=dpi,  # type: ignore[arg-type]
        )


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    yield
    get_settings.cache_clear()
