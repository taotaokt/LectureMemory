"""End-to-end tests for lecture PDF ingestion."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.ingestion.document_processor import PDFProcessingError
from app.models import Lecture
from app.repositories.course_repository import create_course
from app.repositories.errors import LectureNotFoundError
from app.repositories.lecture_repository import create_lecture
from app.repositories.slide_page_repository import list_lecture_pages
from app.schemas import CourseCreate, LectureCreate
from app.services import ingestion_service
from app.services.ingestion_service import (
    DuplicateIngestionError,
    IngestionStatus,
    ingest_lecture_pdf,
)


@pytest.fixture
def ingestion_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "ingestion.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def write_test_pdf(pdf_path: Path, page_titles: list[str]) -> Path:
    document = canvas.Canvas(str(pdf_path), pagesize=letter)
    for page_number, title in enumerate(page_titles, start=1):
        document.setFont("Helvetica-Bold", 22)
        document.drawString(72, 700, title)
        document.setFont("Helvetica", 12)
        document.drawString(72, 665, f"Lecture page {page_number}")
        document.showPage()
    document.save()
    return pdf_path


@pytest.fixture
def lecture_pdf(tmp_path: Path) -> Path:
    return write_test_pdf(
        tmp_path / "lecture.pdf",
        ["Divide and Conquer", "Karatsuba Multiplication"],
    )


def create_test_lecture(factory: sessionmaker[Session]) -> int:
    with session_scope(factory) as session:
        course = create_course(
            session,
            CourseCreate(name="Algorithms", code="CS344"),
        )
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Divide and Conquer", lecture_number=3),
        )
        return lecture.id


def test_ingest_lecture_pdf_creates_source_images_text_and_records(
    ingestion_session_factory: sessionmaker[Session],
    lecture_pdf: Path,
    tmp_path: Path,
) -> None:
    lecture_id = create_test_lecture(ingestion_session_factory)
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"

    with session_scope(ingestion_session_factory) as session:
        summary = ingest_lecture_pdf(
            session,
            lecture_id,
            lecture_pdf,
            raw_root=raw_root,
            rendered_root=rendered_root,
        )

    assert summary.status is IngestionStatus.INGESTED
    assert summary.total_pages == 2
    assert summary.rendered_pages == 2
    assert summary.created_pages == 2
    assert summary.failed_pages == 0
    assert summary.source_pdf == raw_root / "course_1/lecture_1/source.pdf"
    assert summary.source_pdf.read_bytes() == lecture_pdf.read_bytes()

    with ingestion_session_factory() as session:
        lecture = session.get(Lecture, lecture_id)
        assert lecture is not None
        assert lecture.source_pdf == str(summary.source_pdf)

        pages = list_lecture_pages(session, lecture_id)
        assert [page.page_number for page in pages] == [1, 2]
        assert all(Path(page.image_path).is_file() for page in pages)
        assert pages[0].text_content is not None
        assert "Divide and Conquer" in pages[0].text_content
        assert pages[1].text_content is not None
        assert "Karatsuba Multiplication" in pages[1].text_content


def test_ingestion_is_idempotent_for_identical_pdf(
    ingestion_session_factory: sessionmaker[Session],
    lecture_pdf: Path,
    tmp_path: Path,
) -> None:
    lecture_id = create_test_lecture(ingestion_session_factory)
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"

    with session_scope(ingestion_session_factory) as session:
        first_summary = ingest_lecture_pdf(
            session,
            lecture_id,
            lecture_pdf,
            raw_root=raw_root,
            rendered_root=rendered_root,
        )

    with ingestion_session_factory() as session:
        original_page_ids = [page.id for page in list_lecture_pages(session, lecture_id)]

    with session_scope(ingestion_session_factory) as session:
        second_summary = ingest_lecture_pdf(
            session,
            lecture_id,
            lecture_pdf,
            raw_root=raw_root,
            rendered_root=rendered_root,
        )

    assert first_summary.status is IngestionStatus.INGESTED
    assert second_summary.status is IngestionStatus.SKIPPED
    assert second_summary.created_pages == 0
    assert second_summary.rendered_pages == 2

    with ingestion_session_factory() as session:
        current_page_ids = [page.id for page in list_lecture_pages(session, lecture_id)]
    assert current_page_ids == original_page_ids


def test_different_pdf_is_rejected_after_ingestion(
    ingestion_session_factory: sessionmaker[Session],
    lecture_pdf: Path,
    tmp_path: Path,
) -> None:
    lecture_id = create_test_lecture(ingestion_session_factory)
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"
    replacement_pdf = write_test_pdf(tmp_path / "replacement.pdf", ["Different lecture"])

    with session_scope(ingestion_session_factory) as session:
        ingest_lecture_pdf(
            session,
            lecture_id,
            lecture_pdf,
            raw_root=raw_root,
            rendered_root=rendered_root,
        )

    with ingestion_session_factory() as session:
        with pytest.raises(DuplicateIngestionError, match="different or incomplete"):
            ingest_lecture_pdf(
                session,
                lecture_id,
                replacement_pdf,
                raw_root=raw_root,
                rendered_root=rendered_root,
            )


def test_incomplete_existing_ingestion_is_reported(
    ingestion_session_factory: sessionmaker[Session],
    lecture_pdf: Path,
    tmp_path: Path,
) -> None:
    lecture_id = create_test_lecture(ingestion_session_factory)
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"

    with session_scope(ingestion_session_factory) as session:
        summary = ingest_lecture_pdf(
            session,
            lecture_id,
            lecture_pdf,
            raw_root=raw_root,
            rendered_root=rendered_root,
        )

    missing_image = rendered_root / "course_1/lecture_1/page_0002.png"
    missing_image.unlink()

    with ingestion_session_factory() as session:
        with pytest.raises(DuplicateIngestionError, match="different or incomplete"):
            ingest_lecture_pdf(
                session,
                lecture_id,
                lecture_pdf,
                raw_root=raw_root,
                rendered_root=rendered_root,
            )

    assert summary.source_pdf.is_file()


def test_invalid_pdf_leaves_no_ingestion_state(
    ingestion_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id = create_test_lecture(ingestion_session_factory)
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("not a PDF", encoding="utf-8")
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"

    with ingestion_session_factory() as session:
        with pytest.raises(PDFProcessingError, match="Unable to open PDF"):
            ingest_lecture_pdf(
                session,
                lecture_id,
                invalid_pdf,
                raw_root=raw_root,
                rendered_root=rendered_root,
            )

    with ingestion_session_factory() as session:
        lecture = session.get(Lecture, lecture_id)
        assert lecture is not None
        assert lecture.source_pdf is None
        assert list_lecture_pages(session, lecture_id) == []
    assert not raw_root.exists()
    assert not rendered_root.exists()


def test_render_failure_removes_copied_source(
    ingestion_session_factory: sessionmaker[Session],
    lecture_pdf: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    lecture_id = create_test_lecture(ingestion_session_factory)
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"

    def fail_render(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(ingestion_service, "render_pdf_pages", fail_render)

    with ingestion_session_factory() as session:
        with pytest.raises(RuntimeError, match="render failed"):
            ingest_lecture_pdf(
                session,
                lecture_id,
                lecture_pdf,
                raw_root=raw_root,
                rendered_root=rendered_root,
            )

    stored_source = raw_root / "course_1/lecture_1/source.pdf"
    assert not stored_source.exists()
    with ingestion_session_factory() as session:
        assert list_lecture_pages(session, lecture_id) == []


def test_unknown_lecture_is_rejected(
    ingestion_session_factory: sessionmaker[Session],
    lecture_pdf: Path,
    tmp_path: Path,
) -> None:
    with ingestion_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            ingest_lecture_pdf(
                session,
                999,
                lecture_pdf,
                raw_root=tmp_path / "raw",
                rendered_root=tmp_path / "rendered",
            )
