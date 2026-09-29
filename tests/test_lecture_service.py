"""Tests for course workspaces, lecture creation, and uploaded PDF ingestion."""

from collections.abc import Iterator
from datetime import date
from io import BytesIO
from pathlib import Path

import pytest
from pydantic import ValidationError
from reportlab.pdfgen import canvas
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.repositories.course_repository import create_course
from app.repositories.errors import CourseNotFoundError
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import CourseCreate, LectureCreate, NoteCreate, SlidePageCreate
from app.services.lecture_service import (
    create_course_lecture,
    get_course_workspace,
    ingest_uploaded_lecture_pdf,
)


@pytest.fixture
def lecture_service_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "lecture-service.db")
    init_database(engine)
    factory = create_session_factory(engine)

    yield factory

    engine.dispose()


def create_course_fixture(factory: sessionmaker[Session]) -> tuple[int, int, int]:
    with session_scope(factory) as session:
        course = create_course(
            session,
            CourseCreate(
                name="Algorithms",
                code="CS344",
                description="Algorithm design techniques",
            ),
        )
        later = create_lecture(
            session,
            course.id,
            LectureCreate(title="Dynamic Programming", lecture_number=2),
        )
        earlier = create_lecture(
            session,
            course.id,
            LectureCreate(
                title="Divide and Conquer",
                lecture_number=1,
                lecture_date=date(2026, 9, 23),
            ),
        )
        page = create_slide_page(
            session,
            earlier.id,
            SlidePageCreate(page_number=1, image_path="page-1.png"),
        )
        create_slide_page(
            session,
            earlier.id,
            SlidePageCreate(page_number=2, image_path="page-2.png"),
        )
        create_note(
            session,
            earlier.id,
            NoteCreate(content="Attached note", page_id=page.id),
        )
        create_note(session, earlier.id, NoteCreate(content="Lecture note"))
        return course.id, earlier.id, later.id


def make_pdf_bytes(*page_texts: str) -> bytes:
    buffer = BytesIO()
    document = canvas.Canvas(buffer)
    for text in page_texts:
        document.drawString(72, 720, text)
        document.showPage()
    document.save()
    return buffer.getvalue()


def test_get_course_workspace_orders_lectures_and_counts_content(
    lecture_service_factory: sessionmaker[Session],
) -> None:
    course_id, earlier_id, later_id = create_course_fixture(lecture_service_factory)

    with lecture_service_factory() as session:
        workspace = get_course_workspace(session, course_id)

    assert workspace.course.name == "Algorithms"
    assert workspace.course.code == "CS344"
    assert workspace.course.description == "Algorithm design techniques"
    assert workspace.course.lecture_count == 2
    assert [lecture.id for lecture in workspace.lectures] == [earlier_id, later_id]
    assert workspace.lectures[0].slide_count == 2
    assert workspace.lectures[0].note_count == 2
    assert workspace.lectures[0].lecture_date == date(2026, 9, 23)
    assert workspace.lectures[1].slide_count == 0
    assert workspace.lectures[1].note_count == 0


def test_create_course_lecture_returns_empty_summary(
    lecture_service_factory: sessionmaker[Session],
) -> None:
    with session_scope(lecture_service_factory) as session:
        course = create_course(session, CourseCreate(name="Optimization", code="OR301"))
        summary = create_course_lecture(
            session,
            course.id,
            title="  Convex Sets  ",
            lecture_number=4,
            lecture_date=date(2026, 9, 24),
        )
        course_id = course.id

    assert summary.title == "Convex Sets"
    assert summary.lecture_number == 4
    assert summary.slide_count == 0
    assert summary.note_count == 0
    with lecture_service_factory() as session:
        assert get_course_workspace(session, course_id).lectures == (summary,)


def test_create_course_lecture_validates_course_and_input(
    lecture_service_factory: sessionmaker[Session],
) -> None:
    with lecture_service_factory() as session:
        with pytest.raises(CourseNotFoundError, match="Course 999 does not exist"):
            create_course_lecture(
                session,
                999,
                title="Missing",
                lecture_number=1,
            )
        with pytest.raises(ValidationError):
            create_course_lecture(
                session,
                1,
                title=" ",
                lecture_number=1,
            )


def test_ingest_uploaded_lecture_pdf_uses_existing_pipeline(
    lecture_service_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    course_id, _, lecture_id = create_course_fixture(lecture_service_factory)
    raw_root = tmp_path / "raw"
    rendered_root = tmp_path / "rendered"

    with session_scope(lecture_service_factory) as session:
        summary = ingest_uploaded_lecture_pdf(
            session,
            lecture_id,
            filename="lecture-01.PDF",
            content=make_pdf_bytes("Karatsuba", "Master theorem"),
            raw_root=raw_root,
            rendered_root=rendered_root,
        )

    assert summary.total_pages == 2
    assert summary.created_pages == 2
    assert summary.source_pdf == raw_root / f"course_{course_id}/lecture_{lecture_id}/source.pdf"
    assert summary.source_pdf.is_file()
    assert (rendered_root / f"course_{course_id}/lecture_{lecture_id}/page_0001.png").is_file()
    with lecture_service_factory() as session:
        lecture = get_course_workspace(session, course_id).lectures[1]

    assert lecture.source_pdf == str(summary.source_pdf)
    assert lecture.slide_count == 2


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("lecture.txt", b"content", r"\.pdf extension"),
        ("lecture.pdf", b"", "must not be empty"),
    ],
)
def test_ingest_uploaded_lecture_pdf_rejects_invalid_upload(
    filename: str,
    content: bytes,
    message: str,
    lecture_service_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    with lecture_service_factory() as session:
        with pytest.raises(ValueError, match=message):
            ingest_uploaded_lecture_pdf(
                session,
                1,
                filename=filename,
                content=content,
                raw_root=tmp_path / "raw",
                rendered_root=tmp_path / "rendered",
            )


def test_get_course_workspace_rejects_unknown_course(
    lecture_service_factory: sessionmaker[Session],
) -> None:
    with lecture_service_factory() as session:
        with pytest.raises(CourseNotFoundError, match="Course 999 does not exist"):
            get_course_workspace(session, 999)
