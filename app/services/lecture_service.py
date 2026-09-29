"""Application-facing lecture workspace and PDF upload operations."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.models import Course, Lecture, Note, SlidePage
from app.repositories.errors import CourseNotFoundError
from app.repositories.lecture_repository import create_lecture
from app.schemas import (
    CourseSummary,
    CourseWorkspace,
    LectureCreate,
    LectureSummary,
)
from app.services.ingestion_service import IngestionSummary, ingest_lecture_pdf


def get_course_workspace(session: Session, course_id: int) -> CourseWorkspace:
    """Return one course and ordered lecture summaries for its page."""
    course = session.get(Course, course_id)
    if course is None:
        raise CourseNotFoundError(f"Course {course_id} does not exist")

    statement = (
        select(
            Lecture,
            func.count(distinct(SlidePage.id)),
            func.count(distinct(Note.id)),
        )
        .outerjoin(SlidePage, SlidePage.lecture_id == Lecture.id)
        .outerjoin(Note, Note.lecture_id == Lecture.id)
        .where(Lecture.course_id == course_id)
        .group_by(Lecture.id)
        .order_by(Lecture.lecture_number, Lecture.id)
    )
    lectures = tuple(
        _lecture_summary(lecture, slide_count=slide_count, note_count=note_count)
        for lecture, slide_count, note_count in session.execute(statement)
    )
    return CourseWorkspace(
        course=CourseSummary(
            id=course.id,
            name=course.name,
            code=course.code,
            description=course.description,
            lecture_count=len(lectures),
        ),
        lectures=lectures,
    )


def create_course_lecture(
    session: Session,
    course_id: int,
    *,
    title: str,
    lecture_number: int,
    lecture_date: date | None = None,
) -> LectureSummary:
    """Validate and create an empty lecture for a course workspace."""
    lecture = create_lecture(
        session,
        course_id,
        LectureCreate(
            title=title,
            lecture_number=lecture_number,
            lecture_date=lecture_date,
        ),
    )
    return _lecture_summary(lecture, slide_count=0, note_count=0)


def ingest_uploaded_lecture_pdf(
    session: Session,
    lecture_id: int,
    *,
    filename: str,
    content: bytes | bytearray | memoryview,
    raw_root: str | Path,
    rendered_root: str | Path,
) -> IngestionSummary:
    """Validate an uploaded filename and ingest its bytes through the PDF pipeline."""
    cleaned_filename = filename.strip() if isinstance(filename, str) else ""
    if not cleaned_filename or Path(cleaned_filename).suffix.casefold() != ".pdf":
        raise ValueError("uploaded file must have a .pdf extension")

    payload = bytes(content)
    if not payload:
        raise ValueError("uploaded PDF must not be empty")

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="lecture-upload-", suffix=".pdf", delete=False) as file:
            file.write(payload)
            temporary_path = Path(file.name)
        return ingest_lecture_pdf(
            session,
            lecture_id,
            temporary_path,
            raw_root=raw_root,
            rendered_root=rendered_root,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _lecture_summary(
    lecture: Lecture,
    *,
    slide_count: int,
    note_count: int,
) -> LectureSummary:
    return LectureSummary(
        id=lecture.id,
        course_id=lecture.course_id,
        title=lecture.title,
        lecture_number=lecture.lecture_number,
        lecture_date=lecture.lecture_date,
        source_pdf=lecture.source_pdf,
        slide_count=slide_count,
        note_count=note_count,
    )
