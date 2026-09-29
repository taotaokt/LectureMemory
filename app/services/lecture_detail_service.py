"""Application operations for the lecture detail workspace."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Course, Lecture, Note
from app.repositories.concept_repository import list_lecture_concepts
from app.repositories.errors import LectureNotFoundError, NoteNotFoundError
from app.repositories.note_repository import (
    create_note,
    edit_note,
    get_note,
    list_lecture_notes,
)
from app.repositories.slide_page_repository import list_lecture_pages
from app.schemas import (
    CourseSummary,
    LectureConceptDisplay,
    LectureSummary,
    LectureWorkspace,
    NoteCreate,
    NoteDisplay,
    NoteUpdate,
    SlidePageDisplay,
)


def get_lecture_workspace(session: Session, lecture_id: int) -> LectureWorkspace:
    """Return the complete read projection used by a lecture detail page."""
    lecture = session.get(Lecture, lecture_id)
    if lecture is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    course = session.get(Course, lecture.course_id)
    if course is None:  # pragma: no cover - protected by the foreign key
        raise LectureNotFoundError(f"Lecture {lecture_id} has no course")

    slides = list_lecture_pages(session, lecture_id)
    notes = list_lecture_notes(session, lecture_id)
    concepts = tuple(
        LectureConceptDisplay(
            name=association.concept.name,
            normalized_name=association.concept.normalized_name,
            confidence=association.confidence,
            source=association.source,
        )
        for association in list_lecture_concepts(session, lecture_id)
    )
    lecture_count = session.scalar(
        select(func.count(Lecture.id)).where(Lecture.course_id == course.id)
    )

    return LectureWorkspace(
        course=CourseSummary(
            id=course.id,
            name=course.name,
            code=course.code,
            description=course.description,
            lecture_count=lecture_count or 0,
        ),
        lecture=LectureSummary(
            id=lecture.id,
            course_id=lecture.course_id,
            title=lecture.title,
            lecture_number=lecture.lecture_number,
            lecture_date=lecture.lecture_date,
            source_pdf=lecture.source_pdf,
            slide_count=len(slides),
            note_count=len(notes),
        ),
        concepts=concepts,
        slides=tuple(
            SlidePageDisplay(
                id=slide.id,
                page_number=slide.page_number,
                image_path=slide.image_path,
                text_content=slide.text_content,
            )
            for slide in slides
        ),
        notes=tuple(_note_display(note) for note in notes),
    )


def create_lecture_note(
    session: Session,
    lecture_id: int,
    *,
    content: str,
    page_id: int | None = None,
) -> NoteDisplay:
    """Validate and create a note from the lecture detail form."""
    note = create_note(
        session,
        lecture_id,
        NoteCreate(content=content, page_id=page_id),
    )
    return _note_display(note)


def update_lecture_note(
    session: Session,
    lecture_id: int,
    note_id: int,
    *,
    content: str,
    page_id: int | None,
) -> NoteDisplay:
    """Update a note only when it belongs to the active lecture."""
    note = get_note(session, note_id)
    if note is None or note.lecture_id != lecture_id:
        raise NoteNotFoundError(
            f"Note {note_id} does not belong to lecture {lecture_id}"
        )

    updated = edit_note(
        session,
        note_id,
        NoteUpdate(content=content, page_id=page_id),
    )
    if updated is None:  # pragma: no cover - guarded by the lookup above
        raise NoteNotFoundError(f"Note {note_id} does not exist")
    return _note_display(updated)


def _note_display(note: Note) -> NoteDisplay:
    page = note.page if note.page_id is not None else None
    return NoteDisplay(
        id=note.id,
        content=note.content,
        page_id=note.page_id,
        page_number=page.page_number if page is not None else None,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )
