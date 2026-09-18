"""Persistence operations for personal lecture notes."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Lecture, Note, SlidePage
from app.repositories.errors import (
    LectureNotFoundError,
    NotePageMismatchError,
    SlidePageNotFoundError,
)
from app.schemas import NoteCreate, NoteUpdate


def create_note(
    session: Session,
    lecture_id: int,
    note_data: NoteCreate,
) -> Note:
    """Create and flush a note under an existing lecture."""
    if session.get(Lecture, lecture_id) is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")
    _validate_page_association(session, lecture_id, note_data.page_id)

    note = Note(lecture_id=lecture_id, **note_data.model_dump())
    session.add(note)
    session.flush()
    return note


def get_note(session: Session, note_id: int) -> Note | None:
    """Return a note by primary key, or ``None`` when it does not exist."""
    return session.get(Note, note_id)


def edit_note(
    session: Session,
    note_id: int,
    note_data: NoteUpdate,
) -> Note | None:
    """Apply validated changes to a note in the current transaction."""
    note = get_note(session, note_id)
    if note is None:
        return None

    updates = note_data.model_dump(exclude_unset=True)
    if "page_id" in updates:
        _validate_page_association(session, note.lecture_id, updates["page_id"])

    for field, value in updates.items():
        setattr(note, field, value)
    session.flush()
    return note


def delete_note(session: Session, note_id: int) -> bool:
    """Delete a note in the current transaction and report whether it existed."""
    note = get_note(session, note_id)
    if note is None:
        return False

    session.delete(note)
    session.flush()
    return True


def list_lecture_notes(session: Session, lecture_id: int) -> list[Note]:
    """List one lecture's notes in stable creation order."""
    statement = (
        select(Note)
        .where(Note.lecture_id == lecture_id)
        .order_by(Note.created_at, Note.id)
    )
    return list(session.scalars(statement).all())


def _validate_page_association(
    session: Session,
    lecture_id: int,
    page_id: int | None,
) -> None:
    if page_id is None:
        return

    page = session.get(SlidePage, page_id)
    if page is None:
        raise SlidePageNotFoundError(f"Slide page {page_id} does not exist")
    if page.lecture_id != lecture_id:
        raise NotePageMismatchError(
            f"Slide page {page_id} belongs to lecture {page.lecture_id}, not {lecture_id}"
        )
