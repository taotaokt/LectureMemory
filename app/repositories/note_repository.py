"""Persistence operations for personal lecture notes."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Lecture, Note
from app.schemas import NoteCreate, NoteUpdate


class LectureNotFoundError(ValueError):
    """Raised when a note references a lecture that does not exist."""


def create_note(
    session: Session,
    lecture_id: int,
    note_data: NoteCreate,
) -> Note:
    """Create and flush a note under an existing lecture."""
    if session.get(Lecture, lecture_id) is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

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

    for field, value in note_data.model_dump(exclude_unset=True).items():
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
