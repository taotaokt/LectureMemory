"""CRUD and relationship tests for personal lecture notes."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.repositories.course_repository import create_course
from app.repositories.lecture_repository import create_lecture, delete_lecture
from app.repositories.note_repository import (
    LectureNotFoundError,
    create_note,
    delete_note,
    edit_note,
    get_note,
    list_lecture_notes,
)
from app.schemas import (
    CourseCreate,
    LectureCreate,
    NoteCreate,
    NoteRead,
    NoteUpdate,
)


@pytest.fixture
def note_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "notes.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def create_test_lecture(session: Session, lecture_number: int = 1) -> int:
    course = create_course(
        session,
        CourseCreate(name="Algorithms", code="CS344"),
    )
    lecture = create_lecture(
        session,
        course.id,
        LectureCreate(title=f"Lecture {lecture_number}", lecture_number=lecture_number),
    )
    return lecture.id


def test_create_lecture_note(note_session_factory: sessionmaker[Session]) -> None:
    with session_scope(note_session_factory) as session:
        lecture_id = create_test_lecture(session)
        note = create_note(
            session,
            lecture_id,
            NoteCreate(content="  Why do three recursive calls improve complexity?  "),
        )
        note_id = note.id

    with note_session_factory() as session:
        stored_note = get_note(session, note_id)
        assert stored_note is not None
        assert stored_note.lecture_id == lecture_id
        assert stored_note.lecture.id == lecture_id
        assert stored_note.page_id is None
        assert stored_note.content == "Why do three recursive calls improve complexity?"
        assert stored_note.created_at is not None
        assert stored_note.updated_at is not None

        serialized_note = NoteRead.model_validate(stored_note)
        assert serialized_note.id == note_id


def test_create_note_with_optional_page_id(
    note_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(note_session_factory) as session:
        lecture_id = create_test_lecture(session)
        note = create_note(
            session,
            lecture_id,
            NoteCreate(content="Important diagram", page_id=17),
        )
        note_id = note.id

    with note_session_factory() as session:
        stored_note = get_note(session, note_id)
        assert stored_note is not None
        assert stored_note.page_id == 17


def test_list_lecture_notes_filters_and_orders_results(
    note_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(note_session_factory) as session:
        first_lecture_id = create_test_lecture(session, 1)
        other_course = create_course(
            session,
            CourseCreate(name="Optimization", code="OR301"),
        )
        second_lecture = create_lecture(
            session,
            other_course.id,
            LectureCreate(title="Convex Sets", lecture_number=1),
        )
        create_note(session, first_lecture_id, NoteCreate(content="First note"))
        create_note(session, first_lecture_id, NoteCreate(content="Second note"))
        create_note(session, second_lecture.id, NoteCreate(content="Other lecture"))

    with note_session_factory() as session:
        notes = list_lecture_notes(session, first_lecture_id)

    assert [note.content for note in notes] == ["First note", "Second note"]


def test_edit_note_content_and_page_association(
    note_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(note_session_factory) as session:
        lecture_id = create_test_lecture(session)
        note = create_note(
            session,
            lecture_id,
            NoteCreate(content="Draft", page_id=4),
        )
        note_id = note.id

    with session_scope(note_session_factory) as session:
        edited_note = edit_note(
            session,
            note_id,
            NoteUpdate(content="Final explanation", page_id=None),
        )
        assert edited_note is not None
        assert edited_note.content == "Final explanation"
        assert edited_note.page_id is None

    with note_session_factory() as session:
        stored_note = get_note(session, note_id)
        assert stored_note is not None
        assert stored_note.content == "Final explanation"
        assert stored_note.page_id is None


def test_edit_note_returns_none_for_unknown_id(
    note_session_factory: sessionmaker[Session],
) -> None:
    with note_session_factory() as session:
        assert edit_note(session, 999, NoteUpdate(content="Missing")) is None


def test_delete_note(note_session_factory: sessionmaker[Session]) -> None:
    with session_scope(note_session_factory) as session:
        lecture_id = create_test_lecture(session)
        note = create_note(session, lecture_id, NoteCreate(content="Delete me"))
        note_id = note.id

    with session_scope(note_session_factory) as session:
        assert delete_note(session, note_id) is True
        assert delete_note(session, note_id) is False

    with note_session_factory() as session:
        assert get_note(session, note_id) is None


def test_deleting_lecture_cascades_to_notes(
    note_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(note_session_factory) as session:
        lecture_id = create_test_lecture(session)
        note = create_note(session, lecture_id, NoteCreate(content="Cascade me"))
        note_id = note.id

    with session_scope(note_session_factory) as session:
        assert delete_lecture(session, lecture_id) is True

    with note_session_factory() as session:
        assert get_note(session, note_id) is None


def test_create_note_rejects_unknown_lecture(
    note_session_factory: sessionmaker[Session],
) -> None:
    with note_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            create_note(session, 999, NoteCreate(content="Orphan note"))


@pytest.mark.parametrize(
    "note_data",
    [
        {"content": "   "},
        {"content": "Valid", "page_id": 0},
        {"content": "Valid", "page_id": -1},
    ],
)
def test_note_creation_validates_content_and_page_id(note_data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NoteCreate(**note_data)


@pytest.mark.parametrize(
    "note_data",
    [{}, {"content": None}, {"content": "   "}, {"page_id": 0}],
)
def test_note_update_requires_valid_change(note_data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NoteUpdate(**note_data)
