"""Tests for lecture detail projections and note editing operations."""

from collections.abc import Iterator
from datetime import date
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
from app.repositories.concept_repository import (
    assign_lecture_concept,
    get_or_create_concept,
)
from app.repositories.course_repository import create_course
from app.repositories.errors import LectureNotFoundError, NoteNotFoundError
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import CourseCreate, LectureCreate, NoteCreate, SlidePageCreate
from app.services.lecture_detail_service import (
    create_lecture_note,
    get_lecture_workspace,
    update_lecture_note,
)


@pytest.fixture
def detail_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "lecture-detail.db")
    init_database(engine)
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def create_detail_fixture(factory: sessionmaker[Session]) -> tuple[int, int, int, int]:
    with session_scope(factory) as session:
        course = create_course(
            session,
            CourseCreate(name="Algorithms", code="CS344", description="Design techniques"),
        )
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(
                title="Divide and Conquer",
                lecture_number=3,
                lecture_date=date(2026, 9, 23),
                source_pdf="source.pdf",
            ),
        )
        later_page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=2,
                image_path="page-2.png",
                text_content="Master theorem",
            ),
        )
        first_page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=1,
                image_path="page-1.png",
                text_content="Karatsuba multiplication",
            ),
        )
        create_note(session, lecture.id, NoteCreate(content="General explanation"))
        attached = create_note(
            session,
            lecture.id,
            NoteCreate(content="Review this proof", page_id=later_page.id),
        )
        concept, _ = get_or_create_concept(
            session,
            name="Master Theorem",
            normalized_name="master theorem",
        )
        assign_lecture_concept(
            session,
            lecture_id=lecture.id,
            concept_id=concept.id,
            confidence=0.91,
            source="auto_extracted",
        )
        return lecture.id, first_page.id, later_page.id, attached.id


def test_get_lecture_workspace_collects_ordered_content(
    detail_factory: sessionmaker[Session],
) -> None:
    lecture_id, first_page_id, later_page_id, _ = create_detail_fixture(detail_factory)

    with detail_factory() as session:
        workspace = get_lecture_workspace(session, lecture_id)

    assert workspace.course.name == "Algorithms"
    assert workspace.course.code == "CS344"
    assert workspace.course.lecture_count == 1
    assert workspace.lecture.title == "Divide and Conquer"
    assert workspace.lecture.lecture_number == 3
    assert workspace.lecture.slide_count == 2
    assert workspace.lecture.note_count == 2
    assert [slide.id for slide in workspace.slides] == [first_page_id, later_page_id]
    assert [slide.page_number for slide in workspace.slides] == [1, 2]
    assert [note.content for note in workspace.notes] == [
        "General explanation",
        "Review this proof",
    ]
    assert workspace.notes[0].page_number is None
    assert workspace.notes[1].page_number == 2
    assert workspace.concepts[0].name == "Master Theorem"
    assert workspace.concepts[0].confidence == pytest.approx(0.91)


def test_create_and_update_lecture_note(
    detail_factory: sessionmaker[Session],
) -> None:
    lecture_id, first_page_id, _, _ = create_detail_fixture(detail_factory)

    with session_scope(detail_factory) as session:
        created = create_lecture_note(
            session,
            lecture_id,
            content="  Compare the recursive branches  ",
        )

    assert created.content == "Compare the recursive branches"
    assert created.page_number is None

    with session_scope(detail_factory) as session:
        updated = update_lecture_note(
            session,
            lecture_id,
            created.id,
            content="Compare all three branches",
            page_id=first_page_id,
        )

    assert updated.content == "Compare all three branches"
    assert updated.page_id == first_page_id
    assert updated.page_number == 1


def test_update_lecture_note_rejects_note_from_another_lecture(
    detail_factory: sessionmaker[Session],
) -> None:
    lecture_id, _, _, note_id = create_detail_fixture(detail_factory)
    with session_scope(detail_factory) as session:
        other_course = create_course(session, CourseCreate(name="Systems", code="CS416"))
        other_lecture = create_lecture(
            session,
            other_course.id,
            LectureCreate(title="Processes", lecture_number=1),
        )
        other_lecture_id = other_lecture.id

    with detail_factory() as session:
        with pytest.raises(NoteNotFoundError, match="does not belong"):
            update_lecture_note(
                session,
                other_lecture_id,
                note_id,
                content="Wrong lecture",
                page_id=None,
            )

    with detail_factory() as session:
        workspace = get_lecture_workspace(session, lecture_id)
    assert workspace.notes[1].content == "Review this proof"


def test_lecture_note_form_input_is_validated(
    detail_factory: sessionmaker[Session],
) -> None:
    lecture_id, _, _, note_id = create_detail_fixture(detail_factory)
    with detail_factory() as session:
        with pytest.raises(ValidationError):
            create_lecture_note(session, lecture_id, content="   ")
        with pytest.raises(ValidationError):
            update_lecture_note(
                session,
                lecture_id,
                note_id,
                content="   ",
                page_id=None,
            )


def test_get_lecture_workspace_rejects_unknown_lecture(
    detail_factory: sessionmaker[Session],
) -> None:
    with detail_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            get_lecture_workspace(session, 999)
