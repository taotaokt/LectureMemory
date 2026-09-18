"""Persistence and relationship tests for rendered slide pages."""

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
from app.repositories.errors import DuplicateSlidePageError, LectureNotFoundError
from app.repositories.lecture_repository import create_lecture, delete_lecture
from app.repositories.note_repository import create_note, get_note
from app.repositories.slide_page_repository import (
    create_slide_page,
    delete_slide_page,
    get_slide_page,
    list_lecture_pages,
)
from app.schemas import (
    CourseCreate,
    LectureCreate,
    NoteCreate,
    SlidePageCreate,
    SlidePageRead,
)


@pytest.fixture
def slide_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "slide-pages.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def create_test_lecture(
    session: Session,
    *,
    course_code: str = "CS344",
    lecture_number: int = 1,
) -> int:
    course = create_course(
        session,
        CourseCreate(name=f"Course {course_code}", code=course_code),
    )
    lecture = create_lecture(
        session,
        course.id,
        LectureCreate(title=f"Lecture {lecture_number}", lecture_number=lecture_number),
    )
    return lecture.id


def test_create_slide_page_with_lecture_relationship(
    slide_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(slide_session_factory) as session:
        lecture_id = create_test_lecture(session)
        page = create_slide_page(
            session,
            lecture_id,
            SlidePageCreate(
                page_number=1,
                image_path="  rendered/course_1/lecture_1/page_0001.png  ",
                text_content="Convex sets are closed under convex combinations.",
            ),
        )
        page_id = page.id

    with slide_session_factory() as session:
        stored_page = get_slide_page(session, page_id)
        assert stored_page is not None
        assert stored_page.lecture_id == lecture_id
        assert stored_page.lecture.id == lecture_id
        assert stored_page.page_number == 1
        assert stored_page.image_path == "rendered/course_1/lecture_1/page_0001.png"
        assert stored_page.text_content == "Convex sets are closed under convex combinations."
        assert stored_page.created_at is not None

        serialized_page = SlidePageRead.model_validate(stored_page)
        assert serialized_page.id == page_id


def test_list_lecture_pages_filters_and_orders_results(
    slide_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(slide_session_factory) as session:
        first_lecture_id = create_test_lecture(session)
        second_lecture_id = create_test_lecture(
            session,
            course_code="MATH301",
            lecture_number=1,
        )
        create_slide_page(
            session,
            first_lecture_id,
            SlidePageCreate(page_number=3, image_path="page_0003.png"),
        )
        create_slide_page(
            session,
            first_lecture_id,
            SlidePageCreate(page_number=1, image_path="page_0001.png"),
        )
        create_slide_page(
            session,
            second_lecture_id,
            SlidePageCreate(page_number=1, image_path="other/page_0001.png"),
        )

    with slide_session_factory() as session:
        pages = list_lecture_pages(session, first_lecture_id)

    assert [page.page_number for page in pages] == [1, 3]


def test_create_slide_page_rejects_unknown_lecture(
    slide_session_factory: sessionmaker[Session],
) -> None:
    with slide_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            create_slide_page(
                session,
                999,
                SlidePageCreate(page_number=1, image_path="page_0001.png"),
            )


def test_duplicate_page_number_is_rejected(
    slide_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(slide_session_factory) as session:
        lecture_id = create_test_lecture(session)
        page_data = SlidePageCreate(page_number=1, image_path="page_0001.png")
        create_slide_page(session, lecture_id, page_data)

        with pytest.raises(DuplicateSlidePageError, match="already has page 1"):
            create_slide_page(session, lecture_id, page_data)


def test_delete_slide_page_clears_note_association(
    slide_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(slide_session_factory) as session:
        lecture_id = create_test_lecture(session)
        page = create_slide_page(
            session,
            lecture_id,
            SlidePageCreate(page_number=1, image_path="page_0001.png"),
        )
        note = create_note(
            session,
            lecture_id,
            NoteCreate(content="Keep this note", page_id=page.id),
        )
        page_id = page.id
        note_id = note.id

    with session_scope(slide_session_factory) as session:
        assert delete_slide_page(session, page_id) is True
        assert delete_slide_page(session, page_id) is False

    with slide_session_factory() as session:
        assert get_slide_page(session, page_id) is None
        stored_note = get_note(session, note_id)
        assert stored_note is not None
        assert stored_note.page_id is None
        assert stored_note.content == "Keep this note"


def test_deleting_lecture_cascades_to_slide_pages(
    slide_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(slide_session_factory) as session:
        lecture_id = create_test_lecture(session)
        page = create_slide_page(
            session,
            lecture_id,
            SlidePageCreate(page_number=1, image_path="page_0001.png"),
        )
        page_id = page.id

    with session_scope(slide_session_factory) as session:
        assert delete_lecture(session, lecture_id) is True

    with slide_session_factory() as session:
        assert get_slide_page(session, page_id) is None


@pytest.mark.parametrize(
    "page_data",
    [
        {"page_number": 0, "image_path": "page.png"},
        {"page_number": -1, "image_path": "page.png"},
        {"page_number": 1, "image_path": "   "},
    ],
)
def test_slide_page_validates_number_and_image_path(
    page_data: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SlidePageCreate(**page_data)
