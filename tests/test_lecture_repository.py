"""CRUD and relationship tests for lectures."""

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
from app.repositories.course_repository import create_course, delete_course
from app.repositories.lecture_repository import (
    CourseNotFoundError,
    create_lecture,
    delete_lecture,
    get_lecture,
    list_course_lectures,
)
from app.schemas import CourseCreate, LectureCreate, LectureRead


@pytest.fixture
def lecture_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "lectures.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def create_test_course(session: Session, code: str = "CS344") -> int:
    course = create_course(
        session,
        CourseCreate(name=f"Course {code}", code=code),
    )
    return course.id


def test_create_lecture_with_course_relationship(
    lecture_session_factory: sessionmaker[Session],
) -> None:
    lecture_data = LectureCreate(
        title="  Divide and Conquer  ",
        lecture_number=3,
        lecture_date=date(2026, 9, 18),
        source_pdf="courses/cs344/lecture_03.pdf",
    )

    with session_scope(lecture_session_factory) as session:
        course_id = create_test_course(session)
        lecture = create_lecture(session, course_id, lecture_data)
        lecture_id = lecture.id

    with lecture_session_factory() as session:
        stored_lecture = get_lecture(session, lecture_id)
        assert stored_lecture is not None
        assert stored_lecture.course_id == course_id
        assert stored_lecture.course.id == course_id
        assert stored_lecture.title == "Divide and Conquer"
        assert stored_lecture.lecture_number == 3
        assert stored_lecture.lecture_date == date(2026, 9, 18)
        assert stored_lecture.source_pdf == "courses/cs344/lecture_03.pdf"
        assert stored_lecture.created_at is not None
        assert stored_lecture.updated_at is not None

        serialized_lecture = LectureRead.model_validate(stored_lecture)
        assert serialized_lecture.id == lecture_id


def test_list_course_lectures_filters_and_orders_results(
    lecture_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(lecture_session_factory) as session:
        first_course_id = create_test_course(session, "CS344")
        second_course_id = create_test_course(session, "MATH301")
        create_lecture(
            session,
            first_course_id,
            LectureCreate(title="Integer Multiplication", lecture_number=3),
        )
        create_lecture(
            session,
            first_course_id,
            LectureCreate(title="Algorithm Analysis", lecture_number=1),
        )
        create_lecture(
            session,
            second_course_id,
            LectureCreate(title="Convex Sets", lecture_number=1),
        )

    with lecture_session_factory() as session:
        lectures = list_course_lectures(session, first_course_id)

    assert [lecture.lecture_number for lecture in lectures] == [1, 3]
    assert [lecture.title for lecture in lectures] == ["Algorithm Analysis", "Integer Multiplication"]


def test_create_lecture_rejects_unknown_course(
    lecture_session_factory: sessionmaker[Session],
) -> None:
    with lecture_session_factory() as session:
        with pytest.raises(CourseNotFoundError, match="Course 999 does not exist"):
            create_lecture(
                session,
                999,
                LectureCreate(title="Orphan Lecture", lecture_number=1),
            )


def test_get_lecture_returns_none_for_unknown_id(
    lecture_session_factory: sessionmaker[Session],
) -> None:
    with lecture_session_factory() as session:
        assert get_lecture(session, 999) is None


def test_delete_lecture(lecture_session_factory: sessionmaker[Session]) -> None:
    with session_scope(lecture_session_factory) as session:
        course_id = create_test_course(session)
        lecture = create_lecture(
            session,
            course_id,
            LectureCreate(title="Delete Me", lecture_number=1),
        )
        lecture_id = lecture.id

    with session_scope(lecture_session_factory) as session:
        assert delete_lecture(session, lecture_id) is True
        assert delete_lecture(session, lecture_id) is False

    with lecture_session_factory() as session:
        assert get_lecture(session, lecture_id) is None


def test_deleting_course_cascades_to_lectures(
    lecture_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(lecture_session_factory) as session:
        course_id = create_test_course(session)
        lecture = create_lecture(
            session,
            course_id,
            LectureCreate(title="Cascade Me", lecture_number=1),
        )
        lecture_id = lecture.id

    with session_scope(lecture_session_factory) as session:
        assert delete_course(session, course_id) is True

    with lecture_session_factory() as session:
        assert get_lecture(session, lecture_id) is None


@pytest.mark.parametrize(
    ("title", "lecture_number"),
    [("   ", 1), ("Valid title", 0), ("Valid title", -1)],
)
def test_lecture_requires_title_and_positive_number(
    title: str,
    lecture_number: int,
) -> None:
    with pytest.raises(ValidationError):
        LectureCreate(title=title, lecture_number=lecture_number)
