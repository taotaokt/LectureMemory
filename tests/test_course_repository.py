"""CRUD tests for courses."""

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
from app.repositories.course_repository import (
    create_course,
    delete_course,
    get_course,
    list_courses,
)
from app.schemas import CourseCreate, CourseRead


@pytest.fixture
def course_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "courses.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def test_create_course(course_session_factory: sessionmaker[Session]) -> None:
    course_data = CourseCreate(
        name="  Design and Analysis of Algorithms  ",
        code="  CS344  ",
        description="Algorithm design techniques",
    )

    with session_scope(course_session_factory) as session:
        course = create_course(session, course_data)
        course_id = course.id

    with course_session_factory() as session:
        stored_course = get_course(session, course_id)
        assert stored_course is not None
        assert stored_course.name == "Design and Analysis of Algorithms"
        assert stored_course.code == "CS344"
        assert stored_course.description == "Algorithm design techniques"
        assert stored_course.created_at is not None
        assert stored_course.updated_at is not None

        serialized_course = CourseRead.model_validate(stored_course)
        assert serialized_course.id == course_id


def test_list_courses_uses_creation_order(
    course_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(course_session_factory) as session:
        create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        create_course(session, CourseCreate(name="Machine Learning", code="CS478"))

    with course_session_factory() as session:
        courses = list_courses(session)

    assert [course.code for course in courses] == ["CS344", "CS478"]


def test_get_course_returns_none_for_unknown_id(
    course_session_factory: sessionmaker[Session],
) -> None:
    with course_session_factory() as session:
        assert get_course(session, 999) is None


def test_delete_course(course_session_factory: sessionmaker[Session]) -> None:
    with session_scope(course_session_factory) as session:
        course = create_course(session, CourseCreate(name="Linear Optimization", code="OR301"))
        course_id = course.id

    with session_scope(course_session_factory) as session:
        assert delete_course(session, course_id) is True
        assert delete_course(session, course_id) is False

    with course_session_factory() as session:
        assert get_course(session, course_id) is None


@pytest.mark.parametrize("field", ["name", "code"])
def test_course_requires_non_blank_name_and_code(field: str) -> None:
    course_data = {"name": "Algorithms", "code": "CS344", field: "   "}

    with pytest.raises(ValidationError):
        CourseCreate(**course_data)
