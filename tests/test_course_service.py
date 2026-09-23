"""Tests for application-facing course summaries and creation."""

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
from app.repositories.lecture_repository import create_lecture
from app.schemas import CourseCreate, LectureCreate
from app.services.course_service import create_course_from_input, list_course_summaries


@pytest.fixture
def course_service_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "course-service.db")
    init_database(engine)
    factory = create_session_factory(engine)

    yield factory

    engine.dispose()


def test_list_course_summaries_includes_zero_and_multiple_lecture_counts(
    course_service_factory: sessionmaker[Session],
) -> None:
    with session_scope(course_service_factory) as session:
        empty_course = create_course(
            session,
            CourseCreate(name="Linear Optimization", code="OR301"),
        )
        algorithms = create_course(
            session,
            CourseCreate(
                name="Algorithms",
                code="CS344",
                description="Algorithm design techniques",
            ),
        )
        create_lecture(
            session,
            algorithms.id,
            LectureCreate(title="Divide and Conquer", lecture_number=1),
        )
        create_lecture(
            session,
            algorithms.id,
            LectureCreate(title="Dynamic Programming", lecture_number=2),
        )

    with course_service_factory() as session:
        summaries = list_course_summaries(session)

    assert [summary.id for summary in summaries] == [empty_course.id, algorithms.id]
    assert [summary.lecture_count for summary in summaries] == [0, 2]
    assert summaries[1].description == "Algorithm design techniques"


def test_create_course_from_input_validates_and_returns_summary(
    course_service_factory: sessionmaker[Session],
) -> None:
    with session_scope(course_service_factory) as session:
        summary = create_course_from_input(
            session,
            name="  Machine Learning  ",
            code="  CS446  ",
            description="  Statistical learning  ",
        )

    assert summary.name == "Machine Learning"
    assert summary.code == "CS446"
    assert summary.description == "Statistical learning"
    assert summary.lecture_count == 0

    with course_service_factory() as session:
        assert list_course_summaries(session) == (summary,)


def test_create_course_from_input_rejects_blank_required_fields(
    course_service_factory: sessionmaker[Session],
) -> None:
    with course_service_factory() as session:
        with pytest.raises(ValidationError):
            create_course_from_input(session, name=" ", code="CS344")
