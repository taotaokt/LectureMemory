"""Application-facing course operations and read projections."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Course, Lecture
from app.repositories.course_repository import create_course
from app.schemas import CourseCreate, CourseSummary


def list_course_summaries(session: Session) -> tuple[CourseSummary, ...]:
    """Return courses with lecture counts in stable creation order."""
    statement = (
        select(Course, func.count(Lecture.id))
        .outerjoin(Lecture, Lecture.course_id == Course.id)
        .group_by(Course.id)
        .order_by(Course.id)
    )
    return tuple(
        CourseSummary(
            id=course.id,
            name=course.name,
            code=course.code,
            description=course.description,
            lecture_count=lecture_count,
        )
        for course, lecture_count in session.execute(statement)
    )


def create_course_from_input(
    session: Session,
    *,
    name: str,
    code: str,
    description: str | None = None,
) -> CourseSummary:
    """Validate and persist one course from user-facing form values."""
    course = create_course(
        session,
        CourseCreate(
            name=name,
            code=code,
            description=description or None,
        ),
    )
    return CourseSummary(
        id=course.id,
        name=course.name,
        code=course.code,
        description=course.description,
        lecture_count=0,
    )
