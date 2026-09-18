"""Persistence operations for courses."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Course
from app.schemas import CourseCreate


def create_course(session: Session, course_data: CourseCreate) -> Course:
    """Create and flush a course in the current transaction."""
    course = Course(**course_data.model_dump())
    session.add(course)
    session.flush()
    return course


def list_courses(session: Session) -> list[Course]:
    """List courses in stable creation order."""
    statement = select(Course).order_by(Course.id)
    return list(session.scalars(statement).all())


def get_course(session: Session, course_id: int) -> Course | None:
    """Return a course by primary key, or ``None`` when it does not exist."""
    return session.get(Course, course_id)


def delete_course(session: Session, course_id: int) -> bool:
    """Delete a course in the current transaction and report whether it existed."""
    course = get_course(session, course_id)
    if course is None:
        return False

    session.delete(course)
    session.flush()
    return True
