"""Persistence operations for lectures."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Course, Lecture
from app.repositories.errors import CourseNotFoundError
from app.schemas import LectureCreate


def create_lecture(
    session: Session,
    course_id: int,
    lecture_data: LectureCreate,
) -> Lecture:
    """Create and flush a lecture under an existing course."""
    if session.get(Course, course_id) is None:
        raise CourseNotFoundError(f"Course {course_id} does not exist")

    lecture = Lecture(course_id=course_id, **lecture_data.model_dump())
    session.add(lecture)
    session.flush()
    return lecture


def list_course_lectures(session: Session, course_id: int) -> list[Lecture]:
    """List one course's lectures by lecture number and creation order."""
    statement = (
        select(Lecture)
        .where(Lecture.course_id == course_id)
        .order_by(Lecture.lecture_number, Lecture.id)
    )
    return list(session.scalars(statement).all())


def get_lecture(session: Session, lecture_id: int) -> Lecture | None:
    """Return a lecture by primary key, or ``None`` when it does not exist."""
    return session.get(Lecture, lecture_id)


def delete_lecture(session: Session, lecture_id: int) -> bool:
    """Delete a lecture in the current transaction and report whether it existed."""
    lecture = get_lecture(session, lecture_id)
    if lecture is None:
        return False

    session.delete(lecture)
    session.flush()
    return True
