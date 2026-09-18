"""Persistence operations for rendered slide pages."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Lecture, SlidePage
from app.repositories.errors import DuplicateSlidePageError, LectureNotFoundError
from app.schemas import SlidePageCreate


def create_slide_page(
    session: Session,
    lecture_id: int,
    page_data: SlidePageCreate,
) -> SlidePage:
    """Create and flush a uniquely numbered page under an existing lecture."""
    if session.get(Lecture, lecture_id) is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    existing_page_id = session.scalar(
        select(SlidePage.id).where(
            SlidePage.lecture_id == lecture_id,
            SlidePage.page_number == page_data.page_number,
        )
    )
    if existing_page_id is not None:
        raise DuplicateSlidePageError(
            f"Lecture {lecture_id} already has page {page_data.page_number}"
        )

    page = SlidePage(lecture_id=lecture_id, **page_data.model_dump())
    session.add(page)
    session.flush()
    return page


def get_slide_page(session: Session, page_id: int) -> SlidePage | None:
    """Return a slide page by primary key, or ``None`` when it does not exist."""
    return session.get(SlidePage, page_id)


def list_lecture_pages(session: Session, lecture_id: int) -> list[SlidePage]:
    """List one lecture's pages in page-number order."""
    statement = (
        select(SlidePage)
        .where(SlidePage.lecture_id == lecture_id)
        .order_by(SlidePage.page_number, SlidePage.id)
    )
    return list(session.scalars(statement).all())


def delete_slide_page(session: Session, page_id: int) -> bool:
    """Delete a slide page and report whether it existed."""
    page = get_slide_page(session, page_id)
    if page is None:
        return False

    session.delete(page)
    session.flush()
    return True
