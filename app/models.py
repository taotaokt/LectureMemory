"""SQLAlchemy domain models for Lecture Memory."""

from datetime import UTC, date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    """Return the current UTC time for model timestamp defaults."""
    return datetime.now(UTC)


class Course(Base):
    """A university course that owns a collection of lectures."""

    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
    lectures: Mapped[list["Lecture"]] = relationship(
        back_populates="course",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"Course(id={self.id!r}, code={self.code!r}, name={self.name!r})"


class Lecture(Base):
    """A single lecture belonging to a course."""

    __tablename__ = "lectures"

    id: Mapped[int] = mapped_column(primary_key=True)
    course_id: Mapped[int] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    lecture_number: Mapped[int] = mapped_column(nullable=False)
    lecture_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_pdf: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
    course: Mapped[Course] = relationship(back_populates="lectures")
    slide_pages: Mapped[list["SlidePage"]] = relationship(
        back_populates="lecture",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    notes: Mapped[list["Note"]] = relationship(
        back_populates="lecture",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"Lecture(id={self.id!r}, course_id={self.course_id!r}, "
            f"lecture_number={self.lecture_number!r}, title={self.title!r})"
        )


class SlidePage(Base):
    """A rendered page belonging to a lecture PDF."""

    __tablename__ = "slide_pages"
    __table_args__ = (
        UniqueConstraint("lecture_id", "page_number", name="uq_slide_page_lecture_number"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    lecture_id: Mapped[int] = mapped_column(
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(nullable=False)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    text_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    lecture: Mapped[Lecture] = relationship(back_populates="slide_pages")
    notes: Mapped[list["Note"]] = relationship(
        back_populates="page",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return (
            f"SlidePage(id={self.id!r}, lecture_id={self.lecture_id!r}, "
            f"page_number={self.page_number!r})"
        )


class Note(Base):
    """A personal note attached to a lecture and optionally a slide page."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    lecture_id: Mapped[int] = mapped_column(
        ForeignKey("lectures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_id: Mapped[int | None] = mapped_column(
        ForeignKey("slide_pages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )
    lecture: Mapped[Lecture] = relationship(back_populates="notes")
    page: Mapped[SlidePage | None] = relationship(back_populates="notes")

    def __repr__(self) -> str:
        return f"Note(id={self.id!r}, lecture_id={self.lecture_id!r}, page_id={self.page_id!r})"
