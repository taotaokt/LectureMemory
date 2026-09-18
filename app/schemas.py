"""Pydantic request and response schemas for Lecture Memory."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class CourseCreate(BaseModel):
    """Validated input for creating a course."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=255)
    code: str = Field(min_length=1, max_length=64)
    description: str | None = None


class CourseRead(BaseModel):
    """Serializable representation of a stored course."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    code: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class LectureCreate(BaseModel):
    """Validated input for creating a lecture within a course."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=255)
    lecture_number: int = Field(gt=0)
    lecture_date: date | None = None
    source_pdf: str | None = None


class LectureRead(BaseModel):
    """Serializable representation of a stored lecture."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    title: str
    lecture_number: int
    lecture_date: date | None
    source_pdf: str | None
    created_at: datetime
    updated_at: datetime
