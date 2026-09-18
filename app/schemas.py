"""Pydantic request and response schemas for Lecture Memory."""

from datetime import date, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class NoteCreate(BaseModel):
    """Validated input for creating a note within a lecture."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1)
    page_id: int | None = Field(default=None, gt=0)


class NoteUpdate(BaseModel):
    """Validated changes to a note's content or slide association."""

    model_config = ConfigDict(str_strip_whitespace=True)

    content: str | None = Field(default=None, min_length=1)
    page_id: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_update(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("At least one note field must be provided")
        if "content" in self.model_fields_set and self.content is None:
            raise ValueError("Note content cannot be null")
        return self


class NoteRead(BaseModel):
    """Serializable representation of a stored note."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    lecture_id: int
    page_id: int | None
    content: str
    created_at: datetime
    updated_at: datetime
