"""Pydantic request and response schemas for Lecture Memory."""

from datetime import date, datetime
from typing import Literal, Self

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


class SlidePageCreate(BaseModel):
    """Validated metadata for one rendered lecture page."""

    model_config = ConfigDict(str_strip_whitespace=True)

    page_number: int = Field(gt=0)
    image_path: str = Field(min_length=1)
    text_content: str | None = None


class SlidePageRead(BaseModel):
    """Serializable representation of a rendered slide page."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    lecture_id: int
    page_number: int
    image_path: str
    text_content: str | None
    created_at: datetime


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


EmbeddingEntityType = Literal["slide_page", "note"]


class EmbeddingRecordCreate(BaseModel):
    """Validated metadata for one cached embedding vector."""

    model_config = ConfigDict(str_strip_whitespace=True)

    entity_id: int = Field(gt=0)
    entity_type: EmbeddingEntityType
    model_name: str = Field(min_length=1, max_length=255)
    dimension: int = Field(gt=0)
    embedding_path: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class EmbeddingRecordRead(BaseModel):
    """Serializable representation of cached embedding metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    entity_id: int
    entity_type: EmbeddingEntityType
    model_name: str
    dimension: int
    embedding_path: str
    content_hash: str
    created_at: datetime
