"""Validated JSONL dataset primitives for retrieval benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Course, Lecture, SlidePage

BenchmarkLanguage = Literal["en", "zh"]
BenchmarkQueryType = Literal["semantic", "visual", "note"]
BenchmarkDifficulty = Literal["standard", "paraphrase", "confusable", "hard"]


class BenchmarkDatasetError(ValueError):
    """Raised when benchmark data or its referenced materials are invalid."""


class RelevantPage(BaseModel):
    """One relevant PDF page, optionally pinned to a local lecture ID."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(gt=0)
    lecture_id: int | None = Field(default=None, gt=0)


class BenchmarkQuery(BaseModel):
    """One realistic query and its human-labeled relevant pages."""

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    id: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    query: str = Field(min_length=3)
    course: str = Field(min_length=1)
    lecture: str = Field(min_length=1)
    source_file: str = Field(min_length=5)
    relevant_pages: tuple[RelevantPage, ...] = Field(min_length=1)
    language: BenchmarkLanguage
    query_type: BenchmarkQueryType
    difficulty: BenchmarkDifficulty

    @field_validator("source_file")
    @classmethod
    def require_pdf_filename(cls, value: str) -> str:
        path = Path(value)
        if path.name != value or path.suffix.casefold() != ".pdf":
            raise ValueError("source_file must be a plain .pdf filename")
        return value

    @field_validator("relevant_pages")
    @classmethod
    def require_unique_pages(
        cls,
        pages: tuple[RelevantPage, ...],
    ) -> tuple[RelevantPage, ...]:
        keys = [(page.lecture_id, page.page_number) for page in pages]
        if len(keys) != len(set(keys)):
            raise ValueError("relevant_pages must not contain duplicates")
        return pages


@dataclass(frozen=True, slots=True)
class BenchmarkSource:
    """Validated local source PDF metadata."""

    course: str
    source_file: str
    path: Path
    page_count: int
    query_count: int


@dataclass(frozen=True, slots=True)
class ResolvedRelevantPage:
    """A portable benchmark label resolved to local database identifiers."""

    query_id: str
    lecture_id: int
    page_id: int
    page_number: int


def load_benchmark_queries(path: str | Path) -> tuple[BenchmarkQuery, ...]:
    """Load strict JSONL records and reject duplicate IDs or query text."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise BenchmarkDatasetError(f"Benchmark dataset does not exist: {source}")

    queries: list[BenchmarkQuery] = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            raise BenchmarkDatasetError(f"Line {line_number} is blank")
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BenchmarkDatasetError(
                f"Line {line_number} is not valid JSON: {exc.msg}"
            ) from exc
        try:
            queries.append(BenchmarkQuery.model_validate(payload))
        except ValidationError as exc:
            raise BenchmarkDatasetError(
                f"Line {line_number} does not match the benchmark schema: {exc}"
            ) from exc

    if not queries:
        raise BenchmarkDatasetError("Benchmark dataset must contain at least one query")
    _require_unique_values(queries, field="id")
    _require_unique_values(queries, field="query", casefold=True)
    return tuple(queries)


def validate_source_pdfs(
    queries: tuple[BenchmarkQuery, ...],
    raw_root: str | Path,
) -> tuple[BenchmarkSource, ...]:
    """Validate that every portable label points to an existing PDF page."""
    root = Path(raw_root).expanduser().resolve()
    grouped: dict[tuple[str, str], list[BenchmarkQuery]] = {}
    for query in queries:
        grouped.setdefault((query.course, query.source_file), []).append(query)

    sources: list[BenchmarkSource] = []
    for (course, source_file), source_queries in sorted(grouped.items()):
        path = root / course / source_file
        if not path.is_file():
            raise BenchmarkDatasetError(f"Benchmark source PDF does not exist: {path}")
        try:
            document = pymupdf.open(path)
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise BenchmarkDatasetError(f"Could not open benchmark source PDF: {path}") from exc
        with document:
            if document.needs_pass:
                raise BenchmarkDatasetError(f"Benchmark source PDF is password protected: {path}")
            page_count = document.page_count

        for query in source_queries:
            for relevant_page in query.relevant_pages:
                if relevant_page.page_number > page_count:
                    raise BenchmarkDatasetError(
                        f"Query {query.id} references page {relevant_page.page_number}, "
                        f"but {path.name} has {page_count} pages"
                    )
        sources.append(
            BenchmarkSource(
                course=course,
                source_file=source_file,
                path=path,
                page_count=page_count,
                query_count=len(source_queries),
            )
        )
    return tuple(sources)


def resolve_database_references(
    session: Session,
    queries: tuple[BenchmarkQuery, ...],
) -> tuple[ResolvedRelevantPage, ...]:
    """Resolve stable course/lecture/page labels against an ingested database."""
    resolved: list[ResolvedRelevantPage] = []
    lecture_cache: dict[tuple[str, str], Lecture] = {}
    for query in queries:
        key = (query.course, query.lecture)
        lecture = lecture_cache.get(key)
        if lecture is None:
            lectures = tuple(
                session.scalars(
                    select(Lecture)
                    .join(Course)
                    .where(Course.code == query.course, Lecture.title == query.lecture)
                    .order_by(Lecture.id)
                ).all()
            )
            if len(lectures) != 1:
                raise BenchmarkDatasetError(
                    f"Query {query.id} expected one {query.course}/{query.lecture} lecture; "
                    f"found {len(lectures)}"
                )
            lecture = lectures[0]
            lecture_cache[key] = lecture

        for label in query.relevant_pages:
            if label.lecture_id is not None and label.lecture_id != lecture.id:
                raise BenchmarkDatasetError(
                    f"Query {query.id} expects lecture ID {label.lecture_id}, "
                    f"but the stable label resolved to {lecture.id}"
                )
            page = session.scalar(
                select(SlidePage).where(
                    SlidePage.lecture_id == lecture.id,
                    SlidePage.page_number == label.page_number,
                )
            )
            if page is None:
                raise BenchmarkDatasetError(
                    f"Query {query.id} references missing lecture {lecture.id} "
                    f"page {label.page_number}"
                )
            resolved.append(
                ResolvedRelevantPage(
                    query_id=query.id,
                    lecture_id=lecture.id,
                    page_id=page.id,
                    page_number=page.page_number,
                )
            )
    return tuple(resolved)


def _require_unique_values(
    queries: list[BenchmarkQuery],
    *,
    field: Literal["id", "query"],
    casefold: bool = False,
) -> None:
    seen: set[str] = set()
    for query in queries:
        value = getattr(query, field)
        key = value.casefold() if casefold else value
        if key in seen:
            raise BenchmarkDatasetError(f"Duplicate benchmark {field}: {value}")
        seen.add(key)
