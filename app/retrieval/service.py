"""Unified retrieval service for indexed lecture slides and notes."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.embeddings.base import EmbeddingProvider
from app.models import Lecture, Note, SlidePage
from app.retrieval.index import FaissVectorIndex, VectorSearchResult
from app.schemas import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
DEFAULT_PREVIEW_LENGTH = 240


class RetrievalConfigurationError(RuntimeError):
    """Raised when the query provider and index cannot be used together."""


def search_lecture_memory(
    session: Session,
    query: str,
    *,
    provider: EmbeddingProvider,
    index: FaissVectorIndex,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[SearchResult, ...]:
    """Embed a query and return normalized slide and note matches."""
    cleaned_query = _validate_query(query)
    _validate_top_k(top_k)
    if provider.dimension != index.dimension:
        raise RetrievalConfigurationError(
            f"Embedding provider dimension {provider.dimension} does not match "
            f"index dimension {index.dimension}"
        )
    if index.count == 0:
        return ()

    query_vector = provider.embed_query(cleaned_query)
    candidates = index.search(query_vector, top_k=top_k)
    results: list[SearchResult] = []
    for candidate in candidates:
        result = _resolve_candidate(session, candidate, rank=len(results) + 1)
        if result is None:
            logger.warning(
                "Skipping stale vector-index entity",
                extra={
                    "entity_type": candidate.entity_type,
                    "entity_id": candidate.entity_id,
                    "candidate_rank": candidate.rank,
                },
            )
            continue
        results.append(result)
    return tuple(results)


def _resolve_candidate(
    session: Session,
    candidate: VectorSearchResult,
    *,
    rank: int,
) -> SearchResult | None:
    if candidate.entity_type == "slide_page":
        return _resolve_slide_result(session, candidate, rank=rank)
    return _resolve_note_result(session, candidate, rank=rank)


def _resolve_slide_result(
    session: Session,
    candidate: VectorSearchResult,
    *,
    rank: int,
) -> SearchResult | None:
    statement = (
        select(SlidePage)
        .where(SlidePage.id == candidate.entity_id)
        .options(
            joinedload(SlidePage.lecture).joinedload(Lecture.course),
            selectinload(SlidePage.notes),
        )
    )
    page = session.scalar(statement)
    if page is None:
        return None

    lecture = page.lecture
    course = lecture.course
    related_notes = tuple(
        note.content for note in sorted(page.notes, key=lambda note: (note.created_at, note.id))
    )
    return SearchResult(
        result_type="slide",
        entity_id=page.id,
        rank=rank,
        course_id=course.id,
        course_name=course.name,
        course_code=course.code,
        lecture_id=lecture.id,
        lecture_title=lecture.title,
        lecture_number=lecture.lecture_number,
        page_number=page.page_number,
        preview_path=page.image_path,
        raw_similarity=candidate.score,
        text_preview=_text_preview(page.text_content),
        related_notes=related_notes,
    )


def _resolve_note_result(
    session: Session,
    candidate: VectorSearchResult,
    *,
    rank: int,
) -> SearchResult | None:
    statement = (
        select(Note)
        .where(Note.id == candidate.entity_id)
        .options(
            joinedload(Note.lecture).joinedload(Lecture.course),
            joinedload(Note.page),
        )
    )
    note = session.scalar(statement)
    if note is None:
        return None

    lecture = note.lecture
    course = lecture.course
    return SearchResult(
        result_type="note",
        entity_id=note.id,
        rank=rank,
        course_id=course.id,
        course_name=course.name,
        course_code=course.code,
        lecture_id=lecture.id,
        lecture_title=lecture.title,
        lecture_number=lecture.lecture_number,
        page_number=note.page.page_number if note.page is not None else None,
        preview_path=note.page.image_path if note.page is not None else None,
        raw_similarity=candidate.score,
        text_preview=_text_preview(note.content),
    )


def _text_preview(content: str | None) -> str | None:
    if content is None:
        return None
    compact = " ".join(content.split())
    if not compact:
        return None
    if len(compact) <= DEFAULT_PREVIEW_LENGTH:
        return compact
    return f"{compact[: DEFAULT_PREVIEW_LENGTH - 1].rstrip()}…"


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be blank")
    return cleaned_query


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
