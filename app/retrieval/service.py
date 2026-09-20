"""Unified retrieval service for indexed lecture slides and notes."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.embeddings.base import EmbeddingProvider
from app.models import Course, Lecture, Note, SlidePage
from app.repositories.errors import CourseNotFoundError, LectureNotFoundError
from app.retrieval.index import FaissVectorIndex, VectorSearchResult
from app.retrieval.reranker import Reranker
from app.schemas import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10
DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_RERANK_TOP_K = 20
DEFAULT_FINAL_TOP_K = 5
DEFAULT_PREVIEW_LENGTH = 240


class RetrievalConfigurationError(RuntimeError):
    """Raised when the query provider and index cannot be used together."""


class SearchFilterMismatchError(ValueError):
    """Raised when a lecture filter does not belong to the selected course."""


def search_lecture_memory(
    session: Session,
    query: str,
    *,
    provider: EmbeddingProvider,
    index: FaissVectorIndex,
    top_k: int = DEFAULT_TOP_K,
    course_id: int | None = None,
    lecture_id: int | None = None,
) -> tuple[SearchResult, ...]:
    """Embed a query and return normalized matches within an optional scope."""
    cleaned_query = _validate_query(query)
    _validate_top_k(top_k)
    if provider.dimension != index.dimension:
        raise RetrievalConfigurationError(
            f"Embedding provider dimension {provider.dimension} does not match "
            f"index dimension {index.dimension}"
        )
    _validate_search_scope(
        session,
        course_id=course_id,
        lecture_id=lecture_id,
    )
    if index.count == 0:
        return ()

    query_vector = provider.embed_query(cleaned_query)
    has_filter = course_id is not None or lecture_id is not None
    candidate_limit = index.count if has_filter else top_k
    candidates = index.search(query_vector, top_k=candidate_limit)
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
        if not _matches_scope(result, course_id=course_id, lecture_id=lecture_id):
            continue
        results.append(result)
        if len(results) == top_k:
            break
    return tuple(results)


def search_and_rerank(
    session: Session,
    query: str,
    *,
    provider: EmbeddingProvider,
    index: FaissVectorIndex,
    reranker: Reranker,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    course_id: int | None = None,
    lecture_id: int | None = None,
) -> tuple[SearchResult, ...]:
    """Retrieve a broad candidate set, rerank it, and return final results."""
    cleaned_query = _validate_query(query)
    _validate_pipeline_limits(
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
        final_top_k=final_top_k,
    )
    retrieved = search_lecture_memory(
        session,
        cleaned_query,
        provider=provider,
        index=index,
        top_k=retrieval_top_k,
        course_id=course_id,
        lecture_id=lecture_id,
    )
    if not retrieved:
        return ()

    rerank_candidates = retrieved[:rerank_top_k]
    results = reranker.rerank(
        cleaned_query,
        rerank_candidates,
        top_k=final_top_k,
    )
    logger.info(
        "Completed retrieval and reranking",
        extra={
            "retrieved_count": len(retrieved),
            "reranked_count": len(rerank_candidates),
            "returned_count": len(results),
        },
    )
    return results


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


def _validate_pipeline_limits(
    *,
    retrieval_top_k: int,
    rerank_top_k: int,
    final_top_k: int,
) -> None:
    for field_name, value in (
        ("retrieval_top_k", retrieval_top_k),
        ("rerank_top_k", rerank_top_k),
        ("final_top_k", final_top_k),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
    if rerank_top_k > retrieval_top_k:
        raise ValueError("rerank_top_k must not exceed retrieval_top_k")
    if final_top_k > rerank_top_k:
        raise ValueError("final_top_k must not exceed rerank_top_k")


def _validate_search_scope(
    session: Session,
    *,
    course_id: int | None,
    lecture_id: int | None,
) -> None:
    _validate_optional_id(course_id, field_name="course_id")
    _validate_optional_id(lecture_id, field_name="lecture_id")

    course = session.get(Course, course_id) if course_id is not None else None
    if course_id is not None and course is None:
        raise CourseNotFoundError(f"Course {course_id} does not exist")

    lecture = session.get(Lecture, lecture_id) if lecture_id is not None else None
    if lecture_id is not None and lecture is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")
    if course_id is not None and lecture is not None and lecture.course_id != course_id:
        raise SearchFilterMismatchError(
            f"Lecture {lecture.id} belongs to course {lecture.course_id}, not {course_id}"
        )


def _validate_optional_id(value: int | None, *, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer or None")


def _matches_scope(
    result: SearchResult,
    *,
    course_id: int | None,
    lecture_id: int | None,
) -> bool:
    if course_id is not None and result.course_id != course_id:
        return False
    return lecture_id is None or result.lecture_id == lecture_id
