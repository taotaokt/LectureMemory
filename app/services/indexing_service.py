"""Build a complete persisted search index from database-backed content."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import EmbeddingCache, EmbeddingProvider
from app.embeddings.cache import hash_file, hash_text
from app.models import Note, SlidePage
from app.retrieval import FaissVectorIndex, IndexedEntity
from app.schemas import EmbeddingEntityType

logger = logging.getLogger(__name__)

IndexingStatus = Literal["generated", "cached", "failed"]


@dataclass(frozen=True, slots=True)
class IndexingFailure:
    """One entity omitted from a rebuilt index after embedding failed."""

    entity_type: EmbeddingEntityType
    entity_id: int
    lecture_id: int
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class IndexingProgress:
    """Progress event emitted after one entity has been processed."""

    processed: int
    total: int
    entity_type: EmbeddingEntityType
    entity_id: int
    status: IndexingStatus


@dataclass(frozen=True, slots=True)
class IndexBuildSummary:
    """Aggregate outcome of one full cache-and-index rebuild."""

    model_name: str
    dimension: int
    index_dir: Path
    total_slides: int
    total_notes: int
    indexed_slides: int
    indexed_notes: int
    generated_embeddings: int
    cached_embeddings: int
    failures: tuple[IndexingFailure, ...]

    @property
    def total_entities(self) -> int:
        return self.total_slides + self.total_notes

    @property
    def indexed_entities(self) -> int:
        return self.indexed_slides + self.indexed_notes

    @property
    def failed_entities(self) -> int:
        return len(self.failures)

    @property
    def complete(self) -> bool:
        return self.indexed_entities == self.total_entities and not self.failures


def build_search_index(
    session: Session,
    *,
    provider: EmbeddingProvider,
    cache: EmbeddingCache,
    index_dir: str | Path,
    progress_callback: Callable[[IndexingProgress], None] | None = None,
) -> IndexBuildSummary:
    """Embed every current slide and note, then atomically replace the index."""
    slides = tuple(
        session.scalars(
            select(SlidePage).order_by(
                SlidePage.lecture_id,
                SlidePage.page_number,
                SlidePage.id,
            )
        ).all()
    )
    notes = tuple(
        session.scalars(
            select(Note).order_by(Note.lecture_id, Note.created_at, Note.id)
        ).all()
    )

    vectors = []
    entities: list[IndexedEntity] = []
    failures: list[IndexingFailure] = []
    generated_embeddings = 0
    cached_embeddings = 0
    indexed_slides = 0
    indexed_notes = 0
    total_entities = len(slides) + len(notes)
    processed = 0

    for slide in slides:
        image_path = Path(slide.image_path).expanduser().resolve()
        try:
            content_hash = hash_file(image_path)
            with session.begin_nested():
                result = cache.get_or_compute(
                    session,
                    entity_type="slide_page",
                    entity_id=slide.id,
                    model_name=provider.model_name,
                    dimension=provider.dimension,
                    content_hash=content_hash,
                    compute=lambda image_path=image_path: provider.embed_image(image_path),
                )
        except Exception as exc:
            failures.append(
                _failure(
                    entity_type="slide_page",
                    entity_id=slide.id,
                    lecture_id=slide.lecture_id,
                    error=exc,
                )
            )
            processed += 1
            _report_progress(
                progress_callback,
                processed=processed,
                total=total_entities,
                entity_type="slide_page",
                entity_id=slide.id,
                status="failed",
            )
            continue
        vectors.append(result.vector)
        entities.append(IndexedEntity("slide_page", slide.id))
        indexed_slides += 1
        generated_embeddings += int(not result.cache_hit)
        cached_embeddings += int(result.cache_hit)
        processed += 1
        _report_progress(
            progress_callback,
            processed=processed,
            total=total_entities,
            entity_type="slide_page",
            entity_id=slide.id,
            status="cached" if result.cache_hit else "generated",
        )

    for note in notes:
        try:
            content_hash = hash_text(note.content)
            with session.begin_nested():
                result = cache.get_or_compute(
                    session,
                    entity_type="note",
                    entity_id=note.id,
                    model_name=provider.model_name,
                    dimension=provider.dimension,
                    content_hash=content_hash,
                    compute=lambda content=note.content: provider.embed_text(content),
                )
        except Exception as exc:
            failures.append(
                _failure(
                    entity_type="note",
                    entity_id=note.id,
                    lecture_id=note.lecture_id,
                    error=exc,
                )
            )
            processed += 1
            _report_progress(
                progress_callback,
                processed=processed,
                total=total_entities,
                entity_type="note",
                entity_id=note.id,
                status="failed",
            )
            continue
        vectors.append(result.vector)
        entities.append(IndexedEntity("note", note.id))
        indexed_notes += 1
        generated_embeddings += int(not result.cache_hit)
        cached_embeddings += int(result.cache_hit)
        processed += 1
        _report_progress(
            progress_callback,
            processed=processed,
            total=total_entities,
            entity_type="note",
            entity_id=note.id,
            status="cached" if result.cache_hit else "generated",
        )

    destination = Path(index_dir).expanduser().resolve()
    index = FaissVectorIndex.build(
        dimension=provider.dimension,
        vectors=vectors,
        entities=entities,
    )
    index.save(destination)

    summary = IndexBuildSummary(
        model_name=provider.model_name,
        dimension=provider.dimension,
        index_dir=destination,
        total_slides=len(slides),
        total_notes=len(notes),
        indexed_slides=indexed_slides,
        indexed_notes=indexed_notes,
        generated_embeddings=generated_embeddings,
        cached_embeddings=cached_embeddings,
        failures=tuple(failures),
    )
    logger.info(
        "Built unified lecture search index",
        extra={
            "model_name": summary.model_name,
            "dimension": summary.dimension,
            "total_entities": summary.total_entities,
            "indexed_entities": summary.indexed_entities,
            "failed_entities": summary.failed_entities,
            "generated_embeddings": summary.generated_embeddings,
            "cached_embeddings": summary.cached_embeddings,
            "index_dir": str(summary.index_dir),
        },
    )
    return summary


def _failure(
    *,
    entity_type: EmbeddingEntityType,
    entity_id: int,
    lecture_id: int,
    error: Exception,
) -> IndexingFailure:
    failure = IndexingFailure(
        entity_type=entity_type,
        entity_id=entity_id,
        lecture_id=lecture_id,
        error_type=type(error).__name__,
        message=str(error),
    )
    logger.warning(
        "Omitting entity from search index",
        extra={
            "entity_type": failure.entity_type,
            "entity_id": failure.entity_id,
            "lecture_id": failure.lecture_id,
            "error_type": failure.error_type,
            "error": failure.message,
        },
    )
    return failure


def _report_progress(
    callback: Callable[[IndexingProgress], None] | None,
    *,
    processed: int,
    total: int,
    entity_type: EmbeddingEntityType,
    entity_id: int,
    status: IndexingStatus,
) -> None:
    if callback is not None:
        callback(
            IndexingProgress(
                processed=processed,
                total=total,
                entity_type=entity_type,
                entity_id=entity_id,
                status=status,
            )
        )
