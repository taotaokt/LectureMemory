"""Orchestration for embedding personal notes from one lecture."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.embeddings.base import EmbeddingProvider
from app.embeddings.cache import EmbeddingCache, hash_text
from app.models import Lecture
from app.repositories.errors import LectureNotFoundError
from app.repositories.note_repository import list_lecture_notes

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NoteEmbeddingFailure:
    """Details for one note that could not be embedded."""

    note_id: int
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class NoteEmbeddingSummary:
    """Aggregate result for one lecture note-embedding run."""

    lecture_id: int
    model_name: str
    dimension: int
    total_notes: int
    generated_notes: int
    cached_notes: int
    failed_notes: int
    failures: tuple[NoteEmbeddingFailure, ...]

    @property
    def successful_notes(self) -> int:
        """Return notes that were generated or served by the cache."""
        return self.generated_notes + self.cached_notes

    @property
    def complete(self) -> bool:
        """Return whether every persisted note now has a valid embedding."""
        return self.successful_notes == self.total_notes and self.failed_notes == 0


def embed_lecture_notes(
    session: Session,
    lecture_id: int,
    *,
    provider: EmbeddingProvider,
    cache: EmbeddingCache,
) -> NoteEmbeddingSummary:
    """Embed every note in a lecture, reusing unchanged cached vectors."""
    lecture = session.get(Lecture, lecture_id)
    if lecture is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    notes = list_lecture_notes(session, lecture_id)
    generated_notes = 0
    cached_notes = 0
    failures: list[NoteEmbeddingFailure] = []

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
            if result.cache_hit:
                cached_notes += 1
            else:
                generated_notes += 1
        except Exception as exc:
            failure = NoteEmbeddingFailure(
                note_id=note.id,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            failures.append(failure)
            logger.warning(
                "Failed to embed note",
                extra={
                    "lecture_id": lecture_id,
                    "note_id": note.id,
                    "error_type": failure.error_type,
                    "error": failure.message,
                },
            )

    summary = NoteEmbeddingSummary(
        lecture_id=lecture_id,
        model_name=provider.model_name,
        dimension=provider.dimension,
        total_notes=len(notes),
        generated_notes=generated_notes,
        cached_notes=cached_notes,
        failed_notes=len(failures),
        failures=tuple(failures),
    )
    logger.info(
        "Embedded lecture notes",
        extra={
            "lecture_id": summary.lecture_id,
            "model_name": summary.model_name,
            "dimension": summary.dimension,
            "total_notes": summary.total_notes,
            "generated_notes": summary.generated_notes,
            "cached_notes": summary.cached_notes,
            "failed_notes": summary.failed_notes,
        },
    )
    return summary
