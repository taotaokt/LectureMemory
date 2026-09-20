"""Orchestration for embedding rendered pages from one lecture."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.embeddings.base import EmbeddingProvider
from app.embeddings.cache import EmbeddingCache, hash_file
from app.models import Lecture
from app.repositories.errors import LectureNotFoundError
from app.repositories.slide_page_repository import list_lecture_pages

logger = logging.getLogger(__name__)


class LectureHasNoSlidesError(RuntimeError):
    """Raised when slide embedding starts before PDF ingestion."""


@dataclass(frozen=True, slots=True)
class SlideEmbeddingFailure:
    """Details for one page that could not be embedded."""

    page_id: int
    page_number: int
    image_path: Path
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class SlideEmbeddingSummary:
    """Aggregate result for one lecture slide-embedding run."""

    lecture_id: int
    model_name: str
    dimension: int
    total_pages: int
    generated_pages: int
    cached_pages: int
    failed_pages: int
    failures: tuple[SlideEmbeddingFailure, ...]

    @property
    def successful_pages(self) -> int:
        """Return pages that were generated or served by the cache."""
        return self.generated_pages + self.cached_pages

    @property
    def complete(self) -> bool:
        """Return whether every persisted page now has a valid embedding."""
        return self.successful_pages == self.total_pages and self.failed_pages == 0


def embed_lecture_slides(
    session: Session,
    lecture_id: int,
    *,
    provider: EmbeddingProvider,
    cache: EmbeddingCache,
) -> SlideEmbeddingSummary:
    """Embed every rendered page in a lecture, reusing valid cache entries."""
    lecture = session.get(Lecture, lecture_id)
    if lecture is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    pages = list_lecture_pages(session, lecture_id)
    if not pages:
        raise LectureHasNoSlidesError(
            f"Lecture {lecture_id} has no slide pages; ingest a PDF before embedding"
        )

    generated_pages = 0
    cached_pages = 0
    failures: list[SlideEmbeddingFailure] = []

    for page in pages:
        image_path = Path(page.image_path).expanduser().resolve()
        try:
            content_hash = hash_file(image_path)
            with session.begin_nested():
                result = cache.get_or_compute(
                    session,
                    entity_type="slide_page",
                    entity_id=page.id,
                    model_name=provider.model_name,
                    dimension=provider.dimension,
                    content_hash=content_hash,
                    compute=lambda image_path=image_path: provider.embed_image(image_path),
                )
            if result.cache_hit:
                cached_pages += 1
            else:
                generated_pages += 1
        except Exception as exc:
            failure = SlideEmbeddingFailure(
                page_id=page.id,
                page_number=page.page_number,
                image_path=image_path,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            failures.append(failure)
            logger.warning(
                "Failed to embed slide page",
                extra={
                    "lecture_id": lecture_id,
                    "page_id": page.id,
                    "page_number": page.page_number,
                    "image_path": str(image_path),
                    "error_type": failure.error_type,
                    "error": failure.message,
                },
            )

    summary = SlideEmbeddingSummary(
        lecture_id=lecture_id,
        model_name=provider.model_name,
        dimension=provider.dimension,
        total_pages=len(pages),
        generated_pages=generated_pages,
        cached_pages=cached_pages,
        failed_pages=len(failures),
        failures=tuple(failures),
    )
    logger.info(
        "Embedded lecture slide pages",
        extra={
            "lecture_id": summary.lecture_id,
            "model_name": summary.model_name,
            "dimension": summary.dimension,
            "total_pages": summary.total_pages,
            "generated_pages": summary.generated_pages,
            "cached_pages": summary.cached_pages,
            "failed_pages": summary.failed_pages,
        },
    )
    return summary
