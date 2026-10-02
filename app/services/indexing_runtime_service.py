"""Configured assembly for rebuilding the persisted application search index."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings import EmbeddingCache, Qwen3VLEmbeddingProvider
from app.services.indexing_service import (
    IndexBuildSummary,
    IndexingProgress,
    build_search_index,
)


def build_configured_search_index(
    session: Session,
    settings: Settings,
    *,
    progress_callback: Callable[[IndexingProgress], None] | None = None,
) -> IndexBuildSummary:
    """Build the complete index using the application's validated model settings."""
    if settings.embedding_dir is None:
        raise ValueError("EMBEDDING_DIR is not configured")
    if settings.index_dir is None:
        raise ValueError("INDEX_DIR is not configured")

    provider = Qwen3VLEmbeddingProvider(
        model_name=settings.model_name,
        device=settings.device,
        dtype=settings.embedding_dtype,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_pixels=settings.embedding_max_pixels,
        query_instruction=settings.embedding_query_instruction,
    )
    return build_search_index(
        session,
        provider=provider,
        cache=EmbeddingCache(settings.embedding_dir),
        index_dir=settings.index_dir,
        progress_callback=progress_callback,
    )
