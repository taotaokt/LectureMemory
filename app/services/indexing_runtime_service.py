"""Configured assembly for rebuilding the persisted application search index."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings import EmbeddingCache, Qwen3VLEmbeddingProvider
from app.retrieval import FaissVectorIndex, IndexPersistenceError
from app.services.indexing_service import (
    IndexBuildSummary,
    IndexingProgress,
    build_search_index,
    get_search_source_snapshot,
)

SearchIndexState = Literal["missing", "stale", "current"]


@dataclass(frozen=True, slots=True)
class SearchIndexStatus:
    """Read-only freshness assessment for the persisted search index."""

    state: SearchIndexState
    indexed_entities: int
    current_entities: int | None
    reason: str


def get_search_index_status(session: Session, settings: Settings) -> SearchIndexStatus:
    """Compare persisted index metadata with current content and model settings."""
    if settings.index_dir is None:
        return SearchIndexStatus("missing", 0, None, "INDEX_DIR is not configured")

    try:
        index = FaissVectorIndex.load(settings.index_dir)
    except IndexPersistenceError:
        return SearchIndexStatus("missing", 0, None, "No usable search index was found")

    try:
        snapshot = get_search_source_snapshot(session)
    except OSError as exc:
        return SearchIndexStatus(
            "stale",
            index.count,
            None,
            f"Searchable source content could not be read: {exc}",
        )

    if index.dimension != settings.embedding_dimension:
        reason = "The configured embedding dimension has changed"
    elif index.model_name is None or index.source_signature is None:
        reason = "The index predates freshness metadata"
    elif index.model_name != settings.model_name:
        reason = "The configured embedding model has changed"
    elif index.source_signature != snapshot.signature:
        reason = "Slides or notes have changed since the last build"
    else:
        return SearchIndexStatus(
            "current",
            index.count,
            snapshot.entity_count,
            "The search index matches the current content and model",
        )

    return SearchIndexStatus(
        "stale",
        index.count,
        snapshot.entity_count,
        reason,
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
