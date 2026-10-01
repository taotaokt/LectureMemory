"""Configured search runtime assembly for the application interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings import EmbeddingProvider, Qwen3VLEmbeddingProvider
from app.retrieval import (
    FaissVectorIndex,
    IndexPersistenceError,
    Qwen3VLReranker,
    Reranker,
    search_and_rerank,
)
from app.retrieval.index import INDEX_FILENAME, METADATA_FILENAME
from app.schemas import SearchResult


class SearchRuntimeUnavailableError(RuntimeError):
    """Raised when a configured persistent search runtime cannot be loaded."""


@dataclass(frozen=True, slots=True)
class SearchRuntime:
    """Reusable models and vector index required by one search process."""

    provider: EmbeddingProvider
    index: FaissVectorIndex
    reranker: Reranker


def search_index_revision(index_dir: str | Path | None) -> str:
    """Return a lightweight cache key that changes with the saved index snapshot."""
    if index_dir is None:
        return "unconfigured"

    directory = Path(index_dir).expanduser().resolve()
    revisions: list[str] = []
    for filename in (INDEX_FILENAME, METADATA_FILENAME):
        path = directory / filename
        try:
            stat = path.stat()
        except FileNotFoundError:
            revisions.append(f"{filename}:missing")
        else:
            revisions.append(f"{filename}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(revisions)


def build_search_runtime(settings: Settings) -> SearchRuntime:
    """Load the persisted index and lazily configured Qwen adapters."""
    if settings.index_dir is None:
        raise SearchRuntimeUnavailableError("INDEX_DIR is not configured")
    try:
        index = FaissVectorIndex.load(settings.index_dir)
    except IndexPersistenceError as exc:
        raise SearchRuntimeUnavailableError(
            "No usable search index was found. Generate and save the lecture index first."
        ) from exc

    provider = Qwen3VLEmbeddingProvider(
        model_name=settings.model_name,
        device=settings.device,
        dtype=settings.embedding_dtype,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
        max_pixels=settings.embedding_max_pixels,
        query_instruction=settings.embedding_query_instruction,
    )
    reranker = Qwen3VLReranker(
        model_name=settings.reranker_model_name,
        device=settings.device,
        dtype=settings.reranker_dtype,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        min_pixels=settings.reranker_min_pixels,
        max_pixels=settings.reranker_max_pixels,
        instruction=settings.reranker_instruction,
    )
    return SearchRuntime(provider=provider, index=index, reranker=reranker)


def search_course_memory(
    session: Session,
    query: str,
    *,
    course_id: int,
    runtime: SearchRuntime,
    settings: Settings,
    reranker_weight: float | None = None,
) -> tuple[SearchResult, ...]:
    """Run configured retrieval and reranking within one course."""
    return search_and_rerank(
        session,
        query,
        provider=runtime.provider,
        index=runtime.index,
        reranker=runtime.reranker,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
        final_top_k=settings.final_top_k,
        reranker_weight=(
            settings.reranker_weight
            if reranker_weight is None
            else reranker_weight
        ),
        course_id=course_id,
    )
