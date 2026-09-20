"""Search and retrieval components for Lecture Memory."""

from app.retrieval.index import (
    DuplicateEntityError,
    FaissVectorIndex,
    IndexedEntity,
    IndexPersistenceError,
    InvalidVectorError,
    VectorIndexError,
    VectorSearchResult,
)
from app.retrieval.reranker import (
    InvalidRerankerOutputError,
    RawRerankerScores,
    Reranker,
    RerankerError,
)
from app.retrieval.service import (
    DEFAULT_TOP_K,
    RetrievalConfigurationError,
    SearchFilterMismatchError,
    search_lecture_memory,
)

__all__ = [
    "DuplicateEntityError",
    "DEFAULT_TOP_K",
    "FaissVectorIndex",
    "IndexedEntity",
    "IndexPersistenceError",
    "InvalidRerankerOutputError",
    "InvalidVectorError",
    "RawRerankerScores",
    "Reranker",
    "RerankerError",
    "RetrievalConfigurationError",
    "SearchFilterMismatchError",
    "VectorIndexError",
    "VectorSearchResult",
    "search_lecture_memory",
]
