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
from app.retrieval.service import (
    DEFAULT_TOP_K,
    RetrievalConfigurationError,
    search_lecture_memory,
)

__all__ = [
    "DuplicateEntityError",
    "DEFAULT_TOP_K",
    "FaissVectorIndex",
    "IndexedEntity",
    "IndexPersistenceError",
    "InvalidVectorError",
    "RetrievalConfigurationError",
    "VectorIndexError",
    "VectorSearchResult",
    "search_lecture_memory",
]
