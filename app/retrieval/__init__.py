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
from app.retrieval.qwen_reranker import (
    DEFAULT_RERANKER_INSTRUCTION,
    DEFAULT_RERANKER_MODEL_NAME,
    Qwen3VLReranker,
    QwenRerankerDependencyError,
    QwenRerankerError,
)
from app.retrieval.reranker import (
    InvalidRerankerOutputError,
    RawRerankerScores,
    Reranker,
    RerankerError,
)
from app.retrieval.service import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_TOP_K,
    RetrievalConfigurationError,
    SearchFilterMismatchError,
    search_and_rerank,
    search_lecture_memory,
)

__all__ = [
    "DuplicateEntityError",
    "DEFAULT_FINAL_TOP_K",
    "DEFAULT_RERANK_TOP_K",
    "DEFAULT_RETRIEVAL_TOP_K",
    "DEFAULT_TOP_K",
    "DEFAULT_RERANKER_INSTRUCTION",
    "DEFAULT_RERANKER_MODEL_NAME",
    "FaissVectorIndex",
    "IndexedEntity",
    "IndexPersistenceError",
    "InvalidRerankerOutputError",
    "InvalidVectorError",
    "RawRerankerScores",
    "Reranker",
    "RerankerError",
    "Qwen3VLReranker",
    "QwenRerankerDependencyError",
    "QwenRerankerError",
    "RetrievalConfigurationError",
    "SearchFilterMismatchError",
    "VectorIndexError",
    "VectorSearchResult",
    "search_and_rerank",
    "search_lecture_memory",
]
