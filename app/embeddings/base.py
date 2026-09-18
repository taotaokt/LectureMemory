"""Model-independent embedding provider contract."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

EmbeddingVector = NDArray[np.float32]
EmbeddingBatch = NDArray[np.float32]
RawEmbedding = Sequence[float] | NDArray[np.floating]


class EmbeddingError(RuntimeError):
    """Base error raised by embedding providers."""


class InvalidEmbeddingError(EmbeddingError):
    """Raised when a provider returns a malformed embedding vector."""


class EmbeddingProvider(ABC):
    """Stable interface for text, image, and retrieval-query embeddings.

    Provider implementations return raw vectors from the protected ``_embed_*``
    methods. The public methods enforce shape, finite values, float32 dtype, and
    L2 normalization before exposing vectors to the rest of the application.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the stable model identifier used for caching and metadata."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the number of values in each embedding vector."""

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed non-empty document or note text."""
        cleaned_text = _validate_text(text, field_name="text")
        return self._prepare_vector(self._embed_text(cleaned_text))

    def embed_image(self, image_path: str | Path) -> EmbeddingVector:
        """Embed one existing local image."""
        resolved_path = _validate_image_path(image_path)
        return self._prepare_vector(self._embed_image(resolved_path))

    def embed_query(self, query: str) -> EmbeddingVector:
        """Embed a non-empty retrieval query using query-specific behavior."""
        cleaned_query = _validate_text(query, field_name="query")
        return self._prepare_vector(self._embed_query(cleaned_query))

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        """Embed a text batch, with a provider-overridable sequential default."""
        return self._stack_vectors([self.embed_text(text) for text in texts])

    def embed_images(self, image_paths: Sequence[str | Path]) -> EmbeddingBatch:
        """Embed an image batch, with a provider-overridable sequential default."""
        return self._stack_vectors([self.embed_image(path) for path in image_paths])

    def embed_queries(self, queries: Sequence[str]) -> EmbeddingBatch:
        """Embed a query batch, with a provider-overridable sequential default."""
        return self._stack_vectors([self.embed_query(query) for query in queries])

    @abstractmethod
    def _embed_text(self, text: str) -> RawEmbedding:
        """Produce one raw text vector in the provider implementation."""

    @abstractmethod
    def _embed_image(self, image_path: Path) -> RawEmbedding:
        """Produce one raw image vector in the provider implementation."""

    @abstractmethod
    def _embed_query(self, query: str) -> RawEmbedding:
        """Produce one raw query vector in the provider implementation."""

    def _prepare_vector(self, raw_vector: RawEmbedding) -> EmbeddingVector:
        dimension = self._validated_dimension()
        vector = np.asarray(raw_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape != (dimension,):
            raise InvalidEmbeddingError(
                f"{self.model_name} returned shape {vector.shape}; expected ({dimension},)"
            )
        if not np.isfinite(vector).all():
            raise InvalidEmbeddingError(f"{self.model_name} returned non-finite values")

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise InvalidEmbeddingError(f"{self.model_name} returned a zero vector")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    def _stack_vectors(self, vectors: list[EmbeddingVector]) -> EmbeddingBatch:
        if not vectors:
            return np.empty((0, self._validated_dimension()), dtype=np.float32)
        return np.ascontiguousarray(np.stack(vectors), dtype=np.float32)

    def _validated_dimension(self) -> int:
        dimension = self.dimension
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise InvalidEmbeddingError(
                f"{self.model_name} declared invalid embedding dimension {dimension!r}"
            )
        return dimension


def _validate_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned_value = value.strip()
    if not cleaned_value:
        raise ValueError(f"{field_name} must not be blank")
    return cleaned_value


def _validate_image_path(image_path: str | Path) -> Path:
    resolved_path = Path(image_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Image does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Image path is not a file: {resolved_path}")
    return resolved_path
