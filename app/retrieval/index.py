"""FAISS-backed vector index hidden behind application-owned types."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.schemas import EmbeddingEntityType

INDEX_FILENAME = "vectors.faiss"
METADATA_FILENAME = "metadata.json"
METADATA_VERSION = 1
SUPPORTED_ENTITY_TYPES = {"slide_page", "note"}


class VectorIndexError(RuntimeError):
    """Base error raised by the vector index abstraction."""


class InvalidVectorError(VectorIndexError):
    """Raised when vectors or search arguments violate the index contract."""


class DuplicateEntityError(VectorIndexError):
    """Raised when an entity is added to an index more than once."""


class IndexPersistenceError(VectorIndexError):
    """Raised when an index snapshot cannot be saved or loaded safely."""


@dataclass(frozen=True, slots=True)
class IndexedEntity:
    """Application entity associated with one vector in an index."""

    entity_type: EmbeddingEntityType
    entity_id: int

    def __post_init__(self) -> None:
        if self.entity_type not in SUPPORTED_ENTITY_TYPES:
            raise ValueError(f"Unsupported entity type: {self.entity_type!r}")
        if (
            not isinstance(self.entity_id, int)
            or isinstance(self.entity_id, bool)
            or self.entity_id <= 0
        ):
            raise ValueError("entity_id must be a positive integer")


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One entity match returned in descending similarity order."""

    entity_type: EmbeddingEntityType
    entity_id: int
    score: float
    rank: int


class FaissVectorIndex:
    """Exact cosine-similarity index with durable entity mappings."""

    def __init__(self, dimension: int) -> None:
        self._dimension = self._validate_dimension(dimension)
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))
        self._entities_by_vector_id: dict[int, IndexedEntity] = {}
        self._vector_ids_by_entity: dict[IndexedEntity, int] = {}
        self._next_vector_id = 0

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension accepted by this index."""
        return self._dimension

    @property
    def count(self) -> int:
        """Return the number of indexed entities."""
        return len(self._entities_by_vector_id)

    @classmethod
    def build(
        cls,
        *,
        dimension: int,
        vectors: ArrayLike,
        entities: list[IndexedEntity] | tuple[IndexedEntity, ...],
    ) -> Self:
        """Build a new index from a complete vector and entity collection."""
        index = cls(dimension)
        index.add(vectors, entities)
        return index

    def add(
        self,
        vectors: ArrayLike,
        entities: list[IndexedEntity] | tuple[IndexedEntity, ...],
    ) -> tuple[int, ...]:
        """Add vectors atomically and return their internal vector IDs."""
        entity_batch = tuple(entities)
        matrix = self._prepare_matrix(vectors, rows=len(entity_batch))
        self._validate_new_entities(entity_batch)
        if not entity_batch:
            return ()

        vector_ids = np.arange(
            self._next_vector_id,
            self._next_vector_id + len(entity_batch),
            dtype=np.int64,
        )
        try:
            self._index.add_with_ids(matrix, vector_ids)
        except Exception as exc:
            raise VectorIndexError(f"FAISS could not add vectors: {exc}") from exc

        for vector_id, entity in zip(vector_ids.tolist(), entity_batch, strict=True):
            self._entities_by_vector_id[vector_id] = entity
            self._vector_ids_by_entity[entity] = vector_id
        self._next_vector_id += len(entity_batch)
        return tuple(int(vector_id) for vector_id in vector_ids)

    def search(self, query: ArrayLike, *, top_k: int) -> tuple[VectorSearchResult, ...]:
        """Return the nearest entities using cosine similarity."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise InvalidVectorError("top_k must be a positive integer")
        query_vector = self._prepare_query(query)
        if self.count == 0:
            return ()

        result_count = min(top_k, self.count)
        try:
            scores, vector_ids = self._index.search(query_vector[None, :], result_count)
        except Exception as exc:
            raise VectorIndexError(f"FAISS search failed: {exc}") from exc

        results: list[VectorSearchResult] = []
        for rank, (score, vector_id) in enumerate(
            zip(scores[0], vector_ids[0], strict=True),
            start=1,
        ):
            entity = self._entities_by_vector_id.get(int(vector_id))
            if entity is None:
                raise IndexPersistenceError(
                    f"FAISS returned unmapped vector ID {int(vector_id)}"
                )
            results.append(
                VectorSearchResult(
                    entity_type=entity.entity_type,
                    entity_id=entity.entity_id,
                    score=float(score),
                    rank=rank,
                )
            )
        return tuple(results)

    def entity_for_vector_id(self, vector_id: int) -> IndexedEntity | None:
        """Return the application entity mapped to an internal vector ID."""
        return self._entities_by_vector_id.get(vector_id)

    def save(self, directory: str | Path) -> None:
        """Atomically write the FAISS index and its application metadata."""
        target_directory = Path(directory).expanduser().resolve()
        index_temporary: Path | None = None
        metadata_temporary: Path | None = None
        try:
            target_directory.mkdir(parents=True, exist_ok=True)
            index_path = target_directory / INDEX_FILENAME
            metadata_path = target_directory / METADATA_FILENAME

            index_descriptor, index_name = tempfile.mkstemp(
                dir=target_directory,
                prefix=f".{INDEX_FILENAME}.",
                suffix=".tmp",
            )
            os.close(index_descriptor)
            index_temporary = Path(index_name)
            faiss.write_index(self._index, str(index_temporary))

            metadata_descriptor, metadata_name = tempfile.mkstemp(
                dir=target_directory,
                prefix=f".{METADATA_FILENAME}.",
                suffix=".tmp",
            )
            metadata_temporary = Path(metadata_name)
            with os.fdopen(metadata_descriptor, "w", encoding="utf-8") as metadata_file:
                json.dump(self._metadata_payload(), metadata_file, indent=2, sort_keys=True)
                metadata_file.write("\n")
                metadata_file.flush()
                os.fsync(metadata_file.fileno())

            os.replace(index_temporary, index_path)
            index_temporary = None
            os.replace(metadata_temporary, metadata_path)
            metadata_temporary = None
        except Exception as exc:
            raise IndexPersistenceError(
                f"Could not save vector index to {target_directory}: {exc}"
            ) from exc
        finally:
            if index_temporary is not None:
                index_temporary.unlink(missing_ok=True)
            if metadata_temporary is not None:
                metadata_temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path) -> Self:
        """Load and validate a persisted FAISS index and entity mapping."""
        source_directory = Path(directory).expanduser().resolve()
        index_path = source_directory / INDEX_FILENAME
        metadata_path = source_directory / METADATA_FILENAME
        if not index_path.is_file() or not metadata_path.is_file():
            raise IndexPersistenceError(
                f"Index snapshot requires {INDEX_FILENAME} and {METADATA_FILENAME} "
                f"in {source_directory}"
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            dimension, next_vector_id, entities_by_vector_id = cls._parse_metadata(metadata)
            faiss_index = faiss.read_index(str(index_path))
            cls._validate_loaded_faiss_index(
                faiss_index,
                dimension=dimension,
                vector_ids=set(entities_by_vector_id),
            )
        except IndexPersistenceError:
            raise
        except Exception as exc:
            raise IndexPersistenceError(
                f"Could not load vector index from {source_directory}: {exc}"
            ) from exc

        instance = cls(dimension)
        instance._index = faiss_index
        instance._entities_by_vector_id = entities_by_vector_id
        instance._vector_ids_by_entity = {
            entity: vector_id for vector_id, entity in entities_by_vector_id.items()
        }
        instance._next_vector_id = next_vector_id
        return instance

    def _prepare_matrix(self, vectors: ArrayLike, *, rows: int) -> NDArray[np.float32]:
        raw_matrix = np.asarray(vectors, dtype=np.float32)
        if rows == 0 and raw_matrix.size == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        if raw_matrix.ndim != 2 or raw_matrix.shape != (rows, self.dimension):
            raise InvalidVectorError(
                f"Vector batch shape is {raw_matrix.shape}; "
                f"expected ({rows}, {self.dimension})"
            )
        if not np.isfinite(raw_matrix).all():
            raise InvalidVectorError("Vector batch contains non-finite values")
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise InvalidVectorError("Vector batch contains a zero vector")
        return np.ascontiguousarray(raw_matrix / norms, dtype=np.float32)

    def _prepare_query(self, query: ArrayLike) -> NDArray[np.float32]:
        raw_query = np.asarray(query, dtype=np.float32)
        if raw_query.ndim != 1 or raw_query.shape != (self.dimension,):
            raise InvalidVectorError(
                f"Query shape is {raw_query.shape}; expected ({self.dimension},)"
            )
        if not np.isfinite(raw_query).all():
            raise InvalidVectorError("Query contains non-finite values")
        norm = float(np.linalg.norm(raw_query))
        if norm == 0:
            raise InvalidVectorError("Query must not be a zero vector")
        return np.ascontiguousarray(raw_query / norm, dtype=np.float32)

    def _validate_new_entities(self, entities: tuple[IndexedEntity, ...]) -> None:
        seen: set[IndexedEntity] = set()
        for entity in entities:
            if not isinstance(entity, IndexedEntity):
                raise TypeError("entities must contain IndexedEntity values")
            if entity in seen or entity in self._vector_ids_by_entity:
                raise DuplicateEntityError(
                    f"Entity {entity.entity_type}:{entity.entity_id} is already indexed"
                )
            seen.add(entity)

    def _metadata_payload(self) -> dict[str, Any]:
        return {
            "version": METADATA_VERSION,
            "dimension": self.dimension,
            "next_vector_id": self._next_vector_id,
            "entities": [
                {
                    "vector_id": vector_id,
                    "entity_type": entity.entity_type,
                    "entity_id": entity.entity_id,
                }
                for vector_id, entity in sorted(self._entities_by_vector_id.items())
            ],
        }

    @classmethod
    def _parse_metadata(
        cls,
        metadata: Any,
    ) -> tuple[int, int, dict[int, IndexedEntity]]:
        if not isinstance(metadata, dict) or metadata.get("version") != METADATA_VERSION:
            raise IndexPersistenceError("Unsupported or missing index metadata version")
        dimension = cls._validate_dimension(metadata.get("dimension"))
        next_vector_id = metadata.get("next_vector_id")
        if (
            not isinstance(next_vector_id, int)
            or isinstance(next_vector_id, bool)
            or next_vector_id < 0
        ):
            raise IndexPersistenceError("Invalid next_vector_id in index metadata")
        raw_entities = metadata.get("entities")
        if not isinstance(raw_entities, list):
            raise IndexPersistenceError("Index metadata entities must be a list")

        entities_by_vector_id: dict[int, IndexedEntity] = {}
        seen_entities: set[IndexedEntity] = set()
        try:
            for item in raw_entities:
                if not isinstance(item, dict):
                    raise ValueError("entity mapping must be an object")
                vector_id = item.get("vector_id")
                if (
                    not isinstance(vector_id, int)
                    or isinstance(vector_id, bool)
                    or vector_id < 0
                    or vector_id in entities_by_vector_id
                ):
                    raise ValueError("invalid or duplicate vector_id")
                entity = IndexedEntity(
                    entity_type=item.get("entity_type"),
                    entity_id=item.get("entity_id"),
                )
                if entity in seen_entities:
                    raise ValueError("duplicate indexed entity")
                entities_by_vector_id[vector_id] = entity
                seen_entities.add(entity)
        except (TypeError, ValueError) as exc:
            raise IndexPersistenceError(f"Invalid entity mapping: {exc}") from exc

        minimum_next_id = max(entities_by_vector_id, default=-1) + 1
        if next_vector_id < minimum_next_id:
            raise IndexPersistenceError("next_vector_id overlaps an existing vector ID")
        return dimension, next_vector_id, entities_by_vector_id

    @staticmethod
    def _validate_loaded_faiss_index(
        index: Any,
        *,
        dimension: int,
        vector_ids: set[int],
    ) -> None:
        if not isinstance(index, faiss.IndexIDMap2):
            raise IndexPersistenceError("Persisted FAISS index is not an ID-mapped index")
        if index.d != dimension:
            raise IndexPersistenceError(
                f"FAISS dimension {index.d} does not match metadata dimension {dimension}"
            )
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise IndexPersistenceError("Persisted FAISS index does not use inner product")
        stored_ids = {int(value) for value in faiss.vector_to_array(index.id_map)}
        if stored_ids != vector_ids or index.ntotal != len(vector_ids):
            raise IndexPersistenceError("FAISS vectors do not match metadata entity mappings")

    @staticmethod
    def _validate_dimension(dimension: Any) -> int:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise InvalidVectorError("dimension must be a positive integer")
        return dimension
