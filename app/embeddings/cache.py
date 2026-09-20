"""Persistent embedding cache backed by SQLite metadata and NumPy files."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from app.embeddings.base import EmbeddingError, EmbeddingVector, InvalidEmbeddingError
from app.models import EmbeddingRecord
from app.repositories.embedding_repository import (
    get_embedding_record,
    upsert_embedding_record,
)
from app.schemas import EmbeddingEntityType, EmbeddingRecordCreate

logger = logging.getLogger(__name__)


class EmbeddingCacheError(EmbeddingError):
    """Raised when an embedding cache operation cannot complete safely."""


@dataclass(frozen=True, slots=True)
class EmbeddingCacheResult:
    """One cached or newly computed vector and its persistence metadata."""

    vector: EmbeddingVector
    record: EmbeddingRecord
    cache_hit: bool


class EmbeddingCache:
    """Reuse unchanged entity embeddings and recover invalid cache files."""

    def __init__(self, embedding_dir: str | Path) -> None:
        self.embedding_dir = Path(embedding_dir).expanduser().resolve()
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        session: Session,
        *,
        entity_type: EmbeddingEntityType,
        entity_id: int,
        model_name: str,
        dimension: int,
        content_hash: str,
        compute: Callable[[], EmbeddingVector],
    ) -> EmbeddingCacheResult:
        """Return a valid cached vector or compute and atomically persist one."""
        identity = self._validate_identity(
            entity_type=entity_type,
            entity_id=entity_id,
            model_name=model_name,
            dimension=dimension,
            content_hash=content_hash,
        )
        record = get_embedding_record(
            session,
            entity_type=identity.entity_type,
            entity_id=identity.entity_id,
            model_name=identity.model_name,
            dimension=identity.dimension,
        )

        if record is not None and record.content_hash == identity.content_hash:
            cached_vector = self._try_load(record, dimension=identity.dimension)
            if cached_vector is not None:
                return EmbeddingCacheResult(
                    vector=cached_vector,
                    record=record,
                    cache_hit=True,
                )

        vector = self._validate_vector(compute(), dimension=identity.dimension)
        relative_path = self._relative_vector_path(identity)
        absolute_path = self.embedding_dir / relative_path
        self._atomic_save(absolute_path, vector)

        stored_record = upsert_embedding_record(
            session,
            EmbeddingRecordCreate(
                entity_type=identity.entity_type,
                entity_id=identity.entity_id,
                model_name=identity.model_name,
                dimension=identity.dimension,
                embedding_path=relative_path.as_posix(),
                content_hash=identity.content_hash,
            ),
        )
        logger.info(
            "Generated embedding cache entry",
            extra={
                "entity_type": identity.entity_type,
                "entity_id": identity.entity_id,
                "model_name": identity.model_name,
                "dimension": identity.dimension,
            },
        )
        return EmbeddingCacheResult(vector=vector, record=stored_record, cache_hit=False)

    def _try_load(
        self,
        record: EmbeddingRecord,
        *,
        dimension: int,
    ) -> EmbeddingVector | None:
        try:
            path = self._resolve_record_path(record.embedding_path)
            with path.open("rb") as vector_file:
                raw_vector = np.load(vector_file, allow_pickle=False)
            return self._validate_vector(raw_vector, dimension=dimension)
        except (EmbeddingCacheError, InvalidEmbeddingError, OSError, ValueError, EOFError) as exc:
            logger.warning(
                "Ignoring invalid embedding cache entry",
                extra={
                    "record_id": record.id,
                    "embedding_path": record.embedding_path,
                    "reason": str(exc),
                },
            )
            return None

    def _resolve_record_path(self, stored_path: str) -> Path:
        candidate = (self.embedding_dir / stored_path).resolve()
        try:
            candidate.relative_to(self.embedding_dir)
        except ValueError as exc:
            raise EmbeddingCacheError(
                f"Embedding path escapes cache directory: {stored_path}"
            ) from exc
        return candidate

    def _relative_vector_path(self, identity: EmbeddingRecordCreate) -> Path:
        model_hash = hashlib.sha256(identity.model_name.encode("utf-8")).hexdigest()[:12]
        filename = f"{identity.entity_id}-{identity.content_hash[:16]}.npy"
        return (
            Path(identity.entity_type)
            / f"{model_hash}-d{identity.dimension}"
            / filename
        )

    def _atomic_save(self, path: Path, vector: EmbeddingVector) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                np.save(temporary_file, vector, allow_pickle=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            raise EmbeddingCacheError(f"Could not write embedding cache file {path}: {exc}") from exc

    @staticmethod
    def _validate_vector(vector: NDArray[np.floating], *, dimension: int) -> EmbeddingVector:
        array = np.asarray(vector)
        if array.dtype != np.float32:
            raise InvalidEmbeddingError(
                f"Cached embedding dtype is {array.dtype}; expected float32"
            )
        if array.ndim != 1 or array.shape != (dimension,):
            raise InvalidEmbeddingError(
                f"Cached embedding shape is {array.shape}; expected ({dimension},)"
            )
        if not np.isfinite(array).all():
            raise InvalidEmbeddingError("Cached embedding contains non-finite values")
        norm = float(np.linalg.norm(array))
        if not np.isclose(norm, 1.0, rtol=1e-4, atol=1e-5):
            raise InvalidEmbeddingError(
                f"Cached embedding norm is {norm}; expected a normalized vector"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    @staticmethod
    def _validate_identity(
        *,
        entity_type: EmbeddingEntityType,
        entity_id: int,
        model_name: str,
        dimension: int,
        content_hash: str,
    ) -> EmbeddingRecordCreate:
        return EmbeddingRecordCreate(
            entity_type=entity_type,
            entity_id=entity_id,
            model_name=model_name,
            dimension=dimension,
            embedding_path="pending.npy",
            content_hash=content_hash,
        )


def hash_bytes(content: bytes) -> str:
    """Return a stable SHA-256 digest for binary entity content."""
    return hashlib.sha256(content).hexdigest()


def hash_text(content: str) -> str:
    """Return a stable SHA-256 digest for UTF-8 text content."""
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    return hash_bytes(content.encode("utf-8"))


def hash_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a local file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    resolved_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved_path.open("rb") as source_file:
        while chunk := source_file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
