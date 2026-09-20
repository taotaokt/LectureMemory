"""Persistence, invalidation, and recovery tests for the embedding cache."""

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.embeddings.base import InvalidEmbeddingError
from app.embeddings.cache import (
    EmbeddingCache,
    EmbeddingCacheError,
    hash_bytes,
    hash_file,
    hash_text,
)
from app.repositories.embedding_repository import get_embedding_record
from app.schemas import EmbeddingRecordRead


@pytest.fixture
def cache_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "embedding-cache.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


@pytest.fixture
def embedding_cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings")


def normalized_vector(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return np.ascontiguousarray(vector / np.linalg.norm(vector), dtype=np.float32)


def fetch_record(
    session: Session,
    *,
    entity_type: str = "slide_page",
    entity_id: int = 7,
    model_name: str = "test/model",
    dimension: int = 3,
):
    return get_embedding_record(
        session,
        entity_type=entity_type,
        entity_id=entity_id,
        model_name=model_name,
        dimension=dimension,
    )


def test_cache_miss_persists_vector_and_later_hit_reuses_it(
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
) -> None:
    content_hash = hash_text("original slide")
    expected = normalized_vector(3, 4, 0)
    compute_calls = 0

    def compute() -> np.ndarray:
        nonlocal compute_calls
        compute_calls += 1
        return expected

    with session_scope(cache_session_factory) as session:
        first = embedding_cache.get_or_compute(
            session,
            entity_type="slide_page",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=content_hash,
            compute=compute,
        )
        stored_path = embedding_cache.embedding_dir / first.record.embedding_path

    with session_scope(cache_session_factory) as session:
        second = embedding_cache.get_or_compute(
            session,
            entity_type="slide_page",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=content_hash,
            compute=compute,
        )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert compute_calls == 1
    assert stored_path.is_file()
    assert np.array_equal(first.vector, expected)
    assert np.array_equal(second.vector, expected)
    assert first.record.id == second.record.id
    assert first.record.embedding_path == second.record.embedding_path
    assert not list(embedding_cache.embedding_dir.rglob("*.tmp"))

    serialized = EmbeddingRecordRead.model_validate(second.record)
    assert serialized.content_hash == content_hash
    assert serialized.dimension == 3


def test_changed_content_recomputes_and_updates_metadata(
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
) -> None:
    first_vector = normalized_vector(1, 0, 0)
    second_vector = normalized_vector(0, 1, 0)

    with session_scope(cache_session_factory) as session:
        first = embedding_cache.get_or_compute(
            session,
            entity_type="note",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=hash_text("first note"),
            compute=lambda: first_vector,
        )
        first_path = embedding_cache.embedding_dir / first.record.embedding_path
        record_id = first.record.id

    with session_scope(cache_session_factory) as session:
        second = embedding_cache.get_or_compute(
            session,
            entity_type="note",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=hash_text("edited note"),
            compute=lambda: second_vector,
        )
        second_path = embedding_cache.embedding_dir / second.record.embedding_path

    assert second.cache_hit is False
    assert second.record.id == record_id
    assert second.record.content_hash == hash_text("edited note")
    assert second_path != first_path
    assert first_path.is_file()
    assert second_path.is_file()
    assert np.array_equal(second.vector, second_vector)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "wrong-dtype"])
def test_invalid_cache_file_is_recomputed(
    damage: str,
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
) -> None:
    content_hash = hash_text("stable content")
    original = normalized_vector(1, 2, 3)
    replacement = normalized_vector(3, 2, 1)

    with session_scope(cache_session_factory) as session:
        first = embedding_cache.get_or_compute(
            session,
            entity_type="slide_page",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=content_hash,
            compute=lambda: original,
        )
        vector_path = embedding_cache.embedding_dir / first.record.embedding_path

    if damage == "missing":
        vector_path.unlink()
    elif damage == "corrupt":
        vector_path.write_bytes(b"not a numpy file")
    else:
        with vector_path.open("wb") as vector_file:
            np.save(vector_file, original.astype(np.float64), allow_pickle=False)

    compute_calls = 0

    def recompute() -> np.ndarray:
        nonlocal compute_calls
        compute_calls += 1
        return replacement

    with session_scope(cache_session_factory) as session:
        recovered = embedding_cache.get_or_compute(
            session,
            entity_type="slide_page",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=content_hash,
            compute=recompute,
        )

    assert recovered.cache_hit is False
    assert compute_calls == 1
    assert np.array_equal(recovered.vector, replacement)
    with vector_path.open("rb") as vector_file:
        assert np.array_equal(np.load(vector_file, allow_pickle=False), replacement)


def test_path_outside_cache_is_ignored_and_repaired(
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
    tmp_path: Path,
) -> None:
    content_hash = hash_text("safe content")
    outside_path = tmp_path / "outside.npy"
    with outside_path.open("wb") as outside_file:
        np.save(outside_file, normalized_vector(1, 0, 0), allow_pickle=False)

    with session_scope(cache_session_factory) as session:
        initial = embedding_cache.get_or_compute(
            session,
            entity_type="slide_page",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=content_hash,
            compute=lambda: normalized_vector(1, 0, 0),
        )
        initial.record.embedding_path = "../../outside.npy"

    replacement = normalized_vector(0, 1, 0)
    with session_scope(cache_session_factory) as session:
        repaired = embedding_cache.get_or_compute(
            session,
            entity_type="slide_page",
            entity_id=7,
            model_name="test/model",
            dimension=3,
            content_hash=content_hash,
            compute=lambda: replacement,
        )

    repaired_path = (embedding_cache.embedding_dir / repaired.record.embedding_path).resolve()
    assert repaired.cache_hit is False
    assert repaired_path.is_relative_to(embedding_cache.embedding_dir)
    assert repaired_path.is_file()
    assert np.array_equal(repaired.vector, replacement)
    assert outside_path.is_file()


@pytest.mark.parametrize(
    "invalid_vector",
    [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([np.nan, 0.0, 1.0], dtype=np.float32),
        np.array([2.0, 0.0, 0.0], dtype=np.float32),
    ],
)
def test_invalid_computed_vector_is_not_cached(
    invalid_vector: np.ndarray,
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
) -> None:
    with cache_session_factory() as session:
        with pytest.raises(InvalidEmbeddingError):
            embedding_cache.get_or_compute(
                session,
                entity_type="slide_page",
                entity_id=7,
                model_name="test/model",
                dimension=3,
                content_hash=hash_text("content"),
                compute=lambda: invalid_vector,
            )

        assert fetch_record(session) is None
    assert not list(embedding_cache.embedding_dir.rglob("*.npy"))


def test_failed_atomic_replace_removes_temporary_file_and_metadata(
    monkeypatch,
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
) -> None:
    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("app.embeddings.cache.os.replace", fail_replace)

    with cache_session_factory() as session:
        with pytest.raises(EmbeddingCacheError, match="synthetic replace failure"):
            embedding_cache.get_or_compute(
                session,
                entity_type="slide_page",
                entity_id=7,
                model_name="test/model",
                dimension=3,
                content_hash=hash_text("content"),
                compute=lambda: normalized_vector(1, 0, 0),
            )
        assert fetch_record(session) is None

    assert not list(embedding_cache.embedding_dir.rglob("*.tmp"))
    assert not list(embedding_cache.embedding_dir.rglob("*.npy"))


def test_model_dimension_and_entity_type_have_independent_cache_entries(
    cache_session_factory: sessionmaker[Session],
    embedding_cache: EmbeddingCache,
) -> None:
    entries: list[tuple[str, str, int, np.ndarray]] = [
        ("slide_page", "model-a", 3, normalized_vector(1, 0, 0)),
        ("slide_page", "model-b", 3, normalized_vector(0, 1, 0)),
        ("note", "model-a", 3, normalized_vector(0, 0, 1)),
        ("slide_page", "model-a", 2, normalized_vector(1, 1)),
    ]
    paths = set()

    with session_scope(cache_session_factory) as session:
        for entity_type, model_name, dimension, vector in entries:
            result = embedding_cache.get_or_compute(
                session,
                entity_type=entity_type,
                entity_id=7,
                model_name=model_name,
                dimension=dimension,
                content_hash=hash_text("same content"),
                compute=lambda vector=vector: vector,
            )
            paths.add(result.record.embedding_path)

    assert len(paths) == len(entries)


def test_content_hash_helpers_are_stable_and_stream_files(tmp_path: Path) -> None:
    content = b"lecture-memory\x00embedding"
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(content)

    assert hash_bytes(content) == hashlib_sha256(content)
    assert hash_text(content.decode("utf-8")) == hash_bytes(content)
    assert hash_file(source_path, chunk_size=3) == hash_bytes(content)

    source_path.write_bytes(content + b"changed")
    assert hash_file(source_path) != hash_bytes(content)


def hashlib_sha256(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    ("function", "argument", "error"),
    [
        (hash_text, b"not text", TypeError),
        (hash_file, Path("missing-file"), FileNotFoundError),
    ],
)
def test_hash_helpers_reject_invalid_inputs(
    function: Callable,
    argument: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        function(argument)


def test_hash_file_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    source_path = tmp_path / "source.bin"
    source_path.write_bytes(b"content")

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        hash_file(source_path, chunk_size=0)
