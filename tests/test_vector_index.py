"""Behavior and persistence tests for the FAISS index abstraction."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.retrieval.index import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    DuplicateEntityError,
    FaissVectorIndex,
    IndexedEntity,
    IndexPersistenceError,
    InvalidVectorError,
)


def test_build_and_search_maps_ranked_vectors_to_entities() -> None:
    index = FaissVectorIndex.build(
        dimension=3,
        vectors=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        entities=[
            IndexedEntity("slide_page", 11),
            IndexedEntity("note", 21),
            IndexedEntity("slide_page", 12),
        ],
    )

    results = index.search([1.0, 0.0, 0.0], top_k=2)

    assert index.dimension == 3
    assert index.count == 3
    assert [(result.entity_type, result.entity_id) for result in results] == [
        ("slide_page", 11),
        ("note", 21),
    ]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)


def test_add_returns_stable_ids_and_rejects_duplicate_entities() -> None:
    index = FaissVectorIndex(2)
    first = IndexedEntity("slide_page", 1)
    second = IndexedEntity("note", 1)

    assert index.add([[3.0, 4.0]], [first]) == (0,)
    assert index.add([[0.0, 2.0]], [second]) == (1,)
    assert index.entity_for_vector_id(0) == first
    assert index.entity_for_vector_id(1) == second
    assert index.entity_for_vector_id(999) is None

    with pytest.raises(DuplicateEntityError, match="slide_page:1"):
        index.add([[1.0, 0.0]], [first])

    with pytest.raises(DuplicateEntityError, match="note:4"):
        index.add(
            [[1.0, 0.0], [0.0, 1.0]],
            [IndexedEntity("note", 4), IndexedEntity("note", 4)],
        )

    assert index.count == 2


def test_save_and_load_preserve_search_and_allow_incremental_add(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "index"
    original = FaissVectorIndex.build(
        dimension=3,
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        entities=[IndexedEntity("slide_page", 7), IndexedEntity("note", 8)],
    )
    original.save(snapshot)

    loaded = FaissVectorIndex.load(snapshot)
    new_ids = loaded.add([[0.0, 0.0, 1.0]], [IndexedEntity("note", 9)])
    results = loaded.search([0.0, 0.0, 2.0], top_k=5)

    assert (snapshot / INDEX_FILENAME).is_file()
    assert (snapshot / METADATA_FILENAME).is_file()
    assert loaded.dimension == 3
    assert loaded.count == 3
    assert new_ids == (2,)
    assert results[0].entity_type == "note"
    assert results[0].entity_id == 9
    assert results[0].score == pytest.approx(1.0)
    assert len(results) == 3
    assert not list(snapshot.glob("*.tmp"))


def test_empty_index_can_be_searched_and_round_tripped(tmp_path: Path) -> None:
    index = FaissVectorIndex.build(dimension=4, vectors=[], entities=[])

    assert index.search([1.0, 0.0, 0.0, 0.0], top_k=10) == ()

    index.save(tmp_path)
    loaded = FaissVectorIndex.load(tmp_path)
    assert loaded.dimension == 4
    assert loaded.count == 0
    assert loaded.search([1.0, 0.0, 0.0, 0.0], top_k=1) == ()


@pytest.mark.parametrize(
    ("vectors", "entities", "message"),
    [
        ([[1.0, 0.0]], [IndexedEntity("note", 1)], "shape"),
        ([[0.0, 0.0, 0.0]], [IndexedEntity("note", 1)], "zero vector"),
        ([[1.0, float("nan"), 0.0]], [IndexedEntity("note", 1)], "non-finite"),
        ([[1.0, 0.0, 0.0]], [], "shape"),
    ],
)
def test_add_rejects_invalid_vector_batches(
    vectors: list[list[float]],
    entities: list[IndexedEntity],
    message: str,
) -> None:
    index = FaissVectorIndex(3)

    with pytest.raises(InvalidVectorError, match=message):
        index.add(vectors, entities)

    assert index.count == 0


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ([1.0, 0.0], 1, "shape"),
        ([0.0, 0.0, 0.0], 1, "zero vector"),
        ([1.0, float("inf"), 0.0], 1, "non-finite"),
        ([1.0, 0.0, 0.0], 0, "top_k"),
    ],
)
def test_search_rejects_invalid_arguments(
    query: list[float],
    top_k: int,
    message: str,
) -> None:
    index = FaissVectorIndex(3)

    with pytest.raises(InvalidVectorError, match=message):
        index.search(query, top_k=top_k)


@pytest.mark.parametrize("dimension", [0, -1, True, 2.5])
def test_invalid_dimensions_are_rejected(dimension: object) -> None:
    with pytest.raises(InvalidVectorError, match="dimension"):
        FaissVectorIndex(dimension)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "entity",
    [
        ("video", 1),
        ("note", 0),
        ("slide_page", True),
    ],
)
def test_invalid_entity_identifiers_are_rejected(entity: tuple[object, object]) -> None:
    with pytest.raises(ValueError):
        IndexedEntity(entity[0], entity[1])  # type: ignore[arg-type]


def test_load_rejects_missing_or_corrupt_snapshot(tmp_path: Path) -> None:
    with pytest.raises(IndexPersistenceError, match="requires"):
        FaissVectorIndex.load(tmp_path)

    index = FaissVectorIndex.build(
        dimension=2,
        vectors=[[1.0, 0.0]],
        entities=[IndexedEntity("note", 1)],
    )
    index.save(tmp_path)
    (tmp_path / METADATA_FILENAME).write_text("not json", encoding="utf-8")

    with pytest.raises(IndexPersistenceError, match="Could not load"):
        FaissVectorIndex.load(tmp_path)


def test_load_rejects_metadata_that_does_not_match_faiss(tmp_path: Path) -> None:
    index = FaissVectorIndex.build(
        dimension=2,
        vectors=[[1.0, 0.0]],
        entities=[IndexedEntity("note", 1)],
    )
    index.save(tmp_path)
    metadata_path = tmp_path / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["entities"] = []
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IndexPersistenceError, match="do not match"):
        FaissVectorIndex.load(tmp_path)
