"""Contract tests for the model-independent reranker abstraction."""

from collections.abc import Sequence

import numpy as np
import pytest

from app.retrieval import InvalidRerankerOutputError, Reranker
from app.schemas import SearchResult


class MappingReranker(Reranker):
    """Deterministic reranker with observable batches."""

    def __init__(self, scores_by_entity_id: dict[int, float]) -> None:
        self.scores_by_entity_id = scores_by_entity_id
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    @property
    def model_name(self) -> str:
        return "test/reranker"

    def _score(
        self,
        query: str,
        candidates: tuple[SearchResult, ...],
    ) -> Sequence[float]:
        self.calls.append((query, tuple(candidate.entity_id for candidate in candidates)))
        return [self.scores_by_entity_id[candidate.entity_id] for candidate in candidates]


class RawOutputReranker(Reranker):
    def __init__(self, raw_scores: object) -> None:
        self.raw_scores = raw_scores

    @property
    def model_name(self) -> str:
        return "test/invalid-reranker"

    def _score(
        self,
        query: str,
        candidates: tuple[SearchResult, ...],
    ):
        return self.raw_scores


def make_result(
    entity_id: int,
    *,
    rank: int,
    raw_similarity: float,
    result_type: str = "slide",
) -> SearchResult:
    return SearchResult(
        result_type=result_type,
        entity_id=entity_id,
        rank=rank,
        course_id=1,
        course_name="Algorithms",
        course_code="CS344",
        lecture_id=3,
        lecture_title="Divide and Conquer",
        lecture_number=3,
        page_number=entity_id if result_type == "slide" else None,
        preview_path=f"/page-{entity_id}.png" if result_type == "slide" else None,
        raw_similarity=raw_similarity,
        text_preview=f"Candidate {entity_id}",
    )


def test_abstract_reranker_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Reranker()


def test_rerank_orders_scores_and_preserves_embedding_similarity() -> None:
    candidates = (
        make_result(1, rank=1, raw_similarity=0.95),
        make_result(2, rank=2, raw_similarity=0.90, result_type="note"),
        make_result(3, rank=3, raw_similarity=0.80),
    )
    reranker = MappingReranker({1: 0.2, 2: 0.95, 3: 0.5})

    results = reranker.rerank("  three recursive calls  ", candidates)

    assert reranker.calls == [("three recursive calls", (1, 2, 3))]
    assert [result.entity_id for result in results] == [2, 3, 1]
    assert [result.rank for result in results] == [1, 2, 3]
    assert [result.reranker_score for result in results] == [0.95, 0.5, 0.2]
    assert [result.raw_similarity for result in results] == [0.90, 0.80, 0.95]
    assert [candidate.reranker_score for candidate in candidates] == [None, None, None]
    assert [candidate.rank for candidate in candidates] == [1, 2, 3]


def test_equal_scores_preserve_candidate_order() -> None:
    candidates = (
        make_result(7, rank=1, raw_similarity=0.9),
        make_result(8, rank=2, raw_similarity=0.8),
        make_result(9, rank=3, raw_similarity=0.7),
    )
    reranker = MappingReranker({7: 0.5, 8: 0.5, 9: 0.2})

    results = reranker.rerank("stable tie", candidates)

    assert [result.entity_id for result in results] == [7, 8, 9]


def test_top_k_is_applied_after_reranking() -> None:
    candidates = (
        make_result(1, rank=1, raw_similarity=0.9),
        make_result(2, rank=2, raw_similarity=0.8),
        make_result(3, rank=3, raw_similarity=0.7),
    )
    reranker = MappingReranker({1: 0.1, 2: 0.2, 3: 0.9})

    results = reranker.rerank("best last", candidates, top_k=2)

    assert [result.entity_id for result in results] == [3, 2]
    assert [result.rank for result in results] == [1, 2]


def test_empty_candidates_do_not_call_model() -> None:
    reranker = MappingReranker({})

    assert reranker.rerank("valid query", []) == ()
    assert reranker.calls == []


@pytest.mark.parametrize(
    ("query", "top_k", "error_type", "message"),
    [
        ("", None, ValueError, "blank"),
        ("   ", None, ValueError, "blank"),
        (123, None, TypeError, "string"),
        ("valid", 0, ValueError, "top_k"),
        ("valid", True, ValueError, "top_k"),
    ],
)
def test_invalid_inputs_fail_before_scoring(
    query: object,
    top_k: object,
    error_type: type[Exception],
    message: str,
) -> None:
    reranker = MappingReranker({1: 0.5})

    with pytest.raises(error_type, match=message):
        reranker.rerank(
            query,  # type: ignore[arg-type]
            [make_result(1, rank=1, raw_similarity=0.9)],
            top_k=top_k,  # type: ignore[arg-type]
        )

    assert reranker.calls == []


def test_non_search_result_candidate_is_rejected() -> None:
    reranker = MappingReranker({})

    with pytest.raises(TypeError, match="SearchResult"):
        reranker.rerank("query", [object()])  # type: ignore[list-item]

    assert reranker.calls == []


@pytest.mark.parametrize(
    ("raw_scores", "message"),
    [
        ([0.1], "shape"),
        ([[0.1, 0.2]], "shape"),
        ([0.1, np.nan], "non-finite"),
        ([0.1, np.inf], "non-finite"),
        (["high", "low"], "numeric"),
    ],
)
def test_invalid_model_scores_are_rejected(raw_scores: object, message: str) -> None:
    reranker = RawOutputReranker(raw_scores)
    candidates = (
        make_result(1, rank=1, raw_similarity=0.9),
        make_result(2, rank=2, raw_similarity=0.8),
    )

    with pytest.raises(InvalidRerankerOutputError, match=message):
        reranker.rerank("query", candidates)


def test_numpy_scores_are_accepted() -> None:
    reranker = RawOutputReranker(np.asarray([0.25, 0.75], dtype=np.float32))
    candidates = (
        make_result(1, rank=1, raw_similarity=0.9),
        make_result(2, rank=2, raw_similarity=0.8),
    )

    results = reranker.rerank("query", candidates)

    assert [result.entity_id for result in results] == [2, 1]
    assert results[0].reranker_score == pytest.approx(0.75)
