"""Model-independent contract for reranking normalized search results."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from app.schemas import SearchResult

RawRerankerScores = Sequence[float] | NDArray[np.floating]

logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    """Base error raised by reranker implementations."""


class InvalidRerankerOutputError(RerankerError):
    """Raised when a reranker returns malformed candidate scores."""


class Reranker(ABC):
    """Stable interface for scoring and reordering retrieval candidates."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the stable model identifier used for diagnostics."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        *,
        top_k: int | None = None,
        reranker_weight: float = 1.0,
    ) -> tuple[SearchResult, ...]:
        """Score candidates and optionally fuse retrieval and reranker ranks."""
        cleaned_query = _validate_query(query)
        _validate_top_k(top_k)
        _validate_reranker_weight(reranker_weight)
        candidate_batch = tuple(candidates)
        _validate_candidates(candidate_batch)
        if not candidate_batch:
            return ()

        raw_scores = self._score(cleaned_query, candidate_batch)
        scores = _validate_scores(raw_scores, expected_count=len(candidate_batch))
        scored_candidates = tuple(zip(candidate_batch, scores, strict=True))
        reranked = sorted(
            enumerate(scored_candidates),
            key=lambda item: -item[1][1],
        )
        reranker_ranks = {
            candidate_index: rank
            for rank, (candidate_index, _) in enumerate(reranked, start=1)
        }
        ranked = sorted(
            enumerate(scored_candidates),
            key=lambda item: -_fused_reciprocal_rank_score(
                retrieval_rank=item[0] + 1,
                reranker_rank=reranker_ranks[item[0]],
                reranker_weight=reranker_weight,
            ),
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        results = tuple(
            candidate.model_copy(
                update={
                    "rank": rank,
                    "reranker_score": score,
                }
            )
            for rank, (_, (candidate, score)) in enumerate(ranked, start=1)
        )
        logger.info(
            "Reranked search results",
            extra={
                "model_name": self.model_name,
                "candidate_count": len(candidate_batch),
                "returned_count": len(results),
                "reranker_weight": reranker_weight,
            },
        )
        return results

    @abstractmethod
    def _score(
        self,
        query: str,
        candidates: tuple[SearchResult, ...],
    ) -> RawRerankerScores:
        """Return one relevance score for each candidate in input order."""


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be blank")
    return cleaned_query


def _validate_top_k(top_k: int | None) -> None:
    if top_k is None:
        return
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer or None")


def _validate_reranker_weight(reranker_weight: float) -> None:
    if (
        not isinstance(reranker_weight, (int, float))
        or isinstance(reranker_weight, bool)
        or not 0.0 <= float(reranker_weight) <= 1.0
    ):
        raise ValueError("reranker_weight must be a number between 0 and 1")


def _fused_reciprocal_rank_score(
    *,
    retrieval_rank: int,
    reranker_rank: int,
    reranker_weight: float,
) -> float:
    """Combine two rankings without assuming comparable model-score scales."""
    return (
        (1.0 - reranker_weight) / retrieval_rank
        + reranker_weight / reranker_rank
    )


def _validate_candidates(candidates: tuple[SearchResult, ...]) -> None:
    if not all(isinstance(candidate, SearchResult) for candidate in candidates):
        raise TypeError("candidates must contain SearchResult values")


def _validate_scores(
    raw_scores: RawRerankerScores,
    *,
    expected_count: int,
) -> tuple[float, ...]:
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidRerankerOutputError(
            f"Reranker scores must be numeric: {exc}"
        ) from exc
    if scores.ndim != 1 or scores.shape != (expected_count,):
        raise InvalidRerankerOutputError(
            f"Reranker returned score shape {scores.shape}; expected ({expected_count},)"
        )
    if not np.isfinite(scores).all():
        raise InvalidRerankerOutputError("Reranker scores contain non-finite values")
    return tuple(float(score) for score in scores)
