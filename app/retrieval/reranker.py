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
    ) -> tuple[SearchResult, ...]:
        """Score candidates, preserve embedding scores, and return a stable ranking."""
        cleaned_query = _validate_query(query)
        _validate_top_k(top_k)
        candidate_batch = tuple(candidates)
        _validate_candidates(candidate_batch)
        if not candidate_batch:
            return ()

        raw_scores = self._score(cleaned_query, candidate_batch)
        scores = _validate_scores(raw_scores, expected_count=len(candidate_batch))
        ranked = sorted(
            zip(candidate_batch, scores, strict=True),
            key=lambda item: -item[1],
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
            for rank, (candidate, score) in enumerate(ranked, start=1)
        )
        logger.info(
            "Reranked search results",
            extra={
                "model_name": self.model_name,
                "candidate_count": len(candidate_batch),
                "returned_count": len(results),
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
