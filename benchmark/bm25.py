"""Dependency-free BM25 baseline over extracted slide text and attached notes."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload, selectinload

from app.models import Lecture, SlidePage

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


@dataclass(frozen=True, slots=True)
class BM25Document:
    """One page-level lexical document."""

    page_id: int
    lecture_id: int
    course_id: int
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class BM25SearchResult:
    """One ranked page returned by the lexical baseline."""

    page_id: int
    lecture_id: int
    course_id: int
    page_number: int
    score: float
    rank: int


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize English words and Chinese unigrams/bigrams without external models."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text.casefold()):
        value = match.group()
        if not _CJK_PATTERN.fullmatch(value):
            tokens.append(value)
            continue
        characters = tuple(value)
        tokens.extend(characters)
        tokens.extend(
            first + second for first, second in zip(characters, characters[1:], strict=False)
        )
    return tuple(tokens)


def load_bm25_documents(session: Session) -> tuple[BM25Document, ...]:
    """Build page documents from extracted text and notes attached to each page."""
    pages = session.scalars(
        select(SlidePage)
        .options(
            joinedload(SlidePage.lecture).joinedload(Lecture.course),
            selectinload(SlidePage.notes),
        )
        .order_by(SlidePage.lecture_id, SlidePage.page_number, SlidePage.id)
    ).all()
    documents: list[BM25Document] = []
    for page in pages:
        sections = [page.text_content or ""]
        sections.extend(
            note.content
            for note in sorted(page.notes, key=lambda note: (note.created_at, note.id))
        )
        documents.append(
            BM25Document(
                page_id=page.id,
                lecture_id=page.lecture_id,
                course_id=page.lecture.course_id,
                page_number=page.page_number,
                text="\n".join(sections),
            )
        )
    return tuple(documents)


class BM25Index:
    """Small immutable Okapi BM25 index for the evaluation corpus."""

    def __init__(
        self,
        documents: tuple[BM25Document, ...] | list[BM25Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.documents = tuple(documents)
        if len({document.page_id for document in self.documents}) != len(self.documents):
            raise ValueError("documents must have unique page IDs")
        self.k1 = float(k1)
        self.b = float(b)
        self._term_frequencies = tuple(
            Counter(tokenize(document.text)) for document in self.documents
        )
        self._document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self._term_frequencies
        )
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequencies.update(frequencies.keys())
        document_count = len(self.documents)
        self._inverse_document_frequencies = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        course_id: int | None = None,
    ) -> tuple[BM25SearchResult, ...]:
        """Return positive-scoring pages in stable score order."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if course_id is not None and (
            not isinstance(course_id, int) or isinstance(course_id, bool) or course_id <= 0
        ):
            raise ValueError("course_id must be a positive integer or None")
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return ()

        scores: list[tuple[int, float]] = []
        for position, document in enumerate(self.documents):
            if course_id is not None and document.course_id != course_id:
                continue
            score = self._score_document(position, query_terms)
            if score > 0:
                scores.append((position, score))
        scores.sort(key=lambda item: (-item[1], self.documents[item[0]].page_id))

        return tuple(
            BM25SearchResult(
                page_id=self.documents[position].page_id,
                lecture_id=self.documents[position].lecture_id,
                course_id=self.documents[position].course_id,
                page_number=self.documents[position].page_number,
                score=score,
                rank=rank,
            )
            for rank, (position, score) in enumerate(scores[:top_k], start=1)
        )

    def _score_document(self, position: int, query_terms: tuple[str, ...]) -> float:
        frequencies = self._term_frequencies[position]
        document_length = self._document_lengths[position]
        length_ratio = (
            document_length / self._average_document_length
            if self._average_document_length
            else 0.0
        )
        score = 0.0
        for term, query_frequency in Counter(query_terms).items():
            term_frequency = frequencies.get(term, 0)
            inverse_document_frequency = self._inverse_document_frequencies.get(term)
            if not term_frequency or inverse_document_frequency is None:
                continue
            denominator = term_frequency + self.k1 * (1 - self.b + self.b * length_ratio)
            score += (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (self.k1 + 1)
                / denominator
            )
        return score
