"""Model-independent concept extraction with a conservative local baseline."""

from __future__ import annotations

import math
import re
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

_TOKEN_PATTERN = re.compile(r"[^\W_][\w+#-]*", re.UNICODE)
_SEPARATOR_PATTERN = re.compile(r"[^\w+#]+", re.UNICODE)
_CONNECTORS = {"and", "for", "in", "of", "on", "the", "to", "with"}
_STOP_WORDS = _CONNECTORS | {
    "about",
    "after",
    "against",
    "also",
    "are",
    "because",
    "before",
    "between",
    "both",
    "but",
    "can",
    "compare",
    "concept",
    "could",
    "does",
    "each",
    "example",
    "examples",
    "from",
    "have",
    "how",
    "important",
    "into",
    "lecture",
    "more",
    "most",
    "note",
    "notes",
    "other",
    "our",
    "review",
    "show",
    "shows",
    "slide",
    "slides",
    "than",
    "that",
    "their",
    "then",
    "these",
    "this",
    "three",
    "through",
    "two",
    "use",
    "used",
    "uses",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "why",
    "will",
}
_GENERIC_TITLE_PATTERN = re.compile(r"^(?:lecture|class|session)\s*\d*$", re.IGNORECASE)


def normalize_concept_name(name: str) -> str:
    """Return a stable identifier for display-name variants of one concept."""
    if not isinstance(name, str):
        raise TypeError("concept name must be a string")
    normalized = unicodedata.normalize("NFKC", name).casefold().strip()
    normalized = _SEPARATOR_PATTERN.sub(" ", normalized)
    return " ".join(normalized.split())


@dataclass(frozen=True, slots=True)
class ExtractedConcept:
    """One normalized concept proposed by an extractor."""

    name: str
    confidence: float
    normalized_name: str = field(init=False)

    def __post_init__(self) -> None:
        cleaned_name = " ".join(self.name.split()) if isinstance(self.name, str) else ""
        normalized_name = normalize_concept_name(cleaned_name)
        if not normalized_name:
            raise ValueError("concept name must not be blank")
        if not isinstance(self.confidence, (int, float)) or isinstance(
            self.confidence, bool
        ):
            raise TypeError("concept confidence must be numeric")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("concept confidence must be between 0 and 1")
        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "normalized_name", normalized_name)


class ConceptExtractor(ABC):
    """Stable interface for extracting concepts from one lecture's text."""

    @abstractmethod
    def extract(
        self,
        *,
        lecture_title: str,
        slide_texts: Sequence[str],
        notes: Sequence[str],
    ) -> tuple[ExtractedConcept, ...]:
        """Return normalized concepts ordered by descending importance."""


@dataclass(slots=True)
class _CandidateStats:
    displays: Counter[str] = field(default_factory=Counter)
    document_ids: set[int] = field(default_factory=set)
    mentions: int = 0
    title: bool = False
    phrase: bool = False


class RuleBasedConceptExtractor(ConceptExtractor):
    """Extract headings, proper-name phrases, and repeated technical terms locally."""

    def __init__(self, *, max_concepts: int = 12) -> None:
        if (
            not isinstance(max_concepts, int)
            or isinstance(max_concepts, bool)
            or max_concepts <= 0
        ):
            raise ValueError("max_concepts must be a positive integer")
        self._max_concepts = max_concepts

    def extract(
        self,
        *,
        lecture_title: str,
        slide_texts: Sequence[str],
        notes: Sequence[str],
    ) -> tuple[ExtractedConcept, ...]:
        title = _clean_text(lecture_title, field_name="lecture_title")
        slides = _clean_collection(slide_texts, field_name="slide_texts")
        note_texts = _clean_collection(notes, field_name="notes")
        documents = (title, *slides, *note_texts)
        candidates: dict[str, _CandidateStats] = {}

        if not _GENERIC_TITLE_PATTERN.fullmatch(title):
            _record_candidate(candidates, title, document_id=0, title=True, phrase=True)

        token_documents: dict[str, set[int]] = {}
        token_displays: dict[str, Counter[str]] = {}
        for document_id, document in enumerate(documents):
            for phrase in _proper_name_phrases(document):
                _record_candidate(
                    candidates,
                    phrase,
                    document_id=document_id,
                    phrase=True,
                )
            for token in _TOKEN_PATTERN.findall(document):
                normalized_token = normalize_concept_name(token)
                if not _is_content_token(normalized_token):
                    continue
                token_documents.setdefault(normalized_token, set()).add(document_id)
                token_displays.setdefault(normalized_token, Counter())[token] += 1

        for normalized_token, document_ids in token_documents.items():
            if len(document_ids) < 2:
                continue
            display = token_displays[normalized_token].most_common(1)[0][0]
            for document_id in document_ids:
                _record_candidate(candidates, display, document_id=document_id)

        ranked = sorted(
            (
                ExtractedConcept(
                    name=_preferred_display(stats),
                    confidence=_confidence(stats),
                )
                for stats in candidates.values()
            ),
            key=lambda concept: (-concept.confidence, concept.normalized_name),
        )
        selected: list[ExtractedConcept] = []
        for concept in ranked:
            if _is_redundant(concept, selected):
                continue
            selected.append(concept)
            if len(selected) == self._max_concepts:
                break
        return tuple(selected)


def _clean_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be blank")
    return cleaned


def _clean_collection(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must contain only strings")
        compact = "\n".join(
            " ".join(line.split()) for line in value.strip().splitlines()
        ).strip()
        if compact:
            cleaned.append(compact)
    return tuple(cleaned)


def _proper_name_phrases(text: str) -> tuple[str, ...]:
    phrases: list[str] = []
    for segment in re.split(r"[\n.!?;:]+", text):
        tokens = _TOKEN_PATTERN.findall(segment)
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not _is_named_token(token):
                index += 1
                continue

            phrase = [token]
            named_count = 1
            cursor = index + 1
            while cursor < len(tokens):
                candidate = tokens[cursor]
                if _is_named_token(candidate):
                    phrase.append(candidate)
                    named_count += 1
                    cursor += 1
                    continue
                if (
                    candidate.casefold() in _CONNECTORS
                    and cursor + 1 < len(tokens)
                    and _is_named_token(tokens[cursor + 1])
                ):
                    phrase.append(candidate)
                    cursor += 1
                    continue
                break

            if named_count >= 2 or (token.isupper() and len(token) >= 2):
                phrases.append(" ".join(phrase))
            index = max(cursor, index + 1)
    return tuple(phrases)


def _is_named_token(token: str) -> bool:
    return (
        bool(token)
        and token.casefold() not in _STOP_WORDS
        and (token[0].isupper() or (token.isupper() and len(token) >= 2))
    )


def _is_content_token(normalized_token: str) -> bool:
    return (
        len(normalized_token) >= 4
        and normalized_token not in _STOP_WORDS
        and not normalized_token.isdigit()
    )


def _record_candidate(
    candidates: dict[str, _CandidateStats],
    display: str,
    *,
    document_id: int,
    title: bool = False,
    phrase: bool = False,
) -> None:
    cleaned_display = " ".join(display.split()).strip("-–—:;,.")
    normalized = normalize_concept_name(cleaned_display)
    if not normalized:
        return
    stats = candidates.setdefault(normalized, _CandidateStats())
    stats.displays[cleaned_display] += 1
    stats.document_ids.add(document_id)
    stats.mentions += 1
    stats.title = stats.title or title
    stats.phrase = stats.phrase or phrase


def _preferred_display(stats: _CandidateStats) -> str:
    return sorted(
        stats.displays,
        key=lambda display: (-stats.displays[display], -len(display), display.casefold()),
    )[0]


def _confidence(stats: _CandidateStats) -> float:
    if stats.title:
        return 1.0
    score = 0.35
    if stats.phrase:
        score += 0.25
    score += min(len(stats.document_ids), 3) * 0.08
    score += min(stats.mentions, 3) * 0.03
    return round(min(score, 0.98), 4)


def _is_redundant(
    candidate: ExtractedConcept,
    selected: Sequence[ExtractedConcept],
) -> bool:
    candidate_tokens = candidate.normalized_name.split()
    for existing in selected:
        existing_tokens = existing.normalized_name.split()
        shorter, longer = sorted(
            (candidate_tokens, existing_tokens),
            key=len,
        )
        for start in range(len(longer) - len(shorter) + 1):
            if longer[start : start + len(shorter)] == shorter:
                return True
    return False
