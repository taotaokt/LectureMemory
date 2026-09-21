"""Transactional orchestration for extracting concepts from one lecture."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.concepts import ConceptExtractor, ExtractedConcept
from app.models import Lecture
from app.repositories.concept_repository import (
    assign_lecture_concept,
    delete_lecture_concepts_by_source,
    get_or_create_concept,
    list_lecture_concepts,
)
from app.repositories.errors import LectureNotFoundError
from app.schemas import LectureConceptDisplay

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConceptExtractionSummary:
    """Outcome of extracting and storing concepts for one lecture."""

    lecture_id: int
    slide_texts: int
    notes: int
    extracted_concepts: tuple[ExtractedConcept, ...]
    created_concepts: int
    created_associations: int
    replaced_associations: int


def get_lecture_concept_display(
    session: Session,
    lecture_id: int,
) -> tuple[LectureConceptDisplay, ...]:
    """Return ordered, serializable concept metadata for one lecture."""
    if session.get(Lecture, lecture_id) is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    return tuple(
        LectureConceptDisplay(
            name=association.concept.name,
            normalized_name=association.concept.normalized_name,
            confidence=association.confidence,
            source=association.source,
        )
        for association in list_lecture_concepts(session, lecture_id)
    )


def extract_lecture_concepts(
    session: Session,
    lecture_id: int,
    *,
    extractor: ConceptExtractor,
) -> ConceptExtractionSummary:
    """Extract canonical concepts and atomically replace prior automatic links."""
    statement = (
        select(Lecture)
        .where(Lecture.id == lecture_id)
        .options(
            selectinload(Lecture.slide_pages),
            selectinload(Lecture.notes),
        )
    )
    lecture = session.scalar(statement)
    if lecture is None:
        raise LectureNotFoundError(f"Lecture {lecture_id} does not exist")

    slide_texts = tuple(
        page.text_content
        for page in sorted(lecture.slide_pages, key=lambda page: (page.page_number, page.id))
        if page.text_content and page.text_content.strip()
    )
    notes = tuple(
        note.content
        for note in sorted(lecture.notes, key=lambda note: (note.created_at, note.id))
        if note.content.strip()
    )
    extracted = _deduplicate_concepts(
        extractor.extract(
            lecture_title=lecture.title,
            slide_texts=slide_texts,
            notes=notes,
        )
    )

    created_concepts = 0
    created_associations = 0
    with session.begin_nested():
        replaced_associations = delete_lecture_concepts_by_source(
            session,
            lecture_id,
            source="auto_extracted",
        )
        for candidate in extracted:
            concept, was_created = get_or_create_concept(
                session,
                name=candidate.name,
                normalized_name=candidate.normalized_name,
            )
            _, link_created = assign_lecture_concept(
                session,
                lecture_id=lecture_id,
                concept_id=concept.id,
                confidence=candidate.confidence,
                source="auto_extracted",
            )
            created_concepts += int(was_created)
            created_associations += int(link_created)

    summary = ConceptExtractionSummary(
        lecture_id=lecture_id,
        slide_texts=len(slide_texts),
        notes=len(notes),
        extracted_concepts=extracted,
        created_concepts=created_concepts,
        created_associations=created_associations,
        replaced_associations=replaced_associations,
    )
    logger.info(
        "Extracted lecture concepts",
        extra={
            "lecture_id": lecture_id,
            "slide_texts": summary.slide_texts,
            "notes": summary.notes,
            "concepts": len(summary.extracted_concepts),
            "created_concepts": summary.created_concepts,
            "created_associations": summary.created_associations,
            "replaced_associations": summary.replaced_associations,
        },
    )
    return summary


def _deduplicate_concepts(
    concepts: tuple[ExtractedConcept, ...],
) -> tuple[ExtractedConcept, ...]:
    deduplicated: dict[str, ExtractedConcept] = {}
    for concept in concepts:
        if not isinstance(concept, ExtractedConcept):
            raise TypeError("extractor must return ExtractedConcept values")
        existing = deduplicated.get(concept.normalized_name)
        if existing is None or concept.confidence > existing.confidence:
            deduplicated[concept.normalized_name] = concept
    return tuple(deduplicated.values())
