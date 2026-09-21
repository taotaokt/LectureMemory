"""Persistence and orchestration tests for lecture concept extraction."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.concepts import ConceptExtractor, ExtractedConcept
from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.models import Concept, LectureConcept
from app.repositories.concept_repository import (
    assign_lecture_concept,
    get_or_create_concept,
    list_lecture_concepts,
)
from app.repositories.course_repository import create_course
from app.repositories.errors import LectureNotFoundError
from app.repositories.lecture_repository import create_lecture, delete_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import CourseCreate, LectureCreate, NoteCreate, SlidePageCreate
from app.services.concept_service import (
    extract_lecture_concepts,
    get_lecture_concept_display,
)


class FixedConceptExtractor(ConceptExtractor):
    """Return configured concepts while recording gathered lecture sources."""

    def __init__(self, concepts: Sequence[ExtractedConcept]) -> None:
        self.concepts = tuple(concepts)
        self.calls: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    def extract(
        self,
        *,
        lecture_title: str,
        slide_texts: Sequence[str],
        notes: Sequence[str],
    ) -> tuple[ExtractedConcept, ...]:
        self.calls.append((lecture_title, tuple(slide_texts), tuple(notes)))
        return self.concepts


@pytest.fixture
def concept_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "concepts.db")
    init_database(engine)
    factory = create_session_factory(engine)

    yield factory

    engine.dispose()


def create_concept_fixture(factory: sessionmaker[Session]) -> int:
    with session_scope(factory) as session:
        course = create_course(
            session,
            CourseCreate(name="Algorithms", code="CS344"),
        )
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Divide and Conquer", lecture_number=3),
        )
        create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=2,
                image_path="page-2.png",
                text_content="Recurrence Relation",
            ),
        )
        create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=1,
                image_path="page-1.png",
                text_content="Karatsuba Multiplication",
            ),
        )
        create_note(
            session,
            lecture.id,
            NoteCreate(content="Master Theorem"),
        )
        return lecture.id


def test_extract_lecture_concepts_gathers_sources_and_persists_normalized_names(
    concept_session_factory: sessionmaker[Session],
) -> None:
    lecture_id = create_concept_fixture(concept_session_factory)
    extractor = FixedConceptExtractor(
        [
            ExtractedConcept("Karatsuba Multiplication", 0.92),
            ExtractedConcept("  karatsuba   multiplication ", 0.7),
            ExtractedConcept("Master Theorem", 0.8),
        ]
    )

    with session_scope(concept_session_factory) as session:
        summary = extract_lecture_concepts(
            session,
            lecture_id,
            extractor=extractor,
        )

    assert extractor.calls == [
        (
            "Divide and Conquer",
            ("Karatsuba Multiplication", "Recurrence Relation"),
            ("Master Theorem",),
        )
    ]
    assert summary.slide_texts == 2
    assert summary.notes == 1
    assert len(summary.extracted_concepts) == 2
    assert summary.created_concepts == 2
    assert summary.created_associations == 2
    assert summary.replaced_associations == 0

    with concept_session_factory() as session:
        associations = list_lecture_concepts(session, lecture_id)
        display = get_lecture_concept_display(session, lecture_id)

    assert [link.concept.normalized_name for link in associations] == [
        "karatsuba multiplication",
        "master theorem",
    ]
    assert [link.confidence for link in associations] == pytest.approx([0.92, 0.8])
    assert {link.source for link in associations} == {"auto_extracted"}
    assert [item.name for item in display] == [
        "Karatsuba Multiplication",
        "Master Theorem",
    ]
    assert display[0].model_dump() == {
        "name": "Karatsuba Multiplication",
        "normalized_name": "karatsuba multiplication",
        "confidence": 0.92,
        "source": "auto_extracted",
    }


def test_reextract_replaces_automatic_links_but_preserves_manual_links(
    concept_session_factory: sessionmaker[Session],
) -> None:
    lecture_id = create_concept_fixture(concept_session_factory)
    first = FixedConceptExtractor(
        [
            ExtractedConcept("Karatsuba Multiplication", 0.9),
            ExtractedConcept("Master Theorem", 0.8),
        ]
    )
    second = FixedConceptExtractor([ExtractedConcept("Recurrence Relation", 0.95)])

    with session_scope(concept_session_factory) as session:
        extract_lecture_concepts(session, lecture_id, extractor=first)
        manual, _ = get_or_create_concept(
            session,
            name="Asymptotic Analysis",
            normalized_name="asymptotic analysis",
        )
        assign_lecture_concept(
            session,
            lecture_id=lecture_id,
            concept_id=manual.id,
            confidence=1.0,
            source="manual",
        )

    with session_scope(concept_session_factory) as session:
        summary = extract_lecture_concepts(session, lecture_id, extractor=second)

    assert summary.replaced_associations == 2
    assert summary.created_concepts == 1
    assert summary.created_associations == 1
    with concept_session_factory() as session:
        associations = list_lecture_concepts(session, lecture_id)

    assert [(link.concept.normalized_name, link.source) for link in associations] == [
        ("asymptotic analysis", "manual"),
        ("recurrence relation", "auto_extracted"),
    ]


def test_concepts_are_reused_across_lectures(
    concept_session_factory: sessionmaker[Session],
) -> None:
    first_lecture_id = create_concept_fixture(concept_session_factory)
    with session_scope(concept_session_factory) as session:
        course = create_course(session, CourseCreate(name="Complexity", code="CS500"))
        second_lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Analysis", lecture_number=1),
        )
        second_lecture_id = second_lecture.id

    extractor = FixedConceptExtractor([ExtractedConcept("Master Theorem", 0.8)])
    with session_scope(concept_session_factory) as session:
        first = extract_lecture_concepts(session, first_lecture_id, extractor=extractor)
        second = extract_lecture_concepts(session, second_lecture_id, extractor=extractor)

    assert first.created_concepts == 1
    assert second.created_concepts == 0
    with concept_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Concept)) == 1
        assert session.scalar(select(func.count()).select_from(LectureConcept)) == 2


def test_deleting_lecture_cascades_association_but_keeps_shared_concept(
    concept_session_factory: sessionmaker[Session],
) -> None:
    lecture_id = create_concept_fixture(concept_session_factory)
    extractor = FixedConceptExtractor([ExtractedConcept("Master Theorem", 0.8)])
    with session_scope(concept_session_factory) as session:
        extract_lecture_concepts(session, lecture_id, extractor=extractor)
        assert delete_lecture(session, lecture_id) is True

    with concept_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(LectureConcept)) == 0
        assert session.scalar(select(func.count()).select_from(Concept)) == 1


def test_extract_lecture_concepts_rejects_unknown_lecture(
    concept_session_factory: sessionmaker[Session],
) -> None:
    extractor = FixedConceptExtractor([])

    with concept_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            extract_lecture_concepts(session, 999, extractor=extractor)

    assert extractor.calls == []


def test_concept_display_returns_empty_for_lecture_without_concepts(
    concept_session_factory: sessionmaker[Session],
) -> None:
    lecture_id = create_concept_fixture(concept_session_factory)

    with concept_session_factory() as session:
        assert get_lecture_concept_display(session, lecture_id) == ()


def test_concept_display_rejects_unknown_lecture(
    concept_session_factory: sessionmaker[Session],
) -> None:
    with concept_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            get_lecture_concept_display(session, 999)
