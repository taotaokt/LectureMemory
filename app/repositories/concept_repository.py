"""Persistence operations for normalized lecture concepts."""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, joinedload

from app.models import Concept, LectureConcept
from app.schemas import ConceptSource


def get_concept_by_normalized_name(
    session: Session,
    normalized_name: str,
) -> Concept | None:
    """Return a concept by canonical name, or ``None`` when absent."""
    return session.scalar(
        select(Concept).where(Concept.normalized_name == normalized_name)
    )


def get_or_create_concept(
    session: Session,
    *,
    name: str,
    normalized_name: str,
) -> tuple[Concept, bool]:
    """Reuse a canonical concept or create and flush a new one."""
    concept = get_concept_by_normalized_name(session, normalized_name)
    if concept is not None:
        return concept, False

    concept = Concept(name=name, normalized_name=normalized_name)
    session.add(concept)
    session.flush()
    return concept, True


def assign_lecture_concept(
    session: Session,
    *,
    lecture_id: int,
    concept_id: int,
    confidence: float,
    source: ConceptSource,
) -> tuple[LectureConcept, bool]:
    """Create an association or update its extraction metadata."""
    association = session.get(LectureConcept, (lecture_id, concept_id))
    if association is not None:
        if association.source != "manual":
            association.confidence = confidence
            association.source = source
            session.flush()
        return association, False

    association = LectureConcept(
        lecture_id=lecture_id,
        concept_id=concept_id,
        confidence=confidence,
        source=source,
    )
    session.add(association)
    session.flush()
    return association, True


def delete_lecture_concepts_by_source(
    session: Session,
    lecture_id: int,
    *,
    source: ConceptSource,
) -> int:
    """Delete one source's associations and return the affected row count."""
    result = session.execute(
        delete(LectureConcept).where(
            LectureConcept.lecture_id == lecture_id,
            LectureConcept.source == source,
        )
    )
    session.flush()
    return result.rowcount or 0


def list_lecture_concepts(
    session: Session,
    lecture_id: int,
) -> list[LectureConcept]:
    """List a lecture's concepts by confidence, then normalized name."""
    statement = (
        select(LectureConcept)
        .join(LectureConcept.concept)
        .where(LectureConcept.lecture_id == lecture_id)
        .options(joinedload(LectureConcept.concept))
        .order_by(LectureConcept.confidence.desc(), Concept.normalized_name)
    )
    return list(session.scalars(statement).all())
