"""Persistence operations for embedding cache metadata."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import EmbeddingRecord, utc_now
from app.schemas import EmbeddingEntityType, EmbeddingRecordCreate


def get_embedding_record(
    session: Session,
    *,
    entity_type: EmbeddingEntityType,
    entity_id: int,
    model_name: str,
    dimension: int,
) -> EmbeddingRecord | None:
    """Return cache metadata for an entity/model/dimension combination."""
    statement = select(EmbeddingRecord).where(
        EmbeddingRecord.entity_type == entity_type,
        EmbeddingRecord.entity_id == entity_id,
        EmbeddingRecord.model_name == model_name,
        EmbeddingRecord.dimension == dimension,
    )
    return session.scalar(statement)


def upsert_embedding_record(
    session: Session,
    record_data: EmbeddingRecordCreate,
) -> EmbeddingRecord:
    """Create cache metadata or update an existing entity/model record."""
    record = get_embedding_record(
        session,
        entity_type=record_data.entity_type,
        entity_id=record_data.entity_id,
        model_name=record_data.model_name,
        dimension=record_data.dimension,
    )
    if record is None:
        record = EmbeddingRecord(**record_data.model_dump())
        session.add(record)
    else:
        record.embedding_path = record_data.embedding_path
        record.content_hash = record_data.content_hash
        record.created_at = utc_now()
    session.flush()
    return record


def delete_embedding_record(session: Session, record_id: int) -> bool:
    """Delete cache metadata and report whether it existed."""
    record = session.get(EmbeddingRecord, record_id)
    if record is None:
        return False
    session.delete(record)
    session.flush()
    return True
