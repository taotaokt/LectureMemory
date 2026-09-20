"""End-to-end service tests for note text embedding and caching."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.embeddings.base import EmbeddingProvider, RawEmbedding
from app.embeddings.cache import EmbeddingCache
from app.models import EmbeddingRecord, Note
from app.repositories.course_repository import create_course
from app.repositories.errors import LectureNotFoundError
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note, edit_note
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import (
    CourseCreate,
    LectureCreate,
    NoteCreate,
    NoteUpdate,
    SlidePageCreate,
)
from app.services.note_embedding_service import embed_lecture_notes


class TextEmbeddingProvider(EmbeddingProvider):
    """Deterministic text provider with observable inference calls."""

    def __init__(self, *, failing_texts: set[str] | None = None) -> None:
        self.text_calls: list[str] = []
        self.failing_texts = failing_texts or set()

    @property
    def model_name(self) -> str:
        return "test/note-model"

    @property
    def dimension(self) -> int:
        return 3

    def _embed_text(self, text: str) -> RawEmbedding:
        self.text_calls.append(text)
        if text in self.failing_texts:
            raise RuntimeError(f"synthetic failure for {text}")
        encoded = text.encode()
        return [len(encoded) + 1.0, (sum(encoded) % 251) + 1.0, 1.0]

    def _embed_image(self, image_path: Path) -> RawEmbedding:
        return [1.0, 1.0, 0.0]

    def _embed_query(self, query: str) -> RawEmbedding:
        return [1.0, 0.0, 1.0]


@pytest.fixture
def note_embedding_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "note-embeddings.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def create_lecture_with_notes(
    factory: sessionmaker[Session],
    *,
    contents: tuple[str, ...] = ("First explanation", "Second explanation"),
) -> tuple[int, list[int]]:
    with session_scope(factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Divide and Conquer", lecture_number=3),
        )
        notes = [
            create_note(session, lecture.id, NoteCreate(content=content))
            for content in contents
        ]
        return lecture.id, [note.id for note in notes]


def test_note_embeddings_are_generated_then_reused(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, _ = create_lecture_with_notes(note_embedding_session_factory)
    provider = TextEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(note_embedding_session_factory) as session:
        first = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)

    with session_scope(note_embedding_session_factory) as session:
        second = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)
        record_count = session.scalar(select(func.count()).select_from(EmbeddingRecord))

    assert first.complete is True
    assert first.total_notes == 2
    assert first.generated_notes == 2
    assert first.cached_notes == 0
    assert first.failed_notes == 0
    assert first.successful_notes == 2
    assert first.failures == ()
    assert second.complete is True
    assert second.generated_notes == 0
    assert second.cached_notes == 2
    assert second.failed_notes == 0
    assert provider.text_calls == ["First explanation", "Second explanation"]
    assert record_count == 2
    assert len(list(cache.embedding_dir.rglob("*.npy"))) == 2


def test_edited_note_recomputes_only_changed_content(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, note_ids = create_lecture_with_notes(note_embedding_session_factory)
    provider = TextEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(note_embedding_session_factory) as session:
        embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)

    with session_scope(note_embedding_session_factory) as session:
        edit_note(session, note_ids[1], NoteUpdate(content="Revised explanation"))

    with session_scope(note_embedding_session_factory) as session:
        updated = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)

    assert updated.complete is True
    assert updated.generated_notes == 1
    assert updated.cached_notes == 1
    assert provider.text_calls == [
        "First explanation",
        "Second explanation",
        "Revised explanation",
    ]


def test_note_metadata_change_does_not_invalidate_content_embedding(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, note_ids = create_lecture_with_notes(
        note_embedding_session_factory,
        contents=("Stable content",),
    )
    provider = TextEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(note_embedding_session_factory) as session:
        embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)

    with session_scope(note_embedding_session_factory) as session:
        page = create_slide_page(
            session,
            lecture_id,
            SlidePageCreate(page_number=1, image_path="/unused/page.png"),
        )
        edit_note(session, note_ids[0], NoteUpdate(page_id=page.id))

    with session_scope(note_embedding_session_factory) as session:
        repeated = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)

    assert repeated.generated_notes == 0
    assert repeated.cached_notes == 1
    assert provider.text_calls == ["Stable content"]


def test_note_failure_is_reported_without_blocking_other_notes(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    contents = ("First note", "Broken note", "Third note")
    lecture_id, note_ids = create_lecture_with_notes(
        note_embedding_session_factory,
        contents=contents,
    )
    provider = TextEmbeddingProvider(failing_texts={"Broken note"})
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(note_embedding_session_factory) as session:
        summary = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)
        record_count = session.scalar(select(func.count()).select_from(EmbeddingRecord))

    assert summary.complete is False
    assert summary.total_notes == 3
    assert summary.generated_notes == 2
    assert summary.cached_notes == 0
    assert summary.failed_notes == 1
    assert summary.successful_notes == 2
    assert len(summary.failures) == 1
    assert summary.failures[0].note_id == note_ids[1]
    assert summary.failures[0].error_type == "RuntimeError"
    assert "synthetic failure" in summary.failures[0].message
    assert record_count == 2


def test_blank_legacy_note_isolated_and_session_remains_usable(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, _ = create_lecture_with_notes(
        note_embedding_session_factory,
        contents=("Valid note",),
    )
    with session_scope(note_embedding_session_factory) as session:
        blank_note = Note(lecture_id=lecture_id, content="   ")
        session.add(blank_note)
        session.flush()
        blank_note_id = blank_note.id

    provider = TextEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")
    with session_scope(note_embedding_session_factory) as session:
        summary = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)
        record_count = session.scalar(select(func.count()).select_from(EmbeddingRecord))

    assert summary.complete is False
    assert summary.generated_notes == 1
    assert summary.failed_notes == 1
    assert summary.failures[0].note_id == blank_note_id
    assert summary.failures[0].error_type == "ValueError"
    assert provider.text_calls == ["Valid note"]
    assert record_count == 1


def test_lecture_without_notes_returns_complete_empty_summary(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, _ = create_lecture_with_notes(
        note_embedding_session_factory,
        contents=(),
    )
    provider = TextEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with note_embedding_session_factory() as session:
        summary = embed_lecture_notes(session, lecture_id, provider=provider, cache=cache)

    assert summary.complete is True
    assert summary.total_notes == 0
    assert summary.successful_notes == 0
    assert summary.failures == ()
    assert provider.text_calls == []


def test_unknown_lecture_is_rejected(
    note_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    provider = TextEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with note_embedding_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            embed_lecture_notes(session, 999, provider=provider, cache=cache)
