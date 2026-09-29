"""End-to-end tests for rebuilding the unified persisted search index."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.embeddings import EmbeddingCache
from app.embeddings.base import EmbeddingProvider, RawEmbedding
from app.models import Note
from app.repositories.course_repository import create_course
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.retrieval import FaissVectorIndex
from app.schemas import CourseCreate, LectureCreate, NoteCreate, SlidePageCreate
from app.services.indexing_service import build_search_index


class IndexEmbeddingProvider(EmbeddingProvider):
    """Small deterministic provider with observable calls and failures."""

    def __init__(
        self,
        *,
        failing_images: set[str] | None = None,
        failing_texts: set[str] | None = None,
    ) -> None:
        self.image_calls: list[Path] = []
        self.text_calls: list[str] = []
        self.failing_images = failing_images or set()
        self.failing_texts = failing_texts or set()

    @property
    def model_name(self) -> str:
        return "test/index-model"

    @property
    def dimension(self) -> int:
        return 3

    def _embed_image(self, image_path: Path) -> RawEmbedding:
        self.image_calls.append(image_path)
        if image_path.name in self.failing_images:
            raise RuntimeError(f"synthetic image failure for {image_path.name}")
        content = image_path.read_bytes()
        return [len(content) + 1.0, 1.0, 1.0]

    def _embed_text(self, text: str) -> RawEmbedding:
        self.text_calls.append(text)
        if text in self.failing_texts:
            raise RuntimeError(f"synthetic text failure for {text}")
        return [1.0, len(text.encode()) + 1.0, 1.0]

    def _embed_query(self, query: str) -> RawEmbedding:
        return [1.0, 1.0, len(query.encode()) + 1.0]


@pytest.fixture
def indexing_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "indexing.db")
    init_database(engine)
    factory = create_session_factory(engine)

    yield factory

    engine.dispose()


def create_search_content(
    factory: sessionmaker[Session],
    image_root: Path,
) -> tuple[list[int], list[int], list[Path]]:
    image_root.mkdir(parents=True)
    images = [image_root / "page_0001.png", image_root / "page_0002.png"]
    for page_number, image in enumerate(images, start=1):
        image.write_bytes(f"slide {page_number}".encode())

    with session_scope(factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Search Trees", lecture_number=4),
        )
        pages = [
            create_slide_page(
                session,
                lecture.id,
                SlidePageCreate(page_number=number, image_path=str(image)),
            )
            for number, image in enumerate(images, start=1)
        ]
        notes = [
            create_note(session, lecture.id, NoteCreate(content="Balanced tree")),
            create_note(session, lecture.id, NoteCreate(content="Rotation invariant")),
        ]
        return [page.id for page in pages], [note.id for note in notes], images


def test_build_generates_then_reuses_embeddings(
    indexing_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    slide_ids, note_ids, images = create_search_content(
        indexing_session_factory,
        tmp_path / "slides",
    )
    provider = IndexEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")
    index_dir = tmp_path / "index"

    with session_scope(indexing_session_factory) as session:
        first = build_search_index(
            session,
            provider=provider,
            cache=cache,
            index_dir=index_dir,
        )
    with session_scope(indexing_session_factory) as session:
        second = build_search_index(
            session,
            provider=provider,
            cache=cache,
            index_dir=index_dir,
        )

    index = FaissVectorIndex.load(index_dir)
    indexed_entities = {
        (result.entity_type, result.entity_id)
        for result in index.search([1.0, 1.0, 1.0], top_k=10)
    }

    assert first.complete is True
    assert first.total_slides == 2
    assert first.total_notes == 2
    assert first.indexed_entities == 4
    assert first.generated_embeddings == 4
    assert first.cached_embeddings == 0
    assert second.complete is True
    assert second.generated_embeddings == 0
    assert second.cached_embeddings == 4
    assert provider.image_calls == [image.resolve() for image in images]
    assert provider.text_calls == ["Balanced tree", "Rotation invariant"]
    assert index.count == 4
    assert indexed_entities == {
        *(('slide_page', slide_id) for slide_id in slide_ids),
        *(('note', note_id) for note_id in note_ids),
    }


def test_full_rebuild_drops_deleted_entities(
    indexing_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    _, note_ids, _ = create_search_content(
        indexing_session_factory,
        tmp_path / "slides",
    )
    provider = IndexEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")
    index_dir = tmp_path / "index"

    with session_scope(indexing_session_factory) as session:
        build_search_index(
            session,
            provider=provider,
            cache=cache,
            index_dir=index_dir,
        )
    with session_scope(indexing_session_factory) as session:
        session.execute(delete(Note).where(Note.id == note_ids[0]))
    with session_scope(indexing_session_factory) as session:
        summary = build_search_index(
            session,
            provider=provider,
            cache=cache,
            index_dir=index_dir,
        )

    index = FaissVectorIndex.load(index_dir)
    indexed_entities = {
        (result.entity_type, result.entity_id)
        for result in index.search([1.0, 1.0, 1.0], top_k=10)
    }

    assert summary.total_entities == 3
    assert summary.indexed_entities == 3
    assert index.count == 3
    assert ("note", note_ids[0]) not in indexed_entities
    assert ("note", note_ids[1]) in indexed_entities


def test_entity_failure_is_isolated_and_partial_index_is_saved(
    indexing_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    slide_ids, note_ids, images = create_search_content(
        indexing_session_factory,
        tmp_path / "slides",
    )
    provider = IndexEmbeddingProvider(
        failing_images={images[1].name},
        failing_texts={"Balanced tree"},
    )
    index_dir = tmp_path / "index"

    with session_scope(indexing_session_factory) as session:
        summary = build_search_index(
            session,
            provider=provider,
            cache=EmbeddingCache(tmp_path / "embeddings"),
            index_dir=index_dir,
        )

    index = FaissVectorIndex.load(index_dir)
    indexed_entities = {
        (result.entity_type, result.entity_id)
        for result in index.search([1.0, 1.0, 1.0], top_k=10)
    }

    assert summary.complete is False
    assert summary.indexed_slides == 1
    assert summary.indexed_notes == 1
    assert summary.generated_embeddings == 2
    assert summary.failed_entities == 2
    assert {(failure.entity_type, failure.entity_id) for failure in summary.failures} == {
        ("slide_page", slide_ids[1]),
        ("note", note_ids[0]),
    }
    assert index.count == 2
    assert indexed_entities == {
        ("slide_page", slide_ids[0]),
        ("note", note_ids[1]),
    }


def test_empty_database_produces_loadable_empty_index(
    indexing_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / "index"

    with session_scope(indexing_session_factory) as session:
        summary = build_search_index(
            session,
            provider=IndexEmbeddingProvider(),
            cache=EmbeddingCache(tmp_path / "embeddings"),
            index_dir=index_dir,
        )

    index = FaissVectorIndex.load(index_dir)
    assert summary.complete is True
    assert summary.total_entities == 0
    assert summary.indexed_entities == 0
    assert summary.generated_embeddings == 0
    assert summary.cached_embeddings == 0
    assert index.count == 0
