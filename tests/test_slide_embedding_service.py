"""End-to-end service tests for slide image embedding and caching."""

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
from app.models import EmbeddingRecord
from app.repositories.course_repository import create_course
from app.repositories.errors import LectureNotFoundError
from app.repositories.lecture_repository import create_lecture
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import CourseCreate, LectureCreate, SlidePageCreate
from app.services.slide_embedding_service import (
    LectureHasNoSlidesError,
    embed_lecture_slides,
)


class FileEmbeddingProvider(EmbeddingProvider):
    """Deterministic image provider with observable inference calls."""

    def __init__(self, *, failing_names: set[str] | None = None) -> None:
        self.image_calls: list[Path] = []
        self.failing_names = failing_names or set()

    @property
    def model_name(self) -> str:
        return "test/slide-model"

    @property
    def dimension(self) -> int:
        return 3

    def _embed_text(self, text: str) -> RawEmbedding:
        return [1.0, 1.0, 0.0]

    def _embed_query(self, query: str) -> RawEmbedding:
        return [1.0, 0.0, 1.0]

    def _embed_image(self, image_path: Path) -> RawEmbedding:
        self.image_calls.append(image_path)
        if image_path.name in self.failing_names:
            raise RuntimeError(f"synthetic failure for {image_path.name}")
        content = image_path.read_bytes()
        return [len(content) + 1.0, (sum(content) % 251) + 1.0, 1.0]


@pytest.fixture
def slide_embedding_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    database_engine = create_database_engine(tmp_path / "slide-embeddings.db")
    init_database(database_engine)
    factory = create_session_factory(database_engine)

    yield factory

    database_engine.dispose()


def create_lecture_with_pages(
    factory: sessionmaker[Session],
    image_root: Path,
    *,
    page_count: int = 2,
) -> tuple[int, list[Path]]:
    image_root.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for page_number in range(1, page_count + 1):
        path = image_root / f"page_{page_number:04d}.png"
        path.write_bytes(f"page {page_number}".encode())
        image_paths.append(path)

    with session_scope(factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Divide and Conquer", lecture_number=3),
        )
        for page_number, image_path in enumerate(image_paths, start=1):
            create_slide_page(
                session,
                lecture.id,
                SlidePageCreate(page_number=page_number, image_path=str(image_path)),
            )
        return lecture.id, image_paths


def test_slide_embeddings_are_generated_then_reused(
    slide_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, image_paths = create_lecture_with_pages(
        slide_embedding_session_factory,
        tmp_path / "rendered",
    )
    provider = FileEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(slide_embedding_session_factory) as session:
        first = embed_lecture_slides(
            session,
            lecture_id,
            provider=provider,
            cache=cache,
        )

    with session_scope(slide_embedding_session_factory) as session:
        second = embed_lecture_slides(
            session,
            lecture_id,
            provider=provider,
            cache=cache,
        )
        record_count = session.scalar(select(func.count()).select_from(EmbeddingRecord))

    assert first.complete is True
    assert first.total_pages == 2
    assert first.generated_pages == 2
    assert first.cached_pages == 0
    assert first.failed_pages == 0
    assert first.successful_pages == 2
    assert first.failures == ()
    assert second.complete is True
    assert second.generated_pages == 0
    assert second.cached_pages == 2
    assert second.failed_pages == 0
    assert provider.image_calls == [path.resolve() for path in image_paths]
    assert record_count == 2
    assert len(list(cache.embedding_dir.rglob("*.npy"))) == 2


def test_changed_slide_recomputes_only_that_page(
    slide_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, image_paths = create_lecture_with_pages(
        slide_embedding_session_factory,
        tmp_path / "rendered",
    )
    provider = FileEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(slide_embedding_session_factory) as session:
        embed_lecture_slides(session, lecture_id, provider=provider, cache=cache)

    image_paths[1].write_bytes(b"updated second slide")
    with session_scope(slide_embedding_session_factory) as session:
        updated = embed_lecture_slides(session, lecture_id, provider=provider, cache=cache)

    assert updated.complete is True
    assert updated.generated_pages == 1
    assert updated.cached_pages == 1
    assert provider.image_calls == [
        image_paths[0].resolve(),
        image_paths[1].resolve(),
        image_paths[1].resolve(),
    ]


def test_page_failure_is_reported_without_blocking_other_pages(
    slide_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, image_paths = create_lecture_with_pages(
        slide_embedding_session_factory,
        tmp_path / "rendered",
        page_count=3,
    )
    provider = FileEmbeddingProvider(failing_names={image_paths[1].name})
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(slide_embedding_session_factory) as session:
        summary = embed_lecture_slides(
            session,
            lecture_id,
            provider=provider,
            cache=cache,
        )
        record_count = session.scalar(select(func.count()).select_from(EmbeddingRecord))

    assert summary.complete is False
    assert summary.total_pages == 3
    assert summary.generated_pages == 2
    assert summary.cached_pages == 0
    assert summary.failed_pages == 1
    assert summary.successful_pages == 2
    assert len(summary.failures) == 1
    assert summary.failures[0].page_number == 2
    assert summary.failures[0].image_path == image_paths[1].resolve()
    assert summary.failures[0].error_type == "RuntimeError"
    assert "synthetic failure" in summary.failures[0].message
    assert record_count == 2


def test_missing_image_is_reported_and_session_remains_usable(
    slide_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    lecture_id, image_paths = create_lecture_with_pages(
        slide_embedding_session_factory,
        tmp_path / "rendered",
    )
    image_paths[0].unlink()
    provider = FileEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with session_scope(slide_embedding_session_factory) as session:
        summary = embed_lecture_slides(
            session,
            lecture_id,
            provider=provider,
            cache=cache,
        )
        record_count = session.scalar(select(func.count()).select_from(EmbeddingRecord))

    assert summary.complete is False
    assert summary.generated_pages == 1
    assert summary.failed_pages == 1
    assert summary.failures[0].error_type == "FileNotFoundError"
    assert provider.image_calls == [image_paths[1].resolve()]
    assert record_count == 1


def test_unknown_lecture_is_rejected(
    slide_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    provider = FileEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")

    with slide_embedding_session_factory() as session:
        with pytest.raises(LectureNotFoundError, match="Lecture 999 does not exist"):
            embed_lecture_slides(session, 999, provider=provider, cache=cache)


def test_lecture_without_pages_is_rejected(
    slide_embedding_session_factory: sessionmaker[Session],
    tmp_path: Path,
) -> None:
    with session_scope(slide_embedding_session_factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Empty lecture", lecture_number=1),
        )
        lecture_id = lecture.id

    provider = FileEmbeddingProvider()
    cache = EmbeddingCache(tmp_path / "embeddings")
    with slide_embedding_session_factory() as session:
        with pytest.raises(LectureHasNoSlidesError, match="ingest a PDF"):
            embed_lecture_slides(session, lecture_id, provider=provider, cache=cache)
