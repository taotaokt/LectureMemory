"""Unified retrieval tests for mixed slide-page and note results."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.embeddings.base import EmbeddingProvider, RawEmbedding
from app.repositories.course_repository import create_course
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.retrieval import (
    FaissVectorIndex,
    IndexedEntity,
    RetrievalConfigurationError,
    search_lecture_memory,
)
from app.schemas import CourseCreate, LectureCreate, NoteCreate, SlidePageCreate


class QueryEmbeddingProvider(EmbeddingProvider):
    """Small query provider with observable inference calls."""

    def __init__(self, *, dimension: int = 3) -> None:
        self._dimension = dimension
        self.query_calls: list[str] = []

    @property
    def model_name(self) -> str:
        return "test/retrieval-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_text(self, text: str) -> RawEmbedding:
        return [1.0] + [0.0] * (self.dimension - 1)

    def _embed_image(self, image_path: Path) -> RawEmbedding:
        return [1.0] + [0.0] * (self.dimension - 1)

    def _embed_query(self, query: str) -> RawEmbedding:
        self.query_calls.append(query)
        return [1.0] + [0.0] * (self.dimension - 1)


@pytest.fixture
def retrieval_session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "retrieval.db")
    init_database(engine)
    factory = create_session_factory(engine)

    yield factory

    engine.dispose()


def create_search_fixture(factory: sessionmaker[Session]) -> dict[str, int]:
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
        first_page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=17,
                image_path="/rendered/page-17.png",
                text_content="  Three recursive calls\nreduce the exponent.  ",
            ),
        )
        second_page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=18,
                image_path="/rendered/page-18.png",
                text_content="Recurrence tree",
            ),
        )
        attached_note = create_note(
            session,
            lecture.id,
            NoteCreate(content="Karatsuba uses three subproblems.", page_id=first_page.id),
        )
        lecture_note = create_note(
            session,
            lecture.id,
            NoteCreate(content="Compare against ordinary multiplication."),
        )
        return {
            "first_page": first_page.id,
            "second_page": second_page.id,
            "attached_note": attached_note.id,
            "lecture_note": lecture_note.id,
        }


def test_search_returns_normalized_mixed_results_in_similarity_order(
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    ids = create_search_fixture(retrieval_session_factory)
    index = FaissVectorIndex.build(
        dimension=3,
        vectors=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        entities=[
            IndexedEntity("slide_page", ids["first_page"]),
            IndexedEntity("note", ids["attached_note"]),
            IndexedEntity("slide_page", ids["second_page"]),
        ],
    )
    provider = QueryEmbeddingProvider()

    with retrieval_session_factory() as session:
        results = search_lecture_memory(
            session,
            "  why is Karatsuba faster?  ",
            provider=provider,
            index=index,
            top_k=2,
        )

    assert provider.query_calls == ["why is Karatsuba faster?"]
    assert len(results) == 2
    slide, note = results
    assert slide.result_type == "slide"
    assert slide.entity_id == ids["first_page"]
    assert slide.rank == 1
    assert slide.course_id == 1
    assert slide.course_name == "Algorithms"
    assert slide.course_code == "CS344"
    assert slide.lecture_id == 1
    assert slide.lecture_title == "Divide and Conquer"
    assert slide.lecture_number == 3
    assert slide.page_number == 17
    assert slide.preview_path == "/rendered/page-17.png"
    assert slide.raw_similarity == pytest.approx(1.0)
    assert slide.reranker_score is None
    assert slide.text_preview == "Three recursive calls reduce the exponent."
    assert slide.related_notes == ("Karatsuba uses three subproblems.",)
    assert slide.concepts == ()

    assert note.result_type == "note"
    assert note.entity_id == ids["attached_note"]
    assert note.rank == 2
    assert note.page_number == 17
    assert note.preview_path == "/rendered/page-17.png"
    assert note.raw_similarity == pytest.approx(0.8)
    assert note.text_preview == "Karatsuba uses three subproblems."
    assert note.related_notes == ()


def test_lecture_level_note_has_no_page_preview(
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    ids = create_search_fixture(retrieval_session_factory)
    index = FaissVectorIndex.build(
        dimension=3,
        vectors=[[1.0, 0.0, 0.0]],
        entities=[IndexedEntity("note", ids["lecture_note"])],
    )

    with retrieval_session_factory() as session:
        result = search_lecture_memory(
            session,
            "ordinary multiplication",
            provider=QueryEmbeddingProvider(),
            index=index,
            top_k=1,
        )[0]

    assert result.result_type == "note"
    assert result.page_number is None
    assert result.preview_path is None
    assert result.text_preview == "Compare against ordinary multiplication."


def test_stale_index_entities_are_skipped_and_result_ranks_are_compacted(
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    ids = create_search_fixture(retrieval_session_factory)
    index = FaissVectorIndex.build(
        dimension=3,
        vectors=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
        entities=[
            IndexedEntity("slide_page", 999),
            IndexedEntity("note", ids["lecture_note"]),
        ],
    )

    with retrieval_session_factory() as session:
        results = search_lecture_memory(
            session,
            "stale candidate",
            provider=QueryEmbeddingProvider(),
            index=index,
            top_k=2,
        )

    assert len(results) == 1
    assert results[0].entity_id == ids["lecture_note"]
    assert results[0].rank == 1


def test_empty_index_returns_without_loading_query_model(
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    provider = QueryEmbeddingProvider()

    with retrieval_session_factory() as session:
        results = search_lecture_memory(
            session,
            "valid query",
            provider=provider,
            index=FaissVectorIndex(3),
        )

    assert results == ()
    assert provider.query_calls == []


def test_provider_and_index_dimensions_must_match(
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    provider = QueryEmbeddingProvider(dimension=2)
    index = FaissVectorIndex(3)

    with retrieval_session_factory() as session:
        with pytest.raises(RetrievalConfigurationError, match="does not match"):
            search_lecture_memory(
                session,
                "valid query",
                provider=provider,
                index=index,
            )

    assert provider.query_calls == []


@pytest.mark.parametrize(
    ("query", "top_k", "error_type", "message"),
    [
        ("", 1, ValueError, "blank"),
        ("   ", 1, ValueError, "blank"),
        (123, 1, TypeError, "string"),
        ("valid", 0, ValueError, "top_k"),
        ("valid", True, ValueError, "top_k"),
    ],
)
def test_invalid_search_inputs_fail_before_query_inference(
    query: object,
    top_k: object,
    error_type: type[Exception],
    message: str,
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    provider = QueryEmbeddingProvider()

    with retrieval_session_factory() as session:
        with pytest.raises(error_type, match=message):
            search_lecture_memory(
                session,
                query,  # type: ignore[arg-type]
                provider=provider,
                index=FaissVectorIndex(3),
                top_k=top_k,  # type: ignore[arg-type]
            )

    assert provider.query_calls == []


def test_long_text_preview_is_compacted_and_truncated(
    retrieval_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(retrieval_session_factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Long Notes", lecture_number=4),
        )
        note = create_note(
            session,
            lecture.id,
            NoteCreate(content=("concept  " * 80)),
        )
        note_id = note.id

    index = FaissVectorIndex.build(
        dimension=3,
        vectors=[[1.0, 0.0, 0.0]],
        entities=[IndexedEntity("note", note_id)],
    )
    with retrieval_session_factory() as session:
        result = search_lecture_memory(
            session,
            "concept",
            provider=QueryEmbeddingProvider(),
            index=index,
            top_k=1,
        )[0]

    assert result.text_preview is not None
    assert len(result.text_preview) <= 240
    assert result.text_preview.endswith("…")
    assert "  " not in result.text_preview
