"""Tests for the page-level bilingual BM25 baseline."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)
from app.repositories.course_repository import create_course
from app.repositories.lecture_repository import create_lecture
from app.repositories.note_repository import create_note
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import CourseCreate, LectureCreate, NoteCreate, SlidePageCreate
from benchmark.bm25 import BM25Document, BM25Index, load_bm25_documents, tokenize


@pytest.fixture
def bm25_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "bm25.db")
    init_database(engine)
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def test_tokenize_supports_casefolded_english_and_chinese_bigrams() -> None:
    tokens = tokenize("Binary TREE 与平衡树")

    assert tokens[:2] == ("binary", "tree")
    assert "平" in tokens
    assert "衡" in tokens
    assert "平衡" in tokens
    assert "衡树" in tokens


def test_bm25_ranks_lexical_matches_and_applies_course_filter() -> None:
    index = BM25Index(
        [
            BM25Document(1, 10, 100, 1, "binary search tree balance rotation"),
            BM25Document(2, 10, 100, 2, "dynamic programming shortest path"),
            BM25Document(3, 20, 200, 1, "binary classification decision boundary"),
        ]
    )

    all_results = index.search("balanced binary tree", top_k=3)
    filtered = index.search("binary", top_k=3, course_id=200)

    assert all_results[0].page_id == 1
    assert filtered[0].page_id == 3
    assert all(result.course_id == 200 for result in filtered)
    assert index.search("unseen vocabulary", top_k=5) == ()


def test_load_documents_combines_slide_text_and_attached_notes(
    bm25_factory: sessionmaker[Session],
) -> None:
    with session_scope(bm25_factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="Trees", lecture_number=1),
        )
        page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(
                page_number=7,
                image_path="/tmp/page-7.png",
                text_content="AVL rotation diagram",
            ),
        )
        create_note(
            session,
            lecture.id,
            NoteCreate(content="Remember the balance factor", page_id=page.id),
        )
        create_note(session, lecture.id, NoteCreate(content="General lecture note"))

    with bm25_factory() as session:
        documents = load_bm25_documents(session)

    assert len(documents) == 1
    assert documents[0].page_id == page.id
    assert "AVL rotation diagram" in documents[0].text
    assert "Remember the balance factor" in documents[0].text
    assert "General lecture note" not in documents[0].text
    assert BM25Index(documents).search("balance factor", top_k=1)[0].page_id == page.id


def test_bm25_rejects_invalid_configuration_and_arguments() -> None:
    document = BM25Document(1, 1, 1, 1, "text")

    with pytest.raises(ValueError, match="unique"):
        BM25Index([document, document])
    with pytest.raises(ValueError, match="k1"):
        BM25Index([document], k1=0)
    with pytest.raises(ValueError, match="between"):
        BM25Index([document], b=1.1)
    with pytest.raises(ValueError, match="top_k"):
        BM25Index([document]).search("text", top_k=0)
