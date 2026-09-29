"""Tests for retrieval benchmark loading and reference validation."""

import json
from collections.abc import Iterator
from pathlib import Path

import pymupdf
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
from app.repositories.slide_page_repository import create_slide_page
from app.schemas import CourseCreate, LectureCreate, SlidePageCreate
from benchmark import (
    BenchmarkDatasetError,
    BenchmarkQuery,
    load_benchmark_queries,
    resolve_database_references,
    validate_source_pdfs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def benchmark_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(tmp_path / "benchmark.db")
    init_database(engine)
    factory = create_session_factory(engine)
    yield factory
    engine.dispose()


def query_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "cs344-l1-q01",
        "query": "Where was recursive multiplication explained?",
        "course": "CS344",
        "lecture": "L1 - Algorithmic Analysis",
        "source_file": "L1 - Algorithmic Analysis.pdf",
        "relevant_pages": [{"page_number": 2}],
        "language": "en",
        "query_type": "semantic",
        "difficulty": "paraphrase",
    }
    payload.update(overrides)
    return payload


def write_jsonl(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def write_pdf(path: Path, page_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page()
    document.save(path)
    document.close()


def test_committed_benchmark_meets_phase_eight_minimum() -> None:
    queries = load_benchmark_queries(PROJECT_ROOT / "benchmark" / "queries.jsonl")

    assert len(queries) >= 50
    assert {query.language for query in queries} == {"en", "zh"}
    assert {query.query_type for query in queries} == {"semantic", "visual", "note"}
    assert {query.difficulty for query in queries} == {
        "standard",
        "paraphrase",
        "confusable",
        "hard",
    }


def test_load_benchmark_queries_validates_and_preserves_order(tmp_path: Path) -> None:
    dataset = tmp_path / "queries.jsonl"
    write_jsonl(
        dataset,
        query_payload(),
        query_payload(
            id="cs344-l1-q02",
            query="哪一页比较了四个递归分支？",
            relevant_pages=[{"page_number": 4}, {"page_number": 5}],
            language="zh",
            query_type="visual",
            difficulty="confusable",
        ),
    )

    queries = load_benchmark_queries(dataset)

    assert [query.id for query in queries] == ["cs344-l1-q01", "cs344-l1-q02"]
    assert queries[1].relevant_pages[1].page_number == 5


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [query_payload(), query_payload(query="A different query")],
            "Duplicate benchmark id",
        ),
        (
            [
                query_payload(),
                query_payload(id="cs344-l1-q02", query="WHERE WAS RECURSIVE MULTIPLICATION EXPLAINED?"),
            ],
            "Duplicate benchmark query",
        ),
        (
            [query_payload(source_file="../lecture.pdf")],
            "does not match the benchmark schema",
        ),
        (
            [query_payload(relevant_pages=[{"page_number": 2}, {"page_number": 2}])],
            "does not match the benchmark schema",
        ),
    ],
)
def test_load_benchmark_queries_rejects_invalid_or_duplicate_records(
    records: list[dict[str, object]],
    message: str,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "invalid.jsonl"
    write_jsonl(dataset, *records)

    with pytest.raises(BenchmarkDatasetError, match=message):
        load_benchmark_queries(dataset)


def test_load_benchmark_queries_reports_json_line_number(tmp_path: Path) -> None:
    dataset = tmp_path / "broken.jsonl"
    dataset.write_text(f"{json.dumps(query_payload())}\n{{broken\n", encoding="utf-8")

    with pytest.raises(BenchmarkDatasetError, match="Line 2 is not valid JSON"):
        load_benchmark_queries(dataset)


def test_validate_source_pdfs_checks_page_ranges(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "CS344" / "L1 - Algorithmic Analysis.pdf"
    write_pdf(source, page_count=3)
    valid = (BenchmarkQuery.model_validate(query_payload()),)

    sources = validate_source_pdfs(valid, tmp_path / "raw")

    assert len(sources) == 1
    assert sources[0].page_count == 3
    assert sources[0].query_count == 1

    invalid = (
        BenchmarkQuery.model_validate(
            query_payload(relevant_pages=[{"page_number": 4}])
        ),
    )
    with pytest.raises(BenchmarkDatasetError, match="has 3 pages"):
        validate_source_pdfs(invalid, tmp_path / "raw")


def test_resolve_database_references_maps_stable_labels_to_slide_ids(
    benchmark_factory: sessionmaker[Session],
) -> None:
    with session_scope(benchmark_factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        lecture = create_lecture(
            session,
            course.id,
            LectureCreate(title="L1 - Algorithmic Analysis", lecture_number=1),
        )
        page = create_slide_page(
            session,
            lecture.id,
            SlidePageCreate(page_number=2, image_path="page-2.png"),
        )
        lecture_id = lecture.id
        page_id = page.id

    query = BenchmarkQuery.model_validate(query_payload())
    with benchmark_factory() as session:
        resolved = resolve_database_references(session, (query,))

    assert len(resolved) == 1
    assert resolved[0].query_id == query.id
    assert resolved[0].lecture_id == lecture_id
    assert resolved[0].page_id == page_id
    assert resolved[0].page_number == 2


def test_resolve_database_references_rejects_missing_page(
    benchmark_factory: sessionmaker[Session],
) -> None:
    with session_scope(benchmark_factory) as session:
        course = create_course(session, CourseCreate(name="Algorithms", code="CS344"))
        create_lecture(
            session,
            course.id,
            LectureCreate(title="L1 - Algorithmic Analysis", lecture_number=1),
        )

    with benchmark_factory() as session:
        with pytest.raises(BenchmarkDatasetError, match="references missing"):
            resolve_database_references(
                session,
                (BenchmarkQuery.model_validate(query_payload()),),
            )
