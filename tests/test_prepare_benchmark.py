"""Tests for reproducible benchmark database preparation."""

import json
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import func, select

from app.database import create_database_engine, create_session_factory
from app.models import Course, Lecture, SlidePage
from benchmark import BenchmarkDatasetError, BenchmarkQuery, load_benchmark_queries
from scripts.prepare_benchmark import (
    collect_benchmark_lectures,
    prepare_benchmark_database,
)


def make_query(
    *,
    query_id: str,
    query: str,
    course: str = "CS344",
    lecture: str = "L1 - Analysis",
    source_file: str = "L1 - Analysis.pdf",
    page_number: int = 1,
) -> BenchmarkQuery:
    return BenchmarkQuery.model_validate(
        {
            "id": query_id,
            "query": query,
            "course": course,
            "lecture": lecture,
            "source_file": source_file,
            "relevant_pages": [{"page_number": page_number}],
            "language": "en",
            "query_type": "semantic",
            "difficulty": "standard",
        }
    )


def write_pdf(path: Path, page_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    for page_number in range(1, page_count + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {page_number}")
    document.save(path)
    document.close()


def test_collect_benchmark_lectures_preserves_first_seen_course_order() -> None:
    queries = (
        make_query(query_id="q-1", query="First query text"),
        make_query(query_id="q-2", query="Second query text"),
        make_query(
            query_id="q-3",
            query="Third query text",
            lecture="L2 - Recurrences",
            source_file="L2 - Recurrences.pdf",
        ),
        make_query(
            query_id="q-4",
            query="Fourth query text",
            course="CS440",
            lecture="Lecture 2",
            source_file="Lecture 2.pdf",
        ),
    )

    lectures = collect_benchmark_lectures(queries)

    assert [(item.course_code, item.title, item.lecture_number) for item in lectures] == [
        ("CS344", "L1 - Analysis", 1),
        ("CS344", "L2 - Recurrences", 2),
        ("CS440", "Lecture 2", 1),
    ]


def test_collect_benchmark_lectures_rejects_ambiguous_source_mapping() -> None:
    queries = (
        make_query(query_id="q-1", query="First query text"),
        make_query(
            query_id="q-2",
            query="Second query text",
            source_file="Different.pdf",
        ),
    )

    with pytest.raises(BenchmarkDatasetError, match="references both"):
        collect_benchmark_lectures(queries)


def test_prepare_benchmark_database_is_idempotent(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    write_pdf(raw_root / "CS344" / "L1 - Analysis.pdf", page_count=2)
    dataset = tmp_path / "queries.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q-1",
                "query": "Where is page two explained?",
                "course": "CS344",
                "lecture": "L1 - Analysis",
                "source_file": "L1 - Analysis.pdf",
                "relevant_pages": [{"page_number": 2}],
                "language": "en",
                "query_type": "semantic",
                "difficulty": "standard",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    queries = load_benchmark_queries(dataset)
    database_path = tmp_path / "benchmark.db"
    rendered_root = tmp_path / "rendered"

    first = prepare_benchmark_database(
        queries,
        database_path=database_path,
        raw_root=raw_root,
        rendered_root=rendered_root,
    )
    second = prepare_benchmark_database(
        queries,
        database_path=database_path,
        raw_root=raw_root,
        rendered_root=rendered_root,
    )

    engine = create_database_engine(database_path)
    factory = create_session_factory(engine)
    try:
        with factory() as session:
            counts = (
                session.scalar(select(func.count()).select_from(Course)),
                session.scalar(select(func.count()).select_from(Lecture)),
                session.scalar(select(func.count()).select_from(SlidePage)),
            )
    finally:
        engine.dispose()

    assert counts == (1, 1, 2)
    assert first.ingested_lectures == 1
    assert first.skipped_lectures == 0
    assert second.ingested_lectures == 0
    assert second.skipped_lectures == 1
