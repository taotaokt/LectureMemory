"""Tests for retrieval quality metrics and persisted benchmark reports."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.evaluation import (
    QueryEvaluation,
    build_evaluation_report,
    write_evaluation_report,
)


def test_query_metrics_handle_multiple_relevant_pages_and_missing_hits() -> None:
    run = QueryEvaluation(
        method="BM25",
        query_id="q1",
        ranked_pages=((1, 8), (1, 4), (1, 9)),
        relevant_pages=((1, 4), (1, 9)),
    )
    miss = QueryEvaluation(
        method="BM25",
        query_id="q2",
        ranked_pages=((2, 1),),
        relevant_pages=((2, 7),),
    )

    assert run.recall_at(1) == 0.0
    assert run.recall_at(2) == 0.5
    assert run.recall_at(3) == 1.0
    assert run.reciprocal_rank == 0.5
    assert miss.recall_at(10) == 0.0
    assert miss.reciprocal_rank == 0.0


def test_report_aggregates_metrics_and_writes_json_and_csv(tmp_path: Path) -> None:
    runs = [
        QueryEvaluation(
            method="BM25",
            query_id="q1",
            ranked_pages=((1, 4),),
            relevant_pages=((1, 4),),
            retrieval_ms=2.0,
            total_ms=2.0,
        ),
        QueryEvaluation(
            method="BM25",
            query_id="q2",
            ranked_pages=(),
            relevant_pages=((1, 7),),
            retrieval_ms=4.0,
            total_ms=4.0,
        ),
        QueryEvaluation(
            method="Qwen Embedding",
            query_id="q1",
            ranked_pages=((1, 4),),
            relevant_pages=((1, 4),),
            embedding_ms=5.0,
            retrieval_ms=1.0,
            total_ms=6.0,
        ),
        QueryEvaluation(
            method="Qwen Embedding",
            query_id="q2",
            ranked_pages=((1, 8), (1, 7)),
            relevant_pages=((1, 7),),
            embedding_ms=7.0,
            retrieval_ms=1.0,
            total_ms=8.0,
        ),
    ]
    report = build_evaluation_report(
        runs,
        dataset=tmp_path / "queries.jsonl",
        generated_at=datetime(2026, 9, 29, tzinfo=UTC),
    )

    bm25, embedding = report.methods
    assert bm25.recall_at_1 == pytest.approx(0.5)
    assert bm25.mrr == pytest.approx(0.5)
    assert bm25.average_retrieval_ms == pytest.approx(3.0)
    assert embedding.recall_at_1 == pytest.approx(0.5)
    assert embedding.recall_at_5 == pytest.approx(1.0)
    assert embedding.mrr == pytest.approx(0.75)
    assert embedding.average_embedding_ms == pytest.approx(6.0)

    json_path, csv_path = write_evaluation_report(
        report,
        tmp_path / "results",
        run_id="test-run",
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))

    assert payload["generated_at"] == "2026-09-29T00:00:00+00:00"
    assert len(payload["queries"]) == 4
    assert [row["method"] for row in rows] == ["BM25", "Qwen Embedding"]


def test_report_rejects_duplicate_or_incomplete_method_runs(tmp_path: Path) -> None:
    first = QueryEvaluation("BM25", "q1", (), ((1, 1),))
    duplicate = QueryEvaluation("BM25", "q1", (), ((1, 1),))
    other_method = QueryEvaluation("Qwen", "q2", (), ((1, 2),))

    with pytest.raises(ValueError, match="one result"):
        build_evaluation_report([first, duplicate], dataset=tmp_path)
    with pytest.raises(ValueError, match="same query IDs"):
        build_evaluation_report([first, other_method], dataset=tmp_path)
