"""Tests for configured application search runtime assembly."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.retrieval import FaissVectorIndex
from app.schemas import SearchResult
from app.services.search_runtime_service import (
    SearchRuntime,
    SearchRuntimeUnavailableError,
    build_search_runtime,
    search_course_memory,
)


def make_settings(tmp_path: Path, *, index_dir: Path | None = None) -> Settings:
    return Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        INDEX_DIR=index_dir or tmp_path / "index",
        EMBEDDING_DIMENSION=64,
        RETRIEVAL_TOP_K=6,
        RERANK_TOP_K=4,
        FINAL_TOP_K=2,
    )


def test_build_search_runtime_loads_index_and_keeps_models_lazy(tmp_path: Path) -> None:
    index_dir = tmp_path / "saved-index"
    FaissVectorIndex(64).save(index_dir)
    settings = make_settings(tmp_path, index_dir=index_dir)

    runtime = build_search_runtime(settings)

    assert runtime.index.dimension == 64
    assert runtime.index.count == 0
    assert runtime.provider.dimension == 64
    assert runtime.provider.is_loaded is False
    assert runtime.reranker.is_loaded is False


def test_build_search_runtime_reports_missing_snapshot(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(SearchRuntimeUnavailableError, match="No usable search index"):
        build_search_runtime(settings)


def test_search_course_memory_forwards_configured_limits_and_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    result = SearchResult(
        result_type="note",
        entity_id=8,
        rank=1,
        course_id=3,
        course_name="Algorithms",
        course_code="CS344",
        lecture_id=5,
        lecture_title="Divide and Conquer",
        lecture_number=2,
        raw_similarity=0.8,
        reranker_score=0.9,
        text_preview="A useful note",
    )
    captured: dict[str, object] = {}

    def fake_search(session, query, **kwargs):
        captured["session"] = session
        captured["query"] = query
        captured.update(kwargs)
        return (result,)

    monkeypatch.setattr(
        "app.services.search_runtime_service.search_and_rerank",
        fake_search,
    )
    runtime = SearchRuntime(
        provider=SimpleNamespace(),
        index=FaissVectorIndex(64),
        reranker=SimpleNamespace(),
    )
    session = SimpleNamespace()

    results = search_course_memory(
        session,
        "recursive multiplication",
        course_id=3,
        runtime=runtime,
        settings=settings,
    )

    assert results == (result,)
    assert captured == {
        "session": session,
        "query": "recursive multiplication",
        "provider": runtime.provider,
        "index": runtime.index,
        "reranker": runtime.reranker,
        "retrieval_top_k": 6,
        "rerank_top_k": 4,
        "final_top_k": 2,
        "course_id": 3,
    }
