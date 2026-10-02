"""Tests for configured application index-building assembly."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.indexing_runtime_service import build_configured_search_index


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        EMBEDDING_DIMENSION=64,
    )


def test_configured_index_build_forwards_model_and_storage_settings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    provider = SimpleNamespace(model_name="test/provider")
    cache = SimpleNamespace()
    summary = SimpleNamespace(complete=True)
    captured: dict[str, object] = {}

    def fake_provider(**kwargs):
        captured["provider_kwargs"] = kwargs
        return provider

    def fake_cache(path):
        captured["cache_path"] = path
        return cache

    def fake_build(session, **kwargs):
        captured["session"] = session
        captured.update(kwargs)
        return summary

    monkeypatch.setattr(
        "app.services.indexing_runtime_service.Qwen3VLEmbeddingProvider",
        fake_provider,
    )
    monkeypatch.setattr(
        "app.services.indexing_runtime_service.EmbeddingCache",
        fake_cache,
    )
    monkeypatch.setattr(
        "app.services.indexing_runtime_service.build_search_index",
        fake_build,
    )
    def callback(progress) -> None:
        del progress

    session = SimpleNamespace()

    result = build_configured_search_index(
        session,
        settings,
        progress_callback=callback,
    )

    assert result is summary
    assert captured["provider_kwargs"] == {
        "model_name": settings.model_name,
        "device": settings.device,
        "dtype": settings.embedding_dtype,
        "dimension": settings.embedding_dimension,
        "batch_size": settings.embedding_batch_size,
        "max_pixels": settings.embedding_max_pixels,
        "query_instruction": settings.embedding_query_instruction,
    }
    assert captured["cache_path"] == settings.embedding_dir
    assert captured["session"] is session
    assert captured["provider"] is provider
    assert captured["cache"] is cache
    assert captured["index_dir"] == settings.index_dir
    assert captured["progress_callback"] is callback


@pytest.mark.parametrize("field", ["embedding_dir", "index_dir"])
def test_configured_index_build_rejects_missing_storage_path(
    field: str,
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path).model_copy(update={field: None})

    with pytest.raises(ValueError, match=field.upper()):
        build_configured_search_index(SimpleNamespace(), settings)
