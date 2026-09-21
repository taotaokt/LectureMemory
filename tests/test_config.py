"""Tests for application configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import PROJECT_ROOT, Settings, get_settings

ENVIRONMENT_KEYS = (
    "DATA_DIR",
    "DATABASE_PATH",
    "RENDERED_DIR",
    "EMBEDDING_DIR",
    "INDEX_DIR",
    "DEVICE",
    "MODEL_NAME",
    "EMBEDDING_DTYPE",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_MAX_PIXELS",
    "EMBEDDING_QUERY_INSTRUCTION",
    "RERANKER_MODEL_NAME",
    "RERANKER_DTYPE",
    "RERANKER_BATCH_SIZE",
    "RERANKER_MAX_LENGTH",
    "RERANKER_MIN_PIXELS",
    "RERANKER_MAX_PIXELS",
    "RERANKER_INSTRUCTION",
    "RETRIEVAL_TOP_K",
    "RERANK_TOP_K",
    "FINAL_TOP_K",
)


def clear_settings_environment(monkeypatch) -> None:
    """Remove configuration variables inherited by the test process."""
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_settings_load_defaults(monkeypatch) -> None:
    clear_settings_environment(monkeypatch)

    settings = Settings(_env_file=None)

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.database_path == PROJECT_ROOT / "data/database/lecture_memory.db"
    assert settings.rendered_dir == PROJECT_ROOT / "data/rendered"
    assert settings.embedding_dir == PROJECT_ROOT / "data/embeddings"
    assert settings.index_dir == PROJECT_ROOT / "data/indexes"
    assert settings.device == "auto"
    assert settings.model_name == "Qwen/Qwen3-VL-Embedding-2B"
    assert settings.embedding_dtype == "auto"
    assert settings.embedding_dimension == 2048
    assert settings.embedding_batch_size == 1
    assert settings.embedding_max_pixels == 524288
    assert settings.embedding_query_instruction.startswith("Retrieve the lecture slide")
    assert settings.reranker_model_name == "Qwen/Qwen3-VL-Reranker-2B"
    assert settings.reranker_dtype == "auto"
    assert settings.reranker_batch_size == 1
    assert settings.reranker_max_length == 10_240
    assert settings.reranker_min_pixels == 4096
    assert settings.reranker_max_pixels == 524288
    assert settings.reranker_instruction.startswith("Retrieve the lecture slide")
    assert settings.retrieval_top_k == 20
    assert settings.rerank_top_k == 20
    assert settings.final_top_k == 5


def test_environment_overrides_settings(monkeypatch, tmp_path: Path) -> None:
    clear_settings_environment(monkeypatch)
    custom_data_dir = tmp_path / "course-memory"
    custom_database_path = tmp_path / "database/custom.sqlite3"
    custom_rendered_dir = tmp_path / "slides"
    custom_embedding_dir = tmp_path / "vectors"
    custom_index_dir = tmp_path / "search-index"

    monkeypatch.setenv("DATA_DIR", str(custom_data_dir))
    monkeypatch.setenv("DATABASE_PATH", str(custom_database_path))
    monkeypatch.setenv("RENDERED_DIR", str(custom_rendered_dir))
    monkeypatch.setenv("EMBEDDING_DIR", str(custom_embedding_dir))
    monkeypatch.setenv("INDEX_DIR", str(custom_index_dir))
    monkeypatch.setenv("DEVICE", "mps")
    monkeypatch.setenv("MODEL_NAME", "test-embedding-model")
    monkeypatch.setenv("EMBEDDING_DTYPE", "float32")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "512")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "3")
    monkeypatch.setenv("EMBEDDING_MAX_PIXELS", "262144")
    monkeypatch.setenv("EMBEDDING_QUERY_INSTRUCTION", "Find the relevant lecture material.")
    monkeypatch.setenv("RERANKER_MODEL_NAME", "test-reranker-model")
    monkeypatch.setenv("RERANKER_DTYPE", "bfloat16")
    monkeypatch.setenv("RERANKER_BATCH_SIZE", "2")
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "4096")
    monkeypatch.setenv("RERANKER_MIN_PIXELS", "8192")
    monkeypatch.setenv("RERANKER_MAX_PIXELS", "262144")
    monkeypatch.setenv("RERANKER_INSTRUCTION", "Judge relevance for this course.")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "30")
    monkeypatch.setenv("RERANK_TOP_K", "12")
    monkeypatch.setenv("FINAL_TOP_K", "4")

    settings = Settings(_env_file=None)

    assert settings.data_dir == custom_data_dir
    assert settings.database_path == custom_database_path
    assert settings.rendered_dir == custom_rendered_dir
    assert settings.embedding_dir == custom_embedding_dir
    assert settings.index_dir == custom_index_dir
    assert settings.device == "mps"
    assert settings.model_name == "test-embedding-model"
    assert settings.embedding_dtype == "float32"
    assert settings.embedding_dimension == 512
    assert settings.embedding_batch_size == 3
    assert settings.embedding_max_pixels == 262144
    assert settings.embedding_query_instruction == "Find the relevant lecture material."
    assert settings.reranker_model_name == "test-reranker-model"
    assert settings.reranker_dtype == "bfloat16"
    assert settings.reranker_batch_size == 2
    assert settings.reranker_max_length == 4096
    assert settings.reranker_min_pixels == 8192
    assert settings.reranker_max_pixels == 262144
    assert settings.reranker_instruction == "Judge relevance for this course."
    assert settings.retrieval_top_k == 30
    assert settings.rerank_top_k == 12
    assert settings.final_top_k == 4


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"RETRIEVAL_TOP_K": "4", "RERANK_TOP_K": "5"}, "rerank_top_k"),
        ({"RERANK_TOP_K": "4", "FINAL_TOP_K": "5"}, "final_top_k"),
        (
            {"RERANKER_MIN_PIXELS": "8192", "RERANKER_MAX_PIXELS": "4096"},
            "reranker_max_pixels",
        ),
    ],
)
def test_inconsistent_reranking_configuration_is_rejected(
    monkeypatch,
    overrides: dict[str, str],
    message: str,
) -> None:
    clear_settings_environment(monkeypatch)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


def test_data_directory_override_updates_derived_paths(monkeypatch, tmp_path: Path) -> None:
    clear_settings_environment(monkeypatch)
    custom_data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("DATA_DIR", str(custom_data_dir))

    settings = Settings(_env_file=None)

    assert settings.database_path == custom_data_dir / "database/lecture_memory.db"
    assert settings.rendered_dir == custom_data_dir / "rendered"
    assert settings.embedding_dir == custom_data_dir / "embeddings"
    assert settings.index_dir == custom_data_dir / "indexes"


def test_get_settings_creates_storage_directories(monkeypatch, tmp_path: Path) -> None:
    clear_settings_environment(monkeypatch)
    custom_data_dir = tmp_path / "nested/data"
    monkeypatch.setenv("DATA_DIR", str(custom_data_dir))
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.data_dir.is_dir()
    assert settings.database_path is not None
    assert settings.database_path.parent.is_dir()
    assert not settings.database_path.exists()
    assert settings.rendered_dir is not None and settings.rendered_dir.is_dir()
    assert settings.embedding_dir is not None and settings.embedding_dir.is_dir()
    assert settings.index_dir is not None and settings.index_dir.is_dir()

    get_settings.cache_clear()
