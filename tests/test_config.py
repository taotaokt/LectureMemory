"""Tests for application configuration."""

from pathlib import Path

from app.config import PROJECT_ROOT, Settings, get_settings

ENVIRONMENT_KEYS = (
    "DATA_DIR",
    "DATABASE_PATH",
    "RENDERED_DIR",
    "EMBEDDING_DIR",
    "INDEX_DIR",
    "DEVICE",
    "MODEL_NAME",
    "RERANKER_MODEL_NAME",
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
    assert settings.model_name == "Qwen3-VL-Embedding"
    assert settings.reranker_model_name == "Qwen3-VL-Reranker"


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
    monkeypatch.setenv("RERANKER_MODEL_NAME", "test-reranker-model")

    settings = Settings(_env_file=None)

    assert settings.data_dir == custom_data_dir
    assert settings.database_path == custom_database_path
    assert settings.rendered_dir == custom_rendered_dir
    assert settings.embedding_dir == custom_embedding_dir
    assert settings.index_dir == custom_index_dir
    assert settings.device == "mps"
    assert settings.model_name == "test-embedding-model"
    assert settings.reranker_model_name == "test-reranker-model"


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
