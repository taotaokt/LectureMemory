"""Central application configuration for Lecture Memory."""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Configuration loaded from defaults, environment variables, and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    data_dir: Path = Field(default=PROJECT_ROOT / "data", validation_alias="DATA_DIR")
    database_path: Path | None = Field(default=None, validation_alias="DATABASE_PATH")
    rendered_dir: Path | None = Field(default=None, validation_alias="RENDERED_DIR")
    embedding_dir: Path | None = Field(default=None, validation_alias="EMBEDDING_DIR")
    index_dir: Path | None = Field(default=None, validation_alias="INDEX_DIR")

    device: str = Field(default="auto", validation_alias="DEVICE")
    model_name: str = Field(
        default="Qwen3-VL-Embedding",
        validation_alias="MODEL_NAME",
    )
    reranker_model_name: str = Field(
        default="Qwen3-VL-Reranker",
        validation_alias="RERANKER_MODEL_NAME",
    )

    @model_validator(mode="after")
    def resolve_storage_paths(self) -> Self:
        """Resolve storage paths and derive unset child paths from ``data_dir``."""
        self.data_dir = self._resolve_path(self.data_dir)
        self.database_path = self._resolve_path(
            self.database_path or self.data_dir / "database" / "lecture_memory.db"
        )
        self.rendered_dir = self._resolve_path(self.rendered_dir or self.data_dir / "rendered")
        self.embedding_dir = self._resolve_path(
            self.embedding_dir or self.data_dir / "embeddings"
        )
        self.index_dir = self._resolve_path(self.index_dir or self.data_dir / "indexes")
        return self

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        expanded_path = path.expanduser()
        if not expanded_path.is_absolute():
            expanded_path = PROJECT_ROOT / expanded_path
        return expanded_path.resolve()

    def ensure_directories(self) -> None:
        """Create all configured storage directories without creating the database file."""
        if self.database_path is None:
            raise ValueError("database_path must be configured")

        directories = (
            self.data_dir,
            self.database_path.parent,
            self.rendered_dir,
            self.embedding_dir,
            self.index_dir,
        )
        for directory in directories:
            if directory is None:
                raise ValueError("storage directories must be configured")
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings with required directories initialized."""
    settings = Settings()
    settings.ensure_directories()
    return settings
