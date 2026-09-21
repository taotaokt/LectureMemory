"""Central application configuration for Lecture Memory."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

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

    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
        validation_alias="DEVICE",
    )
    model_name: str = Field(
        default="Qwen/Qwen3-VL-Embedding-2B",
        validation_alias="MODEL_NAME",
    )
    embedding_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto",
        validation_alias="EMBEDDING_DTYPE",
    )
    embedding_dimension: int = Field(
        default=2048,
        ge=64,
        validation_alias="EMBEDDING_DIMENSION",
    )
    embedding_batch_size: int = Field(
        default=1,
        ge=1,
        validation_alias="EMBEDDING_BATCH_SIZE",
    )
    embedding_max_pixels: int = Field(
        default=512 * 32 * 32,
        ge=4 * 32 * 32,
        validation_alias="EMBEDDING_MAX_PIXELS",
    )
    embedding_query_instruction: str = Field(
        default=(
            "Retrieve the lecture slide image or note most relevant to the user's query."
        ),
        min_length=1,
        validation_alias="EMBEDDING_QUERY_INSTRUCTION",
    )
    reranker_model_name: str = Field(
        default="Qwen/Qwen3-VL-Reranker-2B",
        validation_alias="RERANKER_MODEL_NAME",
    )
    reranker_dtype: Literal["auto", "float16", "bfloat16", "float32"] = Field(
        default="auto",
        validation_alias="RERANKER_DTYPE",
    )
    reranker_batch_size: int = Field(
        default=1,
        ge=1,
        validation_alias="RERANKER_BATCH_SIZE",
    )
    reranker_max_length: int = Field(
        default=10_240,
        ge=1,
        validation_alias="RERANKER_MAX_LENGTH",
    )
    reranker_min_pixels: int = Field(
        default=4 * 32 * 32,
        ge=1,
        validation_alias="RERANKER_MIN_PIXELS",
    )
    reranker_max_pixels: int = Field(
        default=512 * 32 * 32,
        ge=1,
        validation_alias="RERANKER_MAX_PIXELS",
    )
    reranker_instruction: str = Field(
        default=(
            "Retrieve the lecture slide image or note most relevant to the user's query."
        ),
        min_length=1,
        validation_alias="RERANKER_INSTRUCTION",
    )
    retrieval_top_k: int = Field(
        default=20,
        ge=1,
        validation_alias="RETRIEVAL_TOP_K",
    )
    rerank_top_k: int = Field(
        default=20,
        ge=1,
        validation_alias="RERANK_TOP_K",
    )
    final_top_k: int = Field(
        default=5,
        ge=1,
        validation_alias="FINAL_TOP_K",
    )

    @model_validator(mode="after")
    def resolve_storage_paths(self) -> Self:
        """Resolve storage paths and derive unset child paths from ``data_dir``."""
        if self.reranker_max_pixels < self.reranker_min_pixels:
            raise ValueError(
                "reranker_max_pixels must be greater than or equal to "
                "reranker_min_pixels"
            )
        if self.rerank_top_k > self.retrieval_top_k:
            raise ValueError("rerank_top_k must not exceed retrieval_top_k")
        if self.final_top_k > self.rerank_top_k:
            raise ValueError("final_top_k must not exceed rerank_top_k")

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
