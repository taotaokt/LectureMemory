"""Tests for the SQLAlchemy database infrastructure."""

from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, inspect, select, text

from app.config import get_settings
from app.database import (
    create_database_engine,
    create_session_factory,
    init_database,
    session_scope,
)


def build_test_table(metadata: MetaData) -> Table:
    """Create a small table used only by database infrastructure tests."""
    return Table(
        "test_records",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, nullable=False),
    )


def test_engine_uses_configured_database_path(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "configured/database.sqlite3"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    get_settings.cache_clear()

    database_engine = create_database_engine()

    assert database_engine.url.database == str(database_path)
    assert database_path.parent.is_dir()

    database_engine.dispose()
    get_settings.cache_clear()


def test_database_initialization_is_idempotent(tmp_path: Path) -> None:
    database_path = tmp_path / "database/lecture_memory.db"
    metadata = MetaData()
    build_test_table(metadata)
    database_engine = create_database_engine(database_path)

    init_database(database_engine, metadata)
    init_database(database_engine, metadata)

    assert database_path.is_file()
    assert inspect(database_engine).get_table_names() == ["test_records"]

    database_engine.dispose()


def test_sqlite_foreign_keys_are_enabled(tmp_path: Path) -> None:
    database_engine = create_database_engine(tmp_path / "foreign-keys.db")

    with database_engine.connect() as connection:
        foreign_keys_enabled = connection.scalar(text("PRAGMA foreign_keys"))

    assert foreign_keys_enabled == 1
    database_engine.dispose()


def test_session_scope_commits_successful_transaction(tmp_path: Path) -> None:
    metadata = MetaData()
    records = build_test_table(metadata)
    database_engine = create_database_engine(tmp_path / "commit.db")
    init_database(database_engine, metadata)
    session_factory = create_session_factory(database_engine)

    with session_scope(session_factory) as session:
        session.execute(records.insert().values(name="committed"))

    with session_factory() as session:
        names = session.scalars(select(records.c.name)).all()

    assert names == ["committed"]
    database_engine.dispose()


def test_session_scope_rolls_back_failed_transaction(tmp_path: Path) -> None:
    metadata = MetaData()
    records = build_test_table(metadata)
    database_engine = create_database_engine(tmp_path / "rollback.db")
    init_database(database_engine, metadata)
    session_factory = create_session_factory(database_engine)

    with pytest.raises(RuntimeError, match="force rollback"):
        with session_scope(session_factory) as session:
            session.execute(records.insert().values(name="rolled back"))
            raise RuntimeError("force rollback")

    with session_factory() as session:
        names = session.scalars(select(records.c.name)).all()

    assert names == []
    database_engine.dispose()
