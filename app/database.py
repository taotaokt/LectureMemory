"""SQLAlchemy and SQLite infrastructure for Lecture Memory.

V1 schema initialization intentionally uses ``Base.metadata.create_all``. This
is idempotent for creating missing tables. Versioned or destructive schema
changes will require an explicit migration tool before they are introduced.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

from sqlalchemy import URL, Engine, MetaData, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    """Declarative base shared by all application models."""


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: Any) -> None:
    """Enable foreign-key enforcement for every SQLite connection."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def create_database_engine(database_path: Path | None = None) -> Engine:
    """Create an application SQLite engine for the configured or supplied path."""
    if database_path is None:
        configured_path = get_settings().database_path
        if configured_path is None:
            raise ValueError("DATABASE_PATH must be configured")
        database_path = configured_path

    resolved_path = database_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    database_url = URL.create("sqlite+pysqlite", database=str(resolved_path))
    database_engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
    )
    event.listen(database_engine, "connect", _enable_sqlite_foreign_keys)
    return database_engine


def create_session_factory(database_engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to ``database_engine``."""
    return sessionmaker(
        bind=database_engine,
        autoflush=False,
        expire_on_commit=False,
    )


engine = create_database_engine()
SessionLocal = create_session_factory(engine)


def init_database(
    database_engine: Engine = engine,
    metadata: MetaData = Base.metadata,
) -> None:
    """Create all missing tables registered in the supplied metadata."""
    if metadata is Base.metadata:
        import_module("app.models")
    metadata.create_all(bind=database_engine)


@contextmanager
def session_scope(
    session_factory: sessionmaker[Session] = SessionLocal,
) -> Iterator[Session]:
    """Provide a transactional session that commits or rolls back atomically."""
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
