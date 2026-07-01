"""Database engine, session factory, and the FastAPI dependency that hands out sessions.

This is the thin seam between the web layer and storage. Everything here is standard
SQLAlchemy 2.0: one engine, a session-per-request pattern via ``get_db``, and a
declarative ``Base`` that the models inherit from.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import settings

# SQLite needs one extra flag to be used across threads (FastAPI's default worker
# model). Other databases don't want it, so we add it conditionally.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, echo=settings.sql_echo, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base class shared by every ORM model."""


def get_db() -> Iterator[Session]:
    """Yield a database session and guarantee it's closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Fine for dev/SQLite; production would use Alembic migrations."""
    # Import models so they're registered on Base.metadata before create_all.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
