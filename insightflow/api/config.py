"""Application configuration (12-factor: everything comes from the environment).

The one setting that matters most here is ``DATABASE_URL``. It defaults to a local
SQLite file so you can ``git clone`` and run the API with zero setup — no database
to install, no Docker to boot. In production (or whenever you want the real thing),
point it at PostgreSQL:

    export DATABASE_URL="postgresql+psycopg://insightflow:insightflow@localhost:5432/insightflow"

SQLAlchemy speaks both dialects, so nothing else in the codebase changes.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage. SQLite for frictionless dev; Postgres for production.
    database_url: str = "sqlite:///./insightflow.db"

    # API metadata surfaced in the auto-generated docs.
    app_name: str = "InsightFlow"
    app_version: str = "0.3.0"

    # Echo SQL to the console — handy when debugging queries, noisy otherwise.
    sql_echo: bool = False


settings = Settings()
