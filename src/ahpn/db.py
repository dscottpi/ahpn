from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy import URL, text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ahpn.config import (
    DEFAULT_DATABASE_URL,
    DEFAULT_EMBEDDING_DIMS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_M,
)
from ahpn.models import Base

# Spanish stemming + unaccent. Created by ensure_search_schema before any FTS queries.
AHPN_ES_CONFIG = "ahpn_es"


def database_url_from_env(env: Mapping[str, str] | None = None) -> str:
    """AHPN_DATABASE_URL, else a URL built from AHPN_DB_* parts, else the local Compose default.

    The parts form exists for ECS, where the RDS-managed password arrives as
    its own secret and may contain characters that need escaping in a URL.
    """
    env = os.environ if env is None else env
    if env.get("AHPN_DATABASE_URL"):
        return env["AHPN_DATABASE_URL"]
    host = env.get("AHPN_DB_HOST")
    if not host:
        return DEFAULT_DATABASE_URL
    url = URL.create(
        "postgresql+psycopg",
        username=env.get("AHPN_DB_USER", "ahpn"),
        password=env.get("AHPN_DB_PASSWORD"),
        host=host,
        port=int(env.get("AHPN_DB_PORT", "5432")),
        database=env.get("AHPN_DB_NAME", "ahpn"),
    )
    return url.render_as_string(hide_password=False)


def create_db_engine(database_url: str) -> Engine:
    return create_engine(database_url)


def init_db(database_url: str) -> None:
    engine = create_db_engine(database_url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        ensure_search_schema(conn)
    engine.dispose()


def ensure_search_schema(conn) -> None:
    """Install pgvector/unaccent, the Spanish FTS config, generated tsvector, and HNSW indexes.

    Safe to re-run. Existing RDS databases that predate search need this once after deploy;
    local Compose volumes need a reset (`docker compose down -v`) when switching to pgvector.
    """
    dialect = conn.dialect.name
    if dialect != "postgresql":
        return

    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))

    # Immutable wrapper so unaccent can be used in a generated column.
    conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION ahpn_unaccent(text)
            RETURNS text
            LANGUAGE sql
            IMMUTABLE PARALLEL SAFE STRICT
            AS $$ SELECT public.unaccent('public.unaccent', $1) $$
            """
        )
    )
    conn.execute(
        text(
            f"""
            DO $$
            BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = '{AHPN_ES_CONFIG}') THEN
                CREATE TEXT SEARCH CONFIGURATION {AHPN_ES_CONFIG} (COPY = spanish);
                ALTER TEXT SEARCH CONFIGURATION {AHPN_ES_CONFIG}
                  ALTER MAPPING FOR hword, hword_part, word
                  WITH unaccent, spanish_stem;
              END IF;
            END
            $$
            """
        )
    )
    conn.execute(
        text(
            f"""
            ALTER TABLE chunks
              ADD COLUMN IF NOT EXISTS search_vector tsvector
              GENERATED ALWAYS AS (
                to_tsvector('{AHPN_ES_CONFIG}'::regconfig, coalesce(text, ''))
              ) STORED
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS chunks_search_vector_gin
              ON chunks USING GIN (search_vector)
            """
        )
    )
    # Partial HNSW index per default model. Additional models get their own index via embed.
    conn.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_{_index_slug(DEFAULT_EMBEDDING_MODEL)}
              ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
              WITH (m = {DEFAULT_HNSW_M}, ef_construction = {DEFAULT_HNSW_EF_CONSTRUCTION})
              WHERE model = '{DEFAULT_EMBEDDING_MODEL}'
            """
        )
    )
    # Helpful lookup indexes.
    conn.execute(text("CREATE INDEX IF NOT EXISTS chunk_embeddings_model_idx ON chunk_embeddings (model)"))
    conn.execute(
        text(
            f"""
            DO $$
            BEGIN
              -- Defensive: older rows from a previous schema experiment.
              IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunk_embeddings' AND column_name = 'dims'
              ) THEN
                UPDATE chunk_embeddings SET dims = {DEFAULT_EMBEDDING_DIMS} WHERE dims IS NULL;
              END IF;
            END
            $$
            """
        )
    )


def ensure_embedding_index(conn, model: str, *, m: int = DEFAULT_HNSW_M, ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION) -> None:
    """Create a partial HNSW index for a specific embedding model if missing."""
    if conn.dialect.name != "postgresql":
        return
    slug = _index_slug(model)
    # Index predicates cannot take bind params (Postgres needs a concrete type).
    literal_model = model.replace("'", "''")
    conn.execute(
        text(
            f"""
            CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_{slug}
              ON chunk_embeddings USING hnsw (embedding vector_cosine_ops)
              WITH (m = {int(m)}, ef_construction = {int(ef_construction)})
              WHERE model = '{literal_model}'
            """
        )
    )


def session_factory(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=create_db_engine(database_url), expire_on_commit=False)


def _index_slug(model: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in model)[:48]
