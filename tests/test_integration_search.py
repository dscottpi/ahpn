from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from ahpn.db import ensure_search_schema
from ahpn.models import Base, Chunk, Document
from ahpn.search import SearchFilters, search

pytestmark = pytest.mark.integration


def _postgres_url() -> str | None:
    candidates = []
    if os.environ.get("AHPN_TEST_DATABASE_URL"):
        candidates.append(os.environ["AHPN_TEST_DATABASE_URL"])
    if os.environ.get("AHPN_DATABASE_URL") and "ahpn_test" in os.environ["AHPN_DATABASE_URL"]:
        candidates.append(os.environ["AHPN_DATABASE_URL"])
    candidates.append("postgresql+psycopg://ahpn:ahpn@localhost:5433/ahpn_test")
    for url in candidates:
        try:
            engine = create_engine(url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return url
        except Exception:  # noqa: BLE001
            continue
    return None


def _maybe_testcontainers_url() -> str | None:
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        return None
    try:
        postgres = PostgresContainer("postgres:16")
        postgres.start()
    except Exception:  # noqa: BLE001 - Docker unavailable
        return None
    url = postgres.get_connection_url().replace("psycopg2", "psycopg")
    _maybe_testcontainers_url.container = postgres  # type: ignore[attr-defined]
    return url


@pytest.fixture(scope="module")
def pg_session():
    url = _postgres_url() or _maybe_testcontainers_url()
    if url is None:
        pytest.skip("Postgres is not available")
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS chunks CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS ocr_lines CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS pages CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS documents CASCADE"))
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        ensure_search_schema(conn)
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()


def _seed(session: Session) -> None:
    session.rollback()
    session.execute(text("TRUNCATE chunks, ocr_lines, pages, documents RESTART IDENTITY CASCADE"))
    session.commit()
    docs = [
        Document(
            ahpn_id=501,
            source_page="https://example/501",
            visibility="PUB",
            page_count=1,
            document_type="DOC",
            metadata_json={
                "breadcrumbs": [{"name": "JEFATURA DEPARTAMENTAL DE JALAPA", "type": "folder"}],
                "document": {"date": "1983-03-06"},
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        Document(
            ahpn_id=502,
            source_page="https://example/502",
            visibility="PUB",
            page_count=1,
            document_type="DOC",
            metadata_json={
                "breadcrumbs": [{"name": "DIRECCION GENERAL", "type": "folder"}],
                "document": {"date": "1990-01-01"},
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    ]
    session.add_all(docs)
    session.flush()
    session.add_all(
        [
            Chunk(
                document_id=docs[0].id,
                chunk_index=0,
                text="Informe de detenciones por portación de arma en Jalapa.",
                page_start=1,
                page_end=1,
                char_count=55,
                confidence=95,
                line_refs=[],
            ),
            Chunk(
                document_id=docs[1].id,
                chunk_index=0,
                text="Nómina del personal de la Policía Nacional, Dirección General.",
                page_start=1,
                page_end=1,
                char_count=60,
                confidence=95,
                line_refs=[],
            ),
        ]
    )
    session.commit()


def test_spanish_stemming_and_accents(pg_session: Session) -> None:
    _seed(pg_session)
    hits = search(pg_session, "detencion", mode="bm25", k=5)
    assert hits and hits[0].ahpn_id == 501
    hits = search(pg_session, "Policia", mode="bm25", k=5)
    assert hits and hits[0].ahpn_id == 502


def test_self_retrieval_exact_phrase(pg_session: Session) -> None:
    _seed(pg_session)
    phrase = "Informe de detenciones por portación de arma en Jalapa."
    hits = search(pg_session, phrase, mode="bm25", k=1)
    assert hits[0].ahpn_id == 501


def test_department_filter(pg_session: Session) -> None:
    _seed(pg_session)
    hits = search(
        pg_session,
        "Policia",
        mode="bm25",
        k=5,
        filters=SearchFilters(department="DIRECCION GENERAL"),
    )
    assert hits and all(hit.ahpn_id == 502 for hit in hits)
