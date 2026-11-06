from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ahpn.embed import FakeEmbedder, embed_chunks
from ahpn.models import Base, Chunk, ChunkEmbedding, Document


def test_fake_embedder_is_deterministic() -> None:
    embedder = FakeEmbedder(dims=16)
    a = embedder.embed(["hola mundo"], input_type="search_document")
    b = embedder.embed(["hola mundo"], input_type="search_query")
    assert a == b
    assert len(a[0]) == 16
    assert abs(sum(x * x for x in a[0]) - 1.0) < 1e-6


def test_embed_chunks_upserts_and_skips() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    document = Document(
        ahpn_id=9,
        source_page="https://example/9",
        visibility="PUB",
        page_count=1,
        document_type="DOC",
        metadata_json={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session.add(document)
    session.flush()
    session.add(
        Chunk(
            document_id=document.id,
            chunk_index=0,
            text="circular confidencial jalapa",
            page_start=1,
            page_end=1,
            char_count=28,
            confidence=99,
            line_refs=[],
        )
    )
    session.commit()
    embedder = FakeEmbedder(model="fake-embed-v1")
    first = embed_chunks(session, embedder)
    second = embed_chunks(session, embedder)
    assert first.chunks_embedded == 1
    assert second.chunks_embedded == 0
    assert second.chunks_skipped == 1
    rows = session.scalars(select(ChunkEmbedding)).all()
    assert len(rows) == 1
    assert rows[0].model == "fake-embed-v1"
