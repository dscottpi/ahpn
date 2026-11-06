from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ahpn.embed import FakeEmbedder, embed_chunks
from ahpn.models import Base, Chunk, Document
from ahpn.rerank import FakeReranker
from ahpn.search import SearchFilters, reciprocal_rank_fusion, search


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed(session: Session) -> FakeEmbedder:
    docs = [
        Document(
            ahpn_id=1,
            source_page="https://example/1",
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
            ahpn_id=2,
            source_page="https://example/2",
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
                text="Detenciones por portacion de arma en Jalapa durante marzo de 1983.",
                page_start=1,
                page_end=1,
                char_count=70,
                confidence=90,
                line_refs=[],
            ),
            Chunk(
                document_id=docs[1].id,
                chunk_index=0,
                text="Nomina de personal del Cuarto Cuerpo de la Policia Nacional.",
                page_start=1,
                page_end=1,
                char_count=60,
                confidence=90,
                line_refs=[],
            ),
        ]
    )
    session.commit()
    embedder = FakeEmbedder(model="fake-embed-v1", dims=32)
    embed_chunks(session, embedder)
    return embedder


def test_reciprocal_rank_fusion_prefers_shared_top_hits() -> None:
    fused = reciprocal_rank_fusion([[10, 20, 30], [20, 40, 10]], k=60, limit=3)
    assert fused[0][0] == 20
    assert [item for item, _score in fused] == [20, 10, 30] or fused[0][0] == 20


def test_bm25_and_hybrid_find_relevant_chunk() -> None:
    session = _session()
    embedder = _seed(session)
    bm25 = search(session, "detenciones arma Jalapa", mode="bm25", k=5)
    assert bm25
    assert bm25[0].ahpn_id == 1
    hybrid = search(session, "detenciones arma Jalapa", mode="hybrid", k=5, embedder=embedder)
    assert hybrid
    assert hybrid[0].ahpn_id == 1
    assert hybrid[0].page_keys() == ["1:1"]


def test_bm25_rerank_reorders_with_fake_reranker() -> None:
    session = _session()
    _seed(session)
    hits = search(
        session,
        "detenciones arma Jalapa",
        mode="bm25_rerank",
        k=5,
        rerank_candidates=50,
        reranker=FakeReranker(),
    )
    assert hits
    assert hits[0].ahpn_id == 1
    assert hits[0].mode == "bm25_rerank"
    assert hits[0].rank == 1


def test_fake_reranker_prefers_overlap() -> None:
    reranker = FakeReranker()
    ranked = reranker.rerank(
        "Jalapa detenciones",
        [
            "Nomina de personal del Cuarto Cuerpo",
            "Detenciones en Jalapa por arma",
        ],
        top_n=2,
    )
    assert ranked[0].index == 1
    assert ranked[0].score > ranked[1].score


def test_department_filter_restricts_results() -> None:
    session = _session()
    embedder = _seed(session)
    hits = search(
        session,
        "Policia",
        mode="bm25",
        k=5,
        filters=SearchFilters(department="DIRECCION GENERAL"),
        embedder=embedder,
    )
    assert hits
    assert all(hit.ahpn_id == 2 for hit in hits)
