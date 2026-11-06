from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ahpn.embed import FakeEmbedder
from ahpn.evals.fixtures import load_smoke_corpus
from ahpn.evals.metrics import evaluate_run, load_qrels, load_queries, page_hits_from_chunk_hits
from ahpn.models import Base
from ahpn.rerank import FakeReranker
from ahpn.search import search

FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "smoke"

# Floor thresholds for the synthetic smoke corpus.
MIN_RECALL_AT_10 = 0.8
MIN_NDCG_AT_10 = 0.7


def test_smoke_eval_hybrid_meets_thresholds() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    embedder = FakeEmbedder(model="fake-embed-v1")
    loaded = load_smoke_corpus(session, documents_path=FIXTURES / "documents.jsonl", embedder=embedder)
    assert loaded["n_documents"] == 21
    assert loaded["n_jpeg_pages"] == 24

    queries = load_queries(FIXTURES / "queries.jsonl")
    qrels = load_qrels(FIXTURES / "qrels.tsv")
    run = {}
    for query in queries:
        hits = search(session, query.text, mode="hybrid", k=10, candidates=20, embedder=embedder)
        run[query.id] = page_hits_from_chunk_hits(hits)

    metrics = evaluate_run(qrels, run, metrics=["recall@10", "ndcg@10", "mrr@10"])
    assert metrics["recall@10"] >= MIN_RECALL_AT_10, metrics
    assert metrics["ndcg@10"] >= MIN_NDCG_AT_10, metrics


def test_smoke_eval_bm25_rerank_meets_thresholds() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    loaded = load_smoke_corpus(session, documents_path=FIXTURES / "documents.jsonl")
    assert loaded["n_documents"] == 21

    queries = load_queries(FIXTURES / "queries.jsonl")
    qrels = load_qrels(FIXTURES / "qrels.tsv")
    reranker = FakeReranker()
    run = {}
    for query in queries:
        hits = search(
            session,
            query.text,
            mode="bm25_rerank",
            k=10,
            rerank_candidates=50,
            reranker=reranker,
        )
        run[query.id] = page_hits_from_chunk_hits(hits)

    metrics = evaluate_run(qrels, run, metrics=["recall@10", "ndcg@10", "mrr@10"])
    assert metrics["recall@10"] >= MIN_RECALL_AT_10, metrics
    assert metrics["ndcg@10"] >= MIN_NDCG_AT_10, metrics


def test_smoke_multi_page_query_hits_specific_jpeg_page() -> None:
    """A hit on page 3 of a 3-JPEG document must be scored as 1021:3, not just doc 1021."""
    from ahpn.evals.pages import assemble_page_text, parse_page_docno

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    load_smoke_corpus(session, documents_path=FIXTURES / "documents.jsonl")
    query = "Rosa Maria Gutierrez vehiculo placa C-88210"
    hits = search(
        session,
        query,
        mode="bm25_rerank",
        k=10,
        reranker=FakeReranker(),
    )
    page_texts = {}
    for hit in hits:
        for key in hit.page_keys():
            if key in page_texts:
                continue
            ref = parse_page_docno(key)
            page_texts[key] = assemble_page_text(session, ref.ahpn_id, ref.page_number)
    page_run = page_hits_from_chunk_hits(hits, query=query, page_texts=page_texts)
    assert page_run
    assert page_run[0][0] == "1021:3"
    # Sibling JPEGs in the same document should not inherit credit without overlap.
    assert "1021:1" not in {docno for docno, _score in page_run[:3]}
