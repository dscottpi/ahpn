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

# Floor thresholds for the synthetic smoke corpus with FakeEmbedder + hybrid search.
MIN_RECALL_AT_10 = 0.8
MIN_NDCG_AT_10 = 0.7


def test_smoke_eval_hybrid_meets_thresholds() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    embedder = FakeEmbedder(model="fake-embed-v1")
    loaded = load_smoke_corpus(session, documents_path=FIXTURES / "documents.jsonl", embedder=embedder)
    assert loaded == 20

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
    assert loaded == 20

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
