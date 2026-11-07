from __future__ import annotations

from pathlib import Path

from ahpn.evals.metrics import (
    evaluate_run,
    load_qrels,
    page_hits_from_chunk_hits,
    write_qrels,
)
from ahpn.search import SearchHit


def test_page_hits_from_chunk_hits_dedupes_pages() -> None:
    hits = [
        SearchHit(1, 1, 10, 0, "a", 1, 2, 0.9, 1, "bm25"),
        SearchHit(2, 1, 10, 1, "b", 2, 2, 0.8, 2, "bm25"),
    ]
    pages = page_hits_from_chunk_hits(hits)
    assert [docno for docno, _ in pages] == ["10:1", "10:2"]


def test_page_hits_from_multi_page_chunk_prefer_overlapping_jpeg() -> None:
    hits = [SearchHit(1, 1, 1021, 0, "chunk spanning pages", 1, 3, 0.9, 1, "bm25")]
    page_texts = {
        "1021:1": "Portada del expediente",
        "1021:2": "Continuacion del informe",
        "1021:3": "Testigo clave Rosa Maria Gutierrez placa C-88210",
    }
    pages = page_hits_from_chunk_hits(
        hits,
        query="Rosa Maria Gutierrez placa C-88210",
        page_texts=page_texts,
    )
    assert [docno for docno, _ in pages] == ["1021:3"]


def test_evaluate_run_perfect_ranking() -> None:
    qrels = {"q1": {"1:1": 3, "2:1": 1}}
    run = {"q1": [("1:1", 1.0), ("2:1", 0.5), ("3:1", 0.1)]}
    metrics = evaluate_run(qrels, run, metrics=["recall@10", "ndcg@10", "mrr@10"])
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["ndcg@10"] > 0.9


def test_qrels_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    write_qrels(path, {"q1": {"10:1": 3, "11:2": 1}})
    assert load_qrels(path)["q1"]["10:1"] == 3
