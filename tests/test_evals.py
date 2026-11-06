from __future__ import annotations

from pathlib import Path

from ahpn.evals.generate import assign_splits, known_item_queries, metadata_queries, no_answer_queries
from ahpn.evals.judge import parse_grade_response
from ahpn.evals.kappa import cohen_kappa, meets_calibration_threshold
from ahpn.evals.metrics import (
    QueryRecord,
    evaluate_run,
    load_qrels,
    page_hits_from_chunk_hits,
    write_qrels,
    write_queries,
)
from ahpn.evals.ocr_eval import bag_word_error_rate, char_error_rate, word_error_rate
from ahpn.evals.review import review_queries
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


def test_parse_grade_response_and_kappa() -> None:
    assert parse_grade_response('{"grade": 3, "rationale": "yes"}') == 3
    assert parse_grade_response('```json\n{"grade": 1}\n```') == 1
    kappa = cohen_kappa([3, 2, 0, 1], [3, 2, 0, 2])
    assert 0.5 < kappa < 1.0
    assert meets_calibration_threshold(0.6)
    assert not meets_calibration_threshold(0.59)


def test_query_generators_and_review(tmp_path: Path) -> None:
    meta = metadata_queries(
        ahpn_id=5,
        breadcrumbs=[{"name": "GT PN", "type": "folder"}, {"name": "Jalapa", "type": "folder"}],
        date="1983-03-06",
        start_id=1,
    )
    assert meta and "Jalapa" in meta[0].text
    known = known_item_queries(
        ahpn_id=5,
        page_number=1,
        lines=["x", "Circular Confidencial No. 12/83", "y"],
        start_id=10,
    )
    assert known[0].types
    none = no_answer_queries(100)
    assert any("no_answer" in q.types for q in none)
    split = assign_splits(meta + known + none)
    assert {q.split for q in split} == {"dev", "test"} or len(split) < 2

    pending = tmp_path / "pending.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    write_queries(pending, [QueryRecord(id="p1", text="hola", lang="es", types=["conceptual"], source="llm")])
    answers = iter(["a", "q"])

    kept, edited, rejected = review_queries(
        pending,
        accepted,
        input_fn=lambda _prompt: next(answers),
        print_fn=lambda *_args, **_kwargs: None,
    )
    assert kept == 1
    assert edited == 0
    assert rejected == 0


def test_ocr_error_rates() -> None:
    assert char_error_rate("hola", "hola") == 0.0
    assert word_error_rate("hola mundo", "hola mundo") == 0.0
    assert bag_word_error_rate("a b c", "c a b") == 0.0
    assert bag_word_error_rate("a b c", "a b") > 0.0


def test_qrels_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    write_qrels(path, {"q1": {"10:1": 3, "11:2": 1}})
    assert load_qrels(path)["q1"]["10:1"] == 3
