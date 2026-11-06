from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ahpn.evals.fixtures import load_smoke_corpus
from ahpn.evals.pages import (
    assemble_page_text,
    corpus_page_stats,
    page_docno,
    page_image_key,
    parse_page_docno,
)
from ahpn.models import Base, Page
from ahpn.rerank import FakeReranker
from ahpn.search import search

FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "smoke"


def test_page_docno_roundtrip_and_image_key() -> None:
    assert page_docno(1021, 3) == "1021:3"
    assert parse_page_docno("1021:3").page_number == 3
    assert page_image_key(1021, 3) == "1021/pages/0003.jpg"


def test_smoke_corpus_pages_are_jpegs_and_multi_page_recall() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(engine)
    loaded = load_smoke_corpus(session, documents_path=FIXTURES / "documents.jsonl")
    assert loaded["n_documents"] == 21
    assert loaded["n_pages"] == 24
    assert loaded["n_jpeg_pages"] == 24

    pages = session.scalars(select(Page)).all()
    assert pages
    assert all(page.content_type == "image/jpeg" for page in pages)
    assert all(page.image_key.endswith(".jpg") for page in pages)

    stats = corpus_page_stats(session)
    assert stats["multi_page_documents"] >= 2
    assert stats["unit"] == "page_image"

    # Content unique to page 3 of a 3-page JPEG document must surface as 1021:3.
    hits = search(
        session,
        "Rosa Maria Gutierrez placa C-88210",
        mode="bm25_rerank",
        k=5,
        reranker=FakeReranker(),
    )
    assert hits
    assert "1021:3" in hits[0].page_keys()
    assert assemble_page_text(session, 1021, 3).find("Rosa Maria Gutierrez") >= 0
    # Page 1 of the same document should not contain that witness text.
    assert "Rosa Maria" not in assemble_page_text(session, 1021, 1)
