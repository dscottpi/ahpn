import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from ahpn.chunk import chunk_lines, chunk_stored_document, index_document
from ahpn.chunk import PageImage as IndexPage
from ahpn.models import Base, Chunk, OcrLine
from ahpn.ocr import TextLine, lines_from_blocks
from ahpn.storage import LocalDocumentStore


def _line(page: int, index: int, text: str) -> TextLine:
    return TextLine(
        page_number=page,
        line_index=index,
        text=text,
        confidence=90,
        top=index / 100,
        left=0.1,
        width=0.8,
        height=0.02,
    )


def test_chunk_lines_overlap_and_page_span() -> None:
    lines = [_line(1, index, "a" * 40) for index in range(4)]
    lines.extend(_line(2, index, "b" * 40) for index in range(2))
    chunks = chunk_lines(lines, target_chars=100, overlap_chars=40)

    assert len(chunks) >= 2
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert all(chunk.char_count <= 100 or "\n" not in chunk.text for chunk in chunks)
    # The second chunk repeats the tail of the first.
    assert chunks[1].text.startswith(chunks[0].text.split("\n")[-1])


def test_long_line_does_not_stall_the_chunker() -> None:
    lines = [_line(1, 0, "x" * 500), _line(1, 1, "after")]
    chunks = chunk_lines(lines, target_chars=100, overlap_chars=40)
    assert [chunk.text for chunk in chunks] == ["x" * 500, "after"]


def test_lines_from_textract_blocks_keep_line_order() -> None:
    blocks = [
        {"BlockType": "PAGE", "Id": "page"},
        {
            "BlockType": "LINE",
            "Text": "CIRCULAR",
            "Confidence": 99.1,
            "Geometry": {"BoundingBox": {"Top": 0.2, "Left": 0.4, "Width": 0.2, "Height": 0.03}},
        },
        {"BlockType": "WORD", "Text": "CIRCULAR"},
        {
            "BlockType": "LINE",
            "Text": "Guatemala",
            "Confidence": 80,
            "Geometry": {"BoundingBox": {"Top": 0.4, "Left": 0.1, "Width": 0.3, "Height": 0.02}},
        },
    ]
    lines = lines_from_blocks(3, blocks)
    assert [line.text for line in lines] == ["CIRCULAR", "Guatemala"]
    assert lines[0].line_index == 0
    assert lines[0].page_number == 3
    assert lines[0].top == pytest.approx(0.2)


class FakeOcr:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def detect_lines(self, page_number: int, image: bytes) -> list[TextLine]:
        self.calls.append(page_number)
        return [_line(page_number, index, f"page {page_number} line {index} " + ("palabra " * 12)) for index in range(6)]


def _session() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_index_document_stores_lines_and_reuses_them() -> None:
    ocr = FakeOcr()
    session = _session()
    pages = [
        IndexPage(page_number=1, image=b"one", content_type="image/jpeg", image_key="1/pages/0001.jpg"),
        IndexPage(page_number=2, image=b"two", content_type="image/jpeg", image_key="1/pages/0002.jpg"),
    ]
    kwargs = dict(
        ahpn_id=2127283,
        source_page="https://ahpn.lib.utexas.edu/document/2127283",
        visibility="PUB",
        page_count=2,
        document_type="DOC",
        metadata={"breadcrumbs": []},
        pages=pages,
        ocr=ocr,
        target_chars=180,
        overlap_chars=40,
    )
    first = index_document(session, **kwargs)
    second = index_document(session, **kwargs)

    assert first.pages_ocred == 2
    assert first.chunk_count > 1
    assert second.pages_ocred == 0
    assert second.pages_skipped == 2
    assert second.chunk_count == first.chunk_count
    assert ocr.calls == [1, 2]
    assert session.scalar(select(func.count()).select_from(OcrLine)) == 12
    stored = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert [chunk.chunk_index for chunk in stored] == list(range(len(stored)))
    assert stored[0].line_refs


def test_chunk_stored_document_reads_the_scrape_record(tmp_path: Path) -> None:
    record = {
        "document_id": 7,
        "source_page": "https://ahpn.lib.utexas.edu/document/7",
        "visibility": "PUB",
        "page_count": 1,
        "document": {"document_type": "PAG"},
        "breadcrumbs": [{"id": 1, "name": "Jalapa", "type": "folder"}],
        "pages": [{"page_number": 1, "key": "7/pages/0001.jpg", "content_type": "image/jpeg"}],
    }
    store = LocalDocumentStore(tmp_path)
    store.put("7/record.json", json.dumps(record).encode(), "application/json")
    store.put("7/pages/0001.jpg", b"jpeg", "image/jpeg")
    ocr = FakeOcr()
    result = chunk_stored_document(_session(), store, 7, ocr, target_chars=200, overlap_chars=40)
    assert result.ahpn_id == 7
    assert result.pages_ocred == 1
    assert result.chunk_count >= 1
