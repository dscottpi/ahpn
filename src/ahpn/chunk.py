from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ahpn.client import AhpnError
from ahpn.config import DEFAULT_CHUNK_CHARS, DEFAULT_CHUNK_OVERLAP
from ahpn.models import Chunk, Document, OcrLine, Page
from ahpn.ocr import OcrEngine, TextLine
from ahpn.storage import DocumentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    confidence: float
    line_refs: tuple[dict, ...]

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class PageImage:
    page_number: int
    image: bytes
    content_type: str
    image_key: str


@dataclass(frozen=True)
class IndexResult:
    ahpn_id: int
    pages_ocred: int
    pages_skipped: int
    chunk_count: int


def chunk_lines(
    lines: list[TextLine],
    *,
    target_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkDraft]:
    """Group OCR lines into overlapping chunks, breaking only on line boundaries."""
    if target_chars < 1:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than target_chars")

    chunks: list[ChunkDraft] = []
    start = 0
    while start < len(lines):
        end = start
        size = 0
        while end < len(lines):
            extra = len(lines[end].text) + (1 if end > start else 0)
            if end > start and size + extra > target_chars:
                break
            size += extra
            end += 1
        if end == start:
            end = start + 1
        draft = _draft(lines[start:end], len(chunks))
        if draft is not None:
            chunks.append(draft)
        if end >= len(lines):
            break
        next_start = _overlap_start(lines, start, end, overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def index_document(
    session: Session,
    *,
    ahpn_id: int,
    source_page: str,
    visibility: str,
    page_count: int | None,
    document_type: str | None,
    metadata: dict,
    pages: list[PageImage],
    ocr: OcrEngine,
    target_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
) -> IndexResult:
    """OCR any pages that do not already have lines, then replace the document's chunks."""
    document = _upsert_document(
        session,
        ahpn_id=ahpn_id,
        source_page=source_page,
        visibility=visibility,
        page_count=page_count,
        document_type=document_type,
        metadata=metadata,
    )
    page_rows = {
        page.page_number: page
        for page in session.scalars(select(Page).where(Page.document_id == document.id))
    }
    pages_ocred = 0
    pages_skipped = 0
    for image in pages:
        page = page_rows.get(image.page_number)
        if page is None:
            page = Page(
                document_id=document.id,
                page_number=image.page_number,
                image_key=image.image_key,
                content_type=image.content_type,
                byte_count=len(image.image),
            )
            session.add(page)
            session.flush()
            page_rows[image.page_number] = page
        else:
            page.image_key = image.image_key
            page.content_type = image.content_type
            page.byte_count = len(image.image)

        existing = session.scalars(select(OcrLine.id).where(OcrLine.page_id == page.id).limit(1)).first()
        if existing is not None and not force:
            pages_skipped += 1
            logger.info("page %s already has OCR lines", image.page_number)
            continue

        logger.info("reading page %s", image.page_number)
        lines = ocr.detect_lines(image.page_number, image.image)
        session.execute(delete(OcrLine).where(OcrLine.page_id == page.id))
        session.add_all(_line_rows(page.id, lines))
        pages_ocred += 1

    session.flush()
    stored_lines = list(
        session.execute(
            select(OcrLine, Page.page_number)
            .join(Page, OcrLine.page_id == Page.id)
            .where(Page.document_id == document.id)
            .order_by(Page.page_number, OcrLine.line_index)
        )
    )
    text_lines = [
        TextLine(
            page_number=page_number,
            line_index=line.line_index,
            text=line.text,
            confidence=line.confidence,
            top=line.top,
            left=line.left,
            width=line.width,
            height=line.height,
        )
        for line, page_number in stored_lines
    ]
    drafts = chunk_lines(text_lines, target_chars=target_chars, overlap_chars=overlap_chars)
    session.execute(delete(Chunk).where(Chunk.document_id == document.id))
    session.add_all(
        Chunk(
            document_id=document.id,
            chunk_index=draft.chunk_index,
            text=draft.text,
            page_start=draft.page_start,
            page_end=draft.page_end,
            char_count=draft.char_count,
            confidence=draft.confidence,
            line_refs=list(draft.line_refs),
        )
        for draft in drafts
    )
    session.commit()
    return IndexResult(
        ahpn_id=ahpn_id,
        pages_ocred=pages_ocred,
        pages_skipped=pages_skipped,
        chunk_count=len(drafts),
    )


def chunk_stored_document(
    session: Session,
    store: DocumentStore,
    document_id: int,
    ocr: OcrEngine,
    *,
    target_chars: int = DEFAULT_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
) -> IndexResult:
    """Load a scraped document from storage and index its pages."""
    try:
        raw = store.get(f"{document_id}/record.json")
    except FileNotFoundError as exc:
        raise AhpnError(f"document {document_id} has not been scraped yet") from exc
    record = json.loads(raw)
    document = record.get("document") or {}
    pages = [
        PageImage(
            page_number=page["page_number"],
            image=store.get(page["key"]),
            content_type=page.get("content_type") or "application/octet-stream",
            image_key=page["key"],
        )
        for page in record.get("pages") or []
    ]
    if not pages:
        raise AhpnError(f"document {document_id} record has no page images")
    return index_document(
        session,
        ahpn_id=document_id,
        source_page=record.get("source_page") or "",
        visibility=record.get("visibility") or "",
        page_count=record.get("page_count"),
        document_type=document.get("document_type"),
        metadata={
            "document": document,
            "breadcrumbs": record.get("breadcrumbs") or [],
            "scraped_at": record.get("scraped_at"),
        },
        pages=pages,
        ocr=ocr,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        force=force,
    )


def _upsert_document(
    session: Session,
    *,
    ahpn_id: int,
    source_page: str,
    visibility: str,
    page_count: int | None,
    document_type: str | None,
    metadata: dict,
) -> Document:
    document = session.scalar(select(Document).where(Document.ahpn_id == ahpn_id))
    now = datetime.now(timezone.utc)
    if document is None:
        document = Document(
            ahpn_id=ahpn_id,
            source_page=source_page,
            visibility=visibility,
            page_count=page_count,
            document_type=document_type,
            metadata_json=metadata,
            created_at=now,
            updated_at=now,
        )
        session.add(document)
    else:
        document.source_page = source_page
        document.visibility = visibility
        document.page_count = page_count
        document.document_type = document_type
        document.metadata_json = metadata
        document.updated_at = now
    session.flush()
    return document


def _line_rows(page_id: int, lines: list[TextLine]) -> list[OcrLine]:
    return [
        OcrLine(
            page_id=page_id,
            line_index=line.line_index,
            text=line.text,
            confidence=line.confidence,
            top=line.top,
            left=line.left,
            width=line.width,
            height=line.height,
        )
        for line in lines
    ]


def _draft(group: list[TextLine], chunk_index: int) -> ChunkDraft | None:
    text = "\n".join(line.text for line in group).strip()
    if not text:
        return None
    return ChunkDraft(
        chunk_index=chunk_index,
        text=text,
        page_start=group[0].page_number,
        page_end=group[-1].page_number,
        confidence=sum(line.confidence for line in group) / len(group),
        line_refs=tuple(
            {
                "page_number": line.page_number,
                "line_index": line.line_index,
                "top": line.top,
                "left": line.left,
                "width": line.width,
                "height": line.height,
            }
            for line in group
            if line.text.strip()
        ),
    )


def _overlap_start(lines: list[TextLine], start: int, end: int, overlap_chars: int) -> int:
    """Index of the first line that should repeat at the start of the next chunk."""
    if overlap_chars == 0:
        return end
    size = 0
    index = end
    while index > start + 1:
        size += len(lines[index - 1].text) + (1 if size else 0)
        index -= 1
        if size >= overlap_chars:
            break
    return index
