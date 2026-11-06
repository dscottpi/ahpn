from __future__ import annotations

"""Page-image identity for the AHPN archive.

Every AHPN document is one JPEG page image, or an ordered list of JPEG page
images. The eval harness treats the page image as the atomic relevance unit
(`ahpn_id:page_number`), not the parent document and not the retrieval chunk.
"""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ahpn.models import Document, OcrLine, Page

JPEG_CONTENT_TYPES = frozenset({"image/jpeg", "image/jpg"})


@dataclass(frozen=True)
class PageRef:
    ahpn_id: int
    page_number: int

    @property
    def docno(self) -> str:
        return page_docno(self.ahpn_id, self.page_number)

    @property
    def image_key(self) -> str:
        return page_image_key(self.ahpn_id, self.page_number)


def page_docno(ahpn_id: int, page_number: int) -> str:
    """TREC docno for one page image."""
    return f"{int(ahpn_id)}:{int(page_number)}"


def parse_page_docno(docno: str) -> PageRef:
    ahpn_id_s, page_s = docno.split(":", 1)
    return PageRef(ahpn_id=int(ahpn_id_s), page_number=int(page_s))


def page_image_key(ahpn_id: int, page_number: int) -> str:
    """Canonical storage key for a page JPEG under the document prefix."""
    return f"{int(ahpn_id)}/pages/{int(page_number):04d}.jpg"


def is_jpeg_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() in JPEG_CONTENT_TYPES


def assemble_page_text(session: Session, ahpn_id: int, page_number: int) -> str:
    """Rebuild one page image's OCR text from stored lines (not chunk text)."""
    document = session.scalar(select(Document).where(Document.ahpn_id == ahpn_id))
    if document is None:
        return ""
    page = session.scalar(
        select(Page).where(Page.document_id == document.id, Page.page_number == page_number)
    )
    if page is None:
        return ""
    lines = session.scalars(
        select(OcrLine).where(OcrLine.page_id == page.id).order_by(OcrLine.line_index)
    ).all()
    return "\n".join(line.text for line in lines if line.text)


def corpus_page_stats(session: Session) -> dict[str, float | int]:
    """Summarize the indexed corpus in page-image terms."""
    n_documents = int(session.scalar(select(func.count()).select_from(Document)) or 0)
    n_pages = int(session.scalar(select(func.count()).select_from(Page)) or 0)
    jpeg_pages = 0
    for content_type in session.scalars(select(Page.content_type)):
        if is_jpeg_content_type(content_type):
            jpeg_pages += 1
    multi_page_docs = 0
    if n_documents:
        counts = session.execute(
            select(Page.document_id, func.count())
            .group_by(Page.document_id)
        )
        for _document_id, page_count in counts:
            if int(page_count) > 1:
                multi_page_docs += 1
    return {
        "unit": "page_image",
        "n_documents": n_documents,
        "n_pages": n_pages,
        "n_jpeg_pages": jpeg_pages,
        "multi_page_documents": multi_page_docs,
        "pages_per_doc": (n_pages / n_documents) if n_documents else 0.0,
    }
