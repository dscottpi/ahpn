from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ahpn.chunk import chunk_lines
from ahpn.embed import Embedder, embed_chunks
from ahpn.evals.pages import is_jpeg_content_type, page_image_key
from ahpn.models import Chunk, Document, OcrLine, Page
from ahpn.ocr import TextLine


def load_smoke_corpus(
    session: Session,
    *,
    documents_path: Path,
    embedder: Embedder | None = None,
    target_chars: int = 800,
    overlap_chars: int = 100,
) -> dict[str, int]:
    """Load synthetic OCR-line documents for the committed smoke eval.

    Each document is modeled as an ordered list of JPEG page images. Fixture
    pages may include ``image_key`` / ``content_type``; defaults are
    ``{ahpn_id}/pages/NNNN.jpg`` and ``image/jpeg``.
    """
    n_documents = 0
    n_pages = 0
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ahpn_id = int(row["ahpn_id"])
        pages = list(row.get("pages") or [])
        if not pages:
            raise ValueError(f"document {ahpn_id} has no page images")
        now = datetime.now(timezone.utc)
        document = Document(
            ahpn_id=ahpn_id,
            source_page=f"https://ahpn.lib.utexas.edu/document/{ahpn_id}",
            visibility="PUB",
            page_count=len(pages),
            document_type=row.get("document_type"),
            metadata_json={
                "document": {"date": row.get("date"), "document_type": row.get("document_type")},
                "breadcrumbs": [
                    {"name": "GT PN - Policía Nacional", "type": "folder"},
                    {"name": row.get("department") or "", "type": "folder"},
                ],
            },
            created_at=now,
            updated_at=now,
        )
        session.add(document)
        session.flush()
        text_lines: list[TextLine] = []
        for page in pages:
            page_number = int(page["page_number"])
            content_type = str(page.get("content_type") or "image/jpeg")
            if not is_jpeg_content_type(content_type):
                raise ValueError(
                    f"document {ahpn_id} page {page_number} must be a JPEG "
                    f"(got content_type={content_type!r})"
                )
            image_key = str(page.get("image_key") or page_image_key(ahpn_id, page_number))
            page_row = Page(
                document_id=document.id,
                page_number=page_number,
                image_key=image_key,
                content_type=content_type.split(";", 1)[0].strip().lower(),
                byte_count=int(page.get("byte_count") or 0),
            )
            session.add(page_row)
            session.flush()
            n_pages += 1
            for index, text in enumerate(page.get("lines") or []):
                session.add(
                    OcrLine(
                        page_id=page_row.id,
                        line_index=index,
                        text=text,
                        confidence=95.0,
                        top=index / 100,
                        left=0.1,
                        width=0.8,
                        height=0.03,
                    )
                )
                text_lines.append(
                    TextLine(
                        page_number=page_number,
                        line_index=index,
                        text=text,
                        confidence=95.0,
                        top=index / 100,
                        left=0.1,
                        width=0.8,
                        height=0.03,
                    )
                )
        drafts = chunk_lines(text_lines, target_chars=target_chars, overlap_chars=overlap_chars)
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
        n_documents += 1
    session.commit()
    if embedder is not None:
        embed_chunks(session, embedder)
    return {"n_documents": n_documents, "n_pages": n_pages, "n_jpeg_pages": n_pages}
