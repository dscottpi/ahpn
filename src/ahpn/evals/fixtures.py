from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from ahpn.chunk import chunk_lines
from ahpn.embed import Embedder, embed_chunks
from ahpn.models import Chunk, Document, OcrLine, Page
from ahpn.ocr import TextLine


def load_smoke_corpus(
    session: Session,
    *,
    documents_path: Path,
    embedder: Embedder | None = None,
    target_chars: int = 800,
    overlap_chars: int = 100,
) -> int:
    """Load synthetic OCR-line documents for the committed smoke eval."""
    count = 0
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ahpn_id = int(row["ahpn_id"])
        now = datetime.now(timezone.utc)
        document = Document(
            ahpn_id=ahpn_id,
            source_page=f"https://ahpn.lib.utexas.edu/document/{ahpn_id}",
            visibility="PUB",
            page_count=len(row.get("pages") or []),
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
        for page in row.get("pages") or []:
            page_row = Page(
                document_id=document.id,
                page_number=int(page["page_number"]),
                image_key=f"{ahpn_id}/pages/{int(page['page_number']):04d}.jpg",
                content_type="image/jpeg",
                byte_count=0,
            )
            session.add(page_row)
            session.flush()
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
                        page_number=int(page["page_number"]),
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
        count += 1
    session.commit()
    if embedder is not None:
        embed_chunks(session, embedder)
    return count
