from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ahpn.chunk import chunk_stored_document
from ahpn.client import AhpnClient, AhpnError, RestrictedDocumentError
from ahpn.embed import EmbedResult, Embedder, embed_chunks
from ahpn.models import Chunk, Document
from ahpn.ocr import OcrEngine
from ahpn.scrape import scrape_document
from ahpn.storage import DocumentStore

logger = logging.getLogger(__name__)


@dataclass
class DocumentIngestStats:
    ahpn_id: int
    status: str
    pages: int = 0
    chunks: int = 0
    pages_ocred: int = 0
    pages_skipped: int = 0
    chunks_embedded: int = 0
    error: str | None = None
    elapsed_seconds: float = 0.0


@dataclass
class IngestReport:
    started_at: str
    finished_at: str | None = None
    documents: list[DocumentIngestStats] = field(default_factory=list)
    total_pages: int = 0
    total_chunks: int = 0
    total_textract_pages: int = 0
    total_embedded_chunks: int = 0

    def add(self, stats: DocumentIngestStats) -> None:
        self.documents.append(stats)
        if stats.status == "ok":
            self.total_pages += stats.pages
            self.total_chunks += stats.chunks
            self.total_textract_pages += stats.pages_ocred
            self.total_embedded_chunks += stats.chunks_embedded

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def ingest_documents(
    *,
    ids: list[int],
    client: AhpnClient,
    store: DocumentStore,
    session: Session,
    ocr: OcrEngine,
    embedder: Embedder | None = None,
    target_chars: int = 1500,
    overlap_chars: int = 200,
    force_ocr: bool = False,
    skip_scrape: bool = False,
    skip_embed: bool = False,
    report_path: Path | None = None,
) -> IngestReport:
    """Resumable scrape → OCR/chunk → embed pipeline with cost-facing counters."""
    report = IngestReport(started_at=datetime.now(timezone.utc).isoformat())
    for ahpn_id in ids:
        started = time.perf_counter()
        stats = DocumentIngestStats(ahpn_id=ahpn_id, status="ok")
        try:
            if not skip_scrape:
                try:
                    store.get(f"{ahpn_id}/record.json")
                    logger.info("document %s already scraped", ahpn_id)
                except FileNotFoundError:
                    scrape_document(client, store, ahpn_id)

            existing = session.scalar(select(Document).where(Document.ahpn_id == ahpn_id))
            if existing is not None and not force_ocr:
                chunk_count = session.scalar(
                    select(Chunk.id).where(Chunk.document_id == existing.id).limit(1)
                )
                if chunk_count is not None:
                    logger.info("document %s already chunked", ahpn_id)
                    index_result = None
                else:
                    index_result = chunk_stored_document(
                        session,
                        store,
                        ahpn_id,
                        ocr,
                        target_chars=target_chars,
                        overlap_chars=overlap_chars,
                        force=force_ocr,
                    )
            else:
                index_result = chunk_stored_document(
                    session,
                    store,
                    ahpn_id,
                    ocr,
                    target_chars=target_chars,
                    overlap_chars=overlap_chars,
                    force=force_ocr,
                )

            document = session.scalar(select(Document).where(Document.ahpn_id == ahpn_id))
            if document is None:
                raise AhpnError(f"document {ahpn_id} missing after chunk")
            stats.pages = int(document.page_count or 0)
            stats.chunks = len(
                list(session.scalars(select(Chunk.id).where(Chunk.document_id == document.id)))
            )
            if index_result is not None:
                stats.pages = stats.pages or (index_result.pages_ocred + index_result.pages_skipped)
                stats.pages_ocred = index_result.pages_ocred
                stats.pages_skipped = index_result.pages_skipped
                stats.chunks = index_result.chunk_count

            if embedder is not None and not skip_embed:
                embed_result: EmbedResult = embed_chunks(
                    session,
                    embedder,
                    document_ids=[document.id],
                    force=False,
                )
                stats.chunks_embedded = embed_result.chunks_embedded
        except RestrictedDocumentError as exc:
            stats.status = "restricted"
            stats.error = str(exc)
            logger.warning("%s", exc)
        except Exception as exc:  # noqa: BLE001 - per-document resume
            stats.status = "error"
            stats.error = str(exc)
            logger.exception("ingest failed for %s", ahpn_id)
            session.rollback()
        stats.elapsed_seconds = time.perf_counter() - started
        report.add(stats)
        if report_path is not None:
            report.finished_at = datetime.now(timezone.utc).isoformat()
            report.write(report_path)
        logger.info(
            "ingest %s status=%s pages=%s chunks=%s ocr=%s embedded=%s",
            ahpn_id,
            stats.status,
            stats.pages,
            stats.chunks,
            stats.pages_ocred,
            stats.chunks_embedded,
        )

    report.finished_at = datetime.now(timezone.utc).isoformat()
    if report_path is not None:
        report.write(report_path)
    return report


def project_costs(
    *,
    sample_docs: int,
    sample_pages: int,
    sample_chunks: int,
    target_docs: int = 20_000_000,
    textract_per_1k_pages: float = 1.50,
    embed_per_1k_chunks: float = 0.10,
) -> dict[str, float]:
    """Extrapolate OCR/embed cost from a measured sample to a target corpus size."""
    if sample_docs <= 0:
        raise ValueError("sample_docs must be positive")
    pages_per_doc = sample_pages / sample_docs
    chunks_per_doc = sample_chunks / sample_docs
    projected_pages = pages_per_doc * target_docs
    projected_chunks = chunks_per_doc * target_docs
    return {
        "pages_per_doc": pages_per_doc,
        "chunks_per_doc": chunks_per_doc,
        "projected_pages": projected_pages,
        "projected_chunks": projected_chunks,
        "projected_textract_usd": (projected_pages / 1000.0) * textract_per_1k_pages,
        "projected_embed_usd": (projected_chunks / 1000.0) * embed_per_1k_chunks,
    }
