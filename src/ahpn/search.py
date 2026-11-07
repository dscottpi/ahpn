from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from sqlalchemy import Select, Text, func, literal_column, select, text
from sqlalchemy.orm import Session

from ahpn.config import DEFAULT_RERANK_CANDIDATES, DEFAULT_SEARCH_CANDIDATES, DEFAULT_SEARCH_K
from ahpn.db import AHPN_ES_CONFIG
from ahpn.models import Chunk, Document
from ahpn.rerank import Reranker

logger = logging.getLogger(__name__)

SearchMode = Literal["bm25", "bm25_rerank"]


@dataclass(frozen=True)
class SearchFilters:
    """Optional metadata filters applied before ranking."""

    department: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    document_ids: frozenset[int] | None = None
    ahpn_ids: frozenset[int] | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    document_id: int
    ahpn_id: int
    chunk_index: int
    text: str
    page_start: int
    page_end: int
    score: float
    rank: int
    mode: str

    @property
    def pages(self) -> range:
        return range(self.page_start, self.page_end + 1)

    @property
    def page_key(self) -> str:
        """First page key used for page-level qrels (`ahpn_id:page`)."""
        return f"{self.ahpn_id}:{self.page_start}"

    def page_keys(self) -> list[str]:
        return [f"{self.ahpn_id}:{page}" for page in self.pages]


def search(
    session: Session,
    query: str,
    *,
    mode: SearchMode = "bm25_rerank",
    k: int = DEFAULT_SEARCH_K,
    candidates: int = DEFAULT_SEARCH_CANDIDATES,
    filters: SearchFilters | None = None,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
) -> list[SearchHit]:
    """Search chunks with BM25, optionally reranked."""
    query = query.strip()
    if not query:
        return []
    filters = filters or SearchFilters()

    if mode == "bm25":
        rows = _bm25_search(session, query, limit=max(k, candidates), filters=filters)
        return _hits_from_rows(rows, mode=mode, k=k)

    if mode != "bm25_rerank":
        raise ValueError(f"unknown search mode: {mode}")
    if reranker is None:
        raise ValueError("bm25_rerank requires a reranker")

    first_stage = max(k, rerank_candidates, candidates)
    rows = _bm25_search(session, query, limit=first_stage, filters=filters)
    shortlist = rows[: max(k, rerank_candidates)]
    if not shortlist:
        return []
    ranked = reranker.rerank(query, [row["text"] for row in shortlist], top_n=min(k, len(shortlist)))
    hits: list[SearchHit] = []
    for rank, item in enumerate(ranked[:k], start=1):
        row = shortlist[item.index]
        hits.append(
            SearchHit(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                ahpn_id=row["ahpn_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                score=float(item.score),
                rank=rank,
                mode=mode,
            )
        )
    return hits


def _hits_from_rows(rows: list[dict[str, Any]], *, mode: str, k: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for rank, row in enumerate(rows[:k], start=1):
        hits.append(
            SearchHit(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                ahpn_id=row["ahpn_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                score=float(row["score"]),
                rank=rank,
                mode=mode,
            )
        )
    return hits


def _apply_filters(stmt: Select[Any], filters: SearchFilters) -> Select[Any]:
    if filters.document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(list(filters.document_ids)))
    if filters.ahpn_ids is not None:
        stmt = stmt.where(Document.ahpn_id.in_(list(filters.ahpn_ids)))
    if filters.department:
        pattern = f"%{filters.department}%"
        stmt = stmt.where(func.cast(Document.metadata_json, Text).ilike(pattern))
    if filters.date_from is not None or filters.date_to is not None:
        doc_date = Document.metadata_json["document"]["date"].as_string()
        if filters.date_from is not None:
            stmt = stmt.where(doc_date >= filters.date_from.isoformat())
        if filters.date_to is not None:
            stmt = stmt.where(doc_date <= filters.date_to.isoformat())
    return stmt


def _bm25_search(session: Session, query: str, *, limit: int, filters: SearchFilters) -> list[dict[str, Any]]:
    if session.bind is not None and session.bind.dialect.name != "postgresql":
        return _bm25_fallback(session, query, limit=limit, filters=filters)

    config = text(f"'{AHPN_ES_CONFIG}'::regconfig")
    ts_query = func.plainto_tsquery(config, query)
    search_vector = literal_column("chunks.search_vector")
    rank = func.ts_rank_cd(search_vector, ts_query).label("score")
    stmt = (
        select(
            Chunk.id.label("chunk_id"),
            Chunk.document_id.label("document_id"),
            Document.ahpn_id.label("ahpn_id"),
            Chunk.chunk_index.label("chunk_index"),
            Chunk.text.label("text"),
            Chunk.page_start.label("page_start"),
            Chunk.page_end.label("page_end"),
            rank,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(search_vector.op("@@")(ts_query))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    stmt = _apply_filters(stmt, filters)
    return [dict(row._mapping) for row in session.execute(stmt)]


def _bm25_fallback(session: Session, query: str, *, limit: int, filters: SearchFilters) -> list[dict[str, Any]]:
    """Simple token-count ranking used on SQLite in unit tests."""
    tokens = [token for token in query.lower().split() if token]
    stmt = (
        select(
            Chunk.id.label("chunk_id"),
            Chunk.document_id.label("document_id"),
            Document.ahpn_id.label("ahpn_id"),
            Chunk.chunk_index.label("chunk_index"),
            Chunk.text.label("text"),
            Chunk.page_start.label("page_start"),
            Chunk.page_end.label("page_end"),
        )
        .join(Document, Document.id == Chunk.document_id)
    )
    stmt = _apply_filters(stmt, filters)
    rows: list[dict[str, Any]] = []
    for row in session.execute(stmt):
        text_l = (row.text or "").lower()
        score = float(sum(text_l.count(token) for token in tokens))
        if tokens and score <= 0:
            continue
        rows.append(
            {
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "ahpn_id": row.ahpn_id,
                "chunk_index": row.chunk_index,
                "text": row.text,
                "page_start": row.page_start,
                "page_end": row.page_end,
                "score": score if tokens else 1.0,
            }
        )
    rows.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return rows[:limit]
