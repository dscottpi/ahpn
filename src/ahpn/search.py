from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Sequence

from sqlalchemy import Select, Text, func, literal, literal_column, select, text
from sqlalchemy.orm import Session, aliased

from ahpn.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SEARCH_CANDIDATES,
    DEFAULT_SEARCH_K,
    RRF_K,
)
from ahpn.db import AHPN_ES_CONFIG
from ahpn.embed import Embedder
from ahpn.models import Chunk, ChunkEmbedding, Document
from ahpn.rerank import Reranker

logger = logging.getLogger(__name__)

SearchMode = Literal["bm25", "bm25_rerank", "vector", "hybrid"]


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


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[tuple[int, float]]:
    """Merge ranked id lists with classic RRF. Higher score is better."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return ordered


def chunk_to_page_keys(ahpn_id: int, page_start: int, page_end: int) -> list[str]:
    return [f"{ahpn_id}:{page}" for page in range(page_start, page_end + 1)]


def search(
    session: Session,
    query: str,
    *,
    mode: SearchMode = "bm25_rerank",
    k: int = DEFAULT_SEARCH_K,
    candidates: int = DEFAULT_SEARCH_CANDIDATES,
    filters: SearchFilters | None = None,
    embedder: Embedder | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    reranker: Reranker | None = None,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ef_search: int | None = DEFAULT_HNSW_EF_SEARCH,
    exact_vector: bool = False,
) -> list[SearchHit]:
    """Search chunks with BM25, BM25+rerank, vector, or hybrid (RRF) ranking."""
    query = query.strip()
    if not query:
        return []
    filters = filters or SearchFilters()
    pool = max(k, candidates)

    if mode == "bm25":
        rows = _bm25_search(session, query, limit=pool, filters=filters)
        return _hits_from_rows(rows, mode=mode, k=k)

    if mode == "bm25_rerank":
        if reranker is None:
            raise ValueError("bm25_rerank requires a reranker")
        # Retrieve a wide BM25 pool, then let the reranker order the top N.
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

    if embedder is None:
        raise ValueError("vector and hybrid search require an embedder")
    query_vector = embedder.embed([query], input_type="search_query")[0]
    model = embedder.model or embedding_model

    if mode == "vector":
        rows = _vector_search(
            session,
            query_vector,
            model=model,
            limit=pool,
            filters=filters,
            ef_search=ef_search,
            exact=exact_vector,
        )
        return _hits_from_rows(rows, mode=mode, k=k)

    if mode != "hybrid":
        raise ValueError(f"unknown search mode: {mode}")

    bm25_rows = _bm25_search(session, query, limit=pool, filters=filters)
    vector_rows = _vector_search(
        session,
        query_vector,
        model=model,
        limit=pool,
        filters=filters,
        ef_search=ef_search,
        exact=exact_vector,
    )
    fused = reciprocal_rank_fusion(
        [[row["chunk_id"] for row in bm25_rows], [row["chunk_id"] for row in vector_rows]],
        k=RRF_K,
        limit=k,
    )
    by_id = {row["chunk_id"]: row for row in bm25_rows}
    for row in vector_rows:
        by_id.setdefault(row["chunk_id"], row)
    hits = []
    for rank, (chunk_id, score) in enumerate(fused, start=1):
        row = by_id[chunk_id]
        hits.append(
            SearchHit(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                ahpn_id=row["ahpn_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                score=score,
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
        # Breadcrumbs and other metadata are stored as JSON; match department names as text.
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


def _vector_search(
    session: Session,
    query_vector: list[float],
    *,
    model: str,
    limit: int,
    filters: SearchFilters,
    ef_search: int | None,
    exact: bool,
) -> list[dict[str, Any]]:
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        if ef_search is not None and not exact:
            # SET LOCAL does not accept bind parameters.
            session.execute(text(f"SET LOCAL hnsw.ef_search = {int(ef_search)}"))
        distance = ChunkEmbedding.embedding.cosine_distance(query_vector)
        score = (literal(1.0) - distance).label("score")
        stmt = (
            select(
                Chunk.id.label("chunk_id"),
                Chunk.document_id.label("document_id"),
                Document.ahpn_id.label("ahpn_id"),
                Chunk.chunk_index.label("chunk_index"),
                Chunk.text.label("text"),
                Chunk.page_start.label("page_start"),
                Chunk.page_end.label("page_end"),
                score,
            )
            .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
            .join(Document, Document.id == Chunk.document_id)
            .where(ChunkEmbedding.model == model)
            .order_by(distance, Chunk.id)
            .limit(limit)
        )
        if exact:
            session.execute(text("SET LOCAL enable_indexscan = off"))
            session.execute(text("SET LOCAL enable_bitmapscan = off"))
        stmt = _apply_filters(stmt, filters)
        try:
            return [dict(row._mapping) for row in session.execute(stmt)]
        finally:
            if exact and session.is_active:
                session.execute(text("SET LOCAL enable_indexscan = on"))
                session.execute(text("SET LOCAL enable_bitmapscan = on"))

    emb = aliased(ChunkEmbedding)
    stmt = (
        select(Chunk, Document, emb.embedding)
        .join(emb, emb.chunk_id == Chunk.id)
        .join(Document, Document.id == Chunk.document_id)
        .where(emb.model == model)
    )
    stmt = _apply_filters(stmt, filters)
    scored: list[dict[str, Any]] = []
    for chunk, document, embedding in session.execute(stmt):
        scored.append(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "ahpn_id": document.ahpn_id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "score": _cosine(query_vector, list(embedding)),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["chunk_id"]))
    return scored[:limit]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))
