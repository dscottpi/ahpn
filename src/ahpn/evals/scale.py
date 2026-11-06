from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sqlalchemy.orm import Session

from ahpn.embed import Embedder
from ahpn.evals.metrics import (
    evaluate_run,
    filter_queries,
    load_qrels,
    load_queries,
    page_hits_from_chunk_hits,
)
from ahpn.ingest import project_costs
from ahpn.search import SearchFilters, search

logger = logging.getLogger(__name__)


@dataclass
class AnnProbeResult:
    ef_search: int
    recall_vs_exact: float
    p95_latency_ms: float
    n_queries: int


@dataclass
class DistractorPoint:
    corpus_size: int
    metrics: dict[str, float]


def ann_vs_exact(
    session: Session,
    *,
    queries_path: Path,
    embedder: Embedder,
    ahpn_ids: Sequence[int] | None = None,
    ef_values: Sequence[int] = (20, 40, 80, 120),
    k: int = 10,
    split: str = "dev",
) -> list[AnnProbeResult]:
    """Measure HNSW recall against exact cosine search and latency per ef_search."""
    queries = filter_queries(load_queries(queries_path), split=split)
    filters = SearchFilters(ahpn_ids=frozenset(ahpn_ids)) if ahpn_ids else None
    results: list[AnnProbeResult] = []
    for ef in ef_values:
        overlaps: list[float] = []
        latencies: list[float] = []
        for query in queries:
            exact = search(
                session,
                query.text,
                mode="vector",
                k=k,
                embedder=embedder,
                filters=filters,
                exact_vector=True,
            )
            started = time.perf_counter()
            approx = search(
                session,
                query.text,
                mode="vector",
                k=k,
                embedder=embedder,
                filters=filters,
                ef_search=ef,
                exact_vector=False,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            exact_ids = {hit.chunk_id for hit in exact}
            approx_ids = {hit.chunk_id for hit in approx}
            overlaps.append(len(exact_ids & approx_ids) / max(1, len(exact_ids)))
        results.append(
            AnnProbeResult(
                ef_search=ef,
                recall_vs_exact=sum(overlaps) / len(overlaps) if overlaps else 0.0,
                p95_latency_ms=_percentile(latencies, 95),
                n_queries=len(queries),
            )
        )
        logger.info(
            "ef_search=%s recall_vs_exact=%.3f p95=%.1fms",
            ef,
            results[-1].recall_vs_exact,
            results[-1].p95_latency_ms,
        )
    return results


def distractor_curve(
    session: Session,
    *,
    queries_path: Path,
    qrels_path: Path,
    embedder: Embedder | None,
    layers: dict[str, list[int]],
    mode: str = "hybrid",
    split: str = "dev",
    k: int = 100,
) -> list[DistractorPoint]:
    """Recall/nDCG at nested corpus sizes (100 / 300 / 1000)."""
    queries = filter_queries(load_queries(queries_path), split=split)
    qrels = load_qrels(qrels_path)
    points: list[DistractorPoint] = []
    for label in sorted(layers, key=lambda item: int(item)):
        ids = layers[label]
        filters = SearchFilters(ahpn_ids=frozenset(ids))
        run: dict[str, list[tuple[str, float]]] = {}
        for query in queries:
            hits = search(
                session,
                query.text,
                mode=mode,  # type: ignore[arg-type]
                k=k,
                embedder=embedder,
                filters=filters,
            )
            run[query.id] = page_hits_from_chunk_hits(hits)
        metrics = evaluate_run(qrels, run)
        points.append(DistractorPoint(corpus_size=len(ids), metrics=metrics))
        logger.info("corpus=%s metrics=%s", label, metrics)
    return points


def extrapolate_log(points: Sequence[DistractorPoint], *, metric: str, target_size: int) -> float | None:
    """Log-linear extrapolation of a metric against corpus size."""
    usable = [(math.log(p.corpus_size), p.metrics.get(metric)) for p in points if p.corpus_size > 1]
    usable = [(x, y) for x, y in usable if y is not None]
    if len(usable) < 2:
        return None
    xs = [x for x, _ in usable]
    ys = [float(y) for _, y in usable]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return ys[-1]
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denom
    intercept = y_mean - slope * x_mean
    return intercept + slope * math.log(target_size)


def filtered_ann_probe(
    session: Session,
    *,
    query: str,
    embedder: Embedder,
    department: str,
    k: int = 10,
) -> dict[str, list[int]]:
    """Compare filtered vector search with and without iterative scan when available."""
    filters = SearchFilters(department=department)
    base = [
        hit.chunk_id
        for hit in search(session, query, mode="vector", k=k, embedder=embedder, filters=filters)
    ]
    # Best-effort: set iterative_scan if the server supports it.
    try:
        session.execute(__import__("sqlalchemy").text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        iterative = [
            hit.chunk_id
            for hit in search(session, query, mode="vector", k=k, embedder=embedder, filters=filters)
        ]
    except Exception:  # noqa: BLE001
        iterative = list(base)
    return {"default": base, "iterative_scan": iterative}


def build_cost_projection(
    *,
    sample_docs: int,
    sample_pages: int,
    sample_chunks: int,
    index_bytes: int | None = None,
    build_seconds: float | None = None,
    target_docs: int = 20_000_000,
) -> dict[str, float]:
    projection = project_costs(
        sample_docs=sample_docs,
        sample_pages=sample_pages,
        sample_chunks=sample_chunks,
        target_docs=target_docs,
    )
    if index_bytes is not None and sample_chunks > 0:
        projection["bytes_per_chunk"] = index_bytes / sample_chunks
        projection["projected_index_bytes"] = projection["bytes_per_chunk"] * projection["projected_chunks"]
    if build_seconds is not None and sample_docs > 0:
        projection["seconds_per_doc"] = build_seconds / sample_docs
        projection["projected_build_seconds"] = projection["seconds_per_doc"] * target_docs
    return projection


def write_scale_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def serialize_ann(results: Sequence[AnnProbeResult]) -> list[dict]:
    return [asdict(item) for item in results]


def serialize_distractor(points: Sequence[DistractorPoint]) -> list[dict]:
    return [asdict(item) for item in points]


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)
