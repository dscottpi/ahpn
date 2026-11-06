from __future__ import annotations

import json
import logging
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy.orm import Session

from ahpn.config import DEFAULT_RERANK_CANDIDATES
from ahpn.embed import Embedder
from ahpn.evals.metrics import (
    QueryRecord,
    evaluate_by_group,
    evaluate_run,
    filter_queries,
    load_qrels,
    load_queries,
    page_hits_from_chunk_hits,
)
from ahpn.evals.pages import assemble_page_text, corpus_page_stats, parse_page_docno
from ahpn.rerank import Reranker
from ahpn.search import SearchFilters, SearchMode, search

logger = logging.getLogger(__name__)


@dataclass
class EvalRunResult:
    path: Path
    metrics: dict[str, float]
    by_type: dict[str, dict[str, float]]
    by_lang: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    n_queries: int


@dataclass
class EvalConfig:
    mode: SearchMode = "bm25_rerank"
    split: str = "dev"
    k: int = 100
    candidates: int = 100
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES
    embedding_model: str = ""
    rerank_model: str = ""
    department: str | None = None
    exact_vector: bool = False
    ef_search: int | None = None
    # Default unit is the page image (one JPEG). Document-level is a collapsed view.
    level: str = "page"  # page | document
    relevance_unit: str = "page_image"


def git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return None


def run_eval(
    session: Session,
    *,
    queries_path: Path,
    qrels_path: Path,
    out_dir: Path,
    embedder: Embedder | None,
    config: EvalConfig,
    filters: SearchFilters | None = None,
    reranker: Reranker | None = None,
) -> EvalRunResult:
    queries = filter_queries(load_queries(queries_path), split=config.split)
    qrels = load_qrels(qrels_path)
    if config.level == "document":
        qrels = _to_document_qrels(qrels)

    run: dict[str, list[tuple[str, float]]] = {}
    latencies: list[float] = []
    for query in queries:
        if query.id not in qrels and "no_answer" not in query.types:
            # Still run, but metrics only cover judged queries.
            pass
        started = time.perf_counter()
        hits = search(
            session,
            query.text,
            mode=config.mode,
            k=config.k,
            candidates=config.candidates,
            filters=filters,
            embedder=embedder,
            embedding_model=config.embedding_model or (embedder.model if embedder else ""),
            reranker=reranker,
            rerank_candidates=config.rerank_candidates,
            ef_search=config.ef_search,
            exact_vector=config.exact_vector,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)
        if config.level == "document":
            ranked = [(str(hit.ahpn_id), hit.score) for hit in hits]
            # unique docs preserving order
            seen: set[str] = set()
            unique: list[tuple[str, float]] = []
            for docno, score in ranked:
                if docno in seen:
                    continue
                seen.add(docno)
                unique.append((docno, score))
            run[query.id] = unique
        else:
            page_texts: dict[str, str] = {}
            for hit in hits:
                for key in hit.page_keys():
                    if key in page_texts:
                        continue
                    ref = parse_page_docno(key)
                    page_texts[key] = assemble_page_text(session, ref.ahpn_id, ref.page_number)
            run[query.id] = page_hits_from_chunk_hits(
                hits,
                query=query.text,
                page_texts=page_texts,
            )

    metrics = evaluate_run(qrels, run)
    by_type = evaluate_by_group(queries, qrels, run, group="type")
    by_lang = evaluate_by_group(queries, qrels, run, group="lang")
    latency = {
        "p50": float(statistics.median(latencies)) if latencies else 0.0,
        "p95": float(_percentile(latencies, 95)) if latencies else 0.0,
        "mean": float(statistics.fmean(latencies)) if latencies else 0.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{config.mode}_{config.split}"
    run_path = out_dir / f"{stem}.run.tsv"
    json_path = out_dir / f"{stem}.json"
    _write_trec_run(run_path, run, tag=config.mode)
    page_stats = corpus_page_stats(session)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config": asdict(config),
        "relevance_unit": "page_image" if config.level == "page" else "document",
        "corpus": page_stats,
        "metrics": metrics,
        "by_type": by_type,
        "by_lang": by_lang,
        "latency_ms": latency,
        "n_queries": len(queries),
        "run_file": str(run_path),
        "notes": (
            "AHPN documents are JPEG page images (or ordered lists of JPEGs). "
            "Default metrics score page images via ahpn_id:page qrels."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", json_path)
    return EvalRunResult(
        path=json_path,
        metrics=metrics,
        by_type=by_type,
        by_lang=by_lang,
        latency_ms=latency,
        n_queries=len(queries),
    )


def load_run_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_trec_run(path: Path, run: dict[str, list[tuple[str, float]]], *, tag: str) -> None:
    lines: list[str] = []
    for qid, ranked in run.items():
        for rank, (docno, score) in enumerate(ranked, start=1):
            lines.append(f"{qid} Q0 {docno} {rank} {score:.6f} {tag}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _to_document_qrels(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    collapsed: dict[str, dict[str, int]] = {}
    for qid, graded in qrels.items():
        docs: dict[str, int] = {}
        for docno, grade in graded.items():
            doc_id = docno.split(":", 1)[0]
            docs[doc_id] = max(grade, docs.get(doc_id, 0))
        collapsed[qid] = docs
    return collapsed


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
