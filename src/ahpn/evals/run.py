from __future__ import annotations

import json
import logging
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from sqlalchemy.orm import Session

from ahpn.config import DEFAULT_RERANK_CANDIDATES
from ahpn.evals.metrics import (
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
    latency_ms: dict[str, float]
    n_queries: int


@dataclass
class EvalConfig:
    mode: SearchMode = "bm25_rerank"
    split: str = "dev"
    k: int = 10
    candidates: int = 100
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES
    rerank_model: str = ""
    department: str | None = None
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
    config: EvalConfig,
    filters: SearchFilters | None = None,
    reranker: Reranker | None = None,
) -> EvalRunResult:
    queries = filter_queries(load_queries(queries_path), split=config.split)
    qrels = load_qrels(qrels_path)

    run: dict[str, list[tuple[str, float]]] = {}
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        hits = search(
            session,
            query.text,
            mode=config.mode,
            k=config.k,
            candidates=config.candidates,
            filters=filters,
            reranker=reranker,
            rerank_candidates=config.rerank_candidates,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)

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
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config": asdict(config),
        "relevance_unit": "page_image",
        "corpus": corpus_page_stats(session),
        "metrics": metrics,
        "latency_ms": latency,
        "n_queries": len(queries),
        "run_file": str(run_path),
        "notes": (
            "AHPN documents are JPEG page images (or ordered lists of JPEGs). "
            "Metrics score page images via ahpn_id:page qrels."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", json_path)
    return EvalRunResult(
        path=json_path,
        metrics=metrics,
        latency_ms=latency,
        n_queries=len(queries),
    )


def _write_trec_run(path: Path, run: dict[str, list[tuple[str, float]]], *, tag: str) -> None:
    lines: list[str] = []
    for qid, ranked in run.items():
        for rank, (docno, score) in enumerate(ranked, start=1):
            lines.append(f"{qid} Q0 {docno} {rank} {score:.6f} {tag}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
