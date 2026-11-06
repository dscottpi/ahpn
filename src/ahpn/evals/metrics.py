from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ahpn.search import SearchHit


@dataclass
class QueryRecord:
    id: str
    text: str
    lang: str = "es"
    types: list[str] = field(default_factory=list)
    source: str = "hand"
    split: str = "dev"
    notes: str = ""


def load_queries(path: Path) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        records.append(
            QueryRecord(
                id=str(payload["id"]),
                text=payload["text"],
                lang=payload.get("lang", "es"),
                types=list(payload.get("types") or []),
                source=payload.get("source", "hand"),
                split=payload.get("split", "dev"),
                notes=payload.get("notes", ""),
            )
        )
    return records


def write_queries(path: Path, queries: Sequence[QueryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query in queries:
            handle.write(json.dumps(asdict(query), ensure_ascii=False) + "\n")


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load TREC qrels: qid 0 docno grade. docno is `ahpn_id:page`."""
    qrels: dict[str, dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad qrels line: {line}")
        qid, _iter, docno, grade = parts[0], parts[1], parts[2], int(parts[3])
        qrels.setdefault(qid, {})[docno] = grade
    return qrels


def write_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for qid in sorted(qrels):
        for docno, grade in sorted(qrels[qid].items()):
            lines.append(f"{qid} 0 {docno} {grade}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def page_hits_from_chunk_hits(hits: Sequence[SearchHit]) -> list[tuple[str, float]]:
    """Collapse chunk hits to unique page keys, keeping the best score per page."""
    best: dict[str, float] = {}
    order: list[str] = []
    for hit in hits:
        for key in hit.page_keys():
            if key not in best:
                order.append(key)
                best[key] = hit.score
            else:
                best[key] = max(best[key], hit.score)
    order.sort(key=lambda key: (-best[key], key))
    return [(key, best[key]) for key in order]


def document_hits_from_chunk_hits(hits: Sequence[SearchHit]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    order: list[str] = []
    for hit in hits:
        key = str(hit.ahpn_id)
        if key not in best:
            order.append(key)
            best[key] = hit.score
        else:
            best[key] = max(best[key], hit.score)
    order.sort(key=lambda key: (-best[key], key))
    return [(key, best[key]) for key in order]


def filter_queries(
    queries: Sequence[QueryRecord],
    *,
    split: str | None = None,
    types: Iterable[str] | None = None,
) -> list[QueryRecord]:
    selected = list(queries)
    if split and split != "all":
        selected = [query for query in selected if query.split == split]
    if types:
        wanted = set(types)
        selected = [query for query in selected if wanted.intersection(query.types)]
    return selected


def run_to_ranx(run: dict[str, list[tuple[str, float]]]) -> Any:
    from ranx import Run

    return Run(run)


def qrels_to_ranx(qrels: dict[str, dict[str, int]]) -> Any:
    from ranx import Qrels

    return Qrels(qrels)


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float]]],
    *,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Score a run with ranx. Returns mean metrics."""
    metrics = list(metrics or ("recall@10", "recall@50", "recall@100", "ndcg@10", "mrr@10"))
    try:
        from ranx import evaluate
    except ImportError as exc:
        raise RuntimeError("ranx is required for eval metrics; install the evals extra") from exc

    # Keep only queries present in both.
    shared = sorted(set(qrels) & set(run))
    if not shared:
        return {metric: 0.0 for metric in metrics}
    trimmed_qrels = {qid: qrels[qid] for qid in shared}
    trimmed_run = {qid: dict(run[qid]) for qid in shared}
    scores = evaluate(qrels_to_ranx(trimmed_qrels), run_to_ranx(trimmed_run), list(metrics))
    return {metric: float(scores[metric]) for metric in metrics}


def evaluate_by_group(
    queries: Sequence[QueryRecord],
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float]]],
    *,
    group: str,
) -> dict[str, dict[str, float]]:
    """group is 'lang' or 'type'."""
    buckets: dict[str, list[str]] = {}
    for query in queries:
        if query.id not in qrels or query.id not in run:
            continue
        keys = [query.lang] if group == "lang" else (query.types or ["untagged"])
        for key in keys:
            buckets.setdefault(key, []).append(query.id)
    return {
        key: evaluate_run(
            {qid: qrels[qid] for qid in qids},
            {qid: run[qid] for qid in qids},
        )
        for key, qids in sorted(buckets.items())
    }
