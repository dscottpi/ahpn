from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

from ahpn.evals.metrics import QueryRecord
from ahpn.llm import LlmClient, parse_json_response

logger = logging.getLogger(__name__)

QUERY_GEN_SYSTEM = """You write search queries for a Guatemalan National Police (AHPN) archive.
Given an OCR page excerpt, invent realistic researcher queries that the page would answer.
Rules:
- Paraphrase; do NOT copy long phrases from the excerpt.
- Prefer proper names, places, dates, case numbers, and concrete events.
- Return JSON: {"queries":[{"text":"...","lang":"es"|"en","types":["name"|"place"|"number_date"|"conceptual"|"cross_lingual"|"ocr_noise"]}]}
- Include at least one Spanish and one English query when possible.
- 2 to 4 queries total."""


def generate_queries_from_chunk(
    llm: LlmClient,
    *,
    chunk_text: str,
    ahpn_id: int,
    page_start: int,
    start_id: int,
) -> list[QueryRecord]:
    prompt = f"AHPN document {ahpn_id}, pages {page_start}+\n\nExcerpt:\n{chunk_text[:4000]}"
    raw = llm.complete(prompt, system=QUERY_GEN_SYSTEM)
    payload = parse_json_response(raw)
    rows = payload.get("queries") if isinstance(payload, dict) else payload
    queries: list[QueryRecord] = []
    for offset, row in enumerate(rows or []):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        queries.append(
            QueryRecord(
                id=f"gen-{start_id + offset}",
                text=text,
                lang=str(row.get("lang") or "es"),
                types=list(row.get("types") or ["conceptual"]),
                source="llm",
                split="dev",
                notes=f"seed={ahpn_id}:{page_start}",
            )
        )
    return queries


def metadata_queries(
    *,
    ahpn_id: int,
    breadcrumbs: Sequence[dict],
    date: str | None,
    start_id: int,
    split: str = "dev",
) -> list[QueryRecord]:
    """Cheap queries whose answers are known from folder/case/date metadata."""
    queries: list[QueryRecord] = []
    places = [str(item.get("name") or "") for item in breadcrumbs if item.get("type") in {"folder", "case"}]
    places = [name for name in places if name]
    if places:
        focus = places[-1] if len(places) == 1 else places[1] if len(places) > 1 else places[0]
        queries.append(
            QueryRecord(
                id=f"meta-{start_id}",
                text=focus,
                lang="es",
                types=["place"],
                source="metadata",
                split=split,
                notes=f"ahpn_id={ahpn_id}",
            )
        )
        start_id += 1
    if date:
        year = date[:4]
        place = places[1] if len(places) > 1 else (places[0] if places else "")
        text = f"{place} {year}".strip() if place else year
        queries.append(
            QueryRecord(
                id=f"meta-{start_id}",
                text=text,
                lang="es",
                types=["number_date", "place"],
                source="metadata",
                split=split,
                notes=f"ahpn_id={ahpn_id}",
            )
        )
    return queries


def known_item_queries(
    *,
    ahpn_id: int,
    page_number: int,
    lines: Sequence[str],
    start_id: int,
    split: str = "dev",
) -> list[QueryRecord]:
    """Use a real OCR phrase (including likely OCR noise) as a known-item query."""
    phrase = _best_phrase(lines)
    if not phrase:
        return []
    return [
        QueryRecord(
            id=f"known-{start_id}",
            text=phrase,
            lang="es",
            types=["ocr_noise", "known_item"],
            source="known_item",
            split=split,
            notes=f"ahpn_id={ahpn_id}:{page_number}",
        )
    ]


def no_answer_queries(start_id: int, *, split: str = "dev") -> list[QueryRecord]:
    seeds = [
        ("informes de tráfico espacial en Marte 2099", "es"),
        ("purchase orders for Antarctic penguin census 1920", "en"),
        ("nómina de astronautas de Quetzaltenango 1450", "es"),
    ]
    return [
        QueryRecord(
            id=f"none-{start_id + offset}",
            text=text,
            lang=lang,
            types=["no_answer"],
            source="no_answer",
            split=split,
        )
        for offset, (text, lang) in enumerate(seeds)
    ]


def assign_splits(queries: list[QueryRecord], *, seed: int = 42) -> list[QueryRecord]:
    """Deterministic 50/50 dev/test split by sorted id."""
    ordered = sorted(queries, key=lambda query: query.id)
    mid = len(ordered) // 2
    result: list[QueryRecord] = []
    for index, query in enumerate(ordered):
        split = "dev" if index < mid else "test"
        result.append(
            QueryRecord(
                id=query.id,
                text=query.text,
                lang=query.lang,
                types=list(query.types),
                source=query.source,
                split=split,
                notes=query.notes,
            )
        )
    return result


def _best_phrase(lines: Iterable[str]) -> str | None:
    candidates = []
    for line in lines:
        cleaned = re.sub(r"\s+", " ", line).strip()
        if 12 <= len(cleaned) <= 80:
            candidates.append(cleaned)
    if not candidates:
        return None
    return candidates[len(candidates) // 2]
