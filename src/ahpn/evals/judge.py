from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

from ahpn.llm import LlmClient, parse_json_response

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You grade whether an archive page is relevant to a search query.
Return ONLY JSON: {"grade": <0-3>, "rationale": "<short>"}.
Grades:
0 = not relevant
1 = marginally related (same department/era, no answer)
2 = partially relevant (related facts, incomplete answer)
3 = highly relevant (page answers or directly documents the query)
The archive is Guatemalan National Police records (AHPN), mostly Spanish."""

JUDGE_PROMPT = """Query ({lang}): {query}

Page id: {page_key}
Page text:
\"\"\"
{page_text}
\"\"\"
"""


def grade_pair(
    llm: LlmClient,
    *,
    query: str,
    lang: str,
    page_key: str,
    page_text: str,
    cache_dir: Path | None = None,
) -> tuple[int, str]:
    cache_key = _cache_key(query, page_key, page_text)
    if cache_dir is not None:
        cached = _read_cache(cache_dir, cache_key)
        if cached is not None:
            return int(cached["grade"]), str(cached.get("rationale") or "")

    prompt = JUDGE_PROMPT.format(lang=lang, query=query, page_key=page_key, page_text=page_text[:6000])
    raw = llm.complete(prompt, system=JUDGE_SYSTEM)
    payload = parse_json_response(raw)
    grade = int(payload.get("grade", 0))
    if grade < 0 or grade > 3:
        raise ValueError(f"grade out of range: {grade}")
    rationale = str(payload.get("rationale") or "")
    if cache_dir is not None:
        _write_cache(cache_dir, cache_key, {"grade": grade, "rationale": rationale, "page_key": page_key})
    return grade, rationale


def pool_and_judge(
    llm: LlmClient,
    *,
    queries: Iterable[tuple[str, str, str]],
    pools: dict[str, list[tuple[str, str]]],
    cache_dir: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Judge pooled page candidates.

    queries: iterable of (qid, text, lang)
    pools: qid -> list of (page_key, page_text)
    """
    qrels: dict[str, dict[str, int]] = {}
    for qid, text, lang in queries:
        qrels[qid] = {}
        for page_key, page_text in pools.get(qid, []):
            grade, _rationale = grade_pair(
                llm,
                query=text,
                lang=lang,
                page_key=page_key,
                page_text=page_text,
                cache_dir=cache_dir,
            )
            if grade > 0:
                qrels[qid][page_key] = grade
    return qrels


def parse_grade_response(text: str) -> int:
    payload = parse_json_response(text)
    return int(payload["grade"])


def _cache_key(query: str, page_key: str, page_text: str) -> str:
    blob = f"{query}\n{page_key}\n{page_text}".encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_cache(cache_dir: Path, key: str) -> dict | None:
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(cache_dir: Path, key: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
