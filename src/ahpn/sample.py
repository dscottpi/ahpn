from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ahpn.client import AhpnClient

logger = logging.getLogger(__name__)

PAGE_BUCKETS = (
    ("1", lambda n: n == 1),
    ("2-5", lambda n: 2 <= n <= 5),
    ("6-20", lambda n: 6 <= n <= 20),
    ("21+", lambda n: n >= 21),
)


@dataclass(frozen=True)
class SampledDocument:
    ahpn_id: int
    document_type: str
    page_count: int
    decade: str
    page_bucket: str
    date: str | None
    case_id: int | None
    layer: str  # core | distractor_300 | distractor_1000


@dataclass(frozen=True)
class SamplePlan:
    version: str
    seed: int
    total: int
    core_size: int
    layers: dict[str, list[int]]
    documents: list[SampledDocument]

    def ids_for_layer(self, layer: str) -> list[int]:
        """Nested subset: 100 ⊂ 300 ⊂ 1000."""
        return list(self.layers[layer])


def page_bucket(page_count: int | None) -> str:
    count = int(page_count or 0)
    for name, predicate in PAGE_BUCKETS:
        if predicate(count):
            return name
    return "1"


def decade_of(date_str: str | None) -> str:
    if not date_str:
        return "unknown"
    year = date_str[:4]
    if not year.isdigit():
        return "unknown"
    return f"{year[:3]}0s"


def stratum_key(document_type: str, decade: str, bucket: str) -> str:
    return f"{document_type}|{decade}|{bucket}"


def stratified_sample(
    candidates: Iterable[dict[str, Any]],
    *,
    total: int = 1000,
    core_size: int = 300,
    seed: int = 42,
) -> SamplePlan:
    """Draw a stratified sample with nested distractor layers at 100 / 300 / 1000.

    Candidates are raw `/document/` API dicts. Quotas are spread evenly across
    observed (document_type, decade, page_bucket) strata, then leftovers fill
    from the largest remaining pools.
    """
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    for doc in candidates:
        if doc.get("visibility") and doc.get("visibility") != "PUB":
            continue
        doc_id = int(doc["id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        dtype = str(doc.get("document_type") or "UNK")
        decade = decade_of(doc.get("date"))
        bucket = page_bucket(doc.get("page_count"))
        by_stratum[stratum_key(dtype, decade, bucket)].append(doc)

    for docs in by_stratum.values():
        rng.shuffle(docs)

    strata = sorted(by_stratum)
    if not strata:
        raise ValueError("no public document candidates to sample from")

    quota = max(1, total // len(strata))
    picked: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for key in strata:
        pool = by_stratum[key]
        take = min(quota, len(pool))
        picked.extend(pool[:take])
        leftovers.extend(pool[take:])

    rng.shuffle(leftovers)
    if len(picked) < total:
        picked.extend(leftovers[: total - len(picked)])
    picked = picked[:total]
    rng.shuffle(picked)

    # Nested layers: first 100 = core_small, first 300 = core, rest distractors up to 1000.
    layer_for_index = []
    for index in range(len(picked)):
        if index < min(100, core_size, len(picked)):
            layer_for_index.append("core_100")
        elif index < min(core_size, len(picked)):
            layer_for_index.append("core")
        else:
            layer_for_index.append("distractor")

    documents: list[SampledDocument] = []
    for index, doc in enumerate(picked):
        layer = layer_for_index[index]
        documents.append(
            SampledDocument(
                ahpn_id=int(doc["id"]),
                document_type=str(doc.get("document_type") or "UNK"),
                page_count=int(doc.get("page_count") or 0),
                decade=decade_of(doc.get("date")),
                page_bucket=page_bucket(doc.get("page_count")),
                date=doc.get("date"),
                case_id=doc.get("case"),
                layer="core" if layer.startswith("core") else "distractor",
            )
        )

    ids = [doc.ahpn_id for doc in documents]
    layers = {
        "100": ids[: min(100, len(ids))],
        "300": ids[: min(300, len(ids))],
        "1000": ids[: min(1000, len(ids))],
    }
    return SamplePlan(
        version="v1",
        seed=seed,
        total=len(documents),
        core_size=min(core_size, len(documents)),
        layers=layers,
        documents=documents,
    )


def fetch_document_candidates(
    client: AhpnClient,
    *,
    pages: int = 40,
    page_size: int = 100,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Pull random pages from the public document listing to seed stratification.

    The listing does not honor filters, so diversity comes from sampling many
    random offsets across the ~7.6M public documents.
    """
    head = client.get_json(f"/document/?page=1&page_size={page_size}")
    total = int(head.get("count") or 0)
    total_pages = max(1, (total + page_size - 1) // page_size)
    rng = random.Random(seed)
    page_numbers = {1}
    while len(page_numbers) < min(pages, total_pages):
        page_numbers.add(rng.randint(1, total_pages))

    candidates: list[dict[str, Any]] = list(head.get("results") or [])
    for page in sorted(page_numbers):
        if page == 1:
            continue
        payload = client.get_json(f"/document/?page={page}&page_size={page_size}")
        candidates.extend(payload.get("results") or [])
        logger.info("listed page %s/%s (%s candidates)", page, total_pages, len(candidates))
    return candidates


def write_sample(plan: SamplePlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": plan.version,
        "seed": plan.seed,
        "total": plan.total,
        "core_size": plan.core_size,
        "layers": plan.layers,
        "documents": [asdict(doc) for doc in plan.documents],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ids_path = path.with_suffix(".txt")
    ids_path.write_text("\n".join(str(i) for i in plan.layers["1000"]) + "\n", encoding="utf-8")


def load_sample(path: Path) -> SamplePlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = [SampledDocument(**row) for row in payload["documents"]]
    return SamplePlan(
        version=payload["version"],
        seed=payload["seed"],
        total=payload["total"],
        core_size=payload["core_size"],
        layers={key: list(value) for key, value in payload["layers"].items()},
        documents=documents,
    )


def read_ids_file(path: Path) -> list[int]:
    ids: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(int(line.split()[0]))
    return ids
