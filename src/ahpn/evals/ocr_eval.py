from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrGoldPage:
    ahpn_id: int
    page_number: int
    page_type: str  # typed | handwritten | form | degraded
    text: str


@dataclass
class OcrPageScore:
    ahpn_id: int
    page_number: int
    page_type: str
    cer: float
    wer: float
    bag_wer: float


def load_gold_pages(path: Path) -> list[OcrGoldPage]:
    """Load gold transcriptions from a JSONL file or a directory of .txt files.

    JSONL fields: ahpn_id, page_number, page_type, text
    Directory layout: <ahpn_id>_<page>_<type>.txt
    """
    if path.is_file():
        pages: list[OcrGoldPage] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pages.append(
                OcrGoldPage(
                    ahpn_id=int(row["ahpn_id"]),
                    page_number=int(row["page_number"]),
                    page_type=str(row.get("page_type") or "typed"),
                    text=str(row["text"]),
                )
            )
        return pages

    pages = []
    for file in sorted(path.glob("*.txt")):
        stem = file.stem
        parts = stem.split("_")
        if len(parts) < 3:
            raise ValueError(f"expected <ahpn_id>_<page>_<type>.txt, got {file.name}")
        pages.append(
            OcrGoldPage(
                ahpn_id=int(parts[0]),
                page_number=int(parts[1]),
                page_type="_".join(parts[2:]),
                text=file.read_text(encoding="utf-8"),
            )
        )
    return pages


def score_pages(
    gold_pages: Iterable[OcrGoldPage],
    hypothesis: dict[tuple[int, int], str],
) -> list[OcrPageScore]:
    scores: list[OcrPageScore] = []
    for gold in gold_pages:
        hypo = hypothesis.get((gold.ahpn_id, gold.page_number), "")
        scores.append(
            OcrPageScore(
                ahpn_id=gold.ahpn_id,
                page_number=gold.page_number,
                page_type=gold.page_type,
                cer=char_error_rate(gold.text, hypo),
                wer=word_error_rate(gold.text, hypo),
                bag_wer=bag_word_error_rate(gold.text, hypo),
            )
        )
    return scores


def summarize(scores: list[OcrPageScore]) -> dict:
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    by_type: dict[str, list[OcrPageScore]] = {}
    for score in scores:
        by_type.setdefault(score.page_type, []).append(score)
    return {
        "n_pages": len(scores),
        "cer": avg([s.cer for s in scores]),
        "wer": avg([s.wer for s in scores]),
        "bag_wer": avg([s.bag_wer for s in scores]),
        "by_type": {
            page_type: {
                "n_pages": len(group),
                "cer": avg([s.cer for s in group]),
                "wer": avg([s.wer for s in group]),
                "bag_wer": avg([s.bag_wer for s in group]),
            }
            for page_type, group in sorted(by_type.items())
        },
        "pages": [asdict(score) for score in scores],
    }


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        import jiwer

        return float(jiwer.cer(ref, hyp))
    except ImportError:
        return _edit_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        import jiwer

        return float(jiwer.wer(ref, hyp))
    except ImportError:
        ref_words = ref.split()
        hyp_words = hyp.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        return _edit_distance(ref_words, hyp_words) / len(ref_words)


def bag_word_error_rate(reference: str, hypothesis: str) -> float:
    """Order-insensitive word error: 1 - |intersection| / |reference words|."""
    ref_words = _normalize(reference).split()
    hyp_words = _normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    ref_bag = list(ref_words)
    hyp_bag = list(hyp_words)
    matched = 0
    for word in ref_bag:
        if word in hyp_bag:
            hyp_bag.remove(word)
            matched += 1
    return 1.0 - (matched / len(ref_bag))


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(a, b) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]
