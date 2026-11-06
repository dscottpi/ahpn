from __future__ import annotations

from collections import Counter
from typing import Sequence


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two aligned grade sequences."""
    if len(a) != len(b):
        raise ValueError("grade sequences must have the same length")
    if not a:
        return 0.0
    n = len(a)
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y)
    p_o = agree / n
    labels = sorted(set(a) | set(b))
    ca = Counter(a)
    cb = Counter(b)
    p_e = sum((ca[label] / n) * (cb[label] / n) for label in labels)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def meets_calibration_threshold(kappa: float, *, threshold: float = 0.6) -> bool:
    return kappa >= threshold
