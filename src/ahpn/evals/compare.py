from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ahpn.evals.run import load_run_json


def compare_runs(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Compare two eval JSON reports. Includes paired significance when ranx can load run files."""
    a = load_run_json(path_a)
    b = load_run_json(path_b)
    metrics = sorted(set(a.get("metrics", {})) | set(b.get("metrics", {})))
    delta = {
        metric: float(b.get("metrics", {}).get(metric, 0.0)) - float(a.get("metrics", {}).get(metric, 0.0))
        for metric in metrics
    }
    significance: dict[str, Any] = {}
    run_a = a.get("run_file")
    run_b = b.get("run_file")
    # Significance needs qrels; callers can pass them later. Here we report metric deltas always.
    return {
        "a": str(path_a),
        "b": str(path_b),
        "metrics_a": a.get("metrics", {}),
        "metrics_b": b.get("metrics", {}),
        "delta_b_minus_a": delta,
        "latency_a": a.get("latency_ms", {}),
        "latency_b": b.get("latency_ms", {}),
        "significance": significance,
        "run_a": run_a,
        "run_b": run_b,
    }


def compare_with_qrels(
    path_a: Path,
    path_b: Path,
    qrels_path: Path,
    *,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Paired statistical comparison via ranx.compare when both TREC runs exist."""
    base = compare_runs(path_a, path_b)
    metrics = metrics or ["ndcg@10", "recall@50", "mrr@10"]
    try:
        from ranx import Qrels, Run, compare
    except ImportError:
        base["significance"] = {"error": "ranx not installed"}
        return base

    a = load_run_json(path_a)
    b = load_run_json(path_b)
    run_a_path = Path(a["run_file"])
    run_b_path = Path(b["run_file"])
    qrels = Qrels.from_file(str(qrels_path))
    run_a = Run.from_file(str(run_a_path))
    run_b = Run.from_file(str(run_b_path))
    report = compare(
        qrels,
        [run_a, run_b],
        metrics,
        max_p=0.01,
    )
    base["significance"] = {"report": str(report)}
    return base


def write_compare(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
