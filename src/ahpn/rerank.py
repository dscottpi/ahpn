from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ahpn.config import DEFAULT_RERANK_MODEL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class Reranker(Protocol):
    """Score query/document pairs and return a new ranking."""

    @property
    def model(self) -> str: ...

    def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> list[RerankResult]:
        """Return documents ordered by relevance. index refers to the input list."""


class FakeReranker:
    """Deterministic token-overlap reranker for offline tests and smoke evals."""

    def __init__(self, model: str = "fake-rerank-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> list[RerankResult]:
        query_tokens = set(query.lower().split())
        scored: list[RerankResult] = []
        for index, document in enumerate(documents):
            doc_tokens = set(document.lower().split())
            overlap = len(query_tokens & doc_tokens)
            # Prefer denser overlap; tiny length prior keeps ties stable.
            score = float(overlap) + (1.0 / (1.0 + len(document)))
            scored.append(RerankResult(index=index, score=score))
        scored.sort(key=lambda item: (-item.score, item.index))
        if top_n is not None:
            scored = scored[:top_n]
        return scored


class BedrockReranker:
    """Cohere Rerank on Bedrock via invoke_model (api_version 2)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_RERANK_MODEL,
        region: str | None = None,
        client: Any | None = None,
        max_attempts: int = 4,
        max_doc_chars: int = 4000,
    ) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("Bedrock rerank requires boto3") from exc
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self._model = model
        self.max_attempts = max_attempts
        self.max_doc_chars = max_doc_chars

    @property
    def model(self) -> str:
        return self._model

    def rerank(self, query: str, documents: Sequence[str], *, top_n: int | None = None) -> list[RerankResult]:
        if not documents:
            return []
        truncated = [document[: self.max_doc_chars] for document in documents]
        body: dict[str, Any] = {
            "api_version": 2,
            "query": query,
            "documents": truncated,
        }
        if top_n is not None:
            body["top_n"] = int(top_n)

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self._client.invoke_model(
                    modelId=self._model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                payload = json.loads(response["body"].read())
                rows = payload.get("results") or []
                results = [
                    RerankResult(index=int(row["index"]), score=float(row.get("relevance_score", 0.0)))
                    for row in rows
                ]
                results.sort(key=lambda item: (-item.score, item.index))
                return results
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"rerank failed for model {self._model}") from last_error
