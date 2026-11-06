from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ahpn.config import DEFAULT_EMBEDDING_DIMS, DEFAULT_EMBEDDING_MODEL
from ahpn.db import ensure_embedding_index
from ahpn.models import Chunk, ChunkEmbedding

logger = logging.getLogger(__name__)

InputType = Literal["search_document", "search_query"]


class Embedder(Protocol):
    """Produce dense vectors for documents or queries."""

    @property
    def model(self) -> str: ...

    @property
    def dims(self) -> int: ...

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        """Embed one or more texts. input_type distinguishes indexed docs from queries."""


@dataclass(frozen=True)
class EmbedResult:
    model: str
    chunks_embedded: int
    chunks_skipped: int


class FakeEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for tests and offline smoke evals."""

    def __init__(self, model: str = "fake-embed-v1", dims: int = DEFAULT_EMBEDDING_DIMS) -> None:
        self._model = model
        self._dims = dims

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        del input_type
        return [_fake_vector(text, self._dims) for text in texts]


class BedrockEmbedder:
    """Embed via Amazon Bedrock (Cohere Embed v4 or Titan Text Embeddings V2)."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dims: int = DEFAULT_EMBEDDING_DIMS,
        region: str | None = None,
        client: Any | None = None,
        batch_size: int = 32,
        max_attempts: int = 4,
    ) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("Bedrock embeddings require boto3") from exc
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self._model = model
        self._dims = dims
        self.batch_size = batch_size
        self.max_attempts = max_attempts

    @property
    def model(self) -> str:
        return self._model

    @property
    def dims(self) -> int:
        return self._dims

    def embed(self, texts: Sequence[str], *, input_type: InputType) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(self._embed_batch(batch, input_type=input_type))
        return vectors

    def _embed_batch(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                if self._model.startswith("cohere.") or "cohere" in self._model:
                    return self._cohere(texts, input_type=input_type)
                return self._titan(texts)
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"embedding failed for model {self._model}") from last_error

    def _cohere(self, texts: list[str], *, input_type: InputType) -> list[list[float]]:
        body = {
            "texts": texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        # Cohere v3/v4 on Bedrock accept an optional truncate / output dimension.
        if self._dims:
            body["output_dimension"] = self._dims
        response = self._client.invoke_model(
            modelId=self._model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        embeddings = payload.get("embeddings") or {}
        floats = embeddings.get("float") if isinstance(embeddings, dict) else embeddings
        if not floats:
            raise RuntimeError(f"Cohere response missing embeddings: {payload.keys()}")
        return [list(map(float, row)) for row in floats]

    def _titan(self, texts: list[str]) -> list[list[float]]:
        # Titan v2 accepts one inputText per call.
        vectors: list[list[float]] = []
        for text in texts:
            body = {"inputText": text, "dimensions": self._dims, "normalize": True}
            response = self._client.invoke_model(
                modelId=self._model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            payload = json.loads(response["body"].read())
            vector = payload.get("embedding")
            if not vector:
                raise RuntimeError(f"Titan response missing embedding: {payload.keys()}")
            vectors.append([float(x) for x in vector])
        return vectors


def embed_chunks(
    session: Session,
    embedder: Embedder,
    *,
    document_ids: Sequence[int] | None = None,
    force: bool = False,
    batch_size: int = 64,
) -> EmbedResult:
    """Embed chunks that do not yet have a vector for embedder.model."""
    query = select(Chunk)
    if document_ids is not None:
        query = query.where(Chunk.document_id.in_(list(document_ids)))
    query = query.order_by(Chunk.id)
    chunks = list(session.scalars(query))
    if not chunks:
        return EmbedResult(model=embedder.model, chunks_embedded=0, chunks_skipped=0)

    existing: set[int] = set()
    if not force:
        existing = set(
            session.scalars(
                select(ChunkEmbedding.chunk_id).where(
                    ChunkEmbedding.model == embedder.model,
                    ChunkEmbedding.chunk_id.in_([c.id for c in chunks]),
                )
            )
        )

    pending = [c for c in chunks if c.id not in existing]
    embedded = 0
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.embed([c.text for c in batch], input_type="search_document")
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(batch)} texts")
        for chunk, vector in zip(batch, vectors, strict=True):
            _upsert_embedding(session, chunk_id=chunk.id, model=embedder.model, dims=embedder.dims, vector=vector)
            embedded += 1
        session.commit()
        logger.info("embedded %s/%s chunks (%s)", embedded, len(pending), embedder.model)

    if session.bind is not None and session.bind.dialect.name == "postgresql":
        with session.bind.begin() as conn:
            ensure_embedding_index(conn, embedder.model)

    return EmbedResult(
        model=embedder.model,
        chunks_embedded=embedded,
        chunks_skipped=len(chunks) - embedded,
    )


def _upsert_embedding(
    session: Session,
    *,
    chunk_id: int,
    model: str,
    dims: int,
    vector: list[float],
) -> None:
    if session.bind is not None and session.bind.dialect.name == "postgresql":
        stmt = (
            pg_insert(ChunkEmbedding)
            .values(chunk_id=chunk_id, model=model, dims=dims, embedding=vector)
            .on_conflict_do_update(
                index_elements=["chunk_id", "model"],
                set_={"dims": dims, "embedding": vector},
            )
        )
        session.execute(stmt)
        return
    row = session.scalar(
        select(ChunkEmbedding).where(ChunkEmbedding.chunk_id == chunk_id, ChunkEmbedding.model == model)
    )
    if row is None:
        session.add(ChunkEmbedding(chunk_id=chunk_id, model=model, dims=dims, embedding=vector))
    else:
        row.dims = dims
        row.embedding = vector


def _fake_vector(text: str, dims: int) -> list[float]:
    values = [0.0] * dims
    tokens = text.lower().split()
    if not tokens:
        values[0] = 1.0
        return values
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]
