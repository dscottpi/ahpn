from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LlmClient(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the model text for a single-turn prompt."""


class FakeLlm:
    """Deterministic stand-in for tests. Optional canned responses by substring."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = "{}") -> None:
        self.responses = responses or {}
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        del system
        self.calls.append(prompt)
        for needle, response in self.responses.items():
            if needle in prompt:
                return response
        return self.default


class BedrockLlm:
    """Anthropic messages API via Bedrock Runtime."""

    def __init__(
        self,
        *,
        model: str,
        region: str | None = None,
        client: Any | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("Bedrock LLM requires boto3") from exc
            client = boto3.client("bedrock-runtime", region_name=region)
        self._client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        response = self._client.invoke_model(
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(response["body"].read())
        parts = payload.get("content") or []
        texts = [part.get("text", "") for part in parts if part.get("type") == "text"]
        return "\n".join(texts).strip()


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_json_response(text: str) -> Any:
    """Extract a JSON value from a model response, tolerating fenced markdown."""
    text = text.strip()
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
