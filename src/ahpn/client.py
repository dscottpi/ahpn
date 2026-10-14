from __future__ import annotations

import time
from typing import Any

import httpx

from ahpn.config import API_BASE_URL, DEFAULT_REQUEST_DELAY_SECONDS, USER_AGENT

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class AhpnError(Exception):
    """Base error for archive requests and scrapes."""


class ApiError(AhpnError):
    """The archive API or image service returned an error."""


class RestrictedDocumentError(AhpnError):
    """The document is not marked public, so its pages are left untouched."""


class AhpnClient:
    """HTTP client for the public AHPN API and IIIF image service."""

    def __init__(
        self,
        *,
        api_base_url: str = API_BASE_URL,
        user_agent: str = USER_AGENT,
        delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        max_attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.delay_seconds = delay_seconds
        self.max_attempts = max_attempts
        self._last_request_at = 0.0
        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json, image/jpeg;q=0.9, */*;q=0.8",
            },
            timeout=httpx.Timeout(30.0, read=120.0),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> AhpnClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_json(self, path: str) -> Any:
        url = path if path.startswith("http") else f"{self.api_base_url}/{path.lstrip('/')}"
        response = self._get(url)
        return response.json()

    def get_bytes(self, url: str) -> tuple[bytes, str]:
        response = self._get(url)
        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type.split(";")[0].strip().lower()

    def _get(self, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._pace()
            try:
                response = self._client.get(url)
            except httpx.TransportError as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                self._backoff(attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < self.max_attempts:
                self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise ApiError(f"HTTP {response.status_code} for {url}")
            return response
        raise ApiError(f"failed to fetch {url}") from last_error

    def _pace(self) -> None:
        if self.delay_seconds <= 0:
            self._last_request_at = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if self._last_request_at and remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _backoff(self, attempt: int) -> None:
        time.sleep(self.delay_seconds * attempt)
