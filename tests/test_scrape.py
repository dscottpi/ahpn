import json
from pathlib import Path

import httpx
import pytest

from ahpn.client import AhpnClient, ApiError, RestrictedDocumentError
from ahpn.scrape import scrape_document
from ahpn.storage import LocalDocumentStore, S3DocumentStore

DOCUMENT = {
    "id": 2127283,
    "visibility": "PUB",
    "page_count": 1,
    "name": "",
    "document_type": "DOC",
    "admin_notes": "staff only",
    "fedora_uuid": "AHPN/Documents/example",
}
BREADCRUMBS = [{"id": 52809, "name": "Expediente", "type": "case"}]
MANIFEST = {
    "type": "Manifest",
    "items": [
        {
            "id": "https://example.test/canvas/1",
            "type": "Canvas",
            "items": [
                {
                    "type": "AnnotationPage",
                    "items": [
                        {
                            "type": "Annotation",
                            "body": {
                                "id": "https://iiif.example.test/page1/full/max/0/default.jpg",
                                "type": "Image",
                                "format": "image/jpg",
                            },
                        }
                    ],
                }
            ],
        }
    ],
}
JPEG = b"\xff\xd8\xff\xd9"


def _client(handler: httpx.MockTransport) -> AhpnClient:
    return AhpnClient(
        api_base_url="https://api.example.test",
        delay_seconds=0,
        transport=handler,
    )


def _routes(document: dict | None = None, *, fail_times: int = 0) -> httpx.MockTransport:
    calls = {"document": 0}
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        path = request.url.path
        if path == "/document/2127283/":
            calls["document"] += 1
            if calls["document"] <= fail_times:
                return httpx.Response(503, json={"detail": "busy"})
            return httpx.Response(200, json=document if document is not None else DOCUMENT)
        if path == "/document/2127283/breadcrumbs/":
            return httpx.Response(200, json=BREADCRUMBS)
        if path == "/document/2127283/iiif/":
            return httpx.Response(200, json=MANIFEST)
        if request.url.host == "iiif.example.test":
            return httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})
        return httpx.Response(404, json={"detail": "missing"})

    transport = httpx.MockTransport(handler)
    transport.requested = requested  # type: ignore[attr-defined]
    return transport


def test_scrape_writes_metadata_and_page_image(tmp_path: Path) -> None:
    transport = _routes()
    store = LocalDocumentStore(tmp_path)
    with _client(transport) as client:
        result = scrape_document(client, store, 2127283)

    record = json.loads((tmp_path / "2127283" / "record.json").read_text())
    raw_document = json.loads((tmp_path / "2127283" / "raw" / "document.json").read_text())
    assert result.pages[0].byte_count == len(JPEG)
    assert (tmp_path / "2127283" / "pages" / "0001.jpg").read_bytes() == JPEG
    assert record["document_id"] == 2127283
    assert record["source_page"] == "https://ahpn.lib.utexas.edu/document/2127283"
    assert record["pages"][0]["key"] == "2127283/pages/0001.jpg"
    assert "admin_notes" not in raw_document
    assert raw_document["fedora_uuid"] == "AHPN/Documents/example"
    assert json.loads((tmp_path / "2127283" / "raw" / "breadcrumbs.json").read_text()) == BREADCRUMBS


def test_restricted_document_is_not_downloaded(tmp_path: Path) -> None:
    transport = _routes({"id": 2127283, "visibility": "RST", "page_count": 1})
    with _client(transport) as client:
        with pytest.raises(RestrictedDocumentError):
            scrape_document(client, LocalDocumentStore(tmp_path), 2127283)

    requested = transport.requested  # type: ignore[attr-defined]
    assert requested == ["https://api.example.test/document/2127283/"]
    assert not (tmp_path / "2127283").exists()


def test_retries_transient_api_errors(tmp_path: Path) -> None:
    transport = _routes(fail_times=2)
    with _client(transport) as client:
        result = scrape_document(client, LocalDocumentStore(tmp_path), 2127283)
    assert result.document_id == 2127283


def test_api_error_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "busy"})

    with _client(httpx.MockTransport(handler)) as client:
        with pytest.raises(ApiError):
            client.get_json("/document/2127283/")


def test_s3_store_requires_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_boto3(name: str, *args: object, **kwargs: object) -> object:
        if name == "boto3":
            raise ImportError("boto3 is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_boto3)
    with pytest.raises(RuntimeError, match="s3 extra"):
        S3DocumentStore(bucket="archive")
