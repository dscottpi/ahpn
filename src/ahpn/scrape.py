from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ahpn.client import AhpnClient, AhpnError, RestrictedDocumentError
from ahpn.config import SITE_BASE_URL
from ahpn.storage import DocumentStore

logger = logging.getLogger(__name__)

_STAFF_FIELDS = ("admin_notes",)


@dataclass(frozen=True)
class PageImage:
    page_number: int
    source_url: str
    content_type: str
    byte_count: int
    key: str


@dataclass(frozen=True)
class ScrapeResult:
    document_id: int
    record_key: str
    record_locator: str
    pages: tuple[PageImage, ...]


def scrape_document(client: AhpnClient, store: DocumentStore, document_id: int) -> ScrapeResult:
    """Download one public document: metadata, breadcrumbs, IIIF manifest, and page images."""
    logger.info("fetching document %s", document_id)
    document = _public_document(client.get_json(f"/document/{document_id}/"))
    breadcrumbs = client.get_json(f"/document/{document_id}/breadcrumbs/")
    manifest = client.get_json(f"/document/{document_id}/iiif/")
    images = _manifest_images(manifest)
    if not images:
        raise AhpnError(f"document {document_id} manifest did not include any page images")

    prefix = str(document_id)
    _put_json(store, f"{prefix}/raw/document.json", document)
    _put_json(store, f"{prefix}/raw/breadcrumbs.json", breadcrumbs)
    _put_json(store, f"{prefix}/raw/manifest.json", manifest)

    pages: list[PageImage] = []
    for page_number, source_url, image_format in images:
        logger.info("fetching page %s", page_number)
        data, content_type = client.get_bytes(source_url)
        extension = _extension(content_type, image_format)
        key = f"{prefix}/pages/{page_number:04d}{extension}"
        store.put(key, data, content_type or "application/octet-stream")
        pages.append(
            PageImage(
                page_number=page_number,
                source_url=source_url,
                content_type=content_type,
                byte_count=len(data),
                key=key,
            )
        )

    record = {
        "document_id": document_id,
        "source_page": f"{SITE_BASE_URL}/document/{document_id}",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "visibility": document.get("visibility"),
        "page_count": document.get("page_count"),
        "document": document,
        "breadcrumbs": breadcrumbs,
        "pages": [
            {
                "page_number": page.page_number,
                "source_url": page.source_url,
                "content_type": page.content_type,
                "byte_count": page.byte_count,
                "key": page.key,
            }
            for page in pages
        ],
    }
    record_key = f"{prefix}/record.json"
    record_locator = _put_json(store, record_key, record)
    return ScrapeResult(
        document_id=document_id,
        record_key=record_key,
        record_locator=record_locator,
        pages=tuple(pages),
    )


def _public_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("visibility") != "PUB":
        document_id = document.get("id", "unknown")
        raise RestrictedDocumentError(
            f"document {document_id} is not public (visibility={document.get('visibility')!r})"
        )
    return {key: value for key, value in document.items() if key not in _STAFF_FIELDS}


def _manifest_images(manifest: dict[str, Any]) -> list[tuple[int, str, str | None]]:
    images: list[tuple[int, str, str | None]] = []
    for index, canvas in enumerate(manifest.get("items") or [], start=1):
        for annotation_page in canvas.get("items") or []:
            for annotation in annotation_page.get("items") or []:
                body = annotation.get("body") or {}
                url = body.get("id")
                if body.get("type") == "Image" and isinstance(url, str):
                    images.append((index, url, body.get("format")))
    return images


def _put_json(store: DocumentStore, key: str, payload: Any) -> str:
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    return store.put(key, data, "application/json")


def _extension(content_type: str, image_format: str | None) -> str:
    by_type = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tif",
        "image/jp2": ".jp2",
    }
    if content_type in by_type:
        return by_type[content_type]
    if image_format and "png" in image_format:
        return ".png"
    if image_format and "tif" in image_format:
        return ".tif"
    return ".jpg"
