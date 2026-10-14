from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ahpn.client import AhpnError
from ahpn.config import TEXTRACT_MAX_BYTES


@dataclass(frozen=True)
class TextLine:
    page_number: int
    line_index: int
    text: str
    confidence: float
    top: float
    left: float
    width: float
    height: float


class OcrEngine(Protocol):
    def detect_lines(self, page_number: int, image: bytes) -> list[TextLine]:
        """Read one page image into ordered text lines."""


class TextractOcr:
    """OCR a page image with Amazon Textract DetectDocumentText.

    Textract is the AWS service built for document scans. Rekognition's text
    detection is aimed at signs and scenes, and it does not return the same
    line structure. DetectDocumentText is synchronous and accepts one JPEG,
    PNG, TIFF, or single-page PDF up to 10 MB, which matches the page images
    this archive serves.
    """

    def __init__(self, client: Any | None = None, region: str | None = None) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise RuntimeError("Textract requires boto3") from exc
            client = boto3.client("textract", region_name=region)
        self._client = client

    def detect_lines(self, page_number: int, image: bytes) -> list[TextLine]:
        if len(image) > TEXTRACT_MAX_BYTES:
            raise AhpnError(
                f"page {page_number} is {len(image)} bytes; "
                "Textract's synchronous API accepts images up to 10 MB"
            )
        response = self._client.detect_document_text(Document={"Bytes": image})
        blocks = list(response.get("Blocks") or [])
        token = response.get("NextToken")
        while token:
            response = self._client.detect_document_text(Document={"Bytes": image}, NextToken=token)
            blocks.extend(response.get("Blocks") or [])
            token = response.get("NextToken")
        return lines_from_blocks(page_number, blocks)


def lines_from_blocks(page_number: int, blocks: list[dict[str, Any]]) -> list[TextLine]:
    lines: list[TextLine] = []
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        box = (block.get("Geometry") or {}).get("BoundingBox") or {}
        lines.append(
            TextLine(
                page_number=page_number,
                line_index=len(lines),
                text=str(block.get("Text") or ""),
                confidence=float(block.get("Confidence") or 0),
                top=float(box.get("Top") or 0),
                left=float(box.get("Left") or 0),
                width=float(box.get("Width") or 0),
                height=float(box.get("Height") or 0),
            )
        )
    return lines
