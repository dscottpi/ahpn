from __future__ import annotations

from pathlib import Path
from typing import Protocol


class DocumentStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> str:
        """Store bytes at a document-relative key and return a locator."""

    def get(self, key: str) -> bytes:
        """Read bytes previously stored at a document-relative key."""


class LocalDocumentStore:
    """Write scraped documents under a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def put(self, key: str, data: bytes, content_type: str) -> str:
        del content_type
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def get(self, key: str) -> bytes:
        path = self.root / key
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()


class S3DocumentStore:
    """Write the same keys to S3 for an AWS deployment.

    Install the ``s3`` extra (``uv sync --extra s3``). Credentials come from
    the standard AWS chain: environment variables, shared config, or an
    instance role.
    """

    def __init__(self, bucket: str, prefix: str = "documents", region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3 storage requires the s3 extra: uv sync --extra s3") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = boto3.client("s3", region_name=region)

    def put(self, key: str, data: bytes, content_type: str) -> str:
        object_key = f"{self.prefix}/{key}" if self.prefix else key
        self._client.put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        return f"s3://{self.bucket}/{object_key}"

    def get(self, key: str) -> bytes:
        object_key = f"{self.prefix}/{key}" if self.prefix else key
        response = self._client.get_object(Bucket=self.bucket, Key=object_key)
        return response["Body"].read()
