from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ahpn.chunk import chunk_stored_document
from ahpn.client import AhpnClient, AhpnError
from ahpn.config import (
    API_BASE_URL,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_REQUEST_DELAY_SECONDS,
    EXAMPLE_DOCUMENT_ID,
)
from ahpn.db import database_url_from_env, init_db, session_factory
from ahpn.models import Chunk, Document, Page
from ahpn.ocr import TextractOcr
from ahpn.scrape import scrape_document
from ahpn.storage import DocumentStore, LocalDocumentStore, S3DocumentStore

logger = logging.getLogger("ahpn")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ahpn",
        description="Download public AHPN documents, read them with Textract, and store chunks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape = subparsers.add_parser("scrape", help="Download one public document, including its page images.")
    _add_document_id(scrape)
    _add_storage_args(scrape)
    scrape.add_argument("--api-base-url", default=os.environ.get("AHPN_API_BASE_URL", API_BASE_URL))
    scrape.add_argument(
        "--delay",
        type=float,
        default=float(os.environ.get("AHPN_REQUEST_DELAY", DEFAULT_REQUEST_DELAY_SECONDS)),
        help="Seconds to wait between archive requests",
    )

    db_init = subparsers.add_parser("db-init", help="Create the documents, pages, OCR lines, and chunks tables.")
    _add_database_arg(db_init)

    chunk = subparsers.add_parser("chunk", help="OCR a scraped document and store overlapping text chunks.")
    _add_document_id(chunk)
    _add_storage_args(chunk)
    _add_database_arg(chunk)
    chunk.add_argument("--target-chars", type=int, default=int(os.environ.get("AHPN_CHUNK_CHARS", DEFAULT_CHUNK_CHARS)))
    chunk.add_argument(
        "--overlap-chars",
        type=int,
        default=int(os.environ.get("AHPN_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
    )
    chunk.add_argument(
        "--force",
        action="store_true",
        help="Call Textract again even when OCR lines are already stored",
    )

    show = subparsers.add_parser("show", help="Print a stored document's pages and chunks.")
    _add_document_id(show)
    _add_database_arg(show)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    try:
        if args.command == "scrape":
            _scrape(args)
        elif args.command == "db-init":
            init_db(args.database_url)
            logger.info("database ready")
        elif args.command == "chunk":
            _chunk(args)
        elif args.command == "show":
            _show(args)
    except AhpnError as exc:
        logger.error("%s", exc)
        return 1
    return 0


def _scrape(args: argparse.Namespace) -> None:
    store = _store_from_args(args)
    with AhpnClient(api_base_url=args.api_base_url, delay_seconds=args.delay) as client:
        result = scrape_document(client, store, args.document_id)
    logger.info("saved document %s", result.document_id)
    logger.info("record %s", result.record_locator)
    for page in result.pages:
        logger.info("page %s %s bytes %s", page.page_number, page.byte_count, page.key)


def _chunk(args: argparse.Namespace) -> None:
    store = _store_from_args(args)
    try:
        factory = session_factory(args.database_url)
        init_db(args.database_url)
        with factory() as session:
            result = chunk_stored_document(
                session,
                store,
                args.document_id,
                TextractOcr(region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")),
                target_chars=args.target_chars,
                overlap_chars=args.overlap_chars,
                force=args.force,
            )
    except OperationalError as exc:
        raise AhpnError(
            "could not connect to Postgres. Start it with `docker compose up -d` "
            f"or set AHPN_DATABASE_URL. ({exc.orig})"
        ) from exc
    logger.info("document %s", result.ahpn_id)
    logger.info("ocr pages %s, skipped %s", result.pages_ocred, result.pages_skipped)
    logger.info("chunks %s", result.chunk_count)


def _show(args: argparse.Namespace) -> None:
    try:
        with session_factory(args.database_url)() as session:
            document = session.scalar(select(Document).where(Document.ahpn_id == args.document_id))
            if document is None:
                raise AhpnError(f"document {args.document_id} is not in the database; run `ahpn chunk` first")
            pages = session.scalars(
                select(Page).where(Page.document_id == document.id).order_by(Page.page_number)
            ).all()
            chunks = session.scalars(
                select(Chunk).where(Chunk.document_id == document.id).order_by(Chunk.chunk_index)
            ).all()
    except OperationalError as exc:
        raise AhpnError(f"could not connect to Postgres. ({exc.orig})") from exc
    logger.info("document %s %s", document.ahpn_id, document.source_page)
    for page in pages:
        logger.info("page %s %s", page.page_number, page.image_key)
    for chunk in chunks:
        logger.info(
            "--- chunk %s, pages %s-%s, %s chars, confidence %.1f",
            chunk.chunk_index,
            chunk.page_start,
            chunk.page_end,
            chunk.char_count,
            chunk.confidence,
        )
        logger.info("%s", chunk.text)


def _store_from_args(args: argparse.Namespace) -> DocumentStore:
    if args.storage == "s3":
        if not args.s3_bucket:
            raise AhpnError("S3 storage requires --s3-bucket or AHPN_S3_BUCKET")
        return S3DocumentStore(
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        )
    return LocalDocumentStore(args.data_dir)


def _add_document_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--document-id",
        type=int,
        default=EXAMPLE_DOCUMENT_ID,
        help=f"AHPN document id (default: {EXAMPLE_DOCUMENT_ID})",
    )


def _add_database_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        default=database_url_from_env(),
        help="SQLAlchemy database URL (or AHPN_DATABASE_URL, or AHPN_DB_HOST and related AHPN_DB_* variables)",
    )


def _add_storage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--storage",
        choices=("local", "s3"),
        default=os.environ.get("AHPN_STORAGE", "local"),
        help="Where page images were written (default: local, or AHPN_STORAGE)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("AHPN_DATA_DIR", "data/documents")),
        help="Local document directory (default: data/documents, or AHPN_DATA_DIR)",
    )
    parser.add_argument("--s3-bucket", default=os.environ.get("AHPN_S3_BUCKET"))
    parser.add_argument("--s3-prefix", default=os.environ.get("AHPN_S3_PREFIX", "documents"))


if __name__ == "__main__":
    sys.exit(main())
