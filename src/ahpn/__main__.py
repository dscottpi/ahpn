from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from ahpn.chunk import chunk_stored_document
from ahpn.client import AhpnClient, AhpnError
from ahpn.config import (
    API_BASE_URL,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_RERANK_MODEL,
    DEFAULT_REQUEST_DELAY_SECONDS,
    DEFAULT_SEARCH_K,
    EXAMPLE_DOCUMENT_ID,
)
from ahpn.db import database_url_from_env, init_db, session_factory
from ahpn.evals.run import EvalConfig, run_eval
from ahpn.models import Chunk, Document, Page
from ahpn.ocr import TextractOcr
from ahpn.rerank import BedrockReranker, FakeReranker
from ahpn.scrape import scrape_document
from ahpn.search import SearchFilters, search
from ahpn.storage import DocumentStore, LocalDocumentStore, S3DocumentStore

_SEARCH_MODES = ("bm25", "bm25_rerank")

logger = logging.getLogger("ahpn")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ahpn",
        description="Download public AHPN documents, index them, and run BM25 (+ rerank) search.",
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

    db_init = subparsers.add_parser("db-init", help="Create tables and the Spanish FTS schema.")
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

    search_cmd = subparsers.add_parser("search", help="Search indexed chunks (bm25|bm25_rerank).")
    _add_database_arg(search_cmd)
    search_cmd.add_argument("query")
    search_cmd.add_argument("--mode", choices=_SEARCH_MODES, default="bm25_rerank")
    search_cmd.add_argument("--k", type=int, default=DEFAULT_SEARCH_K)
    search_cmd.add_argument("--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES)
    search_cmd.add_argument(
        "--rerank-model",
        default=os.environ.get("AHPN_RERANK_MODEL", DEFAULT_RERANK_MODEL),
    )
    search_cmd.add_argument("--fake-rerank", action="store_true")
    search_cmd.add_argument("--department")
    search_cmd.add_argument("--date-from", type=_parse_date)
    search_cmd.add_argument("--date-to", type=_parse_date)

    show = subparsers.add_parser("show", help="Print a stored document's pages and chunks.")
    _add_document_id(show)
    _add_database_arg(show)

    eval_parser = subparsers.add_parser("eval", help="Run page-level retrieval metrics.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_sub.add_parser("run", help="Score queries against page-image qrels.")
    _add_database_arg(eval_run)
    eval_run.add_argument("--queries", type=Path, default=Path("evals/fixtures/smoke/queries.jsonl"))
    eval_run.add_argument("--qrels", type=Path, default=Path("evals/fixtures/smoke/qrels.tsv"))
    eval_run.add_argument("--out-dir", type=Path, default=Path("evals/runs"))
    eval_run.add_argument("--mode", choices=_SEARCH_MODES, default="bm25_rerank")
    eval_run.add_argument("--split", default="dev")
    eval_run.add_argument("--k", type=int, default=DEFAULT_SEARCH_K)
    eval_run.add_argument("--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES)
    eval_run.add_argument(
        "--rerank-model",
        default=os.environ.get("AHPN_RERANK_MODEL", DEFAULT_RERANK_MODEL),
    )
    eval_run.add_argument("--fake-rerank", action="store_true")

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
        elif args.command == "search":
            _search(args)
        elif args.command == "show":
            _show(args)
        elif args.command == "eval":
            _eval(args)
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


def _search(args: argparse.Namespace) -> None:
    init_db(args.database_url)
    reranker = _reranker_from_args(args) if args.mode == "bm25_rerank" else None
    filters = SearchFilters(department=args.department, date_from=args.date_from, date_to=args.date_to)
    with session_factory(args.database_url)() as session:
        hits = search(
            session,
            args.query,
            mode=args.mode,
            k=args.k,
            filters=filters,
            reranker=reranker,
            rerank_candidates=args.rerank_candidates,
        )
    for hit in hits:
        logger.info(
            "#%s score=%.4f doc=%s pages=%s-%s\n%s\n",
            hit.rank,
            hit.score,
            hit.ahpn_id,
            hit.page_start,
            hit.page_end,
            hit.text[:500],
        )


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


def _eval(args: argparse.Namespace) -> None:
    init_db(args.database_url)
    reranker = _reranker_from_args(args) if args.mode == "bm25_rerank" else None
    with session_factory(args.database_url)() as session:
        result = run_eval(
            session,
            queries_path=args.queries,
            qrels_path=args.qrels,
            out_dir=args.out_dir,
            reranker=reranker,
            config=EvalConfig(
                mode=args.mode,
                split=args.split,
                k=args.k,
                rerank_candidates=args.rerank_candidates,
                rerank_model=args.rerank_model,
            ),
        )
    logger.info("metrics %s", json.dumps(result.metrics))
    logger.info("wrote %s", result.path)


def _reranker_from_args(args: argparse.Namespace):
    model = getattr(args, "rerank_model", DEFAULT_RERANK_MODEL)
    if getattr(args, "fake_rerank", False):
        return FakeReranker(model=model)
    return BedrockReranker(
        model=model,
        region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
    )


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


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


if __name__ == "__main__":
    sys.exit(main())
