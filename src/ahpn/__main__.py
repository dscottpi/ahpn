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
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_REQUEST_DELAY_SECONDS,
    DEFAULT_SEARCH_K,
    EXAMPLE_DOCUMENT_ID,
)
from ahpn.db import database_url_from_env, init_db, session_factory
from ahpn.embed import BedrockEmbedder, FakeEmbedder, embed_chunks
from ahpn.evals.compare import compare_runs, compare_with_qrels, write_compare
from ahpn.evals.generate import (
    assign_splits,
    known_item_queries,
    metadata_queries,
    no_answer_queries,
)
from ahpn.evals.judge import grade_pair
from ahpn.evals.kappa import cohen_kappa, meets_calibration_threshold
from ahpn.evals.metrics import QueryRecord, load_queries, write_qrels, write_queries
from ahpn.evals.ocr_eval import load_gold_pages, score_pages, summarize
from ahpn.evals.review import review_queries
from ahpn.evals.run import EvalConfig, run_eval
from ahpn.evals.scale import (
    ann_vs_exact,
    build_cost_projection,
    distractor_curve,
    extrapolate_log,
    serialize_ann,
    serialize_distractor,
    write_scale_report,
)
from ahpn.ingest import ingest_documents, project_costs
from ahpn.llm import BedrockLlm, FakeLlm
from ahpn.models import Chunk, Document, OcrLine, Page
from ahpn.ocr import TextractOcr
from ahpn.sample import (
    fetch_document_candidates,
    load_sample,
    read_ids_file,
    stratified_sample,
    write_sample,
)
from ahpn.scrape import scrape_document
from ahpn.search import SearchFilters, search
from ahpn.storage import DocumentStore, LocalDocumentStore, S3DocumentStore

logger = logging.getLogger("ahpn")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ahpn",
        description="Download public AHPN documents, index them, and evaluate retrieval.",
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

    db_init = subparsers.add_parser("db-init", help="Create tables, extensions, FTS config, and vector indexes.")
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

    embed = subparsers.add_parser("embed", help="Embed stored chunks with Bedrock or the fake embedder.")
    _add_database_arg(embed)
    embed.add_argument("--model", default=os.environ.get("AHPN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    embed.add_argument("--fake", action="store_true", help="Use the deterministic FakeEmbedder")
    embed.add_argument("--document-id", type=int, action="append", dest="document_ids")
    embed.add_argument("--force", action="store_true")

    search_cmd = subparsers.add_parser("search", help="Search indexed chunks (bm25|vector|hybrid).")
    _add_database_arg(search_cmd)
    search_cmd.add_argument("query")
    search_cmd.add_argument("--mode", choices=("bm25", "vector", "hybrid"), default="hybrid")
    search_cmd.add_argument("--k", type=int, default=DEFAULT_SEARCH_K)
    search_cmd.add_argument("--model", default=os.environ.get("AHPN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    search_cmd.add_argument("--fake-embed", action="store_true")
    search_cmd.add_argument("--department")
    search_cmd.add_argument("--date-from", type=_parse_date)
    search_cmd.add_argument("--date-to", type=_parse_date)
    search_cmd.add_argument("--exact-vector", action="store_true")

    sample = subparsers.add_parser("sample", help="Draw a stratified 1k-document sample from the public API.")
    sample.add_argument("--api-base-url", default=os.environ.get("AHPN_API_BASE_URL", API_BASE_URL))
    sample.add_argument("--delay", type=float, default=float(os.environ.get("AHPN_REQUEST_DELAY", DEFAULT_REQUEST_DELAY_SECONDS)))
    sample.add_argument("--total", type=int, default=1000)
    sample.add_argument("--core-size", type=int, default=300)
    sample.add_argument("--pages", type=int, default=40, help="Random listing pages to draw candidates from")
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--out", type=Path, default=Path("evals/corpus/sample_v1.json"))

    ingest = subparsers.add_parser("ingest", help="Resumable scrape + chunk + embed for an ids file.")
    _add_storage_args(ingest)
    _add_database_arg(ingest)
    ingest.add_argument("--ids-file", type=Path, required=True)
    ingest.add_argument("--api-base-url", default=os.environ.get("AHPN_API_BASE_URL", API_BASE_URL))
    ingest.add_argument("--delay", type=float, default=float(os.environ.get("AHPN_REQUEST_DELAY", DEFAULT_REQUEST_DELAY_SECONDS)))
    ingest.add_argument("--model", default=os.environ.get("AHPN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    ingest.add_argument("--fake-embed", action="store_true")
    ingest.add_argument("--skip-scrape", action="store_true")
    ingest.add_argument("--skip-embed", action="store_true")
    ingest.add_argument("--force-ocr", action="store_true")
    ingest.add_argument("--report", type=Path, default=Path("evals/corpus/ingest_report.json"))
    ingest.add_argument("--target-chars", type=int, default=int(os.environ.get("AHPN_CHUNK_CHARS", DEFAULT_CHUNK_CHARS)))
    ingest.add_argument("--overlap-chars", type=int, default=int(os.environ.get("AHPN_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)))

    show = subparsers.add_parser("show", help="Print a stored document's pages and chunks.")
    _add_document_id(show)
    _add_database_arg(show)

    eval_parser = subparsers.add_parser("eval", help="Retrieval / OCR evaluation commands.")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    eval_run = eval_sub.add_parser("run", help="Run retrieval metrics with ranx.")
    _add_database_arg(eval_run)
    eval_run.add_argument("--queries", type=Path, default=Path("evals/queries/v1.jsonl"))
    eval_run.add_argument("--qrels", type=Path, default=Path("evals/qrels/v1.tsv"))
    eval_run.add_argument("--out-dir", type=Path, default=Path("evals/runs"))
    eval_run.add_argument("--mode", choices=("bm25", "vector", "hybrid"), default="hybrid")
    eval_run.add_argument("--split", default="dev")
    eval_run.add_argument("--k", type=int, default=100)
    eval_run.add_argument("--model", default=os.environ.get("AHPN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    eval_run.add_argument("--fake-embed", action="store_true")
    eval_run.add_argument("--level", choices=("page", "document"), default="page")

    eval_compare = eval_sub.add_parser("compare", help="Compare two eval JSON reports.")
    eval_compare.add_argument("a", type=Path)
    eval_compare.add_argument("b", type=Path)
    eval_compare.add_argument("--qrels", type=Path)
    eval_compare.add_argument("--out", type=Path)

    eval_review = eval_sub.add_parser("review", help="Accept/edit/reject generated queries.")
    eval_review.add_argument("--pending", type=Path, default=Path("evals/queries/pending.jsonl"))
    eval_review.add_argument("--accepted", type=Path, default=Path("evals/queries/v1.jsonl"))

    eval_gen = eval_sub.add_parser("generate-queries", help="Build metadata/known-item/no-answer seed queries.")
    _add_database_arg(eval_gen)
    eval_gen.add_argument("--out", type=Path, default=Path("evals/queries/pending.jsonl"))
    eval_gen.add_argument("--limit", type=int, default=50)

    eval_judge = eval_sub.add_parser("judge", help="Grade one query/page pair (or calibrate kappa).")
    eval_judge.add_argument("--query", required=True)
    eval_judge.add_argument("--page-key", required=True)
    eval_judge.add_argument("--page-text", required=True)
    eval_judge.add_argument("--lang", default="es")
    eval_judge.add_argument("--fake", action="store_true")
    eval_judge.add_argument("--model", default=os.environ.get("AHPN_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    eval_judge.add_argument("--cache-dir", type=Path, default=Path("evals/judge_cache"))

    eval_kappa = eval_sub.add_parser("kappa", help="Cohen's kappa between two grade TSV files (qid page grade).")
    eval_kappa.add_argument("human", type=Path)
    eval_kappa.add_argument("model", type=Path)

    eval_ocr = eval_sub.add_parser("ocr", help="Score OCR CER/WER against hand transcriptions.")
    _add_database_arg(eval_ocr)
    eval_ocr.add_argument("--gold", type=Path, default=Path("evals/ocr/gold.jsonl"))
    eval_ocr.add_argument("--out", type=Path, default=Path("evals/ocr/report.json"))

    eval_scale = eval_sub.add_parser("scale", help="ANN vs exact, distractor curve, and cost projection.")
    _add_database_arg(eval_scale)
    eval_scale.add_argument("--queries", type=Path, default=Path("evals/queries/v1.jsonl"))
    eval_scale.add_argument("--qrels", type=Path, default=Path("evals/qrels/v1.tsv"))
    eval_scale.add_argument("--sample", type=Path, default=Path("evals/corpus/sample_v1.json"))
    eval_scale.add_argument("--out", type=Path, default=Path("evals/runs/scale_report.json"))
    eval_scale.add_argument("--model", default=os.environ.get("AHPN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    eval_scale.add_argument("--fake-embed", action="store_true")
    eval_scale.add_argument("--split", default="dev")
    eval_scale.add_argument("--sample-docs", type=int)
    eval_scale.add_argument("--sample-pages", type=int)
    eval_scale.add_argument("--sample-chunks", type=int)

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
        elif args.command == "embed":
            _embed(args)
        elif args.command == "search":
            _search(args)
        elif args.command == "sample":
            _sample(args)
        elif args.command == "ingest":
            _ingest(args)
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


def _embed(args: argparse.Namespace) -> None:
    init_db(args.database_url)
    embedder = _embedder_from_args(args)
    with session_factory(args.database_url)() as session:
        document_ids = None
        if args.document_ids:
            rows = session.scalars(select(Document).where(Document.ahpn_id.in_(args.document_ids))).all()
            document_ids = [row.id for row in rows]
        result = embed_chunks(session, embedder, document_ids=document_ids, force=args.force)
    logger.info("model %s embedded %s skipped %s", result.model, result.chunks_embedded, result.chunks_skipped)


def _search(args: argparse.Namespace) -> None:
    init_db(args.database_url)
    embedder = None
    if args.mode in {"vector", "hybrid"}:
        embedder = FakeEmbedder(model=args.model) if args.fake_embed else BedrockEmbedder(
            model=args.model,
            region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        )
    filters = SearchFilters(department=args.department, date_from=args.date_from, date_to=args.date_to)
    with session_factory(args.database_url)() as session:
        hits = search(
            session,
            args.query,
            mode=args.mode,
            k=args.k,
            filters=filters,
            embedder=embedder,
            exact_vector=args.exact_vector,
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


def _sample(args: argparse.Namespace) -> None:
    with AhpnClient(api_base_url=args.api_base_url, delay_seconds=args.delay) as client:
        candidates = fetch_document_candidates(client, pages=args.pages, seed=args.seed)
    plan = stratified_sample(candidates, total=args.total, core_size=args.core_size, seed=args.seed)
    write_sample(plan, args.out)
    logger.info("wrote %s (%s docs)", args.out, plan.total)
    logger.info("layers %s", {key: len(value) for key, value in plan.layers.items()})


def _ingest(args: argparse.Namespace) -> None:
    ids = read_ids_file(args.ids_file)
    store = _store_from_args(args)
    init_db(args.database_url)
    embedder = None
    if not args.skip_embed:
        embedder = _embedder_from_args(args)
    ocr = TextractOcr(region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    with AhpnClient(api_base_url=args.api_base_url, delay_seconds=args.delay) as client:
        with session_factory(args.database_url)() as session:
            report = ingest_documents(
                ids=ids,
                client=client,
                store=store,
                session=session,
                ocr=ocr,
                embedder=embedder,
                target_chars=args.target_chars,
                overlap_chars=args.overlap_chars,
                force_ocr=args.force_ocr,
                skip_scrape=args.skip_scrape,
                skip_embed=args.skip_embed,
                report_path=args.report,
            )
    logger.info(
        "ingest done docs=%s pages=%s chunks=%s textract_pages=%s embedded=%s",
        len(report.documents),
        report.total_pages,
        report.total_chunks,
        report.total_textract_pages,
        report.total_embedded_chunks,
    )
    if report.total_pages and report.documents:
        ok = sum(1 for row in report.documents if row.status == "ok")
        projection = project_costs(
            sample_docs=max(ok, 1),
            sample_pages=report.total_pages,
            sample_chunks=report.total_chunks,
        )
        logger.info("cost projection %s", json.dumps(projection))


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
    if args.eval_command == "run":
        init_db(args.database_url)
        embedder = None
        if args.mode in {"vector", "hybrid"}:
            embedder = _embedder_from_args(args)
        with session_factory(args.database_url)() as session:
            result = run_eval(
                session,
                queries_path=args.queries,
                qrels_path=args.qrels,
                out_dir=args.out_dir,
                embedder=embedder,
                config=EvalConfig(
                    mode=args.mode,
                    split=args.split,
                    k=args.k,
                    embedding_model=args.model,
                    level=args.level,
                ),
            )
        logger.info("metrics %s", json.dumps(result.metrics))
        logger.info("wrote %s", result.path)
    elif args.eval_command == "compare":
        if args.qrels:
            payload = compare_with_qrels(args.a, args.b, args.qrels)
        else:
            payload = compare_runs(args.a, args.b)
        if args.out:
            write_compare(args.out, payload)
        logger.info("%s", json.dumps(payload, indent=2))
    elif args.eval_command == "review":
        kept, edited, rejected = review_queries(args.pending, args.accepted)
        logger.info("accepted=%s edited=%s rejected=%s", kept, edited, rejected)
    elif args.eval_command == "generate-queries":
        init_db(args.database_url)
        queries: list[QueryRecord] = []
        next_id = 1
        with session_factory(args.database_url)() as session:
            documents = session.scalars(select(Document).limit(args.limit)).all()
            for document in documents:
                breadcrumbs = (document.metadata_json or {}).get("breadcrumbs") or []
                date_str = ((document.metadata_json or {}).get("document") or {}).get("date")
                meta = metadata_queries(
                    ahpn_id=document.ahpn_id,
                    breadcrumbs=breadcrumbs,
                    date=date_str,
                    start_id=next_id,
                )
                queries.extend(meta)
                next_id += max(1, len(meta))
                page = session.scalar(
                    select(Page).where(Page.document_id == document.id).order_by(Page.page_number).limit(1)
                )
                if page is None:
                    continue
                lines = [
                    line.text
                    for line in session.scalars(
                        select(OcrLine).where(OcrLine.page_id == page.id).order_by(OcrLine.line_index)
                    )
                ]
                known = known_item_queries(
                    ahpn_id=document.ahpn_id,
                    page_number=page.page_number,
                    lines=lines,
                    start_id=next_id,
                )
                queries.extend(known)
                next_id += max(1, len(known))
        queries.extend(no_answer_queries(next_id))
        queries = assign_splits(queries)
        write_queries(args.out, queries)
        logger.info("wrote %s queries to %s", len(queries), args.out)
    elif args.eval_command == "judge":
        llm = FakeLlm(default='{"grade": 2, "rationale": "fake"}') if args.fake else BedrockLlm(
            model=args.model,
            region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        )
        grade, rationale = grade_pair(
            llm,
            query=args.query,
            lang=args.lang,
            page_key=args.page_key,
            page_text=args.page_text,
            cache_dir=args.cache_dir,
        )
        logger.info("grade=%s rationale=%s", grade, rationale)
    elif args.eval_command == "kappa":
        human = _load_grade_tsv(args.human)
        model = _load_grade_tsv(args.model)
        keys = sorted(set(human) & set(model))
        kappa = cohen_kappa([human[key] for key in keys], [model[key] for key in keys])
        logger.info("n=%s kappa=%.3f calibrated=%s", len(keys), kappa, meets_calibration_threshold(kappa))
    elif args.eval_command == "ocr":
        init_db(args.database_url)
        gold = load_gold_pages(args.gold)
        hypothesis: dict[tuple[int, int], str] = {}
        with session_factory(args.database_url)() as session:
            for page in gold:
                document = session.scalar(select(Document).where(Document.ahpn_id == page.ahpn_id))
                if document is None:
                    continue
                page_row = session.scalar(
                    select(Page).where(Page.document_id == document.id, Page.page_number == page.page_number)
                )
                if page_row is None:
                    continue
                lines = session.scalars(
                    select(OcrLine).where(OcrLine.page_id == page_row.id).order_by(OcrLine.line_index)
                ).all()
                hypothesis[(page.ahpn_id, page.page_number)] = "\n".join(line.text for line in lines)
        report = summarize(score_pages(gold, hypothesis))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("ocr cer=%.3f wer=%.3f by_type=%s", report["cer"], report["wer"], report["by_type"])
    elif args.eval_command == "scale":
        init_db(args.database_url)
        embedder = _embedder_from_args(args)
        sample = load_sample(args.sample) if args.sample.is_file() else None
        with session_factory(args.database_url)() as session:
            ann = serialize_ann(
                ann_vs_exact(
                    session,
                    queries_path=args.queries,
                    embedder=embedder,
                    ahpn_ids=sample.layers.get("1000") if sample else None,
                    split=args.split,
                )
            )
            curve = []
            if sample is not None and args.qrels.is_file():
                curve = serialize_distractor(
                    distractor_curve(
                        session,
                        queries_path=args.queries,
                        qrels_path=args.qrels,
                        embedder=embedder,
                        layers=sample.layers,
                        split=args.split,
                    )
                )
        projection = None
        if args.sample_docs and args.sample_pages is not None and args.sample_chunks is not None:
            projection = build_cost_projection(
                sample_docs=args.sample_docs,
                sample_pages=args.sample_pages,
                sample_chunks=args.sample_chunks,
            )
        from ahpn.evals.scale import DistractorPoint

        extrapolated = None
        if curve:
            extrapolated = extrapolate_log(
                [DistractorPoint(**point) for point in curve],
                metric="recall@50",
                target_size=20_000_000,
            )
        payload = {
            "ann_vs_exact": ann,
            "distractor_curve": curve,
            "recall_at_50_extrapolated_20m": extrapolated,
            "cost_projection": projection,
        }
        write_scale_report(args.out, payload)
        logger.info("wrote %s", args.out)


def _embedder_from_args(args: argparse.Namespace):
    model = getattr(args, "model", DEFAULT_EMBEDDING_MODEL)
    if getattr(args, "fake", False) or getattr(args, "fake_embed", False):
        return FakeEmbedder(model=model)
    return BedrockEmbedder(
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


def _load_grade_tsv(path: Path) -> dict[tuple[str, str], int]:
    grades: dict[tuple[str, str], int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        qid, page_key, grade = line.split()[:3]
        grades[(qid, page_key)] = int(grade)
    return grades


if __name__ == "__main__":
    sys.exit(main())
