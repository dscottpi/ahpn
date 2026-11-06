"""Retrieval and OCR evaluation harness for AHPN.

Relevance is judged per page image (JPEG). Multi-page documents are ordered
lists of page images; chunks may span pages but qrels stay at ``ahpn_id:page``.
"""

from ahpn.evals.metrics import (
    QueryRecord,
    load_qrels,
    load_queries,
    page_hits_from_chunk_hits,
    write_qrels,
    write_queries,
)
from ahpn.evals.pages import page_docno, page_image_key, parse_page_docno

__all__ = [
    "QueryRecord",
    "load_qrels",
    "load_queries",
    "page_docno",
    "page_hits_from_chunk_hits",
    "page_image_key",
    "parse_page_docno",
    "write_qrels",
    "write_queries",
]
