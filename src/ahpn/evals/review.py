from __future__ import annotations

from pathlib import Path

from ahpn.evals.metrics import QueryRecord, load_queries, write_queries


def review_queries(
    pending_path: Path,
    accepted_path: Path,
    *,
    input_fn=input,
    print_fn=print,
) -> tuple[int, int, int]:
    """Interactive terminal loop: accept / edit / reject pending generated queries.

    Returns (accepted, edited, rejected) counts.
    """
    pending = load_queries(pending_path)
    accepted = load_queries(accepted_path) if accepted_path.is_file() else []
    accepted_ids = {query.id for query in accepted}
    kept = 0
    edited = 0
    rejected = 0

    for query in pending:
        if query.id in accepted_ids:
            continue
        print_fn(f"\n[{query.id}] lang={query.lang} types={','.join(query.types)} source={query.source}")
        print_fn(query.text)
        choice = input_fn("[a]ccept / [e]dit / [r]eject / [q]uit > ").strip().lower()
        if choice.startswith("q"):
            break
        if choice.startswith("r"):
            rejected += 1
            continue
        if choice.startswith("e"):
            new_text = input_fn("new text > ").strip()
            if not new_text:
                rejected += 1
                continue
            query = QueryRecord(
                id=query.id,
                text=new_text,
                lang=query.lang,
                types=list(query.types),
                source=query.source,
                split=query.split,
                notes=query.notes,
            )
            edited += 1
        else:
            kept += 1
        accepted.append(query)
        accepted_ids.add(query.id)
        write_queries(accepted_path, accepted)

    write_queries(accepted_path, accepted)
    return kept, edited, rejected
