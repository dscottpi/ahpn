from __future__ import annotations

from ahpn.sample import decade_of, page_bucket, stratified_sample


def test_page_bucket_and_decade() -> None:
    assert page_bucket(1) == "1"
    assert page_bucket(4) == "2-5"
    assert page_bucket(12) == "6-20"
    assert page_bucket(55) == "21+"
    assert decade_of("1983-03-06") == "1980s"
    assert decade_of(None) == "unknown"


def test_stratified_sample_nested_layers() -> None:
    candidates = []
    doc_id = 1
    for dtype in ("DOC", "PAG"):
        for year in (1975, 1983, 1991):
            for pages in (1, 3, 10, 40):
                for _ in range(20):
                    candidates.append(
                        {
                            "id": doc_id,
                            "visibility": "PUB",
                            "document_type": dtype,
                            "page_count": pages,
                            "date": f"{year}-01-01",
                            "case": 100,
                        }
                    )
                    doc_id += 1
    plan = stratified_sample(candidates, total=100, core_size=30, seed=7)
    assert plan.total == 100
    assert len(plan.layers["100"]) == 100
    assert plan.layers["100"] == plan.layers["300"][:100]
    assert plan.layers["300"] == plan.layers["1000"][:300]
    assert len(set(plan.layers["1000"])) == 100
