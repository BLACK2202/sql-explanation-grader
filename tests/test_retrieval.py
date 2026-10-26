"""Dense, BM25 and hybrid retrieval, RRF fusion, coverage and sufficiency."""
from __future__ import annotations

import pytest

from sqlrag.retrieval import Retriever, chunk_covers
from sqlrag.sql_analysis import feature_spec


def test_modes_return_ranked_results(retriever):
    for mode in ("dense", "bm25", "hybrid"):
        hits = retriever.search("LEFT JOIN keeps unmatched rows NULL", mode=mode, k=3)
        assert hits, mode
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
        assert hits[0].doc_id == "fx-joins", mode


def test_hybrid_uses_reciprocal_rank_fusion(retriever):
    hits = retriever.search("HAVING filters groups", mode="hybrid", k=5)
    k = retriever.settings.rrf_k
    for h in hits:
        expected = sum(1.0 / (k + r) for r in (h.dense_rank, h.bm25_rank) if r is not None)
        assert h.fused_score == pytest.approx(expected)


def test_unknown_mode_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_metadata_filter(retriever):
    hits = retriever.search("rows", mode="hybrid", k=10, where={"doc_id": "fx-grouping"})
    assert hits and {h.doc_id for h in hits} == {"fx-grouping"}


def test_sql_retrieval_covers_features_and_labels_sources(retriever):
    res = retriever.retrieve_for_sql("SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL ORDER BY c.name LIMIT 5")
    assert res.sufficient and res.coverage == 1.0
    assert set(res.required_features) == {"left_join", "is_null", "order_by", "limit"}
    assert [c.source_label for c in res.chunks] == [f"S{i}" for i in range(1, len(res.chunks) + 1)]
    assert any("left_join" in c.covers for c in res.chunks)
    assert any(q.feature == "left_join" for q in res.queries)  # LEFT JOIN + IS NULL combined query


def test_uncovered_feature_makes_evidence_insufficient(retriever):
    res = retriever.retrieve_for_sql("SELECT json_extract(data, '$.a') FROM events")
    assert not res.sufficient
    assert "fn:json_extract" in res.missing_features
    assert any("json_extract" in r for r in res.reasons)


def test_relevance_filter_drops_low_similarity(kb, settings):
    settings.min_similarity = 0.99
    res = Retriever(kb, settings).retrieve_for_sql("SELECT name FROM t GROUP BY name HAVING count(*) > 1")
    assert res.chunks == [] and not res.sufficient
    assert any("relevance filter" in r for r in res.reasons)


def test_context_budget_and_top_k(kb, settings):
    settings.top_k = 2
    res = Retriever(kb, settings).retrieve_for_sql("SELECT city, count(*) FROM customers GROUP BY city HAVING count(*) > 2 ORDER BY city DESC LIMIT 3")
    assert len(res.chunks) <= 2
    tight = Retriever(kb, settings).retrieve_for_sql("SELECT city FROM customers ORDER BY city", max_context_tokens=1)
    assert tight.chunks == []


def test_query_strategies(retriever):
    sql = "SELECT city, count(*) FROM customers GROUP BY city HAVING count(*) > 2"
    assert len(retriever.retrieve_for_sql(sql, strategy="raw_sql").queries) == 1
    assert len(retriever.retrieve_for_sql(sql, strategy="probe").queries) == 1
    assert len(retriever.retrieve_for_sql(sql).queries) > 2
    with pytest.raises(ValueError):
        retriever.retrieve_for_sql(sql, strategy="nope")


def test_evidence_terms_are_case_aware():
    case = feature_spec("case")
    assert chunk_covers("The CASE expression returns a value", case)
    assert not chunk_covers("in this case the rows are kept", case)  # prose 'case' is not the keyword
    rank = feature_spec("rank")
    assert chunk_covers("rank() returns the rank with gaps", rank)
    assert not chunk_covers("dense_rank() has no gaps", rank)
