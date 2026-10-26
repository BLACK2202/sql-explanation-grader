"""Embedding, persistent indexing, idempotency and duplicate detection."""
from __future__ import annotations

from conftest import FIXTURES, fixture_items
from sqlrag.index.bm25 import BM25Index, tokenize
from sqlrag.index.embeddings import HashEmbedder
from sqlrag.index.knowledge_base import KnowledgeBase
from sqlrag.index.vector_store import collection_name_for
from sqlrag.ingest.models import DocumentMeta


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(64)
    a, b = e.embed_query("left join null"), e.embed_query("left join null")
    assert a == b and len(a) == 64
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_ingest_is_idempotent(kb):
    first = kb.stats()
    report = kb.ingest(fixture_items())
    assert sorted(report.unchanged) == ["fx-grouping", "fx-joins", "fx-ordering"]
    assert report.chunks_written == 0 and report.chunks_deleted == 0
    assert kb.stats()["chunks"] == first["chunks"] == kb.store.count()


def test_changed_document_replaces_old_chunks(kb, tmp_path):
    path = tmp_path / "ordering.txt"
    path.write_text((FIXTURES / "ordering.txt").read_text() + "\n\nNULLS FIRST changes where NULL values sort.", encoding="utf-8")
    before = kb.store.count_document("fx-ordering")
    report = kb.ingest([(path, DocumentMeta(doc_id="fx-ordering", title="Ordering and limits", license="test"))])
    assert report.updated == ["fx-ordering"]
    assert report.chunks_deleted == before
    texts = [h.text for h in kb.store.get_all(where={"doc_id": "fx-ordering"})]
    assert any("NULLS FIRST" in t for t in texts)
    assert kb.store.count_document("fx-ordering") == kb.registry["documents"]["fx-ordering"]["n_chunks"]


def test_duplicate_document_is_detected(kb):
    dup = (FIXTURES / "joins.html", DocumentMeta(doc_id="copy-of-joins", title="Joins again", license="test"))
    report = kb.ingest([dup])
    assert report.duplicates == [{"doc_id": "copy-of-joins", "duplicate_of": "fx-joins"}]
    assert kb.store.count_document("copy-of-joins") == 0


def test_duplicate_chunks_across_documents_are_skipped(kb):
    text = (FIXTURES / "grouping.md").read_text(encoding="utf-8")
    combined = text + "\n\n## Extra\n\nA brand new paragraph about DISTINCT removing duplicate rows from the result set."
    report = kb.ingest([(combined.encode(), DocumentMeta(doc_id="grouping-plus", title="Grouping rows", license="test"), "g.md")])
    assert report.added == ["grouping-plus"]
    assert report.chunks_skipped_duplicate >= 1


def test_failed_documents_are_reported_not_fatal(kb):
    report = kb.ingest([(b"   ", DocumentMeta(doc_id="empty", title="Empty", license="test"), "empty.txt")])
    assert report.failed and report.failed[0]["doc_id"] == "empty"
    assert kb.registry["documents"]["empty"]["status"] == "failed"


def test_metadata_is_stored_and_filterable(kb):
    hits = kb.store.get_all(where={"doc_type": "tutorial"})
    assert hits and all(h.metadata["license"] == "test" for h in hits)
    md = hits[0].metadata
    for key in ("doc_id", "title", "section", "source_url", "chunk_index", "content_hash", "n_tokens"):
        assert key in md
    joins = kb.store.get_all(where={"cat__join": True})
    assert joins and all(h.metadata["doc_id"] == "fx-joins" for h in joins)


def test_persistent_store_survives_reopen(settings, embedder):
    kb1 = KnowledgeBase(settings, embedder=embedder)
    kb1.ingest(fixture_items())
    n = kb1.store.count()
    kb2 = KnowledgeBase(settings, embedder=embedder)
    assert kb2.store.count() == n
    assert kb2.ingest(fixture_items()).unchanged  # registry persisted too


def test_remove_document(kb):
    n = kb.remove_document("fx-ordering")
    assert n > 0 and kb.store.count_document("fx-ordering") == 0
    assert "fx-ordering" not in kb.registry["documents"]


def test_collection_name_is_safe():
    name = collection_name_for("BAAI/bge-small-en-v1.5-c400")
    assert name == "sql_docs__baai-bge-small-en-v1.5-c400"


def test_bm25_prefers_exact_terms_and_sql_phrases():
    idx = BM25Index().build(["a", "b", "c"], [
        "The LEFT JOIN keeps unmatched rows and fills NULL.",
        "GROUP BY combines rows; HAVING filters groups.",
        "Ordering rows with ORDER BY DESC.",
    ])
    assert idx.search("left join", 3)[0][0] == "a"
    assert idx.search("having groups", 3)[0][0] == "b"
    assert "left_join" in tokenize("a LEFT JOIN b") and "is_null" in tokenize("x IS NULL")
    assert idx.search("nothing matches zzz", 3) == []
