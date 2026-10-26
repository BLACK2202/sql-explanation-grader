"""Shared fixtures. The default suite is offline: hash embeddings, in-memory or
temporary ChromaDB, and a scripted fake LLM. Tests that need a running Ollama
server are marked `ollama` and skipped unless RUN_OLLAMA_TESTS=1."""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlrag.config import get_settings  # noqa: E402
from sqlrag.index.embeddings import HashEmbedder  # noqa: E402
from sqlrag.ingest.models import DocumentMeta  # noqa: E402
from sqlrag.llm import FakeLLMClient  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_OLLAMA_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="needs a running Ollama server (set RUN_OLLAMA_TESTS=1)")
    for item in items:
        if "ollama" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def settings(tmp_path):
    return get_settings(
        store_dir=tmp_path / "store",
        uploads_dir=tmp_path / "uploads",
        results_dir=tmp_path / "results",
        reports_dir=tmp_path / "reports",
        embedding_backend="hash",
        embedding_model="hash-384",
        chunk_size=120,
        chunk_overlap=20,
        min_chunk_tokens=20,
        top_k=4,
        min_similarity=0.0,
        llm_provider="fake",
        generator_model="fake-gen",
        judge_model="fake-judge",
    )


@pytest.fixture
def embedder():
    return HashEmbedder()


def fixture_items() -> list[tuple[pathlib.Path, DocumentMeta]]:
    return [
        (FIXTURES / "joins.html", DocumentMeta(doc_id="fx-joins", title="Joins", license="test", doc_type="tutorial",
                                               categories=("join",), path="tests/fixtures/joins.html")),
        (FIXTURES / "grouping.md", DocumentMeta(doc_id="fx-grouping", title="Grouping", license="test", doc_type="tutorial",
                                                categories=("group_by",), path="tests/fixtures/grouping.md")),
        (FIXTURES / "ordering.txt", DocumentMeta(doc_id="fx-ordering", title="Ordering and limits", license="test",
                                                 doc_type="tutorial", categories=("order_by",), path="tests/fixtures/ordering.txt")),
    ]


@pytest.fixture
def kb(settings, embedder):
    from sqlrag.index.knowledge_base import KnowledgeBase

    k = KnowledgeBase(settings, embedder=embedder, in_memory=True)
    k.store.reset()  # EphemeralClient collections are process-global; start clean
    k.registry["documents"] = {}
    k.ingest(fixture_items())
    return k


@pytest.fixture
def retriever(kb, settings):
    from sqlrag.retrieval import Retriever

    return Retriever(kb, settings)


def scripted_llm(*, rag_text: str | None = None, sufficient: bool = True, grade: str = "good"):
    """Fake LLM that answers like the generator or the judge depending on the schema."""

    def responder(model, system, user, schema):
        name = schema.__name__
        if name == "Explanation":
            return {"explanation": "This lists the names of customers who have no matching order, sorted by name."}
        if name == "RAGExplanation":
            labels = re.findall(r'<source id="(S\d+)"', user)
            first = labels[0] if labels else "S1"
            text = rag_text or f"This lists customers that have no matching order; unmatched rows get empty order values. [{first}]"
            return {"explanation": text, "citations": [{"source_id": first, "supports": "left join keeps unmatched rows"}],
                    "evidence_sufficient": sufficient, "missing_evidence": "" if sufficient else "json functions"}
        if name == "RubricVerdict":
            score = 5 if grade == "good" else 2
            return {"correctness": score, "completeness": score, "clarity": 5, "groundedness": score,
                    "unsupported_claims": [] if grade == "good" else ["invented claim"], "grade": grade,
                    "reason": "scripted verdict for tests"}
        if name == "Verdict":
            return {"grade": grade, "reason": "scripted legacy verdict"}
        raise AssertionError(f"unexpected schema {name}")

    return FakeLLMClient(responder)


@pytest.fixture
def fake_llm():
    return scripted_llm()


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
