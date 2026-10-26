"""Integration tests against a real local Ollama server and the real corpus index.

Skipped by default. Run with:  RUN_OLLAMA_TESTS=1 pytest -m ollama
Requires `python -m sqlrag ingest` and the models named below to be pulled.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.ollama

GEN = os.environ.get("SQLRAG_TEST_GENERATOR", "llama3.2:3b")
JUDGE = os.environ.get("SQLRAG_TEST_JUDGE", "qwen2.5:3b")
SCHEMA = "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);\nCREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);"
SQL = "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL"


@pytest.fixture(scope="module")
def explainer():
    from sqlrag.config import get_settings
    from sqlrag.explain import Explainer

    return Explainer.from_settings(get_settings(generator_model=GEN, judge_model=JUDGE))


def test_real_rag_explanation_is_grounded_or_weak(explainer):
    r = explainer.explain(SQL, SCHEMA, mode="rag")
    assert r["grounding_status"] in {"grounded", "weakly_grounded"}
    assert r["explanation"] and r["metrics"]["in_tok"] > r["metrics"].get("context_tokens", 0)


def test_real_abstention_needs_no_llm(explainer):
    r = explainer.explain("SELECT json_extract(name, '$.a') FROM customers", SCHEMA, mode="rag", policy="abstain")
    assert r["abstained"] and r["metrics"]["in_tok"] == 0


def test_real_judge_returns_rubric(explainer):
    from sqlrag.judge import Judge

    out = Judge(explainer.settings, explainer.client, JUDGE).grade(SQL, SCHEMA, "This lists customers who have never placed an order.")
    assert out["grade"] in {"good", "bad"} and set(out["dimensions"]) == {"correctness", "completeness", "clarity", "groundedness"}
