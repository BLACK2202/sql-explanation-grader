"""FastAPI endpoints with a fake LLM, hash embeddings and a temporary store."""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT, fixture_items, scripted_llm, write_jsonl

SCHEMA = "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);\nCREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);"
SQL = "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL"


@pytest.fixture
def api(tmp_path, monkeypatch):
    env = {
        "SQLRAG_STORE_DIR": str(tmp_path / "store"),
        "SQLRAG_UPLOADS_DIR": str(tmp_path / "uploads"),
        "SQLRAG_RESULTS_DIR": str(tmp_path / "results"),
        "SQLRAG_REPORTS_DIR": str(tmp_path / "reports"),
        "SQLRAG_DATA_DIR": str(tmp_path / "data"),
        "SQLRAG_EMBEDDING_BACKEND": "hash",
        "SQLRAG_EMBEDDING_MODEL": "hash-384",
        "SQLRAG_CHUNK_SIZE": "120",
        "SQLRAG_CHUNK_OVERLAP": "20",
        "SQLRAG_MIN_CHUNK_TOKENS": "20",
        "SQLRAG_MIN_SIMILARITY": "0",
        "SQLRAG_MAX_UPLOAD_MB": "0.01",
        "SQLRAG_ADMIN_TOKEN": "",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "all.jsonl").write_text((ROOT / "data" / "tiny.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    mod = importlib.reload(importlib.import_module("sqlrag.api"))
    mod.services._client = scripted_llm()
    mod.services.kb.ingest(fixture_items())
    return mod


@pytest.fixture
def client(api):
    return TestClient(api.app)


def test_health_and_config(client, api):
    api.services._client.list_models = lambda: [{"name": "qwen2.5:7b"}]
    h = client.get("/api/health").json()
    assert h["index"]["chunks"] > 0 and h["llm"]["reachable"]
    cfg = client.get("/api/config").json()
    assert "openai_api_key" not in cfg and "admin_token" not in cfg and cfg["embedding_backend"] == "hash"


def test_explain_both_modes_with_judge(client):
    r = client.post("/api/explain", json={"sql": SQL, "schema": SCHEMA, "mode": "both", "judge": True})
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert res["baseline"]["grounding_status"] == "not_applicable"
    assert res["rag"]["grounding_status"] == "grounded"
    assert res["rag"]["retrieval"]["chunks"][0]["source_label"] == "S1"
    assert res["rag"]["judge"]["label_provenance"].startswith("LLM judge")
    assert "checks" in res["rag"]


def test_explain_abstains_on_uncovered_feature(client):
    r = client.post("/api/explain", json={"sql": "SELECT json_extract(name, '$.a') FROM customers", "schema": SCHEMA, "mode": "rag"})
    assert r.status_code == 200 and r.json()["results"]["rag"]["abstained"]


@pytest.mark.parametrize("sql", ["DELETE FROM customers", "SELECT 1; DROP TABLE customers", "SELECT * FROM nope"])
def test_explain_rejects_unsafe_sql(client, sql):
    r = client.post("/api/explain", json={"sql": sql, "schema": SCHEMA})
    assert r.status_code == 422 and r.json()["error_type"] == "invalid_sql"


def test_request_validation(client):
    assert client.post("/api/explain", json={"sql": "x", "schema": SCHEMA}).status_code == 422
    assert client.post("/api/explain", json={"sql": SQL, "schema": SCHEMA, "mode": "wrong"}).status_code == 422
    assert client.post("/api/retrieve", json={}).status_code == 422


def test_llm_failure_maps_to_502(client, api):
    from sqlrag.llm import FakeLLMClient

    def down(*_):
        from sqlrag.llm import LLMError
        raise LLMError("connection refused")

    api.services._client = FakeLLMClient(down)
    api.services._explainer = None
    r = client.post("/api/explain", json={"sql": SQL, "schema": SCHEMA, "mode": "baseline"})
    assert r.status_code == 502 and r.json()["error_type"] == "llm_unavailable"


def test_evaluate_endpoint(client):
    r = client.post("/api/evaluate", json={"sql": SQL, "schema": SCHEMA, "explanation": "Customers with no orders."})
    body = r.json()
    assert r.status_code == 200 and body["judge"]["grade"] == "good" and "checks" in body


def test_retrieve_sql_and_free_text(client):
    r = client.post("/api/retrieve", json={"sql": SQL}).json()
    assert r["sufficient"] and r["chunks"]
    r = client.post("/api/retrieve", json={"query": "HAVING filters groups", "mode": "bm25", "top_k": 2}).json()
    assert len(r["chunks"]) <= 2 and r["chunks"][0]["doc_id"] == "fx-grouping"


def test_documents_and_chunks(client):
    d = client.get("/api/documents").json()
    assert d["stats"]["documents_indexed"] == 3
    assert any(m["doc_id"] == "sqlite-json" for m in d["manifest"])
    c = client.get("/api/documents/fx-joins/chunks").json()
    assert c["chunks"] and c["chunks"][0]["doc_id"] == "fx-joins"
    assert client.get("/api/documents/missing/chunks").status_code == 404


def test_upload_ingest_duplicate_and_limits(client):
    md = b"# Notes\n\nThe EXCEPT operator returns rows of the left query that are not in the right query. " * 3
    r = client.post("/api/ingest", files={"file": ("notes.md", md, "text/markdown")}, data={"title": "Set operations", "license": "own work"})
    assert r.status_code == 200, r.text
    assert r.json()["report"]["added"] == ["upload-set-operations"]
    r2 = client.post("/api/ingest", files={"file": ("notes.md", md, "text/markdown")}, data={"title": "Set operations", "license": "own work"})
    assert r2.json()["report"]["unchanged"] == ["upload-set-operations"]
    assert client.post("/api/ingest", files={"file": ("x.exe", b"MZ", "application/octet-stream")}, data={"license": "x"}).status_code == 415
    big = b"a " * 20_000
    assert client.post("/api/ingest", files={"file": ("big.txt", big, "text/plain")}, data={"license": "x"}).status_code == 413
    assert client.delete("/api/documents/upload-set-operations").json()["chunks_deleted"] > 0


def test_admin_token_protects_writes(client, api, monkeypatch):
    monkeypatch.setattr(api.settings, "admin_token", "s3cret")
    files = {"file": ("n.txt", b"ORDER BY sorts rows. " * 10, "text/plain")}
    assert client.post("/api/ingest", files=files, data={"license": "x"}).status_code == 401
    assert client.post("/api/ingest", files=files, data={"license": "x"}, headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_ingest_can_be_disabled(client, api, monkeypatch):
    monkeypatch.setattr(api.settings, "allow_ingest", False)
    r = client.post("/api/ingest", files={"file": ("n.txt", b"x " * 50, "text/plain")}, data={"license": "x"})
    assert r.status_code == 403


def _make_experiment(api):
    from sqlrag.evaluation.agreement import text_hash

    d = api.settings.results_dir / "rag" / "exp_api"
    gens = [
        {"id": "t1", "condition": "baseline", "sql": "SELECT COUNT(*) FROM users WHERE age > 30;", "explanation": "Counts users over 30.",
         "explanation_clean": "Counts users over 30.", "grounding_status": "not_applicable", "metrics": {"total_s": 1.0}, "error": None},
        {"id": "t1", "condition": "rag", "sql": "SELECT COUNT(*) FROM users WHERE age > 30;", "explanation": "Counts users older than 30. [S1]",
         "explanation_clean": "Counts users older than 30.", "grounding_status": "grounded", "metrics": {"total_s": 2.0}, "error": None},
    ]
    write_jsonl(d / "generations.jsonl", gens)
    write_jsonl(d / "judgments__fake__rubric.jsonl", [
        {"id": "t1", "condition": c, "score": 1, "grade": "good", "reason": "ok", "dimensions": {"correctness": 5, "completeness": 5, "clarity": 5, "groundedness": 5},
         "n_unsupported_claims": 0, "rubric_pass": True, "grade_consistent": True, "error": None} for c in ("baseline", "rag")])
    (d / "metadata.json").write_text(json.dumps({"dataset": {"split": "tiny", "n_items": 1}}), encoding="utf-8")
    from sqlrag.evaluation.compare import summarize_experiment

    summarize_experiment(d, d / "judgments__fake__rubric.jsonl")
    return d, text_hash


def test_experiment_endpoints_and_exports(client, api):
    _make_experiment(api)
    exps = client.get("/api/experiments").json()["experiments"]
    assert exps[0]["exp_id"] == "exp_api" and "fake__rubric" in exps[0]["summaries"]
    items = client.get("/api/experiments/exp_api/items").json()
    assert items["items"][0]["rag"]["judge"]["grade"] == "good"
    assert client.get("/api/experiments/exp_api/export?judge=fake__rubric&fmt=csv").status_code == 200
    pdf = client.get("/api/experiments/exp_api/report.pdf?judge=fake__rubric")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.get("/api/experiments/..%2Fetc/items").status_code in (400, 404)
    assert client.get("/api/reports/reports/..%2F..%2Fsecret.pdf").status_code in (400, 404)


def test_blind_labelling_flow(client, api):
    _make_experiment(api)
    nxt = client.get("/api/labels/next", params={"exp_id": "exp_api", "annotator": "LB"}).json()
    assert nxt["condition_token"] not in ("baseline", "rag") and "condition" not in nxt
    payload = {k: nxt[k] for k in ("exp_id", "item_id", "condition_token", "explanation_sha256_12")}
    assert client.post("/api/labels", json={**payload, "label": "good", "annotator": "LB"}).status_code == 200
    bad = {**payload, "condition_token": "0123456789"}
    assert client.post("/api/labels", json={**bad, "label": "good", "annotator": "LB"}).status_code == 400
    rows = [json.loads(x) for x in (api.settings.data_dir / "labels" / "human_labels.jsonl").read_text().splitlines()]
    assert rows[0]["label_source"] == "human" and rows[0]["condition"] in ("baseline", "rag")
    agreement = client.get("/api/agreement").json()
    assert agreement["label_counts"]["human"] == 1


def test_dashboard_and_legacy_routes(client):
    assert "SQL Explanation Grader" in client.get("/").text
    assert client.get("/legacy").status_code == 200
    assert client.get("/docs").status_code == 200
    assert isinstance(client.get("/api/runs").json(), list)
