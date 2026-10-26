"""FastAPI application: JSON API + dashboard.

Run:  python -m sqlrag serve            (http://127.0.0.1:8000)
Docs: http://127.0.0.1:8000/docs        (interactive OpenAPI documentation)

Security notes
  * SQL is never executed against data: it is only compiled (EXPLAIN) in a
    private in-memory database with an authorizer that blocks writes, ATTACH
    and PRAGMA (sql_analysis.validate_sql).
  * Uploaded documents are parsed as text (HTML/Markdown/TXT/PDF) and never
    executed; size and extension are limited; filenames are not trusted.
  * Retrieved passages are treated as untrusted data in prompts.
  * Write endpoints (ingest/delete/labels) can be protected with
    SQLRAG_ADMIN_TOKEN, and ingestion can be disabled with SQLRAG_ALLOW_INGEST=0.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hmac
import json
import logging
import pathlib
import re
import threading
import time
from functools import partial
from typing import Literal

import anyio
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .checks import run_checks
from .config import ROOT, get_settings
from .evaluation.experiment import read_jsonl
from .ingest.parsers import SUPPORTED_EXTENSIONS
from .llm import LLMError
from .sql_analysis import SQLValidationError, analyze_sql, validate_sql

logger = logging.getLogger("sqlrag.api")
settings = get_settings()
WEB_DIR = ROOT / "web"
LEGACY_DIR = ROOT / "dashboard"


# ---------------------------------------------------------------------------
# Lazily created services (model loading takes a few seconds)
# ---------------------------------------------------------------------------
class Services:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._kb = None
        self._client = None
        self._retriever = None
        self._explainer = None

    @property
    def kb(self):
        with self._lock:
            if self._kb is None:
                from .index.knowledge_base import KnowledgeBase

                self._kb = KnowledgeBase(settings)
            return self._kb

    @property
    def retriever(self):
        if self._retriever is None:
            from .retrieval import Retriever

            self._retriever = Retriever(self.kb, settings)
        return self._retriever

    @property
    def client(self):
        if self._client is None:
            from .llm import make_client

            self._client = make_client(settings)
        return self._client

    @property
    def explainer(self):
        if self._explainer is None:
            from .explain import Explainer

            self._explainer = Explainer(settings, self.client, self.retriever)
        return self._explainer

    def judge(self, mode: str = "rubric", model: str | None = None):
        from .judge import make_judge

        return make_judge(settings, self.client, model or settings.judge_model, mode)


services = Services()

app = FastAPI(
    title="SQL Explanation Grader: RAG Enhanced",
    version=__version__,
    description="Retrieval-augmented SQL explanations with source citations, abstention, a schema-aware judge "
                "and a controlled baseline-vs-RAG evaluation. LLM-judge results are unverified measurements.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        logger.info("%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code, (time.time() - t0) * 1000)
    return response


async def run_blocking(fn, *args, timeout: float | None = None, **kwargs):
    """Run blocking work (LLM calls, embeddings) off the event loop, with a timeout."""
    try:
        return await asyncio.wait_for(anyio.to_thread.run_sync(partial(fn, *args, **kwargs)), timeout or settings.request_timeout_s)
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, f"timed out after {timeout or settings.request_timeout_s:.0f}s") from exc


@app.exception_handler(SQLValidationError)
async def _sql_error(_: Request, exc: SQLValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc), "error_type": "invalid_sql"})


@app.exception_handler(LLMError)
async def _llm_error(_: Request, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": f"LLM backend error: {exc}", "error_type": "llm_unavailable"})


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if settings.admin_token and not hmac.compare_digest(x_admin_token or "", settings.admin_token):
        raise HTTPException(401, "missing or wrong X-Admin-Token header")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ExplainRequest(BaseModel):
    sql: str = Field(..., min_length=6, max_length=10_000, examples=["SELECT name FROM customers WHERE city = 'Tunis';"])
    schema_ddl: str = Field("", alias="schema", max_length=20_000,
                            examples=["CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);"])
    mode: Literal["baseline", "rag", "both"] = "rag"
    policy: Literal["abstain", "fallback", "off"] | None = None
    judge: bool = False
    generator_model: str | None = Field(None, max_length=100)

    model_config = {"populate_by_name": True}


class EvaluateRequest(BaseModel):
    sql: str = Field(..., min_length=6, max_length=10_000)
    schema_ddl: str = Field("", alias="schema", max_length=20_000)
    explanation: str = Field(..., min_length=10, max_length=8_000)
    judge_mode: Literal["rubric", "legacy"] = "rubric"
    judge_model: str | None = Field(None, max_length=100)

    model_config = {"populate_by_name": True}


class RetrieveRequest(BaseModel):
    sql: str | None = Field(None, max_length=10_000)
    query: str | None = Field(None, max_length=2_000)
    mode: Literal["hybrid", "dense", "bm25"] | None = None
    top_k: int | None = Field(None, ge=1, le=20)
    strategy: Literal["decomposed", "probe", "raw_sql"] = "decomposed"


class ManifestIngestRequest(BaseModel):
    doc_ids: list[str] | None = None
    include_optional: bool = False
    force: bool = False


class LabelRequest(BaseModel):
    exp_id: str = Field(..., max_length=120)
    item_id: str = Field(..., max_length=40)
    condition_token: str = Field(..., min_length=10, max_length=10, description="Opaque token from /api/labels/next (keeps labelling blind)")
    explanation_sha256_12: str = Field(..., min_length=12, max_length=12)
    label: Literal["good", "bad"]
    annotator: str = Field(..., min_length=2, max_length=60)
    notes: str = Field("", max_length=1000)


# ---------------------------------------------------------------------------
# Health and configuration
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["system"])
async def health():
    out = {"status": "ok", "version": __version__, "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    try:
        out["index"] = {k: v for k, v in services.kb.stats().items() if k in ("collection", "chunks", "documents_indexed", "embedding_model")}
    except Exception as exc:
        out["status"] = "degraded"
        out["index"] = {"error": str(exc)}
    try:
        def _models():
            return [m["name"] for m in services.client.list_models()] if hasattr(services.client, "list_models") else []
        names = await run_blocking(_models, timeout=10)
        out["llm"] = {"provider": settings.llm_provider, "reachable": True, "models": names,
                      "generator_model": settings.generator_model, "generator_available": settings.generator_model in names,
                      "judge_model": settings.judge_model, "judge_available": settings.judge_model in names}
    except Exception as exc:
        out["status"] = "degraded"
        out["llm"] = {"provider": settings.llm_provider, "reachable": False, "error": str(exc)[:200]}
    return out


@app.get("/api/config", tags=["system"])
async def config():
    return settings.public_dict()


# ---------------------------------------------------------------------------
# Explain / evaluate / retrieve
# ---------------------------------------------------------------------------
@app.post("/api/explain", tags=["rag"])
async def explain(req: ExplainRequest):
    validate_sql(req.sql, req.schema_ddl)
    modes = ["baseline", "rag"] if req.mode == "both" else [req.mode]
    analysis = analyze_sql(req.sql)
    results = {}
    for mode in modes:
        r = await run_blocking(services.explainer.explain, req.sql, req.schema_ddl, mode=mode, policy=req.policy,
                               model=req.generator_model)
        if r.get("explanation_clean"):
            r["checks"] = run_checks(r["explanation_clean"], analysis)
            if req.judge:
                r["judge"] = await run_blocking(services.judge("rubric").grade, req.sql, req.schema_ddl, r["explanation_clean"],
                                                analysis=analysis)
                r["judge"]["label_provenance"] = "LLM judge decision (unverified)"
        results[mode] = r
    return {"sql": req.sql, "results": results}


@app.post("/api/evaluate", tags=["evaluation"])
async def evaluate(req: EvaluateRequest):
    validate_sql(req.sql, req.schema_ddl)
    analysis = analyze_sql(req.sql)
    verdict = await run_blocking(services.judge(req.judge_mode, req.judge_model).grade, req.sql, req.schema_ddl, req.explanation,
                                 analysis=analysis)
    return {"judge": {**verdict, "label_provenance": "LLM judge decision (unverified)"},
            "checks": run_checks(req.explanation, analysis), "analysis": analysis.as_dict()}


@app.post("/api/retrieve", tags=["rag"])
async def retrieve(req: RetrieveRequest):
    if not req.sql and not req.query:
        raise HTTPException(422, "provide `sql` or `query`")
    if req.sql:
        res = await run_blocking(services.retriever.retrieve_for_sql, req.sql, mode=req.mode, top_k=req.top_k, strategy=req.strategy)
        return res.as_dict(include_text=True)
    hits = await run_blocking(services.retriever.search, req.query, mode=req.mode, k=req.top_k)
    return {"query": req.query, "chunks": [h.as_dict() for h in hits]}


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
@app.get("/api/documents", tags=["knowledge base"])
async def documents():
    kb = services.kb
    manifest = json.loads(settings.corpus_manifest.read_text(encoding="utf-8"))
    registry = {d["doc_id"]: d for d in kb.list_documents()}
    available = [
        {"doc_id": d["doc_id"], "title": d["title"], "url": d["url"], "license": d["license"], "doc_type": d["doc_type"],
         "ingest_by_default": d.get("ingest_by_default", True), "status": registry.get(d["doc_id"], {}).get("status", "not_indexed"),
         "note": d.get("note")}
        for d in manifest["documents"]
    ]
    return {"stats": kb.stats(), "documents": list(registry.values()), "manifest": available, "licenses": manifest.get("licenses")}


@app.get("/api/documents/{doc_id}/chunks", tags=["knowledge base"])
async def document_chunks(doc_id: str):
    hits = [h for h in services.kb.all_chunks().values() if h.metadata.get("doc_id") == doc_id]
    if not hits:
        raise HTTPException(404, f"no chunks for {doc_id!r}")
    hits.sort(key=lambda h: h.metadata.get("chunk_index", 0))
    return {"doc_id": doc_id, "chunks": [{"chunk_id": h.chunk_id, "text": h.text, **h.metadata} for h in hits]}


def _slug(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return s[:60] or "document"


@app.post("/api/ingest", tags=["knowledge base"], dependencies=[Depends(require_admin)])
async def ingest_upload(file: UploadFile = File(...), title: str = Form(""), license: str = Form("user-provided"),
                        source_url: str = Form(""), doc_id: str = Form("")):
    if not settings.allow_ingest:
        raise HTTPException(403, "ingestion is disabled on this server (SQLRAG_ALLOW_INGEST=0)")
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"unsupported file type {ext!r}; allowed: {sorted(SUPPORTED_EXTENSIONS)}")
    limit = int(settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
    from .ingest.models import DocumentMeta

    did = "upload-" + _slug(doc_id or title or pathlib.Path(file.filename or "document").stem)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    stored = settings.uploads_dir / f"{did}{ext}"
    stored.write_bytes(data)
    meta = DocumentMeta(doc_id=did, title=title.strip()[:200], source_url=source_url.strip()[:500], license=license.strip()[:100] or "user-provided",
                        doc_type="user_upload", path=str(stored.relative_to(ROOT)) if stored.is_relative_to(ROOT) else stored.name,
                        retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    report = await run_blocking(services.kb.ingest, [(data, meta, f"upload{ext}")], timeout=600)
    if report.failed:
        raise HTTPException(422, {"message": "document could not be ingested", "report": report.as_dict()})
    return {"doc_id": did, "report": report.as_dict(), "stats": services.kb.stats()}


@app.post("/api/ingest/manifest", tags=["knowledge base"], dependencies=[Depends(require_admin)])
async def ingest_manifest(req: ManifestIngestRequest):
    if not settings.allow_ingest:
        raise HTTPException(403, "ingestion is disabled on this server (SQLRAG_ALLOW_INGEST=0)")
    report = await run_blocking(services.kb.ingest_manifest, include_optional=req.include_optional, doc_ids=req.doc_ids,
                                force=req.force, timeout=900)
    return {"report": report.as_dict(), "stats": services.kb.stats()}


@app.delete("/api/documents/{doc_id}", tags=["knowledge base"], dependencies=[Depends(require_admin)])
async def delete_document(doc_id: str):
    if not settings.allow_ingest:
        raise HTTPException(403, "ingestion is disabled on this server (SQLRAG_ALLOW_INGEST=0)")
    n = await run_blocking(services.kb.remove_document, doc_id)
    return {"doc_id": doc_id, "chunks_deleted": n, "stats": services.kb.stats()}


# ---------------------------------------------------------------------------
# Experiments, reports and labels
# ---------------------------------------------------------------------------
RAG_RESULTS = settings.results_dir / "rag"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def _exp_dir(exp_id: str) -> pathlib.Path:
    if not _SAFE_ID.match(exp_id):
        raise HTTPException(400, "invalid experiment id")
    d = RAG_RESULTS / exp_id
    if not (d / "generations.jsonl").exists():
        raise HTTPException(404, f"experiment {exp_id!r} not found")
    return d


@app.get("/api/experiments", tags=["evaluation"])
async def experiments():
    out = []
    if RAG_RESULTS.exists():
        for d in sorted(RAG_RESULTS.iterdir()):
            if not (d / "generations.jsonl").exists():
                continue
            meta = json.loads((d / "metadata.json").read_text(encoding="utf-8")) if (d / "metadata.json").exists() else {}
            summaries = {}
            for s in sorted(d.glob("summary__*.json")):
                summaries[s.stem.split("__", 1)[1]] = json.loads(s.read_text(encoding="utf-8"))
            out.append({"exp_id": d.name, "kind": "live experiment", "metadata": meta, "summaries": summaries})
    return {"experiments": out}


@app.get("/api/experiments/{exp_id}/items", tags=["evaluation"])
async def experiment_items(exp_id: str, judge: str | None = Query(None, max_length=120)):
    d = _exp_dir(exp_id)
    gens = [g for g in read_jsonl(d / "generations.jsonl") if not g.get("error")]
    tags = sorted(p.stem.split("__", 1)[1] for p in d.glob("judgments__*.jsonl"))
    tag = judge if judge in tags else (tags[0] if tags else None)
    judg = {(j["id"], j["condition"]): j for j in read_jsonl(d / f"judgments__{tag}.jsonl")} if tag else {}
    items: dict[str, dict] = {}
    for g in gens:
        e = items.setdefault(g["id"], {"id": g["id"], "feature": g.get("feature"), "domain": g.get("domain"), "sql": g["sql"]})
        j = judg.get((g["id"], g["condition"]), {})
        e[g["condition"]] = {
            "explanation": g.get("explanation"), "explanation_clean": g.get("explanation_clean"),
            "grounding_status": g.get("grounding_status"), "abstained": g.get("abstained"),
            "abstention_reason": g.get("abstention_reason"), "citations": g.get("citations"),
            "citation_report": g.get("citation_report"), "retrieval": g.get("retrieval"), "metrics": g.get("metrics"),
            "judge": {k: j.get(k) for k in ("grade", "score", "reason", "dimensions", "unsupported_claims", "rubric_pass")} if j else None,
            "checks": run_checks(g["explanation_clean"], analyze_sql(g["sql"])) if g.get("explanation_clean") else None,
        }
    return {"exp_id": exp_id, "judge_tag": tag, "judge_tags": tags, "items": sorted(items.values(), key=lambda x: x["id"])}


@app.get("/api/experiments/{exp_id}/export", tags=["evaluation"])
async def experiment_export(exp_id: str, judge: str = Query(..., max_length=120), fmt: Literal["csv", "json"] = "csv"):
    d = _exp_dir(exp_id)
    if not _SAFE_ID.match(judge):
        raise HTTPException(400, "invalid judge tag")
    path = d / (f"items__{judge}.csv" if fmt == "csv" else f"summary__{judge}.json")
    if not path.exists():
        raise HTTPException(404, "export not found; run `python -m sqlrag summarize` first")
    return FileResponse(path, filename=f"{exp_id}__{path.name}")


@app.get("/api/experiments/{exp_id}/report.pdf", tags=["evaluation"])
async def experiment_pdf(exp_id: str, judge: str = Query(..., max_length=120)):
    d = _exp_dir(exp_id)
    jpath = d / f"judgments__{judge}.jsonl"
    if not _SAFE_ID.match(judge) or not jpath.exists():
        raise HTTPException(404, "unknown judge tag")
    from .evaluation.compare import summarize_experiment
    from .evaluation.report_pdf import write_experiment_pdf

    def build():
        return write_experiment_pdf(d, summarize_experiment(d, jpath), settings.reports_dir)

    pdf = await run_blocking(build, timeout=120)
    return FileResponse(pdf, media_type="application/pdf", filename=pdf.name)


@app.get("/api/retrieval-eval", tags=["evaluation"])
async def retrieval_eval():
    p = RAG_RESULTS / "retrieval_eval.json"
    if not p.exists():
        return {"available": False}
    return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}


@app.get("/api/agreement", tags=["evaluation"])
async def agreement():
    from .evaluation.agreement import compute_agreement

    return await run_blocking(compute_agreement, settings.data_dir / "labels", settings.results_dir, timeout=60)


@app.get("/api/historical", tags=["evaluation"])
async def historical():
    """Results recorded before the RAG extension (legacy judge, no schema)."""
    runs = []
    for p in sorted(settings.results_dir.glob("*.jsonl")):
        rows = read_jsonl(p)
        scored = [r for r in rows if r.get("error") is None and r.get("score") is not None]
        if not scored:
            continue
        kind = "derived from RAG experiment" if ("-baseline-" in p.name or "-rag-" in p.name) else "historical (pre-RAG)"
        runs.append({"file": p.name, "kind": kind, "n": len(scored), "good": sum(r["score"] for r in scored),
                     "judge_saw_schema": any(r.get("judge_saw_schema") for r in scored)})
    return {"note": "Historical runs used the original judge, which did not see the database schema. "
                    "They are LLM-judge approval rates, not verified accuracy.", "runs": runs}


@app.get("/api/reports", tags=["reports"])
async def reports():
    out = []
    for folder in (settings.reports_dir, settings.results_dir):
        if folder.exists():
            for p in sorted(folder.glob("*.pdf")):
                out.append({"name": p.name, "folder": folder.name, "bytes": p.stat().st_size,
                            "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})
    return {"reports": out}


@app.get("/api/reports/{folder}/{name}", tags=["reports"])
async def report_file(folder: Literal["reports", "results"], name: str):
    base = settings.reports_dir if folder == "reports" else settings.results_dir
    if not re.match(r"^[A-Za-z0-9._-]+\.pdf$", name):
        raise HTTPException(400, "invalid file name")
    path = (base / name).resolve()
    if path.parent != base.resolve() or not path.exists():
        raise HTTPException(404, "report not found")
    return FileResponse(path, media_type="application/pdf", filename=name)


@app.get("/api/examples", tags=["system"])
async def examples():
    out = []
    for split in ("dev", "abstention_probe"):
        p = settings.data_dir / f"{split}.jsonl"
        if p.exists():
            for it in read_jsonl(p):
                out.append({"split": split, "id": it["id"], "feature": it.get("feature"), "sql": it["sql"], "schema": it["schema"],
                            "expected": it.get("expected_behaviour")})
    return {"examples": out}


LABELS_FILE = settings.data_dir / "labels" / "human_labels.jsonl"
_labels_lock = threading.Lock()


@app.get("/api/labels/next", tags=["labelling"])
async def next_label(exp_id: str, annotator: str = Query(..., min_length=2, max_length=60)):
    """Next explanation to label, shown blind (the condition is not revealed)."""
    import hashlib
    import random

    from .evaluation.agreement import text_hash

    d = _exp_dir(exp_id)
    done = {(r["item_id"], r["condition"]) for r in read_jsonl(LABELS_FILE) if r.get("exp_id") == exp_id and r.get("annotator") == annotator}
    gens = [g for g in read_jsonl(d / "generations.jsonl") if not g.get("error") and g.get("explanation_clean")]
    todo = [g for g in gens if (g["id"], g["condition"]) not in done]
    if not todo:
        return {"done": True, "labelled": len(done)}
    # Deterministic shuffle per annotator so conditions are interleaved.
    rnd = random.Random(int(hashlib.sha256(annotator.encode()).hexdigest()[:8], 16))
    rnd.shuffle(todo)
    g = todo[0]
    items = {it["id"]: it for it in read_jsonl(settings.data_dir / "all.jsonl")}
    return {"done": False, "labelled": len(done), "remaining": len(todo), "exp_id": exp_id, "item_id": g["id"],
            "condition_token": _condition_token(exp_id, g["id"], g["condition"]), "sql": g["sql"],
            "schema": items.get(g["id"], {}).get("schema", ""),
            "explanation": g["explanation_clean"], "explanation_sha256_12": text_hash(g["explanation_clean"])}


def _condition_token(exp_id: str, item_id: str, condition: str) -> str:
    import hashlib

    return hashlib.sha256(f"{exp_id}|{item_id}|{condition}|blind-label".encode()).hexdigest()[:10]


@app.post("/api/labels", tags=["labelling"], dependencies=[Depends(require_admin)])
async def add_label(req: LabelRequest):
    _exp_dir(req.exp_id)
    condition = next((c for c in ("baseline", "rag") if _condition_token(req.exp_id, req.item_id, c) == req.condition_token), None)
    if condition is None:
        raise HTTPException(400, "unknown condition token")
    data = req.model_dump(exclude={"condition_token"})
    row = {**data, "condition": condition, "label_source": "human",
           "labelled_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with _labels_lock:
        LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LABELS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Legacy dashboard endpoints (ported from dashboard/server.py)
# ---------------------------------------------------------------------------
@app.get("/api/runs", tags=["legacy dashboard"])
async def legacy_runs():
    runs = []
    for f in sorted(settings.results_dir.glob("*.jsonl")):
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            runs.append({"run_id": parts[0], "split": parts[1]})
    return runs


@app.get("/api/status", tags=["legacy dashboard"])
async def legacy_status(run_id: str = "v2", split: str = "dev"):
    import importlib.util

    spec = importlib.util.spec_from_file_location("legacy_dashboard_server", LEGACY_DIR / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS_DIR = settings.results_dir
    mod.DATA_DIR = settings.data_dir
    if not (_SAFE_ID.match(run_id) and _SAFE_ID.match(split)):
        raise HTTPException(400, "invalid run id")
    return mod.read_results(run_id, split)


@app.get("/api/export-pdf", tags=["legacy dashboard"])
async def legacy_export_pdf(run_id: str = "v2", split: str = "dev"):
    if not (_SAFE_ID.match(run_id) and _SAFE_ID.match(split)):
        raise HTTPException(400, "invalid run id")
    results_file = settings.results_dir / f"{run_id}_{split}.jsonl"
    if not results_file.exists():
        raise HTTPException(404, "results file not found")
    import sys

    sys.path.insert(0, str(ROOT))
    from harness.export_pdf import generate_pdf_report

    pdf_path = settings.results_dir / f"{run_id}_{split}_report.pdf"
    await run_blocking(generate_pdf_report, results_file, pdf_path, settings.data_dir, timeout=120)
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)


@app.get("/legacy", include_in_schema=False)
async def legacy_dashboard():
    return FileResponse(LEGACY_DIR / "index.html", media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
else:  # pragma: no cover
    @app.get("/", include_in_schema=False)
    async def root():
        return PlainTextResponse("SQL Explanation Grader API. See /docs.")
