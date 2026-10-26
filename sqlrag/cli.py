"""Command-line interface: python -m sqlrag <command> [options]

Commands
  ingest           parse, chunk, embed and index the corpus (idempotent)
  stats            show index statistics and the document registry
  search           run a free-text retrieval query
  retrieve         show the evidence retrieved for an SQL query
  explain          explain an SQL query (baseline or RAG) with a local LLM
  retrieval-eval   measure retrieval quality (Recall@k, Precision@k, MRR, nDCG)
  experiment       run the controlled baseline-vs-RAG experiment
  summarize        recompute the summary and reports of a finished experiment
  agreement        measure LLM-judge agreement with reference labels
  serve            start the API and dashboard (uvicorn)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import get_settings


def _settings_from(args) -> "object":
    return get_settings(
        embedding_backend=getattr(args, "embedding_backend", None),
        embedding_model=getattr(args, "embedding_model", None),
        chunk_size=getattr(args, "chunk_size", None),
        chunk_overlap=getattr(args, "chunk_overlap", None),
        retrieval_mode=getattr(args, "mode", None),
        top_k=getattr(args, "top_k", None),
        generator_model=getattr(args, "generator_model", None),
        judge_model=getattr(args, "judge_model", None),
        grounding_policy=getattr(args, "policy", None),
    )


def _kb(settings):
    from .index.knowledge_base import KnowledgeBase

    return KnowledgeBase(settings)


def cmd_ingest(args) -> int:
    settings = _settings_from(args)
    kb = _kb(settings)
    if args.rebuild:
        kb.rebuild()
    if args.files:
        import pathlib

        from .ingest.models import DocumentMeta

        items = []
        for f in args.files:
            p = pathlib.Path(f)
            items.append((p, DocumentMeta(doc_id=args.doc_id or p.stem, title=args.title or "", license=args.license,
                                          doc_type="user_upload", path=str(p))))
        report = kb.ingest(items, force=args.force)
    else:
        report = kb.ingest_manifest(include_optional=args.all, doc_ids=args.doc or None, force=args.force)
    print(json.dumps(report.as_dict(), indent=2))
    print(json.dumps(kb.stats(), indent=2))
    return 1 if report.failed else 0


def cmd_stats(args) -> int:
    kb = _kb(_settings_from(args))
    print(json.dumps({"stats": kb.stats(), "documents": kb.list_documents()}, indent=2))
    return 0


def cmd_search(args) -> int:
    from .retrieval import Retriever

    settings = _settings_from(args)
    r = Retriever(_kb(settings), settings)
    for c in r.search(args.query, mode=args.mode, k=args.top_k or 5):
        print(f"#{c.rank} {c.fused_score:.4f} dense={c.dense_score} bm25={c.bm25_score}  {c.title} > {c.section}")
        print("    " + c.text[:200].replace("\n", " ") + "...")
    return 0


def cmd_retrieve(args) -> int:
    from .retrieval import Retriever

    settings = _settings_from(args)
    r = Retriever(_kb(settings), settings)
    res = r.retrieve_for_sql(args.sql)
    out = res.as_dict(include_text=args.text)
    print(json.dumps(out, indent=2))
    return 0


def cmd_explain(args) -> int:
    from .explain import Explainer

    settings = _settings_from(args)
    explainer = Explainer.from_settings(settings)
    schema = open(args.schema_file, encoding="utf-8").read() if args.schema_file else (args.schema or "")
    result = explainer.explain(args.sql, schema, mode=args.mode_explain)
    print(json.dumps(result, indent=2))
    return 0


def cmd_retrieval_eval(args) -> int:
    from .evaluation.retrieval_eval import main as retrieval_main

    return retrieval_main(args)


def cmd_experiment(args) -> int:
    from .evaluation.experiment import main as experiment_main

    return experiment_main(args)


def cmd_summarize(args) -> int:
    from .evaluation.compare import main as compare_main

    return compare_main(args)


def cmd_agreement(args) -> int:
    from .evaluation.agreement import main as agreement_main

    return agreement_main(args)


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("sqlrag.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m sqlrag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_index_opts(p):
        p.add_argument("--embedding-backend", choices=["sentence-transformers", "hash", "ollama"])
        p.add_argument("--embedding-model")
        p.add_argument("--chunk-size", type=int)
        p.add_argument("--chunk-overlap", type=int)

    p = sub.add_parser("ingest", help="index the corpus (idempotent)")
    add_index_opts(p)
    p.add_argument("--all", action="store_true", help="also ingest optional manifest documents (e.g. sqlite-json)")
    p.add_argument("--doc", action="append", help="only this manifest doc_id (repeatable)")
    p.add_argument("--force", action="store_true", help="re-embed even if unchanged")
    p.add_argument("--rebuild", action="store_true", help="drop the collection first")
    p.add_argument("--files", nargs="*", help="ingest these local files instead of the manifest")
    p.add_argument("--doc-id")
    p.add_argument("--title")
    p.add_argument("--license", default="user-provided")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("stats", help="index statistics")
    add_index_opts(p)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="free-text retrieval")
    add_index_opts(p)
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"])
    p.add_argument("--top-k", type=int)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("retrieve", help="evidence retrieved for an SQL query")
    add_index_opts(p)
    p.add_argument("sql")
    p.add_argument("--mode", choices=["hybrid", "dense", "bm25"])
    p.add_argument("--top-k", type=int)
    p.add_argument("--text", action="store_true", help="include chunk text")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("explain", help="explain an SQL query")
    add_index_opts(p)
    p.add_argument("sql")
    p.add_argument("--schema", default="")
    p.add_argument("--schema-file")
    p.add_argument("--mode-explain", choices=["baseline", "rag"], default="rag")
    p.add_argument("--generator-model")
    p.add_argument("--policy", choices=["abstain", "fallback", "off"])
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("retrieval-eval", help="retrieval metrics and ablations")
    add_index_opts(p)
    p.add_argument("--split", default="all")
    p.add_argument("--ks", default="1,3,5,8")
    p.add_argument("--modes", default="dense,bm25,hybrid")
    p.add_argument("--chunk-sizes", default="", help="comma list for a chunk-size ablation, e.g. 200,300,400,500,600")
    p.add_argument("--out", default="results/rag/retrieval_eval.json")
    p.set_defaults(func=cmd_retrieval_eval)

    p = sub.add_parser("experiment", help="controlled baseline-vs-RAG experiment")
    add_index_opts(p)
    p.add_argument("split", choices=["tiny", "dev", "test", "all", "abstention_probe"])
    p.add_argument("exp_id")
    p.add_argument("--generator-model")
    p.add_argument("--judge-model")
    p.add_argument("--policy", choices=["abstain", "fallback", "off"])
    p.add_argument("--limit", type=int, help="only the first N items (smoke runs)")
    p.add_argument("--conditions", default="baseline,rag")
    p.add_argument("--judge-evidence", choices=["none", "retrieved"], default="none")
    p.add_argument("--judge-mode", choices=["rubric", "legacy", "legacy_schema"], default="rubric",
                   help="rubric = schema-aware rubric judge (new); legacy = original judge.md without schema; legacy_schema = original judge.md + schema")
    p.add_argument("--phase", choices=["all", "generate", "judge", "summarize"], default="all")
    p.add_argument("--no-pdf", action="store_true")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("summarize", help="recompute summary for an experiment directory")
    p.add_argument("exp_dir")
    p.add_argument("--no-pdf", action="store_true")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("agreement", help="judge agreement with reference labels")
    p.add_argument("--labels", default="data/labels")
    p.add_argument("--out", default="results/rag/judge_agreement.json")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("serve", help="run the API + dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "httpcore", "chromadb", "sentence_transformers", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
