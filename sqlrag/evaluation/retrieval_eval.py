"""Retrieval evaluation, independent of any LLM.

For each SQL item, the relevant chunks are those whose (document, section)
matches data/retrieval_qrels.json for at least one of the item's
evidence-requiring features. Metrics:

  feature_recall@k  share of the item's features with >= 1 relevant chunk in the top k
  precision@k       share of the top-k chunks that are relevant to some feature
  mrr               1 / rank of the first relevant chunk (0 if none in the candidate list)
  ndcg@k            binary-relevance nDCG
  context_feature_recall  feature recall of the final context the generator receives
  latency           retrieval wall time per SQL item

Ablations: retrieval mode (dense / bm25 / hybrid), query strategy (decomposed /
probe / raw_sql) and chunk size (each size has its own collection).
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
import re
import time
from statistics import mean

from ..config import ROOT, get_settings
from ..index.knowledge_base import KnowledgeBase
from ..retrieval import RetrievedChunk, Retriever
from ..sql_analysis import DATE_FUNCS, analyze_sql
from .stats import describe

logger = logging.getLogger(__name__)


class Qrels:
    def __init__(self, path: pathlib.Path = ROOT / "data" / "retrieval_qrels.json") -> None:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        self.provenance = raw.get("provenance")
        self.features = {k: [(d, re.compile(rx)) for d, rx in v] for k, v in raw["features"].items()}
        self.families = {k: [(d, re.compile(rx)) for d, rx in v] for k, v in raw["function_families"].items()}

    def rules_for(self, feature: str) -> list[tuple[str, re.Pattern]] | None:
        if feature in self.features:
            return self.features[feature]
        if feature.startswith("fn:"):
            name = feature[3:]
            fam = "date" if name in DATE_FUNCS else "json" if name.startswith("json") else "scalar"
            return self.families[fam]
        return None

    def relevant(self, chunk: RetrievedChunk, feature: str) -> bool:
        rules = self.rules_for(feature) or []
        sections = (chunk.section or "").split(" / ")
        return any(chunk.doc_id == d and any(rx.search(s) for s in sections) for d, rx in rules)


def _ndcg(rels: list[int], k: int, n_relevant_total: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, n_relevant_total)))
    return dcg / ideal if ideal else 0.0


def evaluate(retriever: Retriever, items: list[dict], qrels: Qrels, *, mode: str, ks: list[int],
             strategy: str = "decomposed") -> dict:
    all_chunks = list(retriever.kb.all_chunks().values())
    per_k = {k: {"feature_recall": [], "precision": [], "ndcg": []} for k in ks}
    mrr, ctx_recall, latency = [], [], []
    evaluated = 0
    for it in items:
        analysis = analyze_sql(it["sql"])
        feats = [s.key for s in analysis.specs() if s.requires_evidence and qrels.rules_for(s.key)]
        if not feats:
            continue
        evaluated += 1
        t = time.time()
        res = retriever.retrieve_for_sql(it["sql"], mode=mode, analysis=analysis, strategy=strategy,
                                         top_k=max(ks), max_context_tokens=10**6)
        latency.append(time.time() - t)
        ranked = res.candidates
        n_rel_total = 0
        for h in all_chunks:
            rc = RetrievedChunk(chunk_id=h.chunk_id, doc_id=h.metadata.get("doc_id", ""), title="", section=h.metadata.get("section", ""),
                                source_url="", license="", doc_type="", dialect="", text="", n_tokens=0)
            if any(qrels.relevant(rc, f) for f in feats):
                n_rel_total += 1
        rel_flags = [int(any(qrels.relevant(c, f) for f in feats)) for c in ranked]
        first = next((i for i, r in enumerate(rel_flags) if r), None)
        mrr.append(1 / (first + 1) if first is not None else 0.0)
        for k in ks:
            top = ranked[:k]
            covered = [f for f in feats if any(qrels.relevant(c, f) for c in top)]
            per_k[k]["feature_recall"].append(len(covered) / len(feats))
            per_k[k]["precision"].append(sum(rel_flags[:k]) / k)
            per_k[k]["ndcg"].append(_ndcg(rel_flags, k, n_rel_total))
        default_k = retriever.settings.top_k
        ctx = retriever.retrieve_for_sql(it["sql"], mode=mode, analysis=analysis, strategy=strategy, top_k=default_k)
        ctx_recall.append(sum(1 for f in feats if any(qrels.relevant(c, f) for c in ctx.chunks)) / len(feats))
    return {
        "mode": mode,
        "strategy": strategy,
        "n_items_evaluated": evaluated,
        "mrr": mean(mrr) if mrr else None,
        "at_k": {str(k): {m: (mean(v) if v else None) for m, v in d.items()} for k, d in per_k.items()},
        "context_feature_recall": mean(ctx_recall) if ctx_recall else None,
        "latency_s": describe(latency),
    }


def main(args) -> int:
    logging.getLogger("sqlrag").setLevel(logging.INFO)
    settings = get_settings(embedding_backend=getattr(args, "embedding_backend", None),
                            embedding_model=getattr(args, "embedding_model", None))
    items = [json.loads(line) for line in (settings.data_dir / f"{args.split}.jsonl").open(encoding="utf-8") if line.strip()]
    ks = [int(k) for k in args.ks.split(",")]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    qrels = Qrels()
    chunk_sizes = [int(c) for c in args.chunk_sizes.split(",") if c.strip()] or [args.chunk_size or settings.chunk_size]
    report = {"split": args.split, "n_items": len(items), "ks": ks, "qrels_provenance": qrels.provenance, "runs": []}
    t_all = time.time()
    for cs in chunk_sizes:
        s = get_settings(embedding_backend=settings.embedding_backend, embedding_model=settings.embedding_model, chunk_size=cs,
                         chunk_overlap=min(settings.chunk_overlap, cs // 4))
        kb = KnowledgeBase(s)
        ing = kb.ingest_manifest()
        stats = kb.stats()
        retriever = Retriever(kb, s)
        logger.info("chunk_size=%d: %d chunks (%s)", cs, stats["chunks"], {k: len(v) for k, v in ing.as_dict().items() if isinstance(v, list)})
        strategies = ["decomposed", "probe", "raw_sql"] if cs == settings.chunk_size else ["decomposed"]
        for strategy in strategies:
            for mode in modes:
                if strategy != "decomposed" and mode != "hybrid":
                    continue
                t = time.time()
                r = evaluate(retriever, items, qrels, mode=mode, ks=ks, strategy=strategy)
                r.update({"chunk_size": cs, "chunk_overlap": s.chunk_overlap, "n_chunks": stats["chunks"],
                          "avg_chunk_tokens": stats["avg_chunk_tokens"], "chunks_over_embedding_window": stats["chunks_over_embedding_window"],
                          "seconds": round(time.time() - t, 1)})
                report["runs"].append(r)
                logger.info("cs=%d %s/%s: MRR=%.3f R@5=%.3f P@5=%.3f ctxR=%.3f", cs, strategy, mode, r["mrr"] or 0,
                            r["at_k"].get("5", {}).get("feature_recall") or 0, r["at_k"].get("5", {}).get("precision") or 0,
                            r["context_feature_recall"] or 0)
    report["total_seconds"] = round(time.time() - t_all, 1)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0
