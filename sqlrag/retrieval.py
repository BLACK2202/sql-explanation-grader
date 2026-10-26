"""Hybrid retrieval: dense vectors + BM25, fused with Reciprocal Rank Fusion.

For an SQL query the retriever:
  1. analyses the SQL and builds one focused retrieval query per feature
     (plus a whole-query probe), see sql_analysis.build_retrieval_queries;
  2. runs each query through dense search (ChromaDB) and BM25;
  3. fuses the rankings with Reciprocal Rank Fusion (RRF):
         score(chunk) = sum over rankings of 1 / (rrf_k + rank)
     first across the two retrievers, then across the feature queries;
  4. drops candidates whose best dense similarity is below `min_similarity`
     (relevance filter);
  5. selects up to `top_k` chunks, first one covering chunk per feature, then
     the best remaining ones, within a `max_context_tokens` budget;
  6. assesses evidence sufficiency: the share of evidence-requiring features
     for which at least one selected chunk contains the feature's terms.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field

from .config import Settings
from .index.knowledge_base import KnowledgeBase
from .sql_analysis import FeatureSpec, RetrievalQuery, SQLAnalysis, analyze_sql, build_retrieval_queries


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    source_url: str
    license: str
    doc_type: str
    dialect: str
    text: str
    n_tokens: int
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fused_score: float = 0.0
    rank: int = 0
    matched_queries: list[str] = field(default_factory=list)
    covers: list[str] = field(default_factory=list)
    source_label: str = ""

    def as_dict(self, include_text: bool = True) -> dict:
        d = asdict(self)
        if not include_text:
            d.pop("text")
        return d


@dataclass
class RetrievalResult:
    analysis: SQLAnalysis
    queries: list[RetrievalQuery]
    chunks: list[RetrievedChunk]          # selected context, labelled S1..Sn
    candidates: list[RetrievedChunk]      # fused candidates before selection (for inspection)
    required_features: list[str]
    covered_features: list[str]
    missing_features: list[str]
    coverage: float
    sufficient: bool
    reasons: list[str]
    config: dict
    timings: dict

    def as_dict(self, include_text: bool = True) -> dict:
        return {
            "analysis": self.analysis.as_dict(),
            "queries": [{"feature": q.feature, "text": q.text} for q in self.queries],
            "chunks": [c.as_dict(include_text) for c in self.chunks],
            "candidates": [c.as_dict(False) for c in self.candidates[:20]],
            "required_features": self.required_features,
            "covered_features": self.covered_features,
            "missing_features": self.missing_features,
            "coverage": round(self.coverage, 3),
            "sufficient": self.sufficient,
            "reasons": self.reasons,
            "config": self.config,
            "timings": self.timings,
        }

    @property
    def context_tokens(self) -> int:
        return sum(c.n_tokens for c in self.chunks)


def _term_pattern(term: str) -> re.Pattern:
    """Keywords written in upper case match case-sensitively (LIMIT, CASE, NULL);
    other terms match case-insensitively. Alphanumeric edges get word boundaries."""
    flags = 0 if (term.strip().isupper() and any(c.isalpha() for c in term)) else re.IGNORECASE
    body = re.escape(term)
    if term[:1].isalnum() or term[:1] == "_":
        body = r"(?<![A-Za-z0-9_])" + body
    if term[-1:].isalnum() or term[-1:] == "_":
        body = body + r"(?![A-Za-z0-9_])"
    return re.compile(body, flags)


_PATTERN_CACHE: dict[str, re.Pattern] = {}


def chunk_covers(text: str, spec: FeatureSpec) -> bool:
    for term in spec.evidence_terms:
        pat = _PATTERN_CACHE.get(term)
        if pat is None:
            pat = _PATTERN_CACHE[term] = _term_pattern(term)
        if pat.search(text):
            return True
    return False


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class Retriever:
    def __init__(self, kb: KnowledgeBase, settings: Settings) -> None:
        self.kb = kb
        self.settings = settings

    # ------------------------------------------------------------------
    # Single free-text query
    # ------------------------------------------------------------------
    def search(self, query: str, *, mode: str | None = None, k: int | None = None, candidates: int | None = None,
               where: dict | None = None, query_vector: list[float] | None = None) -> list[RetrievedChunk]:
        mode = mode or self.settings.retrieval_mode
        k = k or self.settings.top_k
        n_cand = candidates or self.settings.candidates
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"unknown retrieval mode {mode!r}")
        chunks = self.kb.all_chunks()
        allowed = None
        if where:
            allowed = {h.chunk_id for h in self.kb.store.get_all(where=where)}
        results: dict[str, RetrievedChunk] = {}

        def entry(cid: str) -> RetrievedChunk:
            if cid not in results:
                h = chunks[cid]
                md = h.metadata
                results[cid] = RetrievedChunk(
                    chunk_id=cid, doc_id=md.get("doc_id", ""), title=md.get("title", ""), section=md.get("section", ""),
                    source_url=md.get("source_url", ""), license=md.get("license", ""), doc_type=md.get("doc_type", ""),
                    dialect=md.get("dialect", ""), text=h.text, n_tokens=int(md.get("n_tokens", 0)),
                )
            return results[cid]

        if mode in {"dense", "hybrid"}:
            qvec = query_vector if query_vector is not None else self.kb.embedder.embed_query(query)
            for rank, hit in enumerate(self.kb.store.query(qvec, n_cand, where=where), start=1):
                if hit.chunk_id not in chunks:
                    continue
                e = entry(hit.chunk_id)
                e.dense_rank, e.dense_score = rank, round(float(hit.similarity or 0.0), 4)
        if mode in {"bm25", "hybrid"}:
            for rank, (cid, score) in enumerate(self.kb.bm25().search(query, n_cand, allowed=allowed), start=1):
                e = entry(cid)
                e.bm25_rank, e.bm25_score = rank, round(score, 4)

        rrf_k = self.settings.rrf_k
        for e in results.values():
            if mode == "dense":
                e.fused_score = e.dense_score or 0.0
            elif mode == "bm25":
                e.fused_score = e.bm25_score or 0.0
            else:
                e.fused_score = sum(1.0 / (rrf_k + r) for r in (e.dense_rank, e.bm25_rank) if r is not None)
        ranked = sorted(results.values(), key=lambda e: (-e.fused_score, e.chunk_id))
        for i, e in enumerate(ranked, start=1):
            e.rank = i
        return ranked[:k]

    # ------------------------------------------------------------------
    # SQL-aware retrieval for explanation generation
    # ------------------------------------------------------------------
    def retrieve_for_sql(self, sql: str, *, mode: str | None = None, top_k: int | None = None,
                         min_similarity: float | None = None, max_context_tokens: int | None = None,
                         analysis: SQLAnalysis | None = None, strategy: str = "decomposed") -> RetrievalResult:
        """strategy: decomposed (one query per feature + probe, default) | probe (one
        feature-list query) | raw_sql (embed the SQL text itself; naive baseline)."""
        t0 = time.time()
        mode = mode or self.settings.retrieval_mode
        top_k = top_k or self.settings.top_k
        min_sim = self.settings.min_similarity if min_similarity is None else min_similarity
        budget = max_context_tokens or self.settings.max_context_tokens
        analysis = analysis or analyze_sql(sql)
        queries = build_retrieval_queries(analysis)
        if strategy == "probe":
            queries = [q for q in queries if q.feature is None]
        elif strategy == "raw_sql":
            queries = [RetrievalQuery(None, sql)]
        elif strategy != "decomposed":
            raise ValueError(f"unknown retrieval strategy {strategy!r}")
        per_query_k = self.settings.per_query_k or max(4, top_k)

        fused: dict[str, RetrievedChunk] = {}
        best_dense: dict[str, float] = {}
        query_vecs: dict[int, list[float]] = {}
        rrf_k = self.settings.rrf_k
        for qi, rq in enumerate(queries):
            if mode != "bm25":
                query_vecs[qi] = self.kb.embedder.embed_query(rq.text)
            hits = self.search(rq.text, mode=mode, k=per_query_k, query_vector=query_vecs.get(qi))
            for hit in hits:
                agg = fused.setdefault(hit.chunk_id, RetrievedChunk(**{**hit.__dict__, "matched_queries": [], "fused_score": 0.0}))
                agg.fused_score += 1.0 / (rrf_k + hit.rank)
                agg.matched_queries.append(rq.feature or "whole_query")
                if hit.dense_score is not None:
                    best_dense[hit.chunk_id] = max(best_dense.get(hit.chunk_id, -1.0), hit.dense_score)
                    agg.dense_score = max(agg.dense_score or -1.0, hit.dense_score)
                    agg.dense_rank = min(agg.dense_rank or 10**6, hit.dense_rank or 10**6)
                if hit.bm25_score is not None:
                    agg.bm25_score = max(agg.bm25_score or 0.0, hit.bm25_score)
                    agg.bm25_rank = min(agg.bm25_rank or 10**6, hit.bm25_rank or 10**6)
        t_search = time.time()

        # Relevance filter on dense similarity (BM25-only hits get their similarity computed).
        reasons: list[str] = []
        if mode != "bm25" and fused:
            missing = [cid for cid in fused if cid not in best_dense]
            if missing:
                embs = self.kb.store.get_embeddings(missing)
                for cid, emb in embs.items():
                    best_dense[cid] = max(_cos(qv, emb) for qv in query_vecs.values())
                    fused[cid].dense_score = round(best_dense[cid], 4)
            kept = {cid: c for cid, c in fused.items() if best_dense.get(cid, 0.0) >= min_sim}
            dropped = len(fused) - len(kept)
            if dropped:
                reasons.append(f"relevance filter removed {dropped} candidate(s) below similarity {min_sim}")
            fused = kept
        candidates = sorted(fused.values(), key=lambda c: (-c.fused_score, c.chunk_id))
        for i, c in enumerate(candidates, start=1):
            c.rank = i

        # Which features does each candidate cover (lexically)?
        required = [s for s in analysis.specs() if s.requires_evidence]
        for c in candidates:
            c.covers = [s.key for s in required if chunk_covers(c.text, s)]

        # Coverage-aware selection under the token budget.
        selected: list[RetrievedChunk] = []
        used_tokens = 0

        def try_add(c: RetrievedChunk) -> bool:
            nonlocal used_tokens
            if c in selected or len(selected) >= top_k or used_tokens + c.n_tokens > budget:
                return False
            selected.append(c)
            used_tokens += c.n_tokens
            return True

        for spec in required:
            if any(spec.key in c.covers for c in selected):
                continue
            own = [c for c in candidates if spec.key in c.covers and spec.key in c.matched_queries]
            other = [c for c in candidates if spec.key in c.covers and c not in own]
            for c in own + other:
                if try_add(c):
                    break
        if self.settings.fill_context or not selected:
            for c in candidates:
                if len(selected) >= top_k:
                    break
                try_add(c)
        selected.sort(key=lambda c: c.rank)
        for i, c in enumerate(selected, start=1):
            c.source_label = f"S{i}"

        covered = [s.key for s in required if any(s.key in c.covers for c in selected)]
        missing_feats = [s.key for s in required if s.key not in covered]
        coverage = (len(covered) / len(required)) if required else 1.0
        sufficient = coverage >= self.settings.min_feature_coverage and (bool(selected) or not required)
        if missing_feats:
            reasons.append("no retrieved passage explains: " + ", ".join(missing_feats))
        if required and not selected:
            reasons.append("no passage passed the relevance filter")

        t_end = time.time()
        return RetrievalResult(
            analysis=analysis,
            queries=queries,
            chunks=selected,
            candidates=candidates,
            required_features=[s.key for s in required],
            covered_features=covered,
            missing_features=missing_feats,
            coverage=coverage,
            sufficient=sufficient,
            reasons=reasons,
            config={
                "mode": mode, "top_k": top_k, "min_similarity": min_sim, "max_context_tokens": budget,
                "rrf_k": rrf_k, "per_query_k": per_query_k, "min_feature_coverage": self.settings.min_feature_coverage,
                "strategy": strategy, "fill_context": self.settings.fill_context,
                "embedding_model": self.kb.embedder.name, "collection": self.kb.collection_name,
            },
            timings={
                "search_s": round(t_search - t0, 4),
                "selection_s": round(t_end - t_search, 4),
                "total_s": round(t_end - t0, 4),
            },
        )
