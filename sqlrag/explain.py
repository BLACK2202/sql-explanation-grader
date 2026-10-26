"""Baseline and retrieval-augmented explanation generation.

Grounding status values (RAG mode):
  grounded              evidence covered every feature, the model used valid citations
  weakly_grounded       evidence covered the features, but the answer has no valid
                        citation or cites a source that was not provided
  abstained             the system refused because evidence was insufficient
                        (pre-generation coverage check, or the model reported it)
  fallback_ungrounded   evidence was insufficient and policy=fallback produced a
                        prompt-only answer; it is NOT evidence-grounded
  ungrounded_by_policy  evidence was insufficient but policy=off generated anyway
Baseline mode always reports `not_applicable`: it uses no retrieval.
"""
from __future__ import annotations

import time

from .config import Settings
from .grounding import validate_citations
from .llm import LLMClient, make_client
from .prompting import baseline_user_message, load_prompt, rag_user_message, strip_markers
from .retrieval import RetrievalResult, Retriever
from .schemas import Explanation, RAGExplanation
from .sql_analysis import analyze_sql

POLICIES = ("abstain", "fallback", "off")


class Explainer:
    def __init__(self, settings: Settings, client: LLMClient, retriever: Retriever | None) -> None:
        self.settings = settings
        self.client = client
        self.retriever = retriever
        self.baseline_prompt, self.baseline_hash = load_prompt(settings.baseline_prompt)
        self.rag_prompt, self.rag_hash = load_prompt(settings.rag_prompt)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Explainer":
        from .index.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(settings)
        return cls(settings, make_client(settings), Retriever(kb, settings))

    # ------------------------------------------------------------------
    def explain(self, sql: str, schema: str, *, mode: str = "rag", policy: str | None = None,
                model: str | None = None, retrieval: RetrievalResult | None = None) -> dict:
        if mode == "baseline":
            return self.explain_baseline(sql, schema, model=model)
        if mode == "rag":
            return self.explain_rag(sql, schema, policy=policy, model=model, retrieval=retrieval)
        raise ValueError(f"unknown mode {mode!r}")

    def explain_baseline(self, sql: str, schema: str, *, model: str | None = None) -> dict:
        model = model or self.settings.generator_model
        t0 = time.time()
        out, m = self.client.generate_json(model, self.baseline_prompt, baseline_user_message(sql, schema),
                                           Explanation, self.settings.generation_options())
        total = time.time() - t0
        return {
            "mode": "baseline",
            "explanation": out.explanation,
            "explanation_clean": out.explanation,
            "citations": [],
            "grounding_status": "not_applicable",
            "abstained": False,
            "abstention_reason": None,
            "prompt_file": str(self.settings.baseline_prompt.name),
            "prompt_hash": self.baseline_hash,
            "model": model,
            "metrics": {"retrieval_s": 0.0, "generation_s": m["latency"], "total_s": round(total, 3),
                        "in_tok": m["in_tok"], "out_tok": m["out_tok"], "attempts": m.get("attempts", 1)},
            "analysis": analyze_sql(sql).as_dict(),
            "retrieval": None,
            "citation_report": None,
        }

    def explain_rag(self, sql: str, schema: str, *, policy: str | None = None, model: str | None = None,
                    retrieval: RetrievalResult | None = None) -> dict:
        if self.retriever is None and retrieval is None:
            raise RuntimeError("RAG mode needs a retriever")
        policy = policy or self.settings.grounding_policy
        if policy not in POLICIES:
            raise ValueError(f"unknown grounding policy {policy!r}")
        model = model or self.settings.generator_model
        t0 = time.time()
        if retrieval is None:
            retrieval = self.retriever.retrieve_for_sql(sql)
        retrieval_s = retrieval.timings.get("total_s", round(time.time() - t0, 4))
        base = {
            "mode": "rag",
            "prompt_file": str(self.settings.rag_prompt.name),
            "prompt_hash": self.rag_hash,
            "model": model,
            "policy": policy,
            "analysis": retrieval.analysis.as_dict(),
            "retrieval": retrieval.as_dict(include_text=True),
        }

        if not retrieval.sufficient and policy != "off":
            reason = "; ".join(retrieval.reasons) or "retrieved evidence does not cover the query"
            if policy == "abstain":
                return {
                    **base,
                    "explanation": None,
                    "explanation_clean": None,
                    "citations": [],
                    "grounding_status": "abstained",
                    "abstained": True,
                    "abstention_stage": "retrieval",
                    "abstention_reason": f"Not enough supporting documentation to explain this query reliably ({reason}).",
                    "metrics": {"retrieval_s": retrieval_s, "generation_s": 0.0, "total_s": round(time.time() - t0, 3),
                                "in_tok": 0, "out_tok": 0, "attempts": 0},
                    "citation_report": None,
                }
            # policy == fallback: prompt-only answer, clearly labelled.
            fb = self.explain_baseline(sql, schema, model=model)
            return {
                **base,
                "explanation": fb["explanation"],
                "explanation_clean": fb["explanation"],
                "citations": [],
                "grounding_status": "fallback_ungrounded",
                "abstained": False,
                "abstention_reason": None,
                "fallback_reason": reason,
                "metrics": {**fb["metrics"], "retrieval_s": retrieval_s, "total_s": round(time.time() - t0, 3)},
                "citation_report": None,
            }

        user = rag_user_message(sql, schema, retrieval.chunks)
        out, m = self.client.generate_json(model, self.rag_prompt, user, RAGExplanation, self.settings.generation_options())
        report = validate_citations(out.explanation, out.citations, retrieval.chunks)
        metrics = {"retrieval_s": retrieval_s, "generation_s": m["latency"], "total_s": round(time.time() - t0, 3),
                   "in_tok": m["in_tok"], "out_tok": m["out_tok"], "attempts": m.get("attempts", 1),
                   "context_tokens": retrieval.context_tokens}
        result = {
            **base,
            "explanation": out.explanation,
            "explanation_clean": strip_markers(out.explanation),
            "citations": [c.model_dump() for c in out.citations],
            "model_evidence_sufficient": out.evidence_sufficient,
            "model_missing_evidence": out.missing_evidence,
            "abstained": False,
            "abstention_reason": None,
            "metrics": metrics,
            "citation_report": report.as_dict(),
        }
        if not retrieval.sufficient:  # policy == off
            result["grounding_status"] = "ungrounded_by_policy"
            return result
        if not out.evidence_sufficient and policy == "abstain":
            result.update({
                "draft_explanation": out.explanation,
                "explanation": None,
                "explanation_clean": None,
                "grounding_status": "abstained",
                "abstained": True,
                "abstention_stage": "generation",
                "abstention_reason": "The model reported that the sources do not support an explanation"
                                     + (f" ({out.missing_evidence})" if out.missing_evidence else "") + ".",
            })
            return result
        valid_cites = bool(report.markers) and not report.invalid_markers and not report.invalid_listed
        result["grounding_status"] = "grounded" if valid_cites else "weakly_grounded"
        return result
