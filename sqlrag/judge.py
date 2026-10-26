"""Schema-aware rubric judge.

Fixes the original limitation: the judge now receives the database schema
(and an automatic inventory of the query's operations) in addition to the SQL
and the explanation. It returns 1-5 scores for correctness, completeness,
clarity and groundedness, a list of unsupported claims, and the legacy
good/bad grade so old and new runs remain comparable.

The judge is an LLM and can be wrong. Its decisions are reported as "judge
approval", never as ground truth; see sqlrag.evaluation.agreement for how it
is checked against reference labels.
"""
from __future__ import annotations

import pathlib
import re

from .config import ROOT, Settings
from .llm import LLMClient
from .prompting import judge_user_message, legacy_judge_user_message, load_prompt
from .retrieval import RetrievedChunk
from .schemas import RubricVerdict, Verdict
from .sql_analysis import SQLAnalysis, analyze_sql

DIMENSIONS = ("correctness", "completeness", "clarity", "groundedness")


def judge_tag(model: str, mode: str) -> str:
    """Filesystem-safe id of a judge configuration, e.g. qwen2.5-3b__rubric."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-") + "__" + mode


class LegacyJudge:
    """The original judge: prompts/judge.md, binary verdict, input WITHOUT the schema.

    Kept only to reproduce historical numbers and to measure what the schema
    fix changes. New experiments use `Judge` (mode "rubric").
    """

    def __init__(self, settings: Settings, client: LLMClient, model: str | None = None,
                 prompt_path: pathlib.Path | None = None, include_schema: bool = False) -> None:
        self.settings = settings
        self.client = client
        self.model = model or settings.judge_model
        self.include_schema = include_schema
        # "legacy" = original input; "legacy_schema" = original prompt + schema (harness/run.py default now)
        self.mode = "legacy_schema" if include_schema else "legacy"
        guide, self.prompt_hash = load_prompt(prompt_path or ROOT / "prompts" / "judge.md")
        self.system = f"{guide}\n\nGive a grade (good or bad) and a short reason."
        self.prompt_file = "judge.md"

    def grade(self, sql: str, schema: str, explanation: str, *, analysis: SQLAnalysis | None = None,
              evidence: list[RetrievedChunk] | None = None) -> dict:
        user = legacy_judge_user_message(sql, explanation)
        if self.include_schema:
            user = f"Schema:\n{schema}\n{user}"  # identical to harness/run.py judge_input()
        v, m = self.client.generate_json(self.model, self.system, user, Verdict, self.settings.generation_options())
        return {
            "score": int(v.grade == "good"), "grade": v.grade, "reason": v.reason,
            "dimensions": None, "unsupported_claims": None, "n_unsupported_claims": None,
            "rubric_pass": None, "grade_consistent": None,
            "judge_model": self.model, "judge_mode": self.mode, "judge_prompt": self.prompt_file,
            "judge_prompt_hash": self.prompt_hash, "judge_saw_schema": self.include_schema, "judge_saw_evidence": False,
            "cost": m,
        }


class Judge:
    mode = "rubric"

    def __init__(self, settings: Settings, client: LLMClient, model: str | None = None) -> None:
        self.settings = settings
        self.client = client
        self.model = model or settings.judge_model
        guide, self.prompt_hash = load_prompt(settings.judge_prompt)
        self.system = guide + "\n\nReturn JSON with correctness, completeness, clarity, groundedness, unsupported_claims, grade and reason."
        self.prompt_file = settings.judge_prompt.name

    def grade(self, sql: str, schema: str, explanation: str, *, analysis: SQLAnalysis | None = None,
              evidence: list[RetrievedChunk] | None = None) -> dict:
        analysis = analysis or analyze_sql(sql)
        user = judge_user_message(sql, schema, explanation, analysis.inventory(), evidence)
        v, m = self.client.generate_json(self.model, self.system, user, RubricVerdict, self.settings.generation_options())
        return {
            "score": int(v.grade == "good"),          # legacy binary metric
            "grade": v.grade,
            "reason": v.reason,
            "dimensions": {d: getattr(v, d) for d in DIMENSIONS},
            "unsupported_claims": v.unsupported_claims,
            "n_unsupported_claims": len(v.unsupported_claims),
            "rubric_pass": v.rubric_pass,
            "grade_consistent": (v.grade == "good") == v.rubric_pass,
            "judge_model": self.model,
            "judge_mode": self.mode,
            "judge_prompt": self.prompt_file,
            "judge_prompt_hash": self.prompt_hash,
            "judge_saw_schema": True,
            "judge_saw_evidence": bool(evidence),
            "cost": m,
        }


def make_judge(settings: Settings, client: LLMClient, model: str | None = None, mode: str = "rubric"):
    if mode == "rubric":
        return Judge(settings, client, model)
    if mode == "legacy":
        return LegacyJudge(settings, client, model)
    if mode == "legacy_schema":
        return LegacyJudge(settings, client, model, include_schema=True)
    raise ValueError(f"unknown judge mode {mode!r}")
