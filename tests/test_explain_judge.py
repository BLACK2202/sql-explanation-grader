"""Generation policies (grounded / abstain / fallback / off), judge and deterministic checks."""
from __future__ import annotations

import pytest

from conftest import scripted_llm
from sqlrag.checks import run_checks
from sqlrag.explain import Explainer
from sqlrag.judge import Judge, LegacyJudge, judge_tag, make_judge
from sqlrag.schemas import RubricVerdict
from sqlrag.sql_analysis import analyze_sql

SCHEMA = "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);\nCREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);"
COVERED = "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL ORDER BY c.name"
UNCOVERED = "SELECT json_extract(name, '$.a') FROM customers"


def test_baseline_uses_original_prompt_and_message(settings, retriever, fake_llm):
    ex = Explainer(settings, fake_llm, retriever)
    r = ex.explain(COVERED, SCHEMA, mode="baseline")
    call = fake_llm.calls[-1]
    assert call["schema"] == "Explanation" and call["user"] == f"Schema:\n{SCHEMA}\nSQL:\n{COVERED}"
    assert call["system"] == settings.baseline_prompt.read_text(encoding="utf-8")
    assert call["options"] == settings.generation_options()
    assert r["grounding_status"] == "not_applicable" and r["retrieval"] is None


def test_rag_grounded_answer_with_valid_citation(settings, retriever, fake_llm):
    r = Explainer(settings, fake_llm, retriever).explain(COVERED, SCHEMA, mode="rag")
    assert r["grounding_status"] == "grounded"
    assert r["citation_report"]["citation_validity"] == 1.0
    assert "[S" not in r["explanation_clean"] and "[S" in r["explanation"]
    assert fake_llm.calls[-1]["options"] == settings.generation_options()  # same settings as baseline


def test_abstain_policy_skips_the_llm(settings, retriever, fake_llm):
    r = Explainer(settings, fake_llm, retriever).explain(UNCOVERED, SCHEMA, mode="rag", policy="abstain")
    assert r["abstained"] and r["grounding_status"] == "abstained" and r["abstention_stage"] == "retrieval"
    assert r["explanation"] is None and "json_extract" in r["abstention_reason"]
    assert fake_llm.calls == []


def test_fallback_policy_is_labelled_ungrounded(settings, retriever, fake_llm):
    r = Explainer(settings, fake_llm, retriever).explain(UNCOVERED, SCHEMA, mode="rag", policy="fallback")
    assert r["grounding_status"] == "fallback_ungrounded" and not r["abstained"]
    assert r["citations"] == [] and r["explanation"]
    assert fake_llm.calls[-1]["schema"] == "Explanation"


def test_off_policy_generates_but_says_ungrounded(settings, retriever, fake_llm):
    r = Explainer(settings, fake_llm, retriever).explain(UNCOVERED, SCHEMA, mode="rag", policy="off")
    assert r["grounding_status"] == "ungrounded_by_policy"


def test_model_reported_insufficient_evidence_abstains(settings, retriever):
    llm = scripted_llm(sufficient=False)
    r = Explainer(settings, llm, retriever).explain(COVERED, SCHEMA, mode="rag", policy="abstain")
    assert r["abstained"] and r["abstention_stage"] == "generation" and r["draft_explanation"]


def test_invalid_citation_is_weakly_grounded(settings, retriever):
    llm = scripted_llm(rag_text="Customers without orders are listed. [S9]")
    r = Explainer(settings, llm, retriever).explain(COVERED, SCHEMA, mode="rag")
    assert r["grounding_status"] == "weakly_grounded"
    assert r["citation_report"]["invalid_markers"] == ["S9"]


def test_unknown_policy_rejected(settings, retriever, fake_llm):
    with pytest.raises(ValueError):
        Explainer(settings, fake_llm, retriever).explain(COVERED, SCHEMA, mode="rag", policy="maybe")


def test_rubric_judge_receives_schema_and_inventory(settings, fake_llm):
    out = Judge(settings, fake_llm).grade(COVERED, SCHEMA, "Lists customers with no orders.")
    user = fake_llm.calls[-1]["user"]
    assert "Database schema:" in user and "CREATE TABLE orders" in user and "LEFT JOIN with orders" in user
    assert out["score"] == 1 and out["dimensions"]["correctness"] == 5 and out["grade_consistent"]
    assert out["judge_saw_schema"] is True


def test_legacy_judge_reproduces_old_input(settings, fake_llm):
    out = LegacyJudge(settings, fake_llm).grade(COVERED, SCHEMA, "Lists customers.")
    user = fake_llm.calls[-1]["user"]
    assert "CREATE TABLE" not in user and user.startswith("SQL:\n")
    assert out["judge_saw_schema"] is False and out["dimensions"] is None


def test_legacy_schema_judge_matches_run_py_input(settings, fake_llm):
    out = make_judge(settings, fake_llm, mode="legacy_schema").grade(COVERED, SCHEMA, "Lists customers.")
    user = fake_llm.calls[-1]["user"]
    assert user == f"Schema:\n{SCHEMA}\nSQL:\n{COVERED}\nExplanation:\nLists customers."
    assert out["judge_saw_schema"] is True and out["judge_mode"] == "legacy_schema"


def test_bad_grade_and_consistency(settings):
    out = Judge(settings, scripted_llm(grade="bad")).grade(COVERED, SCHEMA, "Wrong.")
    assert out["score"] == 0 and out["n_unsupported_claims"] == 1 and out["grade_consistent"]


def test_rubric_scores_are_clamped():
    v = RubricVerdict(correctness=9, completeness="0", clarity=3.6, groundedness=None, grade="bad", reason="clamping test")
    assert (v.correctness, v.completeness, v.clarity, v.groundedness) == (5, 1, 4, 1)
    assert not v.rubric_pass


def test_judge_factory_and_tags(settings, fake_llm):
    assert isinstance(make_judge(settings, fake_llm, mode="legacy"), LegacyJudge)
    with pytest.raises(ValueError):
        make_judge(settings, fake_llm, mode="other")
    assert judge_tag("qwen2.5:7b", "rubric") == "qwen2.5-7b__rubric"


def test_deterministic_checks():
    a = analyze_sql("SELECT city, COUNT(*) FROM customers GROUP BY city ORDER BY COUNT(*) DESC LIMIT 3")
    good = run_checks("For each city this counts the customers and shows the three cities with the most customers.", a)
    assert good["checks"] == {"limit_value": True, "order_direction": True, "aggregate_count": True, "grouping": True, "tables": True}
    bad = run_checks("This lists 25 cities alphabetically using GROUP BY.", a)
    assert bad["checks"]["limit_value"] is False and bad["checks"]["order_direction"] is False
    assert bad["invented_numbers"] == ["25"] and bad["sql_keyword_leaks"] == ["GROUP BY"]
    assert 0 <= bad["coverage_score"] < good["coverage_score"] == 1.0


def test_null_filter_check():
    a = analyze_sql("SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL")
    assert run_checks("Customers who have never placed an order.", a)["checks"]["null_filter"]
    assert not run_checks("Customers and all their orders.", a)["checks"]["null_filter"]
