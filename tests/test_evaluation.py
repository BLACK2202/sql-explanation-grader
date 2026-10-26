"""Statistics, experiment runner (checkpoint/resume), summaries, exports and judge agreement."""
from __future__ import annotations

import json

import pytest

from conftest import scripted_llm, write_jsonl
from sqlrag.evaluation import stats
from sqlrag.evaluation.agreement import compute_agreement, text_hash
from sqlrag.evaluation.compare import summarize_experiment
from sqlrag.evaluation.experiment import Experiment, export_legacy, read_jsonl
from sqlrag.explain import Explainer
from sqlrag.judge import Judge

SCHEMA = "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);\nCREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);"
ITEMS = [
    {"id": "t1", "feature": "LEFT JOIN with IS NULL filter", "domain": "retail", "schema": SCHEMA,
     "sql": "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL"},
    {"id": "t2", "feature": "GROUP BY with HAVING", "domain": "retail", "schema": SCHEMA,
     "sql": "SELECT city, COUNT(*) FROM customers GROUP BY city HAVING COUNT(*) > 2"},
    {"id": "t3", "feature": "JSON", "domain": "retail", "schema": SCHEMA, "should_abstain": True,
     "sql": "SELECT json_extract(name, '$.a') FROM customers"},
]


# ---------------------------------------------------------------- statistics
def test_wilson_interval():
    lo, hi = stats.wilson_interval(15, 50)
    assert lo == pytest.approx(0.1909, abs=1e-3) and hi == pytest.approx(0.4370, abs=1e-3)
    assert stats.wilson_interval(0, 0) == (None, None)


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) is None
    assert stats.mcnemar_exact(5, 5) == 1.0
    assert stats.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_paired_bootstrap_contains_mean():
    r = stats.paired_bootstrap([1, 0, 1, 1, 0, 1, 1, 1], iterations=500)
    assert r["ci_low"] <= r["mean"] <= r["ci_high"] and r["n"] == 8


def test_binary_classification_metrics():
    m = stats.binary_classification([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(0.6) and m["precision_good"] == pytest.approx(2 / 3)
    assert m["recall_good"] == pytest.approx(2 / 3) and m["f1_good"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"]["rows_true_cols_pred"] == [[2, 1], [1, 1]]
    assert m["cohen_kappa"] == pytest.approx(1 / 6)


# ---------------------------------------------------------------- experiment
def _explainer(settings, retriever, llm):
    return Explainer(settings, llm, retriever)


def test_experiment_generate_resume_judge_summarize(settings, retriever, tmp_path):
    exp = Experiment(settings.results_dir / "rag" / "exp1", settings)
    llm = scripted_llm()
    ex = _explainer(settings, retriever, llm)
    exp.generate(ITEMS[:2], ex)
    rows = read_jsonl(exp.gen_path)
    assert {(r["id"], r["condition"]) for r in rows} == {("t1", "baseline"), ("t1", "rag"), ("t2", "baseline"), ("t2", "rag")}
    assert [r["condition"] for r in rows[:2]] == ["baseline", "rag"] and [r["condition"] for r in rows[2:]] == ["rag", "baseline"]

    # Resume: only the new item is generated (plus one warm-up call).
    calls_before = len(llm.calls)
    exp.generate(ITEMS, ex)
    new_calls = [c for c in llm.calls[calls_before:] if "Say ok" not in c["user"]]
    assert len(new_calls) == 1  # t3 baseline only; t3 RAG abstains without an LLM call
    rows = read_jsonl(exp.gen_path)
    assert len(rows) == 6
    t3 = next(r for r in rows if r["id"] == "t3" and r["condition"] == "rag")
    assert t3["abstained"] and t3["should_abstain"] is True

    judge = Judge(settings, scripted_llm(grade="good"))
    jpath = exp.judge(ITEMS, judge)
    judged = read_jsonl(jpath)
    assert len(judged) == 6
    assert next(j for j in judged if j["id"] == "t3" and j["condition"] == "rag")["skipped"] == "abstained"
    assert len(read_jsonl(exp.judge(ITEMS, judge))) == 6  # judging again changes nothing

    summary = summarize_experiment(exp.dir, jpath)
    b, r = summary["conditions"]["baseline"], summary["conditions"]["rag"]
    assert b["judge_approval"]["k"] == 3 and b["judge_approval"]["n"] == 3
    assert r["n_abstained"] == 1 and r["judge_approval"]["n"] == 2 and r["judge_approval_itt"]["n"] == 3
    assert r["abstention_eval"]["abstained_correctly"] == 1 and r["abstention_eval"]["recall"] == 1.0
    assert summary["paired"]["n_pairs"] == 2 and summary["paired"]["outcomes"] == {"both_good": 2}
    assert summary["label_provenance"]["judge_approval"].startswith("LLM judge decision")
    assert (exp.dir / f"summary__{summary['judge_tag']}.json").exists()
    assert (exp.dir / f"items__{summary['judge_tag']}.csv").read_text(encoding="utf-8").startswith("id,condition")

    paths = export_legacy(exp.dir, jpath, "dev", settings.results_dir)
    legacy = read_jsonl(paths[0])
    assert set(legacy[0]) >= {"id", "input", "output", "score", "reason", "sys_cost", "judge_cost", "error"}

    from sqlrag.evaluation.report_pdf import write_experiment_pdf

    pdf = write_experiment_pdf(exp.dir, summary, settings.reports_dir)
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_generation_errors_are_recorded_and_retried(settings, retriever):
    def broken(model, system, user, schema):
        raise RuntimeError("model crashed")

    from sqlrag.llm import FakeLLMClient

    exp = Experiment(settings.results_dir / "rag" / "exp2", settings)
    exp.generate(ITEMS[:1], _explainer(settings, retriever, FakeLLMClient(broken)))
    rows = read_jsonl(exp.gen_path)
    assert all(r["error"] for r in rows)
    exp.generate(ITEMS[:1], _explainer(settings, retriever, scripted_llm()))  # failed rows are retried
    ok = [r for r in read_jsonl(exp.gen_path) if not r.get("error")]
    assert len(ok) == 2


def test_agreement_separates_label_sources_and_checks_hashes(settings, retriever, tmp_path):
    exp = Experiment(settings.results_dir / "rag" / "exp3", settings)
    exp.generate(ITEMS[:2], _explainer(settings, retriever, scripted_llm()))
    exp.judge(ITEMS[:2], Judge(settings, scripted_llm(grade="good")))
    gens = {(g["id"], g["condition"]): g for g in read_jsonl(exp.gen_path)}

    def lab(iid, cond, label, src, text=None):
        return {"exp_id": "exp3", "item_id": iid, "condition": cond, "label": label, "label_source": src,
                "explanation_sha256_12": text_hash(text if text is not None else gens[(iid, cond)]["explanation_clean"])}

    labels_dir = tmp_path / "labels"
    write_jsonl(labels_dir / "human.jsonl", [lab("t1", "baseline", "good", "human"), lab("t2", "baseline", "bad", "human")])
    write_jsonl(labels_dir / "ai.jsonl", [lab("t1", "rag", "good", "ai_reference"), lab("t2", "rag", "good", "ai_reference", text="other text")])
    rep = compute_agreement(labels_dir, settings.results_dir)
    human = rep["sources"]["human"]["judges"]
    tag = next(iter(human))
    assert human[tag]["n"] == 2 and human[tag]["accuracy"] == 0.5 and human[tag]["fp"] == 1
    ai = rep["sources"]["ai_reference"]
    assert ai["hash_mismatches"] == 1 and ai["judges"][tag]["n"] == 1
    assert rep["label_counts"] == {"human": 2, "ai_reference": 2}
    json.dumps(rep)  # serialisable
