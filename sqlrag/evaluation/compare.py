"""Summaries of a baseline-vs-RAG experiment.

Every number is computed from the stored rows; nothing is estimated or filled
in. Terminology used in the output:

  judge_approval       share of judged explanations the LLM judge graded
                       "good". An LLM decision, NOT verified accuracy.
  approval_itt         intention-to-treat: approvals / all items, so an
                       abstention counts as "not approved".
  reference agreement  how often the judge matches reference labels; see
                       agreement.py (kept separate on purpose).
"""
from __future__ import annotations

import csv
import json
import pathlib
from collections import Counter, defaultdict
from statistics import mean

from ..checks import run_checks
from ..sql_analysis import analyze_sql
from .experiment import CONDITIONS, read_jsonl
from .stats import describe, mcnemar_exact, paired_bootstrap, wilson_interval

DIMENSIONS = ("correctness", "completeness", "clarity", "groundedness")


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def summarize_experiment(exp_dir: pathlib.Path, judgments_path: pathlib.Path, *, write: bool = True) -> dict:
    exp_dir = pathlib.Path(exp_dir)
    meta = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8")) if (exp_dir / "metadata.json").exists() else {}
    gens_all = read_jsonl(exp_dir / "generations.jsonl")
    gens: dict[tuple[str, str], dict] = {}
    gen_errors: Counter = Counter()
    for g in gens_all:
        if g.get("error"):
            gen_errors[g["condition"]] += 1
            continue
        gens[(g["id"], g["condition"])] = g
    judg = {}
    judge_errors: Counter = Counter()
    for j in read_jsonl(judgments_path):
        if j.get("error"):
            judge_errors[j["condition"]] += 1
            continue
        judg[(j["id"], j["condition"])] = j
    tag = judgments_path.stem.split("__", 1)[1]
    item_ids = sorted({k[0] for k in gens})

    # Deterministic checks per generation (cheap; recomputed so they track checks.py)
    checks: dict[tuple[str, str], dict] = {}
    for key, g in gens.items():
        if g.get("explanation_clean"):
            checks[key] = run_checks(g["explanation_clean"], analyze_sql(g["sql"]))

    per_condition = {}
    for cond in CONDITIONS:
        keys = [k for k in gens if k[1] == cond]
        if not keys:
            continue
        rows = [gens[k] for k in keys]
        judged = [judg[k] for k in keys if k in judg and judg[k].get("score") is not None]
        abstained = [g for g in rows if g.get("abstained")]
        good = sum(j["score"] for j in judged)
        dims = {d: describe([j["dimensions"][d] for j in judged if j.get("dimensions")]) for d in DIMENSIONS}
        unsupported = [j.get("n_unsupported_claims") for j in judged if j.get("n_unsupported_claims") is not None]
        rubric = [j for j in judged if j.get("rubric_pass") is not None]
        consistent = [j for j in judged if j.get("grade_consistent") is not None]
        ch = [checks[k] for k in keys if k in checks]
        per_check: dict[str, list[bool]] = defaultdict(list)
        for c in ch:
            for name, ok in c["checks"].items():
                per_check[name].append(ok)
        metrics = [g.get("metrics") or {} for g in rows if not g.get("abstained")]
        statuses = Counter(g.get("grounding_status") for g in rows)
        cond_summary = {
            "n_items": len(rows),
            "generation_errors": gen_errors.get(cond, 0),
            "judge_errors": judge_errors.get(cond, 0),
            "n_judged": len(judged),
            "n_abstained": len(abstained),
            "abstention_rate": _rate(len(abstained), len(rows)),
            "judge_approval": _rate(good, len(judged)),
            "judge_approval_itt": _rate(good, len(rows)),
            "rubric_pass": _rate(sum(1 for j in rubric if j["rubric_pass"]), len(rubric)) if rubric else None,
            "grade_rubric_consistency": _rate(sum(1 for j in consistent if j["grade_consistent"]), len(consistent)) if consistent else None,
            "dimensions": dims,
            "unsupported_claims_per_explanation": describe(unsupported),
            "explanations_with_unsupported_claims": _rate(sum(1 for u in unsupported if u > 0), len(unsupported)) if unsupported else None,
            "deterministic_checks": {
                "coverage_score": describe([c["coverage_score"] for c in ch if c["coverage_score"] is not None]),
                "per_check_pass_rate": {k: _rate(sum(v), len(v)) for k, v in sorted(per_check.items())},
                "explanations_with_invented_numbers": _rate(sum(1 for c in ch if c["invented_numbers"]), len(ch)),
                "explanations_with_sql_keyword_leaks": _rate(sum(1 for c in ch if c["sql_keyword_leaks"]), len(ch)),
            },
            "latency_s": {
                "retrieval": describe([m.get("retrieval_s") for m in metrics]),
                "generation": describe([m.get("generation_s") for m in metrics]),
                "end_to_end": describe([m.get("total_s") for m in metrics]),
                "judge": describe([(j.get("cost") or {}).get("latency") for j in judged]),
            },
            "tokens": {
                "input": describe([m.get("in_tok") for m in metrics]),
                "output": describe([m.get("out_tok") for m in metrics]),
                "context": describe([m.get("context_tokens") for m in metrics if m.get("context_tokens") is not None]),
            },
            "explanation_words": describe([len((g.get("explanation_clean") or "").split()) for g in rows if g.get("explanation_clean")]),
            "grounding_status": dict(statuses),
        }
        if cond == "rag":
            reports = [g.get("citation_report") for g in rows if g.get("citation_report")]
            retr = [g.get("retrieval") for g in rows if g.get("retrieval")]
            cond_summary["grounding"] = {
                "answers_with_any_citation": _rate(sum(1 for r in reports if r["markers"]), len(reports)),
                "citation_validity": describe([r["citation_validity"] for r in reports if r.get("citation_validity") is not None]),
                "answers_with_invalid_citation": _rate(sum(1 for r in reports if r["invalid_markers"] or r["invalid_listed"]), len(reports)),
                "cited_sentence_share": describe([r["n_cited_sentences"] / r["n_sentences"] for r in reports if r.get("n_sentences")]),
                "lexical_claim_support_rate": describe([r["support_rate"] for r in reports if r.get("support_rate") is not None]),
                "model_reported_insufficient": sum(1 for g in rows if g.get("model_evidence_sufficient") is False),
            }
            labelled = [g for g in rows if g.get("should_abstain") is not None]
            if labelled:
                tp = sum(1 for g in labelled if g["should_abstain"] and g.get("abstained"))
                fn = sum(1 for g in labelled if g["should_abstain"] and not g.get("abstained"))
                fp = sum(1 for g in labelled if not g["should_abstain"] and g.get("abstained"))
                tn = sum(1 for g in labelled if not g["should_abstain"] and not g.get("abstained"))
                cond_summary["abstention_eval"] = {
                    "n": len(labelled),
                    "should_abstain": tp + fn,
                    "abstained_correctly": tp,
                    "answered_when_should_abstain": fn,
                    "abstained_when_should_answer": fp,
                    "answered_correctly": tn,
                    "precision": tp / (tp + fp) if (tp + fp) else None,
                    "recall": tp / (tp + fn) if (tp + fn) else None,
                    "by_stage": dict(Counter(g.get("abstention_stage") for g in labelled if g.get("abstained"))),
                }
            cond_summary["retrieval"] = {
                "feature_coverage": describe([r.get("coverage") for r in retr]),
                "evidence_sufficient": _rate(sum(1 for r in retr if r.get("sufficient")), len(retr)),
                "chunks_per_query": describe([len(r.get("chunks") or []) for r in retr]),
                "missing_features": dict(Counter(f for r in retr for f in (r.get("missing_features") or []))),
            }
        per_condition[cond] = cond_summary

    # Paired comparison on items where both conditions were judged.
    paired = {}
    both = [i for i in item_ids if (i, "baseline") in judg and (i, "rag") in judg
            and judg[(i, "baseline")].get("score") is not None and judg[(i, "rag")].get("score") is not None]
    if both:
        outcome = Counter()
        diffs = []
        for i in both:
            b, r = judg[(i, "baseline")]["score"], judg[(i, "rag")]["score"]
            outcome[{(1, 1): "both_good", (0, 0): "both_bad", (1, 0): "only_baseline_good", (0, 1): "only_rag_good"}[(b, r)]] += 1
            diffs.append(r - b)
        dim_deltas = {}
        for d in DIMENSIONS:
            dd = [judg[(i, "rag")]["dimensions"][d] - judg[(i, "baseline")]["dimensions"][d]
                  for i in both if judg[(i, "rag")].get("dimensions") and judg[(i, "baseline")].get("dimensions")]
            dim_deltas[d] = paired_bootstrap(dd) if dd else None
        cov = [checks[(i, "rag")]["coverage_score"] - checks[(i, "baseline")]["coverage_score"]
               for i in both if (i, "rag") in checks and (i, "baseline") in checks
               and checks[(i, "rag")]["coverage_score"] is not None and checks[(i, "baseline")]["coverage_score"] is not None]
        paired = {
            "n_pairs": len(both),
            "outcomes": dict(outcome),
            "approval_delta_rag_minus_baseline": paired_bootstrap([float(x) for x in diffs]),
            "mcnemar_exact_p": mcnemar_exact(outcome["only_baseline_good"], outcome["only_rag_good"]),
            "dimension_deltas": dim_deltas,
            "check_coverage_delta": paired_bootstrap(cov) if cov else None,
        }

    # Breakdown by the dataset's SQL feature label.
    by_feature: dict[str, dict] = {}
    feats = sorted({g.get("feature") or "unknown" for g in gens.values()})
    for f in feats:
        entry = {}
        for cond in CONDITIONS:
            ks = [k for k, g in gens.items() if k[1] == cond and (g.get("feature") or "unknown") == f]
            js = [judg[k] for k in ks if k in judg and judg[k].get("score") is not None]
            cs = [checks[k]["coverage_score"] for k in ks if k in checks and checks[k]["coverage_score"] is not None]
            entry[cond] = {
                "n": len(ks),
                "approval": _rate(sum(j["score"] for j in js), len(js)),
                "abstained": sum(1 for k in ks if gens[k].get("abstained")),
                "mean_correctness": mean([j["dimensions"]["correctness"] for j in js if j.get("dimensions")]) if any(j.get("dimensions") for j in js) else None,
                "check_coverage": mean(cs) if cs else None,
            }
        by_feature[f] = entry

    summary = {
        "exp_id": exp_dir.name,
        "judge_tag": tag,
        "judge": (meta.get("judges") or {}).get(tag),
        "dataset": meta.get("dataset"),
        "generator": meta.get("generator"),
        "label_provenance": {
            "judge_approval": "LLM judge decision (unverified; not ground truth)",
            "deterministic_checks": "rule-based heuristics computed from the SQL structure (no LLM)",
            "grounding": "deterministic citation validation; lexical support is a heuristic",
            "reference_labels": "see results/rag/judge_agreement.json; human and AI-assisted labels are reported separately",
        },
        "conditions": per_condition,
        "paired": paired,
        "by_feature": by_feature,
    }
    if write:
        (exp_dir / f"summary__{tag}.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        write_csv(exp_dir, tag, gens, judg, checks)
    return summary


def write_csv(exp_dir: pathlib.Path, tag: str, gens: dict, judg: dict, checks: dict) -> pathlib.Path:
    out = exp_dir / f"items__{tag}.csv"
    fields = ["id", "condition", "feature", "domain", "grounding_status", "abstained", "grade", "score", "correctness",
              "completeness", "clarity", "groundedness", "n_unsupported_claims", "check_coverage", "invented_numbers",
              "sql_keyword_leaks", "citation_validity", "lexical_support", "retrieval_coverage", "generation_s", "total_s",
              "in_tok", "out_tok", "judge_reason", "sql", "explanation"]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in sorted(gens):
            g, j, c = gens[key], judg.get(key, {}), checks.get(key, {})
            m = g.get("metrics") or {}
            dims = j.get("dimensions") or {}
            cr = g.get("citation_report") or {}
            w.writerow({
                "id": key[0], "condition": key[1], "feature": g.get("feature"), "domain": g.get("domain"),
                "grounding_status": g.get("grounding_status"), "abstained": g.get("abstained"), "grade": j.get("grade"),
                "score": j.get("score"), **{d: dims.get(d) for d in DIMENSIONS},
                "n_unsupported_claims": j.get("n_unsupported_claims"), "check_coverage": c.get("coverage_score"),
                "invented_numbers": " ".join(c.get("invented_numbers") or []),
                "sql_keyword_leaks": " ".join(c.get("sql_keyword_leaks") or []),
                "citation_validity": cr.get("citation_validity"), "lexical_support": cr.get("support_rate"),
                "retrieval_coverage": (g.get("retrieval") or {}).get("coverage"),
                "generation_s": m.get("generation_s"), "total_s": m.get("total_s"), "in_tok": m.get("in_tok"),
                "out_tok": m.get("out_tok"), "judge_reason": j.get("reason"), "sql": g.get("sql"),
                "explanation": g.get("explanation") or g.get("abstention_reason"),
            })
    return out


def main(args) -> int:
    exp_dir = pathlib.Path(args.exp_dir)
    for path in sorted(exp_dir.glob("judgments__*.jsonl")):
        summary = summarize_experiment(exp_dir, path)
        print(json.dumps({k: summary[k] for k in ("exp_id", "judge_tag", "paired")}, indent=2, default=str))
        if not getattr(args, "no_pdf", False):
            from ..config import get_settings
            from .report_pdf import write_experiment_pdf

            print("PDF:", write_experiment_pdf(exp_dir, summary, get_settings().reports_dir))
    return 0
