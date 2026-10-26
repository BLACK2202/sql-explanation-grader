"""Comparative baseline-vs-RAG PDF report (reportlab, like harness/export_pdf.py)."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..config import ROOT
from .experiment import read_jsonl

INK = colors.HexColor("#1f2937")
MUTED = colors.HexColor("#6b7280")
LINE = colors.HexColor("#d1d5db")
BASE = colors.HexColor("#2563eb")
RAG = colors.HexColor("#0f766e")


def _pct(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    lo, hi = r.get("ci95", [None, None])
    ci = f" [{lo * 100:.0f}-{hi * 100:.0f}]" if lo is not None else ""
    return f"{r['rate'] * 100:.1f}% ({r['k']}/{r['n']}){ci}"


def _num(d: dict | None, key: str = "mean", digits: int = 2) -> str:
    if not d or d.get(key) is None:
        return "n/a"
    return f"{d[key]:.{digits}f}"


def _table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK)]
    t.setStyle(TableStyle(style))
    return t


def write_experiment_pdf(exp_dir: pathlib.Path, summary: dict, reports_dir: pathlib.Path) -> pathlib.Path:
    exp_dir = pathlib.Path(exp_dir)
    reports_dir = pathlib.Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"rag_comparison_{summary['exp_id']}__{summary['judge_tag']}.pdf"
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=18, textColor=INK, alignment=0, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, textColor=INK, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12.5, textColor=INK)
    small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10, textColor=MUTED)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=10)

    b = summary["conditions"].get("baseline", {})
    r = summary["conditions"].get("rag", {})
    gen = summary.get("generator") or {}
    judge = summary.get("judge") or {}
    ds = summary.get("dataset") or {}
    story = [
        Paragraph("SQL Explanation Grader: baseline vs RAG", h1),
        Paragraph(
            f"Experiment <b>{escape(summary['exp_id'])}</b> &middot; split <b>{escape(str(ds.get('split')))}</b> "
            f"({ds.get('n_items')} items) &middot; generator <b>{escape(str(gen.get('model')))}</b> &middot; "
            f"judge <b>{escape(str(judge.get('model')))}</b> ({escape(str(judge.get('mode')))}) &middot; "
            f"generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", small),
        Spacer(1, 6),
        Paragraph(
            "<b>How to read this report.</b> \"Judge approval\" is the share of explanations an LLM judge graded "
            "good. It is an automated, unverified measurement, not ground truth. Deterministic checks are rule-based "
            "heuristics computed from the SQL structure. Agreement of the judge with reference labels is reported "
            "separately (results/rag/judge_agreement.json).", body),
        Paragraph("Headline comparison", h2),
    ]
    rows = [["Metric", "Baseline (prompt only)", "RAG"]]
    rows += [
        ["Judge approval (95% CI)", _pct(b.get("judge_approval")), _pct(r.get("judge_approval"))],
        ["Approval, abstentions count as fail", _pct(b.get("judge_approval_itt")), _pct(r.get("judge_approval_itt"))],
        ["Rubric pass (judge's own scores)", _pct(b.get("rubric_pass")), _pct(r.get("rubric_pass"))],
        ["Mean correctness (1-5)", _num(b.get("dimensions", {}).get("correctness")), _num(r.get("dimensions", {}).get("correctness"))],
        ["Mean completeness (1-5)", _num(b.get("dimensions", {}).get("completeness")), _num(r.get("dimensions", {}).get("completeness"))],
        ["Mean clarity (1-5)", _num(b.get("dimensions", {}).get("clarity")), _num(r.get("dimensions", {}).get("clarity"))],
        ["Mean groundedness (1-5)", _num(b.get("dimensions", {}).get("groundedness")), _num(r.get("dimensions", {}).get("groundedness"))],
        ["Explanations with unsupported claims (judge)", _pct(b.get("explanations_with_unsupported_claims")), _pct(r.get("explanations_with_unsupported_claims"))],
        ["Deterministic check coverage", _num(b.get("deterministic_checks", {}).get("coverage_score")), _num(r.get("deterministic_checks", {}).get("coverage_score"))],
        ["Invented numbers (rule-based)", _pct(b.get("deterministic_checks", {}).get("explanations_with_invented_numbers")), _pct(r.get("deterministic_checks", {}).get("explanations_with_invented_numbers"))],
        ["Raw SQL keywords in text", _pct(b.get("deterministic_checks", {}).get("explanations_with_sql_keyword_leaks")), _pct(r.get("deterministic_checks", {}).get("explanations_with_sql_keyword_leaks"))],
        ["Abstentions", _pct(b.get("abstention_rate")), _pct(r.get("abstention_rate"))],
        ["Generation errors", str(b.get("generation_errors", 0)), str(r.get("generation_errors", 0))],
        ["Mean end-to-end latency (s)", _num(b.get("latency_s", {}).get("end_to_end"), digits=1), _num(r.get("latency_s", {}).get("end_to_end"), digits=1)],
        ["Mean retrieval time (s)", _num(b.get("latency_s", {}).get("retrieval"), digits=2), _num(r.get("latency_s", {}).get("retrieval"), digits=2)],
        ["Mean input tokens", _num(b.get("tokens", {}).get("input"), digits=0), _num(r.get("tokens", {}).get("input"), digits=0)],
        ["Mean output tokens", _num(b.get("tokens", {}).get("output"), digits=0), _num(r.get("tokens", {}).get("output"), digits=0)],
    ]
    story.append(_table(rows, [6.4 * cm, 5.3 * cm, 5.3 * cm]))

    try:
        from .agreement import load_labels, reference_comparison

        refs = reference_comparison(load_labels(ROOT / "data" / "labels"), ROOT / "results")
    except Exception:  # labels are optional
        refs = {}
    names = {"human": "Human labels", "ai_reference": "AI-assisted reference labels (not human)"}
    for src, by_exp in refs.items():
        rc = by_exp.get(summary["exp_id"])
        if not rc or not rc.get("n_pairs"):
            continue
        d = rc.get("delta_rag_minus_baseline") or {}
        cb, cr = rc["conditions"]["baseline"], rc["conditions"]["rag"]
        story += [
            Paragraph(f"{names.get(src, src)}: share labelled good", h2),
            _table([
                ["", "Baseline", "RAG"],
                ["Labelled good", f"{cb['good']}/{cb['n']}", f"{cr['good']}/{cr['n']}"],
                ["Paired: only baseline good / only RAG good", str(rc["only_baseline_good"]), str(rc["only_rag_good"])],
                ["RAG minus baseline (95% bootstrap CI)",
                 f"{(d.get('mean') or 0) * 100:+.1f} pts ({(d.get('ci_low') or 0) * 100:+.1f} to {(d.get('ci_high') or 0) * 100:+.1f})",
                 f"McNemar p = {rc['mcnemar_exact_p']:.3f}" if rc.get("mcnemar_exact_p") is not None else ""],
            ], [7 * cm, 5 * cm, 5 * cm]),
            Paragraph("Labels and their decision rules: data/labels/. They are not produced by the judge in this report.", small),
        ]

    p = summary.get("paired") or {}
    if p:
        d = p.get("approval_delta_rag_minus_baseline") or {}
        oc = p.get("outcomes", {})
        story += [
            Paragraph("Paired comparison (same items, both judged)", h2),
            Paragraph(
                f"{p.get('n_pairs')} pairs. Both good: {oc.get('both_good', 0)}; both bad: {oc.get('both_bad', 0)}; "
                f"only baseline good: {oc.get('only_baseline_good', 0)}; only RAG good: {oc.get('only_rag_good', 0)}. "
                f"Approval difference RAG minus baseline: {((d.get('mean') or 0) * 100):+.1f} points "
                f"(95% bootstrap CI {((d.get('ci_low') or 0) * 100):+.1f} to {((d.get('ci_high') or 0) * 100):+.1f}). "
                f"Exact McNemar p = {p.get('mcnemar_exact_p') if p.get('mcnemar_exact_p') is None else round(p['mcnemar_exact_p'], 4)}.",
                body),
        ]
        drows = [["Dimension", "Mean RAG minus baseline", "95% CI"]]
        for dim, v in (p.get("dimension_deltas") or {}).items():
            if v:
                drows.append([dim, f"{v['mean']:+.2f}", f"{v['ci_low']:+.2f} to {v['ci_high']:+.2f}"])
        if len(drows) > 1:
            story.append(_table(drows, [5 * cm, 5 * cm, 5 * cm]))

    g = r.get("grounding")
    if g:
        rr = r.get("retrieval", {})
        story += [
            Paragraph("Grounding and retrieval (RAG condition)", h2),
            _table([
                ["Metric", "Value"],
                ["Answers with at least one citation", _pct(g.get("answers_with_any_citation"))],
                ["Citation validity (cited source was provided)", _num(g.get("citation_validity"), digits=3)],
                ["Answers citing a source that was not provided", _pct(g.get("answers_with_invalid_citation"))],
                ["Share of sentences carrying a citation", _num(g.get("cited_sentence_share"), digits=3)],
                ["Lexical support of cited sentences (heuristic)", _num(g.get("lexical_claim_support_rate"), digits=3)],
                ["Model said evidence was insufficient", str(g.get("model_reported_insufficient"))],
                ["Retrieval feature coverage", _num(rr.get("feature_coverage"), digits=3)],
                ["Evidence judged sufficient before generation", _pct(rr.get("evidence_sufficient"))],
                ["Grounding status counts", escape(json.dumps(r.get("grounding_status", {})))],
            ], [8 * cm, 8 * cm]),
        ]

    retr_path = ROOT / "results" / "rag" / "retrieval_eval.json"
    if retr_path.exists():
        re_ = json.loads(retr_path.read_text(encoding="utf-8"))
        rows = [["Chunk", "Strategy", "Mode", "MRR", "Recall@5", "Prec@5", "nDCG@5", "Ctx recall", "ms/query"]]
        for run in re_.get("runs", []):
            k5 = run["at_k"].get("5", {})
            rows.append([run["chunk_size"], run["strategy"], run["mode"], f"{run['mrr']:.3f}", f"{k5.get('feature_recall', 0):.3f}",
                         f"{k5.get('precision', 0):.3f}", f"{k5.get('ndcg', 0):.3f}", f"{run['context_feature_recall']:.3f}",
                         f"{run['latency_s']['mean'] * 1000:.0f}"])
        story += [Paragraph("Retrieval evaluation (no LLM involved)", h2),
                  Paragraph(escape(re_.get("qrels_provenance", "")), small),
                  _table(rows, [1.3 * cm, 2.2 * cm, 1.6 * cm, 1.5 * cm, 1.8 * cm, 1.6 * cm, 1.7 * cm, 1.9 * cm, 1.8 * cm])]

    story += [PageBreak(), Paragraph("Breakdown by SQL feature (dataset label)", h2)]
    rows = [["Feature", "n", "Baseline approval", "RAG approval", "Base checks", "RAG checks"]]
    for feat, e in sorted((summary.get("by_feature") or {}).items()):
        be, re2 = e.get("baseline", {}), e.get("rag", {})
        rows.append([Paragraph(escape(feat), cell), be.get("n"), _pct(be.get("approval")), _pct(re2.get("approval")),
                     "n/a" if be.get("check_coverage") is None else f"{be['check_coverage']:.2f}",
                     "n/a" if re2.get("check_coverage") is None else f"{re2['check_coverage']:.2f}"])
    story.append(_table(rows, [5.2 * cm, 0.9 * cm, 3.4 * cm, 3.4 * cm, 1.8 * cm, 1.8 * cm]))

    # Disagreements for failure inspection.
    gens = {(x["id"], x["condition"]): x for x in read_jsonl(exp_dir / "generations.jsonl") if not x.get("error")}
    judg = {(x["id"], x["condition"]): x for x in read_jsonl(exp_dir / f"judgments__{summary['judge_tag']}.jsonl") if x.get("score") is not None}
    disagreements = [i for (i, c) in judg if c == "baseline" and (i, "rag") in judg and judg[(i, "baseline")]["score"] != judg[(i, "rag")]["score"]]
    if disagreements:
        story += [Paragraph("Items where the two conditions were graded differently", h2)]
        for iid in sorted(disagreements)[:12]:
            gb, gr = gens.get((iid, "baseline"), {}), gens.get((iid, "rag"), {})
            jb, jr = judg[(iid, "baseline")], judg[(iid, "rag")]
            story += [
                Paragraph(f"<b>{escape(iid)}</b> &middot; {escape(str(gb.get('feature')))}<br/><font face='Courier' size='7.5'>{escape(gb.get('sql', ''))}</font>", cell),
                _table([
                    [Paragraph(f"<b>Baseline: {jb['grade']}</b><br/>{escape(gb.get('explanation_clean') or '')}<br/><i>Judge: {escape(jb.get('reason') or '')}</i>", cell),
                     Paragraph(f"<b>RAG: {jr['grade']}</b><br/>{escape(gr.get('explanation') or '')}<br/><i>Judge: {escape(jr.get('reason') or '')}</i>", cell)],
                ], [8.5 * cm, 8.5 * cm], header=False),
                Spacer(1, 6),
            ]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(2 * cm, 1.2 * cm, "SQL Explanation Grader: RAG evaluation (LLM-judge results are unverified measurements)")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=f"RAG comparison {summary['exp_id']}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out
