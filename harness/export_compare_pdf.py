"""Generate a compact PDF report for benchmark-v3 A/B results."""
from __future__ import annotations

import argparse
import json
import pathlib
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

from metrics import summarize


def load(path: pathlib.Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def generate_pdf_report(result_path: pathlib.Path, output_path: pathlib.Path) -> pathlib.Path:
    rows = load(result_path)
    summary = summarize(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.5, leading=10))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=6.5, leading=8))

    story = [
        Paragraph("SQL Explanation Grader — Two-Model Benchmark", styles["Title"]),
        Paragraph(
            f"{summary.get('a_name', 'Model A')} vs {summary.get('b_name', 'Model B')} | "
            "judge pass rate, not true accuracy unless externally validated",
            styles["Small"],
        ),
        Spacer(1, 6),
    ]

    if not summary.get("n_valid"):
        story.append(Paragraph("No valid comparisons found.", styles["BodyText"]))
        doc.build(story)
        return output_path

    boot = summary["bootstrap"]
    summary_data = [
        ["Metric", summary["a_name"], summary["b_name"]],
        ["Judge pass rate", f"{summary['a_pass_rate_pct']:.1f}%", f"{summary['b_pass_rate_pct']:.1f}%"],
        ["Correctness", f"{summary['a_dimensions_pct']['correctness']:.1f}%", f"{summary['b_dimensions_pct']['correctness']:.1f}%"],
        ["Completeness", f"{summary['a_dimensions_pct']['completeness']:.1f}%", f"{summary['b_dimensions_pct']['completeness']:.1f}%"],
        ["Hallucination-free", f"{summary['a_dimensions_pct']['hallucination_free']:.1f}%", f"{summary['b_dimensions_pct']['hallucination_free']:.1f}%"],
        ["Clarity", f"{summary['a_dimensions_pct']['clarity']:.1f}%", f"{summary['b_dimensions_pct']['clarity']:.1f}%"],
        ["Avg latency", f"{summary['a_latency_avg_s']:.3f}s", f"{summary['b_latency_avg_s']:.3f}s"],
        ["Avg output tokens*", f"{summary['a_out_tok_avg']:.1f}", f"{summary['b_out_tok_avg']:.1f}"],
        ["Avg output tok/s", f"{summary['a_tok_per_s_avg']:.1f}", f"{summary['b_tok_per_s_avg']:.1f}"],
    ]
    t = Table(summary_data, colWidths=[55 * mm, 65 * mm, 65 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [
        t,
        Spacer(1, 6),
        Paragraph(
            f"Paired difference B−A: {boot['delta_pct']:+.1f} pp; "
            f"95% bootstrap CI [{boot['ci_low_pct']:+.1f}, {boot['ci_high_pct']:+.1f}]. "
            f"Exact McNemar p={summary['mcnemar_exact_p'] if summary['mcnemar_exact_p'] is not None else '—'}.",
            styles["Small"],
        ),
        Spacer(1, 8),
        Paragraph("Pair outcomes", styles["Heading2"]),
    ]

    po = [["Outcome", "Count"]] + [[k, summary["pair_counts"].get(k, 0)] for k in ("both_correct", "only_a_correct", "only_b_correct", "both_wrong", "evaluation_error")]
    pt = Table(po, colWidths=[70 * mm, 25 * mm])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .25, colors.grey),
    ]))
    story += [pt, Spacer(1, 8), Paragraph("SQL category performance", styles["Heading2"])]

    cat = [["Category", "N", summary["a_name"] + " pass", summary["b_name"] + " pass", "Disagree"]]
    for key, data in summary["category"].items():
        cat.append([key, data["n"], f"{data['a_pass_rate_pct']:.1f}%", f"{data['b_pass_rate_pct']:.1f}%", f"{data['disagreement_pct']:.1f}%"])
    ct = Table(cat, repeatRows=1, colWidths=[45 * mm, 15 * mm, 35 * mm, 35 * mm, 30 * mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), .25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(ct)

    disagreements = [r for r in rows if r.get("disagreement") and not r.get("error")]
    if disagreements:
        story.append(PageBreak())
        story.append(Paragraph("Disagreement cases", styles["Heading2"]))
        for r in disagreements[:30]:
            story.append(Paragraph(
                f"<b>{escape(str(r['id']))}</b> — {escape(str(r.get('outcome')))} — "
                f"{escape(', '.join(r.get('categories', [])) or 'basic SELECT')}",
                styles["Small"],
            ))
            story.append(Paragraph(f"SQL: {escape(r['sql'])}", styles["Tiny"]))
            story.append(Paragraph(
                f"A ({escape(r.get('model_a_name', summary['a_name']))}): {escape(r['model_a'].get('explanation', ''))}",
                styles["Tiny"],
            ))
            story.append(Paragraph(
                f"B ({escape(r.get('model_b_name', summary['b_name']))}): {escape(r['model_b'].get('explanation', ''))}",
                styles["Tiny"],
            ))
            story.append(Paragraph(
                f"Judge A: {escape(r['model_a'].get('judge_reason', ''))} | "
                f"Judge B: {escape(r['model_b'].get('judge_reason', ''))}",
                styles["Tiny"],
            ))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "* Token counts are model/tokenizer-specific and should not be treated as identical units across architectures.",
        styles["Tiny"],
    ))
    doc.build(story)
    return output_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("-o", "--output", type=pathlib.Path)
    args = ap.parse_args()
    out = args.output or args.results.with_name(args.results.stem + "_comparison.pdf")
    print(generate_pdf_report(args.results, out))
