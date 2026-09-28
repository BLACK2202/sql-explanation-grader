"""
harness/export_pdf.py — Generate comprehensive, publication-ready PDF evaluation reports.

Usage:
    python harness/export_pdf.py results/v2_dev.jsonl
    python harness/export_pdf.py results/v2_dev.jsonl -o reports/v2_dev_report.pdf
"""

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime
from typing import Dict, List, Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError:
    print("Error: reportlab is required for PDF generation. Install with: pip install reportlab")
    sys.exit(1)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(54, 750, "SQL Explanation Grader — Evaluation Report")
            self.drawRightString(612 - 54, 750, datetime.now().strftime("%Y-%m-%d %H:%M"))
            self.setStrokeColor(colors.HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)

        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 32, footer_text)
        self.drawString(54, 32, "Confidential — Automated Evaluation Pipeline")
        self.restoreState()


def load_dataset_metadata(data_dir: pathlib.Path) -> Dict[str, dict]:
    """Load metadata (domain, feature, schema) from data splits keyed by ID."""
    meta = {}
    if not data_dir.exists():
        return meta

    for jsonl_file in data_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    item_id = item.get("id")
                    if item_id:
                        meta[item_id] = item
        except Exception:
            pass
    return meta


def generate_pdf_report(
    results_path: pathlib.Path,
    output_pdf_path: pathlib.Path,
    data_dir: pathlib.Path = pathlib.Path("data"),
) -> pathlib.Path:
    """Generate a rich PDF evaluation report from a results JSONL file."""
    results_path = pathlib.Path(results_path)
    output_pdf_path = pathlib.Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Read results
    items: List[dict] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    metadata = load_dataset_metadata(data_dir)

    # Aggregate statistics
    total = len(items)
    passed = sum(1 for i in items if i.get("score") == 1)
    failed = sum(1 for i in items if i.get("score") == 0 and not i.get("error"))
    errors = sum(1 for i in items if i.get("error") or i.get("score") is None)
    pass_rate = (passed / total * 100) if total > 0 else 0.0

    # Latency & token costs
    gen_latencies = [i.get("sys_cost", {}).get("latency", 0) for i in items if i.get("sys_cost")]
    judge_latencies = [i.get("judge_cost", {}).get("latency", 0) for i in items if i.get("judge_cost")]
    avg_gen_lat = sum(gen_latencies) / len(gen_latencies) if gen_latencies else 0.0
    avg_judge_lat = sum(judge_latencies) / len(judge_latencies) if judge_latencies else 0.0

    # Feature breakdown
    features: Dict[str, Dict[str, int]] = {}
    domains: Dict[str, Dict[str, int]] = {}

    for i in items:
        item_id = i.get("id", "")
        meta_item = metadata.get(item_id, {})
        feat = meta_item.get("feature", "General / Standard")
        dom = meta_item.get("domain", "General")
        is_pass = i.get("score") == 1

        if feat not in features:
            features[feat] = {"total": 0, "pass": 0}
        features[feat]["total"] += 1
        if is_pass:
            features[feat]["pass"] += 1

        if dom not in domains:
            domains[dom] = {"total": 0, "pass": 0}
        domains[dom]["total"] += 1
        if is_pass:
            domains[dom]["pass"] += 1

    # Document setup
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6,
    )
    card_label = ParagraphStyle(
        "CardLabel",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=1,  # Center
        textColor=colors.HexColor("#64748b"),
    )
    card_val_pass = ParagraphStyle(
        "CardValPass",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#059669"),
    )
    card_val_fail = ParagraphStyle(
        "CardValFail",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#dc2626"),
    )
    card_val_neutral = ParagraphStyle(
        "CardValNeutral",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#0f172a"),
    )
    table_hdr = ParagraphStyle(
        "TableHdr",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1e293b"),
    )
    table_cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
    )
    code_text = ParagraphStyle(
        "CodeText",
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#0f172a"),
    )
    badge_pass = ParagraphStyle(
        "BadgePass",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#065f46"),
    )
    badge_fail = ParagraphStyle(
        "BadgeFail",
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#991b1b"),
    )

    story = []

    # 1. Title Banner
    run_name = results_path.stem
    story.append(Paragraph("SQL Explanation Grader — Benchmark Report", title_style))
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            f"<b>Run ID:</b> {run_name} &nbsp;|&nbsp; <b>Evaluated At:</b> {datetime.now().strftime('%B %d, %Y - %H:%M:%S')} &nbsp;|&nbsp; <b>Source File:</b> {results_path.name}",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    # 2. Key Metrics Cards
    rate_style = card_val_pass if pass_rate >= 75.0 else card_val_fail
    cards_data = [
        [
            Paragraph("PASS RATE", card_label),
            Paragraph("TOTAL QUERIES", card_label),
            Paragraph("PASSED (GOOD)", card_label),
            Paragraph("FAILED (BAD)", card_label),
            Paragraph("AVG GEN LATENCY", card_label),
        ],
        [
            Paragraph(f"{pass_rate:.1f}%", rate_style),
            Paragraph(str(total), card_val_neutral),
            Paragraph(str(passed), card_val_pass),
            Paragraph(str(failed), card_val_fail),
            Paragraph(f"{avg_gen_lat:.2f}s", card_val_neutral),
        ],
    ]

    cards_table = Table(cards_data, colWidths=[100, 100, 100, 100, 104])
    cards_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(cards_table)
    story.append(Spacer(1, 16))

    # 3. SQL Feature Performance Breakdown
    story.append(Paragraph("Performance Breakdown by SQL Feature", h2_style))
    feat_rows = [[
        Paragraph("SQL Feature", table_hdr),
        Paragraph("Total Queries", table_hdr),
        Paragraph("Passed", table_hdr),
        Paragraph("Failed", table_hdr),
        Paragraph("Accuracy", table_hdr),
    ]]

    for feat, stats in sorted(features.items(), key=lambda x: x[1]["total"], reverse=True):
        f_tot = stats["total"]
        f_pass = stats["pass"]
        f_fail = f_tot - f_pass
        f_rate = (f_pass / f_tot * 100) if f_tot > 0 else 0.0
        acc_color = "#059669" if f_rate >= 75 else ("#d97706" if f_rate >= 50 else "#dc2626")
        feat_rows.append([
            Paragraph(feat, table_cell),
            Paragraph(str(f_tot), table_cell),
            Paragraph(str(f_pass), table_cell),
            Paragraph(str(f_fail), table_cell),
            Paragraph(f"<font color='{acc_color}'><b>{f_rate:.1f}%</b></font>", table_cell),
        ])

    feat_table = Table(feat_rows, colWidths=[204, 75, 75, 75, 75])
    feat_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ]
        )
    )
    story.append(feat_table)
    story.append(Spacer(1, 18))

    # 4. Detailed Test Case Inspection
    story.append(Paragraph("Itemized Test Case Results", h2_style))
    story.append(
        Paragraph(
            "Detailed breakdown of query input, generated explanation, and judge critique for each test item.",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))

    for idx, item in enumerate(items, 1):
        item_id = item.get("id", f"Item-{idx}")
        score = item.get("score")
        is_good = score == 1
        meta_item = metadata.get(item_id, {})
        feat = meta_item.get("feature", "N/A")
        dom = meta_item.get("domain", "N/A")
        sql = item.get("input", "")
        output = item.get("output", "No explanation generated.")
        reason = item.get("reason", item.get("error", "No critique recorded."))
        lat = item.get("sys_cost", {}).get("latency", 0)

        badge_bg = colors.HexColor("#d1fae5") if is_good else colors.HexColor("#fee2e2")
        badge_text = (
            Paragraph(f"<b>PASS (Good)</b> &nbsp;[{lat:.1f}s]", badge_pass)
            if is_good
            else Paragraph(f"<b>FAIL (Bad)</b> &nbsp;[{lat:.1f}s]", badge_fail)
        )

        item_header_data = [
            [
                Paragraph(f"<b>#{idx}. {item_id}</b> &nbsp;|&nbsp; Feature: <i>{feat}</i> &nbsp;|&nbsp; Domain: <i>{dom}</i>", table_hdr),
                badge_text,
            ]
        ]
        item_header_table = Table(item_header_data, colWidths=[384, 120])
        item_header_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f1f5f9")),
                    ("BACKGROUND", (1, 0), (1, 0), badge_bg),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ]
            )
        )

        # SQL block, output block, reason block
        body_data = [
            [Paragraph("<b>SQL Query:</b>", table_hdr)],
            [Paragraph(sql.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_text)],
            [Paragraph("<b>Generated Explanation:</b>", table_hdr)],
            [Paragraph(output.replace("\n", "<br/>"), table_cell)],
            [Paragraph("<b>Judge Critique / Reason:</b>", table_hdr)],
            [Paragraph(f"<i>{reason}</i>", table_cell)],
        ]
        body_table = Table(body_data, colWidths=[504])
        body_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ]
            )
        )

        case_block = [item_header_table, body_table, Spacer(1, 10)]
        story.append(KeepTogether(case_block))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_pdf_path


def main():
    parser = argparse.ArgumentParser(description="Export benchmark results to a formatted PDF report.")
    parser.add_argument("results", type=pathlib.Path, help="Path to results JSONL file (e.g. results/v2_dev.jsonl)")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None, help="Output PDF path.")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"), help="Data directory.")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: Results file '{args.results}' does not exist.")
        sys.exit(1)

    if args.output is None:
        args.output = args.results.parent / f"{args.results.stem}_report.pdf"

    print(f"Generating PDF report from {args.results}...")
    pdf_path = generate_pdf_report(args.results, args.output, args.data_dir)
    print(f"[OK] Report generated successfully: {pdf_path}")


if __name__ == "__main__":
    main()
