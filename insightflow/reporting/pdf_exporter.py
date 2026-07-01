"""Exporting an experiment report to a shareable PDF.

Stakeholders live in email and Slack, not the dashboard. A one-page PDF is how an
experiment result actually travels through an org. We build it with ReportLab (pure
Python — no headless browser, no system libraries), so PDF generation works anywhere
the rest of the package does, including CI.

The layout mirrors the on-screen report: a colored recommendation banner up top, the
key numbers as a clean table, then the narrative. Deliberately one page and skimmable.
"""

from __future__ import annotations

import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .report_generator import ExperimentReport

# Recommendation -> banner color, matching the dashboard's semantics.
_BANNER = {
    "SHIP": colors.HexColor("#2ea44f"),
    "DO NOT SHIP": colors.HexColor("#dc2626"),
    "EXTEND": colors.HexColor("#d97706"),
    "INVALID": colors.HexColor("#6b7280"),
}
_NAVY = colors.HexColor("#0f3460")


def report_to_pdf_bytes(report: ExperimentReport) -> bytes:
    """Render the report to PDF and return the raw bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        title=f"InsightFlow — {report.name}",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, textColor=_NAVY, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#64748b"))
    banner = ParagraphStyle("banner", parent=styles["Title"], fontSize=15,
                            textColor=colors.white, alignment=1, spaceBefore=6, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=15)

    story = []
    story.append(Paragraph(f"InsightFlow — {report.name}", h1))
    story.append(Paragraph(
        f"Status: {report.status} &nbsp;·&nbsp; Metric: {report.metric_type} &nbsp;·&nbsp; "
        f"Generated {report.generated_at:%Y-%m-%d %H:%M UTC}", sub,
    ))
    story.append(Spacer(1, 14))

    # Recommendation banner.
    banner_color = _BANNER.get(report.recommendation, _NAVY)
    banner_tbl = Table([[Paragraph(f"RECOMMENDATION: {report.recommendation}", banner)]], colWidths=[6.9 * inch])
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), banner_color),
        ("BOX", (0, 0), (-1, -1), 0, banner_color),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(report.headline, body))
    story.append(Spacer(1, 16))

    # Key-numbers table.
    rows = [
        ["Metric", "Value"],
        ["Samples", f"{report.n_control:,} control / {report.n_treatment:,} treatment"],
        ["Primary test", report.primary_test_name],
        ["p-value", f"{report.p_value:.4g} ({'significant' if report.significant else 'not significant'})"],
    ]
    if report.effect_size is not None:
        rows.append([report.effect_size_name or "effect size", f"{report.effect_size:.4g}"])
    if report.ci_lower is not None:
        rows.append(["95% CI", f"[{report.ci_lower:.4g}, {report.ci_upper:.4g}]"])
    if report.bayesian:
        rows.append(["P(treatment best)", f"{report.bayesian['prob_treatment_best']:.1%}"])
        rows.append(["Expected uplift", f"{report.bayesian['expected_relative_uplift']:+.1%}"])
    if report.srm:
        rows.append(["SRM check", "MISMATCH" if report.srm["mismatch_detected"] else "healthy"])
    if report.top_quintile_lift is not None:
        rows.append(["Top-quintile uplift", f"{report.top_quintile_lift:.2f}x overall ATE"])

    table = Table(rows, colWidths=[2.3 * inch, 4.6 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)

    if report.narrative:
        story.append(Spacer(1, 16))
        story.append(Paragraph("Summary", ParagraphStyle("sh", parent=styles["Heading2"], textColor=_NAVY)))
        story.append(Paragraph(report.narrative, body))

    doc.build(story)
    return buffer.getvalue()


def report_to_pdf(report: ExperimentReport, path: str) -> str:
    """Write the report PDF to ``path`` and return the path."""
    with open(path, "wb") as f:
        f.write(report_to_pdf_bytes(report))
    return path
