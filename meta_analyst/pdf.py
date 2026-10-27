from __future__ import annotations

import html
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from meta_analyst.charts import ad_leaderboard, campaign_scorecard, kpi_metrics
from meta_analyst.config import CURRENCY_SYMBOL

TEAL = colors.HexColor("#0f766e")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#e2e8f0")
SOFT = colors.HexColor("#f8fafc")
GOOD = colors.HexColor("#047857")
BAD = colors.HexColor("#b91c1c")
MID = colors.HexColor("#d97706")

_CHART_TITLES = {
    "trend": "Performance over time",
    "objective_spend": "Spend by objective",
    "objective_spend_revenue": "Spend vs revenue by objective",
    "campaign_spend": "Top campaigns by spend",
    "funnel": "Conversion funnel",
    "scatter_spend_roas": "Spend vs ROAS efficiency",
    "adset_perf": "Ad set performance",
    "roas_distribution": "ROAS distribution",
    "cost_efficiency": "Cost efficiency",
    "reach_frequency": "Reach & frequency",
    "ctr_by_campaign": "CTR by campaign",
}


def _pdf_money(value: float, decimals: int = 0) -> str:
    sym = CURRENCY_SYMBOL if str(CURRENCY_SYMBOL).isascii() else "Rs."
    return f"{sym}{value:,.{decimals}f}"


def _titleize(key: str) -> str:
    return _CHART_TITLES.get(key, re.sub(r"_+", " ", key).strip().title())


def _safe_insight(text: str) -> str:
    cleaned = html.escape((text or "").strip())
    return cleaned.replace("\n", "<br/>") if cleaned else ""


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=18,
            leading=22,
            spaceAfter=2,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        "sub": ParagraphStyle(
            "Sub",
            parent=base["Normal"],
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "section": ParagraphStyle(
            "SectionHead",
            parent=base["Heading2"],
            fontSize=11,
            leading=13,
            spaceBefore=4,
            spaceAfter=4,
            textColor=TEAL,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155"),
            alignment=TA_LEFT,
        ),
        "kpi_label": ParagraphStyle(
            "KpiLabel",
            parent=base["Normal"],
            fontSize=7,
            textColor=MUTED,
            leading=9,
        ),
        "kpi_value": ParagraphStyle(
            "KpiValue",
            parent=base["Normal"],
            fontSize=10,
            leading=12,
            textColor=INK,
            fontName="Helvetica-Bold",
        ),
        "th": ParagraphStyle(
            "Th",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=MUTED,
            fontName="Helvetica-Bold",
        ),
        "td": ParagraphStyle(
            "Td",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=INK,
            leading=9,
        ),
        "td_right": ParagraphStyle(
            "TdR",
            parent=base["Normal"],
            fontSize=7.5,
            textColor=INK,
            alignment=TA_RIGHT,
        ),
    }


def _pdf_safe_text(value: str) -> str:
    """Replace Unicode currency glyphs Helvetica cannot draw."""
    sym = CURRENCY_SYMBOL if str(CURRENCY_SYMBOL).isascii() else "Rs."
    if CURRENCY_SYMBOL and CURRENCY_SYMBOL != sym:
        value = value.replace(CURRENCY_SYMBOL, sym)
    # common extras
    return value.replace("→", "->").replace("–", "-").replace("—", "-")


def _kpi_grid(df, styles) -> Table:
    kpis = [(_pdf_safe_text(label), _pdf_safe_text(value)) for label, value in kpi_metrics(df)]
    cards = []
    for label, value in kpis:
        inner = Table(
            [[Paragraph(label, styles["kpi_label"])], [Paragraph(value, styles["kpi_value"])]],
            colWidths=[1.28 * inch],
        )
        inner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                    ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        cards.append(inner)

    # 5 per row to match dashboard density (10 KPIs)
    rows = [cards[i : i + 5] for i in range(0, len(cards), 5)]
    # pad last row
    while rows and len(rows[-1]) < 5:
        rows[-1].append("")
    grid = Table(rows, colWidths=[1.38 * inch] * 5, hAlign="CENTER")
    grid.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return grid


def _chart_cell(name: str, img_path: str, insight: str, styles: dict, wide: bool = False) -> Table:
    w = 6.8 * inch if wide else 3.25 * inch
    h = 2.45 * inch if wide else 2.35 * inch
    rows = [
        [Paragraph(_titleize(name), styles["section"])],
        [Image(img_path, width=w, height=h)],
    ]
    if insight:
        rows.append([Paragraph(f"<b>AI Insight:</b> {insight}", styles["body"])])
    cell = Table(rows, colWidths=[w + 0.1 * inch])
    cell.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return cell


def _scorecard_table(df, styles) -> Table | None:
    rows = campaign_scorecard(df)
    if not rows:
        return None
    header = [
        Paragraph(h, styles["th"])
        for h in ["Campaign", "Spend", "Revenue", "ROAS", "CTR", "CPC", "Clicks", "Purch."]
    ]
    data = [header]
    for r in rows:
        roas_color = GOOD if r["roas"] >= 2 else (BAD if r["roas"] < 1 else MID)
        data.append(
            [
                Paragraph(html.escape(r["campaign"][:42]), styles["td"]),
                Paragraph(_pdf_money(r["spend"]), styles["td_right"]),
                Paragraph(_pdf_money(r["revenue"]), styles["td_right"]),
                Paragraph(f'<font color="{roas_color.hexval()}"><b>{r["roas"]:.2f}</b></font>', styles["td_right"]),
                Paragraph(f'{r["ctr"]:.2f}%', styles["td_right"]),
                Paragraph(_pdf_money(r["cpc"], 2), styles["td_right"]),
                Paragraph(f'{r["clicks"]:,}', styles["td_right"]),
                Paragraph(f'{r["purchases"]:,}', styles["td_right"]),
            ]
        )
    table = Table(
        data,
        colWidths=[
            2.0 * inch,
            0.75 * inch,
            0.8 * inch,
            0.55 * inch,
            0.55 * inch,
            0.65 * inch,
            0.6 * inch,
            0.55 * inch,
        ],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table


def _ads_table(rows: list[dict], styles, title: str, good: bool) -> Table | None:
    if not rows:
        return None
    header = [Paragraph(h, styles["th"]) for h in ["Ad", "Spend", "ROAS", "Purch."]]
    data = [header]
    for r in rows:
        tone = GOOD if good else BAD
        data.append(
            [
                Paragraph(
                    f'{html.escape(r["ad"][:28])}<br/><font color="#64748b" size="6">{html.escape(r["campaign"][:28])}</font>',
                    styles["td"],
                ),
                Paragraph(_pdf_money(r["spend"]), styles["td_right"]),
                Paragraph(f'<font color="{tone.hexval()}"><b>{r["roas"]:.2f}</b></font>', styles["td_right"]),
                Paragraph(str(r["purchases"]), styles["td_right"]),
            ]
        )
    table = Table(data, colWidths=[1.9 * inch, 0.7 * inch, 0.55 * inch, 0.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    # wrap with title
    wrap = Table(
        [[Paragraph(title, styles["section"])], [table]],
        colWidths=[3.7 * inch],
    )
    wrap.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return wrap


def generate_full_pdf(df, figs, insights, filename="Meta_Report.pdf"):
    styles = _styles()
    row_count = len(df)
    camps = int(df["campaign_name"].nunique()) if "campaign_name" in df.columns else 0
    objs = int(df["objective"].nunique()) if "objective" in df.columns else 0

    story = [
        Paragraph("Meta Ads Performance Report", styles["title"]),
        Paragraph(
            f"{row_count:,} ads · {camps} campaigns · {objs} objectives — same view as the dashboard",
            styles["sub"],
        ),
        Paragraph("KPI Summary", styles["section"]),
        _kpi_grid(df, styles),
        Spacer(1, 8),
    ]

    # Full-width trend first (matches dashboard)
    items = list(figs.items())
    trend = None
    rest = []
    for name, fig in items:
        if name == "trend":
            trend = (name, fig)
        else:
            rest.append((name, fig))

    if trend:
        name, fig = trend
        path = f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        insight = _safe_insight((insights or {}).get(name, ""))
        story.append(_chart_cell(name, path, insight, styles, wide=True))
        story.append(Spacer(1, 6))

    cells = []
    for name, fig in rest:
        path = f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
        insight = _safe_insight((insights or {}).get(name, ""))
        cells.append(_chart_cell(name, path, insight, styles, wide=False))

    for i in range(0, len(cells), 2):
        if i + 1 < len(cells):
            pair = Table([[cells[i], cells[i + 1]]], colWidths=[3.5 * inch, 3.5 * inch])
            pair.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(pair)
        else:
            story.append(cells[i])
        story.append(Spacer(1, 2))

    # Campaign scorecard
    score = _scorecard_table(df, styles)
    if score:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Campaign scorecard", styles["section"]))
        story.append(
            Paragraph("Ranked by spend · ROAS and CTR for quick triage", styles["body"])
        )
        story.append(Spacer(1, 4))
        story.append(score)

    # Strongest / weakest ads
    top = _ads_table(ad_leaderboard(df, strongest=True), styles, "Strongest ads", True)
    weak = _ads_table(ad_leaderboard(df, strongest=False), styles, "Weak ads", False)
    if top or weak:
        story.append(Spacer(1, 8))
        story.append(
            Table(
                [[top or "", weak or ""]],
                colWidths=[3.5 * inch, 3.5 * inch],
            )
        )

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
        title="Meta Ads Performance Report",
    )
    doc.build(story)
    return filename
