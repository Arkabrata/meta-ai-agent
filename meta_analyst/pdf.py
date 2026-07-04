from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

from meta_analyst.config import CURRENCY_SYMBOL


def generate_full_pdf(df, figs, insights, filename="Meta_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Meta Ads Performance Report</b>", styles["Title"]),
        Spacer(1, 20),
        Paragraph("<b>KPI Summary</b>", styles["Heading2"]),
    ]

    kpis = [
        f"Total Spend: {CURRENCY_SYMBOL}{df['spend'].sum():,.0f}",
        f"Total Revenue: {CURRENCY_SYMBOL}{df['revenue'].sum():,.0f}",
        f"Total Impressions: {df['impressions'].sum():,.0f}",
        f"Total Clicks: {df['clicks'].sum():,.0f}",
        f"Overall ROAS: {df['revenue'].sum() / df['spend'].sum():.2f}",
    ]
    for kpi in kpis:
        story.append(Paragraph(f"- {kpi}", styles["Normal"]))
    story.append(Spacer(1, 20))

    for name, fig in figs.items():
        story.append(Paragraph(f"<b>{name.replace('_', ' ').title()}</b>", styles["Heading2"]))
        img_path = f"{name}.png"
        fig.savefig(img_path, dpi=140, bbox_inches="tight")
        story.append(Image(img_path, width=6.2 * inch, height=3.2 * inch))
        story.append(Spacer(1, 10))

        if name in insights:
            story.append(
                Paragraph(
                    f"<b>AI Insight:</b><br/>{insights[name].replace(chr(10), '<br/>')}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 14))
        story.append(PageBreak())

    doc.build(story)
    return filename
