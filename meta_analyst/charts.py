import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from meta_analyst.config import CURRENCY_SYMBOL


def pastel_colors(n):
    base = [
        "#A3CEF1",
        "#F9C8C8",
        "#C6E5B1",
        "#F7E8A4",
        "#DCC4FF",
        "#B5EAEA",
        "#FFB5E8",
        "#FFCBC1",
        "#C0FFD4",
    ]
    return base[:n]


def kpi_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    tot_spend = df["spend"].sum()
    tot_rev = df["revenue"].sum()
    tot_impr = df["impressions"].sum()
    tot_clicks = df["clicks"].sum()
    avg_roas = tot_rev / tot_spend if tot_spend else 0
    avg_cpc = df["cpc"].mean()
    avg_cpm = df["cpm"].mean() if "cpm" in df.columns else df.get("cpm_calc", pd.Series([0])).mean()
    avg_ctr = df["ctr_calc"].mean()
    avg_lpv = df["lpv_lc"].mean() if "lpv_lc" in df.columns else 0
    avg_imp_lpv = df["imp_lpv"].mean() if "imp_lpv" in df.columns else 0

    return [
        ("Total Spend", f"{CURRENCY_SYMBOL}{tot_spend:,.0f}"),
        ("Total Revenue", f"{CURRENCY_SYMBOL}{tot_rev:,.0f}"),
        ("Total Impressions", f"{tot_impr:,.0f}"),
        ("Total Clicks", f"{tot_clicks:,.0f}"),
        ("Overall ROAS", f"{avg_roas:,.2f}"),
        ("Avg CPC", f"{CURRENCY_SYMBOL}{avg_cpc:,.2f}"),
        ("Avg CPM", f"{CURRENCY_SYMBOL}{avg_cpm:,.2f}"),
        ("Avg CTR", f"{avg_ctr * 100:,.2f}%"),
        ("Avg LPview/LC%", f"{avg_lpv:,.2f}%"),
        ("Avg Imp → LPV%", f"{avg_imp_lpv:,.2f}%"),
    ]


def create_charts(df: pd.DataFrame) -> dict:
    figs = {}

    if "objective" in df.columns:
        spend_pie = df.groupby("objective")["spend"].sum()
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        ax.pie(
            spend_pie.values,
            labels=spend_pie.index,
            autopct="%1.1f%%",
            colors=pastel_colors(len(spend_pie)),
        )
        ax.set_title("Spend Share by Objective", fontsize=11)
        figs["objective_spend_pie"] = fig

        summary = df.groupby("objective").agg({"revenue": "sum", "spend": "sum"})
        summary["roas"] = summary["revenue"] / summary["spend"]
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        summary["roas"].sort_values(ascending=False).plot(kind="barh", ax=ax, color="#A3CEF1")
        ax.set_title("ROAS by Objective", fontsize=11)
        ax.set_xlabel("ROAS")
        figs["objective_roas"] = fig

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.scatter(df["spend"], df["roas"], s=14, alpha=0.5)
    ax.set_title("Spend vs ROAS", fontsize=11)
    ax.set_xlabel("Spend")
    ax.set_ylabel("ROAS")
    figs["scatter_spend_roas"] = fig

    if "campaign_name" in df.columns:
        heat = (
            df.groupby("campaign_name")["ctr_calc"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .to_frame()
        )
        fig, ax = plt.subplots(figsize=(3.8, 2.9))
        sns.heatmap(heat, cmap="Blues", ax=ax, annot=True, fmt=".2f")
        ax.set_title("CTR Heatmap (Top Campaigns)", fontsize=11)
        figs["ctr_heatmap"] = fig

    return figs
