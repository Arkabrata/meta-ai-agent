from __future__ import annotations

import pandas as pd

from meta_analyst.config import CURRENCY_SYMBOL

DASH_COLORS = ["#0f766e", "#d97706", "#164e63", "#b45309", "#0e7490", "#a16207", "#115e59", "#c2410c"]


def pastel_colors(n):
    return DASH_COLORS[:n] if n <= len(DASH_COLORS) else (DASH_COLORS * ((n // len(DASH_COLORS)) + 1))[:n]


def kpi_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    tot_spend = float(df["spend"].sum()) if "spend" in df.columns else 0.0
    tot_rev = float(df["revenue"].sum()) if "revenue" in df.columns else 0.0
    tot_impr = float(df["impressions"].sum()) if "impressions" in df.columns else 0.0
    tot_clicks = float(df["clicks"].sum()) if "clicks" in df.columns else 0.0
    avg_roas = tot_rev / tot_spend if tot_spend else 0
    avg_cpc = float(df["cpc"].mean()) if "cpc" in df.columns else 0.0
    if "cpm" in df.columns:
        avg_cpm = float(df["cpm"].mean())
    elif "cpm_calc" in df.columns:
        avg_cpm = float(df["cpm_calc"].mean())
    else:
        avg_cpm = 0.0
    avg_ctr = float(df["ctr_calc"].mean()) if "ctr_calc" in df.columns else 0.0
    avg_lpv = float(df["lpv_lc"].mean()) if "lpv_lc" in df.columns else 0.0
    avg_imp_lpv = float(df["imp_lpv"].mean()) if "imp_lpv" in df.columns else 0.0

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


def _short_objective(name: object) -> str:
    s = str(name).replace("OUTCOME_", "").replace("_", " ").strip().title()
    return s or "Unknown"


def _short(name: object, n: int = 18) -> str:
    s = str(name)
    return s if len(s) <= n else s[: n - 1] + "…"


def create_charts(df: pd.DataFrame) -> dict:
    """Build matplotlib figures that mirror the React dashboard panels."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figs: dict = {}
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlepad": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#e8e4dc",
            "axes.grid": True,
            "grid.color": "#e8e4dc",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    spend_label = f"Spend ({CURRENCY_SYMBOL if str(CURRENCY_SYMBOL).isascii() else 'INR'})"

    # --- Trend over time ---
    if "date_start" in df.columns and "spend" in df.columns:
        dated = df.dropna(subset=["date_start"])
        if dated["date_start"].nunique() > 1:
            agg = {"spend": ("spend", "sum")}
            if "revenue" in dated.columns:
                agg["revenue"] = ("revenue", "sum")
            trend = dated.groupby(dated["date_start"].dt.date).agg(**agg).sort_index().fillna(0)
            if "revenue" in trend.columns:
                trend["roas"] = trend["revenue"] / trend["spend"].replace(0, np.nan)
            else:
                trend["revenue"] = 0
                trend["roas"] = 0
            trend = trend.fillna(0)
            fig, ax = plt.subplots(figsize=(7.2, 2.6))
            x = np.arange(len(trend))
            w = 0.38
            ax.bar(x - w / 2, trend["spend"], w, label="Spend", color="#0f766e")
            ax.bar(x + w / 2, trend["revenue"], w, label="Revenue", color="#f0b35a")
            ax2 = ax.twinx()
            ax2.plot(x, trend["roas"], color="#164e63", linewidth=1.8, label="ROAS")
            ax2.set_ylabel("ROAS")
            ax2.grid(False)
            ax.set_xticks(x[:: max(1, len(x) // 10)])
            ax.set_xticklabels([str(d)[5:] for d in trend.index[:: max(1, len(x) // 10)]], fontsize=7)
            ax.set_ylabel(spend_label)
            ax.set_title("Performance over time")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=7, frameon=False)
            fig.tight_layout()
            figs["trend"] = fig

    # --- Spend by objective (donut) ---
    if "objective" in df.columns and "spend" in df.columns:
        spend_pie = df.groupby("objective")["spend"].sum().sort_values(ascending=False)
        if len(spend_pie) and spend_pie.sum() > 0:
            labels = [_short_objective(i) for i in spend_pie.index]
            fig, ax = plt.subplots(figsize=(3.6, 3.0))
            wedges, _ = ax.pie(
                spend_pie.values,
                labels=None,
                colors=pastel_colors(len(spend_pie)),
                wedgeprops={"width": 0.42, "linewidth": 1.5, "edgecolor": "white"},
                startangle=90,
            )
            total = float(spend_pie.sum())
            pcts = [v / total * 100 for v in spend_pie.values]
            legend_labels = [f"{lab} ({p:.1f}%)" for lab, p in zip(labels, pcts)]
            ax.legend(
                wedges,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.95, 0.5),
                frameon=False,
                fontsize=7,
            )
            ax.set_title("Spend by objective")
            fig.tight_layout()
            figs["objective_spend"] = fig

        # Spend vs revenue by objective (composed)
        if "revenue" in df.columns:
            summary = (
                df.groupby("objective")
                .agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
                .sort_values("spend", ascending=False)
            )
            summary["roas"] = summary["revenue"] / summary["spend"].replace(0, np.nan)
            summary = summary.fillna(0)
            names = [_short_objective(i) for i in summary.index]
            fig, ax = plt.subplots(figsize=(3.6, 3.0))
            x = np.arange(len(summary))
            w = 0.35
            ax.bar(x - w / 2, summary["spend"], w, label="Spend", color="#0f766e")
            ax.bar(x + w / 2, summary["revenue"], w, label="Revenue", color="#d97706")
            ax2 = ax.twinx()
            ax2.plot(x, summary["roas"], color="#164e63", marker="o", markersize=4, linewidth=1.6, label="ROAS")
            ax2.set_ylabel("ROAS")
            ax2.grid(False)
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=7, rotation=20, ha="right")
            ax.set_title("Spend vs revenue by objective")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=7, frameon=False)
            fig.tight_layout()
            figs["objective_spend_revenue"] = fig

    # --- Top campaigns by spend ---
    if "campaign_name" in df.columns and "spend" in df.columns:
        camp = df.groupby("campaign_name").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum") if "revenue" in df.columns else ("spend", "sum"),
        )
        camp = camp.sort_values("spend", ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        y = np.arange(len(camp))
        ax.barh(y - 0.18, camp["spend"], 0.35, label="Spend", color="#0f766e")
        if "revenue" in df.columns:
            ax.barh(y + 0.18, camp["revenue"], 0.35, label="Revenue", color="#f0b35a")
        ax.set_yticks(y)
        ax.set_yticklabels([_short(i, 16) for i in camp.index], fontsize=7)
        ax.set_title("Top campaigns by spend")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        figs["campaign_spend"] = fig

    # --- Conversion funnel ---
    if "impressions" in df.columns:
        stages = [("Impressions", float(df["impressions"].sum()))]
        if "clicks" in df.columns:
            stages.append(("Clicks", float(df["clicks"].sum())))
        if "landing_page_views" in df.columns:
            stages.append(("LP Views", float(df["landing_page_views"].sum())))
        elif "lpv" in df.columns:
            stages.append(("LP Views", float(df["lpv"].sum())))
        if "adds_to_cart" in df.columns:
            stages.append(("Add to Cart", float(df["adds_to_cart"].sum())))
        if "checkouts" in df.columns:
            stages.append(("Checkouts", float(df["checkouts"].sum())))
        if "purchases" in df.columns:
            stages.append(("Purchases", float(df["purchases"].sum())))
        if len(stages) >= 3:
            top = stages[0][1] or 1
            fig, ax = plt.subplots(figsize=(3.6, 3.0))
            names = [s[0] for s in stages]
            vals = [s[1] for s in stages]
            bars = ax.bar(names, vals, color=pastel_colors(len(stages)))
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v / top * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
            ax.set_title("Conversion funnel")
            ax.tick_params(axis="x", labelsize=7, rotation=15)
            fig.tight_layout()
            figs["funnel"] = fig

    # --- Spend vs ROAS scatter (bubble ≈ purchases) ---
    if {"spend", "roas"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        purchases = df["purchases"] if "purchases" in df.columns else pd.Series(1, index=df.index)
        sizes = 20 + (purchases.fillna(0) / max(purchases.max() or 1, 1)) * 180
        ax.scatter(
            df["spend"],
            df["roas"],
            s=sizes,
            c="#d97706",
            alpha=0.65,
            edgecolors="none",
        )
        ax.set_title("Spend vs ROAS efficiency")
        ax.set_xlabel(spend_label)
        ax.set_ylabel("ROAS")
        fig.tight_layout()
        figs["scatter_spend_roas"] = fig

    # --- Ad set performance ---
    if {"adset_name", "spend"}.issubset(df.columns):
        aset = df.groupby("adset_name").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum") if "revenue" in df.columns else ("spend", "sum"),
        )
        aset["roas"] = aset["revenue"] / aset["spend"].replace(0, np.nan)
        aset = aset.fillna(0).sort_values("spend", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        x = np.arange(len(aset))
        ax.bar(x, aset["spend"], color="#0e7490", label="Spend")
        ax2 = ax.twinx()
        ax2.plot(x, aset["roas"], color="#d97706", marker="o", markersize=4, linewidth=1.6, label="ROAS")
        ax2.set_ylabel("ROAS")
        ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(i, 12) for i in aset.index], fontsize=6.5, rotation=28, ha="right")
        ax.set_title("Ad set performance")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout()
        figs["adset_perf"] = fig

    # --- ROAS distribution ---
    if {"roas", "spend"}.issubset(df.columns):
        bins = [0, 0.5, 1, 1.5, 2, 3, 5, float("inf")]
        labels = ["<0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–5", "5+"]
        valid = df.dropna(subset=["roas", "spend"]).copy()
        valid["bucket"] = pd.cut(valid["roas"].clip(lower=0), bins=bins, labels=labels, right=False)
        dist = valid.groupby("bucket", observed=False).agg(ads=("roas", "size"), spend=("spend", "sum"))
        colors = ["#b91c1c", "#b91c1c", "#d97706", "#d97706", "#0f766e", "#0f766e", "#0f766e"]
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        x = np.arange(len(dist))
        ax.bar(x, dist["spend"], color=colors[: len(dist)], alpha=0.85, label="Spend")
        ax2 = ax.twinx()
        ax2.plot(x, dist["ads"], color="#164e63", marker="o", markersize=4, linewidth=1.6, label="Ads")
        ax2.set_ylabel("Ads")
        ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in dist.index], fontsize=7)
        ax.set_title("ROAS distribution")
        lines, labels_ = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels_ + labels2, fontsize=7, frameon=False)
        fig.tight_layout()
        figs["roas_distribution"] = fig

    # --- Cost efficiency ---
    if {"campaign_name", "spend", "clicks", "impressions"}.issubset(df.columns):
        eff = df.groupby("campaign_name").agg(
            spend=("spend", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum")
        )
        eff["cpc"] = eff["spend"] / eff["clicks"].replace(0, np.nan)
        eff["cpm"] = eff["spend"] / eff["impressions"].replace(0, np.nan) * 1000
        eff = eff.fillna(0).sort_values("spend", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        x = np.arange(len(eff))
        w = 0.35
        ax.bar(x - w / 2, eff["cpc"], w, label="CPC", color="#0f766e")
        ax.bar(x + w / 2, eff["cpm"], w, label="CPM", color="#a16207")
        ax.set_xticks(x)
        ax.set_xticklabels([_short(i, 12) for i in eff.index], fontsize=6.5, rotation=28, ha="right")
        ax.set_title("Cost efficiency")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        figs["cost_efficiency"] = fig

    # --- Reach & frequency ---
    if {"campaign_name", "reach"}.issubset(df.columns):
        rf = df.groupby("campaign_name").agg(
            reach=("reach", "sum"),
            impressions=("impressions", "sum") if "impressions" in df.columns else ("reach", "sum"),
        )
        rf["frequency"] = rf["impressions"] / rf["reach"].replace(0, np.nan)
        rf = rf.fillna(0).sort_values("reach", ascending=False).head(8)
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        x = np.arange(len(rf))
        ax.bar(x, rf["reach"], color="#164e63", label="Reach")
        ax2 = ax.twinx()
        ax2.plot(x, rf["frequency"], color="#d97706", marker="o", markersize=4, linewidth=1.6, label="Frequency")
        ax2.set_ylabel("Frequency")
        ax2.grid(False)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(i, 12) for i in rf.index], fontsize=6.5, rotation=28, ha="right")
        ax.set_title("Reach & frequency")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, frameon=False)
        fig.tight_layout()
        figs["reach_frequency"] = fig

    # --- CTR by campaign ---
    if "campaign_name" in df.columns and "ctr_calc" in df.columns:
        ctr = (
            df.groupby("campaign_name")["ctr_calc"]
            .mean()
            .sort_values(ascending=False)
            .head(8)
            * 100
        )
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        ax.bar([_short(i, 12) for i in ctr.index], ctr.values, color="#164e63")
        ax.set_ylabel("CTR %")
        ax.set_title("CTR by campaign")
        ax.tick_params(axis="x", labelsize=6.5, rotation=28)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig.tight_layout()
        figs["ctr_by_campaign"] = fig

    return figs


def campaign_scorecard(df: pd.DataFrame, limit: int = 10) -> list[dict]:
    if "campaign_name" not in df.columns or "spend" not in df.columns:
        return []
    g = df.groupby("campaign_name").agg(
        spend=("spend", "sum"),
        revenue=("revenue", "sum") if "revenue" in df.columns else ("spend", "sum"),
        clicks=("clicks", "sum") if "clicks" in df.columns else ("spend", "size"),
        impressions=("impressions", "sum") if "impressions" in df.columns else ("spend", "size"),
        purchases=("purchases", "sum") if "purchases" in df.columns else ("spend", "size"),
    )
    g["roas"] = g["revenue"] / g["spend"].replace(0, pd.NA)
    g["ctr"] = g["clicks"] / g["impressions"].replace(0, pd.NA) * 100
    g["cpc"] = g["spend"] / g["clicks"].replace(0, pd.NA)
    g = g.fillna(0).sort_values("spend", ascending=False).head(limit)
    return [
        {
            "campaign": str(i),
            "spend": round(float(r.spend), 0),
            "revenue": round(float(r.revenue), 0),
            "roas": round(float(r.roas), 2),
            "ctr": round(float(r.ctr), 2),
            "cpc": round(float(r.cpc), 2),
            "clicks": int(r.clicks),
            "purchases": int(r.purchases),
        }
        for i, r in g.iterrows()
    ]


def ad_leaderboard(df: pd.DataFrame, *, strongest: bool = True, limit: int = 8) -> list[dict]:
    if not {"ad_name", "roas", "spend"}.issubset(df.columns):
        return []
    work = df.dropna(subset=["roas"]).copy()
    if work.empty:
        return []
    work = work.sort_values("roas", ascending=not strongest).head(limit)
    rows = []
    for _, r in work.iterrows():
        rows.append(
            {
                "ad": str(r.get("ad_name", "")),
                "campaign": str(r.get("campaign_name", "")),
                "spend": round(float(r.get("spend", 0) or 0), 0),
                "roas": round(float(r.get("roas", 0) or 0), 2),
                "purchases": int(r.get("purchases", 0) or 0),
            }
        )
    return rows
