from __future__ import annotations

from typing import BinaryIO

import numpy as np
import pandas as pd


def pick_col(candidates, all_cols):
    for name in candidates:
        if name in all_cols:
            return name
    return None


def _read_dataframe(source: str | BinaryIO) -> pd.DataFrame:
    if isinstance(source, str):
        if source.lower().endswith(".csv"):
            return pd.read_csv(source)
        return pd.read_excel(source)

    name = getattr(source, "name", "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(source)
    return pd.read_excel(source)


def load_and_clean_data(source: str | BinaryIO, *, add_recommendations: bool = False) -> pd.DataFrame:
    """Load a Meta Ads export and normalize columns, metrics, and optional recommendations."""
    df = _read_dataframe(source)
    cols = df.columns

    col_campaign = pick_col(["Campaign name", "campaign_name"], cols)
    col_adset = pick_col(["Ad set name", "adset_name"], cols)
    col_ad = pick_col(["Ad name", "ad_name"], cols)
    col_obj = pick_col(["Objective", "objective", "Campaign objective"], cols)
    col_status = pick_col(["Ad delivery", "Delivery", "status"], cols)
    col_impr = pick_col(["Impressions", "impressions"], cols)
    col_clicks = pick_col(["Clicks (all)", "Clicks", "clicks"], cols)
    col_link_clicks = pick_col(["Link clicks", "Outbound clicks", "link_clicks"], cols)
    col_spend = pick_col(["Amount spent (INR)", "Amount spent", "Spend", "spend"], cols)
    col_purchases = pick_col(["Purchases", "Website purchases", "purchases"], cols)
    col_rev = pick_col(
        [
            "Purchases conversion value",
            "Website purchases conversion value",
            "Conversion value",
            "revenue",
        ],
        cols,
    )
    col_ctr = pick_col(["CTR (all)", "CTR", "ctr"], cols)
    col_cpc_all = pick_col(["CPC (all) (INR)", "CPC (all)", "cpc_all"], cols)
    col_cpm = pick_col(
        [
            "CPM (cost per 1,000 impressions) (INR)",
            "CPM (cost per 1,000 impressions)",
            "CPM",
            "cpm",
        ],
        cols,
    )
    col_lpv = pick_col(["LPview/LC%"], cols)
    col_imp_lpv = pick_col(["Imp to LPV %"], cols)
    col_date_start = pick_col(["Reporting starts", "date_start"], cols)
    col_date_end = pick_col(["Reporting ends", "date_end"], cols)

    rename_map = {}
    mapping = {
        col_campaign: "campaign_name",
        col_adset: "adset_name",
        col_ad: "ad_name",
        col_obj: "objective",
        col_status: "status",
        col_impr: "impressions",
        col_clicks: "clicks",
        col_link_clicks: "link_clicks",
        col_spend: "spend",
        col_purchases: "purchases",
        col_rev: "revenue",
        col_ctr: "ctr",
        col_cpc_all: "cpc_all",
        col_cpm: "cpm",
        col_lpv: "lpv_lc",
        col_imp_lpv: "imp_lpv",
        col_date_start: "date_start",
        col_date_end: "date_end",
    }
    for source_col, target_col in mapping.items():
        if source_col:
            rename_map[source_col] = target_col

    df = df.rename(columns=rename_map)

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        df = df[df["status"] == "active"].copy()

    num_cols = [
        "impressions",
        "clicks",
        "link_clicks",
        "spend",
        "purchases",
        "revenue",
        "cpc_all",
        "cpm",
        "ctr",
        "lpv_lc",
        "imp_lpv",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    if "date_end" in df.columns:
        df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")

    if "spend" in df.columns and "clicks" in df.columns:
        df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
    if "spend" in df.columns and "impressions" in df.columns:
        df["cpm_calc"] = (df["spend"] / df["impressions"].replace(0, np.nan)) * 1000
        if "cpm" not in df.columns:
            df["cpm"] = df["cpm_calc"]
    if "revenue" in df.columns and "spend" in df.columns:
        df["roas"] = df["revenue"] / df["spend"].replace(0, np.nan)
    if "spend" in df.columns and "purchases" in df.columns:
        df["cpa"] = df["spend"] / df["purchases"].replace(0, np.nan)
    if "clicks" in df.columns and "impressions" in df.columns:
        df["ctr_calc"] = df["clicks"] / df["impressions"].replace(0, np.nan)

    if add_recommendations:
        df["recommendation"] = df.apply(_classify_ad, axis=1)

    return df


def _classify_ad(row) -> str:
    spend = row.get("spend", 0)
    roas = row.get("roas", np.nan)
    ctr_val = row.get("ctr", np.nan)
    if pd.isna(ctr_val):
        ctr_val = row.get("ctr_calc", np.nan)
    impressions = row.get("impressions", 0)
    clicks = row.get("clicks", 0)

    if spend and spend > 5000 and (not pd.isna(roas) and roas < 1):
        return "PAUSE_LOW_ROAS"
    if spend and spend > 3000 and (not pd.isna(ctr_val) and ctr_val < 0.005):
        return "PAUSE_LOW_CTR"
    if impressions and impressions > 50000 and clicks == 0:
        return "PAUSE_NO_CLICKS"
    if spend and spend > 3000 and (not pd.isna(roas) and roas >= 2):
        return "SCALE_HIGH_ROAS"
    if (not pd.isna(ctr_val) and ctr_val >= 0.01) and (not pd.isna(roas) and roas >= 1.5):
        return "SCALE_GOOD_CTR_ROAS"
    return "MONITOR"


def build_summaries(df: pd.DataFrame):
    if "campaign_name" in df.columns:
        camp_summary = (
            df.groupby("campaign_name", dropna=False)
            .agg(
                {
                    "spend": "sum" if "spend" in df.columns else "size",
                    "revenue": "sum" if "revenue" in df.columns else "size",
                    "impressions": "sum" if "impressions" in df.columns else "size",
                    "clicks": "sum" if "clicks" in df.columns else "size",
                    "purchases": "sum" if "purchases" in df.columns else "size",
                    "roas": "mean" if "roas" in df.columns else "size",
                }
            )
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        camp_summary = []

    cols_for_ads = [
        "campaign_name",
        "adset_name",
        "ad_name",
        "status",
        "objective",
        "impressions",
        "clicks",
        "spend",
        "purchases",
        "revenue",
        "ctr",
        "roas",
        "cpc",
        "cpm_calc",
        "recommendation",
    ]
    existing_cols = [c for c in cols_for_ads if c in df.columns]
    ads_data = df[existing_cols].to_dict(orient="records")

    start_date = df["date_start"].min().date() if "date_start" in df.columns and df["date_start"].notna().any() else None
    end_date = df["date_end"].max().date() if "date_end" in df.columns and df["date_end"].notna().any() else None

    return camp_summary, ads_data, start_date, end_date
