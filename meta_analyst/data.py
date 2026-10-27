from __future__ import annotations

from typing import Any, BinaryIO

import numpy as np
import pandas as pd

# Full Meta Ads Manager export structure supported by this product.
# `aliases` are accepted header names from Ads Manager / API / normalized files.
META_REPORT_FIELDS: list[dict[str, Any]] = [
    {
        "key": "campaign_name",
        "label": "Campaign name",
        "group": "Identity",
        "required": True,
        "aliases": ["Campaign name", "campaign_name", "Campaign"],
    },
    {
        "key": "adset_name",
        "label": "Ad set name",
        "group": "Identity",
        "required": False,
        "aliases": ["Ad set name", "Adset name", "adset_name", "Ad Set Name"],
    },
    {
        "key": "ad_name",
        "label": "Ad name",
        "group": "Identity",
        "required": False,
        "aliases": ["Ad name", "ad_name", "Ad Name"],
    },
    {
        "key": "objective",
        "label": "Objective",
        "group": "Identity",
        "required": False,
        "aliases": ["Objective", "objective", "Campaign objective", "Campaign Objective"],
    },
    {
        "key": "status",
        "label": "Ad delivery",
        "group": "Identity",
        "required": False,
        "aliases": ["Ad delivery", "Delivery", "Delivery status", "status", "Ad Delivery"],
    },
    {
        "key": "date_start",
        "label": "Reporting starts",
        "group": "Dates",
        "required": False,
        "aliases": ["Reporting starts", "Reporting Starts", "date_start", "Start", "Day"],
    },
    {
        "key": "date_end",
        "label": "Reporting ends",
        "group": "Dates",
        "required": False,
        "aliases": ["Reporting ends", "Reporting Ends", "date_end", "End"],
    },
    {
        "key": "impressions",
        "label": "Impressions",
        "group": "Delivery",
        "required": True,
        "aliases": ["Impressions", "impressions"],
    },
    {
        "key": "reach",
        "label": "Reach",
        "group": "Delivery",
        "required": False,
        "aliases": ["Reach", "reach"],
    },
    {
        "key": "frequency",
        "label": "Frequency",
        "group": "Delivery",
        "required": False,
        "aliases": ["Frequency", "frequency"],
    },
    {
        "key": "clicks",
        "label": "Clicks (all)",
        "group": "Engagement",
        "required": True,
        "aliases": ["Clicks (all)", "Clicks", "clicks", "All clicks"],
    },
    {
        "key": "link_clicks",
        "label": "Link clicks",
        "group": "Engagement",
        "required": False,
        "aliases": ["Link clicks", "Outbound clicks", "link_clicks", "Unique link clicks"],
    },
    {
        "key": "landing_page_views",
        "label": "Landing page views",
        "group": "Engagement",
        "required": False,
        "aliases": ["Landing page views", "Landing Page Views", "landing_page_views", "LPV"],
    },
    {
        "key": "spend",
        "label": "Amount spent",
        "group": "Spend",
        "required": True,
        "aliases": [
            "Amount spent (INR)",
            "Amount spent (USD)",
            "Amount spent",
            "Amount Spent",
            "Spend",
            "spend",
            "Cost",
        ],
    },
    {
        "key": "cpc_all",
        "label": "CPC (all)",
        "group": "Spend",
        "required": False,
        "aliases": [
            "CPC (all) (INR)",
            "CPC (all) (USD)",
            "CPC (all)",
            "CPC",
            "cpc_all",
            "Cost per click (all)",
        ],
    },
    {
        "key": "cpm",
        "label": "CPM",
        "group": "Spend",
        "required": False,
        "aliases": [
            "CPM (cost per 1,000 impressions) (INR)",
            "CPM (cost per 1,000 impressions) (USD)",
            "CPM (cost per 1,000 impressions)",
            "CPM",
            "cpm",
        ],
    },
    {
        "key": "ctr",
        "label": "CTR (all)",
        "group": "Rates",
        "required": False,
        "aliases": ["CTR (all)", "CTR", "ctr", "Click-through rate"],
    },
    {
        "key": "lpv_lc",
        "label": "LPview/LC%",
        "group": "Rates",
        "required": False,
        "aliases": ["LPview/LC%", "LPV/LC%", "Landing page view rate"],
    },
    {
        "key": "imp_lpv",
        "label": "Imp to LPV %",
        "group": "Rates",
        "required": False,
        "aliases": ["Imp to LPV %", "Impressions to LPV %", "imp_lpv"],
    },
    {
        "key": "purchases",
        "label": "Purchases",
        "group": "Conversions",
        "required": False,
        "aliases": [
            "Purchases",
            "Website purchases",
            "purchases",
            "Omni purchases",
            "Results",
        ],
    },
    {
        "key": "revenue",
        "label": "Purchases conversion value",
        "group": "Conversions",
        "required": False,
        "aliases": [
            "Purchases conversion value",
            "Website purchases conversion value",
            "Conversion value",
            "Purchase conversion value",
            "revenue",
            "Website purchase ROAS (return on ad spend)",
        ],
    },
    {
        "key": "adds_to_cart",
        "label": "Adds to cart",
        "group": "Conversions",
        "required": False,
        "aliases": ["Adds to cart", "Website adds to cart", "adds_to_cart", "ATC"],
    },
    {
        "key": "checkouts",
        "label": "Checkouts initiated",
        "group": "Conversions",
        "required": False,
        "aliases": [
            "Checkouts initiated",
            "Website checkouts initiated",
            "checkouts",
            "Initiate checkout",
        ],
    },
    {
        "key": "leads",
        "label": "Leads",
        "group": "Conversions",
        "required": False,
        "aliases": ["Leads", "Website leads", "leads", "On-Facebook leads"],
    },
]


def meta_report_schema() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for field in META_REPORT_FIELDS:
        groups.setdefault(field["group"], []).append(
            {
                "key": field["key"],
                "label": field["label"],
                "required": field["required"],
                "aliases": field["aliases"],
            }
        )
    return {
        "description": (
            "Export from Meta Ads Manager as CSV or Excel at Ad / Ad set / Campaign level. "
            "Include the columns below for best results. Currency columns (INR/USD) are auto-detected."
        ),
        "required_keys": [f["key"] for f in META_REPORT_FIELDS if f["required"]],
        "fields": META_REPORT_FIELDS,
        "groups": groups,
        "template_headers": [f["label"] for f in META_REPORT_FIELDS],
    }


def pick_col(candidates, all_cols):
    col_list = list(all_cols)
    lower_map = {str(c).strip().lower(): c for c in col_list}
    for name in candidates:
        if name in all_cols:
            return name
        hit = lower_map.get(str(name).strip().lower())
        if hit is not None:
            return hit
    # Prefix match for "Amount spent (ANY_CURRENCY)"
    for name in candidates:
        prefix = str(name).split("(")[0].strip().lower()
        if not prefix:
            continue
        for raw, original in lower_map.items():
            if raw.startswith(prefix):
                return original
    return None


def detect_column_mapping(columns) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    for field in META_REPORT_FIELDS:
        mapping[field["key"]] = pick_col(field["aliases"], columns)
    return mapping


def mapping_report(source_columns, mapping: dict[str, str | None]) -> dict[str, Any]:
    found = []
    missing_required = []
    missing_optional = []
    for field in META_REPORT_FIELDS:
        key = field["key"]
        src = mapping.get(key)
        entry = {
            "key": key,
            "label": field["label"],
            "group": field["group"],
            "required": field["required"],
            "source_column": src,
            "mapped": bool(src),
        }
        if src:
            found.append(entry)
        elif field["required"]:
            missing_required.append(entry)
        else:
            missing_optional.append(entry)
    unmapped = [
        c
        for c in source_columns
        if c not in {m["source_column"] for m in found if m["source_column"]}
    ]
    return {
        "found": found,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "unmapped_source_columns": unmapped,
        "coverage": {
            "mapped": len(found),
            "total_supported": len(META_REPORT_FIELDS),
            "required_ok": len(missing_required) == 0,
        },
    }


def _read_dataframe(source: str | BinaryIO) -> pd.DataFrame:
    if isinstance(source, str):
        if source.lower().endswith(".csv"):
            return pd.read_csv(source)
        return pd.read_excel(source)

    name = getattr(source, "name", "").lower()
    if name.endswith(".csv") or not name:
        return pd.read_csv(source)
    return pd.read_excel(source)


def load_and_clean_data(
    source: str | BinaryIO | pd.DataFrame,
    *,
    add_recommendations: bool = False,
    active_only: bool = True,
) -> pd.DataFrame:
    """Load a Meta Ads export (or API DataFrame) and normalize columns/metrics."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = _read_dataframe(source)

    source_columns = [str(c) for c in df.columns]
    mapping = detect_column_mapping(df.columns)
    report = mapping_report(source_columns, mapping)
    rename_map = {src: key for key, src in mapping.items() if src}
    df = df.rename(columns=rename_map)
    df.attrs["column_mapping"] = mapping
    df.attrs["mapping_report"] = report

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        if active_only:
            df = df[df["status"].str.contains("active", na=False)].copy()

    num_cols = [
        "impressions",
        "reach",
        "frequency",
        "clicks",
        "link_clicks",
        "landing_page_views",
        "spend",
        "purchases",
        "revenue",
        "adds_to_cart",
        "checkouts",
        "leads",
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
    elif "ctr" in df.columns:
        # Meta CTR is often already a percent (e.g. 1.2); normalize if values look like percent
        ctr = df["ctr"]
        df["ctr_calc"] = np.where(ctr > 1, ctr / 100.0, ctr)

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

    start_date = (
        df["date_start"].min().date()
        if "date_start" in df.columns and df["date_start"].notna().any()
        else None
    )
    end_date = (
        df["date_end"].max().date()
        if "date_end" in df.columns and df["date_end"].notna().any()
        else None
    )

    return camp_summary, ads_data, start_date, end_date


def build_template_csv() -> str:
    headers = [f["label"] for f in META_REPORT_FIELDS]
    sample = {
        "Campaign name": "Prospecting - Purchases",
        "Ad set name": "Lookalike 1%",
        "Ad name": "Creative A",
        "Objective": "OUTCOME_SALES",
        "Ad delivery": "active",
        "Reporting starts": "2026-01-01",
        "Reporting ends": "2026-01-31",
        "Impressions": "100000",
        "Reach": "80000",
        "Frequency": "1.25",
        "Clicks (all)": "2500",
        "Link clicks": "2200",
        "Landing page views": "1800",
        "Amount spent": "50000",
        "CPC (all)": "20",
        "CPM": "500",
        "CTR (all)": "2.5",
        "LPview/LC%": "81.8",
        "Imp to LPV %": "1.8",
        "Purchases": "120",
        "Purchases conversion value": "180000",
        "Adds to cart": "400",
        "Checkouts initiated": "200",
        "Leads": "0",
    }
    row = [str(sample.get(h, "")) for h in headers]
    lines = [",".join(headers), ",".join(row)]
    return "\n".join(lines) + "\n"
