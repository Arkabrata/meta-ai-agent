import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# --------- CONFIG ---------
EXCEL_PATH = "data/meta_report.xlsx" 
MODEL_NAME = "gpt-5-mini"            
# --------------------------


def pick_col(candidates, all_cols):
    """
    Return the first column name from `candidates` that exists in `all_cols`,
    or None if none found.
    """
    for c in candidates:
        if c in all_cols:
            return c
    return None


def load_env_and_client():
    """Load .env and create OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Put it in .env file.")
    client = OpenAI(api_key=api_key)
    return client


def load_and_clean_data(excel_path: str) -> pd.DataFrame:
    """Load Meta Excel/CSV and clean/rename key columns."""
    # Read file (supports both .xlsx and .csv)
    if excel_path.lower().endswith(".csv"):
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)

    cols = df.columns

    # Dynamically detect important columns
    col_campaign = pick_col(["Campaign name", "campaign_name"], cols)
    col_adset = pick_col(["Ad set name", "adset_name"], cols)
    col_ad = pick_col(["Ad name", "ad_name"], cols)
    col_status = pick_col(["Ad delivery", "Delivery", "status"], cols)

    col_impr = pick_col(["Impressions", "impressions"], cols)
    col_clicks = pick_col(["Clicks (all)", "Clicks", "clicks"], cols)
    col_link_clicks = pick_col(["Link clicks", "Outbound clicks", "link_clicks"], cols)

    col_spend = pick_col(
        ["Amount spent (INR)", "Amount spent", "Spend", "spend"], cols
    )

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

    col_date_start = pick_col(["Reporting starts", "date_start"], cols)
    col_date_end = pick_col(["Reporting ends", "date_end"], cols)

    # Build rename map only for found columns
    rename_map = {}
    if col_campaign:
        rename_map[col_campaign] = "campaign_name"
    if col_adset:
        rename_map[col_adset] = "adset_name"
    if col_ad:
        rename_map[col_ad] = "ad_name"
    if col_status:
        rename_map[col_status] = "status"

    if col_impr:
        rename_map[col_impr] = "impressions"
    if col_clicks:
        rename_map[col_clicks] = "clicks"
    if col_link_clicks:
        rename_map[col_link_clicks] = "link_clicks"

    if col_spend:
        rename_map[col_spend] = "spend"
    if col_purchases:
        rename_map[col_purchases] = "purchases"
    if col_rev:
        rename_map[col_rev] = "revenue"

    if col_ctr:
        rename_map[col_ctr] = "ctr"
    if col_cpc_all:
        rename_map[col_cpc_all] = "cpc_all"
    if col_cpm:
        rename_map[col_cpm] = "cpm"

    if col_date_start:
        rename_map[col_date_start] = "date_start"
    if col_date_end:
        rename_map[col_date_end] = "date_end"

    df = df.rename(columns=rename_map)

    # 🔹 Keep only active ads (status = active)
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        before = len(df)
        df = df[df["status"] == "active"].copy()
        after = len(df)
        print(f"Filtered to active ads only: {before} → {after} rows")

    # numeric conversions (only where present)
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
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # date conversions
    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    if "date_end" in df.columns:
        df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")

    # derived metrics (only if ingredients exist)
    if "spend" in df.columns and "clicks" in df.columns:
        df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
    else:
        df["cpc"] = np.nan

    if "spend" in df.columns and "impressions" in df.columns:
        df["cpm_calc"] = (df["spend"] / df["impressions"].replace(0, np.nan)) * 1000
    else:
        df["cpm_calc"] = np.nan

    if "revenue" in df.columns and "spend" in df.columns:
        df["roas"] = df["revenue"] / df["spend"].replace(0, np.nan)
    else:
        df["roas"] = np.nan

    if "spend" in df.columns and "purchases" in df.columns:
        df["cpa"] = df["spend"] / df["purchases"].replace(0, np.nan)
    else:
        df["cpa"] = np.nan

    if "clicks" in df.columns and "impressions" in df.columns:
        df["ctr_calc"] = df["clicks"] / df["impressions"].replace(0, np.nan)
    else:
        df["ctr_calc"] = np.nan

    # basic rule-based recommendation tag
    def classify(row):
        spend = row.get("spend", 0)
        roas = row.get("roas", np.nan)
        ctr_val = row.get("ctr", np.nan)
        if pd.isna(ctr_val):
            ctr_val = row.get("ctr_calc", np.nan)
        impressions = row.get("impressions", 0)
        clicks = row.get("clicks", 0)

        # Pause logic
        if spend and spend > 5000 and (not pd.isna(roas) and roas < 1):
            return "PAUSE_LOW_ROAS"
        if spend and spend > 3000 and (not pd.isna(ctr_val) and ctr_val < 0.005):
            return "PAUSE_LOW_CTR"
        if impressions and impressions > 50000 and clicks == 0:
            return "PAUSE_NO_CLICKS"

        # Scale logic
        if spend and spend > 3000 and (not pd.isna(roas) and roas >= 2):
            return "SCALE_HIGH_ROAS"
        if (not pd.isna(ctr_val) and ctr_val >= 0.01) and (not pd.isna(roas) and roas >= 1.5):
            return "SCALE_GOOD_CTR_ROAS"

        return "MONITOR"

    if "spend" in df.columns:
        df["recommendation"] = df.apply(classify, axis=1)
    else:
        df["recommendation"] = "UNKNOWN"

    return df


def build_summaries(df: pd.DataFrame):
    """Build compact summaries for GPT (campaign + ALL ads)."""

    # Campaign level summary over the FULL dataset
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

    # Ad-level data: ALL active ads
    cols_for_ads = [
        "campaign_name",
        "adset_name",
        "ad_name",
        "status",
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
    existing_cols_for_ads = [c for c in cols_for_ads if c in df.columns]

    ads_data = df[existing_cols_for_ads].to_dict(orient="records")

    # Date range info
    if "date_start" in df.columns and df["date_start"].notna().any():
        start_date = df["date_start"].min().date()
    else:
        start_date = None

    if "date_end" in df.columns and df["date_end"].notna().any():
        end_date = df["date_end"].max().date()
    else:
        end_date = None

    return camp_summary, ads_data, start_date, end_date


def call_chatgpt_for_analysis(
    client: OpenAI,
    camp_summary,
    ads_data,
    start_date=None,
    end_date=None,
    model_name: str = MODEL_NAME,
) -> str:
    """Send data to ChatGPT and get analysis text."""
    date_info = f"{start_date} to {end_date}" if start_date and end_date else "the given date range"

    system_prompt = """
You are a senior performance marketing analyst for a D2C fashion brand.
You analyze Meta (Facebook/Instagram) ads and give clear, actionable recommendations.

Goals:
- Maximize revenue and ROAS
- Reduce wasted spend
- Give specific suggestions on what to pause, scale, or test.

You will receive:
- A campaign-level summary.
- A list of ALL active ads with their metrics (one object per row).
"""

    user_prompt = f"""
Here is Meta Ads performance data for {date_info}.

1) Campaign-level summary (list of objects):
{json.dumps(camp_summary, indent=2)}

2) Full ad-level data (each object is an active ad row, not truncated):
{json.dumps(ads_data, indent=2)}

Please:
- Identify the best and worst performing campaigns and explain why (use metrics).
- Tell me which ads I should PAUSE and why (refer to ad name and campaign).
- Tell me which ads I should SCALE and why.
- Comment on overall account health (CTR, CPC, ROAS, CPM).
- Suggest:
  - 3–5 concrete optimization actions (budget shifts, bids, placements, audiences, creatives).
  - 2–3 experiments to run (new creative angles, audience tests, campaign structure changes).

Focus on practical, direct recommendations I can implement in Ads Manager.
Use bullet points and headings.
"""

    try:
        response = client.responses.create(
            model=model_name,
            reasoning={"effort": "medium"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        print("❌ OpenAI API call failed:")
        print(e)
        return "OpenAI API error (likely quota/billing/permissions issue). Please check your OpenAI dashboard and try again."

    # Preferred helper – flattens text for us
    if getattr(response, "output_text", None):
        return response.output_text

    # Fallback if structure is unexpected
    print("❌ Unexpected response structure. Raw response object:")
    print(response)
    return "Unexpected API response structure. Check console logs for the raw response."




def main():
    print("🔹 Loading data...")
    df = load_and_clean_data(EXCEL_PATH)
    print(f"   Loaded {len(df)} active rows and {len(df.columns)} columns.")

    print("🔹 Building summaries (campaign + ALL active ads)...")
    camp_summary, ads_data, start_date, end_date = build_summaries(df)
    print(f"   Campaigns: {len(camp_summary)}, Active ads rows: {len(ads_data)}")

    print("🔹 Connecting to ChatGPT API...")
    client = load_env_and_client()

    print("🔹 Requesting analysis from ChatGPT...\n")
    analysis = call_chatgpt_for_analysis(
        client,
        camp_summary,
        ads_data,
        start_date,
        end_date,
    )

    print("\n================= META ADS ANALYSIS =================\n")
    print(analysis)
    print("\n=====================================================\n")

    export_cols = [
        "campaign_name",
        "adset_name",
        "ad_name",
        "status",
        "impressions",
        "clicks",
        "spend",
        "purchases",
        "revenue",
        "roas",
        "cpc",
        "cpm_calc",
        "recommendation",
    ]
    existing_export_cols = [c for c in export_cols if c in df.columns]

    export_path = "/Users/rarerabbit/Documents/Data Analysis/meta_ai_agent/output/meta_ads_with_recommendations.csv"
    df[existing_export_cols].to_csv(export_path, index=False)
    print(f"\n📁 Exported ad-level recommendations to: {export_path}\n")



if __name__ == "__main__":
    main()
