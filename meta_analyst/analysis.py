import json

import numpy as np
import pandas as pd
from openai import OpenAI

from meta_analyst.config import ANALYSIS_MODEL, CHAT_MODEL, CLI_MODEL


def get_openai_client() -> OpenAI:
    from meta_analyst.config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")
    return OpenAI(api_key=OPENAI_API_KEY)


def llm_chart_insight(client: OpenAI, df: pd.DataFrame, chart_key: str) -> str | None:
    if chart_key == "objective_spend_pie":
        data = df.groupby("objective")["spend"].sum().to_dict()
        title = "Spend Distribution by Objective"
    elif chart_key == "objective_roas":
        data = (
            df.groupby("objective")
            .apply(lambda x: x["revenue"].sum() / x["spend"].sum())
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .to_dict()
        )
        title = "ROAS by Objective"
    elif chart_key == "scatter_spend_roas":
        data = {
            "spend_min": float(df["spend"].min()),
            "spend_max": float(df["spend"].max()),
            "roas_min": float(df["roas"].min()),
            "roas_max": float(df["roas"].max()),
        }
        title = "Spend vs ROAS Relationship"
    elif chart_key == "ctr_heatmap":
        data = (
            df.groupby("campaign_name")["ctr_calc"]
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .to_dict()
        )
        title = "CTR by Campaign"
    else:
        return None

    system_prompt = f"""
You are a senior Meta Ads performance analyst.
Write a sharp insight block for the chart titled: {title}

Rules:
- Start with 1 crisp summary sentence
- Then give 4-6 bullet points
- Keep it actionable (budget shift, creative ideas, audience issues)
- Keep it short and high signal only
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(data, indent=2)},
        ],
    )
    return response.output_text.strip()


def llm_detailed_analysis(client: OpenAI, df: pd.DataFrame) -> str:
    camp_summary = (
        df.groupby("campaign_name")
        .agg({"spend": "sum", "revenue": "sum", "impressions": "sum", "roas": "mean"})
        .reset_index()
        .to_dict(orient="records")
    )
    ads_data = df.sort_values("spend", ascending=False).head(200).to_dict(orient="records")

    system = """
You are a senior Meta Ads strategist.
Return JSON like:

{
 "overall_summary":"",
 "priority_blocks":[
     {"priority":"HIGH","title":"","details":""},
     {"priority":"MEDIUM","title":"","details":""},
     {"priority":"LOW","title":"","details":""}
 ]
}
"""

    user = f"""
Campaign Summary:
{json.dumps(camp_summary, indent=2)}

Top Ads:
{json.dumps(ads_data, indent=2)}
"""

    response = client.responses.create(
        model=ANALYSIS_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.output_text


def llm_chat_answer(client: OpenAI, df: pd.DataFrame, user_input: str) -> str:
    compact = {
        "campaign_summary": df.groupby("campaign_name")
        .agg({"spend": "sum", "revenue": "sum", "roas": "mean"})
        .reset_index()
        .to_dict(orient="records"),
        "top_ads": df.sort_values("spend", ascending=False).head(100).to_dict(orient="records"),
    }

    system = """
You are a senior Meta Ads performance strategist.
Answer clearly, concisely, and with specific recommendations.
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{json.dumps(compact, indent=2)}"},
            {"role": "user", "content": user_input},
        ],
    )
    return response.output_text


def call_cli_analysis(
    client: OpenAI,
    camp_summary,
    ads_data,
    start_date=None,
    end_date=None,
    model_name: str = CLI_MODEL,
) -> str:
    date_info = f"{start_date} to {end_date}" if start_date and end_date else "the given date range"

    system_prompt = """
You are a senior performance marketing analyst for a D2C brand.
You analyze Meta (Facebook/Instagram) ads and give clear, actionable recommendations.

Goals:
- Maximize revenue and ROAS
- Reduce wasted spend
- Give specific suggestions on what to pause, scale, or test.
"""

    user_prompt = f"""
Here is Meta Ads performance data for {date_info}.

1) Campaign-level summary:
{json.dumps(camp_summary, indent=2)}

2) Full ad-level data:
{json.dumps(ads_data, indent=2)}

Please:
- Identify the best and worst performing campaigns and explain why.
- Tell me which ads to PAUSE and which to SCALE.
- Comment on overall account health (CTR, CPC, ROAS, CPM).
- Suggest 3-5 optimization actions and 2-3 experiments.

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
    except Exception as exc:
        print("OpenAI API call failed:", exc)
        return "OpenAI API error. Check your API key, quota, and billing settings."

    if getattr(response, "output_text", None):
        return response.output_text
    return "Unexpected API response structure."
