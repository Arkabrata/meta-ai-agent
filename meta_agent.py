"""Meta Analyst CLI — batch analysis and CSV export."""

import os

from meta_analyst.analysis import call_cli_analysis, get_openai_client
from meta_analyst.config import CLI_MODEL, META_INPUT_PATH, META_OUTPUT_PATH
from meta_analyst.data import build_summaries, load_and_clean_data


def main():
    print("Meta Analyst CLI")
    print("Loading data...")
    df = load_and_clean_data(META_INPUT_PATH, add_recommendations=True)
    print(f"Loaded {len(df)} active rows and {len(df.columns)} columns.")

    print("Building summaries...")
    camp_summary, ads_data, start_date, end_date = build_summaries(df)
    print(f"Campaigns: {len(camp_summary)}, Active ads: {len(ads_data)}")

    print("Connecting to OpenAI...")
    client = get_openai_client()

    print(f"Requesting analysis ({CLI_MODEL})...\n")
    analysis = call_cli_analysis(
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
    existing_cols = [c for c in export_cols if c in df.columns]
    os.makedirs(os.path.dirname(META_OUTPUT_PATH) or ".", exist_ok=True)
    df[existing_cols].to_csv(META_OUTPUT_PATH, index=False)
    print(f"Exported recommendations to: {META_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
