"""Generate generic Meta Ads report rows for local demos."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

from meta_analyst.data import load_and_clean_data

OBJECTIVES = [
    "OUTCOME_SALES",
    "OUTCOME_TRAFFIC",
    "OUTCOME_ENGAGEMENT",
    "OUTCOME_LEADS",
    "OUTCOME_AWARENESS",
]
CAMPAIGNS = [
    ("Prospecting - Purchases", "OUTCOME_SALES"),
    ("Retargeting - ATC", "OUTCOME_SALES"),
    ("Brand Awareness - Video", "OUTCOME_AWARENESS"),
    ("Traffic - Blog", "OUTCOME_TRAFFIC"),
    ("Lead Gen - Webinar", "OUTCOME_LEADS"),
    ("Engagement - Social", "OUTCOME_ENGAGEMENT"),
    ("ASC+ Catalog", "OUTCOME_SALES"),
    ("Lookalike - Checkout", "OUTCOME_SALES"),
]
ADSETS = [
    "Broad 25-45",
    "Lookalike 1%",
    "Lookalike 3%",
    "Interest - Fashion",
    "Retarget 7d",
    "Retarget 30d",
    "Advantage+ audience",
]
CREATIVES = [
    "UGC Hook A",
    "Carousel B",
    "Static Offer C",
    "Video 15s D",
    "Testimonial E",
    "Collection F",
]


def generate_raw_meta_rows(n: int = 150, *, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    end = date.today()
    window_start = end - timedelta(days=29)
    rows = []
    for i in range(n):
        # Daily-breakdown style: each row is one ad on one day, like a Meta export with "Day" breakdown
        day = window_start + timedelta(days=rng.randint(0, 29))
        start = day
        end_day = day
        camp, obj = CAMPAIGNS[i % len(CAMPAIGNS)]
        # Slight objective noise
        if rng.random() < 0.15:
            obj = rng.choice(OBJECTIVES)
        adset = ADSETS[i % len(ADSETS)]
        ad = f"{CREATIVES[i % len(CREATIVES)]} #{i + 1:03d}"
        impressions = rng.randint(8_000, 220_000)
        ctr_pct = rng.uniform(0.4, 3.8)
        clicks = max(1, int(impressions * ctr_pct / 100))
        link_clicks = max(1, int(clicks * rng.uniform(0.75, 0.95)))
        spend = round(rng.uniform(800, 45_000), 2)
        # Performance tiers
        tier = rng.random()
        if tier < 0.2:
            roas = rng.uniform(0.2, 0.9)
        elif tier < 0.55:
            roas = rng.uniform(1.0, 1.8)
        else:
            roas = rng.uniform(1.9, 4.5)
        revenue = round(spend * roas, 2)
        purchases = max(0, int(revenue / rng.uniform(800, 2200))) if obj == "OUTCOME_SALES" else rng.randint(0, 8)
        if obj == "OUTCOME_LEADS":
            leads = rng.randint(5, 120)
            purchases = 0
            revenue = 0.0
        else:
            leads = rng.randint(0, 15)
        atc = purchases * rng.randint(2, 6) if purchases else rng.randint(0, 40)
        checkouts = max(purchases, int(atc * rng.uniform(0.3, 0.7))) if atc else 0
        lpv = max(1, int(link_clicks * rng.uniform(0.55, 0.9)))
        reach = max(1, int(impressions / rng.uniform(1.05, 1.8)))
        frequency = round(impressions / reach, 2)
        cpc = round(spend / clicks, 2)
        cpm = round(spend / impressions * 1000, 2)
        status = "active" if rng.random() > 0.12 else "inactive"
        rows.append(
            {
                "Campaign name": camp,
                "Ad set name": adset,
                "Ad name": ad,
                "Objective": obj,
                "Ad delivery": status,
                "Reporting starts": start.isoformat(),
                "Reporting ends": end_day.isoformat(),
                "Impressions": impressions,
                "Reach": reach,
                "Frequency": frequency,
                "Clicks (all)": clicks,
                "Link clicks": link_clicks,
                "Landing page views": lpv,
                "Amount spent": spend,
                "CPC (all)": cpc,
                "CPM": cpm,
                "CTR (all)": round(ctr_pct, 2),
                "LPview/LC%": round(lpv / link_clicks * 100, 2),
                "Imp to LPV %": round(lpv / impressions * 100, 2),
                "Purchases": purchases,
                "Purchases conversion value": revenue,
                "Adds to cart": atc,
                "Checkouts initiated": checkouts,
                "Leads": leads,
            }
        )
    return pd.DataFrame(rows)


def generate_clean_sample(n: int = 150, *, seed: int = 42) -> pd.DataFrame:
    raw = generate_raw_meta_rows(n, seed=seed)
    return load_and_clean_data(raw, active_only=False)
