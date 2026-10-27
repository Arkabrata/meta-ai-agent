"""Meta (Facebook) Marketing / Reporting API helpers.

Meta does not accept Facebook username/password for the Marketing API.
Use a Graph API access token from Meta for Developers / Business Suite,
then select an ad account.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

GRAPH_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"


class MetaAPIError(Exception):
    """Raised when the Meta Graph API returns an error."""


def _raise_for_graph(payload: dict[str, Any]) -> None:
    err = payload.get("error")
    if not err:
        return
    message = err.get("message") or "Unknown Meta API error"
    code = err.get("code")
    raise MetaAPIError(f"Meta API error {code}: {message}" if code else message)


def _get(path: str, access_token: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    query = {"access_token": access_token, **(params or {})}
    url = path if path.startswith("http") else f"{GRAPH_BASE}/{path.lstrip('/')}"
    response = requests.get(url, params=query, timeout=60)
    try:
        payload = response.json()
    except ValueError as exc:
        raise MetaAPIError(f"Invalid JSON from Meta API (HTTP {response.status_code})") from exc
    _raise_for_graph(payload)
    if response.status_code >= 400:
        raise MetaAPIError(f"HTTP {response.status_code}: {payload}")
    return payload


def verify_token(access_token: str) -> dict[str, Any]:
    """Return /me profile for a valid user/system token."""
    return _get("me", access_token, {"fields": "id,name"})


def list_ad_accounts(access_token: str) -> list[dict[str, Any]]:
    """List ad accounts available to the token."""
    accounts: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "fields": "id,account_id,name,currency,account_status,business_name",
        "limit": 100,
    }
    path = "me/adaccounts"
    while path:
        payload = _get(path, access_token, params)
        accounts.extend(payload.get("data") or [])
        next_url = (payload.get("paging") or {}).get("next")
        path = next_url or ""
        params = {}  # next URL already includes query
    return accounts


def _normalize_account_id(account_id: str) -> str:
    account_id = str(account_id).strip()
    if account_id.startswith("act_"):
        return account_id
    return f"act_{account_id}"


# Fields requested from Graph API — mapped onto the same structure as the upload template
INSIGHT_FIELDS = [
    "campaign_name",
    "adset_name",
    "ad_name",
    "objective",
    "impressions",
    "reach",
    "frequency",
    "clicks",
    "inline_link_clicks",
    "spend",
    "ctr",
    "cpc",
    "cpm",
    "actions",
    "action_values",
    "date_start",
    "date_stop",
]

PURCHASE_TYPES = ("purchase", "omni_purchase", "offsite_conversion.fb_pixel_purchase")
ATC_TYPES = ("add_to_cart", "omni_add_to_cart", "offsite_conversion.fb_pixel_add_to_cart")
CHECKOUT_TYPES = (
    "initiate_checkout",
    "omni_initiated_checkout",
    "offsite_conversion.fb_pixel_initiate_checkout",
)
LEAD_TYPES = ("lead", "onsite_conversion.lead_grouped", "offsite_conversion.fb_pixel_lead")
LPV_TYPES = ("landing_page_view", "omni_landing_page_view")


def fetch_ad_insights(
    access_token: str,
    account_id: str,
    *,
    date_start: date | None = None,
    date_end: date | None = None,
    level: str = "ad",
    daily: bool = False,
) -> pd.DataFrame:
    """Pull ad-level insights and return a DataFrame shaped like a Meta Ads export.

    Output columns match `meta_analyst.data.META_REPORT_FIELDS` so API and upload
    flows share the same structure. `daily=True` adds one row per ad per day.
    """
    if date_end is None:
        date_end = date.today()
    if date_start is None:
        date_start = date_end - timedelta(days=29)

    act = _normalize_account_id(account_id)
    params: dict[str, Any] = {
        "level": level,
        "fields": ",".join(INSIGHT_FIELDS),
        "time_range": f'{{"since":"{date_start.isoformat()}","until":"{date_end.isoformat()}"}}',
        "limit": 500,
        "filtering": '[{"field":"ad.effective_status","operator":"IN","value":["ACTIVE","PAUSED","CAMPAIGN_PAUSED","ADSET_PAUSED"]}]',
    }
    if daily:
        params["time_increment"] = 1

    rows: list[dict[str, Any]] = []
    path = f"{act}/insights"
    while path:
        payload = _get(path, access_token, params)
        rows.extend(payload.get("data") or [])
        next_url = (payload.get("paging") or {}).get("next")
        path = next_url or ""
        params = {}

    if not rows:
        return pd.DataFrame()

    records = [_insight_to_export_row(r) for r in rows]
    return pd.DataFrame(records)


def _action_value(actions: list[dict] | None, action_types: tuple[str, ...]) -> float:
    if not actions:
        return 0.0
    total = 0.0
    for item in actions:
        if item.get("action_type") in action_types:
            try:
                total += float(item.get("value") or 0)
            except (TypeError, ValueError):
                continue
    return total


def _num(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _insight_to_export_row(row: dict[str, Any]) -> dict[str, Any]:
    """Shape one Graph API insight row into the full Meta report template structure."""
    actions = row.get("actions")
    purchases = _action_value(actions, PURCHASE_TYPES)
    revenue = _action_value(row.get("action_values"), PURCHASE_TYPES)
    adds_to_cart = _action_value(actions, ATC_TYPES)
    checkouts = _action_value(actions, CHECKOUT_TYPES)
    leads = _action_value(actions, LEAD_TYPES)
    lpv = _action_value(actions, LPV_TYPES)

    impressions = _num(row.get("impressions"))
    link_clicks = _num(row.get("inline_link_clicks"))
    lpv_lc = round(lpv / link_clicks * 100, 2) if link_clicks else None
    imp_lpv = round(lpv / impressions * 100, 2) if impressions else None

    return {
        "Campaign name": row.get("campaign_name"),
        "Ad set name": row.get("adset_name"),
        "Ad name": row.get("ad_name"),
        "Objective": row.get("objective"),
        "Ad delivery": "active",
        "Reporting starts": row.get("date_start"),
        "Reporting ends": row.get("date_stop"),
        "Impressions": impressions,
        "Reach": _num(row.get("reach")),
        "Frequency": _num(row.get("frequency")),
        "Clicks (all)": _num(row.get("clicks")),
        "Link clicks": link_clicks,
        "Landing page views": lpv or None,
        "Amount spent": _num(row.get("spend")),
        "CPC (all)": _num(row.get("cpc")),
        "CPM": _num(row.get("cpm")),
        "CTR (all)": _num(row.get("ctr")),
        "LPview/LC%": lpv_lc,
        "Imp to LPV %": imp_lpv,
        "Purchases": purchases,
        "Purchases conversion value": revenue,
        "Adds to cart": adds_to_cart,
        "Checkouts initiated": checkouts,
        "Leads": leads,
    }


def account_label(account: dict[str, Any]) -> str:
    name = account.get("name") or "Unnamed"
    aid = account.get("account_id") or account.get("id", "")
    currency = account.get("currency") or ""
    status = account.get("account_status")
    status_txt = {1: "ACTIVE", 2: "DISABLED", 3: "UNSETTLED"}.get(status, str(status or ""))
    bits = [f"{name} ({aid})"]
    if currency:
        bits.append(currency)
    if status_txt:
        bits.append(status_txt)
    return " · ".join(bits)
