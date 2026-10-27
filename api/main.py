"""FastAPI backend for Meta Analyst React dashboard."""

from __future__ import annotations

import io
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.session import Session, store
from meta_analyst import __product_name__, __version__
from meta_analyst.analysis import llm_chart_insight, llm_chat_answer, llm_detailed_analysis
from meta_analyst.charts import create_charts, kpi_metrics
from meta_analyst.config import (
    APP_PASSWORD,
    COMPANY_NAME,
    CURRENCY_SYMBOL,
    META_ACCESS_TOKEN,
    META_AD_ACCOUNT_ID,
    PRODUCT_NAME,
)
from meta_analyst.data import (
    build_template_csv,
    detect_column_mapping,
    load_and_clean_data,
    mapping_report,
    meta_report_schema,
)
from meta_analyst.meta_api import (
    MetaAPIError,
    account_label,
    fetch_ad_insights,
    list_ad_accounts,
    verify_token,
)
from meta_analyst.pdf import generate_full_pdf
from meta_analyst.providers import (
    DEFAULT_MODELS,
    PROVIDER_MODELS,
    PROVIDER_OPTIONS,
    ProviderConfig,
    get_session_client,
)
from meta_analyst.store import (
    SAMPLE_DATASET_ID,
    db_info,
    init_db,
    list_datasets,
    load_clean_dataset,
    save_dataframe,
)

app = FastAPI(title=PRODUCT_NAME, version=__version__)


@app.on_event("startup")
def _startup() -> None:
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5280",
        "http://127.0.0.1:5280",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_session(authorization: str | None = Header(default=None)) -> Session:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    session = store.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session


# ----- models -----


class LoginRequest(BaseModel):
    password: str


class MetaConnectRequest(BaseModel):
    access_token: str
    display_name: str = ""


class MetaFetchRequest(BaseModel):
    account_id: str
    date_start: date | None = None
    date_end: date | None = None
    daily: bool = False


class ProviderUpsert(BaseModel):
    provider: str
    api_key: str
    model: str = ""
    base_url: str = ""
    label: str = ""


class ActiveProviderRequest(BaseModel):
    provider: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


# ----- auth / health -----


@app.get("/api/health")
def health():
    return {"ok": True, "product": __product_name__, "version": __version__}


@app.get("/api/brand")
def brand():
    return {
        "product": PRODUCT_NAME,
        "company": COMPANY_NAME,
        "currency": CURRENCY_SYMBOL,
        "meta_token_configured": bool(META_ACCESS_TOKEN),
        "default_ad_account": META_AD_ACCOUNT_ID or None,
    }


@app.post("/api/auth/login")
def login(body: LoginRequest):
    if not APP_PASSWORD:
        raise HTTPException(500, "APP_PASSWORD is not configured on the server")
    if body.password != APP_PASSWORD:
        raise HTTPException(401, "Incorrect password")
    session = store.create()
    return {"token": session.token, "product": PRODUCT_NAME, "company": COMPANY_NAME}


@app.post("/api/auth/logout")
def logout(session: Session = Depends(require_session)):
    store.delete(session.token)
    return {"ok": True}


@app.get("/api/session")
def session_info(session: Session = Depends(require_session)):
    active = session.provider_store.get_active()
    return {
        "has_data": session.df is not None and not session.df.empty,
        "row_count": 0 if session.df is None else len(session.df),
        "meta": {
            "connected": bool(session.meta.get("access_token")),
            "display_name": session.meta.get("display_name"),
            "user_name": session.meta.get("user_name"),
            "account_label": session.meta.get("account_label"),
            "account_id": session.meta.get("account_id"),
        },
        "provider": None
        if not active
        else {
            "id": active.provider,
            "label": active.label or active.provider,
            "model": active.resolved_model(),
            "has_key": bool(active.api_key),
        },
    }


# ----- upload / dataset -----


def _df_preview(df: pd.DataFrame, n: int = 25) -> list[dict[str, Any]]:
    preview = df.head(n).copy()
    for col in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[col]):
            preview[col] = preview[col].astype(str)
    return json.loads(preview.to_json(orient="records", date_format="iso"))


@app.get("/api/data/schema")
def data_schema(session: Session = Depends(require_session)):
    return meta_report_schema()


@app.get("/api/data/store")
def data_store_info(session: Session = Depends(require_session)):
    """SQLite file store — no external database server."""
    return db_info()


@app.get("/api/data/datasets")
def data_datasets(session: Session = Depends(require_session)):
    return {"datasets": list_datasets()}


class LoadDatasetRequest(BaseModel):
    dataset_id: str = SAMPLE_DATASET_ID


@app.post("/api/data/load")
def data_load(body: LoadDatasetRequest, session: Session = Depends(require_session)):
    try:
        df = load_clean_dataset(body.dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, f"Could not load dataset: {exc}") from exc
    session.df = df
    session.chart_insights.clear()
    session.detailed_analysis = None
    session.chat_history.clear()
    return {
        "dataset_id": body.dataset_id,
        "row_count": len(df),
        "columns": list(df.columns),
        "preview": _df_preview(df),
    }


@app.post("/api/data/sample")
def data_sample(session: Session = Depends(require_session)):
    """Load the built-in 150-row generic Meta demo into the session."""
    df = load_clean_dataset(SAMPLE_DATASET_ID)
    session.df = df
    session.chart_insights.clear()
    session.detailed_analysis = None
    session.chat_history.clear()
    return {
        "dataset_id": SAMPLE_DATASET_ID,
        "row_count": len(df),
        "columns": list(df.columns),
        "preview": _df_preview(df),
        "store": db_info(),
    }


@app.get("/api/data/template.csv")
def data_template(session: Session = Depends(require_session)):
    from fastapi.responses import Response

    return Response(
        content=build_template_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="meta_ads_report_template.csv"'},
    )


@app.post("/api/data/upload")
async def upload_report(
    file: UploadFile = File(...),
    session: Session = Depends(require_session),
):
    raw = await file.read()
    name = (file.filename or "report.csv").lower()
    bio = io.BytesIO(raw)
    bio.name = name
    try:
        if name.endswith(".csv"):
            source_df = pd.read_csv(io.BytesIO(raw))
        else:
            source_df = pd.read_excel(io.BytesIO(raw))
        mapping = detect_column_mapping(source_df.columns)
        report = mapping_report([str(c) for c in source_df.columns], mapping)
        bio = io.BytesIO(raw)
        bio.name = name
        df = load_and_clean_data(bio)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse file: {exc}") from exc
    if report["missing_required"]:
        missing = ", ".join(m["label"] for m in report["missing_required"])
        raise HTTPException(
            400,
            f"Missing required Meta columns: {missing}. Download the template for the full structure.",
        )
    # Persist upload into embedded SQLite (no DB server)
    dataset_id = f"upload_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
    save_dataframe(
        dataset_id,
        name=file.filename or dataset_id,
        source="upload",
        df=source_df,
        meta={"filename": file.filename},
    )
    session.df = df
    session.chart_insights.clear()
    session.detailed_analysis = None
    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "preview": _df_preview(df),
        "mapping": report,
        "dataset_id": dataset_id,
    }


@app.get("/api/data/preview")
def data_preview(session: Session = Depends(require_session)):
    if session.df is None:
        raise HTTPException(404, "No dataset loaded")
    return {
        "row_count": len(session.df),
        "columns": list(session.df.columns),
        "preview": _df_preview(session.df),
    }


# ----- Meta API -----


@app.post("/api/meta/connect")
def meta_connect(body: MetaConnectRequest, session: Session = Depends(require_session)):
    token = body.access_token.strip() or META_ACCESS_TOKEN
    if not token:
        raise HTTPException(400, "Access token is required")
    try:
        profile = verify_token(token)
        accounts = list_ad_accounts(token)
    except MetaAPIError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Could not reach Meta API: {exc}") from exc

    session.meta = {
        "access_token": token,
        "display_name": body.display_name.strip() or profile.get("name") or "Meta user",
        "user_id": profile.get("id"),
        "user_name": profile.get("name"),
    }
    session.meta_accounts = accounts
    return {
        "user": profile,
        "accounts": [
            {
                "id": a.get("id"),
                "account_id": a.get("account_id"),
                "name": a.get("name"),
                "currency": a.get("currency"),
                "account_status": a.get("account_status"),
                "label": account_label(a),
            }
            for a in accounts
        ],
    }


@app.get("/api/meta/accounts")
def meta_accounts(session: Session = Depends(require_session)):
    if not session.meta.get("access_token"):
        raise HTTPException(400, "Connect a Meta token first")
    return {
        "accounts": [
            {
                "id": a.get("id"),
                "account_id": a.get("account_id"),
                "name": a.get("name"),
                "currency": a.get("currency"),
                "account_status": a.get("account_status"),
                "label": account_label(a),
            }
            for a in session.meta_accounts
        ]
    }


@app.post("/api/meta/fetch")
def meta_fetch(body: MetaFetchRequest, session: Session = Depends(require_session)):
    token = session.meta.get("access_token")
    if not token:
        raise HTTPException(400, "Connect a Meta token first")
    date_end = body.date_end or date.today()
    date_start = body.date_start or (date_end - timedelta(days=29))
    try:
        raw = fetch_ad_insights(
            token,
            body.account_id,
            date_start=date_start,
            date_end=date_end,
            daily=body.daily,
        )
    except MetaAPIError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Failed to fetch insights: {exc}") from exc

    if raw.empty:
        raise HTTPException(404, "No insight rows returned for this range")

    df = load_and_clean_data(raw, active_only=False)
    session.df = df
    session.chart_insights.clear()
    session.detailed_analysis = None

    match = next(
        (
            a
            for a in session.meta_accounts
            if str(a.get("account_id")) == str(body.account_id)
            or str(a.get("id")) == str(body.account_id)
            or f"act_{a.get('account_id')}" == str(body.account_id)
        ),
        None,
    )
    session.meta["account_id"] = body.account_id
    session.meta["account_label"] = account_label(match) if match else body.account_id

    return {
        "row_count": len(df),
        "account_label": session.meta["account_label"],
        "preview": _df_preview(df),
    }


# ----- providers -----


@app.get("/api/providers")
def list_providers(session: Session = Depends(require_session)):
    items = []
    for key, conf in session.provider_store.configs.items():
        masked = ("*" * 8 + conf.api_key[-4:]) if len(conf.api_key) >= 4 else "****"
        items.append(
            {
                "provider": key,
                "label": conf.label or key,
                "model": conf.resolved_model(),
                "base_url": conf.base_url,
                "masked_key": masked,
                "active": key == session.provider_store.active,
            }
        )
    return {
        "options": [{"id": k, "label": v} for k, v in PROVIDER_OPTIONS],
        "defaults": DEFAULT_MODELS,
        "models": PROVIDER_MODELS,
        "active": session.provider_store.active,
        "configured": items,
        "usage": session.usage.as_dict(),
    }


@app.get("/api/providers/usage")
def provider_usage(session: Session = Depends(require_session)):
    return session.usage.as_dict()


@app.delete("/api/providers/usage")
def reset_provider_usage(session: Session = Depends(require_session)):
    from meta_analyst.providers import UsageMeter

    session.usage = UsageMeter()
    return session.usage.as_dict()


@app.post("/api/providers")
def upsert_provider(body: ProviderUpsert, session: Session = Depends(require_session)):
    if not body.api_key.strip():
        raise HTTPException(400, "API key is required")
    label_map = dict(PROVIDER_OPTIONS)
    session.provider_store.upsert(
        ProviderConfig(
            provider=body.provider,
            api_key=body.api_key.strip(),
            model=body.model.strip() or DEFAULT_MODELS.get(body.provider, ""),
            base_url=body.base_url.strip(),
            label=body.label.strip() or label_map.get(body.provider, body.provider),
        )
    )
    session.chart_insights.clear()
    return {"ok": True, "active": session.provider_store.active}


@app.post("/api/providers/active")
def set_active_provider(body: ActiveProviderRequest, session: Session = Depends(require_session)):
    if body.provider not in session.provider_store.configs:
        raise HTTPException(404, "Provider not configured")
    session.provider_store.active = body.provider
    session.chart_insights.clear()
    return {"ok": True, "active": body.provider}


@app.delete("/api/providers/{provider}")
def delete_provider(provider: str, session: Session = Depends(require_session)):
    if not session.provider_store.remove(provider):
        raise HTTPException(404, "Provider not configured")
    session.chart_insights.clear()
    return {"ok": True, "active": session.provider_store.active or None}


def _llm(session: Session, *, feature: str = ""):
    try:
        return get_session_client(session.provider_store, session.usage, feature=feature)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ----- dashboard / analysis -----


def _require_df(session: Session) -> pd.DataFrame:
    if session.df is None or session.df.empty:
        raise HTTPException(400, "Load data via Meta API or upload first")
    return session.df


def _short_label(name: str, n: int = 22) -> str:
    s = str(name)
    return s if len(s) <= n else s[: n - 1] + "…"


@app.get("/api/dashboard")
def dashboard(session: Session = Depends(require_session)):
    df = _require_df(session)
    kpis = [{"name": n, "value": v} for n, v in kpi_metrics(df)]
    currency = CURRENCY_SYMBOL
    charts: dict[str, Any] = {}

    # Objective: spend share + ROAS + spend/revenue paired
    if "objective" in df.columns and "spend" in df.columns:
        spend = df.groupby("objective")["spend"].sum().sort_values(ascending=False)
        total_spend = float(spend.sum()) or 1.0
        charts["objective_spend"] = [
            {
                "name": str(k).replace("OUTCOME_", ""),
                "full_name": str(k),
                "value": float(v),
                "pct": round(float(v) / total_spend * 100, 1),
            }
            for k, v in spend.items()
        ]
        if "revenue" in df.columns:
            summary = df.groupby("objective").agg(
                revenue=("revenue", "sum"),
                spend=("spend", "sum"),
                impressions=("impressions", "sum") if "impressions" in df.columns else ("spend", "size"),
                clicks=("clicks", "sum") if "clicks" in df.columns else ("spend", "size"),
            )
            summary["roas"] = summary["revenue"] / summary["spend"].replace(0, pd.NA)
            summary = summary.fillna(0).sort_values("spend", ascending=False)
            charts["objective_roas"] = [
                {
                    "name": str(i).replace("OUTCOME_", ""),
                    "full_name": str(i),
                    "value": round(float(r.roas), 2),
                    "spend": round(float(r.spend), 0),
                    "revenue": round(float(r.revenue), 0),
                }
                for i, r in summary.iterrows()
            ]
            charts["objective_spend_revenue"] = [
                {
                    "name": str(i).replace("OUTCOME_", ""),
                    "spend": round(float(r.spend), 0),
                    "revenue": round(float(r.revenue), 0),
                    "roas": round(float(r.roas), 2),
                }
                for i, r in summary.iterrows()
            ]

    # Campaign performance bars + table
    if "campaign_name" in df.columns and "spend" in df.columns:
        camp = df.groupby("campaign_name").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum") if "revenue" in df.columns else ("spend", "size"),
            impressions=("impressions", "sum") if "impressions" in df.columns else ("spend", "size"),
            clicks=("clicks", "sum") if "clicks" in df.columns else ("spend", "size"),
            purchases=("purchases", "sum") if "purchases" in df.columns else ("spend", "size"),
        )
        if "revenue" in df.columns:
            camp["roas"] = camp["revenue"] / camp["spend"].replace(0, pd.NA)
        else:
            camp["roas"] = 0
        if "clicks" in df.columns and "impressions" in df.columns:
            camp["ctr"] = camp["clicks"] / camp["impressions"].replace(0, pd.NA) * 100
        else:
            camp["ctr"] = 0
        if "clicks" in df.columns:
            camp["cpc"] = camp["spend"] / camp["clicks"].replace(0, pd.NA)
        else:
            camp["cpc"] = 0
        camp = camp.fillna(0).sort_values("spend", ascending=False)
        charts["campaign_spend"] = [
            {
                "name": _short_label(i, 18),
                "full_name": str(i),
                "spend": round(float(r.spend), 0),
                "revenue": round(float(r.revenue), 0),
                "roas": round(float(r.roas), 2),
                "ctr": round(float(r.ctr), 2),
            }
            for i, r in camp.head(10).iterrows()
        ]
        charts["ctr_by_campaign"] = [
            {
                "name": _short_label(i, 16),
                "full_name": str(i),
                "value": round(float(r.ctr), 2),
                "spend": round(float(r.spend), 0),
            }
            for i, r in camp.sort_values("ctr", ascending=False).head(10).iterrows()
        ]
        charts["campaign_table"] = [
            {
                "campaign": str(i),
                "spend": round(float(r.spend), 0),
                "revenue": round(float(r.revenue), 0),
                "roas": round(float(r.roas), 2),
                "ctr": round(float(r.ctr), 2),
                "cpc": round(float(r.cpc), 2),
                "clicks": int(r.clicks),
                "purchases": int(r.purchases) if "purchases" in camp.columns else 0,
            }
            for i, r in camp.head(12).iterrows()
        ]

    # Funnel
    funnel_steps = []
    if "impressions" in df.columns:
        funnel_steps.append(("Impressions", float(df["impressions"].sum())))
    if "clicks" in df.columns:
        funnel_steps.append(("Clicks", float(df["clicks"].sum())))
    if "landing_page_views" in df.columns:
        funnel_steps.append(("LP Views", float(df["landing_page_views"].sum())))
    elif "link_clicks" in df.columns:
        funnel_steps.append(("Link clicks", float(df["link_clicks"].sum())))
    if "adds_to_cart" in df.columns and df["adds_to_cart"].sum() > 0:
        funnel_steps.append(("Add to cart", float(df["adds_to_cart"].sum())))
    if "purchases" in df.columns:
        funnel_steps.append(("Purchases", float(df["purchases"].sum())))
    if len(funnel_steps) >= 2:
        top = funnel_steps[0][1] or 1.0
        charts["funnel"] = [
            {
                "name": name,
                "value": round(val, 0),
                "pct": round(val / top * 100, 2),
            }
            for name, val in funnel_steps
        ]

    # Scatter with campaign labels + purchases size
    if {"spend", "roas"}.issubset(df.columns):
        sample = df.dropna(subset=["spend", "roas"]).copy()
        if "campaign_name" in sample.columns:
            sample = sample.sort_values("spend", ascending=False).head(200)
        else:
            sample = sample.head(200)
        charts["scatter_spend_roas"] = [
            {
                "spend": float(r.spend),
                "roas": float(r.roas),
                "campaign": str(getattr(r, "campaign_name", ""))[:28] if "campaign_name" in sample.columns else "",
                "purchases": float(getattr(r, "purchases", 0) or 0),
            }
            for r in sample.itertuples()
        ]

    # Top / bottom ads
    if {"ad_name", "spend"}.issubset(df.columns):
        ads = df.copy()
        if "roas" not in ads.columns and {"revenue", "spend"}.issubset(ads.columns):
            ads["roas"] = ads["revenue"] / ads["spend"].replace(0, pd.NA)
        ads = ads.fillna({"roas": 0, "purchases": 0})
        top = ads.sort_values(["roas", "spend"], ascending=[False, False]).head(8)
        weak = ads[ads["spend"] > ads["spend"].quantile(0.4)].sort_values("roas").head(8)
        charts["top_ads"] = [
            {
                "ad": str(r.ad_name)[:40],
                "campaign": str(getattr(r, "campaign_name", ""))[:28],
                "spend": round(float(r.spend), 0),
                "roas": round(float(getattr(r, "roas", 0) or 0), 2),
                "purchases": int(getattr(r, "purchases", 0) or 0),
            }
            for r in top.itertuples()
        ]
        charts["weak_ads"] = [
            {
                "ad": str(r.ad_name)[:40],
                "campaign": str(getattr(r, "campaign_name", ""))[:28],
                "spend": round(float(r.spend), 0),
                "roas": round(float(getattr(r, "roas", 0) or 0), 2),
                "purchases": int(getattr(r, "purchases", 0) or 0),
            }
            for r in weak.itertuples()
        ]

    # Trend over time (needs per-day rows: daily API fetch or daily-breakdown export)
    if "date_start" in df.columns and "spend" in df.columns:
        dated = df.dropna(subset=["date_start"])
        if dated["date_start"].nunique() > 1:
            agg: dict[str, Any] = {"spend": ("spend", "sum")}
            if "revenue" in dated.columns:
                agg["revenue"] = ("revenue", "sum")
            if "clicks" in dated.columns:
                agg["clicks"] = ("clicks", "sum")
            if "impressions" in dated.columns:
                agg["impressions"] = ("impressions", "sum")
            if "purchases" in dated.columns:
                agg["purchases"] = ("purchases", "sum")
            trend = dated.groupby(dated["date_start"].dt.date).agg(**agg).sort_index()
            if "revenue" in trend.columns:
                trend["roas"] = trend["revenue"] / trend["spend"].replace(0, pd.NA)
            if {"clicks", "impressions"}.issubset(trend.columns):
                trend["ctr"] = trend["clicks"] / trend["impressions"].replace(0, pd.NA) * 100
            trend = trend.fillna(0)
            charts["trend"] = [
                {
                    "date": str(idx),
                    "spend": round(float(r.spend), 0),
                    "revenue": round(float(getattr(r, "revenue", 0) or 0), 0),
                    "roas": round(float(getattr(r, "roas", 0) or 0), 2),
                    "ctr": round(float(getattr(r, "ctr", 0) or 0), 2),
                    "purchases": int(getattr(r, "purchases", 0) or 0),
                }
                for idx, r in trend.iterrows()
            ]

    # Ad set performance
    if {"adset_name", "spend"}.issubset(df.columns):
        aset = df.groupby("adset_name").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum") if "revenue" in df.columns else ("spend", "size"),
            clicks=("clicks", "sum") if "clicks" in df.columns else ("spend", "size"),
            impressions=("impressions", "sum") if "impressions" in df.columns else ("spend", "size"),
        )
        aset["roas"] = (
            aset["revenue"] / aset["spend"].replace(0, pd.NA) if "revenue" in df.columns else 0
        )
        aset["ctr"] = (
            aset["clicks"] / aset["impressions"].replace(0, pd.NA) * 100
            if {"clicks", "impressions"}.issubset(df.columns)
            else 0
        )
        aset = aset.fillna(0).sort_values("spend", ascending=False).head(10)
        charts["adset_perf"] = [
            {
                "name": _short_label(i, 18),
                "full_name": str(i),
                "spend": round(float(r.spend), 0),
                "revenue": round(float(r.revenue), 0),
                "roas": round(float(r.roas), 2),
                "ctr": round(float(r.ctr), 2),
            }
            for i, r in aset.iterrows()
        ]

    # ROAS distribution (spend-weighted buckets)
    if {"roas", "spend"}.issubset(df.columns):
        bins = [0, 0.5, 1, 1.5, 2, 3, 5, float("inf")]
        labels = ["<0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", "3–5", "5+"]
        valid = df.dropna(subset=["roas", "spend"]).copy()
        valid["bucket"] = pd.cut(valid["roas"].clip(lower=0), bins=bins, labels=labels, right=False)
        dist = valid.groupby("bucket", observed=False).agg(
            ads=("roas", "size"), spend=("spend", "sum")
        )
        charts["roas_distribution"] = [
            {"bucket": str(b), "ads": int(r.ads), "spend": round(float(r.spend), 0)}
            for b, r in dist.iterrows()
        ]

    # Cost efficiency: CPC & CPM by campaign
    if {"campaign_name", "spend", "clicks", "impressions"}.issubset(df.columns):
        eff = df.groupby("campaign_name").agg(
            spend=("spend", "sum"), clicks=("clicks", "sum"), impressions=("impressions", "sum")
        )
        eff["cpc"] = eff["spend"] / eff["clicks"].replace(0, pd.NA)
        eff["cpm"] = eff["spend"] / eff["impressions"].replace(0, pd.NA) * 1000
        eff = eff.fillna(0).sort_values("spend", ascending=False).head(10)
        charts["cost_efficiency"] = [
            {
                "name": _short_label(i, 16),
                "full_name": str(i),
                "cpc": round(float(r.cpc), 2),
                "cpm": round(float(r.cpm), 2),
                "spend": round(float(r.spend), 0),
            }
            for i, r in eff.iterrows()
        ]

    # Reach & frequency by campaign (when exported)
    if {"campaign_name", "reach", "frequency"}.issubset(df.columns):
        rf = df.groupby("campaign_name").agg(
            reach=("reach", "sum"), impressions=("impressions", "sum") if "impressions" in df.columns else ("reach", "sum")
        )
        rf["frequency"] = rf["impressions"] / rf["reach"].replace(0, pd.NA)
        rf = rf.fillna(0).sort_values("reach", ascending=False).head(10)
        charts["reach_frequency"] = [
            {
                "name": _short_label(i, 16),
                "full_name": str(i),
                "reach": round(float(r.reach), 0),
                "frequency": round(float(r.frequency), 2),
            }
            for i, r in rf.iterrows()
        ]

    # Status mix
    if "status" in df.columns and "spend" in df.columns:
        st = df.groupby(df["status"].astype(str).str.contains("active", case=False, na=False).map({True: "Active", False: "Other"}))[
            "spend"
        ].sum()
        charts["status_spend"] = [
            {"name": str(k), "value": float(v)} for k, v in st.items()
        ]

    return {
        "kpis": kpis,
        "charts": charts,
        "insights": session.chart_insights,
        "row_count": len(df),
        "currency": currency,
        "summary": {
            "campaigns": int(df["campaign_name"].nunique()) if "campaign_name" in df.columns else 0,
            "ads": int(df["ad_name"].nunique()) if "ad_name" in df.columns else len(df),
            "objectives": int(df["objective"].nunique()) if "objective" in df.columns else 0,
        },
    }


@app.post("/api/analysis/chart-insight/{chart_key}")
def chart_insight(chart_key: str, session: Session = Depends(require_session)):
    df = _require_df(session)
    if chart_key in session.chart_insights:
        return {"chart_key": chart_key, "insight": session.chart_insights[chart_key], "cached": True}
    client = _llm(session, feature=f"chart:{chart_key}")
    # Map frontend chart keys to analysis keys
    key_map = {
        "objective_spend": "objective_spend_pie",
        "objective_roas": "objective_roas",
        "scatter_spend_roas": "scatter_spend_roas",
        "ctr_by_campaign": "ctr_heatmap",
    }
    analysis_key = key_map.get(chart_key, chart_key)
    insight = llm_chart_insight(client, df, analysis_key)
    session.chart_insights[chart_key] = insight
    return {"chart_key": chart_key, "insight": insight, "cached": False}


@app.post("/api/analysis/detailed")
def detailed_analysis(session: Session = Depends(require_session)):
    df = _require_df(session)
    client = _llm(session, feature="detailed_analysis")
    raw = llm_detailed_analysis(client, df)
    session.detailed_analysis = raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "parsed": None}
    return {"raw": raw, "parsed": parsed}


@app.get("/api/chat")
def get_chat(session: Session = Depends(require_session)):
    return {"messages": session.chat_history}


@app.post("/api/chat")
def post_chat(body: ChatRequest, session: Session = Depends(require_session)):
    df = _require_df(session)
    client = _llm(session, feature="chat")
    history = list(session.chat_history)
    session.chat_history.append({"role": "user", "content": body.message})
    answer = llm_chat_answer(client, df, body.message, history=history)
    session.chat_history.append({"role": "assistant", "content": answer})
    return {"reply": answer, "messages": session.chat_history}


@app.delete("/api/chat")
def clear_chat(session: Session = Depends(require_session)):
    session.chat_history.clear()
    return {"ok": True}


@app.post("/api/report/pdf")
def pdf_report(session: Session = Depends(require_session)):
    df = _require_df(session)
    try:
        figs = create_charts(df)
    except Exception as exc:
        raise HTTPException(500, f"Chart generation failed: {exc}") from exc

    insights: dict[str, str] = {}
    # PDF chart keys align with dashboard insight keys
    key_map = {
        "objective_spend": "objective_spend",
        "objective_spend_revenue": "objective_roas",
        "scatter_spend_roas": "scatter_spend_roas",
        "ctr_by_campaign": "ctr_by_campaign",
    }
    # AI insights are optional — PDF still downloads without a configured provider
    client = None
    try:
        client = get_session_client(session.provider_store, session.usage, feature="pdf_report")
    except ValueError:
        client = None

    for fig_key in figs:
        front_key = key_map.get(fig_key, fig_key)
        if front_key in session.chart_insights:
            insights[fig_key] = session.chart_insights[front_key] or ""
        elif client is not None:
            try:
                insight = llm_chart_insight(client, df, fig_key)
                session.chart_insights[front_key] = insight
                insights[fig_key] = insight or ""
            except Exception:
                insights[fig_key] = ""
        else:
            insights[fig_key] = ""

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "Meta_Report.pdf"
            # generate_full_pdf writes chart pngs in cwd — run inside tmp
            import os

            prev = os.getcwd()
            try:
                os.chdir(tmp)
                path = generate_full_pdf(df, figs, insights, filename=str(out.name))
                data = Path(path).read_bytes()
            finally:
                os.chdir(prev)
                for fig in figs.values():
                    try:
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                    except Exception:
                        pass
    except Exception as exc:
        raise HTTPException(500, f"PDF build failed: {exc}") from exc

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(data)
    tmp_file.close()
    return FileResponse(
        tmp_file.name,
        media_type="application/pdf",
        filename="Meta_Report.pdf",
    )


# Serve built React app if present
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
