# Meta Analyst

**Turn Meta Ads exports into actionable insights in minutes.**

Meta Analyst is an AI-powered analytics product for Meta (Facebook/Instagram) Ads. Connect the Marketing API or upload a CSV/Excel export, then explore a React KPI dashboard, chart-level AI insights, a strategy chat agent, and client-ready PDF reports.

Built by **RareBox**.

---

## Stack

| Layer | Tech |
| --- | --- |
| Frontend | React + Vite + Recharts |
| Backend | FastAPI (`api/main.py`) |
| Core | `meta_analyst/` (data, Meta API, multi-LLM providers, PDF) |

---

## Quick Start

### 1. Backend

```bash
python3.12 -m venv .venv   # 3.11/3.12 recommended
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set APP_PASSWORD and at least one AI key
uvicorn api.main:app --reload --port 8810
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5280`, sign in with `APP_PASSWORD`.

Ports are set to **5280** (UI) and **8810** (API) so they don’t clash with other local apps on 5173/8000.

### Production-style (API serves built UI)

```bash
cd frontend && npm run build && cd ..
uvicorn api.main:app --host 0.0.0.0 --port 8810
```
Then open `http://localhost:8810`.

---

## Product Features

### Meta Reporting API
- Connect with a Graph API **access token** (not Facebook password)
- Select an ad account and date range
- Pull live insights into the dashboard

### Dashboard
- KPI cards: spend, revenue, impressions, clicks, ROAS, CPC, CPM, CTR, LPV rates
- Spend by objective, ROAS by objective, spend vs ROAS, CTR by campaign
- On-demand AI insight blocks + PDF export

### AI Providers
- OpenAI (ChatGPT), Anthropic (Claude), xAI (Grok), Gemini, or custom OpenAI-compatible
- Keys stored per API session; optional `.env` fallbacks

### Upload + CLI
- CSV/XLSX upload still supported
- CLI batch mode: `META_INPUT_PATH=data/my_report.csv python meta_agent.py`

---

## Configuration

| Variable | Description |
| --- | --- |
| `APP_PASSWORD` | Required for login |
| `OPENAI_API_KEY` | OpenAI / ChatGPT |
| `ANTHROPIC_API_KEY` | Claude |
| `GROK_API_KEY` / `XAI_API_KEY` | Grok |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini |
| `META_ACCESS_TOKEN` | Optional default Meta Graph token |
| `META_AD_ACCOUNT_ID` | Optional default ad account |
| `META_CURRENCY` | Currency symbol (default `₹`) |

---

## Project Structure

```
meta-ai-agent/
├── api/
│   ├── main.py            # FastAPI app
│   └── session.py         # In-memory sessions
├── frontend/              # React dashboard (Vite)
├── meta_analyst/          # Core product package
│   ├── meta_api.py        # Meta Marketing / Reporting API
│   ├── providers.py       # Multi-LLM providers
│   ├── data.py            # Load / clean
│   ├── analysis.py        # LLM analysis
│   ├── charts.py          # KPI helpers (+ matplotlib for PDF)
│   └── pdf.py             # PDF report generation
├── meta_agent.py          # CLI entrypoint
├── .env.example
└── requirements.txt
```

---

## Security

Do not commit `.env` or API keys. Report vulnerabilities per [SECURITY.md](SECURITY.md).

---

## License

Private product by RareBox. All rights reserved.

Developed by Arka
