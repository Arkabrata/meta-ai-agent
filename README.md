# Meta Analyst

**Turn Meta Ads exports into actionable insights in minutes.**

Meta Analyst is an AI-powered analytics product for Meta (Facebook/Instagram) Ads. Upload a CSV or Excel export from Ads Manager, and get a KPI dashboard, chart-level AI insights, a strategy chat agent, and client-ready PDF reports — without manual spreadsheet work.

Built by **RareBox**.

---

## Why Meta Analyst

| Problem | Meta Analyst |
| --- | --- |
| Raw Meta exports use inconsistent column names | Auto-detects and normalizes columns |
| ROAS/CPC/CTR calculations are repetitive | Computes derived metrics instantly |
| Performance reviews take hours | AI summarizes charts and prioritizes actions |
| Recommendations live in Slack threads | Chat agent answers from your uploaded data |
| Client reporting is manual | One-click PDF export with KPIs + insights |

---

## Product Features

### Dashboard
- KPI cards: spend, revenue, impressions, clicks, ROAS, CPC, CPM, CTR, LPV rates
- Spend share by objective, ROAS by objective, spend vs. ROAS scatter, CTR heatmap
- Per-chart AI insight blocks with actionable recommendations

### AI Analysis
- **Detailed analysis** — prioritized HIGH / MEDIUM / LOW recommendations
- **Strategy chat** — ask about scaling, pausing, budget shifts, and campaign health
- **CLI batch mode** — run analysis from the terminal and export ad-level recommendations to CSV

### Security
- Password-protected access via `APP_PASSWORD`
- API keys stored in `.env` (never committed)

---

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd meta-ai-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
APP_PASSWORD=your-access-password
```

### 3. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`, sign in, upload a Meta Ads report, then explore **Dashboard** and **AI Chat**.

---

## How It Works

```
Meta Ads Export (CSV/XLSX)
        │
        ▼
   Auto-clean & normalize
        │
        ├── KPI Dashboard + Charts
        ├── AI insights per chart
        ├── Detailed priority analysis
        ├── Strategy chat
        └── PDF report export
```

**Supported inputs:** Meta Ads Manager exports with columns such as Campaign name, Amount spent, Impressions, Clicks, Purchases conversion value, Objective, and Ad delivery status.

**Default currency:** INR (`₹`). Override with `META_CURRENCY` in `.env`.

---

## CLI Usage

For batch analysis without the web UI:

```bash
META_INPUT_PATH=data/my_report.csv python meta_agent.py
```

| Variable | Default | Description |
| --- | --- | --- |
| `META_INPUT_PATH` | `data/meta_report.xlsx` | Input report path |
| `META_OUTPUT_PATH` | `output/meta_ads_with_recommendations.csv` | Recommendations CSV |
| `META_MODEL_NAME` | `gpt-5-mini` | OpenAI model for CLI analysis |

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `APP_PASSWORD` | — | Required for web app access |
| `META_ANALYSIS_MODEL` | `gpt-4.1-mini` | Model for detailed dashboard analysis |
| `META_CHAT_MODEL` | `gpt-4.1-mini` | Model for chat and chart insights |
| `META_CURRENCY` | `₹` | Currency symbol in KPI cards |

---

## Project Structure

```
meta-ai-agent/
├── app.py                 # Streamlit product (web UI)
├── meta_agent.py          # CLI entrypoint
├── meta_analyst/          # Core product package
│   ├── config.py          # Environment-based settings
│   ├── data.py            # Load, clean, summarize
│   ├── analysis.py        # OpenAI integrations
│   ├── charts.py          # KPIs and visualizations
│   ├── pdf.py             # PDF report generation
│   └── ui.py              # Streamlit theme and components
├── .env.example
├── .streamlit/config.toml
├── requirements.txt
└── RareBoxLogo.png
```

---

## Dev Container

The included `.devcontainer/` config runs Python 3.11, installs dependencies, and launches the Streamlit app on port `8501` in Codespaces or VS Code Dev Containers.

---

## Security

Do not commit `.env` or API keys. Report vulnerabilities per [SECURITY.md](SECURITY.md).

---

## License

Private product by RareBox. All rights reserved.

Developed by Arka
