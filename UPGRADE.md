# Meta Analyst — Upgrade Notes

Status: **v2.0.0** (Streamlit removed → React + FastAPI)  
Date: 2026-10-27

---

## Done in v2.0

- Replaced Streamlit UI with **React (Vite) dashboard**
- Added **FastAPI** backend (`api/main.py`) with session auth
- Kept core package: Meta Reporting API, multi-LLM providers, data clean, PDF, CLI

## Run

```bash
# API
uvicorn api.main:app --reload --port 8810

# UI → http://localhost:5280
cd frontend && npm install && npm run dev
```

## Remaining backlog (from earlier plan)

- Persist sessions / datasets (DB or object storage)
- Real multi-user auth
- Soft chart/column guards + insight caching hardening
- Unit tests for `data.py` + Meta API adapters
- Pin lockfiles (`uv` / `package-lock` already via npm)
