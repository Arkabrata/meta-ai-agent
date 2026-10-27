import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PRODUCT_NAME = "Meta Analyst"
COMPANY_NAME = "RareBox"
LOGO_PATH = ROOT / "RareBoxLogo.png"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Optional Meta Marketing API defaults (UI can override per session)
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
META_AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID", "")

ANALYSIS_MODEL = os.getenv("META_ANALYSIS_MODEL", "gpt-4.1-mini")
CHAT_MODEL = os.getenv("META_CHAT_MODEL", "gpt-4.1-mini")
CLI_MODEL = os.getenv("META_MODEL_NAME", "gpt-5-mini")
CURRENCY_SYMBOL = os.getenv("META_CURRENCY", "₹")

META_INPUT_PATH = os.getenv("META_INPUT_PATH", "data/meta_report.xlsx")
META_OUTPUT_PATH = os.getenv("META_OUTPUT_PATH", "output/meta_ads_with_recommendations.csv")
