import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# exact Gemini model IDs; Flash-Lite is used because its free daily limit is far higher than Flash (20 a day for gemini-3.6-flash)
MAIN_MODEL = os.getenv("MAIN_MODEL", "gemini-3.5-flash-lite")
LITE_MODEL = os.getenv("LITE_MODEL", "gemini-3.1-flash-lite")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")

# used instead of MAIN_MODEL when it is overloaded (error 503), times out, or its daily quota is used up
BACKUP_MODEL = os.getenv("BACKUP_MODEL", "gemini-3.1-flash-lite")
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))

# optional, shared with Wikipedia and OpenAlex so they can contact you if your script misbehaves
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "")

CACHE_DIR = ROOT / "data" / "cache"
RUNS_DIR = ROOT / "runs"
# every finished PDF is also copied here (the main project folder) so reports are easy to find
REPORTS_COPY_DIR = ROOT
TRUST_FILE = ROOT / "agent" / "trusted_sources.yaml"

REQUEST_TIMEOUT = 20
MAX_DOWNLOAD_BYTES = 8_000_000
MAX_PAGE_CHARS = 20_000
MIN_PAGE_CHARS = 200
MAX_PDF_PAGES = 30
SAME_SITE_DELAY = 1.0

# how much work one research run is allowed to do; recheck = weak claims that get a targeted search
DEPTH_SETTINGS = {
    "quick": {"questions": 4, "max_sources": 10, "max_steps": 12, "recheck": 2},
    "standard": {"questions": 6, "max_sources": 20, "max_steps": 25, "recheck": 4},
    "deep": {"questions": 8, "max_sources": 35, "max_steps": 40, "recheck": 6},
}
DEPTH = os.getenv("DEPTH", "standard")


def user_agent():
    contact = f"; {CONTACT_EMAIL}" if CONTACT_EMAIL else ""
    return f"DeepResearchAgent/0.1 (B.Tech student project{contact})"
