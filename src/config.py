"""Configuration loader from .env at project root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = ROOT / "corpus"
CHROMA_DB_DIR = ROOT / "chroma_db"
DOCS_DIR = ROOT / "docs"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# Remote Colab ngrok endpoint for MedGemma
COLAB_API_BASE = os.environ.get("COLAB_API_BASE", "").rstrip("/")
COLAB_API_KEY = os.environ.get("COLAB_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-4b-it")
PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "gemma").lower()
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
RESERVE_OUTPUT_TOKENS = int(os.environ.get("RESERVE_OUTPUT_TOKENS", "640"))

# Auth secrets
JWT_SECRET = os.environ.get("JWT_SECRET", os.environ.get("APP_SECRET", "dev-only-insecure-secret-change-in-production"))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
