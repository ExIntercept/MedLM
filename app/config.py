"""Configuration, loaded from .env at the project root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
USERS_DIR = DATA_DIR / "users"


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

COLAB_API_BASE = os.environ.get("COLAB_API_BASE", "").rstrip("/")
COLAB_API_KEY = os.environ.get("COLAB_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "epfl-llm/meditron-7b")
PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "base").lower()
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "2048"))
RESERVE_OUTPUT_TOKENS = int(os.environ.get("RESERVE_OUTPUT_TOKENS", "384"))
APP_SECRET = os.environ.get("APP_SECRET", "dev-secret-change-me")

# How many past turns to consider before the token budget trims them further.
MAX_HISTORY_TURNS = 12

DATA_DIR.mkdir(exist_ok=True)
USERS_DIR.mkdir(exist_ok=True)
