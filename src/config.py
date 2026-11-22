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

# ── Model-agnostic LLM backend (used by the defense gateway) ─────────────────
# LLM_PROVIDER selects how the protected model is called:
#   completions  -> OpenAI-compatible /v1/completions; the gateway renders the
#                   chat template itself (LLM_TEMPLATE). Default; matches the
#                   original Colab/vLLM setup.
#   openai_chat  -> OpenAI-compatible /v1/chat/completions; the serving stack
#                   applies the model's own chat template. Works with vLLM, TGI,
#                   Ollama, LM Studio, llama.cpp server, OpenAI and similar APIs.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "completions").lower()
LLM_BASE_URL = (os.environ.get("LLM_BASE_URL") or COLAB_API_BASE).rstrip("/")
if LLM_BASE_URL.endswith("/v1"):
    LLM_BASE_URL = LLM_BASE_URL[: -len("/v1")]
LLM_API_KEY = os.environ.get("LLM_API_KEY") or COLAB_API_KEY
LLM_MODEL = os.environ.get("LLM_MODEL") or MODEL_ID
# Template for LLM_PROVIDER=completions: gemma | chatml | llama3 | mistral | plain.
# Falls back to PROMPT_STYLE ("chat" -> chatml, "base" -> plain).
LLM_TEMPLATE = (os.environ.get("LLM_TEMPLATE") or PROMPT_STYLE).lower()
# Set to 0 for models whose chat template rejects a system role (e.g. Gemma 2).
LLM_SYSTEM_ROLE = os.environ.get("LLM_SYSTEM_ROLE", "1") == "1"
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Auth secrets
JWT_SECRET = os.environ.get("JWT_SECRET", os.environ.get("APP_SECRET", "dev-only-insecure-secret-change-in-production"))
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
ADMIN_USERNAMES = {
    username.strip()
    for username in os.environ.get("ADMIN_USERNAMES", "").split(",")
    if username.strip()
}

# ── Defense / Security pipeline ──────────────────────────────────────────────
# Set DEFENSE_ENABLED=0 in .env to disable all layers (useful for ablation).
DEFENSE_ENABLED = os.environ.get("DEFENSE_ENABLED", "1") == "1"

# Layer A – Prompt Injection neural classifier (open-source DeBERTa-v3 or Llama-Prompt-Guard)
PROMPT_GUARD_MODEL_ID = os.environ.get(
    "PROMPT_GUARD_MODEL_ID", "ProtectAI/deberta-v3-base-prompt-injection-v2"
)

# Layer G – Llama Guard input moderation, served behind an OpenAI-compatible
# /v1/chat/completions endpoint (Ollama: GUARD_BASE_URL=http://localhost:11434,
# GUARD_MODEL=llama-guard3:1b; vLLM: meta-llama/Llama-Guard-3-1B). Unset = layer off.
GUARD_BASE_URL = os.environ.get("GUARD_BASE_URL", "").rstrip("/")
GUARD_MODEL = os.environ.get("GUARD_MODEL", "")
GUARD_API_KEY = os.environ.get("GUARD_API_KEY", "")

# Layer B – Template Matcher embedding model (all-MiniLM-L6-v2 is already cached)
TEMPLATE_EMBED_MODEL_ID = os.environ.get(
    "TEMPLATE_EMBED_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)

# Path to calibrated thresholds JSON; written by scripts/calibrate_defenses.py.
# Falls back to hard-coded defaults in decision.py if the file is missing.
DEFENSE_THRESHOLDS_PATH = ROOT / "src" / "security" / "thresholds.json"

# JSONL log of every defense decision (one line per request).
DEFENSE_LOG_PATH = DATA_DIR / "defense_log.jsonl"
PIPELINE_LOG_PATH = DATA_DIR / "pipeline_log.jsonl"

# Input size limits applied by the sanitizer.
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "6000"))
MAX_PROFILE_FIELD_CHARS = int(os.environ.get("MAX_PROFILE_FIELD_CHARS", "300"))
