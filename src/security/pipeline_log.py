"""Per-request pipeline log for the admin console.

One JSON line per chat turn: who asked, what each of the nine stages did, how
long it took, and what the user was finally shown. The defense decision log
(``defense_log.jsonl``) stays as it is; this file adds the whole-pipeline view.
"""
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.config import PIPELINE_LOG_PATH

logger = logging.getLogger(__name__)
_lock = threading.Lock()

EXCERPT_CHARS = 500


def _cut(text: Optional[str]) -> str:
    text = text or ""
    return text if len(text) <= EXCERPT_CHARS else text[:EXCERPT_CHARS] + "…"


def build_record(result, *, status: str, username: str, role: str,
                 conversation_id: Optional[int], mode: str, message: str) -> dict:
    """The log entry for one finished gateway run."""
    steps = list(getattr(result, "steps", None) or [])
    return {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "username": username,
        "role": role,
        "conversation_id": conversation_id,
        "mode": mode,
        "status": status,
        "flagged": bool(getattr(result, "flagged", False)),
        "message": _cut(message),
        "response": _cut(getattr(result, "response_text", "")),
        "model": getattr(result, "model", None),
        "steps": steps,
        "total_ms": round(sum(s.get("ms", 0.0) for s in steps), 1),
    }


def append(record: dict) -> None:
    with _lock:
        try:
            PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(PIPELINE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.error("Failed to write pipeline log to %s: %s", PIPELINE_LOG_PATH, exc)


def read(limit: int = 100, after: Optional[str] = None) -> list[dict]:
    """Newest first. With ``after`` (an entry id), only entries newer than it."""
    if not PIPELINE_LOG_PATH.exists():
        return []
    out: list[dict] = []
    for line in reversed(PIPELINE_LOG_PATH.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if after and rec.get("id") == after:
            break
        out.append(rec)
        if len(out) >= limit:
            break
    return out
