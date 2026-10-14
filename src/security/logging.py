"""Thread-safe JSONL defense decision logger."""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import DEFENSE_LOG_PATH
from src.security.decision import DefenseDecision

logger = logging.getLogger(__name__)
_log_lock = threading.Lock()


def log_decision(
    decision: DefenseDecision,
    conversation_id: Optional[int],
    user_id: Optional[int],
    message_detector_text: str,
) -> None:
    """Append one JSON line per request to DEFENSE_LOG_PATH.

    Never logs HF_TOKEN, API keys, or full profile field values.
    Logs profile field names with scores only.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_id": conversation_id,
        "user_id": user_id,
        "action": decision.action,
        "category": decision.category,
        "pg_score": decision.pg_score,
        "tm_max_sim": round(decision.tm_max_sim, 4),
        "tm_n_components": decision.tm_n_components,
        "top_templates": decision.top_templates[:3],
        "sanitize_notes": decision.sanitize_notes,
        "reasons": decision.reasons,
        "latency_ms": decision.latency_ms,
        "message_excerpt": message_detector_text[:500],
    }
    with _log_lock:
        try:
            DEFENSE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(DEFENSE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("Failed to write defense log to %s: %s", DEFENSE_LOG_PATH, exc)
