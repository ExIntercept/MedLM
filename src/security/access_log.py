"""Audit trail of admins viewing user data.

Admins can read patients' conversations to debug the system; every such view
is recorded here (who looked, at whose data, when), and the log itself is
visible in the admin console.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

from src.config import DATA_DIR

ACCESS_LOG_PATH = DATA_DIR / "admin_access_log.jsonl"
logger = logging.getLogger(__name__)
_lock = threading.Lock()


def record(admin: str, action: str, subject_user: Optional[str] = None, **detail) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "admin": admin,
        "action": action,
        "subject_user": subject_user,
        **{k: v for k, v in detail.items() if v is not None},
    }
    with _lock:
        try:
            ACCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ACCESS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error("Failed to write access log: %s", exc)


def read(limit: int = 200) -> list[dict]:
    if not ACCESS_LOG_PATH.exists():
        return []
    out = []
    for line in reversed(ACCESS_LOG_PATH.read_text(encoding="utf-8").splitlines()):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out
