"""EWMA session risk for gradual, multi-turn adversarial escalation."""
from __future__ import annotations

import threading


class SessionRisk:
    def __init__(self, decay: float = 0.60, limit: int = 10_000):
        self.decay = decay
        self.limit = limit
        self._scores: dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, session_id: str, risk: float) -> float:
        key = str(session_id or "default")
        with self._lock:
            previous = self._scores.get(key, 0.0)
            current = self.decay * previous + (1.0 - self.decay) * max(0.0, min(1.0, risk))
            self._scores[key] = current
            if len(self._scores) > self.limit:
                oldest = next(iter(self._scores))
                self._scores.pop(oldest, None)
            return current

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._scores.pop(str(session_id or "default"), None)


session_risk = SessionRisk()
