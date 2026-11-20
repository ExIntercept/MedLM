"""Layer G - Llama Guard input moderation.

Layers A and B recognise how an attack is phrased, and Layer H scores harmful
*clinical* requests (it was trained on MedSafetyBench). None of them was trained
on plainly worded requests to hurt another person or to make a dangerous
substance, and those passed in UI testing. Llama Guard 3 is trained for exactly
those hazard categories, so it classifies the user's message before it reaches
the protected model.

The guard runs behind any OpenAI-compatible chat endpoint (Ollama, vLLM, ...),
whose chat template turns the message into Llama Guard's classification prompt.
If the endpoint is not configured or fails, the layer is bypassed like the
other layers, and the decision records that.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Llama Guard 3 hazard taxonomy (MLCommons).
CATEGORIES = {
    "S1": "Violent crimes", "S2": "Non-violent crimes", "S3": "Sex-related crimes",
    "S4": "Child sexual exploitation", "S5": "Defamation", "S6": "Specialized advice",
    "S7": "Privacy", "S8": "Intellectual property", "S9": "Indiscriminate weapons",
    "S10": "Hate", "S11": "Suicide & self-harm", "S12": "Sexual content",
    "S13": "Elections", "S14": "Code interpreter abuse",
}

# S6 (specialized advice) fires on ordinary medical questions, which are this
# product's purpose, so it alone never blocks. S11 is handled by emergency
# triage, which routes self-harm to crisis support instead of refusing.
NON_BLOCKING = {"S6", "S11"}


@dataclass
class GuardResult:
    unsafe: Optional[bool]
    """True/False as classified; None if the output could not be parsed."""

    categories: list = field(default_factory=list)
    raw: str = ""
    latency_ms: float = 0.0

    @property
    def blocks(self) -> bool:
        return bool(self.unsafe) and any(c not in NON_BLOCKING for c in self.categories or ["?"])


def parse_guard_output(text: str) -> GuardResult:
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return GuardResult(None, raw=text or "")
    verdict = lines[0].lower()
    if verdict.startswith("safe"):
        return GuardResult(False, raw=text)
    if verdict.startswith("unsafe"):
        cats = []
        if len(lines) > 1:
            cats = [c.strip().upper() for c in lines[1].replace(" ", ",").split(",") if c.strip()]
        return GuardResult(True, cats, raw=text)
    return GuardResult(None, raw=text)


class InputGuard:
    def __init__(self, adapter):
        self.adapter = adapter
        self.name = getattr(adapter, "name", "guard")

    def classify(self, user_text: str) -> GuardResult:
        start = time.perf_counter()
        out = "".join(self.adapter.stream([{"role": "user", "content": user_text}],
                                          temperature=0.0, max_tokens=20))
        result = parse_guard_output(out)
        result.latency_ms = round((time.perf_counter() - start) * 1000.0, 1)
        return result


_instance: Optional[InputGuard] = None
_lock = threading.Lock()


def get_input_guard() -> Optional[InputGuard]:
    """Lazy singleton; None when GUARD_BASE_URL / GUARD_MODEL are not set."""
    global _instance
    if _instance is not None:
        return _instance
    from src.config import GUARD_API_KEY, GUARD_BASE_URL, GUARD_MODEL

    if not (GUARD_BASE_URL and GUARD_MODEL):
        return None
    with _lock:
        if _instance is None:
            from src.llm.adapters import OpenAIChatAdapter

            _instance = InputGuard(OpenAIChatAdapter(GUARD_BASE_URL, GUARD_MODEL, api_key=GUARD_API_KEY))
    return _instance
