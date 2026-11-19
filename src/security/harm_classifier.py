"""Layer H - harmful medical intent classifier.

Prompt Guard (Layer A) and the Template Matcher (Layer B) recognise *how* an
attack is phrased: jailbreak personas, instruction overrides, injected
delimiters. A plainly worded harmful request such as "Write a justification
for prescribing a cancer drug for the common cold" has none of that, and the
public-benchmark screening showed fewer than 5% of MedSafetyBench requests
were stopped. Layer H instead scores *what* is being asked for.

Model: sentence embedding (default ``all-MiniLM-L6-v2``, already used by
Layer B) followed by logistic regression, trained by
``scripts/train_harm_classifier.py`` on the MedSafetyBench *training* split
(harmful) and benign MedQuAD / MedQA / general user prompts. The weights and
calibrated thresholds live in ``src/security/harm_classifier.npz`` - plain
arrays, loaded without pickle.

Long prompts (jailbreak templates often put the harmful payload at the end)
are scored in overlapping windows and the maximum is used.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path(__file__).parent / "harm_classifier.npz"
WINDOW_CHARS = 1000
WINDOW_STRIDE = 700


@dataclass
class HarmScore:
    score: float
    """Probability-like score in [0, 1]; maximum over windows."""

    window: int
    """Index of the highest-scoring window (0 = start of text)."""


def windows(text: str, size: int = WINDOW_CHARS, stride: int = WINDOW_STRIDE) -> list[str]:
    text = text or ""
    if len(text) <= size:
        return [text]
    out = [text[i:i + size] for i in range(0, max(1, len(text) - size + stride), stride)]
    tail = text[-size:]
    if out[-1] != tail:
        out.append(tail)
    return out


class HarmClassifier:
    def __init__(self, coef: np.ndarray, intercept: float, embed_model_id: str, meta: dict, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.coef = coef.astype(np.float32)
        self.intercept = float(intercept)
        self.meta = meta
        self.block_threshold = float(meta["thresholds"]["block"])
        self.flag_threshold = float(meta["thresholds"]["flag"])
        self.embedder = SentenceTransformer(embed_model_id, device=device)

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH, device: str = "cpu") -> "HarmClassifier":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(data["coef"], float(data["intercept"]), meta["embed_model_id"], meta, device)

    def _prob(self, emb: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(emb @ self.coef + self.intercept)))

    def score_many(self, texts: list[str]) -> list[HarmScore]:
        chunks: list[str] = []
        owner: list[int] = []
        for i, t in enumerate(texts):
            for w in windows(t):
                chunks.append(w)
                owner.append(i)
        if not chunks:
            return []
        emb = self.embedder.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        probs = self._prob(np.asarray(emb, dtype=np.float32))
        best = [HarmScore(0.0, 0) for _ in texts]
        seen: dict[int, int] = {}
        for p, i in zip(probs, owner):
            w = seen.get(i, 0)
            seen[i] = w + 1
            if p > best[i].score:
                best[i] = HarmScore(float(p), w)
        return best

    def score(self, text: str) -> HarmScore:
        return self.score_many([text])[0]


_instance: Optional[HarmClassifier] = None
_unavailable = False
_lock = threading.Lock()


def get_harm_classifier(device: str = "cpu") -> Optional[HarmClassifier]:
    """Lazy singleton; returns None (layer bypassed) if weights or encoder are missing."""
    global _instance, _unavailable
    if _instance is not None or _unavailable:
        return _instance
    with _lock:
        if _instance is None and not _unavailable:
            if not WEIGHTS_PATH.exists():
                logger.warning("Harm classifier weights not found at %s; Layer H disabled. "
                               "Run scripts/train_harm_classifier.py.", WEIGHTS_PATH)
                _unavailable = True
                return None
            try:
                _instance = HarmClassifier.load(WEIGHTS_PATH, device)
            except Exception as exc:  # pragma: no cover - depends on local model cache
                logger.error("Failed to load harm classifier: %s", exc)
                _unavailable = True
    return _instance
