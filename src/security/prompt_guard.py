"""Layer A — Prompt Guard 2 86M (meta-llama/Llama-Prompt-Guard-2-86M).

Wraps the HuggingFace sequence-classification model with:
  - Lazy singleton loading with clear error messages when the model is gated.
  - 510-token sliding windows with stride 384.
  - Batched single forward-pass per call.
  - Thread-safe inference lock.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 510
_STRIDE = 384


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PGResult:
    """Result of a Prompt Guard scoring call."""

    max_score: Optional[float]
    """Maximum malicious-class probability across all windows. None if model unavailable."""

    per_chunk: list[float]
    """Per-window malicious scores in order."""

    n_chunks: int
    """Number of token windows evaluated."""

    ms: float
    """Wall-clock time for the call in milliseconds."""


# ---------------------------------------------------------------------------
# PromptGuard class
# ---------------------------------------------------------------------------

class PromptGuard:
    """Wrapper around Llama-Prompt-Guard-2-86M for jailbreak / injection detection.

    Args:
        model_id: HuggingFace model identifier.
        device:   Compute device string ("cpu", "cuda", "mps").
    """

    def __init__(self, model_id: str, device: str) -> None:
        import os
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = device
        self._lock = threading.Lock()
        token = os.environ.get("HF_TOKEN")

        logger.info("Loading Prompt Guard tokenizer/model: %s on %s", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        # Handle MPS: fall back to CPU if MPS raises
        try:
            self._model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token)
            self._model = self._model.to(device)
        except Exception as exc:
            if device == "mps":
                logger.warning(
                    "Prompt Guard: MPS load failed (%s), falling back to CPU.", exc
                )
                self._device = "cpu"
                self._model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token)
                self._model = self._model.to("cpu")
            else:
                raise

        self._model.eval()

        # Resolve the index of the MALICIOUS / INJECTION label
        id2label: dict[int, str] = self._model.config.id2label
        malicious_idx: Optional[int] = None
        for idx, label in id2label.items():
            if label.upper() in ("MALICIOUS", "LABEL_1", "INJECTION"):
                malicious_idx = idx
                break
        if malicious_idx is None:
            raise ValueError(
                f"Cannot find MALICIOUS/INJECTION label in model.config.id2label: {id2label}. "
                "Expected a label whose uppercase is 'MALICIOUS', 'INJECTION', or 'LABEL_1'."
            )
        self._malicious_idx = malicious_idx
        logger.info(
            "Prompt Guard ready. Malicious label index: %d (%s)",
            malicious_idx,
            id2label[malicious_idx],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_token_ids(self, token_ids: list[int]) -> list[list[int]]:
        """Split a flat token-id list into overlapping windows with special boundary tokens."""
        if not token_ids:
            return []

        bos = [self._tokenizer.bos_token_id] if self._tokenizer.bos_token_id is not None else ([self._tokenizer.cls_token_id] if self._tokenizer.cls_token_id is not None else [])
        eos = [self._tokenizer.eos_token_id] if self._tokenizer.eos_token_id is not None else ([self._tokenizer.sep_token_id] if self._tokenizer.sep_token_id is not None else [])

        effective_window = max(16, _WINDOW_SIZE - len(bos) - len(eos))
        chunks: list[list[int]] = []
        start = 0
        while start < len(token_ids):
            chunk_slice = token_ids[start : start + effective_window]
            chunks.append(bos + chunk_slice + eos)
            if start + effective_window >= len(token_ids):
                break
            start += _STRIDE
        return chunks

    def _score_chunks(self, chunks: list[list[int]]) -> list[float]:
        """Run a single batched forward pass and return per-chunk malicious scores."""
        import torch
        import torch.nn.functional as F

        # Tokenize all chunks together as a padded batch
        encodings = self._tokenizer.pad(
            {"input_ids": chunks},
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self._device)
        attention_mask = encodings["attention_mask"].to(self._device)

        with self._lock:
            with torch.inference_mode():
                logits = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits  # (batch, num_labels)

        probs = F.softmax(logits, dim=-1)  # (batch, num_labels)
        scores: list[float] = probs[:, self._malicious_idx].tolist()
        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> PGResult:
        """Score a single text string.

        Args:
            text: Input text to evaluate.

        Returns:
            A :class:`PGResult`.
        """
        t0 = time.perf_counter()

        if not text or not text.strip():
            return PGResult(max_score=0.0, per_chunk=[], n_chunks=0, ms=0.0)

        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        chunks = self._chunk_token_ids(token_ids)

        if not chunks:
            return PGResult(max_score=0.0, per_chunk=[], n_chunks=0, ms=0.0)

        per_chunk = self._score_chunks(chunks)
        max_score = max(per_chunk)
        ms = (time.perf_counter() - t0) * 1000.0

        return PGResult(
            max_score=max_score,
            per_chunk=per_chunk,
            n_chunks=len(per_chunk),
            ms=ms,
        )

    def score_many(self, texts: list[str]) -> list[PGResult]:
        """Score multiple texts, each independently chunked and scored.

        All chunks from all texts are batched into a single forward pass for
        efficiency when texts are short, while still respecting per-text chunking
        semantics.

        Args:
            texts: List of input strings.

        Returns:
            List of :class:`PGResult`, one per input text, in order.
        """
        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()

        # Fast path: empty-ish texts
        results: list[Optional[PGResult]] = [None] * len(texts)
        pending_indices: list[int] = []
        all_chunks: list[list[int]] = []
        chunk_map: list[int] = []  # chunk_idx -> text_idx

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = PGResult(max_score=0.0, per_chunk=[], n_chunks=0, ms=0.0)
                continue
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            chunks = self._chunk_token_ids(token_ids)
            if not chunks:
                results[i] = PGResult(max_score=0.0, per_chunk=[], n_chunks=0, ms=0.0)
                continue
            pending_indices.append(i)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_map.append(i)

        if not all_chunks:
            # All texts were empty
            ms = (time.perf_counter() - t0) * 1000.0
            return [r or PGResult(max_score=0.0, per_chunk=[], n_chunks=0, ms=ms) for r in results]

        # Single batched forward pass
        encodings = self._tokenizer.pad(
            {"input_ids": all_chunks},
            padding=True,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self._device)
        attention_mask = encodings["attention_mask"].to(self._device)

        with self._lock:
            with torch.inference_mode():
                logits = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        all_scores: list[float] = probs[:, self._malicious_idx].tolist()

        # Reassemble per-text
        per_text_chunks: dict[int, list[float]] = {i: [] for i in pending_indices}
        for chunk_idx, score in enumerate(all_scores):
            text_idx = chunk_map[chunk_idx]
            per_text_chunks[text_idx].append(score)

        ms = (time.perf_counter() - t0) * 1000.0
        for i in pending_indices:
            per_chunk = per_text_chunks[i]
            results[i] = PGResult(
                max_score=max(per_chunk),
                per_chunk=per_chunk,
                n_chunks=len(per_chunk),
                ms=ms,
            )

        return [r or PGResult(max_score=0.0, per_chunk=[], n_chunks=0, ms=ms) for r in results]


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_pg_instance: Optional[PromptGuard] = None
_pg_lock = threading.Lock()
_pg_unavailable = False


def get_prompt_guard(
    model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M",
    device: str = "cpu",
) -> Optional[PromptGuard]:
    """Return the singleton :class:`PromptGuard`, loading it on first call.

    If the model is gated (missing HF_TOKEN or licence not accepted), logs a
    clear actionable message and returns ``None`` — does **not** crash the server.

    Args:
        model_id: HuggingFace model identifier.
        device:   Compute device string.

    Returns:
        A :class:`PromptGuard` instance, or ``None`` if loading failed.
    """
    global _pg_instance, _pg_unavailable

    if _pg_unavailable:
        return None
    if _pg_instance is not None:
        return _pg_instance

    with _pg_lock:
        # Double-checked locking
        if _pg_instance is not None:
            return _pg_instance
        if _pg_unavailable:
            return None

        import os
        token = os.environ.get("HF_TOKEN")
        if not token and model_id.startswith("meta-llama/"):
            _pg_unavailable = True
            logger.warning(
                "Prompt Guard unavailable: HF_TOKEN is not set in environment or .env. "
                "Set HF_TOKEN in .env and accept the Llama 4 Community License at "
                "https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M"
            )
            return None

        try:
            _pg_instance = PromptGuard(model_id=model_id, device=device)
            return _pg_instance
        except Exception as exc:
            _pg_unavailable = True
            logger.error("Prompt Guard load failed: %s", exc)
            return None
