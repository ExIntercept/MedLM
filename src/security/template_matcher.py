"""Layer B — Jailbreak template matcher using dense semantic embeddings.

Loads a JSONL corpus of known jailbreak template patterns, encodes the index
split, and uses cosine similarity (dot product on L2-normalised embeddings) to
detect structural similarity between user inputs and known attack patterns.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# The 9 roleplay-component names tracked by the matcher
COMPONENT_NAMES: tuple[str, ...] = (
    "fictional_frame",
    "amoral_character",
    "unconditional_obedience",
    "ethics_waiver",
    "realism_demand",
    "stakes_pressure",
    "private_audience_claim",
    "redaction_claim",
    "forced_opening",
)

# Sentence-splitting pattern
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TMResult:
    """Result of a template-matcher query."""

    max_sim: float
    """Maximum cosine similarity between any query window and any index record."""

    top: list[dict]
    """Top-5 matches: each dict has keys id, family, component, sim."""

    components: dict[str, float]
    """Per-component best similarity (only for records in the roleplay_component family)."""

    n_components: int
    """Number of components whose best_sim >= tau_component."""

    ms: float
    """Wall-clock time in milliseconds."""


# ---------------------------------------------------------------------------
# TemplateMatcher class
# ---------------------------------------------------------------------------

class TemplateMatcher:
    """Dense-embedding template matcher for jailbreak detection.

    Args:
        embed_model_id: SentenceTransformer model identifier.
        device:         Compute device string.
        corpus_path:    Path to the JSONL corpus file.
        tau_component:  Similarity threshold for counting active components.
        cache_dir:      Directory to persist/load the index embeddings cache.
    """

    def __init__(
        self,
        embed_model_id: str,
        device: str,
        corpus_path: Path,
        tau_component: float = 0.65,
        cache_dir: Optional[Path] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._tau = tau_component
        self._lock = threading.Lock()

        logger.info(
            "Loading SentenceTransformer: %s on %s", embed_model_id, device
        )
        try:
            self._embedder = SentenceTransformer(embed_model_id, device=device, local_files_only=True)
        except Exception:
            self._embedder = SentenceTransformer(embed_model_id, device=device)

        # Load corpus
        self._index_records: list[dict] = []
        self._load_corpus(corpus_path)

        if not self._index_records:
            logger.warning(
                "TemplateMatcher: no index records found in %s", corpus_path
            )
            self._index_embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
            return

        # Build or load cached embeddings
        texts = [r["text"] for r in self._index_records]
        self._index_embeddings = self._load_or_compute_embeddings(
            texts=texts,
            model_id=embed_model_id,
            cache_dir=cache_dir,
        )

    # ------------------------------------------------------------------
    # Corpus loading
    # ------------------------------------------------------------------

    def _load_corpus(self, corpus_path: Path) -> None:
        """Parse JSONL and keep only records with split == "index"."""
        if not corpus_path.exists():
            logger.error("Corpus file not found: %s", corpus_path)
            return
        with corpus_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)
                    continue
                if record.get("split") == "index":
                    self._index_records.append(record)
        logger.info("TemplateMatcher: loaded %d index records", len(self._index_records))

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    @staticmethod
    def _corpus_key(texts: list[str], model_id: str) -> str:
        """SHA-256 fingerprint of the corpus texts joined with the model id."""
        joined = "\n".join(texts) + model_id
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def _load_or_compute_embeddings(
        self,
        texts: list[str],
        model_id: str,
        cache_dir: Optional[Path],
    ) -> np.ndarray:
        """Return index embeddings, loading from cache when the key matches."""
        expected_key = self._corpus_key(texts, model_id)

        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            npy_path = cache_dir / "index_embeddings.npy"
            key_path = cache_dir / "index_embeddings_key.json"

            if npy_path.exists() and key_path.exists():
                try:
                    stored_key = json.loads(key_path.read_text(encoding="utf-8")).get("key", "")
                    if stored_key == expected_key:
                        embeddings = np.load(str(npy_path))
                        logger.info(
                            "TemplateMatcher: loaded cached embeddings from %s", npy_path
                        )
                        return embeddings
                    else:
                        logger.info(
                            "TemplateMatcher: cache key mismatch — recomputing embeddings."
                        )
                except Exception as exc:
                    logger.warning("TemplateMatcher: cache read failed (%s) — recomputing.", exc)

        # Compute embeddings
        logger.info("TemplateMatcher: encoding %d index texts …", len(texts))
        embeddings = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)

        if cache_dir is not None:
            try:
                np.save(str(npy_path), embeddings)
                key_path.write_text(
                    json.dumps({"key": expected_key}), encoding="utf-8"
                )
                logger.info("TemplateMatcher: saved embeddings cache to %s", npy_path)
            except Exception as exc:
                logger.warning("TemplateMatcher: failed to write cache: %s", exc)

        return embeddings

    # ------------------------------------------------------------------
    # Query segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _segment(text: str) -> list[str]:
        """Return full text + 1/2/3-sentence windows, capped at 64 longest."""
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        windows: list[str] = [text]  # always include the full text

        for n in (1, 2, 3):
            for i in range(len(sentences) - n + 1):
                window = " ".join(sentences[i : i + n])
                if window:
                    windows.append(window)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for w in windows:
            if w not in seen:
                seen.add(w)
                unique.append(w)

        # Cap at 64: keep 64 longest by character length
        if len(unique) > 64:
            unique = sorted(unique, key=len, reverse=True)[:64]

        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, text: str) -> TMResult:
        """Score text against the index and return a :class:`TMResult`.

        Args:
            text: Input text (detector_text from sanitizer is recommended).

        Returns:
            A :class:`TMResult` with similarity scores and component breakdown.
        """
        t0 = time.perf_counter()

        if not self._index_records or self._index_embeddings.shape[0] == 0:
            return TMResult(
                max_sim=0.0,
                top=[],
                components={c: 0.0 for c in COMPONENT_NAMES},
                n_components=0,
                ms=0.0,
            )

        windows = self._segment(text)

        with self._lock:
            query_embeddings: np.ndarray = self._embedder.encode(
                windows,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

        # Cosine similarity matrix: (n_windows x n_index)
        # Embeddings are L2-normalised, so dot product == cosine similarity
        sim_matrix: np.ndarray = query_embeddings @ self._index_embeddings.T  # (W, N)

        # max_sim across all windows and all index records
        global_max: float = float(sim_matrix.max())

        # Per-index-record best similarity (max over query windows)
        per_record_best: np.ndarray = sim_matrix.max(axis=0)  # (N,)

        # Top-5 records
        n_top = min(5, len(self._index_records))
        top5_indices = np.argpartition(per_record_best, -n_top)[-n_top:]
        top5_indices = top5_indices[np.argsort(per_record_best[top5_indices])[::-1]]

        top: list[dict] = []
        for idx in top5_indices:
            rec = self._index_records[int(idx)]
            top.append(
                {
                    "id": rec.get("id", int(idx)),
                    "family": rec.get("family", ""),
                    "component": rec.get("component", ""),
                    "sim": round(float(per_record_best[idx]), 4),
                }
            )

        # Per-component best similarity (only roleplay_component family)
        components: dict[str, float] = {c: 0.0 for c in COMPONENT_NAMES}
        for idx, rec in enumerate(self._index_records):
            if rec.get("family") != "roleplay_component":
                continue
            comp = rec.get("component", "")
            if comp not in components:
                continue
            sim = float(per_record_best[idx])
            if sim > components[comp]:
                components[comp] = sim

        n_components = sum(1 for v in components.values() if v >= self._tau)

        ms = (time.perf_counter() - t0) * 1000.0
        return TMResult(
            max_sim=round(global_max, 4),
            top=top,
            components=components,
            n_components=n_components,
            ms=ms,
        )


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_tm_instance: Optional[TemplateMatcher] = None
_tm_lock = threading.Lock()
_tm_init_key: Optional[tuple] = None


def get_template_matcher(
    embed_model_id: str,
    device: str,
    corpus_path: Path,
    tau_component: float = 0.65,
    cache_dir: Optional[Path] = None,
) -> TemplateMatcher:
    """Return the singleton :class:`TemplateMatcher`, loading it on first call.

    If called again with the same arguments, returns the cached instance.
    If arguments differ, raises ``RuntimeError`` (re-initialisation not supported at runtime).

    Args:
        embed_model_id: SentenceTransformer model ID.
        device:         Compute device.
        corpus_path:    Path to jailbreak_templates.jsonl.
        tau_component:  Component similarity threshold.
        cache_dir:      Optional cache directory.

    Returns:
        A :class:`TemplateMatcher` instance.
    """
    global _tm_instance, _tm_init_key

    init_key = (embed_model_id, device, str(corpus_path), tau_component, str(cache_dir))

    if _tm_instance is not None:
        if _tm_init_key != init_key:
            raise RuntimeError(
                "TemplateMatcher singleton already initialised with different arguments. "
                "Re-initialisation is not supported. Restart the process to change parameters."
            )
        return _tm_instance

    with _tm_lock:
        if _tm_instance is not None:
            return _tm_instance
        _tm_instance = TemplateMatcher(
            embed_model_id=embed_model_id,
            device=device,
            corpus_path=corpus_path,
            tau_component=tau_component,
            cache_dir=cache_dir,
        )
        _tm_init_key = init_key
        return _tm_instance
