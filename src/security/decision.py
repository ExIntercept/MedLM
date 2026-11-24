"""Defense decision engine — orchestrates Layer A (Prompt Guard) and
Layer B (Template Matcher) and emits a structured :class:`DefenseDecision`.
"""
from __future__ import annotations

import json
import logging
import time
import random
import statistics
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)

from src.security.prompt_guard import get_prompt_guard
from src.security.template_matcher import COMPONENT_NAMES, get_template_matcher

# ---------------------------------------------------------------------------
# Default threshold configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: dict = {
    "pg_block": 0.90,
    "pg_flag": 0.50,
    "tm_block": 0.80,
    "tm_flag": 0.70,
    "tau_component": 0.65,
    "components_block": 3,
    "components_flag": 2,
    "session_block": 0.55,
    "smooth_block": 0.90,
    "smooth_flag": 0.50,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DefenseDecision:
    """Full result of the input defense pipeline."""

    action: Literal["ALLOW", "FLAG", "BLOCK"]
    """Gating decision: ALLOW, FLAG (manual review), or BLOCK (hard reject)."""

    category: Optional[Literal["JAILBREAK", "INJECTION", "STRUCTURAL", "HARMFUL_REQUEST"]]
    """Attack category when action != ALLOW. None when ALLOW."""

    pg_score: Optional[float]
    """Maximum Prompt Guard malicious score across all scored texts. None if unavailable."""

    tm_max_sim: float
    """Maximum template-matcher cosine similarity across all scored texts."""

    tm_components: dict
    """Per-roleplay-component best similarity scores."""

    tm_n_components: int
    """Number of roleplay components above tau_component threshold."""

    top_templates: list
    """Top matched templates from the template matcher."""

    sanitize_notes: list
    """Transformation notes from the sanitizer (e.g. control_token:end_of_turn)."""

    reasons: list
    """Human-readable list of triggered rules, e.g. ["pg_score_high", "tm_sim_high"]."""

    latency_ms: dict
    """Per-layer latency in milliseconds: {"pg": …, "tm": …, "harm": …, "total": …}."""

    harm_score: Optional[float] = None
    """Layer H harmful-intent score (max over texts). None if the layer did not run."""
    guard_unsafe: Optional[bool] = None
    """Layer G (Llama Guard) verdict; None when the layer did not run."""

    guard_categories: Optional[list] = None
    """Llama Guard hazard codes, e.g. ["S1"]."""
    pg_smooth_mean: Optional[float] = None
    """Prompt Guard mean over smoothed character perturbations."""
    pg_smooth_std: Optional[float] = None
    """Prompt Guard score instability across perturbations."""
    session_risk: float = 0.0
    """EWMA risk across turns in the current conversation."""
    fused_risk: float = 0.0
    """Domain-fused detector risk score."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_thresholds(path: Path) -> dict:
    """Load thresholds from a JSON file, falling back to defaults.

    Args:
        path: Path to the JSON thresholds file.

    Returns:
        Merged dict: defaults overridden by any keys present in the file.
    """
    thresholds = dict(DEFAULT_THRESHOLDS)
    if path.exists():
        try:
            overrides = json.loads(path.read_text(encoding="utf-8"))
            thresholds.update(overrides)
            logger.debug("Loaded thresholds from %s", path)
        except Exception as exc:
            logger.warning("Failed to load thresholds from %s: %s. Using defaults.", path, exc)
    return thresholds


def _smoothed_prompt_guard(prompt_guard, text: str, copies: int = 4, q: float = 0.08) -> tuple[float, float]:
    """Score small character perturbations with Prompt Guard."""
    if not text.strip():
        return 0.0, 0.0
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    rng = random.Random(0)
    variants = []
    source = text[:2000]
    for index in range(max(1, copies)):
        chars = list(source)
        count = max(1, int(len(chars) * q))
        positions = rng.sample(range(len(chars)), min(count, len(chars)))
        mode = index % 3
        if mode == 0:
            for pos in positions:
                chars[pos] = rng.choice(alphabet)
        elif mode == 1:
            for _ in range(count):
                chars.insert(rng.randrange(len(chars) + 1), rng.choice(alphabet))
        else:
            start = rng.randrange(max(1, len(chars) - count + 1))
            chars[start:start + count] = [rng.choice(alphabet) for _ in range(count)]
        variants.append("".join(chars))
    results = prompt_guard.score_many(variants)
    scores = [r.max_score for r in results if r.max_score is not None]
    if not scores:
        return 0.0, 0.0
    return float(statistics.mean(scores)), float(statistics.pstdev(scores))


# ---------------------------------------------------------------------------
# Main defense function
# ---------------------------------------------------------------------------

def run_input_defenses(
    sanitized: "SanitizedInputs",  # noqa: F821 — imported below to avoid circular imports
    prior_user_turns: list[str],
    disable_layer_a: bool = False,
    disable_layer_b: bool = False,
    disable_layer_h: bool = False,
    disable_layer_g: bool = False,
    session_id: Optional[str] = None,
) -> DefenseDecision:
    """Run Layer A (Prompt Guard), Layer B (Template Matcher) and Layer H
    (harmful-intent classifier) on sanitized inputs.

    Args:
        sanitized:        Output of :func:`sanitize_user_inputs`.
        prior_user_turns: Previous user messages in the conversation (most recent last).
        disable_layer_a:  If True, skip Prompt Guard scoring.
        disable_layer_b:  If True, skip Template Matcher scoring.
        disable_layer_h:  If True, skip harmful-intent scoring.

    Returns:
        A :class:`DefenseDecision` with action, category, scores, and latency.
    """
    # Deferred imports to keep module-level import cost zero
    from src.config import (
        DEFENSE_ENABLED,
        DEFENSE_THRESHOLDS_PATH,
        PROMPT_GUARD_MODEL_ID,
        TEMPLATE_EMBED_MODEL_ID,
    )

    # ------------------------------------------------------------------
    # Fast path: defense disabled
    # ------------------------------------------------------------------
    if not DEFENSE_ENABLED:
        return DefenseDecision(
            action="ALLOW",
            category=None,
            pg_score=None,
            tm_max_sim=0.0,
            tm_components={},
            tm_n_components=0,
            top_templates=[],
            sanitize_notes=[],
            reasons=["defense_disabled"],
            latency_ms={},
        )

    # ------------------------------------------------------------------
    # Wrap all detector code in a broad try/except
    # ------------------------------------------------------------------
    try:
        return _run_defenses_inner(
            sanitized=sanitized,
            prior_user_turns=prior_user_turns,
            prompt_guard_model_id=PROMPT_GUARD_MODEL_ID,
            template_embed_model_id=TEMPLATE_EMBED_MODEL_ID,
            thresholds_path=DEFENSE_THRESHOLDS_PATH,
            disable_layer_a=disable_layer_a,
            disable_layer_b=disable_layer_b,
            disable_layer_h=disable_layer_h,
            disable_layer_g=disable_layer_g,
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("Unexpected error in defense pipeline: %s", exc)
        # Collect sanitize notes for the response even in error path
        sanitize_notes = list(sanitized.message.notes)
        for sf in sanitized.profile.values():
            sanitize_notes.extend(sf.notes)
        return DefenseDecision(
            action="FLAG",
            category=None,
            pg_score=None,
            tm_max_sim=0.0,
            tm_components={},
            tm_n_components=0,
            top_templates=[],
            sanitize_notes=sanitize_notes,
            reasons=["detector_error"],
            latency_ms={},
        )


def _run_defenses_inner(
    sanitized: "SanitizedInputs",  # noqa: F821
    prior_user_turns: list[str],
    prompt_guard_model_id: str,
    template_embed_model_id: str,
    thresholds_path: Path,
    disable_layer_a: bool = False,
    disable_layer_b: bool = False,
    disable_layer_h: bool = False,
    disable_layer_g: bool = False,
    session_id: Optional[str] = None,
) -> DefenseDecision:
    """Internal implementation; called from inside the try/except in run_input_defenses."""
    from pathlib import Path as _Path

    from src.security.device import get_device

    t_total_start = time.perf_counter()

    device = get_device()
    thresholds = load_thresholds(thresholds_path)

    pg_block: float = thresholds["pg_block"]
    pg_flag: float = thresholds["pg_flag"]
    tm_block: float = thresholds["tm_block"]
    tm_flag: float = thresholds["tm_flag"]
    tau_component: float = thresholds["tau_component"]
    components_block: int = int(thresholds["components_block"])
    components_flag: int = int(thresholds["components_flag"])

    # ------------------------------------------------------------------
    # Collect sanitize notes
    # ------------------------------------------------------------------
    sanitize_notes: list[str] = list(sanitized.message.notes)
    for sf in sanitized.profile.values():
        sanitize_notes.extend(sf.notes)

    # ------------------------------------------------------------------
    # Build the three texts to score
    # ------------------------------------------------------------------
    text_message = sanitized.message.detector_text

    profile_parts: list[str] = []
    for key, sf in sanitized.profile.items():
        val = sf.detector_text.strip()
        if val:
            profile_parts.append(f"{key}: {val}")
    text_profile = "\n".join(profile_parts)

    last_two = prior_user_turns[-2:] if len(prior_user_turns) >= 2 else prior_user_turns
    text_multiturn = "\n".join(last_two + [text_message])

    texts = [text_message, text_profile, text_multiturn]

    # ------------------------------------------------------------------
    # Layer A — Prompt Guard
    # ------------------------------------------------------------------
    t_pg_start = time.perf_counter()
    pg_score: Optional[float] = None
    pg_unavailable = True
    pg_ms = 0.0
    pg_results: list = []
    pg_smooth_mean: Optional[float] = None
    pg_smooth_std: Optional[float] = None

    if not disable_layer_a:
        pg = get_prompt_guard(prompt_guard_model_id, device)
        pg_unavailable = pg is None
        if not pg_unavailable:
            pg_results = pg.score_many(texts)  # type: ignore[union-attr]
            scores = [r.max_score for r in pg_results if r.max_score is not None]
            pg_score = max(scores) if scores else None
            if text_message.strip():
                pg_smooth_mean, pg_smooth_std = _smoothed_prompt_guard(pg, text_message)
        pg_ms = (time.perf_counter() - t_pg_start) * 1000.0

    # ------------------------------------------------------------------
    # Layer B — Template Matcher
    # ------------------------------------------------------------------
    t_tm_start = time.perf_counter()
    tm_max_sim: float = 0.0
    tm_components: dict[str, float] = {c: 0.0 for c in COMPONENT_NAMES}
    tm_n_components: int = 0
    top_templates: list[dict] = []
    tm_ms: float = 0.0

    if not disable_layer_b:
        corpus_path = _Path(__file__).parent / "data" / "jailbreak_templates.jsonl"
        cache_dir = _Path(__file__).parent / "data"

        tm = get_template_matcher(
            embed_model_id=template_embed_model_id,
            device=device,
            corpus_path=corpus_path,
            tau_component=tau_component,
            cache_dir=cache_dir,
        )

        tm_results = [tm.match(t) for t in texts]
        tm_ms = (time.perf_counter() - t_tm_start) * 1000.0

        tm_max_sim = max(r.max_sim for r in tm_results)

        for r in tm_results:
            for comp, sim in r.components.items():
                if sim > tm_components.get(comp, 0.0):
                    tm_components[comp] = sim

        tm_n_components = sum(1 for v in tm_components.values() if v >= tau_component)

        all_top: list[dict] = []
        for r in tm_results:
            all_top.extend(r.top)
        seen_ids: dict = {}
        for entry in all_top:
            eid = entry["id"]
            if eid not in seen_ids or entry["sim"] > seen_ids[eid]["sim"]:
                seen_ids[eid] = entry
        top_templates = sorted(seen_ids.values(), key=lambda x: x["sim"], reverse=True)[:5]

    # ------------------------------------------------------------------
    # Layer H — Harmful-intent classifier (what is asked, not how)
    # ------------------------------------------------------------------
    t_h_start = time.perf_counter()
    harm_score: Optional[float] = None
    harm_block = harm_flag = None
    harm_ms = 0.0
    already_blocked = (
        (pg_score is not None and pg_score >= pg_block)
        or tm_max_sim >= tm_block
        or tm_n_components >= components_block
    )
    # Skip when Layers A/B already block: saves seconds on long jailbreak prompts.
    if not disable_layer_h and not already_blocked:
        from src.security.harm_classifier import get_harm_classifier

        hc = get_harm_classifier(device)
        if hc is not None:
            harm_texts = list(dict.fromkeys(t for t in (text_message, text_profile, text_multiturn) if t.strip()))
            harm_score = max(h.score for h in hc.score_many(harm_texts)) if harm_texts else 0.0
            harm_block = float(thresholds.get("harm_block", hc.block_threshold))
            harm_flag = float(thresholds.get("harm_flag", hc.flag_threshold))
        harm_ms = (time.perf_counter() - t_h_start) * 1000.0

    # Layer G - Llama Guard on the message (with recent turns for multi-turn
    # attacks). Skipped when A/B already block or Layer H already blocks.
    guard = None
    harm_blocks = harm_score is not None and harm_block is not None and harm_score >= harm_block
    if not disable_layer_g and not already_blocked and not harm_blocks:
        from src.llm.adapters import LLMUnavailable
        from src.security.llama_guard import get_input_guard

        g = get_input_guard()
        if g is not None:
            try:
                guard = g.classify(text_multiturn)
            except LLMUnavailable as exc:
                logger.warning("Llama Guard unavailable; Layer G bypassed: %s", exc)

    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    latency_ms = {"pg": round(pg_ms, 1), "tm": round(tm_ms, 1), "harm": round(harm_ms, 1),
                  "guard": guard.latency_ms if guard else 0.0, "total": round(total_ms, 1)}

    import math
    fused_logit = (
        -4.5
        + 4.0 * (pg_score or 0.0)
        + 3.0 * (pg_smooth_mean or 0.0)
        + 1.5 * tm_max_sim
        + 1.5 * (tm_n_components / 9.0)
        + 2.5 * (harm_score or 0.0)
        + 3.0 * (1.0 if guard is not None and guard.unsafe else 0.0)
    )
    fused_risk = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, fused_logit))))

    current_session_risk = 0.0
    if session_id:
        from src.security.session_risk import session_risk
        current_session_risk = session_risk.update(
            session_id,
            max(fused_risk, pg_score or 0.0, tm_max_sim, harm_score or 0.0),
        )
    session_block = float(thresholds.get("session_block", 0.55))

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------
    reasons: list[str] = []

    # BLOCK conditions
    do_block = False
    if pg_score is not None and pg_score >= pg_block:
        do_block = True
        reasons.append("pg_score_block")
    if tm_max_sim >= tm_block:
        do_block = True
        reasons.append("tm_sim_block")
    if tm_n_components >= components_block:
        do_block = True
        reasons.append("tm_components_block")
    if harm_score is not None and harm_score >= harm_block:
        do_block = True
        reasons.append("harm_block")
    if guard is not None and guard.blocks:
        do_block = True
        reasons.append("guard_block")
    if pg_smooth_mean is not None and pg_smooth_mean >= float(thresholds.get("smooth_block", 0.90)):
        do_block = True
        reasons.append("pg_smooth_block")
    if current_session_risk >= session_block:
        do_block = True
        reasons.append("session_risk_block")

    # FLAG conditions
    do_flag = False
    if pg_score is not None and pg_score >= pg_flag:
        do_flag = True
        reasons.append("pg_score_flag")
    if tm_max_sim >= tm_flag:
        do_flag = True
        reasons.append("tm_sim_flag")
    if tm_n_components >= components_flag:
        do_flag = True
        reasons.append("tm_components_flag")
    if harm_score is not None and harm_score >= harm_flag:
        do_flag = True
        reasons.append("harm_flag")
    if any(n.startswith("control_token:") for n in sanitize_notes):
        do_flag = True
        reasons.append("control_token_detected")
    if pg_smooth_mean is not None and pg_smooth_mean >= float(thresholds.get("smooth_flag", 0.50)):
        do_flag = True
        reasons.append("pg_smooth_flag")

    # Determine action
    if do_block:
        action: Literal["ALLOW", "FLAG", "BLOCK"] = "BLOCK"
    elif do_flag:
        action = "FLAG"
    else:
        action = "ALLOW"

    # ------------------------------------------------------------------
    # Category classification
    # ------------------------------------------------------------------
    category: Optional[Literal["JAILBREAK", "INJECTION", "STRUCTURAL", "HARMFUL_REQUEST"]] = None

    if action != "ALLOW":
        # Did Layer A fire on the message specifically?
        pg_fired_message = (
            pg_results
            and pg_results[0].max_score is not None
            and pg_results[0].max_score >= pg_flag
        )
        # Did Layer A fire on profile or history only?
        pg_fired_profile_or_history = (
            pg_results
            and not pg_fired_message
            and any(
                r.max_score is not None and r.max_score >= pg_flag
                for r in pg_results[1:]
            )
        )
        # Did Layer B match roleplay-persona patterns?
        tm_roleplay_match = tm_n_components >= components_flag

        attack_framing = (
            (pg_score is not None and pg_score >= pg_flag)
            or tm_max_sim >= tm_flag
            or tm_n_components >= components_flag
        )
        harm_fired = (harm_score is not None and harm_score >= harm_flag) or (
            guard is not None and guard.blocks)
        only_sanitizer = (
            not do_block
            and not attack_framing
            and not harm_fired
            and any(n.startswith("control_token:") for n in sanitize_notes)
        )

        if harm_fired and not attack_framing:
            # Plainly worded harmful request: no jailbreak or injection framing.
            category = "HARMFUL_REQUEST"
        elif only_sanitizer:
            category = "STRUCTURAL"
        elif tm_roleplay_match or pg_fired_message:
            category = "JAILBREAK"
        elif pg_fired_profile_or_history:
            category = "INJECTION"
        else:
            # Generic fallback
            category = "JAILBREAK"

    return DefenseDecision(
        action=action,
        category=category,
        pg_score=round(pg_score, 4) if pg_score is not None else None,
        tm_max_sim=tm_max_sim,
        tm_components=tm_components,
        tm_n_components=tm_n_components,
        top_templates=top_templates,
        sanitize_notes=sanitize_notes,
        reasons=reasons,
        latency_ms=latency_ms,
        harm_score=round(harm_score, 4) if harm_score is not None else None,
        guard_unsafe=guard.unsafe if guard else None,
        guard_categories=guard.categories if guard else None,
        pg_smooth_mean=round(pg_smooth_mean, 4) if pg_smooth_mean is not None else None,
        pg_smooth_std=round(pg_smooth_std, 4) if pg_smooth_std is not None else None,
        session_risk=round(current_session_risk, 4),
        fused_risk=round(fused_risk, 4),
    )
