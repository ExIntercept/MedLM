"""Security defense pipeline: sanitize, detect, decide.

Public API:
    sanitize_user_inputs  - Layer C part 1: neutralise control tokens and invisible chars
    run_input_defenses    - Layers A+B: Prompt Guard + Template Matcher + decision
    DefenseDecision       - Result dataclass from run_input_defenses
    warmup                - Load both detectors and run one dummy inference
"""
from src.security.sanitize import sanitize_user_inputs, SanitizedInputs, SanitizedField  # noqa: F401
from src.security.decision import run_input_defenses, DefenseDecision  # noqa: F401


def warmup() -> None:
    """Pre-load both detectors and run a dummy inference.

    Call from FastAPI startup in a background thread so the first real
    request does not pay the model-load cost.
    """
    import logging
    logger = logging.getLogger(__name__)
    from src.config import DEFENSE_ENABLED, PROMPT_GUARD_MODEL_ID, TEMPLATE_EMBED_MODEL_ID, DEFENSE_THRESHOLDS_PATH
    from src.security.device import get_device

    if not DEFENSE_ENABLED:
        logger.info("Defense pipeline disabled (DEFENSE_ENABLED=0). Skipping warmup.")
        return

    device = get_device()
    logger.info("[Security warmup] Device: %s", device)

    # Layer A: Prompt Guard
    try:
        from src.security.prompt_guard import get_prompt_guard
        pg = get_prompt_guard(PROMPT_GUARD_MODEL_ID, device)
        if pg is not None:
            pg.score("warmup")
            logger.info("[Security warmup] Prompt Guard ready.")
        else:
            logger.warning("[Security warmup] Prompt Guard unavailable (gating issue). Layer A disabled.")
    except Exception as exc:
        logger.error("[Security warmup] Prompt Guard failed: %s", exc)

    # Layer B: Template Matcher
    try:
        from pathlib import Path
        from src.security.template_matcher import get_template_matcher
        corpus_path = Path(__file__).parent / "data" / "jailbreak_templates.jsonl"
        cache_dir = Path(__file__).parent / "data"
        thresholds = {}
        if DEFENSE_THRESHOLDS_PATH.exists():
            import json
            thresholds = json.loads(DEFENSE_THRESHOLDS_PATH.read_text(encoding="utf-8"))
        tau = thresholds.get("tau_component", 0.65)
        tm = get_template_matcher(TEMPLATE_EMBED_MODEL_ID, device, corpus_path, tau, cache_dir)
        tm.match("warmup")
        logger.info("[Security warmup] Template Matcher ready.")
    except Exception as exc:
        logger.error("[Security warmup] Template Matcher failed: %s", exc)

    # Layer H: harmful-intent classifier
    try:
        from src.security.harm_classifier import get_harm_classifier
        hc = get_harm_classifier(device)
        if hc is not None:
            hc.score("warmup")
            logger.info("[Security warmup] Harm classifier ready.")
    except Exception as exc:
        logger.error("[Security warmup] Harm classifier failed: %s", exc)

    logger.info("[Security warmup] Complete.")
