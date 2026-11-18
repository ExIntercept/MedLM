"""Model-agnostic clinical defense gateway.

``ClinicalDefenseGateway`` sits between the user and *any* LLM. Every request
passes through the same ordered stages, and the protected model is only ever
reached through an :class:`~src.llm.adapters.LLMAdapter`:

    1. sanitize   - strip invisible chars, defang every model family's control tokens
    2. triage     - deterministic emergency red-flag escalation
    3. guardrails - hard-coded contraindication intercepts
    4. detect     - Prompt Guard + Template Matcher -> ALLOW / FLAG / BLOCK
    5. retrieve   - optional evidence retrieval (injected callable)
    6. harden     - nonce-tagged system/user messages (no model-specific markers)
    7. generate   - adapter.stream(...)  <- the only model-specific step
    8. verify     - output clean-up + evidence audit; unverifiable answers withheld

The gateway is transport-free: it reports progress through an ``on_event``
callback and returns a :class:`GatewayResult`. The FastAPI server turns those
into SSE events and database rows; evaluation scripts can call it directly with
any adapter (including :class:`~src.llm.adapters.CallableAdapter`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.security.sanitize import SanitizedField, SanitizedInputs, sanitize_user_inputs

logger = logging.getLogger(__name__)

# Patient-facing refusal for security blocks. Deliberately vague: reveals
# nothing about which detector fired.
BLOCK_MESSAGE = (
    "I can only help with genuine health questions, and I can't respond to role-play or "
    "instruction-override requests. If you have a medical question, please ask it directly "
    "and I'll do my best to help. If this is an emergency, contact your local emergency number."
)

# Shown when the model's answer cannot be verified against retrieved evidence.
WITHHELD_MESSAGE = (
    "I couldn't verify enough of this response against the retrieved clinical "
    "evidence to present it safely. Please consult a qualified healthcare "
    "professional for individualized advice."
)

DEFAULT_TOGGLES: dict[str, bool] = {
    "defense_enabled": True,
    "layer_sanitize": True,
    "layer_triage": True,
    "layer_guardrails": True,
    "layer_prompt_guard": True,
    "layer_template_matcher": True,
    "layer_prompt_hardening": True,
}

Evidence = list  # list[tuple[chunk_id, text, meta]]
Retriever = Callable[[str], Evidence]
EventSink = Callable[[str, Any], None]


@dataclass
class GatewayResult:
    """Outcome of one request through the gateway."""

    status: str
    """ANSWERED, BLOCKED, EMERGENCY_ESCALATION or INTERCEPTED."""

    response_text: str
    """The text that is safe to show the user."""

    user_text: str
    """The sanitized user message (what should be persisted)."""

    decision: Any = None
    """The DefenseDecision from the detect stage, if it ran."""

    audit: Optional[dict] = None
    """Evidence audit from the verify stage, if it ran."""

    sources: list = field(default_factory=list)
    withheld: bool = False
    """True when the model's answer was replaced by WITHHELD_MESSAGE."""

    model: Optional[str] = None
    """Name of the adapter that generated the answer, if any."""

    trace: list = field(default_factory=list)
    """Ordered list of stages that ran, for audit and the UI."""

    meta: dict = field(default_factory=dict)
    """Stage-specific metadata to persist with the assistant message."""


class ClinicalDefenseGateway:
    """Run the defense pipeline around an arbitrary LLM adapter."""

    def __init__(
        self,
        llm,
        retriever: Optional[Retriever] = None,
        toggles: Optional[dict] = None,
        temperature: float = 0.2,
        max_tokens: int = 640,
    ):
        """
        Args:
            llm:         Any object satisfying the ``LLMAdapter`` protocol.
            retriever:   ``fn(query) -> [(chunk_id, text, meta), ...]``; None disables RAG.
            toggles:     Layer switches (see DEFAULT_TOGGLES). The dict is read on every
                         request, so the admin console can change it live.
            temperature: Sampling temperature passed to the adapter.
            max_tokens:  Generation budget passed to the adapter.
        """
        self.llm = llm
        self.retriever = retriever
        self.toggles = toggles if toggles is not None else dict(DEFAULT_TOGGLES)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _on(self, key: str) -> bool:
        return self.toggles.get(key, DEFAULT_TOGGLES.get(key, True))

    # ------------------------------------------------------------------
    def run(
        self,
        message: str,
        profile: Any = None,
        history: Optional[list[dict]] = None,
        mode: str = "patient",
        conversation_id: Optional[int] = None,
        on_event: Optional[EventSink] = None,
    ) -> GatewayResult:
        """Process one user turn end to end.

        Args:
            message:         Raw user message.
            profile:         Patient profile (pydantic model, dict, or None).
            history:         Server-side conversation history, oldest first,
                             as ``[{"role": ..., "content": ...}]``. Never pass
                             client-supplied history: it enables turn forgery.
            mode:            "patient" or "clinician" (controls tone).
            conversation_id: Used only for the defense log.
            on_event:        ``fn(event_name, data)`` progress callback.
        """
        emit: EventSink = on_event or (lambda _e, _d: None)
        history = history or []
        trace: list[str] = []

        # ── 1. Sanitize ────────────────────────────────────────────────
        if self._on("layer_sanitize"):
            san = sanitize_user_inputs(message, profile)
            trace.append("sanitize")
        else:
            san = SanitizedInputs(
                message=SanitizedField(model_text=message, detector_text=message, modified=False, notes=[]),
                profile={},
            )
        user_text = san.message.model_text

        # ── 2. Emergency triage ────────────────────────────────────────
        if self._on("layer_triage"):
            from src.agents.emergency_triage import EMERGENCY_WARNING, check_emergency_triage

            trace.append("triage")
            category = check_emergency_triage(user_text)
            if category:
                emit("sources", [])
                emit("triage_status", {"level": "EMERGENCY", "category": category})
                emit("token", EMERGENCY_WARNING)
                return GatewayResult(
                    status="EMERGENCY_ESCALATION",
                    response_text=EMERGENCY_WARNING,
                    user_text=user_text,
                    trace=trace,
                    meta={"emergency_triage": category},
                )

        # ── 3. Contraindication guardrails ─────────────────────────────
        if self._on("layer_guardrails"):
            from src.agents.guardrails import check_hard_rules, format_guardrail_intercept

            trace.append("guardrails")
            hit = check_hard_rules(user_text)
            if hit:
                entity, _, warning = hit
                text = format_guardrail_intercept(entity, warning, mode=mode)
                emit("sources", [])
                emit("token", text)
                return GatewayResult(
                    status="INTERCEPTED",
                    response_text=text,
                    user_text=user_text,
                    trace=trace,
                    meta={"guardrail_intercept": True},
                )

        # ── 4. Adversarial detection ───────────────────────────────────
        decision = None
        flagged = False
        if self._on("defense_enabled"):
            from src.security.decision import run_input_defenses
            from src.security.logging import log_decision

            trace.append("detect")
            prior_user_turns = [t["content"] for t in history if t.get("role") == "user"][-2:]
            decision = run_input_defenses(
                san,
                prior_user_turns,
                disable_layer_a=not self._on("layer_prompt_guard"),
                disable_layer_b=not self._on("layer_template_matcher"),
            )
            log_decision(decision, conversation_id, None, san.message.detector_text)

            if decision.action == "BLOCK":
                emit("sources", [])
                emit("security_status", {"action": "BLOCK", "category": decision.category})
                emit("token", BLOCK_MESSAGE)
                return GatewayResult(
                    status="BLOCKED",
                    response_text=BLOCK_MESSAGE,
                    user_text=user_text,
                    decision=decision,
                    trace=trace,
                    meta={
                        "security_block": decision.category,
                        "pg": decision.pg_score,
                        "tm": decision.tm_max_sim,
                        "n_components": decision.tm_n_components,
                    },
                )
            if decision.action == "FLAG":
                emit("security_status", {"action": "FLAG"})
                flagged = True

        from src.agents.intake_extraction import extract_intake_fields

        extracted = extract_intake_fields(user_text)
        if extracted:
            emit("profile_update", extracted)

        # ── 5. Retrieval ───────────────────────────────────────────────
        from src.agents.guardrails import parse_embedded_options

        vignette_text, detected_options = parse_embedded_options(user_text)
        evidence: Evidence = []
        if self.retriever is not None:
            trace.append("retrieve")
            evidence = list(self.retriever(vignette_text) or [])

        # ── 6. Prompt hardening, fitted to this model's context window ─
        from src.agents.prompting import format_patient_profile

        hardened = self._on("layer_prompt_hardening")
        trace.append("harden" if hardened else "prompt")
        profile_note = format_patient_profile(
            **{k: (san.profile[k].model_text if k in san.profile else None)
               for k in ("age", "sex", "duration", "conditions", "medications")}
        )
        forced_tone = "patient" if mode == "patient" else "clinical_qa"
        model_profile = getattr(self.llm, "profile", None)
        max_tokens = self.max_tokens
        if model_profile is not None:
            # Never let the output budget eat more than half of a small context.
            max_tokens = min(max_tokens, model_profile.context_window // 2)

        def compose(ev, hist, vignette):
            if hardened:
                from src.agents.prompting import build_messages

                return build_messages(
                    vignette, detected_options, _format_evidence(ev), hist,
                    forced_tone=forced_tone, patient_profile_text=profile_note, flagged=flagged,
                )
            return _unhardened_messages(vignette, detected_options, _format_evidence(ev), hist, forced_tone, profile_note)

        messages, evidence, fit_notes = fit_to_context(
            compose, evidence, history, vignette_text, model_profile, max_tokens
        )
        if fit_notes:
            trace.append("fit_context")

        sources = [
            {
                "chunk_id": str(chunk_id),
                "title": meta.get("title", "Guideline"),
                "source": meta.get("source", ""),
                "score": round(float(meta.get("cross_score", 0.0)), 4),
                "excerpt": text[:280],
            }
            for chunk_id, text, meta in evidence
        ]
        emit("sources", sources)

        # ── 7. Generate (the only model-specific step) ─────────────────
        trace.append("generate")
        raw_answer = "".join(
            self.llm.stream(messages, temperature=self.temperature, max_tokens=max_tokens)
        )

        # ── 8. Output normalisation, verification and gate ─────────────
        from src.agents.prompting import sanitize_patient_output
        from src.agents.verifier import verify_evidence

        trace.append("verify")
        answer = sanitize_patient_output(clean_model_output(raw_answer, model_profile))
        audit = verify_evidence(answer, evidence, vignette_text)
        withheld = audit["status"] == "FLAGGED"
        if withheld:
            answer = WITHHELD_MESSAGE
            emit("verification_status", {"status": "FLAGGED"})
        emit("token", answer)

        return GatewayResult(
            status="ANSWERED",
            response_text=answer,
            user_text=user_text,
            decision=decision,
            audit=audit,
            sources=sources,
            withheld=withheld,
            model=getattr(self.llm, "name", type(self.llm).__name__),
            trace=trace,
            meta={
                "faithfulness_score": audit["score"],
                "status": audit["status"],
                "model": getattr(self.llm, "name", type(self.llm).__name__),
                "model_family": model_profile.family if model_profile else None,
                "context_fit": fit_notes,
                "security": {
                    "action": decision.action if decision else "BYPASSED",
                    "pg": decision.pg_score if decision else None,
                    "tm_max": decision.tm_max_sim if decision else 0.0,
                    "n_components": decision.tm_n_components if decision else 0,
                },
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_evidence(evidence: Evidence) -> str:
    out = ""
    for idx, (_, text, meta) in enumerate(evidence, 1):
        out += f"### [Evidence {idx}] {meta.get('title', 'Guideline')} (`{meta.get('source', '')}`)\n"
        out += f"*MedCPT Rerank Score:* `{meta.get('cross_score', 0):.4f}`\n\n> {text}\n\n---\n"
    return out


def _unhardened_messages(vignette_text, detected_options, evidence_md, history, forced_tone, profile_note):
    """Ablation path: the legacy un-tagged prompt as a single user message."""
    from src.agents.prompting import build_prompt

    prompt = build_prompt(
        vignette_text,
        detected_options,
        evidence_md,
        history,
        forced_tone=forced_tone,
        prompt_style="plain",
        patient_profile_text=profile_note,
        hardened=False,
    )
    return [{"role": "user", "content": prompt}]


# ---------------------------------------------------------------------------
# Model-specific adaptation (driven by ModelProfile, never by model name)
# ---------------------------------------------------------------------------

_MESSAGE_OVERHEAD_TOKENS = 8
_SAFETY_MARGIN = 0.92  # token estimates are approximate; stay under the limit


def estimate_tokens(messages: list, chars_per_token: float = 3.4) -> int:
    return sum(int(len(m["content"]) / chars_per_token) + _MESSAGE_OVERHEAD_TOKENS for m in messages)


def fit_to_context(compose, evidence, history, vignette, model_profile, max_tokens):
    """Shrink the prompt until it fits ``context_window - max_tokens``.

    Small-context medical models (Meditron-7B, BioMistral: 2048 tokens) cannot
    take the full hardened prompt. Content is dropped in order of least
    clinical value: older history, then lower-ranked evidence, then evidence
    length, and only as a last resort the user's own question. The safety
    instructions and delimiter rules are never trimmed.

    Returns ``(messages, evidence_used, notes)``.
    """
    messages = compose(evidence, history, vignette)
    if model_profile is None:
        return messages, evidence, []

    budget = int((model_profile.context_window - max_tokens) * _SAFETY_MARGIN)
    cpt = model_profile.chars_per_token

    def fits(msgs):
        return estimate_tokens(msgs, cpt) <= budget

    notes: list[str] = []
    if fits(messages):
        return messages, evidence, notes

    # 1. Drop older history, keeping the most recent exchange as long as possible.
    for keep in (2, 0):
        if len(history) > keep:
            history = history[-keep:] if keep else []
            notes.append(f"history_kept:{keep}")
            messages = compose(evidence, history, vignette)
            if fits(messages):
                return messages, evidence, notes

    # 2. Drop lowest-ranked evidence chunks (retriever returns best first).
    while len(evidence) > 1:
        evidence = evidence[:-1]
        notes.append(f"evidence_kept:{len(evidence)}")
        messages = compose(evidence, history, vignette)
        if fits(messages):
            return messages, evidence, notes

    # 3. Shorten the remaining evidence chunk.
    if evidence:
        chunk_id, text, meta = evidence[0]
        overflow_chars = int((estimate_tokens(messages, cpt) - budget) * cpt) + 50
        new_len = max(200, len(text) - overflow_chars)
        if new_len < len(text):
            evidence = [(chunk_id, text[:new_len].rsplit(" ", 1)[0] + " ...", meta)]
            notes.append(f"evidence_truncated:{new_len}")
            messages = compose(evidence, history, vignette)
            if fits(messages):
                return messages, evidence, notes

    # 4. Last resort: shorten the user's question itself.
    overflow_chars = int((estimate_tokens(messages, cpt) - budget) * cpt) + 50
    if overflow_chars > 0 and len(vignette) > 200:
        vignette = vignette[: max(200, len(vignette) - overflow_chars)]
        notes.append("question_truncated")
        messages = compose(evidence, history, vignette)

    if not fits(messages):
        notes.append("over_budget")
        logger.warning("Prompt still exceeds %s context after trimming: %s", model_profile.family, notes)
    return messages, evidence, notes


# A base model often keeps writing the next turn of a transcript it imagines.
_FABRICATED_TURN_RE = re.compile(
    r"\n\s*(?:#{1,4}\s*)?(?:User|Patient|Human|Question|Instruction|Current Question|"
    r"Previous Case History|Retrieved Guidelines)\s*:?\s*\n|\n\s*(?:User|Human|Patient|Question)\s*:",
    re.IGNORECASE,
)
_PROMPT_ECHO_RE = re.compile(r"^\s*(?:Provide your response:|Answer:)\s*", re.IGNORECASE)


def clean_model_output(text: str, model_profile=None) -> str:
    """Remove model-specific artifacts so every model's answer looks the same.

    * all families' leftover control tokens and echoed delimiter tags
    * an echoed "Provide your response:" cue at the start
    * for base (non-instruction-tuned) models, anything after a fabricated
      next turn such as "\nUser:" or "### Current Question"
    """
    from src.security.sanitize import strip_control_tokens

    text = strip_control_tokens(text)
    text = _PROMPT_ECHO_RE.sub("", text)
    if model_profile is not None and not model_profile.instruction_tuned:
        m = _FABRICATED_TURN_RE.search(text)
        if m:
            text = text[: m.start()]
    return text.strip()
