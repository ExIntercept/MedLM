"""Profiles for the medical LLMs the defense gateway can protect.

Medical models differ in ways that matter to a defense layer:

* **Prompt format** - MedGemma uses Gemma turns, OpenBioLLM / Med42-v2 use
  Llama 3 headers, BioMistral uses Mistral ``[INST]``, Meditron / PMC-LLaMA are
  base models with no chat format at all.
* **Context window** - Meditron-7B and BioMistral were trained at 2048 tokens,
  so the full hardened prompt (rules + evidence + history) must be trimmed.
* **System role** - Mistral-family templates reject a ``system`` message.
* **Instruction tuning** - base models do not follow refusal rules reliably,
  so the gateway's pre-model (detect/BLOCK) and post-model (verify) stages
  carry more of the load for them.

``resolve_profile(model_id)`` matches a Hugging Face model id against the
registry. Unknown models fall back to the ``LLM_TEMPLATE`` / ``MAX_MODEL_LEN``
settings, so any other model still works with explicit configuration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ModelProfile:
    family: str
    """Human-readable model family, e.g. "MedGemma"."""

    template: str
    """Template for raw completions: gemma | chatml | llama3 | mistral | plain."""

    context_window: int
    """Maximum tokens (prompt + output) the model was trained for."""

    system_role: bool = True
    """Whether the chat template accepts a ``system`` message."""

    instruction_tuned: bool = True
    """False for base models that only continue text."""

    chars_per_token: float = 3.4
    """Rough characters-per-token ratio used for context budgeting."""

    notes: str = ""


# Ordered: first matching pattern wins, so specific entries come before generic ones.
MEDICAL_MODEL_PROFILES: list[tuple[str, ModelProfile]] = [
    (r"medgemma", ModelProfile(
        "MedGemma", "gemma", 8192,
        notes="google/medgemma-4b-it, medgemma-27b-text-it; Gemma 3 turns. "
              "Supports far longer contexts; 8192 matches the Colab vLLM setting.")),
    (r"meditron.*70b", ModelProfile(
        "Meditron-70B", "plain", 4096, system_role=False, instruction_tuned=False,
        notes="Llama 2 base model; no chat template.")),
    (r"meditron", ModelProfile(
        "Meditron-7B", "plain", 2048, system_role=False, instruction_tuned=False,
        notes="epfl-llm/meditron-7b; Llama 2 base model, 2048-token context.")),
    (r"biomistral", ModelProfile(
        "BioMistral", "mistral", 2048, system_role=False,
        notes="BioMistral/BioMistral-7B; Mistral [INST] format, trained at 2048 tokens.")),
    (r"openbiollm", ModelProfile(
        "OpenBioLLM", "llama3", 8192,
        notes="aaditya/Llama3-OpenBioLLM-8B / 70B; Llama 3 instruct format.")),
    (r"med42.*(v2|llama3|llama-3)|llama3-med42", ModelProfile(
        "Med42-v2", "llama3", 8192,
        notes="m42-health/Llama3-Med42-8B / 70B; Llama 3 instruct format.")),
    (r"pmc[_-]?llama", ModelProfile(
        "PMC-LLaMA", "plain", 2048, system_role=False, instruction_tuned=False,
        notes="axiong/PMC_LLaMA_13B; LLaMA base model.")),
    (r"clinicalcamel|clinical-camel", ModelProfile(
        "ClinicalCamel", "plain", 4096, system_role=False,
        notes="wanglab/ClinicalCamel-70B; Llama 2 dialogue fine-tune without a standard chat template.")),
    (r"huatuogpt-o1-(7|72)b", ModelProfile(
        "HuatuoGPT-o1 (Qwen2.5)", "chatml", 32768,
        notes="FreedomIntelligence/HuatuoGPT-o1-7B / 72B; Qwen 2.5 ChatML format.")),
    (r"huatuogpt-o1-(8|70)b", ModelProfile(
        "HuatuoGPT-o1 (Llama 3.1)", "llama3", 32768,
        notes="FreedomIntelligence/HuatuoGPT-o1-8B / 70B; Llama 3.1 format.")),
    (r"bio-medical-llama-3|jsl-medllama-3|medllama3", ModelProfile(
        "Medical Llama 3 fine-tune", "llama3", 8192)),
    # Not a medical model: the general-purpose control for cross-model evaluation.
    (r"meta-llama/(meta-)?llama-3(\.\d)?-\d+b-instruct", ModelProfile(
        "Llama 3.x Instruct (general control)", "llama3", 8192,
        notes="meta-llama/Llama-3.1-8B-Instruct; shows whether medical fine-tuning changed vulnerability.")),
]


def resolve_profile(
    model_id: str,
    fallback_template: Optional[str] = None,
    fallback_context: Optional[int] = None,
) -> ModelProfile:
    """Return the profile for ``model_id``; unknown ids get a generic profile.

    Args:
        model_id:          Model name as served, e.g. ``google/medgemma-4b-it``.
        fallback_template: Template for unknown models (defaults to ``plain``).
        fallback_context:  Context window for unknown models (defaults to 4096).
    """
    lowered = (model_id or "").lower()
    for pattern, profile in MEDICAL_MODEL_PROFILES:
        if re.search(pattern, lowered):
            return profile
    return ModelProfile(
        family="Unregistered model",
        template=fallback_template or "plain",
        context_window=fallback_context or 4096,
        notes="Not in the medical model registry; using LLM_TEMPLATE / MAX_MODEL_LEN.",
    )


def profile_from_config() -> ModelProfile:
    """Profile for the configured model, with explicit ``.env`` overrides applied.

    ``LLM_TEMPLATE``, ``LLM_SYSTEM_ROLE`` and ``MAX_MODEL_LEN`` override the
    registry only when they are set explicitly in the environment.
    """
    import os

    from src.config import LLM_MODEL, LLM_TEMPLATE, MAX_MODEL_LEN

    profile = resolve_profile(LLM_MODEL, fallback_template=LLM_TEMPLATE, fallback_context=MAX_MODEL_LEN)
    overrides = {}
    if os.environ.get("LLM_TEMPLATE"):
        overrides["template"] = os.environ["LLM_TEMPLATE"].lower()
    if os.environ.get("LLM_SYSTEM_ROLE"):
        overrides["system_role"] = os.environ["LLM_SYSTEM_ROLE"] == "1"
    if os.environ.get("MAX_MODEL_LEN"):
        # The server's real limit always wins: prompts larger than it are rejected.
        overrides["context_window"] = min(profile.context_window, MAX_MODEL_LEN) \
            if profile.family != "Unregistered model" else MAX_MODEL_LEN
    return replace(profile, **overrides) if overrides else profile
