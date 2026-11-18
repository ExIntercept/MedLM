"""Input sanitization for the clinical AI defense pipeline — Layer C part 1.

Applies the following transformations IN ORDER before text reaches the model:
  1. Strip invisible / format (Cf) characters.
  2. Neutralise Gemma / ChatML control tokens.
  3. Neutralise forged application delimiter tags.
  4. Truncate to max_chars.
  5. Build detector_text via NFKC normalisation (model_text is NOT normalised).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SanitizedField:
    """Result of sanitizing a single text field."""

    model_text: str
    """Text that will be sent to the generative model (not NFKC-normalised)."""

    detector_text: str
    """NFKC-normalised version used by all detectors."""

    modified: bool
    """True if any transformation changed the original text."""

    notes: list[str]
    """Ordered list of transformation notes, e.g. ["zero_width:3", "truncated"]."""


@dataclass(frozen=True)
class SanitizedInputs:
    """Sanitized user message and patient profile fields."""

    message: SanitizedField
    """Sanitized free-text message."""

    profile: dict[str, SanitizedField]
    """Sanitized profile fields. Only non-empty fields are present.

    Possible keys: age, sex, duration, conditions, medications.
    """


# ---------------------------------------------------------------------------
# Internal regex patterns (compiled once at module load)
# ---------------------------------------------------------------------------

# Step 1: invisible / format characters
_ZERO_WIDTH_RE = re.compile(
    "[\u200b-\u200f\u2060-\u2064\ufeff]"
)

_BIDI_CTRL_RE = re.compile(
    "[\u202a-\u202e\u2066-\u2069]"
)

_TAG_BLOCK_RE = re.compile("[\U000e0000-\U000e007f]")

# Step 2: Gemma / ChatML control tokens
_GEMMA_TOKEN_RE = re.compile(
    r"<\s*/?\s*(start_of_turn|end_of_turn|bos|eos|pad|unk|mask"
    r"|start_of_image|end_of_image|image_soft_token"
    r"|unused\d+)\s*>",
    re.IGNORECASE,
)

# <|im_start|>, <|im_end|>, and generic <|...|> delimiters
# (also covers Llama 3 <|begin_of_text|>, <|start_header_id|>, <|eot_id|>, Phi, Qwen)
_CHATML_TOKEN_RE = re.compile(r"<\|[^|>]{1,40}\|>", re.IGNORECASE)

# Llama 2 / Mistral instruction markers: [INST] [/INST] [SYSTEM_PROMPT] [TOOL_CALLS] ...
_BRACKET_TOKEN_RE = re.compile(
    r"\[\s*/?\s*(INST|SYSTEM_PROMPT|AVAILABLE_TOOLS|TOOL_CALLS|TOOL_RESULTS)\s*\]",
    re.IGNORECASE,
)

# Llama 2 system delimiters: <<SYS>> <</SYS>>
_SYS_TOKEN_RE = re.compile(r"<<\s*/?\s*SYS\s*>>", re.IGNORECASE)

# SentencePiece sequence markers used by Llama 2 / Mistral templates: <s> </s>
_SEQ_TOKEN_RE = re.compile(r"<\s*/?\s*s\s*>", re.IGNORECASE)

# Step 3: Forged application delimiter tags
_APP_TAG_RE = re.compile(
    r"</?(user_message|conversation_log|reference_material|patient_profile)"
    r"(?:_[0-9a-fA-F]+)?\s*/?>",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core sanitization logic
# ---------------------------------------------------------------------------

def _neutralise_angle(match: re.Match) -> str:  # type: ignore[type-arg]
    """Replace < with \u2039 (U+2039) and > with \u203a (U+203A) in a matched token."""
    s = match.group(0)
    return "\u2039" + s[1:-1] + "\u203a"


def neutralise_control_tokens(text: str) -> tuple[str, list[str]]:
    """Defang chat-template control tokens for every supported model family.

    Covers Gemma (``<start_of_turn>``), ChatML / Llama 3 / Qwen / Phi
    (``<|...|>``), Llama 2 / Mistral (``[INST]``, ``<<SYS>>``, ``<s>``). Tokens
    are made inert by swapping their delimiters for look-alike glyphs, so the
    text stays readable but can no longer be parsed as a turn boundary by any
    tokenizer. Returns the new text and a list of ``control_token:<name>`` notes.

    This is model-agnostic on purpose: the gateway may route the same request
    to any backend, so every family's markers are neutralised regardless of
    which model is configured.
    """
    notes: list[str] = []

    def _replace_gemma(m: re.Match) -> str:  # type: ignore[type-arg]
        notes.append(f"control_token:{m.group(1).lower()}")
        return _neutralise_angle(m)

    def _replace_chatml(m: re.Match) -> str:  # type: ignore[type-arg]
        notes.append(f"control_token:{m.group(0)[2:-2].strip()}")
        return _neutralise_angle(m)

    def _replace_bracket(m: re.Match) -> str:  # type: ignore[type-arg]
        notes.append(f"control_token:{m.group(1).lower()}")
        s = m.group(0)
        return "\u2045" + s[1:-1] + "\u2046"

    def _replace_sys(m: re.Match) -> str:  # type: ignore[type-arg]
        notes.append("control_token:sys")
        s = m.group(0)
        return "\u2039\u2039" + s[2:-2] + "\u203a\u203a"

    def _replace_seq(m: re.Match) -> str:  # type: ignore[type-arg]
        notes.append("control_token:s")
        return _neutralise_angle(m)

    text = _GEMMA_TOKEN_RE.sub(_replace_gemma, text)
    text = _CHATML_TOKEN_RE.sub(_replace_chatml, text)
    text = _BRACKET_TOKEN_RE.sub(_replace_bracket, text)
    text = _SYS_TOKEN_RE.sub(_replace_sys, text)
    text = _SEQ_TOKEN_RE.sub(_replace_seq, text)
    return text, notes


def strip_control_tokens(text: str) -> str:
    """Remove (not defang) every model family's control tokens and app tags.

    Used on model *output*, where leftover template markers or echoed
    delimiter tags are artifacts to delete rather than evidence to keep.
    """
    for pattern in (_GEMMA_TOKEN_RE, _CHATML_TOKEN_RE, _BRACKET_TOKEN_RE, _SYS_TOKEN_RE, _SEQ_TOKEN_RE, _APP_TAG_RE):
        text = pattern.sub("", text)
    return text


def sanitize_text(text: str, max_chars: int) -> SanitizedField:
    """Sanitize a single text field, applying all rules in order.

    Args:
        text:      Raw input text from the user or external source.
        max_chars: Maximum allowed length of the sanitized model_text.

    Returns:
        A :class:`SanitizedField` with ``model_text``, ``detector_text``,
        ``modified`` flag, and ``notes`` list.
    """
    original = text
    notes: list[str] = []

    # ------------------------------------------------------------------
    # Step 1 — Strip invisible / format (Cf) characters
    # ------------------------------------------------------------------

    # 1a. Zero-width and BOM characters (U+200B-U+200F, U+2060-U+2064, U+FEFF)
    zw_count = len(_ZERO_WIDTH_RE.findall(text))
    if zw_count:
        text = _ZERO_WIDTH_RE.sub("", text)
        notes.append(f"zero_width:{zw_count}")

    # 1b. Bidi control characters (U+202A-U+202E, U+2066-U+2069)
    bidi_count = len(_BIDI_CTRL_RE.findall(text))
    if bidi_count:
        text = _BIDI_CTRL_RE.sub("", text)
        notes.append(f"bidi_control:{bidi_count}")

    # 1c. Unicode tag block (U+E0000-U+E007F)
    tag_count = len(_TAG_BLOCK_RE.findall(text))
    if tag_count:
        text = _TAG_BLOCK_RE.sub("", text)
        notes.append(f"tag_block:{tag_count}")

    # 1d. Remaining Cf (format) characters not caught above
    cf_chars = [c for c in text if unicodedata.category(c) == "Cf"]
    if cf_chars:
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
        notes.append(f"cf_other:{len(cf_chars)}")

    # ------------------------------------------------------------------
    # Step 2 — Neutralise chat-template control tokens (all model families)
    # ------------------------------------------------------------------

    text, token_notes = neutralise_control_tokens(text)
    notes.extend(token_notes)

    # ------------------------------------------------------------------
    # Step 3 — Neutralise forged application delimiter tags
    # ------------------------------------------------------------------

    def _replace_app_tag(m: re.Match) -> str:  # type: ignore[type-arg]
        notes.append("app_tag_forged")
        return _neutralise_angle(m)

    text, _ = _APP_TAG_RE.subn(_replace_app_tag, text)

    # ------------------------------------------------------------------
    # Step 4 — Truncate
    # ------------------------------------------------------------------

    if len(text) > max_chars:
        text = text[:max_chars]
        notes.append("truncated")

    # ------------------------------------------------------------------
    # Step 5 — Build detector_text (NFKC-normalised); model_text is NOT
    # ------------------------------------------------------------------

    model_text = text
    detector_text = unicodedata.normalize("NFKC", text)

    modified = model_text != original

    return SanitizedField(
        model_text=model_text,
        detector_text=detector_text,
        modified=modified,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROFILE_KEYS: tuple[str, ...] = ("age", "sex", "duration", "conditions", "medications")


def sanitize_user_inputs(
    message: str,
    profile: Optional[Any],
    max_message_chars: int = 6000,
    max_profile_chars: int = 300,
) -> SanitizedInputs:
    """Sanitize the user message and patient profile.

    Args:
        message:           Raw free-text message from the user.
        profile:           Patient profile. Accepts a Pydantic model (with
                           ``.model_dump()``), a plain ``dict``, or ``None``.
        max_message_chars: Character limit for the message field (default 6000).
        max_profile_chars: Character limit per profile field (default 300).

    Returns:
        A :class:`SanitizedInputs` instance with sanitized message and profile.
    """
    sanitized_message = sanitize_text(message, max_message_chars)

    # Resolve profile to a plain dict
    if profile is None:
        raw_profile: dict[str, Any] = {}
    elif hasattr(profile, "model_dump"):
        # Pydantic v2 model
        raw_profile = profile.model_dump()
    elif isinstance(profile, dict):
        raw_profile = profile
    else:
        try:
            raw_profile = dict(profile)  # type: ignore[call-overload]
        except Exception:
            raw_profile = {}

    sanitized_profile: dict[str, SanitizedField] = {}
    for key in _PROFILE_KEYS:
        raw_value = raw_profile.get(key)
        if raw_value is None:
            continue
        str_value = str(raw_value).strip()
        if not str_value:
            continue
        sanitized_profile[key] = sanitize_text(str_value, max_profile_chars)

    return SanitizedInputs(message=sanitized_message, profile=sanitized_profile)
